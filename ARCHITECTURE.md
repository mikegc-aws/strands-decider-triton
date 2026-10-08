# Architecture

Companion to [README.md](README.md). The README tells you how to build, deploy and call
this thing. This document explains **how it works and why it is shaped this way**, for a
developer who understands what a decider model does but does not want to become an ML
engineer to operate one.

The last section answers the question you probably arrived with: **is this different from
the plain server in the `strands-decider-2b` repo, or is it the same thing in a bigger
box?** Short answer: the model is identical, the serving layer is not, and the difference
is worth about 8x.

---

## Words you will see

Five terms do all the work. Everything else follows from them.

| term | plain meaning |
| --- | --- |
| **forward pass** | One trip of some text through the model's weights. The unit of work and the unit of cost. Think "one query against a database that has no index": you pay for the trip, not for the row you wanted. |
| **state** | The document, ticket, email or transcript being asked about. Usually the long part. |
| **question** | One typed question plus its option list. Usually the short part — a few dozen tokens. |
| **row** | One (state, question) pair as the model sees it. A request with 7 questions is 7 rows. |
| **cache** | What the model remembers about text it has already read, so it does not have to re-read it. Encode a state once, keep the cache, and a question can be appended for the price of the question alone. |

---

## 1. The mental model

**One forward pass costs the same whether you put a little or a lot through it.**

That single fact explains every design decision in this repository. Measured on the L4 in
`ml.g6.xlarge`:

```
cost of one forward pass  =  max( 45 ms ,  tokens × 0.0935 ms )
                                 ↑                 ↑
                           fixed floor        marginal cost
```

The 45 ms floor is not arithmetic. It is the CPU issuing ~5,676 individual instructions to
the GPU and waiting. (For scale: physically streaming the model's 3.8 GB of weights
through the card's memory bandwidth accounts for only 12.7 ms of it.) At realistic ticket
sizes — a few hundred tokens — **you are paying for the trip, not the cargo.**

So the entire serving problem is:

> **Make as few trips as possible, and fill each one.**

A useful analogy: a forward pass is a delivery van that costs £45 to send out regardless of
how full it is, plus about 9p per parcel. Nobody sends a van out with one parcel in it if
another parcel is waiting at the depot.

### The three ways to fill a van

```
1. Questions about the same state      →  encode the state once, add questions to it
   (one request, many questions)          ~free: 1 question 47 ms, 7 questions 67 ms

2. Requests arriving at the same time  →  put them in the same pass
   (many requests in flight)              8 tickets: 16 passes → 2 passes

3. Requests sharing a state            →  encode that state once for all of them
   (two question sets, one document)      de-duplicated by token ids
```

Level 1 is the decider model's native trick and it already exists in the base package
(`SystemOneEngine`). **Levels 2 and 3 are what this repository adds**, and they are the
reason it exists. See §7.

### Why the state is the expensive part

A request is `state + N questions`. The naive way to answer N questions is N passes of
`state + question_i`, which reads the state N times. The engine instead reads the state
once, keeps its cache, and runs the N short questions against copies of it:

```
naive      :  N × (state + question)     2,000-token state, 5 × 40-token questions = ~10,200 tokens
this engine :  state + N × question                                                 =  ~2,200 tokens
```

This is why **long documents are this deployment's strength rather than its weakness**.
Nearly doubling the input (601 → 1,025 tokens) costs about 14% of throughput, because the
document is paid for once per ticket instead of once per question.

### The one-pass / two-pass decision

Sharing a state is not always worth it, because *forking the cache costs an extra pass*.
`BatchedSystemOneEngine` therefore counts how many state tokens would be duplicated and
picks a route:

```
duplicated state tokens ≤ 480   →  ONE pass over "state + question" per row
duplicated state tokens >  480  →  TWO passes: states first, then questions against them
```

480 is just `45 ms ÷ 0.0935 ms` — the point where re-reading the state becomes more
expensive than an extra trip. It predicts all four measured cases:

| questions | duplicated tokens | one pass | two passes | chosen |
| --- | --- | --- | --- | --- |
| 1 | 0 | 45.6 ms | 94 ms | one |
| 3 | 180 | 48.6 ms | 93 ms | one |
| 7 | 540 | 97.9 ms | 101 ms | two |
| 14 | 1,170 | 190.9 ms | 158 ms | two |

This guard is why a one-question request costs ~45 ms and not ~94 ms. It matters because
the "clever" path is the wrong path for small requests, and a batch-only benchmark never
notices.

---

## 2. The pieces

```
   client
     │  {"state": "...", "questions": {...}}   (JSON)
     ▼
┌──────────────────────────────────────────────────────────────────────┐
│ SageMaker real-time endpoint — ml.g6.xlarge, one NVIDIA L4           │
│                                                                      │
│   POST /invocations  :8080        ← SigV4; the only public door      │
│        │                                                             │
│        │  the Triton DLC proxies this to KServe v2 `infer`           │
│        ▼                                                             │
│   Triton Inference Server                                            │
│        │   dynamic batcher:  queue ≤ 256, wait ≤ 2 ms,               │
│        │                     coalesce up to 8 requests               │
│        ▼                                                             │
│   python backend stub process  (one per instance_group count)        │
│        │                                                             │
│        │   model_repository/decider/1/model.py :: execute(requests)  │
│        │       the ADAPTER. Decodes JSON, calls the engine once,     │
│        │       returns one response per request, in order.           │
│        ▼                                                             │
│   BatchedSystemOneEngine.evaluate_many([...])                        │
│        │       pass 1: every distinct state (left-padded)            │
│        │       pass 2: every question row, against its state's cache │
│        ▼                                                             │
│   merged Qwen3.5-2B torso  +  pointer head          [on the GPU]     │
└──────────────────────────────────────────────────────────────────────┘
```

Who owns what, and the rule that keeps it honest:

| component | owns | notes |
| --- | --- | --- |
| `src/strands_decider/` | the model, prompt rendering, the window fit, both readout paths, temperatures, confidence formulas | **Unchanged from the base package.** Nothing in this repo reimplements inference. |
| `src/strands_decider/batch_engine.py` | cross-request batching, the cache gather, the one/two-pass routing | The new serving capability |
| `src/strands_decider/cuda_graphs.py` | CUDA graph capture of the forward passes: the state bank, shape bucketing, the padding masks | Opt-in, `SD_CUDA_GRAPHS`. Falls back to the eager pass per shape; see §9 |
| `src/strands_decider/merged_engine.py` | loading a torso with the LoRA already folded in | No PEFT at runtime |
| `src/decider_triton/wire.py` | JSON ↔ KServe v2 envelope | Deliberately free of Triton *and* torch imports, so it is unit-testable on a laptop |
| `model_repository/decider/1/model.py` | Triton's batch → one engine call → responses in order | Adapter only. ~390 lines, no maths |
| `model_repository/decider/config.pbtxt` | batching, queueing, instance count | The operational knobs |
| `deploy/` | the image, the in-region build, the endpoint | |
| `src/strands_decider/server.py`, `scheduler.py` | the plain FastAPI deployable | Kept for comparison and for running without Triton — see §7 |

---

## 3. Why Triton — and what it is *not* for

This is the question worth being precise about, because the package already ships a
perfectly good HTTP server with a bounded queue and a worker pool. Triton is not here for
any of that.

### What Triton actually provides

1. **A dynamic batcher.** This is the whole reason. Requests that arrive within 2 ms of
   each other are coalesced into a single `execute()` call, so the backend is handed a
   *batch* rather than a request. Everything downstream can then amortise the 45 ms pass
   floor across all of them.
2. **A bounded queue that sheds.** `max_queue_size: 256` with `timeout_action: REJECT`.
   Overload becomes an immediate refusal a load balancer can act on, rather than unbounded
   latency that looks like a hang to every caller.
3. **The SageMaker contract, for free.** The image is built `FROM` the SageMaker Triton
   DLC, which already implements the `serve` entrypoint verb, `/invocations` and `/ping` on
   port 8080, and the CloudWatch metrics plumbing. Not having to reimplement that is worth
   a lot of unglamorous debugging.
4. **Process management per model copy.** `instance_group count: N` gives N stub processes,
   each with its own weights, without writing any supervisor code.

### What Triton does *not* provide

**Triton does not make the forward pass batched.** It hands you a list of requests; turning
that list into two GPU passes is `evaluate_many`'s job, in this repository. This division
is the thing most worth internalising:

> **Triton gathers the parcels. The engine is what makes a full van cheaper than five empty
> ones.**

Put Triton in front of an engine that still loops one request at a time and you get almost
nothing — you have added a 2 ms queue delay and changed the wire format. That was in fact
the first version of this backend ("one engine call per distinct state"), and the comment
block in `model.py::execute` records exactly what it cost: 8 requests × 7 questions ran 16
passes and ~790 ms, where the same token count in 2 passes takes ~434 ms.

### Why not vLLM behind Triton

vLLM batches internally across its own scheduler, so a second batching queue in front of it
adds latency and coalesces nothing. More importantly, vLLM has no way to express "encode
this state once and fork it across questions" — each question must carry the whole state
again. That is the optimisation that makes long documents cheap here, so the torch path
keeps it. `model.py` refuses `SD_ENGINE=vllm` explicitly rather than silently serving a
different forward pass.

---

## 4. The life of one request

```
 1. Client POSTs to the endpoint. The System One body travels inside a KServe v2
    envelope (see §5). SigV4 signed.

 2. SageMaker's front door forwards it to the container's /invocations on :8080.
    In-region, this costs ~8-10 ms.

 3. The DLC proxies /invocations to Triton's POST /v2/models/decider/infer.

 4. Triton's dynamic batcher holds the request for up to 2 ms, hoping for company.
    If 8 gather first, it dispatches immediately. If the queue already holds 256,
    this request is rejected now rather than queued.

 5. The python backend's execute() receives the batch. For each request it decodes
    the JSON and validates it against the pydantic schema. A bad payload is answered
    immediately with an error body and dropped from the batch -- one caller's broken
    JSON must not fail its neighbours.

 6. The surviving requests go to evaluate_many() as one list. The engine:
      - renders each question to text and tokenises it
      - gives the QUESTION first claim on the context window, truncating the state
        (and front-truncating an over-long question, so the options and the <answer>
        marker always survive)
      - de-duplicates identical states by token ids
      - counts duplicated state tokens and picks one pass or two
      - runs the GPU work
      - reads out one probability distribution per row

 7. execute() maps results back: a ValueError becomes a caller error (matching the
    FastAPI server's 422), anything else becomes an internal error, and a success
    becomes a response body with latency_ms attached.

 8. Exactly one response per request, in the original order. Triton requires this;
    a missing response hangs that client.
```

Two things about step 6 are load-bearing and invisible if they are wrong:

- **States of different lengths must be left-padded, not right-padded.** 18 of the 24 model
  layers are recurrent — they consume tokens in order and carry state forward. Right-padding
  would make them consume the padding *last*, corrupting the state at exactly the point the
  second pass picks it up, and would put each question a different fictitious distance after
  its state. Left-padding puts every real state's end at the same position. Get this wrong
  and you do not get a crash: you get a confidently wrong probability at HTTP 200.
- **The per-row cache must be new storage, not a view.** A recurrent layer updates its state
  in place during the second pass, so a shared buffer would corrupt the state other rows
  still need.

Both are why the verification harness (`tools/batch_parity.py`) gates on *zero decision
flips against the single-request path* using **mixed state lengths** — equal lengths would
pad to nothing and pass while the logic was broken.

---

## 5. Why the body is JSON inside a tensor

Triton models normally declare a typed tensor per field. This one declares a single string:

```
input  [ { name: "REQUEST_JSON",  data_type: TYPE_STRING, dims: [1] } ]
output [ { name: "RESPONSE_JSON", data_type: TYPE_STRING, dims: [1] } ]
```

Two reasons.

**There is no fixed tensor schema for the request.** A decider request is a state plus an
arbitrary dict of named questions, each carrying an option list of 2–255 entries whose
*labels are defined by the request*, not by the model. That genericity is the whole point of
the architecture. It does not express as fixed dimensions.

**Identical shapes are what let the batcher work at all.** Every request is `dims: [1]`
regardless of content. A ragged tensor input would force Triton into separate batches, which
would defeat the only reason Triton is here.

### The gotcha that is worth reading twice

SageMaker's `/invocations` on the Triton DLC is **a thin proxy to KServe v2 `infer`, not a
raw-JSON endpoint.** This deployable is therefore *not* drop-in for `POST /v1/systemone`
clients. The same JSON travels, but wrapped:

```json
{"inputs": [{"name": "REQUEST_JSON", "shape": [1, 1],
             "datatype": "BYTES", "data": ["<the System One JSON>"]}]}
```

`shape` is `[1, 1]` and not `[1]` because the model sets `max_batch_size > 0`, so Triton
prepends a batch dimension. Sending `[1]` is rejected. A bare System One body returns an
opaque `Unable to parse 'inputs'` from Triton's JSON parser, which no amount of backend code
can report nicely.

`src/decider_triton/wire.py` (`encode_v2_request` / `decode_v2_response`) is the one
implementation of this wrapping, shared by the client, the tests and every harness. Use it
rather than hand-rolling the envelope.

### The error contract

Deliberately matched to the FastAPI server so a client cannot tell the two deployables
apart:

| situation | FastAPI server | this deployable |
| --- | --- | --- |
| malformed JSON, schema violation, too many options, prompt truncated through its options | HTTP 422 | `{"error": {"type": "invalid_request", ...}}` in the body |
| broken deployment, CUDA OOM | HTTP 500 | `{"error": {"type": "internal", ...}}` |
| queue full | HTTP 429 | Triton rejects the request |

A caller error is returned **as a body, not as a Triton-level failure**, because Triton
reports an exception from `execute()` as a server error for the whole batch — which would
make a client's own bad payload indistinguishable from the model being down, and would take
out the innocent requests batched alongside it.

---

## 6. The image, and why the weights are inside it

A two-stage build (`deploy/Dockerfile.triton`):

**Stage 1 — fold the LoRA, on CPU.** The published checkpoint is a base model plus a small
rank-16 adapter. Served as-is, every adapted layer runs `base(x) + B(A(x))` — three matrix
multiplies where one would do, which on this model is 558 `aten::mm` calls per pass instead
of 186, costing 1.59x–1.86x on the torso forward. Folding the adapter into the weights is
exact arithmetic and happens once, at build time. `peft` exists **only** in this stage; it is
deliberately absent from the runtime image so nobody can accidentally serve an unmerged
checkpoint, which would answer plausibly as the *un-adapted base model*.

**Stage 2 — the runtime.** The merged torso, the checkpoint (for the head, tokenizer and
fitted temperatures), the application code, and the model repository at `/opt/ml/model`.

Three operational consequences:

- **The weights are baked into the image, and there is no `ModelDataUrl`.** Scale-out does
  not pay a 4.6 GB S3 download per instance. The cost is a 20.6 GB image and a ~12 minute
  cold start dominated by the pull — which is why the autoscaling floor is 1 instance and
  scale-to-zero is not appropriate here.
- **Warm-up happens in `initialize()`, before Triton reports ready.** The recurrent layers'
  GPU kernels compile *per shape*, and the two readout paths are different code: one question
  takes the plain path, several take the shared-prefix path. Measured, the first pass costs
  54.2 s cold against ~67 ms warm. `initialize()` therefore sweeps three state lengths across
  both paths, so Triton's readiness signal is honest and no real caller pays for a compile.
- **The backend asserts its own fast path at startup.** This is the project's nastiest
  failure mode and it is silent: without a C compiler at runtime, the GPU kernel compiler
  cannot build its driver shim, and the linear-attention library *catches* that and falls
  back to a CPU reference path. The server starts, reports healthy, answers **correctly**,
  and runs 18 of 24 layers slowly. Nothing else would report it, so `_assert_fla_on_gpu`
  refuses to start.

---

## 7. How this differs from the basic server

Both deployables are in this repository, so you can compare them directly. **The basic
server is `src/strands_decider/server.py` + `scheduler.py`** — the same FastAPI +
bounded-queue path the base package ships, vendored here unchanged.

### What is identical

Everything that determines the *answer*:

- the model weights and the pointer-head readout
- prompt rendering (`prompting.py`) — the same `<state>` / `<question>` / `<options>` /
  `<answer>` text, option numbering and spans
- the question-first window fit and front truncation (`_fit`, shared by both paths — the
  window policy must not depend on how busy the server is, or the same payload would answer
  differently)
- the fitted per-primitive temperatures and both confidence formulas
- the JSON request and response bodies, field for field

`BatchedSystemOneEngine` is a *subclass* of the base engine and inherits `evaluate`
untouched. The gate on the new path is zero decision flips against the old one.

### What is different

| | basic server (`/v1/systemone`) | this deployable (`/invocations`) |
| --- | --- | --- |
| **front door** | FastAPI + uvicorn, bare JSON body | SageMaker → Triton DLC → KServe v2, JSON in a tensor envelope |
| **queue** | `Scheduler`: bounded (256), deadline, 429 / 503 | Triton dynamic batcher: bounded (256), REJECT |
| **requests per forward pass** | **one** | **up to 8 coalesced** |
| **cross-request batching** | no — the scheduler drains a batch then evaluates it **one entry at a time** | yes — `evaluate_many`, all distinct states in one pass |
| **state de-duplication across callers** | no | yes, by token ids |
| **LoRA adapter** | loaded via PEFT, unmerged at runtime | folded into the weights at build time |
| **engine class** | `SystemOneEngine` | `BatchedSystemOneEngine` over a merged torso |
| **warm-up** | `Scheduler.warmup()`, gates `/ready` | `initialize()`, gates Triton readiness |
| **health** | `/health` (liveness), `/ready` (503 until warm) | `/ping`, plus Triton's own health routes |
| **scaling out** | separate processes, by hand | `instance_group count`, or SageMaker autoscaling |

### What that is worth

The basic server's ceiling is one forward pass at a time. Its own scheduler docstring is
candid about this — the drain "is the seam a future cross-request batched forward pass slots
into. It is **NOT** yet a shared forward pass." Measured on the torch path, that ceiling is
**12 req/s at one worker, and *worse* with more** (5.2 at 4 workers, 3.8 at 8), because
concurrent passes contend for the same GPU without any of them going faster. Adding workers
to the basic server does not help; the queue exists precisely to serialise it.

With cross-request batching:

```
8 tickets × 7 questions, ~90-token states, same token count both ways

   per-request (basic)   16 passes   ~790 ms   (measured)
   per-batch   (this)     2 passes   ~434 ms

The entire saving is passes, not arithmetic.
```

End to end on the live endpoint: **~99 decisions/s** (~14 tickets/s, ~36,000 tickets/hour) at
about **$0.022 per 1,000 tickets**, with a 7-question ticket at **67 ms** server-side.

### So which should you use?

| use | because |
| --- | --- |
| **this deployable** for concurrent traffic on SageMaker | many callers, overlapping tickets — the batcher has something to batch, and the per-pass floor gets spread across it. Experimental: see the warning at the top of [README.md](README.md) before putting anything you care about through it |
| **the basic server** for local development, a laptop, MPS or CPU, CI, or a single-caller batch job | no Triton, no image build, no AWS; at concurrency 1 the two are within a few ms of each other, because with one request in flight there is nothing to coalesce |

The second row is the honest caveat: **at concurrency 1 this deployment buys you almost
nothing** beyond the merged torso. Its advantage is entirely in what happens when requests
overlap.

---

## 8. The knobs, and how they interact

Only one relationship really matters:

> **`max_batch_size` × typical questions per request should land at or just under
> `max_rows`.**

`max_batch_size` (8) is how many requests Triton coalesces. `max_rows` (128) is how many
question rows share one GPU pass. 8 × 7 = 56 rows, comfortably inside 128, so a full batch
is one pass pair.

Exceeding `max_rows` does not fail — it chunks, which means you paid the queueing delay to
assemble a big batch and then split it anyway. That is measurable: raising
`max_batch_size` to 32 gave **56 decisions/s against 101.5**, with server p50 going
611 ms → 2,031 ms, because Triton occasionally formed batches of 26 requests (182 rows).
**Bigger batches are not better past the point the engine can run them in one pass.**

| knob | where | value | note |
| --- | --- | --- | --- |
| `max_batch_size` | `config.pbtxt` | 8 | measured; see above |
| `max_queue_delay_microseconds` | `config.pbtxt` | 2 ms | pure added latency for a request arriving into an empty queue, so keep it small relative to ~67 ms of work. A copy-pasted 100 ms would dominate the request |
| `max_queue_size` | `config.pbtxt` | 256, then REJECT | bounded shedding beats unbounded latency |
| `instance_group count` | `config.pbtxt` | 1 | the model is ~5 GB on a 24 GB card, so several fit; raising it overlaps one batch's CPU work with another's GPU work. **Untested — the cheapest untried lever** |
| `max_rows` | `BatchedSystemOneEngine` | 128 | activation memory for the whole in-flight batch |
| `DUP_TOKEN_BUDGET` | `BatchedSystemOneEngine` | 480 | the one-pass/two-pass threshold, in duplicated state tokens |
| `SD_ENGINE` | container env | `merged` | `merged` folds the LoRA; `hf` is for A/B only and needs an image built with `peft` |

---

## 9. What this does not do

Stated plainly so you do not discover it in production:

- **A small forward pass is dispatch-bound.** The ~45 ms floor is CPU kernel-launch
  overhead, and prefill runs at ~34% of the card's peak — so a pass with one question in it
  is paying for launches, not arithmetic. A pass the batcher has filled is not.

  **CUDA graph capture closes that for the one-pass route, and is opt-in**
  (`SD_CUDA_GRAPHS=1`, off by default): a one-question request goes 45.9 ms → 21.5 ms
  server-side. This section used to say capture "does not work — the shapes that go fast
  return wrong values". That was a true measurement of *plain* capture, and the reason is
  worth knowing because it generalises: transformers builds its attention masks inside the
  forward, that code reads device data back to the host and branches on it, and a capture
  records kernels rather than branches — so the branch's *result* is baked in and the graph
  replays one set of sequence lengths for ever. Hoist the masks out (transformers 5 accepts
  `attention_mask` as a dict keyed by layer type) and replay is bit-identical to eager.
  `src/strands_decider/cuda_graphs.py` carries the full account, and the two-pass route is
  implemented but measured slower and left off; see README "Known limits".
- **A single question costs almost as much as three.** Same reason: you are paying for the
  pass, not the work.
- **Under load, latency becomes queueing.** At 32 requests in flight a 7-question ticket sits
  at ~615 ms p50. Autoscaling exists to keep you off that; point it at a target well below
  the measured ceiling so a second instance comes up while p50 is still healthy.
- **Cold start is minutes, not seconds** (~12 min to `InService`, dominated by the 20.6 GB
  image pull). Keep a warm floor of at least 1 instance.
- **Text only.** Images are rejected rather than ignored; the vision path needs a different
  torso.
- **The engine seam is `create_app(engine=...)` / `evaluate_many`**, so an alternative
  executor can be dropped in without touching the server. A vLLM pooling engine is
  reasonable at short inputs but gives up the shared-state optimisation — which is the thing
  that makes long documents cheap here.
