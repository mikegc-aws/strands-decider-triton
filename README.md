# strands-decider-triton

**An experiment in cheap decider hosting.** A Triton Inference Server backend for
**`strands-decider-2B-hobson-v21`**, deployed to an AWS SageMaker real-time endpoint.

Send a document and a set of typed questions; get back calibrated probabilities. No text
generation, no decode loop, no prompt parsing on the way out.

> [!WARNING]
> **Experimental. Not a supported product, and not production-hardened.**
>
> This was built to answer a question — can a decider model be served cheaply on one GPU,
> and where does the cost actually go? — and it is published so other people can poke at
> the answer, not because it is ready to carry your traffic.
>
> **What is real:** it works, and every number in this README was *measured* on a live
> endpoint rather than estimated, including a correctness gate of zero decision mismatches
> against the model's published reference values.
>
> **What is not:** no SLA, no support, no security review, and no load testing beyond what
> `tools/` does. It has run on exactly one instance type (`ml.g6.xlarge`, one L4). Several
> tuning levers are explicitly untested — `instance_group count: 2` among them. It is a
> carve-out from a larger private tree, so some cited paths are not here (see
> [What this repo is *not*](#what-this-repo-is-not)). Read
> [Known limits](#known-limits) before relying on any of it.
>
> Apache-2.0, so experiment freely. If you find it wrong, an issue or a PR is welcome.

```
POST /invocations
  { "state": "<the ticket, email, transcript, document>",
    "questions": {
      "urgent":     { "type": "noul",   "instructions": "Does this convey urgency?" },
      "department": { "type": "choice", "instructions": "Who handles this?",
                      "criteria": { "billing": "payments", "technical": "bugs" } },
      "severity":   { "type": "score",  "instructions": "How severe?",
                      "criteria": ["none", "minor", "major", "critical"] } } }

  -> { "answers": {
         "urgent":     { "type": "noul",   "noul": 0.8754 },
         "department": { "type": "choice", "choice": "billing", "confidence": 0.834,
                         "probabilities": { "billing": 0.889, "technical": 0.057 } },
         "severity":   { "type": "score",  "score": 1.072, "confidence": 0.601,
                         "probabilities": { "0": 0.14, "1": 0.648, "2": 0.212 } } },
       "usage": { "input_tokens": 518, "output_tokens": 3 }, "latency_ms": 67.2 }
```

## What it is

Three question types, each returning a probability distribution rather than a sampled
token:

| type | returns |
| --- | --- |
| `noul` | `P(true)` for a statement about the state |
| `choice` | the chosen option, plus a probability for every option, plus a confidence |
| `score` | an expected level on an ordered rubric, plus per-level probabilities and a confidence |

Confidences are **derived from the distribution**, not predicted by a second head —
normalised max-probability for `choice`, normalised standard deviation for `score`.

Under the hood the model is a Qwen3.5-2B hybrid torso (18 of 24 layers are Gated DeltaNet
linear attention, 6 are full attention) with its language-model head replaced by a ~1M
parameter pointer head. The head compares the hidden state at the `<answer>` position
against the hidden state at each option's last token. One prefill, one readout, done.

**One state, many questions.** Each question becomes its own row sharing the state, so
extra questions are nearly free: on one L4, 1 question costs 47 ms and 7 cost 67 ms.
The engine encodes the state once and forks its KV/recurrent cache across the questions
when that is cheaper than re-encoding it, and takes a single combined pass when it is not.
It applies the same trick **across concurrent requests**, so a batch of tickets costs two
forward passes in total rather than two per ticket.

## Measured on the live endpoint

`ml.g6.xlarge` (one NVIDIA L4), warm, server-side, 7 questions per ticket.

| | |
| --- | --- |
| Latency, 1 question | **47 ms** |
| Latency, 7 questions | **67 ms** (9.6 ms per decision) |
| Latency, 7 questions + a ~400-token attached document | 104 ms |
| Throughput at saturation | **~99 decisions/s** (~14 tickets/s, ~50,000 tickets/hour) |
| Cost | $1.1267/hr hosting → **~$0.022 per 1,000 tickets** at full load |
| Correctness | **0 decision mismatches** against the model's published reference values (max Δp 0.0020) |
| Cold start | ~12 min to `InService` (dominated by the 20.6 GB image pull); ~25 s container start with a warm kernel cache |

A caller inside the same AWS region adds ~8–10 ms. Under load, latency becomes queueing:
at 32 requests in flight a 7-question ticket sits at ~615 ms p50. Autoscaling exists to
keep you off that.

**Every number above is with CUDA graphs off, which is the default.** Turning them on
(`SD_CUDA_GRAPHS=1`) takes a 1-question ticket from ~46 ms to **21 ms** and a 2-question one
from ~48 ms to **30 ms**, and leaves the 7-question figure alone. See
[Known limits](#known-limits) for why the split falls there.

Long states are this deployment's strength: nearly doubling the input (601 → 1,025 tokens)
costs ~14% of throughput, because the state is paid once per ticket rather than once per
question.

## Layout

[ARCHITECTURE.md](ARCHITECTURE.md) explains how the serving layer works and why — the
cost model, what Triton is and is not for, and how this deployable differs from the plain
FastAPI server. Read it before changing the tuning knobs below.

[CLIENT.md](CLIENT.md) is the handover document for another project calling the deployed
endpoint: coordinates, IAM, the envelope, a drop-in client, and the error contract.
[notebooks/decider_playground.ipynb](notebooks/decider_playground.ipynb) is the same
client in a notebook, with runnable examples of all three question types, the batching
and concurrency effects, and the error contract.

```
src/strands_decider/      the model, prompt rendering, engines
  modeling.py             torso + pointer head + the masked softmax readout
  cuda_graphs.py          CUDA graph capture of the forward passes (opt-in, see below)
  prompting.py            renders <state>/<question>/<options>/<answer>
  infer.py                SystemOneEngine: window fit, option spans, single-request paths
  batch_engine.py         cross-request batching; the one-pass/two-pass decision
  merged_engine.py        loads the pre-merged torso (no PEFT at runtime)
  schema.py               the request/response contract (pydantic)
  server.py, scheduler.py FastAPI + bounded-queue path, for running without Triton
src/decider_triton/
  wire.py                 JSON <-> KServe v2 envelope. Triton-free and torch-free.
model_repository/decider/
  config.pbtxt            dynamic batching, instance group, warm-up policy
  1/model.py              the Triton python backend
deploy/
  Dockerfile.triton       two-stage: fold the LoRA on CPU, then the runtime image
  build_on_box.sh         in-region amd64 build + push to ECR, via SSM
  create_endpoint.py      model, endpoint config, endpoint, autoscaling (boto3)
serving/merge_lora.py     the build-time LoRA merge
tools/                    verification and measurement, see below
tests/                    110 unit tests, no GPU or AWS needed
```

### What this repo is *not*

It is the **serving slice** of a larger private tree, extracted so the deployable can stand
on its own. Training, evaluation, the data pipeline and the vLLM deployable stayed behind.

Comments and docstrings here still cite that tree, because the citation is where a measured
number came from and removing it would leave a bare claim. Those paths are **not in this
repository** and that is expected, not a broken link:

| cited | what it was | here? |
| --- | --- | --- |
| `evaluation/`, JevBench | the accuracy and device-parity suites | no |
| `data/collate.py` | training-time option shuffling | no |
| `serving/README.md`, `serving/vllm_server.py`, `vllm_engine.py` | the vLLM deployable and its notes | no |
| `tests/test_prefix_cache.py`, `tests/test_mps_kernels.py` | tests for code that is included | no |
| `research/…/BENCHMARK.md`, `LambdaGpuLaunchDemo` | the comparison numbers and the build-box pattern | no |
| `/opt/prof/*`, `/opt/tctx/*` | scratch paths on the GPU build box | no |
| `strands-decider serve`, `… calibrate`, `… data build` | the parent package's CLI | no — there is no `strands-decider` console script here |

Two consequences worth knowing before you read the source. `src/strands_decider/` is
importable but has **no CLI**, so where a docstring says `strands-decider serve <ckpt>`,
the equivalent here is `strands_decider.server.create_app(...)` or `serve(...)`. And
`mlx_engine.py` / `vision.py` / `mps_kernels.py` are carried along because `infer.py`
imports into them, but neither the MLX nor the vision path is exercised by this
deployable — the Triton image is CUDA and text-only.

## Running it

### 1. Tests (laptop, no GPU, no AWS)

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q          # 110 passed
```

Use the venv. The system interpreter usually lacks `fastapi`/`torch`, and the failure
looks like broken tests rather than a missing dependency.

### 2. Build the image

Build **in-region on an amd64 GPU host**, not on a laptop: the base image alone is 27.7 GB
and the result is ~20.6 GB compressed.

```bash
TRITON_IMAGE=763104351884.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tritonserver:25.04-py3 \
  deploy/build_on_box.sh v23-triton-onepass
```

`BOX_ID`, `REGION`, `REPO` and `BUCKET` are environment overrides; set `HF_TOKEN` if the
checkpoint repo requires one (it is passed as a BuildKit secret and never lands in a
layer).

**`TRITON_IMAGE` must be a CUDA 12 tag.** `25.04-py3` is CUDA 12.9. The newer DLC tags
(`25.09`, `26.03`–`26.05`) are CUDA 13 and cannot be placed on every GPU fleet — some
SageMaker hosts carry NVIDIA drivers as old as `470.256.02`, below the minimum for any
current DLC. The build asserts its own dependency versions (`torch 2.7.1+cu126`,
`triton >= 3.3`, `transformers >= 5.18`) and fails rather than shipping a mismatch.

### 3. Deploy

```bash
deploy/create_endpoint.py \
  --image <acct>.dkr.ecr.us-west-2.amazonaws.com/strands-decider-serving:v23-triton-onepass \
  --role  arn:aws:iam::<acct>:role/service-role/<SageMakerExecutionRole> \
  --instance-type ml.g6.xlarge \
  --name strands-decider-g6
```

Then attach autoscaling once it is `InService` — re-running the same command is safe, it
reuses the model and config and leaves a healthy endpoint alone:

```bash
deploy/create_endpoint.py --image <same> --role <same> \
  --name strands-decider-g6 --target-invocations 250
```

**Pick this target from the measured ceiling, and note it is per instance per *minute*.**
Saturation is ~14 tickets/s, i.e. ~840 invocations/instance/minute, so:

| target | share of ceiling | effect |
| --- | --- | --- |
| 250 | ~30% | a second instance is requested while p50 is still ~100 ms. Recommended |
| 600 | ~70% | scale-out is requested only once the instance is well into queueing — p50 is heading toward the ~615 ms above before help arrives |

600 was deployed first, described here as "~25% of the ceiling", and that was wrong
arithmetic rather than a different measurement. It matters more than it looks because a new
instance takes **~12 minutes** to serve traffic, so the target has to fire well before the
current one is in trouble — target tracking is not a brake you can apply late.

Default capacity is 1–4 instances with a **warm floor of 1**: GPU cold start is minutes, so
scale-to-zero is not appropriate for latency-sensitive traffic.

Teardown: `deploy/create_endpoint.py --name strands-decider-g6 --delete`.

**Target `ml.g6`/`ml.g6e` (Ada).** If capacity is short, launch several
`create_endpoint.py` attempts concurrently under different `--name` values and keep
whichever lands — a capacity failure takes ~31 minutes to surface, so running the ladder
in parallel beats running it in series. Delete the losers.

### 4. Calling it

`/invocations` on the Triton DLC is a proxy to Triton's KServe v2 `infer`, so the body
travels inside a tensor envelope. `shape` is `[1, 1]` because the model sets
`max_batch_size > 0` and Triton prepends the batch dimension:

```json
{"inputs": [{"name": "REQUEST_JSON", "shape": [1, 1],
             "datatype": "BYTES", "data": ["<the JSON request above>"]}]}
```

The response comes back the same way under `RESPONSE_JSON`.
`src/decider_triton/wire.py` (`encode_v2_request` / `decode_v2_response`) is the one
implementation of this, shared by the client, the tests and the harnesses — use it rather
than hand-rolling the envelope.

A malformed request returns `{"error": {"type": "invalid_request", "message": ...}}` in the
response body; it is a caller error, not a server fault, and it does not affect other
requests batched alongside it.

## Verification and measurement

| tool | what it establishes |
| --- | --- |
| `tools/reference_check.py` | reproduces the model's published reference values. The strongest end-to-end check: it validates prompt rendering, the window fit, option spans, the pointer readout, the fitted temperatures, the confidence formulas, the folded LoRA and the wire format at once. Gate: zero decision mismatches. |
| `tools/triton_smoke.sh` | starts the image the way SageMaker does (`docker run <image> serve`) and checks readiness, all three primitives, batching, caller errors, and that warm-up covered the shapes. |
| `tools/batch_parity.py` | cross-request batching against the single-request path. Gate: **zero decision flips**; probability drift is advisory (≤7e-3 — batching changes bf16 reduction order). `--cuda-graphs` / `--cuda-graphs-two-pass` put the batched side on the graph path and additionally **fail if nothing was captured or replayed**, so a silent fallback cannot pass the gate. |
| `tools/bench_tickets.py` | tickets/s, decisions/s and per-decision latency by question count, ticket length and concurrency. Use `--tickets distinct` (the default). |
| `tools/sm_sweep.py`, `tools/loadsweep_triton.py` | load sweeps through the endpoint and straight to the container. Both record GPU utilisation alongside throughput, so a saturated card is distinguishable from a starved load generator. |

Run load tests **in-region**. From a laptop the round trip dominates and you measure the
internet, not the model.

Two startup guards worth knowing about, because both failure modes are silent rather than
loud:

- The backend asserts `fla` is on CUDA. Without a runtime C compiler, `flash-linear-attention`
  catches the error and falls back to a CPU reference path — the server starts, reports
  healthy, and answers *correctly* with all 18 Gated DeltaNet layers on the slow path.
- `initialize()` sweeps several state lengths across both readout paths before reporting
  ready. The DeltaNet kernels compile per shape (~54 s cold against ~40 ms warm), so an
  unwarmed shape means the first real caller pays for compilation.

## Tuning

| knob | where | note |
| --- | --- | --- |
| `max_batch_size` | `config.pbtxt` | Requests coalesced per `execute()`. **8**, measured. Keep `max_batch_size × typical questions` at or just under `max_rows`; 32 was tried and is worse. |
| `max_queue_delay_microseconds` | `config.pbtxt` | 2 ms. Pure added latency for a request arriving into an empty queue, so keep it small relative to the work. |
| `max_queue_size` | `config.pbtxt` | 256, then reject. Bounded shedding beats unbounded latency — a load balancer can act on a refusal. |
| `instance_group count` | `config.pbtxt` | 1. The model is ~5 GB on a 24 GB card so several fit; raising it overlaps one batch's CPU work with another's GPU work. Untested. |
| `max_rows` | `BatchedSystemOneEngine` | 128 question rows per pass. Activation memory for the whole in-flight batch. |
| `DUP_TOKEN_BUDGET` | `BatchedSystemOneEngine` | 480. Above this many duplicated state tokens, encoding the state once and forking the cache beats a single combined pass. |
| `SD_ENGINE`, `SD_PREFIX_CACHE` | container env | `merged` folds the LoRA into the torso (no PEFT at runtime). Prefix caching on. |
| `SD_CUDA_GRAPHS` | container env | `0`. `1` graphs the one-pass route (1 question 45.9 → 21.5 ms). `all` also graphs the state/row pair, which measured 0.73x–1.02x and is therefore not in `1`. Falls back to eager per shape. See [Known limits](#known-limits). |

## Known limits

- **A small forward pass is dispatch-bound; a full one is not.** One pass has a ~45 ms
  floor on an L4 that is CPU kernel-launch overhead (~5,676 launches) against a 12.7 ms
  weight-streaming floor, and prefill runs at ~34% of the card's peak. That floor dominates
  a *small* pass, which is why a one-question request costs almost as much as three. Once
  the batcher fills a pass it stops being the limit.

  **CUDA graph capture now exists and is opt-in: `SD_CUDA_GRAPHS=1`.** It is off by
  default. This section used to say capture "does not work — the shapes that go fast return
  wrong values", which was a true measurement of *plain* capture: transformers builds its
  attention masks inside the forward, that code reads device data back to the host and
  branches on it, and a host-side branch inside a capture has its *result* baked in, so the
  graph replays one set of sequence lengths for ever. Hand the masks in precomputed —
  transformers 5 takes `attention_mask` as a dict keyed by layer type — and replay is
  bit-identical to an eager pass over the same buffers. See
  [`src/strands_decider/cuda_graphs.py`](src/strands_decider/cuda_graphs.py), ported from
  `kev` (credited in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).

  **What it buys, and the shape of it.** `evaluate_many` timed in-process, eager and
  graphed alternated per iteration, median of 21, on an otherwise idle L4 (inter-quartile
  spread ≤1.6 ms):

  | request | state tokens | eager | `SD_CUDA_GRAPHS=1` | |
  | --- | --- | --- | --- | --- |
  | 1 question | 75 | 40.9 ms | **19.9 ms** | 2.06x |
  | 1 question | 145 | 41.9 ms | **22.4 ms** | 1.87x |
  | 1 question | 245 | 42.1 ms | 28.7 ms | 1.47x |
  | 1 question | 335 | 43.7 ms | 35.9 ms | 1.22x |
  | 1 question | 725 | 58.7 ms | 61.2 ms | 0.96x |
  | 2 questions | 75 | 44.0 ms | 27.9 ms | 1.58x |
  | 2 requests × 1 question | 75 | 43.0 ms | 28.3 ms | 1.52x |

  Read down that table and the 45 ms floor is visible directly: **eager costs ~41 ms
  whatever you put in it up to a few hundred tokens, and the graph costs what the tokens
  actually cost** — 20 ms, 22 ms, 29 ms, 36 ms. The graph does not make the GPU faster; it
  removes a fixed CPU cost and leaves a variable GPU one. So the win is 2x on the smallest
  pass, shrinks as the pass fills, and is gone by ~400–500 tokens.

  End to end through the container (`tools/bench_tickets.py --base`, concurrency 1, on an
  otherwise-idle card). Run as eager → graphed → eager, so the two eager runs bracket the
  graphed one; they agree within ~4%, which is the drift control, and the eager column below
  is their range:

  | | eager (bracketing runs) | `SD_CUDA_GRAPHS=1` |
  | --- | --- | --- |
  | 1 question | 19.8–20.5 decisions/s, p50 45.1–47.0 ms | **40.1 decisions/s, p50 21.0 ms** |
  | 2 questions | 38.0–39.5 decisions/s, p50 47.2–49.1 ms | **59.6 decisions/s, p50 30.0 ms** |
  | 3 questions | 56.2–57.8 decisions/s, p50 48.5–49.9 ms | 67.2 decisions/s, p50 40.6 ms |
  | 7 questions | 63.9–66.7 decisions/s, p50 101.8–106.2 ms | 63.9 decisions/s, p50 103.3 ms |
  | 7 questions + ~400-token document | 58.8–59.7 decisions/s, p50 114.1–116.1 ms | 58.8 decisions/s, p50 115.4 ms |

  1,420 graph replays over 2,204 requests, 10 graphs captured, none failed. The 7-question
  rows are unchanged **by design** — see below. Concurrency 8 and 32 are left out because
  they did not settle at a 15-second window: the same cell measured twice inside one run
  gave 37.8 and 78.4 decisions/s. That is queueing, and it is also the regime where graphs
  are expected to do least, since a pass the batcher has filled is no longer paying for
  launches.

  **What is NOT fixed: the 7-question request, which is the headline number.** It takes the
  engine's *two-pass* route (`DUP_TOKEN_BUDGET`), and graphing that pair measured
  **slower** — 0.73x–1.02x. The cause is not capture. A graph cannot branch, so the state
  pass must always carry an explicit `Sb × Sb` additive mask, which puts the 6 attention
  layers on the masked SDPA path instead of the causal flash one that the eager path gets
  for free whenever a batch's states are the same length; past a few hundred state tokens
  that costs more than the launches it saves. That route is implemented, gated
  (`SD_CUDA_GRAPHS=all`), correctness-tested, and **off**.

  **What to try next, concretely.** Keep the state pass eager — it is compute-bound anyway
  and it keeps the flash kernel — and graph only the row pass, whose mask is
  `Lb × (Sr + Lb)` with `Lb` in the tens. The row pass is ~45 ms of the 92 ms a
  one-request/7-question call takes, so this is worth about 1.3x on exactly the case the
  table above misses. The reason it is not done here is memory: the row pass has to read
  its states from fixed addresses, which is what the state bank exists for, and that bank
  is the ~1.1 GB that made capture fail with CUDA OOM on a shared card. The way out is
  visible but untried: do the per-row state gather **outside** the captured region, writing
  straight into the row buffers. It is ~48 kernel launches against the 5,676 a pass costs,
  so hoisting it out gives up nothing, and it deletes the bank entirely.

  Two other limits, both deliberate: a pass is graphed all-or-nothing, so one state over
  1,024 tokens in a batch sends that whole batch to the eager path; and the buffers are
  fixed (64 MiB for the one-pass route, 1,188 MiB with the bank), so on a card shared with
  anything else capture can fail with CUDA OOM — in which case it says so, stops trying,
  and runs eagerly.
- **`instance_group count: 2` is untested** and is the cheapest untried lever.
- A single question costs almost as much as three, for the same reason: you are paying for
  the pass, not the work.
- Text only. Images are rejected; the vision path needs a different torso.
- The engine seam is `create_app(engine=...)` / `evaluate_many`, so an alternative executor
  can be dropped in without changing the server. A vLLM pooling engine is a reasonable
  alternative at short inputs but gives up the shared-state optimisation, which is what
  makes long documents cheap here.

## Licence

Apache-2.0 — see [LICENSE](LICENSE).

[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) records what this derives from: the Gated
DeltaNet chunk rule in `mps_kernels.py` is a modified Apache-2.0 file from `transformers`,
and the scheduler's shape and the hybrid cache fork are credited there too. All three
upstreams are Apache-2.0.

**No model weights are in this repository.** The image downloads the checkpoint at build
time; it carries its own licence, which this repository neither alters nor restates. Check
it before redistributing an image you have built.
