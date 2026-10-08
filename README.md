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
tests/                    80 unit tests, no GPU or AWS needed
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
.venv/bin/python -m pytest -q          # 80 passed
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
| `tools/batch_parity.py` | cross-request batching against the single-request path. Gate: **zero decision flips**; probability drift is advisory (≤7e-3 — batching changes bf16 reduction order). |
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

## Known limits

- **Throughput is dispatch-bound, not compute-bound.** One forward pass has a ~45 ms floor
  on an L4 that is CPU kernel-launch overhead (~5,676 launches) against a 12.7 ms
  weight-streaming floor, and prefill runs at ~34% of the card's peak. Cutting that floor
  is the largest remaining serving win. Plain CUDA-graph capture was measured on this torso
  and does not work — the shapes that go fast return wrong values.
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
