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
> `tools/` does. It has run on two instance types — `ml.g6.xlarge` (one L4) and
> `ml.g6e.xlarge` (one L40S) — and most numbers here are the L4. It is a carve-out from a
> larger private tree, so some cited paths are not here (see
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
  up.py                   ONE COMMAND: role -> image -> endpoint -> a real invocation
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
`mlx_engine.py` / `vision.py` / `mps_kernels.py` are carried along but **not exercised by
this deployable** — the Triton image is CUDA and text-only. Each is reached by exactly one
entry point, in all three cases through a function-local import, so nothing on the CUDA
path imports them at all:

| file | reached only by | why it is kept |
| --- | --- | --- |
| `mlx_engine.py` | `infer.load_mlx_engine` (`--device mlx`) | the Apple-silicon half of the "use the plain server on a laptop" recommendation in [ARCHITECTURE.md](ARCHITECTURE.md) §7 |
| `mps_kernels.py` | `infer.py`'s device setup; `install()` is a no-op once `fla` is bound, so it is inert on CUDA | same, plus it is the reference implementation the Triton backend's `_assert_fla_on_gpu` guard exists to keep you off |
| `vision.py` | `server.create_app(..., vision=True)` | answers "can it do images?" without rebuilding the path; `schema.py` refuses images on a text-only engine rather than ignoring them |

None of the three has tests here — those suites stayed in the private tree, which is an
uncovered regression risk if you edit them.

## Running it

### 1. Tests (laptop, no GPU, no AWS)

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q          # 80 passed
```

Use the venv. The system interpreter usually lacks `fastapi`/`torch`, and the failure
looks like broken tests rather than a missing dependency.

### 2. One command to a working endpoint

If an image already exists in your account's ECR — which is the usual case after the first
build — this is the whole path from a fresh clone plus credentials:

```bash
deploy/up.py --name my-decider            # creates the role, the endpoint, and verifies it
deploy/up.py --name my-decider --down     # deletes all of it
```

`up.py` orchestrates the two scripts below rather than replacing them. It creates a
least-privilege execution role if you do not pass `--role` (ECR pull plus scoped
CloudWatch Logs — *not* `AmazonSageMakerFullAccess`), waits for IAM to propagate, hands off
to `create_endpoint.py`, and then **invokes the endpoint for real** and checks all three
primitives came back. That last step is the point: `InService` only means `/ping` answered,
and a wrong `SD_ENGINE` or an unreadable model repository produces an endpoint that is
`InService` and fails every request.

To build the image too, add `--build --box-id <instance>`:

```bash
deploy/up.py --name my-decider --build --box-id i-0123456789abcdef0 --tag v23-triton-onepass
```

**`--build` needs a GPU box you already have**, and that is the one genuine gap in the
one-command story: this build cannot run on a laptop (see below), so `up.py` will not
invent an instance for you. It does create the S3 build-context bucket, which
`build_on_box.sh` needs and does not create itself.

It creates billable resources — ~$1.13/hr for `ml.g6.xlarge`, charged whether or not
anything calls it — so `--down` is part of the workflow, not an afterthought. `--down`
leaves the ECR image and the build bucket alone deliberately: they cost cents and are what
make the next `up.py` take minutes instead of an hour.

### 3. Build the image

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

### 4. Deploy by hand

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

### Picking an instance: `ml.g6e.xlarge` is 2.7× under load

| instance | GPU | GPU mem | bandwidth | vCPU | $/hr hosting, us-west-2 |
| --- | --- | --- | --- | --- | --- |
| **`ml.g6.xlarge`** | L4 | 22.9 GB | 300 GB/s | 4 | **1.1267** — the default; most numbers here |
| `ml.g6.2xlarge` | L4 | 22.9 GB | 300 GB/s | 8 | 1.2220 |
| `ml.g6e.xlarge` | L40S | 45.8 GB | 864 GB/s | 4 | 2.6054 |
| `ml.g6e.2xlarge` | L40S | 45.8 GB | 864 GB/s | 8 | 2.8026 |

**`ml.g6e.xlarge` needs no rebuild** — the L40S is sm89 and `deploy/Dockerfile.triton`
already targets `8.0;8.6;8.9`. Measured through two live endpoints, same harness, same
in-region load generator, 7 questions, distinct tickets:

| | concurrency 1 | | concurrency 32 (saturation) | | |
| --- | --- | --- | --- | --- | --- |
| | decisions/s | server p50 | decisions/s | server p50 | tickets/s |
| `g6` (L4) | 63.4 | 102.5 ms | 99.0 | 620 ms | 14.15 |
| `g6e` (L40S) | 66.1 | 95.7 ms | **269.8** | **209 ms** | **38.55** |
| | +4% | −7% | **+173%** | **−66%** | +172% |

**Read those two halves together, because they look contradictory and are not.** At
concurrency 1 the bigger card buys ~4%: a single small pass is bound by the CPU issuing
~5,676 kernel launches, and no GPU can help with that. Under load the batcher fills each
pass with up to 8 requests × 7 questions = 56 rows, so the pass carries thousands of tokens
and has crossed out of the fixed-floor regime into the marginal one — where it is bound by
weight streaming and arithmetic, and the L40S's 2.88× memory bandwidth shows up almost
linearly as the measured 2.7×.

So **"this model is dispatch-bound" is true of one request and false of a saturated
server**, and which regime you care about picks the instance. The same fact explains why
`instance_group count: 2` lost on the L4: under load that card is genuinely saturated, not
merely busy, so a second process found no idle GPU to overlap into.

Cost per unit of work therefore **favours `g6e` at full load**, despite 2.31× the hourly
rate — ~$0.0188 per 1,000 tickets against ~$0.0221, about 15% cheaper. It is worse value
only if your traffic never leaves concurrency 1, where you would pay 2.31× for 4%.

Three things to know before you rely on this:

- **The `g6e` row is a lower bound.** Its server-side p50 was only 209 ms — the server was
  not deeply queued — so 269.8 decisions/s is what the 4-vCPU *load generator* could drive,
  not the endpoint's ceiling. The `g6` row is a real ceiling (p50 620 ms is the server
  queueing). A fatter load generator would raise one number and not the other. 1–2 requests
  of ~770 also errored in the `g6e` c=16/32 cells, undiagnosed.
- **Those are SageMaker *hosting* rates, not EC2 rates.** `g6e.xlarge` on EC2 on-demand is
  ~$1.86/hr; as a SageMaker endpoint it is $2.6054/hr. The EC2 number under-budgets by ~40%.
- **The endpoint-usage quota is per instance type and they differ.** In the account this was
  built in, `ml.g6.xlarge for endpoint usage` is **4** but `ml.g6e.xlarge` is **1** — so on
  `g6e.xlarge` autoscaling has nowhere to go and the 2.7× has to be enough by itself.
  `create_endpoint.py` now reads the real quota and clamps, because `application-autoscaling`
  accepts an impossible maximum without complaint and records the failed scale-out only in a
  scaling activity log.

And still **not `ml.g5`**: that fleet's host drivers (470.x, 535.x) are too old for any
current Triton DLC. It was tried on both CUDA 12 and CUDA 13 and failed both times; the
details are in `deploy/create_endpoint.py`.

### 5. Calling it

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
| `tools/fused_ab.py` | the fused kernels (`SD_FUSE_LAYERS=1`) against the reference torso: same requests, both readout routes, max and mean \|Δp\| per primitive, plus the torso forward timed both ways. `--fp32-reference` runs the same torso in fp32 as an arbiter, because two bf16 paths can differ by more than either differs from the exact answer. `--batch-time` times `evaluate_many` on a full Triton batch in-process, with no HTTP and no load generator — the only speed measurement this box can take honestly. Gate: **zero decision flips**. |
| `tools/bench_tickets.py` | tickets/s, decisions/s and per-decision latency by question count, ticket length and concurrency. Use `--tickets distinct` (the default). |
| `tools/sm_sweep.py`, `tools/loadsweep_triton.py` | load sweeps through the endpoint and straight to the container. Both record GPU utilisation alongside throughput, so a saturated card is distinguishable from a starved load generator. |

Run load tests **in-region**. From a laptop the round trip dominates and you measure the
internet, not the model.

> [!WARNING]
> **Load-testing an endpoint that has autoscaling attached will scale it out, and you pay
> for that for about half an hour.** Observed: a ~3-minute `bench_tickets.py` run against
> `strands-decider-g6` (target 250 invocations/instance/minute, max 4) pushed the high alarm
> into `ALARM` and set desired capacity to **3**. Because a new instance needs ~12 minutes
> to serve, and `ScaleInCooldown` is 600 s, the endpoint then sits above its floor for
> ~25–30 minutes after the load stops — roughly $1 of instances for a 3-minute test, and
> it also means a *second* test started during that window is measuring 3 instances rather
> than 1.
>
> Either point load tests at an endpoint with no scaling policy (which is what
> `deploy/up.py --name <something-else>` gives you by default), or measure the container
> directly with `--base http://localhost:8100`, which is what the `instance_group` table
> above was done with. Check `CurrentInstanceCount` before trusting any throughput number
> from a scaled endpoint.

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
| `instance_group count` | `config.pbtxt` | **1**, measured. 2 and 3 both fit in memory and are both *worse* — 91.4 and 79.8 decisions/s against 101.5. See [below](#instance_group-count-measured-1-wins). |
| `max_rows` | `BatchedSystemOneEngine` | 128 question rows per pass. Activation memory for the whole in-flight batch. |
| `DUP_TOKEN_BUDGET` | `BatchedSystemOneEngine` | 480. Above this many duplicated state tokens, encoding the state once and forking the cache beats a single combined pass. |
| `SD_ENGINE`, `SD_PREFIX_CACHE` | container env | `merged` folds the LoRA into the torso (no PEFT at runtime). Prefix caching on. |
| `SD_FUSE_LAYERS` | container env | **0 (off)**. `1` swaps the torso's decoder layers for `flash-linear-attention`'s Triton kernels — see [Fused kernels](#fused-kernels-sd_fuse_layers1) below. Measured **1.29x** on a full 56-row batch (103.8 → 133.7 decisions/s) and *slower* at batch 1. Correctness gates pass; the numbers move. |

## Fused kernels (`SD_FUSE_LAYERS=1`)

`src/strands_decider/fused_layers.py` rewrites the torso's decoder layers for inference
using `flash-linear-attention`'s Triton kernels: one projection GEMM per DeltaNet mixer
instead of four, fla's causal conv started from the cached conv state rather than from a
`torch.cat` of it, the gate / beta sigmoid / q-k L2 norm inside the chunk kernel, fla's
fused gated RMSNorm, one GEMM plus a fused SwiGLU for the MLP, and the two zero-centred
RMSNorms as fla's fused norm with the second adding the residual. It is a port of
[`kev`](https://github.com/jaredpalmer/kev)'s `fused_qwen35.py` — see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

**It is off by default, and that is a measurement, not caution.** Measured on this
deployable's L4, `tools/fused_ab.py`:

| | reference | fused | |
| --- | --- | --- | --- |
| **Full batch, 8 tickets × 7 questions** (56 rows — the shape `max_batch_size: 8` produces), least-contended pair | 540 ms / **103.8 decisions/s** | 419 ms / **133.7 decisions/s** | **1.29x** |
| …across 7 interleaved rounds | 540–1,154 ms | 419–895 ms | 1.05–1.99x, fused faster in **7 of 7** |
| Torso forward, batch 1, 128 tokens | 42 ms | 49 ms | **0.85x — fused is slower** |
| Torso forward, batch 1, 1,024 tokens | 84 ms | 73 ms | 1.14x |

The reference row is a useful check on the method: measured in-process it lands at
~101–104 decisions/s across rounds, against the ~99 decisions/s this README's live-endpoint
table reports at saturation.

So the win is in the **saturated** regime and the loss is at batch 1. That is consistent
with what the two regimes are: one small pass is CPU-dispatch-bound, and fla's ops carry
more Python per call (`input_guard`, autotune lookups, an autograd `Function`) even though
they launch fewer kernels — while a 56-row pass is memory-bandwidth-bound, where doing less
arithmetic and moving less data is what helps. On a 4-vCPU `g6.xlarge` the Python side is
not cheap.

These come from `tools/fused_ab.py --batch-time`, which calls the engine **in-process** with
no HTTP and no load generator. That is deliberate: see
[ARCHITECTURE.md §9](ARCHITECTURE.md#9-what-this-does-not-do) for why a co-resident
`bench_tickets.py` cannot measure this model on a 4-vCPU box.

Correctness, all three gates green with fusion on:

| check | result |
| --- | --- |
| `tools/reference_check.py` | **0 decision mismatches** against the published v21 values (Δ noul 0.0018, Δp 0.0020, Δscore 0.0031) |
| `tools/batch_parity.py` | **0 decision flips**; max \|Δp\| 0.0086 (0.0064 unfused) |
| `tools/fused_ab.py`, 80 answers over both readout routes | **0 decision flips**; mean \|Δp\| 0.0011–0.0019, max 0.0102 |

The max is above this project's 7e-3 advisory band and the mean is well inside it. `--fp32-reference`
settles which: run the *same* torso in fp32 as an arbiter and the reference bf16 path sits
0.00105 from it on average, the fused path 0.00124 — a 1.18x difference, with the
per-primitive maxima not ordering consistently. Both bf16 paths are about equally close to
the exact answer; they are simply not close to each other, which is what rounding looks like
and a systematic kernel error does not. `kev` records the same magnitude for its own bf16
path on an L40S (max 0.0133, mean 0.0014 from fp32, zero argmax flips).

**But it does move the numbers a caller sees**, by up to ~0.01 on a probability, so it is
opt-in. Turning it on refuses rather than degrades: `fuse_torso` checks the layer layout and
runs five kernel-contract probes on the GPU before rewriting anything, and raises — failing
`initialize()` and keeping Triton from reporting ready — if any of them has moved.

### `instance_group count`: measured, 1 wins

This was the README's "cheapest untried lever". It is now tried, and it does not pay.
`ml.g6.xlarge` (one L4, **4 vCPU**), 7 questions per ticket, distinct tickets,
`tools/bench_tickets.py --tickets distinct`, server-side latency, each row at the
concurrency where that setting peaks (c=32):

| `count` | decisions/s | tickets/s | server p50 | server p95 | GPU util (mean / median) | GPU memory |
| --- | --- | --- | --- | --- | --- | --- |
| **1** | **101.5** | **14.5** | **611 ms** | **615 ms** | 83% / 85% | 4.7 GB |
| 2 | 91.4 | 13.1 | 1,334 ms | 1,348 ms | 89% / 100% | 9.2 GB |
| 3 | 79.8 | 11.4 | 1,893 ms | 2,819 ms | 86% / 97% | 13.7 GB |

Zero errors in every cell, and `count: 1` reproduced the 101.5 decisions/s and 611 ms p50
already recorded in [ARCHITECTURE.md](ARCHITECTURE.md) §8, which is what makes the other
two rows comparable rather than merely adjacent.

**Memory was never the constraint** — 4,474 MiB per stub process plus 250 MiB for
`tritonserver`, so even 3 copies sit inside the L4's 22.9 GB with ~9 GB spare. The
constraint is CPU and batch geometry:

- **It halves the batch.** Triton gives each instance its own dynamic batcher and spreads
  arrivals between them, so 2 instances each form batches of ~4 instead of 8.
  `max_batch_size: 8` is tuned so 8 × 7 = 56 rows run as *one* pass pair; two batches of 4
  are *two* pass pairs, paying the ~45 ms dispatch floor twice for the same work. It is the
  same mistake as `max_batch_size: 32`, reached from the opposite direction.
- **It splits 4 vCPU.** The 45 ms floor is CPU kernel-launch overhead (~5,676 launches per
  pass), issued single-threaded per stub. A second dispatcher does not get a free core on a
  4-vCPU box; it takes one from the first.

The tell is in the utilisation column: `count: 2` runs the card at a *higher* median
utilisation (100% against 85%) while delivering 10% *less* work. That is contention, not
overlap — and it is why "GPU util is high" is not evidence that a GPU is the bottleneck on
this model.

**Revisit it only with more vCPU**, and only together with `max_batch_size`, so that
`count × max_batch_size × questions` still lands at or just under `max_rows`.
`ml.g6.2xlarge` (8 vCPU, same L4, ~8% more money) is the cheap place to retest.

## Known limits

- **Latency is dispatch-bound; throughput at saturation is not.** One *small* forward pass
  has a ~45 ms floor on an L4 that is CPU kernel-launch overhead (~5,676 launches) against a
  12.7 ms weight-streaming floor, and that floor is what a single request pays — cutting it
  is the largest remaining win for latency. Plain CUDA-graph capture was measured on this
  torso and does not work: the shapes that go fast return wrong values. But once the batcher
  has filled a pass with 56 rows the pass leaves that regime, and a **loaded** server is
  bound by the card: an L40S with 2.88x the memory bandwidth delivers 2.7x the decisions/s
  on the same configuration. An earlier version of this README claimed throughput was
  dispatch-bound full stop; that was an over-generalisation from concurrency-1 profiling.
- **The two regimes reward opposite optimisations.** Fused kernels (`SD_FUSE_LAYERS=1`) do
  less arithmetic and move less data but carry more Python per op, so they help a full pass
  (1.07-1.42x at 56 rows) and *hurt* a small one (0.85x at batch 1). Choose for the regime
  you are in.
- **`instance_group count: 2` is now measured and is worse** (91.4 decisions/s against
  101.5, with p50 doubling). More GPU-side parallelism is not the lever on a 4-vCPU host --
  see [above](#instance_group-count-measured-1-wins). The remaining levers all attack the
  dispatch floor itself, or buy more vCPU to issue it with.
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
