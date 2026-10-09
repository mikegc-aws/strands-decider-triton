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
  up.py                   ONE COMMAND: role -> image -> endpoint -> a real invocation
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
.venv/bin/python -m pytest -q          # 110 passed
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

**Target `ml.g6`/`ml.g6e` (Ada), and use `--instance-pools` when capacity is short.**

```
deploy/create_endpoint.py --image <uri> --role <arn> \
    --instance-pools ml.g6e.4xlarge,ml.g6e.2xlarge,ml.g6e.xlarge,ml.g6e.8xlarge
```

SageMaker takes up to five instance types per production variant in priority order and
falls back automatically on `InsufficientInstanceCapacity`, so **one** `CreateEndpoint`
covers the whole ladder. This replaces the old advice here — launch several endpoints under
different `--name` values and keep whichever lands — which worked but needed a human
watching three deploys and left two to tear down.

Measured, and the margin is the point. Ada capacity in this account was genuinely short:
a pooled attempt over `4xlarge → 2xlarge → xlarge` exhausted all three and failed in
**20 minutes**; the same list retried with `8xlarge` appended landed on the **priority-4
fallback in 9 minutes**. Serially, discovering that would have been four
`CreateEndpoint` attempts at ~31 minutes each — about **two hours** — and the first three
would each have looked like a dead end.

Two details worth knowing:

- `InstancePools` **replaces** `InstanceType`; sending both is a validation error.
- `VariantInstanceProvisionTimeoutInSeconds` (300–3600, set to 900 here) bounds the *total*
  attempt across all pools. It is what converts a half-hour dead end into a 15-minute one,
  and the whole value of pooling is a fast failure.

Because the pool decides *after* you ask, `create_endpoint.py` prints which type actually
landed (`describe_placement`) — a throughput number attributed to the type you requested
rather than the one that served it is simply wrong.

> **Quota note.** Endpoint-usage quota in this account is **1** for every `ml.g6e` size
> (and 0 for 24xlarge/48xlarge). That also means `UpdateEndpoint` cannot move a live g6e
> endpoint to a new config: blue/green needs `2 ×` the instance count at once, so the update
> is refused for want of capacity and the endpoint keeps serving the old config — i.e. the
> deploy looks like it did nothing. **Delete-and-recreate is the only path**, which is why a
> `config.pbtxt` sweep costs a full teardown per cell. A bump to 2 would unblock in-place
> updates *and* make the autoscaling policy more than decorative; it is the cheapest single
> unblock available here.

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

- **The `g6e` row was published as a lower bound, and that caveat was wrong** — see
  [the L40S ladder](#the-l40s-ladder-vcpu-is-not-the-lever) below. The reasoning was that a
  server-side p50 of only 209 ms meant the server was not deeply queued. It does not:
  `latency_ms` is timed from the top of `execute()`, so it is the cost of a *batch* and
  **excludes the dynamic batcher's queue wait entirely**. A server p50 that stops moving is
  evidence the batcher is always full, not evidence that it is empty. Re-measured from a
  dedicated 32-vCPU load generator, 4 vCPU reaches the same decisions/s as 64 vCPU at the
  same concurrency. 1–2 requests of ~770 did error in the `g6e` c=16/32 cells, undiagnosed;
  none of the ~45,000 requests in the ladder below errored.
- **Those are SageMaker *hosting* rates, not EC2 rates.** `g6e.xlarge` on EC2 on-demand is
  ~$1.86/hr; as a SageMaker endpoint it is $2.6054/hr. The EC2 number under-budgets by ~40%.
- **The endpoint-usage quota is per instance type and they differ.** In the account this was
  built in, `ml.g6.xlarge for endpoint usage` is **4** but **every `ml.g6e` size from
  `xlarge` to `16xlarge` is 1** (`24xlarge` and `48xlarge` are 0) — so on any `g6e`
  autoscaling has nowhere to go, and a config change cannot use `UpdateEndpoint` either,
  because blue/green needs a second instance the quota will not allow. Delete and recreate.
  `create_endpoint.py` reads the real quota and clamps, because `application-autoscaling`
  accepts an impossible maximum without complaint and records the failed scale-out only in a
  scaling activity log.

### The L40S ladder: vCPU is not the lever

The open question after the table above was whether 269.8 decisions/s was the card or the
4-vCPU host issuing kernel launches. **It is neither: it is `max_batch_size`.**

Measured through live endpoints, 7 questions, distinct tickets, from a **dedicated
`c7i.8xlarge` load generator** (32 vCPU, nothing else on it) with
`tools/bench_tickets.py --processes 16`. Every cell's load-generator CPU is in the table
because without it a throughput number is not a measurement of the server:

| instance | vCPU | $/hr | c | decisions/s | tickets/s | server p50 | e2e p50 | GPU | loadgen CPU | $/1,000 tickets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ml.g6e.4xlarge` | 16 | 3.7553 | 1 | 68.2 | 9.8 | 93 ms | 103 ms | — | 0.1% | — |
| | | | 8 | 214.2 | 30.6 | 121 ms | 245 ms | — | 0.2% | $0.0341 |
| | | | 32 | 277.6 | 39.7 | 205 ms | 828 ms | — | 0.3% | $0.0263 |
| | | | 128 | 311.2 | 44.5 | 205 ms | 3,318 ms | — | 0.5% | $0.0235 |
| | | | **192** | **325.8** | **46.6** | 205 ms | 4,979 ms | — | 0.5% | **$0.0224** |
| `ml.g6e.16xlarge` | 64 | 9.4715 | 1 | 69.6 | 10.0 | 92 ms | 101 ms | — | 0.1% | — |
| | | | 8 | 210.7 | 30.1 | 138 ms | 240 ms | — | 0.2% | $0.0874 |
| | | | 32 | 273–279 | 39.0–39.8 | 206 ms | 835 ms | 71–80% | 0.4% | $0.0663 |
| | | | 128 | 310.8 | 44.4 | 207 ms | 3,342 ms | — | 0.4% | $0.0593 |
| | | | **192** | **331.1** | **47.3** | 208 ms | 5,019 ms | — | 0.6% | **$0.0556** |

`ml.g6e.xlarge` (4 vCPU) and `ml.g6e.2xlarge` (8 vCPU) are **missing because they had no
capacity** — see [capacity](#capacity-instancepools-not-a-retry-loop).

**Read the two instances against each other and they are the same machine.** 16 vCPU and
64 vCPU agree to within noise at every concurrency — 311.2 against 310.8 at c=128, server
p50 205 ms against 207 ms — and both agree at c=32 with the 269.8 decisions/s already
published for **4** vCPU. Sixteen times the vCPU buys nothing. The hypothesis that the
4-vCPU host was the limit is dead.

**What the limit actually is.** At saturation server p50 pins at ~205 ms and never moves,
while end-to-end p50 rises exactly linearly with concurrency (828 → 1,670 → 3,318 →
4,979 ms at c=32/64/128/192). That is a queue in front of a server running flat out, and
the arithmetic closes: `max_batch_size: 8` × 7 questions = 56 rows per pass pair, 56 rows
in 205 ms is **273 decisions/s**, which is the c=32 number. The rest — the climb to ~331 at
c=192 — is a deeper queue keeping the batcher always full rather than occasionally
dispatching a short batch. So **throughput ≈ `max_batch_size` ÷ batch latency**, and the
next lever is batch geometry, not the host.

The 205 ms is the card: the same 56-row pass costs ~611 ms on an L4 (README's
`instance_group` table), and 611/205 = 2.98× against the L40S's 2.88× memory-bandwidth
advantage.

**Proof that the load generator was not the limit**, both tests, in every cell:

- load-generator CPU never exceeded **0.7%** of a 32-vCPU box while the server's
  end-to-end p50 was in seconds, and
- doubling the client processes at fixed offered load moved nothing: c=64 gave 289.4
  decisions/s at `--processes 16` and 289.4 at `--processes 32`; c=128 gave 310.8 and
  311.8. If the client had been the constraint, more of it would have bought more
  throughput.

**Where to operate.** c=8 is the interesting row: 214 decisions/s at a 245 ms end-to-end
p50, i.e. 65–69% of the ceiling for ~1/14th of the tail latency at c=192. Past c≈32 you are
buying throughput with seconds of queueing.

### `instance_group count` on an L40S: +7% at best, N× the latency always

On the L4 this was measured at **−10%** and the explanation was that the card was genuinely
saturated. On the L40S `count: 1` leaves the GPU at 71–80% with the host using ~1.5 of its
64 cores, so there was real headroom to aim at. All three rows are the **same instance
type** (`ml.g6e.16xlarge`), the same harness and the same load generator; only the overlaid
`config.pbtxt` differs. Decisions/s:

| c | `count: 1` | `count: 2` | `count: 4` |
| --- | --- | --- | --- |
| 1 | 69.6 | 68.6 | 63.0 |
| 8 | **210.7** | 194.6 | 140.3 |
| 16 | **273.0** | 205.1 | 255.8 |
| 32 | 273–279 | **292.2** | 140.7 ⁽*⁾ |
| 64 | 289.4 | 303.8 | **305.6** |
| 128 | 310.8 | **325.5–343.3** | 318.8 |
| 192 | 331.1 | 353.5 | **355.6** |
| 256 | not measured | 364.7 | **376.9** |
| server p50 at saturation | **205 ms** | 371–392 ms | 754–768 ms |
| server p95 at c=8 | **141 ms** | 225 ms | 1,822 ms |
| GPU util under load | 71–80% | 93–100% | 100% |
| GPU memory | 14.2% | 27.3% | 53.8% |
| $/1,000 tickets, best cell | $0.0556 | $0.0505 | $0.0490 |

⁽*⁾ a `preferred_batch_size` stall, not a `count: 4` property — see
[Known limits](#known-limits). Its p95 was 7.8 s.

**`count: 1` stays the default, and the table says why more plainly than the L4 one did.**
Going from 1 → 2 → 4 model copies buys +7% and then +0.6% at the very top of the
concurrency range, is **worse at every concurrency a latency-sensitive caller would use**
(0.67× at c=8 for `count: 4`), and multiplies the server-side p50 by the count — each
batcher runs its own full 56-row pass against a card it now shares. `count: 4`'s p95 at
c=8 is **13× `count: 1`'s**.

**The utilisation column is the lesson.** `count: 2` takes the GPU from 71–80% to 93–100%
and returns 7% more work; `count: 4` pins it at 100% and returns nothing further. The
20–25% of "idle" GPU was not 20–25% of available throughput: a filled pass is
memory-bandwidth-bound, so a second process competing for the same bandwidth mostly makes
both passes slower. The L4's −10% and the L40S's +7% are one mechanism with different
amounts of slack, which is why "GPU util is high" remains not-evidence that the GPU is the
bottleneck — and "GPU util is 75%" is not evidence that 25% is available either.

Memory never binds: 4 copies of the weights sit in 53.8% of the L40S's 45.8 GB.

### Capacity: `InstancePools`, not a retry loop

`ml.g6e` capacity in us-west-2 was the hard constraint on this ladder, not money or time.
Measured over one session: **six** single-instance-type `CreateEndpoint` attempts across
`xlarge`, `2xlarge` and `4xlarge` all failed `InsufficientInstanceCapacity`, each taking
~31 minutes to say so. `16xlarge` landed twice in ~8 minutes.

The remedy is in the error message AWS returns, and it works: a production variant may
carry up to **five** `InstancePools` entries with `Priority` 1–5, and SageMaker places the
first that has capacity. One pooled attempt at `[4xlarge, 2xlarge, xlarge]` landed
`4xlarge` immediately after three serial attempts had failed. `DescribeEndpoint` reports
which pool won under `ProductionVariants[].InstancePools[].InstanceType`.

Two constraints worth knowing: the list is capped at 5 (a 6th is a `ValidationException`),
and it needs a recent botocore — it is absent from boto3 1.42.97 and present in 1.43.110.

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
| `tools/batch_parity.py` | cross-request batching against the single-request path. Gate: **zero decision flips**; probability drift is advisory (≤7e-3 — batching changes bf16 reduction order). `--cuda-graphs` / `--cuda-graphs-two-pass` put the batched side on the graph path and additionally **fail if nothing was captured or replayed**, so a silent fallback cannot pass the gate. |
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
| `max_batch_size` | `config.pbtxt` | Requests coalesced per `execute()`. **8**, measured, and **finished** — 16 was deployed and is 1.6–12% *slower* at every concurrency while doubling server p50 (204 → 417 ms), and 32 is much worse. Pass time is linear in rows with no measurable fixed term, so a wider batch has nothing left to amortise. See [Batch geometry](#batch-geometry-max_batch_size-is-not-the-ceiling). |
| `max_queue_delay_microseconds` | `config.pbtxt` | 2 ms. Pure added latency for a request arriving into an empty queue, so keep it small relative to the work. |
| `preferred_batch_size` | `config.pbtxt` | `[4, 8]`. **Suspect.** Two cells of 17 on the L40S delivered 59–60% of their neighbours (165 against 273–311 decisions/s) at an unchanged server p50, which is what forming 5-request batches instead of 8 looks like. See [Known limits](#known-limits). |
| `max_queue_size` | `config.pbtxt` | 256, then reject. Bounded shedding beats unbounded latency — a load balancer can act on a refusal. |
| `instance_group count` | `config.pbtxt` | **1**. On the L4, 2 and 3 are *worse* (91.4 and 79.8 against 101.5). On an L40S with 64 vCPU, 2 is **0.75× below c≈32 and 1.07× above it**, 4 adds nothing beyond that, and each multiplies server p50 by the count — see [below](#instance_group-count-measured-1-wins) and [the L40S result](#instance_group-count-on-an-l40s-7-at-best-n-the-latency-always). Changing it needs no rebuild: pack `decider/` with the edited `config.pbtxt` and pass `--model-data-url`. |
| `max_rows` | `SD_MAX_ROWS` in `config.pbtxt` | 128 question rows per pass. Activation memory for the whole in-flight batch — measured at only ~6.5 GB of the L40S's 45.8 GB at `max_batch_size 8`, so memory is not what bounds it. Needs an image built from this commit or later; `model.py` refuses to start rather than serve a value it cannot honour. |
| `preferred_batch_size` | `config.pbtxt` | `[ 4, 8 ]`. `[ 8 ]` alone is a wash at saturation (±1%) and ~9% *worse* at concurrency 4, because a queue of four is then not a preferred size and waits out the queue delay. |
| `DUP_TOKEN_BUDGET` | `BatchedSystemOneEngine` | 480. Above this many duplicated state tokens, encoding the state once and forking the cache beats a single combined pass. |
| `SD_ENGINE`, `SD_PREFIX_CACHE` | container env | `merged` folds the LoRA into the torso (no PEFT at runtime). Prefix caching on. |
| `SD_FUSE_LAYERS` | container env | **0 (off)**. `1` swaps the torso's decoder layers for `flash-linear-attention`'s Triton kernels — see [Fused kernels](#fused-kernels-sd_fuse_layers1) below. Measured **1.29x** on a full 56-row batch (103.8 → 133.7 decisions/s) and *slower* at batch 1. Correctness gates pass; the numbers move. |
| `SD_CUDA_GRAPHS` | container env | `0`. `1` graphs the one-pass route (1 question 45.9 → 21.5 ms). `all` also graphs the state/row pair, which measured 0.73x–1.02x and is therefore not in `1`. Falls back to eager per shape. See [Known limits](#known-limits). |

## Batch geometry: `max_batch_size` is not the ceiling

The standing hypothesis was that throughput was limited by `max_batch_size`: 8 requests ×
7 questions = 56 rows against a `max_rows` budget of 128, so the engine's own row budget was
only 44% used, and widening the batch should buy amortisation. **It does not.** Measured on
one L40S (`ml.g6e.8xlarge`, 32 vCPU), in-region 32-vCPU load generator, 7 questions,
distinct tickets from a pool of 512, `reference_check` clean on every configuration.

**The cheap experiment first, because it needs no redeploy.** Rows per pass is
`max_batch_size × questions`, so holding `max_batch_size` at 8 and varying the question
count sweeps the row geometry for free:

| questions | rows/pass | server p50 | ms per row | decisions/s (c=32) |
| --- | --- | --- | --- | --- |
| 1 | 8 | 48.4 ms | 6.050 | 160.1 |
| 7 | 56 | 203.6 ms | 3.636 | 281.4 |
| 14 | 112 | 410.8 ms | 3.668 | 291.9 |
| 28 | 224 | 806.4 ms | 3.600 | 292.6 |

Fitting `time = a + b × rows` across the three shared-prefix rows gives `a` between −3.6 and
+15 ms against passes of 204–806 ms — **a fixed per-pass cost indistinguishable from zero** —
and `b ≈ 3.6 ms` per row. Rows per second is therefore constant, and decisions/s is flat at
~290 whether a pass carries 56, 112 or 224 rows.

The ~45 ms dispatch floor that motivated all of this is real, and the 8-row cell above *is*
it. But it is a **single-request** phenomenon, and at 56 rows it has already been amortised
to invisibility. There is nothing left for a wider batch to recover.

**Confirmed head-on**, with `max_batch_size: 16` actually deployed (112 rows at 7 questions):

| concurrency | 1 | 8 | 16 | 32 | 64 | 128 | 192 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **mbs 8** decisions/s | 68.6 | 217.0 | 266.7 | 281.4 | 291.6 | 314.3 | **336.3** |
| **mbs 16** decisions/s | 67.9 | 190.4 | — | 258.3 | 287.0 | 308.3 | 325.2 |
| **mbs 8** server p50 | 94 ms | 120 ms | 204 ms | 203 ms | 204 ms | 204 ms | 204 ms |
| **mbs 16** server p50 | 93 ms | 156 ms | — | 415 ms | 415 ms | 416 ms | 417 ms |

16 is slower everywhere and doubles server-side latency. **Keep 8.** It is the smallest
batch already on the linear part of the curve, so it collects the full amortisation at half
the latency of 16.

### So what *is* the ceiling?

The card — but not in the way the repo previously said. "A filled pass is
memory-bandwidth-bound" does not survive the arithmetic:

- **The card swap cannot discriminate.** L40S/L4 is 2.88× on bandwidth (864/300 GB/s) and
  2.99× on dense BF16 tensor cores (362/121 TFLOPS). The measured 2.98× fits both equally,
  so that experiment proves nothing about which.
- **Weight streaming is 2.3% of the pass.** 2B parameters × 2 bytes = 4.0 GB, which at
  864 GB/s is 4.6 ms against a measured 203.6 ms pass. Not a bandwidth wall — and that is
  the same fact as the linear row cost above, because weight-bound batching would be nearly
  free.
- **It is arithmetic-shaped at ~26% of peak.** A 56-row pass pushes ~4,808 input tokens in
  203.6 ms ≈ 23,600 tok/s, i.e. ~94 TFLOPS for a 2B model, against the L40S's 362 TFLOPS.

So there is roughly 4× of headroom still on this card, and it is reachable only through
**kernel efficiency** — `SD_FUSE_LAYERS`, CUDA graphs, a compiled torso — not through batch
geometry, which is now closed from both directions.

### Operating point, not just peak

Peak throughput here costs five seconds of queueing, which is not a recommendation:

| concurrency | decisions/s | server p50 | end-to-end p50 | $/1,000 tickets (g6e.4xlarge) |
| --- | --- | --- | --- | --- |
| 8 | 217.0 | 120 ms | **242 ms** | $0.0336 |
| 16 | 266.7 | 204 ms | 411 ms | $0.0274 |
| 32 | 281.4 | 203 ms | 820 ms | $0.0259 |
| 64 | 291.6 | 204 ms | 1,645 ms | $0.0250 |
| 192 | 336.3 | 204 ms | 4,932 ms | $0.0217 |

**Concurrency 16 is the recommendation**: 79% of peak throughput at 8% of peak tail latency.
Concurrency 8 is the pick if 250 ms matters more than money.

Note the trap in that table, and it has already caused one wrong conclusion in this project:
**server p50 is flat at 204 ms from concurrency 16 upward while end-to-end p50 grows 12×.**
`latency_ms` is timed from the top of `execute()`, which Triton calls *after* batch
formation, so it excludes queue wait entirely. A flat server p50 means the batcher is
**always full**, i.e. saturated — not that there is headroom. Judge saturation from
end-to-end latency rising at constant throughput.

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

**Retested with more vCPU, and the penalty was the batch geometry, not the CPU.** On a
64-vCPU host (`ml.g6.16xlarge`, same single L4), holding `count × max_batch_size × 7 = 56`
rows constant:

| `count` | `max_batch_size` | peak decisions/s | server p50 |
| --- | --- | --- | --- |
| 1 | 8 | 103.8 | 618 ms |
| 2 | 4 | 103.1 | 601 ms |
| 4 | 2 | 95.2 | 665 ms |

So `count: 2` with `max_batch_size` moved to match is **completely neutral** — the 10%
loss and doubled p50 seen at `count: 2, mbs: 8` were the halved batch, and removing the
geometry change removes the whole penalty. The batch-splitting mechanism above is
confirmed; the 4-vCPU contention story cannot be separated out and may have contributed.

But `count: 4` still loses ~8% with rows held constant, which sharpens the rule: what
matters is rows **per pass**, not rows in total. Four passes of 14 rows pay the fixed
per-pass cost four times and no amount of vCPU fixes that. So keeping
`count × max_batch_size × questions` at `max_rows` is **necessary but not sufficient** —
dropping `max_batch_size` below ~8 is itself the harm. `count: 1` stays the default;
`count: 2` is a free no-op rather than a win.

That revisit has now happened on an L40S with 64 vCPU, which is the most favourable host
this fleet offers, and `count: 2` is worth **+7% at deep saturation and −25% at moderate
load**, with `count: 4` adding nothing further —
[the L40S result](#instance_group-count-on-an-l40s-7-at-best-n-the-latency-always). More
vCPU was not the missing ingredient; it was never the constraint.

## Known limits

- **`latency_ms` does not include the queue, so it cannot tell you whether the server is
  saturated.** It is timed from the top of `execute()`, which Triton calls *after* the
  dynamic batcher has formed a batch. Under load it therefore converges on the cost of one
  full pass pair and stays there — 205 ms on an L40S, 611 ms on an L4 — however deep the
  queue gets. An earlier version of this README read a flat 209 ms as "the server was not
  deeply queued" and published a real ceiling as a lower bound because of it. The two
  signals that actually answer the question are **end-to-end latency rising linearly at
  constant throughput** (a queue in front of a server running flat out) and **load-generator
  CPU** (`loadgen_cpu_pct`, in every `bench_tickets.py` row).
- **Throughput on a fast card is set by `max_batch_size`, not by the host.** 16 vCPU and 64
  vCPU deliver the same decisions/s on an L40S at every concurrency, and both match the
  4-vCPU figure at the same concurrency. 56 rows in 205 ms is 273 decisions/s and that is
  what the card delivers. See [the L40S ladder](#the-l40s-ladder-vcpu-is-not-the-lever).
  Raising `max_batch_size` together with `max_rows` is the untried lever this points at, and
  the one caution is that 32 was already tried *without* raising `max_rows` and was much
  worse.
- **The dynamic batcher looks bistable at `preferred_batch_size: [4, 8]`.** Two cells out of
  17 across two instance types came in at 23.4 and 23.7 tickets/s where their neighbours
  (and a repeat of the same cell seconds later) gave 39–44, with server p50 unchanged at
  205 ms and zero errors. A constant batch cost at 60% of the throughput means 60% of the
  batch size, i.e. ~5 requests per pass instead of 8. Not diagnosed; the cheap experiment is
  `preferred_batch_size: [ 8 ]`, so the batcher has one target instead of two.
- **No pushed image supports `SD_FUSE_LAYERS` or `SD_CUDA_GRAPHS`.** Both knobs are read by
  `model.py` and both were added *after* the newest image in ECR
  (`v23-triton-onepass`, 2026-10-07) was built, so the fused-kernel and CUDA-graph numbers
  in this README are in-process measurements from the build box and **have never been
  measured through a deployed endpoint**. Doing so needs a rebuild.
- **A model-repository overlay replaces the repository, not the package it imports.**
  `--model-data-url` is extracted over `/opt/ml/model`, so the `decider/1/model.py` in the
  archive must be the same vintage as the `strands_decider` baked into the image. Shipping
  HEAD's `model.py` against an older image fails `initialize()` with
  `TypeError: load_merged_engine() got an unexpected keyword argument`, the container never
  passes `/ping`, and the endpoint fails ~30 minutes later with "did not pass the ping
  health check". Take both files from the commit that built the image.
- **Latency is dispatch-bound; throughput at saturation is not.** One *small* forward pass
  has a ~45 ms floor on an L4 that is CPU kernel-launch overhead (~5,676 launches) against a
  12.7 ms weight-streaming floor, and that floor is what a single request pays — cutting it
  is the largest remaining win for latency. But once the batcher
  has filled a pass with 56 rows the pass leaves that regime, and a **loaded** server is
  bound by the card: an L40S with 2.88x the memory bandwidth delivers 2.7x the decisions/s
  on the same configuration. An earlier version of this README claimed throughput was
  dispatch-bound full stop; that was an over-generalisation from concurrency-1 profiling.
- **CUDA graph capture exists for the one-pass route and is opt-in: `SD_CUDA_GRAPHS=1`.**
  It is off by
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
- **The two regimes reward opposite optimisations.** Fused kernels (`SD_FUSE_LAYERS=1`) do
  less arithmetic and move less data but carry more Python per op, so they help a full pass
  (1.07-1.42x at 56 rows) and *hurt* a small one (0.85x at batch 1). Choose for the regime
  you are in.
- **`instance_group count: 2` is measured and buys nothing.** It is neutral when
  `max_batch_size` moves with it and 10% worse when it does not; `count: 4` loses ~8% even
  with rows held constant. See [above](#instance_group-count-measured-1-wins).
- **More vCPU does not help, and this is measured, not assumed.** On a fixed single L4,
  throughput is flat across a **16x** span of vCPU: **101.5 → 102.9 → 103.8 decisions/s at
  4 → 32 → 64 vCPU**, and equally flat at concurrency 1 (63.4 → 64.4 → 62.9) with the
  single-pass floor unmoved at 45 → 43 → 45 ms. The reason is in the sentence above: the
  ~5,676 launches are issued **single-threaded per stub**, and a serial instruction stream
  does not go faster on more cores. Extra vCPU can only relieve *contention* between stubs,
  `tritonserver` and a co-resident load generator — so it buys nothing once the load
  generator is on its own host. An earlier version of this section suggested buying vCPU to
  issue the dispatch stream faster; that was wrong.

  It is worse than neutral on cost, because price is not flat: **$/1,000 tickets degrades
  3.7x, $0.022 → $0.080**, from 4 to 64 vCPU. **On a fixed GPU, buy the smallest vCPU count
  that fits.** The levers that remain all attack the dispatch floor itself (`SD_CUDA_GRAPHS`)
  or the GPU work (`SD_FUSE_LAYERS`, a faster card).
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
