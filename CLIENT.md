# Calling the endpoint

Everything another project needs to call `strands-decider-2B-hobson-v21` on the live
SageMaker endpoint. Verified against the running endpoint on 2026-10-08.

> [!WARNING]
> **This is an experimental endpoint. Do not build anything you care about on top of it.**
>
> It is one `ml.g6.xlarge` instance with no SLA, no support and no uptime commitment, run
> for experimentation. **It may be shut down at any time, without notice** — GPU hosting
> costs ~$27/day whether anyone calls it or not, so it will not stay up indefinitely. Make
> your client fail clearly rather than hang if it disappears (see
> [§7, Checking it's alive](#7-checking-its-alive)).
>
> The *contract* below is stable and worth coding against — it is the same JSON the
> non-Triton server speaks. The *deployment* is not. If you need something durable, deploy
> your own from this repo rather than depending on this instance.

> **There is no plain URL you can `curl`.** This is a SageMaker real-time endpoint: every
> request must be SigV4-signed with AWS credentials that hold `sagemaker:InvokeEndpoint`.
> Use an AWS SDK (`boto3`, `@aws-sdk/client-sagemaker-runtime`, …) and give it the
> **endpoint name**, not a URL.

---

## 1. Coordinates

| | |
| --- | --- |
| **Endpoint name** | `strands-decider-g6` |
| **Region** | `us-west-2` |
| **Account** | `<ACCOUNT_ID>` — ask whoever owns the deployment, or run `aws sts get-caller-identity --query Account --output text` while assuming the hosting account's role |
| **Endpoint ARN** | `arn:aws:sagemaker:us-west-2:<ACCOUNT_ID>:endpoint/strands-decider-g6` |
| **Content-Type** | `application/json` |
| **Underlying URL** | `https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/strands-decider-g6/invocations` |

`<ACCOUNT_ID>` is deliberately not written down here, so this file can be shared without
carrying the hosting account's identity. You only need it for the IAM policy below — the
client code never references it, because `boto3` resolves the account from the caller's
own credentials.

The URL is given for completeness — if you are signing SigV4 by hand, that is the path.
Service name for signing is `sagemaker`. Almost nobody should do this; use the SDK.

### IAM the caller needs

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "sagemaker:InvokeEndpoint",
    "Resource": "arn:aws:sagemaker:us-west-2:<ACCOUNT_ID>:endpoint/strands-decider-g6"
  }]
}
```

If the calling project runs in a different AWS account, it needs a role in the hosting
account that it can assume — `sagemaker:InvokeEndpoint` is not cross-account by default,
and SageMaker endpoints take no resource-based policy, so assuming a role is the only
route.

---

## 2. The one thing that trips everyone up

The decider's own request body is **not** what you send. SageMaker's `/invocations` on the
Triton DLC is a thin proxy to Triton's KServe v2 `infer` API, so the body travels **inside
a tensor envelope**:

```
┌─ what you POST ──────────────────────────────────────────────┐
│ {"inputs": [{"name": "REQUEST_JSON",                         │
│              "shape": [1, 1],                                │
│              "datatype": "BYTES",                            │
│              "data": ["<the decider request, as a STRING>"]}]}│
└──────────────────────────────────────────────────────────────┘
                         │
                         ▼  the decider request, JSON-encoded into that string
          {"state": "...", "questions": {...}}
```

Two details that are not negotiable:

- **`data[0]` is a JSON *string*, not a nested object.** You `json.dumps` the decider
  request, then put that string in the list.
- **`shape` is `[1, 1]`, not `[1]`.** The model declares `max_batch_size > 0`, so Triton
  prepends a batch dimension. Sending `[1]` is rejected.

Sending a bare decider body gets you an opaque HTTP 500:
`Unable to parse 'inputs': attempt to access non-existing object member 'inputs'`.

The response comes back the same way, under `RESPONSE_JSON` — and note the asymmetry,
**the response output's `shape` is `[1]`**:

```json
{"model_name": "decider", "model_version": "1",
 "outputs": [{"name": "RESPONSE_JSON", "datatype": "BYTES", "shape": [1],
              "data": ["<the decider response, as a string>"]}]}
```

---

## 3. Drop-in client

Copy this into the calling project. No dependency on this repo.

```python
"""Client for the strands-decider SageMaker endpoint."""

from __future__ import annotations

import json
from typing import Any

import boto3
from botocore.config import Config

ENDPOINT = "strands-decider-g6"
REGION = "us-west-2"


class DeciderError(RuntimeError):
    """The endpoint answered, but with an error body. See `.kind`."""

    def __init__(self, message: str, kind: str) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind          # "invalid_request" (your payload) | "internal" (the service)


def make_client(region: str = REGION, max_in_flight: int = 64):
    """One client, reused. The pool size is load-bearing: botocore's default is 10, which
    silently serialises anything above 10 concurrent calls and makes you measure the client
    rather than the model."""
    return boto3.client(
        "sagemaker-runtime",
        region_name=region,
        config=Config(
            max_pool_connections=max(10, max_in_flight),
            retries={"max_attempts": 2, "mode": "standard"},
            # SageMaker's own hard ceiling for a real-time invocation is 60 s.
            read_timeout=60,
            connect_timeout=10,
        ),
    )


def decide(client, state: Any, questions: dict, *, endpoint: str = ENDPOINT) -> dict:
    """Ask `questions` about `state`. Returns the decider response body.

    `state` may be a string, or a dict/list (rendered deterministically server-side).
    Raises DeciderError for a caller or server error, ClientError for a transport failure.
    """
    inner = json.dumps({"state": state, "questions": questions})
    envelope = {
        "inputs": [{
            "name": "REQUEST_JSON",
            "shape": [1, 1],            # NOT [1] -- Triton prepends the batch dimension
            "datatype": "BYTES",
            "data": [inner],            # a JSON *string*, not a nested object
        }]
    }

    resp = client.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Body=json.dumps(envelope),
    )
    out = json.loads(resp["Body"].read().decode("utf-8"))

    # A Triton-level failure (bad envelope) has no `outputs`.
    if "error" in out and "outputs" not in out:
        raise DeciderError(str(out["error"]), "envelope")

    for tensor in out.get("outputs") or []:
        if tensor.get("name") == "RESPONSE_JSON":
            data = tensor.get("data") or []
            if not data:
                raise DeciderError("RESPONSE_JSON carried no data", "internal")
            body = json.loads(data[0])
            # The service reports caller errors in the body at HTTP 200, deliberately, so a
            # malformed payload is distinguishable from the model being down.
            if "error" in body:
                err = body["error"]
                raise DeciderError(err.get("message", "unknown"),
                                   err.get("type", "invalid_request"))
            return body

    raise DeciderError(f"no RESPONSE_JSON in {sorted(out)}", "internal")
```

### Using it

```python
client = make_client()

answers = decide(client,
    state="Help! My payouts have been failing for 3 days!",
    questions={
        "urgency":     {"type": "noul",
                        "instructions": "Does this convey urgency?"},
        "team":        {"type": "choice",
                        "instructions": "Which team should handle this?",
                        "criteria": {"billing": None, "sales": None, "retail": None}},
        "frustration": {"type": "score",
                        "instructions": "How frustrated is the writer?",
                        "criteria": ["calm", "frustrated", "depressed"]},
    })

print(answers["answers"]["team"]["choice"])                  # billing
print(answers["answers"]["team"]["confidence"])              # 0.8367
print(answers["answers"]["urgency"]["noul"])                  # 0.8748
```

---

## 4. Request shape

```json
{
  "state": "<string, or any JSON object/array>",
  "questions": {
    "<your name for it>": { ... one of the three types below ... }
  }
}
```

`state` may be `""` if the question carries the whole task. `questions` must hold at least
one entry. Question names are yours and appear verbatim as the keys of `answers`.

| type | required fields | `criteria` |
| --- | --- | --- |
| `noul` | `instructions` | optional `{"true": "...", "false": "..."}` to sharpen the boundary |
| `choice` | `instructions`, `criteria` | `{name: description-or-null}`, **2 to 255** options |
| `score` | `instructions`, `criteria` | an **ascending** list of 2 to 10 level names; index 0 is the low end |

A `choice` description may be `null` for a bare label, or structured data (a dict/list),
which is rendered deterministically.

**Ask many questions in one call.** This is the whole point of the deployment: extra
questions about the same state are nearly free (1 question ≈ 47 ms, 7 questions ≈ 67 ms).
Do not loop one question per request — that is up to 7x the cost for the same answers.

---

## 5. Response shape

Verified live. One entry in `answers` per question, keyed by your name:

```json
{
  "model": "strands-decider-triton",
  "answers": {
    "urgency": {"type": "noul", "noul": 0.8748},

    "team": {"type": "choice", "choice": "billing", "confidence": 0.8367,
             "probabilities": {"billing": 0.8912, "sales": 0.0564, "retail": 0.0524}},

    "frustration": {"type": "score", "score": 1.0713, "confidence": 0.6009,
                    "legend": {"0": "calm", "1": "frustrated", "2": "depressed"},
                    "probabilities": {"0": 0.1404, "1": 0.6479, "2": 0.2117}}
  },
  "usage": {"input_tokens": 238, "output_tokens": 3},
  "latency_ms": 49.15
}
```

| field | meaning |
| --- | --- |
| `noul` | `P(statement is true)`. There is no confidence field — with two outcomes the probability *is* the uncertainty |
| `choice` | the argmax option name. `probabilities` covers every option and sums to 1 |
| `score` | the **expected** level index, so `1.07` means "mostly level 1, leaning 2". Not an argmax — it can be fractional |
| `legend` | what each `score` level index meant, in the order you declared them |
| `confidence` | derived from the distribution, not a second prediction. `choice`: normalised max-probability (uniform → 0, one-hot → 1), stable across option counts. `score`: normalised standard deviation, so mass on *adjacent* levels is confident, mass at both ends is not |
| `latency_ms` | server-side only; excludes your round trip |

All probabilities are rounded to 4 decimal places.

### Errors

A caller error returns **HTTP 200 with an error body**, deliberately, so your bad payload
cannot be confused with the service being down:

```json
{"error": {"type": "invalid_request",
           "message": "request was not valid JSON: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"}}
```

| `type` | meaning | what to do |
| --- | --- | --- |
| `invalid_request` | malformed JSON, schema violation, too many options, prompt truncated through its option list | fix the payload; retrying will not help |
| `internal` | the inference failed | retry with backoff; alert |
| `envelope` *(client-side label)* | Triton could not parse the envelope | your `inputs` wrapper is wrong — check `shape: [1, 1]` |

A malformed request from one caller **does not** affect other requests batched alongside
it. Transport-level problems (throttling, endpoint not in service) surface as a botocore
`ClientError`, not as an error body.

---

## 6. Operational notes for the calling project

**Latency.** ~50–70 ms server-side for a typical request, plus ~8–10 ms if you call from
inside `us-west-2`. From outside the region the round trip dominates — the same request
measuring 44 ms server-side takes 500–630 ms from a laptop. **Co-locate if latency
matters.**

**Concurrency is where this endpoint earns its keep.** Requests arriving together are
coalesced into one GPU pass (up to 8), so throughput improves markedly under concurrent
load: ~99 decisions/s (~14 requests/s at 7 questions) at saturation on the single instance.
At concurrency 1 you get none of that. Send concurrently if you have batch work.

**Back-pressure.** The queue is bounded at 256 and then **rejects** rather than queueing
further. Treat a rejection as "slow down", not as a failure — retry with backoff.

**Under heavy load, latency becomes queueing.** At 32 requests in flight a 7-question
request sits at ~615 ms p50. Set your client timeout accordingly (60 s is SageMaker's hard
ceiling anyway).

**Limits.** Max request payload 6 MB; max invocation duration 60 s. The model's context
window is 3,072 tokens — longer states are truncated from the end, and the *question*
always keeps its reserve, so your options and the answer marker are never lost.

**Capacity today.** One `ml.g6.xlarge` with autoscaling 1→4 instances, target tracking at
600 invocations/instance/minute. Two things to know: a new instance takes **~12 minutes**
to come up (GPU cold start, 20.6 GB image pull), so this does not absorb a sudden spike;
and 600/min is ~71% of one instance's measured ceiling, so scale-out is requested fairly
late. If this project is going to drive sustained load, lower the target to ~200–250 first.

**Text only.** The `images` field exists in the schema but a text-only engine refuses it
rather than ignoring it. Don't send images.

---

## 7. Checking it's alive

The endpoint has no public health URL. Check status with the control-plane API:

```bash
aws sagemaker describe-endpoint \
  --endpoint-name strands-decider-g6 --region us-west-2 \
  --query '{status:EndpointStatus,instances:ProductionVariants[0].CurrentInstanceCount}'
```

`EndpointStatus` must be `InService`. Anything else (`Creating`, `Updating`, `Failed`,
or a `ValidationException` meaning it does not exist) and invocations will fail — this is
a dev endpoint and **it may be torn down between uses**, so make the calling project fail
clearly rather than hang if it is gone.
