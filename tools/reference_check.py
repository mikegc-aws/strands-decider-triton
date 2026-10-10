"""Check the served model against a named checkpoint's published reference values.

Pick the checkpoint with `--model` (default `v21`), or ask which one is deployed with
`--identify`.

The strongest correctness check available without a second deployable: the model's own
README records exact outputs for three specific requests against
`strands-decider-2B-hobson-v21`. If this deployable reproduces them, then the whole chain
is right -- prompt rendering, the window fit, option spans, the pointer readout, the fitted
temperatures, the confidence formulas, the merged torso, and the Triton wire format.

Published in `README.md` of strands-labs/strands-decider for v21, state
"Help! My payouts have been failing for 3 days! ":

    noul  "Does this convey urgency?"                          -> 0.875
    choice "Which team should handle this?=billing,sales,retail"
             -> billing (confidence 0.835); billing 0.890, sales 0.056, retail 0.054
    score  "How frustrated is the writer?=calm,frustrated,depressed"
             -> 1.07 (confidence 0.602); 0: 0.140, 1: 0.648, 2: 0.212

Tolerance. These are NOT expected to match to the last digit and the README says as much --
it records that a retrained checkpoint gives "somewhat different numbers with the same
answers", and that answers differ in the final digits across kernel paths (`noul` 0.784 vs
0.7875). The gate here is therefore the same one used everywhere else in this project:

    the DECISION must match exactly       (argmax option, side of 0.5, rounded level)
    |delta p| must be within --warn        (default 0.02)

0.02 is chosen to be looser than the 0.0078 that `ab.py` measured for the LoRA merge plus
the ~0.0065 recorded across kernel paths, with headroom -- it is a smoke gate, not a
calibration test.

**The probability check is a gate, not an advisory, and that is a change.** It used to
warn. It had to become a gate because this tool now knows six checkpoints and all six
return the SAME three decisions for these requests -- so "0 decision mismatches" is true
of every one of them and identifies none. Confirmed the expensive way on 2026-10-10: a
freshly built qwen3.5-v1 endpoint was gated against v21's numbers and printed
`PASS: 0 decision mismatch(es)` with p[billing] off by 0.116. The decisions were genuinely
right; the verdict line was just not scoped to the model under test.

Adjacent checkpoints differ by 0.05-0.12 on these probabilities and drift within one
checkpoint is under 0.008, so 0.02 separates "wrong model" from "same model, different
card" with an order of magnitude to spare in both directions.

Usage:
    reference_check.py --base http://localhost:8100 --path /invocations --triton
    reference_check.py --endpoint sd-l40s-2xl --region us-west-2
    reference_check.py --endpoint sd-g6 --model qwen3.5-v1
    reference_check.py --endpoint sd-g6 --identify

`--endpoint` exists because this gate has to be runnable against a *card*, and some of them
only come as a SageMaker endpoint. The L40S instance types (`ml.g6e.*`) are hosted, not
shelled into: there is no container port to point `--base` at, so before this flag the
project's governing rule -- zero decision mismatches before any throughput number is
trusted -- could not be applied to a new instance type at all. The decision gate is the
thing that matters here rather than the transport: bf16 reduction order depends on the
kernels the card selects, so "it was right on an L4" is not evidence about an L40S.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

STATE = "Help! My payouts have been failing for 3 days! "

# The three requests. Identical for every checkpoint -- only the expected answers differ,
# which is what makes them usable both as a gate and as a fingerprint (see `--identify`).
CASES: list[dict] = [
    {
        "name": "noul_urgency",
        "questions": {"urgency": {"type": "noul",
                                  "instructions": "Does this convey urgency?"}},
    },
    {
        "name": "choice_team",
        "questions": {"team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            # Bare labels, no descriptions -- matching the CLI form `=billing,sales,retail`.
            "criteria": {"billing": None, "sales": None, "retail": None},
        }},
    },
    {
        "name": "score_frustration",
        "questions": {"frustration": {
            "type": "score",
            "instructions": "How frustrated is the writer?",
            "criteria": ["calm", "frustrated", "depressed"],
        }},
    },
]


def _expect(noul, conf_c, billing, sales, retail, score, conf_s, p0, p1, p2) -> dict:
    """One checkpoint's published answers to the three CASES above, in README order."""
    return {
        "noul_urgency": {"urgency": {"kind": "noul", "noul": noul}},
        "choice_team": {"team": {
            "kind": "choice", "choice": "billing", "confidence": conf_c,
            "probabilities": {"billing": billing, "sales": sales, "retail": retail}}},
        "score_frustration": {"frustration": {
            "kind": "score", "score": score, "confidence": conf_s,
            "probabilities": {"0": p0, "1": p1, "2": p2}}},
    }


# Transcribed from the "Use" section of each checkpoint's own README on the Hub, verified
# against the published files on 2026-10-10. All six give the SAME three decisions, which is
# exactly why a decision-only gate cannot tell them apart -- see `--identify`.
#
# The v21 row corrects four digits this file previously carried (confidence 0.835 -> 0.837,
# billing 0.890 -> 0.891, sales 0.056 -> 0.055, score confidence 0.602 -> 0.603, p[1] 0.648
# -> 0.649, p[2] 0.212 -> 0.211). All within 0.002, so no gate result changes; the published
# README is the authority and now matches.
REFERENCES: dict[str, dict] = {
    "v21":               _expect(0.875, 0.837, 0.891, 0.055, 0.054, 1.07, 0.603, 0.140, 0.649, 0.211),
    "qwen3.5-v1":        _expect(0.858, 0.659, 0.773, 0.121, 0.106, 1.03, 0.509, 0.196, 0.574, 0.230),
    "E2B-gemma4-v1":     _expect(0.863, 0.903, 0.935, 0.038, 0.027, 1.01, 0.521, 0.202, 0.585, 0.212),
    "E4B-gemma4-v1":     _expect(0.959, 0.951, 0.967, 0.019, 0.013, 0.97, 0.848, 0.102, 0.823, 0.075),
    "12B-gemma4-v1":     _expect(0.956, 0.909, 0.939, 0.040, 0.021, 0.99, 0.799, 0.106, 0.794, 0.101),
    "26B-A4B-gemma4-v1": _expect(0.981, 0.712, 0.808, 0.127, 0.065, 1.00, 0.828, 0.094, 0.812, 0.094),
}

# Hub ids, so the banner can name what was actually expected rather than a short key.
HUB_IDS: dict[str, str] = {
    "v21": "StrandsAgents/strands-decider-2B-hobson-v21",
    "qwen3.5-v1": "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610",
    "E2B-gemma4-v1": "StrandsAgents/strands-decider-E2B-gemma4-v1-2610",
    "E4B-gemma4-v1": "StrandsAgents/strands-decider-E4B-gemma4-v1-2610",
    "12B-gemma4-v1": "StrandsAgents/strands-decider-12B-gemma4-v1-2610",
    "26B-A4B-gemma4-v1": "StrandsAgents/strands-decider-26B-A4B-gemma4-v1-2610",
}


def envelope(payload: dict) -> dict:
    """Wrap a System One request in the KServe v2 envelope.

    `shape: [1, 1]` because the model sets `max_batch_size > 0` and Triton prepends the
    batch dimension. A hand-rolled `[1]` is the most common way to get an opaque
    "Unable to parse 'inputs'" out of this endpoint.
    """
    return {"inputs": [{"name": "REQUEST_JSON", "shape": [1, 1],
                        "datatype": "BYTES", "data": [json.dumps(payload)]}]}


def unwrap(out: dict) -> dict:
    if "error" in out and "outputs" not in out:
        raise ValueError(f"Triton error: {out['error']}")
    for o in out.get("outputs") or []:
        if o.get("name") == "RESPONSE_JSON":
            return json.loads(o["data"][0])
    raise ValueError(f"no RESPONSE_JSON in {sorted(out)}")


def post(url: str, payload: dict, triton: bool, timeout: float = 300.0) -> dict:
    if triton:
        payload = envelope(payload)
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode())
    if not triton:
        return out
    return unwrap(out)


def post_endpoint(endpoint: str, region: str, payload: dict,
                  timeout: float = 300.0) -> dict:
    """The same request through `InvokeEndpoint`, for a card with no reachable HTTP port.

    boto3 is imported here rather than at module scope on purpose: the HTTP path needs
    nothing but the standard library, and a laptop checking a local container should not be
    made to install boto3 to do it.

    Retries are off. A retried request would turn a server error into latency, and this is
    a correctness gate -- a request that had to be retried to succeed is a result worth
    seeing, not one worth hiding.
    """
    import boto3
    from botocore.config import Config

    rt = boto3.client("sagemaker-runtime", region_name=region,
                      config=Config(read_timeout=timeout, connect_timeout=30,
                                    retries={"max_attempts": 0}))
    raw = rt.invoke_endpoint(EndpointName=endpoint, ContentType="application/json",
                             Body=json.dumps(envelope(payload)))["Body"].read().decode()
    return unwrap(json.loads(raw))


def deltas(answer: dict, expect: dict) -> list[float]:
    """Every |observed - published| this pair can produce, decisions aside.

    Used two ways: as the gate's drift measure, and by `--identify` to score the response
    against every known checkpoint. Kept free of printing so both callers can share it.
    """
    out: list[float] = []
    kind = expect["kind"]
    if answer.get("type") != kind:
        return [float("inf")]
    if kind == "noul":
        out.append(abs(float(answer["noul"]) - expect["noul"]))
    else:
        if kind == "score":
            out.append(abs(float(answer["score"]) - expect["score"]))
        out.append(abs(float(answer["confidence"]) - expect["confidence"]))
    got_probs = answer.get("probabilities") or {}
    for label, want_p in (expect.get("probabilities") or {}).items():
        if label not in got_probs:
            out.append(float("inf"))
        else:
            out.append(abs(float(got_probs[label]) - want_p))
    return out


def check(answer: dict, expect: dict, warn: float) -> tuple[bool, bool, list[str]]:
    """Returns (decision_ok, probabilities_ok, notes).

    Both are gates now. The decision check is unchanged and remains the primary signal.
    The probability check is new: a drift above `warn` FAILS instead of printing an
    advisory, because a decision-only gate cannot distinguish the six published
    checkpoints -- all six give these same three decisions, so the old verdict line read
    PASS against v21's numbers while serving any of them. Confirmed live on 2026-10-10
    serving qwen3.5-v1, where p[billing] was off by 0.116 and the gate still passed.

    The threshold stays at the 0.02 this file already chose, which is comfortably looser
    than the 0.0078 measured for the LoRA merge plus ~0.0065 across kernel paths, and far
    tighter than the 0.05-0.12 that separates adjacent checkpoints. Served qwen3.5-v1
    matched its OWN published values to 0.002, so the headroom is real in both directions.
    """
    notes: list[str] = []
    kind = expect["kind"]
    assert answer.get("type") == kind, f"type {answer.get('type')} != {kind}"

    decision_ok = True
    if kind == "noul":
        got, want = float(answer["noul"]), expect["noul"]
        if (got >= 0.5) != (want >= 0.5):
            decision_ok = False
            notes.append(f"DECISION DIFFERS: P(true) {got} vs published {want} "
                         "(opposite sides of 0.5)")
        notes.append(f"noul {got:.4f} vs {want} (delta {abs(got - want):.4f})")
    elif kind == "choice":
        if answer["choice"] != expect["choice"]:
            decision_ok = False
            notes.append(f"DECISION DIFFERS: chose {answer['choice']!r}, "
                         f"published {expect['choice']!r}")
        else:
            notes.append(f"choice {answer['choice']} (matches)")
        notes.append(f"confidence {answer['confidence']:.4f} vs {expect['confidence']}")
    else:
        got, want = float(answer["score"]), expect["score"]
        if round(got) != round(want):
            decision_ok = False
            notes.append(f"DECISION DIFFERS: score rounds to {round(got)}, "
                         f"published rounds to {round(want)}")
        notes.append(f"score {got:.4f} vs {want} (delta {abs(got - want):.4f})")
        notes.append(f"confidence {answer['confidence']:.4f} vs {expect['confidence']}")

    for label, want_p in (expect.get("probabilities") or {}).items():
        got_p = float((answer.get("probabilities") or {}).get(label, float("nan")))
        delta = abs(got_p - want_p)
        flag = "  <-- OVER TOLERANCE" if delta > warn else ""
        notes.append(f"  p[{label}] {got_p:.4f} vs {want_p} (delta {delta:.4f}){flag}")

    worst = max(deltas(answer, expect), default=0.0)
    probabilities_ok = worst <= warn
    if not probabilities_ok:
        notes.append(f"PROBABILITIES DIFFER: worst delta {worst:.4f} > tolerance {warn} "
                     "-- either this is not the checkpoint named above, or something in "
                     "the chain changed. Run --identify.")
    return decision_ok, probabilities_ok, notes


def fetch_all(args, url: str) -> dict:
    """Run the three CASES once and return `{case_name: answers}`.

    One pass, shared by the gate and `--identify`, so identifying a checkpoint costs the
    same three requests as checking one.
    """
    collected: dict[str, dict] = {}
    for case in CASES:
        request = {"state": STATE, "questions": case["questions"]}
        if args.endpoint:
            body = post_endpoint(args.endpoint, args.region, request)
        else:
            body = post(url, request, args.triton)
        if "error" in body:
            raise ValueError(f"server error: {body['error']}")
        collected[case["name"]] = body["answers"]
    return collected


def identify(args, url: str, target: str) -> int:
    """Rank every known checkpoint by how well it explains the observed answers.

    This exists because the decision gate cannot do it. All six published checkpoints
    return the same three decisions for these requests, so "0 decision mismatches" is true
    of every one of them and says nothing about which is deployed. The probabilities DO
    separate them -- adjacent checkpoints differ by 0.05-0.12 where kernel and merge drift
    is under 0.008 -- so the worst per-checkpoint delta is a usable fingerprint.

    Reports the margin as well as the winner. A small margin means the fingerprint did not
    actually discriminate, which is a different statement from "it is this one" and should
    not be read as the latter.
    """
    print(f"[ref] {target}")
    print("[ref] identifying: scoring against every published checkpoint\n")
    try:
        observed = fetch_all(args, url)
    except Exception as exc:
        print(f"[ref] REQUEST FAILED: {exc}")
        print("[ref] REFERENCE_CHECK_FINISHED")
        return 1

    scores: list[tuple[float, str]] = []
    for model, references in REFERENCES.items():
        worst = 0.0
        for case_name, expectations in references.items():
            for qname, expect in expectations.items():
                answer = observed[case_name].get(qname)
                if answer is None:
                    worst = float("inf")
                    continue
                worst = max(worst, max(deltas(answer, expect), default=0.0))
        scores.append((worst, model))
    scores.sort()

    for worst, model in scores:
        mark = "  <== best match" if model == scores[0][1] else ""
        print(f"[ref]   {model:<20} worst delta {worst:7.4f}{mark}")

    best_delta, best_model = scores[0]
    runner_up = scores[1][0] if len(scores) > 1 else float("inf")
    margin = runner_up - best_delta
    print(f"\n[ref] best match: {best_model}  ({HUB_IDS[best_model]})")
    print(f"[ref] worst delta {best_delta:.4f}, next closest is {runner_up:.4f} "
          f"(margin {margin:.4f})")

    if best_delta > args.warn:
        print(f"[ref] INCONCLUSIVE: even the best match is outside tolerance "
              f"({args.warn}). This endpoint is serving something not in this table, or "
              "something in the chain is wrong.")
        verdict = "FAIL"
    elif margin < best_delta:
        print("[ref] WEAK: the margin to the runner-up is smaller than the best match's "
              "own error, so this does not reliably discriminate. Treat it as 'consistent "
              "with' rather than 'is'.")
        verdict = "WEAK"
    else:
        verdict = "PASS"
    print(f"[ref] {verdict}")
    print("[ref] REFERENCE_CHECK_FINISHED")
    return 0 if verdict != "FAIL" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://localhost:8100")
    ap.add_argument("--path", default="/invocations")
    ap.add_argument("--endpoint", default="",
                    help="SageMaker endpoint name. Goes through InvokeEndpoint instead of "
                         "HTTP, which is the only way to reach a hosted card (ml.g6e.*), "
                         "and always uses the KServe v2 envelope")
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--triton", action="store_true",
                    help="wrap in the KServe v2 envelope (the Triton deployable). Implied "
                         "by --endpoint")
    ap.add_argument("--warn", type=float, default=0.02,
                    help="probability tolerance. Deltas above it now FAIL the gate, not "
                         "just warn: with six published checkpoints giving identical "
                         "decisions, the decision check alone cannot tell them apart")
    ap.add_argument("--model", default="v21", choices=sorted(REFERENCES),
                    help="which checkpoint's published values to expect (default: v21, "
                         "which is what this tool has always compared against)")
    ap.add_argument("--identify", action="store_true",
                    help="score the responses against EVERY known checkpoint and rank "
                         "them, instead of gating against one. Answers 'which model is "
                         "this endpoint actually serving?'")
    args = ap.parse_args()

    url = args.base.rstrip("/") + args.path
    target = f"sagemaker://{args.endpoint}" if args.endpoint else url

    if args.identify:
        return identify(args, url, target)

    print(f"[ref] {target}")
    print(f"[ref] expecting {args.model}  ({HUB_IDS[args.model]})")
    references = REFERENCES[args.model]
    failures = 0
    for case in CASES:
        print(f"\n[ref] --- {case['name']}")
        request = {"state": STATE, "questions": case["questions"]}
        try:
            if args.endpoint:
                body = post_endpoint(args.endpoint, args.region, request)
            else:
                body = post(url, request, args.triton)
        # Broad on purpose. This is a gate: any failure to get an answer is a failure of
        # the gate, and the exception types differ by transport (urllib's URLError here,
        # botocore's ClientError/EndpointConnectionError there). Naming them all would mean
        # importing botocore on the HTTP path, and missing one would turn a red gate into a
        # traceback that reads like a bug in this script.
        except Exception as exc:
            detail = ""
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    detail = exc.read().decode()[:300]
                except Exception:
                    pass
            print(f"[ref]     REQUEST FAILED: {exc} {detail}")
            failures += 1
            continue
        if "error" in body:
            print(f"[ref]     SERVER ERROR: {body['error']}")
            failures += 1
            continue
        for name, expect in references[case["name"]].items():
            ok, probs_ok, notes = check(body["answers"][name], expect, args.warn)
            for n in notes:
                print(f"[ref]     {n}")
            if not ok:
                failures += 1
            if not probs_ok:
                failures += 1
            print(f"[ref]     {'DECISION MATCHES' if ok else 'DECISION DIFFERS'}")

    verdict = "PASS" if failures == 0 else "FAIL"
    print(f"\n[ref] {verdict}: {failures} mismatch(es) against the published "
          f"{args.model} values")
    if failures:
        print("[ref] If the decisions all matched and only the probabilities drifted, the "
              "likeliest cause is that this endpoint serves a different checkpoint than "
              "--model names. `--identify` will say which.")
    print("[ref] REFERENCE_CHECK_FINISHED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
