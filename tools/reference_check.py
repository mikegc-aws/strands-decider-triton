"""Check the served model against the published reference values for v21.

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
    |delta p| is reported, and warns above 0.02 rather than failing

0.02 is chosen to be looser than the 0.0078 that `ab.py` measured for the LoRA merge plus
the ~0.0065 recorded across kernel paths, with headroom -- it is a smoke gate, not a
calibration test.

Usage:
    reference_check.py --base http://localhost:8100 --path /invocations --triton
    reference_check.py --endpoint sd-l4-vcpu8          # a deployed SageMaker endpoint
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

STATE = "Help! My payouts have been failing for 3 days! "

# Each case: the request, and the published expectation.
CASES: list[dict] = [
    {
        "name": "noul_urgency",
        "questions": {"urgency": {"type": "noul",
                                  "instructions": "Does this convey urgency?"}},
        "expect": {"urgency": {"kind": "noul", "noul": 0.875}},
    },
    {
        "name": "choice_team",
        "questions": {"team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            # Bare labels, no descriptions -- matching the CLI form `=billing,sales,retail`.
            "criteria": {"billing": None, "sales": None, "retail": None},
        }},
        "expect": {"team": {"kind": "choice", "choice": "billing", "confidence": 0.835,
                            "probabilities": {"billing": 0.890, "sales": 0.056,
                                              "retail": 0.054}}},
    },
    {
        "name": "score_frustration",
        "questions": {"frustration": {
            "type": "score",
            "instructions": "How frustrated is the writer?",
            "criteria": ["calm", "frustrated", "depressed"],
        }},
        "expect": {"frustration": {"kind": "score", "score": 1.07, "confidence": 0.602,
                                   "probabilities": {"0": 0.140, "1": 0.648,
                                                     "2": 0.212}}},
    },
]


def _wrap(payload: dict) -> dict:
    """The KServe v2 envelope. `shape: [1, 1]` because the model sets max_batch_size > 0 and
    Triton prepends the batch dimension; `[1]` is the most common way to get an opaque
    "Unable to parse 'inputs'" out of this deployable."""
    return {"inputs": [{"name": "REQUEST_JSON", "shape": [1, 1],
                        "datatype": "BYTES", "data": [json.dumps(payload)]}]}


def _unwrap(out: dict) -> dict:
    if "error" in out and "outputs" not in out:
        raise ValueError(f"Triton error: {out['error']}")
    for o in out.get("outputs") or []:
        if o.get("name") == "RESPONSE_JSON":
            return json.loads(o["data"][0])
    raise ValueError(f"no RESPONSE_JSON in {sorted(out)}")


def post(url: str, payload: dict, triton: bool, timeout: float = 300.0) -> dict:
    if triton:
        payload = _wrap(payload)
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode())
    if not triton:
        return out
    return _unwrap(out)


def invoke(rt, endpoint: str, payload: dict) -> dict:
    """The same three cases against a deployed SageMaker endpoint.

    Added because this gate could previously only be run against a container over plain
    HTTP (`--base`), so the one check the project calls its strongest -- zero decision
    mismatches against the published v21 values -- could not be run on a real endpoint
    without standing up an unsigned proxy beside it. `/invocations` on the Triton DLC is a
    proxy to KServe v2 `infer`, so the envelope is the same one `--base --triton` sends;
    only the transport and the SigV4 signing differ.
    """
    raw = rt.invoke_endpoint(EndpointName=endpoint, ContentType="application/json",
                             Body=json.dumps(_wrap(payload)))["Body"].read().decode()
    return _unwrap(json.loads(raw))


def check(answer: dict, expect: dict, warn: float) -> tuple[bool, list[str]]:
    """Decision match is the gate; probability deltas are reported."""
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
        flag = "  <-- above warn threshold" if delta > warn else ""
        notes.append(f"  p[{label}] {got_p:.4f} vs {want_p} (delta {delta:.4f}){flag}")

    return decision_ok, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="",
                    help="HTTP base of a container, e.g. http://localhost:8100 "
                         "(default when --endpoint is not given)")
    ap.add_argument("--path", default="/invocations")
    ap.add_argument("--triton", action="store_true",
                    help="wrap in the KServe v2 envelope (the Triton deployable)")
    ap.add_argument("--endpoint", default="",
                    help="SageMaker endpoint name; mutually exclusive with --base. Implies "
                         "the KServe v2 envelope, which is the only thing /invocations "
                         "accepts on the Triton DLC")
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--warn", type=float, default=0.02)
    args = ap.parse_args()

    if args.endpoint and args.base:
        ap.error("pass exactly one of --endpoint (SageMaker) or --base (direct HTTP)")

    rt = None
    # urllib's errors plus ValueError cover the --base path. The SageMaker path adds
    # botocore's ClientError, which is NOT an OSError -- leaving it out means a ModelError
    # from a sick endpoint escapes as a traceback instead of being counted as the failure
    # it is, and the gate prints no verdict at all.
    transport_errors: tuple[type[BaseException], ...] = (
        urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError)
    if args.endpoint:
        import boto3  # only the SageMaker path needs it; --base stays stdlib-only
        from botocore.config import Config
        from botocore.exceptions import BotoCoreError, ClientError
        transport_errors += (ClientError, BotoCoreError)
        # Retries off, so a refusal is an error rather than latency in disguise -- the same
        # reason tools/bench_tickets.py disables them.
        rt = boto3.client("sagemaker-runtime", region_name=args.region,
                          config=Config(retries={"max_attempts": 0}, read_timeout=300))
        target = f"endpoint {args.endpoint} ({args.region})"
    else:
        url = (args.base or "http://localhost:8100").rstrip("/") + args.path
        target = url
    print(f"[ref] {target}  (published v21 reference values)")
    failures = 0
    for case in CASES:
        print(f"\n[ref] --- {case['name']}")
        try:
            request = {"state": STATE, "questions": case["questions"]}
            body = (invoke(rt, args.endpoint, request) if rt is not None
                    else post(url, request, args.triton))
        except transport_errors as exc:
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
        for name, expect in case["expect"].items():
            ok, notes = check(body["answers"][name], expect, args.warn)
            for n in notes:
                print(f"[ref]     {n}")
            if not ok:
                failures += 1
            print(f"[ref]     {'DECISION MATCHES' if ok else 'DECISION DIFFERS'}")

    print(f"\n[ref] {'PASS' if failures == 0 else 'FAIL'}: "
          f"{failures} decision mismatch(es) against the published v21 values")
    print("[ref] REFERENCE_CHECK_FINISHED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
