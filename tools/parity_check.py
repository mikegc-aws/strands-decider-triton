"""Compare two Decider deployables on the same requests, and gate on decision flips.

The gate is **zero decision flips as hard pass/fail, with `|Δp|` advisory only.**

Not a tight probability gate, deliberately. Kernel paths differ between these deployables
and the recorded `v19-vllm` validation measured max `|Δnoul|` 0.0065 and max `|Δscore|`
0.0050 across them; `serving/README.md` records `noul` 0.784 against 0.7875 (3.5e-3) as
normal drift, not a bug. A 1e-3 gate would fail spuriously and get loosened ad hoc
mid-run, which is worse than no gate. For a decision model the flip is what matters --
whether the answer you would act on changed -- and the probability delta is diagnostic.

A "flip" is defined per primitive:

    noul    the side of 0.5 changed            (what a yes/no gate acts on)
    choice  the argmax option changed          (what a router acts on)
    score   the rounded level changed          (what a threshold acts on)

Usage:
    # record a baseline from a running server
    parity_check.py --base http://localhost:8001 --save baseline.json

    # compare another server against it
    parity_check.py --base http://localhost:8002 --against baseline.json

    # or do both ends in one go
    parity_check.py --base http://localhost:8001 --other http://localhost:8002

Triton's own HTTP API is not spoken here. Both deployables expose `/v1/systemone`, and the
Triton one is reached through the SageMaker `/invocations` path, so `--path` selects it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# A fixed, varied suite. Fixed so two runs are comparable; varied so the comparison covers
# all three primitives, several option counts, and both readout paths (1 question takes the
# plain batched path, several take the shared-prefix path).
SUITE: list[dict] = [
    {
        "name": "noul_single_short",
        "state": "Help! My payouts have been failing for 3 days!",
        "questions": {"urgent": {"type": "noul",
                                 "instructions": "Does this convey urgency?"}},
    },
    {
        "name": "choice_three",
        "state": "Help! My payouts have been failing for 3 days!",
        "questions": {"team": {"type": "choice",
                               "instructions": "Which team should handle this?",
                               "criteria": {"billing": "payments, invoices, payouts",
                                            "technical": "bugs, outages, API errors",
                                            "sales": "pricing and upgrades"}}},
    },
    {
        "name": "score_three",
        "state": "Help! My payouts have been failing for 3 days!",
        "questions": {"frustration": {"type": "score",
                                      "instructions": "How frustrated is the writer?",
                                      "criteria": ["calm", "frustrated", "depressed"]}},
    },
    {
        # The shared-prefix path: three questions, one state, encoded once.
        "name": "all_three_primitives",
        "state": "Help! My payouts have been failing for 3 days!",
        "questions": {
            "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
            "team": {"type": "choice", "instructions": "Which team should handle this?",
                     "criteria": {"billing": "money", "technical": "bugs",
                                  "sales": "pricing"}},
            "frustration": {"type": "score", "instructions": "How frustrated?",
                            "criteria": ["calm", "frustrated", "depressed"]},
        },
    },
    {
        "name": "choice_eight_options",
        "state": "The dashboard shows a 502 on every page load since the deploy.",
        "questions": {"route": {"type": "choice", "instructions": "Route this ticket.",
                                "criteria": {k: "" for k in
                                             ["billing", "technical", "sales", "legal",
                                              "security", "docs", "onboarding", "other"]}}},
    },
    {
        "name": "score_five_levels",
        "state": "The dashboard shows a 502 on every page load since the deploy.",
        "questions": {"severity": {"type": "score", "instructions": "How severe is this?",
                                   "criteria": ["none", "low", "medium", "high",
                                                "critical"]}},
    },
    {
        "name": "empty_state",
        "state": "",
        "questions": {"maths": {"type": "noul",
                                "instructions": "Is 17 a prime number?"}},
    },
    {
        "name": "structured_state",
        "state": {"ticket": 4471, "tier": "enterprise",
                  "body": "Our nightly export has silently produced empty files all week."},
        "questions": {"escalate": {"type": "noul",
                                   "instructions": "Should this be escalated?"},
                      "area": {"type": "choice", "instructions": "Which area?",
                               "criteria": {"data": "pipelines and exports",
                                            "auth": "login and permissions",
                                            "ui": "front-end"}}},
    },
    {
        # ~1,200 tokens: past the launch-bound regime, into the shared-prefix win.
        "name": "long_state_multi_question",
        "state": ("A customer writes in about a recurring billing discrepancy. "
                  "Each month the invoice total exceeds the sum of the line items by "
                  "a small amount, and support has not been able to explain it. " * 18),
        "questions": {
            "urgent": {"type": "noul", "instructions": "Is this urgent?"},
            "team": {"type": "choice", "instructions": "Route it.",
                     "criteria": {"billing": "money", "technical": "bugs"}},
            "severity": {"type": "score", "instructions": "How severe?",
                         "criteria": ["minor", "moderate", "major"]},
        },
    },
]


def _v2_wrap(payload: dict) -> dict:
    """The KServe v2 envelope Triton requires. Mirrors decider_triton.wire, duplicated
    here deliberately so this script stays runnable standalone on a bare box with no
    PYTHONPATH -- it gets copied around on its own."""
    return {"inputs": [{"name": "REQUEST_JSON", "shape": [1, 1],
                        "datatype": "BYTES", "data": [json.dumps(payload)]}]}


def _v2_unwrap(response: dict) -> dict:
    if "error" in response and "outputs" not in response:
        raise ValueError(f"Triton error: {response['error']}")
    for out in response.get("outputs") or []:
        if out.get("name") == "RESPONSE_JSON":
            data = out.get("data") or []
            if not data:
                raise ValueError("RESPONSE_JSON carried no data")
            return json.loads(data[0])
    raise ValueError(f"no RESPONSE_JSON in {sorted(response)}")


def post(url: str, payload: dict, timeout: float = 300.0,
         triton: bool = False) -> tuple[dict, float]:
    """POST a System One request.

    `triton=True` wraps it in the KServe v2 envelope. Needed because SageMaker's
    `/invocations` on the Triton DLC is a thin proxy to `POST /v2/models/<n>/infer`, not a
    raw-JSON endpoint -- sending the bare body returns HTTP 500 "Unable to parse 'inputs'".
    The FastAPI deployable takes the bare body, hence the flag rather than one format.
    """
    body = json.dumps(_v2_wrap(payload) if triton else payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode())
    elapsed = (time.perf_counter() - started) * 1000
    return (_v2_unwrap(out) if triton else out), elapsed


def collect(base: str, path: str, repeats: int = 1,
            triton: bool = False) -> dict:
    """Run the suite against one server. Returns {case_name: response}."""
    url = base.rstrip("/") + path
    results: dict = {}
    for case in SUITE:
        payload = {"state": case["state"], "questions": case["questions"]}
        last_exc = None
        for attempt in range(repeats):
            try:
                out, ms = post(url, payload, triton=triton)
                results[case["name"]] = {"response": out, "latency_ms": round(ms, 2)}
                last_exc = None
                break
            except (urllib.error.URLError, urllib.error.HTTPError,
                    TimeoutError, ValueError) as exc:
                last_exc = exc
                detail = ""
                if isinstance(exc, urllib.error.HTTPError):
                    try:
                        detail = exc.read().decode()[:300]
                    except Exception:
                        pass
                print(f"  [{case['name']}] attempt {attempt + 1} failed: {exc} {detail}",
                      file=sys.stderr)
                time.sleep(2)
        if last_exc is not None:
            results[case["name"]] = {"error": str(last_exc)}
        else:
            print(f"  [{case['name']}] ok "
                  f"{results[case['name']]['latency_ms']}ms", flush=True)
    return results


def _decision(answer: dict) -> tuple[str, object]:
    """The thing a caller would act on, per primitive."""
    kind = answer.get("type")
    if kind == "noul":
        return "noul", answer["noul"] >= 0.5
    if kind == "choice":
        return "choice", answer["choice"]
    if kind == "score":
        return "score", round(float(answer["score"]))
    return kind or "?", json.dumps(answer, sort_keys=True)


def _probs(answer: dict) -> dict[str, float]:
    kind = answer.get("type")
    if kind == "noul":
        return {"noul": float(answer["noul"])}
    return {k: float(v) for k, v in (answer.get("probabilities") or {}).items()}


def compare(a: dict, b: dict, advisory: float) -> dict:
    """Decision flips (hard) and probability deltas (advisory)."""
    flips: list[dict] = []
    deltas: list[dict] = []
    missing: list[str] = []
    errors: list[str] = []

    for name in sorted(set(a) | set(b)):
        ra, rb = a.get(name), b.get(name)
        if ra is None or rb is None:
            missing.append(name)
            continue
        if "error" in ra or "error" in rb:
            errors.append(f"{name}: A={ra.get('error', 'ok')} B={rb.get('error', 'ok')}")
            continue
        aa = ra["response"].get("answers", {})
        ab = rb["response"].get("answers", {})
        if set(aa) != set(ab):
            missing.append(f"{name}: answer keys differ {sorted(aa)} vs {sorted(ab)}")
            continue
        for q in sorted(aa):
            da, db = _decision(aa[q]), _decision(ab[q])
            if da != db:
                flips.append({"case": name, "question": q,
                              "a": str(da[1]), "b": str(db[1])})
            pa, pb = _probs(aa[q]), _probs(ab[q])
            for k in sorted(set(pa) | set(pb)):
                d = abs(pa.get(k, 0.0) - pb.get(k, 0.0))
                deltas.append({"case": name, "question": q, "label": k, "delta": d})

    worst = sorted(deltas, key=lambda x: -x["delta"])[:8]
    max_delta = worst[0]["delta"] if worst else 0.0
    return {
        "compared_cases": len([n for n in set(a) & set(b)]),
        "flips": flips,
        "max_abs_delta": max_delta,
        "advisory_threshold": advisory,
        "advisory_exceeded": max_delta > advisory,
        "worst_deltas": worst,
        "missing": missing,
        "errors": errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="server to run the suite against")
    ap.add_argument("--path", default="/v1/systemone",
                    help="use /invocations for the SageMaker/Triton deployable")
    ap.add_argument("--other", help="second server, compared against --base")
    ap.add_argument("--other-path", default=None)
    ap.add_argument("--save", help="write --base's results here")
    ap.add_argument("--against", help="compare --base against a saved baseline")
    ap.add_argument("--advisory", type=float, default=7e-3,
                    help="probability delta that is reported but does not fail")
    ap.add_argument("--triton", action="store_true",
                    help="wrap requests in the KServe v2 envelope; required for the "
                         "Triton deployable's /invocations")
    ap.add_argument("--repeats", type=int, default=3,
                    help="retries per case, for a server that is still warming")
    args = ap.parse_args()

    print(f"[parity] collecting from {args.base}{args.path}", flush=True)
    a = collect(args.base, args.path, args.repeats, triton=args.triton)

    if args.save:
        with open(args.save, "w") as fh:
            json.dump(a, fh, indent=2)
        print(f"[parity] wrote {args.save}")
        if not (args.other or args.against):
            failed = [k for k, v in a.items() if "error" in v]
            print(f"[parity] {len(a) - len(failed)}/{len(a)} cases answered")
            return 1 if failed else 0

    if args.against:
        with open(args.against) as fh:
            b = json.load(fh)
        label = args.against
    elif args.other:
        path = args.other_path or args.path
        print(f"[parity] collecting from {args.other}{path}", flush=True)
        b = collect(args.other, path, args.repeats, triton=args.triton)
        label = args.other
    else:
        return 0

    report = compare(a, b, args.advisory)
    print(f"\n[parity] ==== {args.base} vs {label} ====")
    print(f"[parity] cases compared     : {report['compared_cases']}")
    print(f"[parity] DECISION FLIPS     : {len(report['flips'])}  <- the hard gate")
    for f in report["flips"]:
        print(f"[parity]   FLIP {f['case']}.{f['question']}: {f['a']} -> {f['b']}")
    print(f"[parity] max |delta p|      : {report['max_abs_delta']:.2e} "
          f"(advisory {args.advisory:.0e}"
          f"{', EXCEEDED' if report['advisory_exceeded'] else ''})")
    for d in report["worst_deltas"][:5]:
        print(f"[parity]   {d['delta']:.2e}  {d['case']}.{d['question']}[{d['label']}]")
    for m in report["missing"]:
        print(f"[parity] MISSING: {m}")
    for e in report["errors"]:
        print(f"[parity] ERROR: {e}")

    ok = not report["flips"] and not report["missing"] and not report["errors"]
    print(f"[parity] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
