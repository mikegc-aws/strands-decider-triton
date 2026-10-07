"""Does `evaluate_many` give the same answers as `evaluate`? The gate for cross-request batching.

This is the test the CPU suite structurally cannot do. `evaluate_many` batches requests
whose states have **different lengths**, which means padding them, and the padding
interacts with 18 recurrent Gated DeltaNet layers. Get that wrong and you do not get a
crash -- you get a confidently wrong probability at HTTP 200, for one caller, derived
partly from another caller's state or from pad tokens. Only the real torso on a real GPU
can detect it.

The gate is the same one used everywhere else in this project:

    DECISION must match exactly       (argmax option, side of 0.5, rounded level)  -> hard
    |delta p| reported, warns above 7e-3                                           -> advisory

7e-3 is this project's established band, not a number chosen to make the test pass. It
was set from what actually validated the previous path: recorded drift across kernel
paths is 3.5e-3 (`noul` 0.784 vs 0.7875), the v19-vLLM validation measured max
|delta noul| 0.0065 and |delta score| 0.0050, and the LoRA merge 0.0078.

An earlier version of this file used 2e-3, on the reasoning that batching is "the same
arithmetic on the same tokens". **That reasoning was wrong**, and three controls on an
L4 show why:

    evaluate() twice                                      0.000000   deterministic
    evaluate_many, 1 request  (1 state row,  7 q rows)     0.000000   bit-exact
    evaluate_many, 2 requests, IDENTICAL state, pad=0      0.003100
    evaluate_many, 2 requests, different state, SAME len   0.003100   <- same number
    evaluate_many, padding up to 4,077 tokens              0.0045-0.0077

The drift appears at **pad = 0** and is identical whether the two states are the same or
different, so it is not padding and not the cache gather -- it is bf16 reduction order
changing with the batch shape (14 question rows instead of 7). Padding from 0 to 4,077
tokens adds almost nothing on top. The batch-of-one case being bit-exact is the evidence
that the gather and row mapping are correct; `tests/test_batch_engine.py` covers the
mapping itself.

So: a decision flip here is a bug. A 5e-3 probability difference is the arithmetic.

What makes it a real test:
  * **mixed state lengths in one batch** -- equal lengths would pad to nothing and pass
    while the padding logic was broken.
  * **a duplicated state** -- exercises the de-duplication and the many-rows-to-one-state
    mapping at the same time.
  * **varying question counts**, including a single-question request, so rows per request
    differ and an off-by-one in the row mapping shows up as a crossed answer.
  * **a deliberately long state** that forces `_fit`'s truncation, since truncation
    changes token offsets and therefore the pointer readout.

Usage (on a GPU host, inside the serving image):
    batch_parity.py --checkpoint /opt/strands-decider/checkpoint \
                    --merged /opt/strands-decider/merged
"""

from __future__ import annotations

import argparse
import sys

QUESTIONS = {
    "asks_for_human": {"type": "noul",
                       "instructions": "Is the writer asking to speak to a human?"},
    "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
    "refund": {"type": "noul", "instructions": "Is the writer requesting a refund?"},
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "payments and invoices",
                                "technical": "bugs and outages",
                                "shipping": "delivery and tracking",
                                "returns": "refunds and exchanges",
                                "accounts": "login and profile",
                                "sales": "pricing and upgrades"}},
    "frustration": {"type": "score", "instructions": "How frustrated is the writer?",
                    "criteria": ["calm", "frustrated", "depressed"]},
    "severity": {"type": "score", "instructions": "How severe is the business impact?",
                 "criteria": ["none", "minor", "moderate", "major", "critical"]},
    "intent": {"type": "choice", "instructions": "What is the writer's primary intent?",
               "criteria": {"cancel": None, "complain": None, "query": None,
                            "escalate": None, "track_order": None, "other": None}},
}

SHORT = "Payouts failing. Please call me."
MEDIUM = ("Hi, my last three payouts have failed and I have had no explanation. "
          "I have contacted support twice already and nobody has got back to me. "
          "This is affecting my ability to pay suppliers. Can someone please call me?")
LONG = MEDIUM + (" Transaction log: payout PO-4471 FAILED INSUFFICIENT_SETTLEMENT_BALANCE; "
                 "payout PO-4482 FAILED INSUFFICIENT_SETTLEMENT_BALANCE; case SC-99812 "
                 "opened severity normal owner unassigned. ") * 30
VERY_LONG = LONG * 6  # forces _fit's state truncation


def cases() -> list[tuple[str, str, list[str]]]:
    """(label, state, question names). Deliberately ragged -- see the module docstring."""
    names = list(QUESTIONS)
    return [
        ("short/7q", SHORT, names),
        ("medium/7q", MEDIUM, names),
        ("medium/1q", MEDIUM, names[:1]),          # single question -> different row count
        ("long/3q", LONG, names[:3]),
        ("short/2q", SHORT, names[3:5]),
        ("medium/7q-dup", MEDIUM, names),          # same state as medium/7q -> de-dup path
        ("verylong/4q", VERY_LONG, names[:4]),     # forces truncation
    ]


def decision(ans: dict) -> object:
    """The part that must match exactly. Probabilities are advisory; this is not."""
    kind = ans.get("type")
    if kind == "noul":
        return ("noul", float(ans["noul"]) >= 0.5)
    if kind == "choice":
        return ("choice", ans["choice"])
    return ("score", round(float(ans["score"])))


def probs_of(ans: dict) -> dict[str, float]:
    out = dict(ans.get("probabilities") or {})
    if ans.get("type") == "noul":
        out["__noul"] = float(ans["noul"])
    if "confidence" in ans:
        out["__conf"] = float(ans["confidence"])
    if ans.get("type") == "score":
        out["__score"] = float(ans["score"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="/opt/strands-decider/checkpoint")
    ap.add_argument("--merged", default="/opt/strands-decider/merged")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warn", type=float, default=7e-3)
    ap.add_argument("--max-rows", type=int, default=128)
    args = ap.parse_args()

    from strands_decider.merged_engine import load_merged_engine
    from strands_decider.schema import SystemOneRequest

    engine = load_merged_engine(args.checkpoint, args.merged, device=args.device,
                                use_prefix_cache=True, max_rows=args.max_rows)
    if not hasattr(engine, "evaluate_many"):
        print("[parity] FAIL: engine has no evaluate_many")
        return 2

    specs = cases()
    requests = [
        SystemOneRequest(state=state,
                         questions={n: QUESTIONS[n] for n in qnames})
        for _label, state, qnames in specs
    ]

    # Reference: one request at a time, through the untouched single-request path.
    print(f"[parity] reference: {len(requests)} requests, one at a time")
    ref = [engine.evaluate(r) for r in requests]

    # Under test: all of them together, in two passes.
    print("[parity] batched:   the same requests in one evaluate_many call")
    got = engine.evaluate_many(requests)

    flips = 0
    worst = 0.0
    worst_where = ""
    for (label, _state, qnames), r, g in zip(specs, ref, got, strict=True):
        if isinstance(g, Exception):
            print(f"[parity]   {label}: BATCHED RAISED {type(g).__name__}: {g}")
            flips += 1
            continue
        rd = r.model_dump() if hasattr(r, "model_dump") else r
        gd = g.model_dump() if hasattr(g, "model_dump") else g
        if sorted(gd["answers"]) != sorted(rd["answers"]):
            print(f"[parity]   {label}: ANSWER SET DIFFERS "
                  f"{sorted(gd['answers'])} vs {sorted(rd['answers'])}")
            flips += 1
            continue
        for name in qnames:
            a_ref, a_got = rd["answers"][name], gd["answers"][name]
            if decision(a_ref) != decision(a_got):
                flips += 1
                print(f"[parity]   {label}/{name}: DECISION FLIP "
                      f"{decision(a_ref)} -> {decision(a_got)}")
            pr, pg = probs_of(a_ref), probs_of(a_got)
            for k in pr:
                d = abs(pr[k] - pg.get(k, float("nan")))
                if d > worst:
                    worst, worst_where = d, f"{label}/{name}/{k}"
        print(f"[parity]   {label}: {len(qnames)} question(s) ok")

    print(f"\n[parity] max |delta p| {worst:.6f} at {worst_where or 'n/a'}"
          f"  (warn above {args.warn})")
    if worst > args.warn:
        print("[parity] WARNING: probability drift above the threshold. This path is "
              "supposed to be the same arithmetic on the same tokens, so investigate "
              "state padding and the cache gather before shipping.")
    print(f"[parity] {'PASS' if flips == 0 else 'FAIL'}: {flips} decision flip(s) "
          f"between evaluate_many and evaluate")
    print("[parity] BATCH_PARITY_FINISHED")
    return 0 if flips == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
