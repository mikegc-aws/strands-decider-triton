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

# States sized to sit INSIDE the CUDA-graph envelope (`cuda_graphs.GRAPH_STATE`), so the
# graphed batch below actually reaches a graph. Still mixed lengths -- that is the property
# that detects a padding mistake, and it is as necessary of the bank as of the eager path.
_LOG = (" payout PO-4471 FAILED INSUFFICIENT_SETTLEMENT_BALANCE; case SC-99812 opened. ")
MID_200 = MEDIUM + _LOG * 3
MID_400 = MEDIUM + _LOG * 9
MID_700 = MEDIUM + _LOG * 18


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


def one_pass_cases() -> list[tuple[str, str, list[str]]]:
    """A batch that takes the ONE-pass route, which is the route graphs are on by default.

    `graph_cases()` below duplicates each state across seven questions, which pushes
    `evaluate_many` past `DUP_TOKEN_BUDGET` and onto the two-pass route for the whole
    chunk -- so it never reaches the `combined` graph at all. (That is how this was found:
    `--cuda-graphs` reported `captured: 0, pending: 0` and the gate caught it. It took two
    attempts: a first version of this list still duplicated 730 state tokens across its
    rows, which is also past the budget. The budget is counted over the WHOLE batch, not
    per request.)

    So: one question per request and a distinct state each, which duplicates nothing, plus
    one three-question request over the shortest state -- 20 duplicated tokens, well under
    the budget, and it is what makes two rows of one state appear in a one-pass batch.
    Mixed state lengths throughout, for the same reason as everywhere else here: equal
    lengths pad to nothing and would pass while the padding was broken.
    """
    names = list(QUESTIONS)
    return [
        ("o-short/1q", SHORT, names[:1]),
        ("o-medium/1q", MEDIUM, names[1:2]),
        ("o-mid200/1q", MID_200, names[2:3]),
        ("o-mid400/1q", MID_400, names[3:4]),
        ("o-mid700/1q", MID_700, names[4:5]),
        ("o-mid400/1q-b", MID_400, names[5:6]),
        ("o-short/3q", SHORT, names[:3]),           # the only duplication, ~20 tokens
    ]
    # Deliberately NO out-of-envelope state in this batch. `admits_combined` is
    # all-or-nothing over a pass, so one 1,000-token row sends every row in the batch to
    # the eager path -- which is exactly what happened the first time, and the gate's
    # "nothing was captured" check caught it. The fallback is gated by `cases()` above,
    # which carries a state long enough to force it.


def graph_cases() -> list[tuple[str, str, list[str]]]:
    """A batch whose shapes the graphed route admits, covering BOTH of its passes.

    `cases()` above is deliberately ragged enough that one of its states exceeds the
    graphed envelope, which sends the whole chunk to the eager path -- useful (it is the
    fallback, and the fallback must stay right) but it means that batch never tests a
    replay. These cases are sized so it does:

      * 1 and 2 questions over a short state  -> little duplication, so the ONE-pass route,
        which is the `combined` graph: one pass over `state + question` per row.
      * 7 questions over 200-700 token states -> duplication past `DUP_TOKEN_BUDGET`, so
        the TWO-pass route: the `states` graph into the bank, then the `rows` graph out of
        it.
      * a duplicated state, so two bank entries' worth of rows map back through one entry.
      * four distinct state lengths in one pass, so the bank's right-alignment and the row
        mask's `Sr - plen` offset are both exercised with real padding.
    """
    names = list(QUESTIONS)
    return [
        ("g-short/1q", SHORT, names[:1]),
        ("g-short/2q", SHORT, names[3:5]),
        ("g-medium/7q", MEDIUM, names),
        ("g-mid200/7q", MID_200, names),
        ("g-mid400/7q", MID_400, names),
        ("g-mid700/5q", MID_700, names[:5]),
        ("g-mid200/7q-dup", MID_200, names),
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
    ap.add_argument("--cuda-graphs", action="store_true",
                    help="put evaluate_many on the captured-graph path (SD_CUDA_GRAPHS=1 "
                         "in the container). The reference side stays eager either way, so "
                         "this is the gate on the graph path specifically.")
    ap.add_argument("--cuda-graphs-two-pass", action="store_true",
                    help="also graph the state/row pair (SD_CUDA_GRAPHS=all). Off by "
                         "default in the engine because it measured slower; this gates it "
                         "for correctness anyway, since it is kept in the tree.")
    args = ap.parse_args()

    from strands_decider.merged_engine import load_merged_engine
    from strands_decider.schema import SystemOneRequest

    engine = load_merged_engine(args.checkpoint, args.merged, device=args.device,
                                use_prefix_cache=True, max_rows=args.max_rows,
                                cuda_graphs=args.cuda_graphs,
                                cuda_graphs_two_pass=args.cuda_graphs_two_pass)
    if not hasattr(engine, "evaluate_many"):
        print("[parity] FAIL: engine has no evaluate_many")
        return 2

    # Two batches: the ragged one, which overflows the CUDA-graph envelope and so also
    # gates the fallback, and one sized to stay inside it. Both run whatever path the
    # engine picks; the reference side is `evaluate`, which never touches a graph.
    batches = [("ragged", cases())]
    if args.cuda_graphs:
        batches.append(("one-pass", one_pass_cases()))
    if args.cuda_graphs_two_pass:
        batches.append(("two-pass", graph_cases()))

    flips = 0
    worst = 0.0
    worst_where = ""
    per_batch: list[tuple[str, float, str]] = []

    for batch_label, specs in batches:
        batch_worst, batch_where = 0.0, ""
        requests = [
            SystemOneRequest(state=state,
                             questions={n: QUESTIONS[n] for n in qnames})
            for _label, state, qnames in specs
        ]

        # Reference: one request at a time, through the untouched single-request path.
        print(f"\n[parity] === {batch_label}: {len(requests)} requests")
        print(f"[parity] reference: {len(requests)} requests, one at a time")
        ref = [engine.evaluate(r) for r in requests]

        if args.cuda_graphs or args.cuda_graphs_two_pass:
            # Run the batch a few times and capture in between. A captured graph and the
            # eager run of the same bucket are DIFFERENT code paths, so a parity check
            # that only ever saw the bucket's eager warm-up run would pass while every
            # replay was wrong -- which is precisely what the earlier probe found.
            print("[parity] cuda graphs: warming the buckets, then capturing")
            for _ in range(3):
                engine.evaluate_many(requests)
            engine.capture_pending()
            print(f"[parity] cuda graphs: {engine.graph_stats()}")

        # Under test: all of them together, in the engine's batched route.
        print("[parity] batched:   the same requests in one evaluate_many call")
        got = engine.evaluate_many(requests)

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
                    if d > batch_worst:
                        batch_worst, batch_where = d, f"{label}/{name}/{k}"
                    if d > worst:
                        worst, worst_where = d, f"{label}/{name}/{k}"
            print(f"[parity]   {label}: {len(qnames)} question(s) ok")
        per_batch.append((batch_label, batch_worst, batch_where))

    for batch_label, batch_worst, batch_where in per_batch:
        print(f"[parity] {batch_label}: max |delta p| {batch_worst:.6f} at "
              f"{batch_where or 'n/a'}")

    if args.cuda_graphs or args.cuda_graphs_two_pass:
        stats = engine.graph_stats()
        print(f"\n[parity] cuda graphs: {stats}")
        if stats.get("failed"):
            # Not a correctness failure -- a failed capture falls back to eager, which is
            # right -- but it is a silent loss of the whole point of this path, so it is
            # said out loud rather than left in a stats dict nobody reads.
            print(f"[parity] NOTE: {stats['failed']} shape(s) could not be captured and "
                  "are running eagerly. The speed-up does not apply to them.")
        # Without this the gate is vacuous: every shape could have quietly fallen back to
        # the eager path and the run would still report PASS.
        if not stats.get("captured") or not stats.get("replays"):
            print("[parity] FAIL: --cuda-graphs was asked for but nothing was captured or "
                  "replayed, so this run did not test the graph path at all")
            return 2

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
