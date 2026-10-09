"""Fused kernels vs the reference torso: same requests, same engine, both routes.

The gate for `SD_FUSE_LAYERS=1`, and the thing neither `batch_parity.py` nor
`reference_check.py` can establish on its own. Those two compare the deployable against
*itself* and against the published values; this compares the fused arithmetic against the
reference arithmetic directly, request by request, on identical inputs.

Why it has to be a direct A/B. Fusing changes the rounding -- fla's kernels keep fp32 where
`Qwen3_5RMSNormGated` rounds to bf16 mid-formula, and the gate, the beta sigmoid and the
q/k L2 norm move inside the chunk kernel. Every one of those shifts a probability a little,
and none of them can shift a probability a lot without something being wrong. So:

    DECISION must match exactly       (argmax option, side of 0.5, rounded level)  -> hard
    |delta p| reported per primitive, warns above 7e-3                             -> advisory

7e-3 is this project's established band and is not negotiable upward to make this pass. It
came from what validated the earlier paths: recorded drift across kernel paths 3.5e-3, the
v19-vLLM validation max |delta noul| 0.0065 / |delta score| 0.0050, the LoRA merge 0.0078,
cross-request batching 0.0031-0.0077. A fused pass that lands inside it is the same class of
change as those. A fused pass that lands outside it is a finding -- report it, do not widen
the band.

Both engines are loaded in the SAME process, one after the other, with the first freed
before the second loads. Two 2B torsos fit on a 24 GB L4, but a box is often shared, and
loading sequentially also means the numbers come from one tokeniser, one prompt renderer and
one head -- so a difference is attributable to the kernels and nothing else.

It also times the torso forward itself (`--time`), because that is where a fused pass is
supposed to win and an end-to-end throughput number mixes it with queueing and the head.

Usage (on the GPU host, inside the serving image):
    fused_ab.py --checkpoint /opt/strands-decider/checkpoint \
                --merged /opt/strands-decider/merged --time
"""

from __future__ import annotations

import argparse
import json
import sys
import time

# Mirrors tools/batch_parity.py's set deliberately: three noul, two choice, two score, with
# a 6-option and an 11-option choice so the pointer readout is exercised at more than two
# widths. Keeping the two tools on the same questions means a drift number here is directly
# comparable with the batching drift number there.
QUESTIONS: dict[str, dict] = {
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
    "intent": {"type": "choice", "instructions": "What is the writer's primary intent?",
               "criteria": {"cancel": None, "complain": None, "query": None,
                            "escalate": None, "track_order": None, "other": None}},
    "frustration": {"type": "score", "instructions": "How frustrated is the writer?",
                    "criteria": ["calm", "frustrated", "depressed"]},
    "severity": {"type": "score", "instructions": "How severe is the business impact?",
                 "criteria": ["none", "minor", "moderate", "major", "critical"]},
}

SHORT = "Payouts failing. Please call me."
MEDIUM = ("Hi, my last three payouts have failed and I have had no explanation. "
          "I have contacted support twice already and nobody has got back to me. "
          "This is affecting my ability to pay suppliers. Can someone please call me?")
LONG = MEDIUM + (" Transaction log: payout PO-4471 FAILED INSUFFICIENT_SETTLEMENT_BALANCE; "
                 "payout PO-4482 FAILED INSUFFICIENT_SETTLEMENT_BALANCE; case SC-99812 "
                 "opened severity normal owner unassigned. ") * 30
VERY_LONG = LONG * 6   # forces `_fit`'s state truncation, which moves every token offset

# The published v21 reference state, so this A/B covers the exact rows reference_check.py
# gates on as well as the ragged batch ones.
PUBLISHED = "Help! My payouts have been failing for 3 days! "


def cases() -> list[tuple[str, str, list[str]]]:
    """(label, state, question names).

    Deliberately ragged, and for the same reasons batch_parity.py is: a single-question
    request takes the one-pass route while a seven-question one takes the two-pass route,
    and the two routes drive the DeltaNet cache completely differently -- one with no cache
    at all, one filling a cache and then continuing from it. A fused conv-state convention
    that was subtly wrong would show up only on the second. A truncated state is included
    because truncation shifts the pointer readout's offsets.
    """
    names = list(QUESTIONS)
    return [
        ("published/1q-noul", PUBLISHED, ["urgent"]),
        ("published/7q", PUBLISHED, names),
        ("short/1q", SHORT, names[:1]),
        ("short/7q", SHORT, names),
        ("medium/3q", MEDIUM, names[:3]),
        ("medium/7q", MEDIUM, names),
        ("long/7q", LONG, names),
        ("verylong/7q", VERY_LONG, names),
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
    """Every number the caller actually sees, flattened. The derived figures (confidence,
    expected score) are included because they are what a caller thresholds on, and a
    confidence is a sharper detector of drift than the probabilities it comes from."""
    out = {k: float(v) for k, v in (ans.get("probabilities") or {}).items()}
    if ans.get("type") == "noul":
        out["__noul"] = float(ans["noul"])
    if ans.get("type") == "score":
        out["__score"] = float(ans["score"])
    if "confidence" in ans:
        out["__conf"] = float(ans["confidence"])
    return out


def run(checkpoint: str, merged: str, device: str, max_rows: int, *,
        fused: bool, specs: list[tuple[str, str, list[str]]],
        torch_dtype: str | None = None) -> tuple[dict, dict]:
    """Answer every case through one engine, both the single-request and batched routes.

    Returns `({label/name/route -> answer dict}, {timing})`. Both routes are recorded because
    the Triton backend uses `evaluate_many` in production and `evaluate` as its fallback, so
    both have to agree with the reference -- a fused path that is right only on the route the
    benchmark happens to take is not right.

    `torch_dtype` is for `--fp32-reference` only; it bypasses `load_merged_engine` because
    that function is for serving and should not grow a precision knob.
    """
    from strands_decider.merged_engine import load_merged_engine
    from strands_decider.schema import SystemOneRequest

    t0 = time.perf_counter()
    if torch_dtype:
        from strands_decider.batch_engine import BatchedSystemOneEngine
        from strands_decider.infer import EngineConfig
        from strands_decider.merged_engine import build_merged_model

        model = build_merged_model(checkpoint, merged, torch_dtype=torch_dtype)
        engine = BatchedSystemOneEngine(
            model, EngineConfig(device=device, use_prefix_cache=True,
                                model_name="strands-decider-ab"),
            max_rows=max_rows)
    else:
        engine = load_merged_engine(checkpoint, merged, device=device,
                                    use_prefix_cache=True, max_rows=max_rows,
                                    fuse_layers=fused)
    load_s = time.perf_counter() - t0

    from strands_decider.fused_layers import is_fused
    actually_fused = is_fused(engine.model.torso)
    if actually_fused != fused:
        # The one thing this harness must never do is attribute a number to the wrong torso.
        raise SystemExit(f"[ab] asked for fused={fused} but the torso reports "
                         f"fused={actually_fused}; refusing to report a mislabelled A/B")

    requests = [SystemOneRequest(state=s, questions={n: QUESTIONS[n] for n in qn})
                for _label, s, qn in specs]

    out: dict[str, dict] = {}
    for (label, _s, qnames), req in zip(specs, requests, strict=True):
        resp = engine.evaluate(req)
        dumped = resp.model_dump() if hasattr(resp, "model_dump") else resp
        for n in qnames:
            out[f"{label}/{n}/evaluate"] = dumped["answers"][n]

    for (label, _s, qnames), resp in zip(specs, engine.evaluate_many(requests), strict=True):
        if isinstance(resp, Exception):
            raise SystemExit(f"[ab] evaluate_many raised for {label}: "
                             f"{type(resp).__name__}: {resp}")
        dumped = resp.model_dump() if hasattr(resp, "model_dump") else resp
        for n in qnames:
            out[f"{label}/{n}/evaluate_many"] = dumped["answers"][n]

    timing = {"load_s": round(load_s, 1)}
    return out, timing


def time_torso(checkpoint: str, merged: str, device: str, *, fused: bool,
               token_counts: list[int], repeats: int) -> dict[str, float]:
    """Time the torso forward alone, batch 1, at a few prompt lengths.

    Isolated from the engine on purpose. End-to-end `decisions_per_s` mixes the forward with
    queueing, the pointer head, prompt rendering and the HTTP envelope; this is the number
    the fused kernels are supposed to move, and the only one a kernel change can be held
    responsible for. Batch 1 because the per-pass floor is CPU dispatch, so batch 1 is where
    launch count dominates most -- exactly the regime this repository's ceiling sits in.
    """
    import torch

    from strands_decider.fused_layers import fuse_torso, is_fused
    from strands_decider.merged_engine import build_merged_model

    model = build_merged_model(checkpoint, merged)
    torso = model.torso.to(device).eval()
    if fused:
        fuse_torso(torso)
    if is_fused(torso) != fused:
        raise SystemExit("[ab] torso fusion state does not match the label")

    out: dict[str, float] = {}
    with torch.inference_mode():
        for n in token_counts:
            ids = torch.randint(1000, 5000, (1, n), device=device)
            mask = torch.ones_like(ids)
            # The fla/Triton kernels compile per shape (54.2 s cold, 20.7 s warm against
            # ~67 ms steady state), so an unwarmed shape would be measured as a compile.
            # Three, not one: autotuning can recompile on the second launch of a new shape.
            warmups, t0 = 3, 0.0
            for i in range(warmups + repeats):
                if i == warmups:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                torso(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
            torch.cuda.synchronize()
            out[str(n)] = round((time.perf_counter() - t0) * 1000.0 / repeats, 2)

    torch.cuda.empty_cache()
    return out


BATCH_REQUESTS = 8   # config.pbtxt's max_batch_size: what a full Triton batch looks like


def time_batch(checkpoint: str, merged: str, device: str, max_rows: int, *,
               fused: bool, repeats: int) -> dict[str, float]:
    """Time `evaluate_many` on a full Triton batch: 8 distinct requests x 7 questions.

    This is the measurement that matters and the only one this box can take honestly.

    `bench_tickets.py` drives the server over HTTP from a load generator, and on this
    g6.xlarge the generator is **co-resident on 4 vCPUs** with a model whose cost is partly
    CPU dispatch. The client and the server then compete for the exact resource under test,
    and the numbers wander by 2-3x between runs -- at concurrency 8 one run gave 0.67
    requests/s with a 13.4 s server p50, which is a measurement of the client, not the model.
    A second agent sharing the card makes it worse.

    So: no HTTP, no threads, no queueing. One process calls the engine directly on the shape
    `config.pbtxt` actually produces (8 requests x 7 questions = 56 question rows, over the
    `DUP_TOKEN_BUDGET` so it takes the two-pass route), with `cuda.synchronize` around it.
    What is left is GPU work and this process's own dispatch, which is what a kernel change
    is allowed to be judged on.

    56 rows is also the regime the deploy measurements say is **memory-bandwidth-bound**
    rather than dispatch-bound (an L40S with 2.88x the bandwidth of an L4 gave 2.7x the
    throughput, near-linear). Fused kernels reduce arithmetic and memory traffic, so this is
    where they should pay even though they lose at batch 1.
    """
    import torch

    from strands_decider.merged_engine import load_merged_engine
    from strands_decider.schema import SystemOneRequest

    engine = load_merged_engine(checkpoint, merged, device=device, use_prefix_cache=True,
                               max_rows=max_rows, fuse_layers=fused)

    from strands_decider.fused_layers import is_fused
    if is_fused(engine.model.torso) != fused:
        raise SystemExit("[ab] torso fusion state does not match the label")

    # Distinct states, so nothing de-duplicates and the batch is the honest 8-state shape.
    states = [f"Document {i}: {MEDIUM}" for i in range(BATCH_REQUESTS)]
    requests = [SystemOneRequest(state=s, questions=dict(QUESTIONS)) for s in states]
    n_decisions = BATCH_REQUESTS * len(QUESTIONS)

    for _ in range(3):
        engine.evaluate_many(requests)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        engine.evaluate_many(requests)
    torch.cuda.synchronize()
    per_batch_ms = (time.perf_counter() - t0) * 1000.0 / repeats

    return {"ms_per_batch": round(per_batch_ms, 2),
            "decisions_per_s": round(n_decisions * 1000.0 / per_batch_ms, 1),
            "rows": float(n_decisions)}


def compare(ref: dict, got: dict, warn: float) -> int:
    """Report the comparison and return the number of decision flips."""
    missing = sorted(set(ref) ^ set(got))
    if missing:
        print(f"[ab] FAIL: the two runs answered different question sets: {missing[:6]}")
        return len(missing)

    import statistics

    flips = 0
    worst_by_kind: dict[str, tuple[float, str]] = {}
    worst_by_route: dict[str, tuple[float, str]] = {}
    all_by_kind: dict[str, list[float]] = {}
    for key in sorted(ref):
        a, b = ref[key], got[key]
        kind = str(a.get("type"))
        route = key.rsplit("/", 1)[-1]
        if decision(a) != decision(b):
            flips += 1
            print(f"[ab]   {key}: DECISION FLIP {decision(a)} -> {decision(b)}")
        pa, pb = probs_of(a), probs_of(b)
        for k, v in pa.items():
            d = abs(v - pb.get(k, float("nan")))
            all_by_kind.setdefault(kind, []).append(d)
            if d > worst_by_kind.get(kind, (0.0, ""))[0]:
                worst_by_kind[kind] = (d, f"{key}/{k}")
            if d > worst_by_route.get(route, (0.0, ""))[0]:
                worst_by_route[route] = (d, f"{key}/{k}")

    print(f"\n[ab] |delta p| per primitive  (max warns above {warn:g})")
    print(f"[ab]   {'primitive':<8} {'n':>4} {'mean':>10} {'max':>10}")
    for kind in sorted(worst_by_kind):
        d, where = worst_by_kind[kind]
        vals = all_by_kind[kind]
        flag = "   <-- max ABOVE THE BAND" if d > warn else ""
        print(f"[ab]   {kind:<8} {len(vals):>4} {statistics.fmean(vals):>10.6f} "
              f"{d:>10.6f}  at {where}{flag}")
    print("[ab] max |delta p| per readout route")
    for route in sorted(worst_by_route):
        d, where = worst_by_route[route]
        print(f"[ab]   {route:<14} {d:.6f}  at {where}")

    overall = max((d for d, _ in worst_by_kind.values()), default=0.0)
    if overall > warn:
        print(f"\n[ab] FINDING: the worst single probability moves by {overall:.6f}, above "
              f"this project's {warn:g} band. The band is not a tolerance to widen on "
              "request, and this does NOT by itself say which path is wrong -- two bf16 "
              "paths can differ by more than either differs from the exact answer. Run with "
              "--fp32-reference to separate rounding from a kernel error, and read the mean "
              "column above rather than the max.")
    return flips


def triangulate(exact: dict, ref: dict, fused: dict) -> dict[str, dict[str, float]]:
    """Is the fused/reference difference rounding, or is a kernel wrong?

    Comparing two bf16 paths against each other cannot answer that: it gives a magnitude and
    no direction. Comparing both against the SAME torso run in fp32 can, because bf16
    rounding is a distance from the exact answer and a kernel bug is a much larger one.

    **Read the mean, not only the max.** The max is a max over a few hundred probabilities
    and is dominated by whichever one sat nearest a decision boundary; it moves between runs
    and it moves between primitives for no deeper reason than which question was asked. The
    mean is the statistic that separates "both paths are rounding" from "one path is wrong",
    and it is the one `kev`'s own `runs/fused-*` reports lead with (`mean_dp` 1.4e-3 against
    `max_dp` 1.3e-2 for its bf16 path against fp32 on an L40S, with zero argmax flips).

    The verdict therefore keys on the mean ratio, and the threshold is deliberately generous
    in the direction of suspicion: the fused path is allowed to be no more than twice as far
    from fp32 on average as the reference bf16 path is. Beyond that, two bf16 paths that
    differ only in rounding cannot explain it.
    """
    import statistics

    acc: dict[str, dict[str, list[float]]] = {}
    for key in sorted(exact):
        if key not in ref or key not in fused:
            continue
        kind = str(exact[key].get("type"))
        pe, pr, pf = probs_of(exact[key]), probs_of(ref[key]), probs_of(fused[key])
        row = acc.setdefault(kind, {"reference": [], "fused": []})
        for k, v in pe.items():
            row["reference"].append(abs(v - pr.get(k, float("nan"))))
            row["fused"].append(abs(v - pf.get(k, float("nan"))))

    rows: dict[str, dict[str, float]] = {}
    print("\n[ab] distance from the same torso in fp32 (lower is more accurate)")
    print(f"[ab]   {'primitive':<8} {'n':>4} "
          f"{'ref mean':>10} {'fused mean':>11} {'ref max':>10} {'fused max':>10}")
    for kind in sorted(acc):
        r, f = acc[kind]["reference"], acc[kind]["fused"]
        rows[kind] = {
            "n": float(len(r)),
            "reference_mean": statistics.fmean(r), "fused_mean": statistics.fmean(f),
            "reference_max": max(r), "fused_max": max(f),
        }
        v = rows[kind]
        print(f"[ab]   {kind:<8} {len(r):>4} {v['reference_mean']:>10.6f} "
              f"{v['fused_mean']:>11.6f} {v['reference_max']:>10.6f} "
              f"{v['fused_max']:>10.6f}")

    all_ref = [d for v in acc.values() for d in v["reference"]]
    all_fused = [d for v in acc.values() for d in v["fused"]]
    mean_ref, mean_fused = statistics.fmean(all_ref), statistics.fmean(all_fused)
    ratio = mean_fused / mean_ref if mean_ref else float("inf")
    rows["__overall"] = {"reference_mean": mean_ref, "fused_mean": mean_fused,
                         "fused_over_reference": ratio}
    print(f"[ab]   overall mean distance from fp32: reference {mean_ref:.6f}, "
          f"fused {mean_fused:.6f}  ({ratio:.2f}x)")

    if ratio > 2.0:
        print("[ab] FINDING: on average the fused path is more than twice as far from fp32 "
              "as the reference bf16 path. Two bf16 paths differing only in rounding do not "
              "do that, so a kernel is computing the wrong thing. Do not ship this.")
    else:
        print("[ab] The two bf16 paths sit the same distance from fp32 on average, so the "
              "fused-vs-reference difference is bf16 rounding rather than a kernel error: "
              "both paths are about equally close to the exact answer, they are simply not "
              "close to each other. Per-primitive maxima do not order consistently, which is "
              "what rounding looks like and a systematic bug does not. It is still a real "
              "change to the numbers a caller sees, so the decision gates are the thing to "
              "trust, not this.")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="/opt/strands-decider/checkpoint")
    ap.add_argument("--merged", default="/opt/strands-decider/merged")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warn", type=float, default=7e-3)
    ap.add_argument("--max-rows", type=int, default=128)
    ap.add_argument("--time", action="store_true",
                    help="also time the torso forward alone, both ways")
    ap.add_argument("--fp32-reference", action="store_true",
                    help="also run the unfused torso in fp32, to tell rounding from a bug")
    ap.add_argument("--fp32-max-rows", type=int, default=16,
                    help="row chunk for the fp32 arm; fp32 doubles weights and activations")
    ap.add_argument("--tokens", default="128,512,1024,3072")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", default="", help="write the full comparison to this JSON file")
    ap.add_argument("--batch-time", action="store_true",
                    help="time evaluate_many on a full Triton batch (8 requests x 7 questions)")
    ap.add_argument("--only", choices=["reference", "fused"], default="",
                    help="run one torso only and exit. With --batch-time, lets a shell loop "
                         "interleave the two modes so background load on a shared box "
                         "averages across both instead of landing on one")
    args = ap.parse_args()

    if args.only:
        # One mode, one line, nothing else. The interleaving is the caller's job.
        if not args.batch_time:
            raise SystemExit("[ab] --only is for --batch-time; without it there is nothing "
                             "to compare against")
        out = time_batch(args.checkpoint, args.merged, args.device, args.max_rows,
                         fused=args.only == "fused", repeats=args.repeats)
        print(f"[ab] BATCHTIME mode={args.only} rows={out['rows']:.0f} "
              f"ms_per_batch={out['ms_per_batch']} "
              f"decisions_per_s={out['decisions_per_s']}")
        print("[ab] FUSED_AB_FINISHED")
        return 0

    specs = cases()
    print(f"[ab] {len(specs)} cases, "
          f"{sum(len(q) for _, _, q in specs)} questions, two readout routes each")

    print("\n[ab] --- reference torso (SD_FUSE_LAYERS=0)")
    ref, ref_t = run(args.checkpoint, args.merged, args.device, args.max_rows,
                     fused=False, specs=specs)
    print(f"[ab]     loaded in {ref_t['load_s']}s, {len(ref)} answers recorded")

    # Free the reference torso before the fused one loads. Two 2B torsos fit on an L4, but
    # this box is shared and an OOM here would look like a fusion failure.
    import gc

    import torch
    gc.collect()
    torch.cuda.empty_cache()

    print("\n[ab] --- fused torso (SD_FUSE_LAYERS=1)")
    got, got_t = run(args.checkpoint, args.merged, args.device, args.max_rows,
                     fused=True, specs=specs)
    print(f"[ab]     loaded in {got_t['load_s']}s, {len(got)} answers recorded")

    print()
    flips = compare(ref, got, args.warn)

    # Timing before the fp32 arbiter, deliberately: fp32 needs twice the weights and is the
    # arm that runs out of memory on a shared card, and losing the timings to someone else's
    # allocation would mean another whole pair of model loads.
    timings: dict[str, dict] = {}
    if args.time:
        counts = [int(x) for x in args.tokens.split(",") if x.strip()]
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n[ab] torso forward, batch 1, {args.repeats} reps after 3 warm-ups (ms)")
        timings["unfused"] = time_torso(args.checkpoint, args.merged, args.device,
                                        fused=False, token_counts=counts,
                                        repeats=args.repeats)
        timings["fused"] = time_torso(args.checkpoint, args.merged, args.device,
                                      fused=True, token_counts=counts,
                                      repeats=args.repeats)
        print(f"[ab]   {'tokens':>8} {'unfused':>10} {'fused':>10} {'speedup':>9}")
        for n in counts:
            u, f = timings["unfused"][str(n)], timings["fused"][str(n)]
            print(f"[ab]   {n:>8} {u:>10.2f} {f:>10.2f} {u / f:>8.2f}x")

    triangle: dict[str, dict[str, float]] = {}
    if args.fp32_reference:
        gc.collect()
        torch.cuda.empty_cache()
        print("\n[ab] --- the same torso in fp32, unfused (the arbiter)")
        # Smaller row chunks than the bf16 arms: fp32 doubles the weights AND the
        # activations, and this is the arm that OOMs on a shared card. Chunking changes
        # nothing it is being asked -- rows are independent and the answers are reassembled
        # by name -- and in fp32 the reduction-order effect that chunking has in bf16 is
        # below anything this comparison can see.
        exact, exact_t = run(args.checkpoint, args.merged, args.device, args.fp32_max_rows,
                             fused=False, specs=specs, torch_dtype="float32")
        print(f"[ab]     loaded in {exact_t['load_s']}s")
        triangle = triangulate(exact, ref, got)

    print(f"\n[ab] {'PASS' if flips == 0 else 'FAIL'}: {flips} decision flip(s) between the "
          "fused and reference torsos")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"reference": ref, "fused": got, "timings": timings,
                       "fp32_triangulation": triangle}, fh, indent=1)
        print(f"[ab] wrote {args.out}")
    print("[ab] FUSED_AB_FINISHED")
    return 0 if flips == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
