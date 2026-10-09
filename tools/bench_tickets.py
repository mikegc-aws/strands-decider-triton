"""Benchmark in the shape of the Lambda-GPU/vLLM Decider benchmark, for comparison.

Why this exists rather than reusing `sm_sweep.py`: the published Lambda numbers
(research/strands-decider/BENCHMARK.md, 2026-10-06) are quoted per *ticket* with the
**production set of 7 questions**, and in tickets/s plus decisions/s. This repo's earlier
sweeps used 3 questions and quoted req/s. Those are not the same unit, and comparing them
directly flatters whichever one happens to have fewer questions. So: same question count,
same reported units, three sections matching the three claims being compared.

  1. concurrency sweep at 7 questions  -> tickets/s, decisions/s, p50 "inside the function"
  2. question-count sweep at c=1       -> fixed per-request cost vs marginal cost per question
  3. ticket length at 7 questions      -> the "~400-token document costs ~4x throughput" claim

"Inside the function" is their phrase for latency excluding the caller's round trip. The
equivalent here is the server's own `latency_ms`, which the backend reports in every
response, so that column is like-for-like. The end-to-end column is reported separately
because it is what a caller actually waits for, and it is NOT comparable across the two
systems (different front doors, different regions, different client locations).

Run it in-region. From outside, round-trip dominates and the throughput number measures the
internet rather than the GPU.

WHY `--processes`, AND WHY EVERY ROW CARRIES `loadgen_cpu_pct`. A load generator written in
Python signs every request (SigV4) and parses every response *in the interpreter*, and that
work holds the GIL -- so one process offers only about one core's worth of load however many
threads it is given. That is not a hypothetical: this repository's `ml.g6e.xlarge` figure of
269.8 decisions/s was taken with a single-process generator on the 4-vCPU endpoint host, and
its server-side p50 was only 209 ms. A server that is not queueing is not the thing being
measured, so that number is published as a LOWER BOUND (README.md, "Picking an instance").
`--processes` spreads the client across cores, `--concurrency` stays the TOTAL number of
requests in flight so it remains comparable with every earlier number, and
`loadgen_cpu_pct` is the evidence for whether the client had headroom left. Read it
together with `server_p50_ms`: a server p50 that climbs with concurrency while loadgen CPU
stays low is a measurement of the endpoint; the reverse is a measurement of the client.

Usage:
    bench_tickets.py --endpoint strands-decider-g6 --out bench.json
    bench_tickets.py --endpoint sd-l40s-2xl --processes 16 --concurrency 32,64,128
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import boto3
from botocore.config import Config

# A plausible production triage set of 7, covering the three primitives and the task types
# the published accuracy numbers describe (department routing, "asks for a human",
# frustration). Held fixed across every cell so a change in throughput is attributable.
Q7 = {
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
                            "update": None, "escalate": None, "praise": None,
                            "report_bug": None, "request_refund": None,
                            "track_order": None, "change_plan": None, "other": None}},
    "frustration": {"type": "score", "instructions": "How frustrated is the writer?",
                    "criteria": ["calm", "frustrated", "depressed"]},
    "severity": {"type": "score", "instructions": "How severe is the business impact?",
                 "criteria": ["none", "minor", "moderate", "major", "critical"]},
}

# Subsets and a superset, for the marginal-cost-per-question measurement.
_KEYS = list(Q7)


def questions(n: int) -> dict:
    """n questions. Above 7, the set is repeated under distinct names -- which is the point:
    it measures the cost of *another slot in the same forward pass*, not of new content."""
    if n <= len(_KEYS):
        return {k: Q7[k] for k in _KEYS[:n]}
    out = dict(Q7)
    for i in range(n - len(_KEYS)):
        k = _KEYS[i % len(_KEYS)]
        out[f"{k}_dup{i}"] = Q7[k]
    return out


PLAIN = ("Hi, my last three payouts have failed and I have had no explanation. "
         "I have contacted support twice already and nobody has got back to me. "
         "This is affecting my ability to pay suppliers. Can someone please call me?")

# A pool of DISTINCT tickets, and the reason it has to exist.
#
# Sending the same ticket from every loadgen thread makes this backend look better than it
# is: `_merge_same_state` in model.py collapses requests that share a state into one encode,
# so a batch of 8 identical tickets costs roughly one ticket's prefill. Measured, that is not
# a small effect -- reported `input_tokens` per request drops from 568 to 528 as concurrency
# rises, which is the merge working. Real traffic is all-distinct, and the published
# comparison numbers were taken on real support tickets, so `--tickets distinct` is the
# default and the honest setting. `identical` is kept because the gap between the two
# quantifies what the merge is worth.
_SUBJECTS = [
    ("payouts have failed", "pay suppliers", "PO-4471"),
    ("subscription was charged twice", "reconcile my accounts", "INV-8820"),
    ("order never arrived", "give my customer a date", "ORD-11204"),
    ("return label will not generate", "ship the item back", "RMA-3391"),
    ("login keeps rejecting my password", "access my dashboard", "ACC-7756"),
    ("API keys were revoked without notice", "keep my integration running", "KEY-5503"),
    ("invoice shows the wrong tax rate", "file my quarterly return", "INV-9142"),
    ("refund was approved but never paid", "close my books", "REF-2208"),
]
_OPENERS = ["Hi,", "Hello,", "Good morning,", "Hi there,", "To whom it may concern,",
            "Hey,", "Dear support,", "Morning,"]
_CLOSERS = ["Can someone please call me?", "I need this resolved today.",
            "Please escalate this.", "Who can actually help here?",
            "I would like to speak to a person.", "This needs a human, not a bot.",
            "Please reply with a timeline.", "I am losing patience."]


def ticket_pool(n: int = 64) -> list[str]:
    """`n` distinct tickets of comparable length. Varied on several axes at once so the
    tokeniser sees genuinely different prefixes rather than one template with a changed
    number -- a shared long prefix would partially defeat the point of using distinct
    tickets at all."""
    out = []
    for i in range(n):
        subject, impact, ref = _SUBJECTS[i % len(_SUBJECTS)]
        opener = _OPENERS[(i // len(_SUBJECTS)) % len(_OPENERS)]
        closer = _CLOSERS[(i * 3 + 1) % len(_CLOSERS)]
        out.append(
            f"{opener} my {subject} and I have had no explanation. Reference {ref}, "
            f"raised {(i % 28) + 1} days ago. I have contacted support "
            f"{(i % 4) + 2} times already and nobody has got back to me. This is "
            f"affecting my ability to {impact}, and the amount involved is "
            f"${(i * 137) % 9000 + 250}.{(i * 7) % 100:02d}. {closer}")
    return out

# The "parsed document" case the published benchmark calls out as ~4x worse for throughput.
_DOC = ("Transaction log excerpt: payout PO-4471 attempted 2026-09-28 status FAILED "
        "reason INSUFFICIENT_SETTLEMENT_BALANCE; payout PO-4482 attempted 2026-09-30 "
        "status FAILED reason INSUFFICIENT_SETTLEMENT_BALANCE; support case SC-99812 "
        "opened 2026-10-01 severity normal owner unassigned; escalation policy requires "
        "response within two business days for severity normal. ")
WITH_DOC = PLAIN + " Attached document follows. " + _DOC * 4

TICKETS = {"plain": PLAIN, "with_document": WITH_DOC}


def pct(srt: list[float], p: float) -> float:
    return round(srt[min(len(srt) - 1, int(p * len(srt)))] * 1000, 1) if srt else -1.0


def drive(invoke, payloads: list[str], start: int, stride: int, deadline: float,
          lats: list[float], server: list[float], tokens: list[int],
          errs: list[str]) -> None:
    """One loadgen thread. Module level and fully parameterised: closing over the sweep
    loop's variables is how a harness silently measures the wrong cell.

    Each thread walks the payload pool from its own offset, so the requests in flight at any
    instant are different tickets rather than the same one N times.
    """
    i = start
    while time.monotonic() < deadline:
        payload = payloads[i % len(payloads)]
        i += stride
        try:
            elapsed, server_ms, tok = invoke(payload)
            lats.append(elapsed)
            if server_ms is not None:
                server.append(server_ms)
            if tok is not None:
                tokens.append(tok)
        except Exception as exc:
            # A failed request is a result, not a reason to abort the sweep.
            errs.append(repr(exc)[:150])


def body(ticket: str, qs: dict) -> str:
    """The KServe v2 envelope the DLC's /invocations proxy expects.

    `shape: [1, 1]` because the model sets `max_batch_size > 0` and Triton prepends the
    batch dimension; `[1]` is the most common way to get an opaque "Unable to parse
    'inputs'" out of this endpoint.
    """
    inner = json.dumps({"state": ticket, "questions": qs})
    return json.dumps({"inputs": [{"name": "REQUEST_JSON", "shape": [1, 1],
                                   "datatype": "BYTES", "data": [inner]}]})


def make_invoke(endpoint: str, base: str, region: str):
    """Build the one-request callable, per PROCESS.

    Constructed here rather than passed in because a botocore client is not picklable and
    must not be shared across a fork: its connection pool would be inherited by every child
    and the children would then race on the same sockets.

    `retries={"max_attempts": 0}` is deliberate -- with retries on, a throttled or failed
    request is retried inside `invoke_endpoint` and shows up as *latency* rather than as an
    error, which hides exactly what a load test is for. `max_pool_connections` is well above
    any per-process concurrency, because botocore's default of 10 would serialise the rest
    and measure the client.
    """
    if endpoint:
        cfg = Config(max_pool_connections=256, retries={"max_attempts": 0},
                     read_timeout=300, connect_timeout=30)
        rt = boto3.client("sagemaker-runtime", region_name=region, config=cfg)
        session = None
        url = ""
    else:
        import requests  # local-only dependency; not needed for the SageMaker path

        rt = None
        session = requests.Session()
        session.mount("http://", requests.adapters.HTTPAdapter(
            pool_connections=256, pool_maxsize=256, max_retries=0))
        url = base.rstrip("/") + "/invocations"

    def invoke(payload: str):
        t0 = time.monotonic()
        if rt is not None:
            r = rt.invoke_endpoint(EndpointName=endpoint,
                                   ContentType="application/json", Body=payload)
            env = json.loads(r["Body"].read().decode())
        else:
            resp = session.post(url, data=payload, timeout=300,
                                headers={"Content-Type": "application/json"})
            env = resp.json()
        out = None
        for o in env.get("outputs") or []:
            if o.get("name") == "RESPONSE_JSON":
                out = json.loads(o["data"][0])
        if out is None or "error" in out:
            raise RuntimeError(str(out)[:150])
        usage = out.get("usage") or {}
        return time.monotonic() - t0, out.get("latency_ms"), usage.get("input_tokens")

    return invoke


def cpu_jiffies() -> tuple[int, int] | None:
    """(total, idle) CPU jiffies for the whole box, or None where there is no /proc.

    Read from /proc/stat rather than taken from `psutil` so the load generator needs no
    dependency beyond boto3, and returns None on macOS instead of raising -- the harness
    runs on a laptop too, and a missing CPU figure must degrade to "unknown" rather than
    break the sweep. `loadgen_cpu_pct: null` in a row is then an honest statement that this
    run cannot prove the client was not the bottleneck.
    """
    try:
        with open("/proc/stat") as fh:
            parts = fh.readline().split()
    except OSError:
        return None
    if not parts or parts[0] != "cpu":
        return None
    vals = [int(v) for v in parts[1:]]
    # idle + iowait. iowait counts as not-busy on purpose: a load generator blocked on a
    # socket is waiting for the server, which is the state we WANT it in.
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    return sum(vals), idle


def cpu_busy_pct(before: tuple[int, int] | None,
                 after: tuple[int, int] | None) -> float | None:
    """Mean busy CPU across all cores over the window, as a percentage of the whole box."""
    if before is None or after is None:
        return None
    total = after[0] - before[0]
    idle = after[1] - before[1]
    if total <= 0:
        return None
    return round(100.0 * (total - idle) / total, 1)


def split_workers(total: int, procs: int) -> list[int]:
    """Spread `total` in-flight requests over at most `procs` processes.

    `--concurrency` is the TOTAL number of requests in flight, not a per-process number, so
    that a c=32 row stays comparable with every c=32 row measured before `--processes`
    existed. Remainders go to the first processes rather than being rounded away: dropping
    them would silently offer less load than the sweep says it did.
    """
    procs = max(1, min(procs, total))
    base, extra = divmod(total, procs)
    return [base + (1 if i < extra else 0) for i in range(procs)]


def _run_block(spec: dict) -> dict:
    """One load-generating process: its own client, its own threads, a shared window.

    Module level and fed a plain dict because this is the function `ProcessPoolExecutor`
    pickles. The processes align on `start_at` -- an absolute wall-clock time -- so all of
    them are offering load over the same interval; without that the first process to start
    would measure an idle server and the last an already-loaded one, and the aggregate
    throughput would be lower than either.
    """
    invoke = make_invoke(spec["endpoint"], spec["base"], spec["region"])
    while True:
        wait = spec["start_at"] - time.time()
        if wait <= 0:
            break
        time.sleep(min(wait, 0.01))
    deadline = time.monotonic() + spec["seconds"]
    lats: list[float] = []
    server: list[float] = []
    tokens: list[int] = []
    errs: list[str] = []
    with ThreadPoolExecutor(max_workers=spec["workers"]) as ex:
        # Stride by the GLOBAL worker count and start at this process's global offset, so
        # the requests in flight across the whole generator are different tickets rather
        # than each process replaying the same slice of the pool.
        futs = [ex.submit(drive, invoke, spec["payloads"], spec["offset"] + w,
                          spec["stride"], deadline, lats, server, tokens, errs)
                for w in range(spec["workers"])]
        for fut in futs:
            fut.result()
    return {"lats": lats, "server": server, "tokens": tokens, "errs": errs}


def summarise(label: str, n_q: int, conc: int, seconds: float, ticket_mode: str,
              pool: int, procs: int, blocks: list[dict],
              loadgen_cpu: float | None) -> dict:
    """Merge the per-process results into one row. Pure, so it can be tested without AWS."""
    lats = [v for b in blocks for v in b["lats"]]
    server = [v for b in blocks for v in b["server"]]
    tokens = [v for b in blocks for v in b["tokens"]]
    errs = [v for b in blocks for v in b["errs"]]
    tickets_s = len(lats) / seconds
    srt = sorted(lats)
    return {"label": label, "ticket": label.split("/")[0], "questions": n_q,
            "ticket_mode": ticket_mode, "pool": pool,
            "concurrency": conc, "processes": procs,
            "tickets": len(lats), "errors": len(errs), "error_sample": errs[:2],
            "tickets_per_s": round(tickets_s, 2),
            "decisions_per_s": round(tickets_s * n_q, 1),
            "ms_per_decision_gpu": round(1000.0 / (tickets_s * n_q), 2) if tickets_s else None,
            "server_p50_ms": round(statistics.median(server), 1) if server else None,
            "server_p95_ms": round(sorted(server)[min(len(server) - 1,
                                                      int(0.95 * len(server)))], 1)
                              if server else None,
            "e2e_p50_ms": pct(srt, 0.50), "e2e_p95_ms": pct(srt, 0.95),
            "input_tokens": statistics.mode(tokens) if tokens else None,
            # The honesty column. See the module docstring: a throughput figure taken with
            # a saturated client is a measurement of the client.
            "loadgen_cpu_pct": loadgen_cpu}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="",
                    help="SageMaker endpoint name; mutually exclusive with --base")
    ap.add_argument("--base", default="",
                    help="HTTP base of a container, e.g. http://localhost:8100. Use this to "
                         "compare engine settings without redeploying an endpoint.")
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--concurrency", default="1,4,8,16,32")
    ap.add_argument("--qcounts", default="1,3,7,14")
    ap.add_argument("--tickets", default="distinct", choices=["distinct", "identical"],
                    help="distinct (default, realistic) sends a different ticket per "
                         "request; identical sends one ticket, which lets the backend's "
                         "same-state merge collapse the batch and inflates throughput")
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--processes", type=int, default=1,
                    help="load-generating PROCESSES to spread --concurrency over. 1 (the "
                         "default) reproduces every number measured before this flag "
                         "existed. Raise it on a box with spare cores: a single Python "
                         "process offers about one core's worth of load because SigV4 "
                         "signing and JSON parsing hold the GIL, and a client at its "
                         "ceiling measures itself rather than the endpoint")
    ap.add_argument("--sections", default="1,2,3",
                    help="which sweep sections to run; '1' is the concurrency sweep alone, "
                         "which is what a ladder across instance types needs and costs a "
                         "third of the GPU minutes. '4' is the --qcounts x --concurrency "
                         "grid, for comparing a configuration knob across both regimes")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if bool(args.endpoint) == bool(args.base):
        ap.error("pass exactly one of --endpoint (SageMaker) or --base (direct HTTP)")

    # One client in the parent, for warming a shape only. The load itself is driven by
    # `_run_block`, which builds its own per process -- see `make_invoke`.
    warm_invoke = make_invoke(args.endpoint, args.base, args.region)
    sections = {s.strip() for s in args.sections.split(",") if s.strip()}

    def cell(label: str, tickets: list[str], qs: dict, conc: int) -> dict:
        payloads = [body(t, qs) for t in tickets]
        warm_invoke(payloads[0])  # warm this shape; a new shape compiles kernels first call
        workers = split_workers(conc, args.processes)
        offsets = [sum(workers[:i]) for i in range(len(workers))]
        # 1 s of slack so every process has built its client and is parked on the barrier
        # before the window opens. A process still importing boto3 when the clock starts
        # would otherwise be counted as offered load it never offered.
        start_at = time.time() + (1.0 if len(workers) > 1 else 0.0)
        specs = [{"endpoint": args.endpoint, "base": args.base, "region": args.region,
                  "payloads": payloads, "workers": w, "offset": off, "stride": max(conc, 1),
                  "seconds": args.seconds, "start_at": start_at}
                 for w, off in zip(workers, offsets, strict=True)]

        if len(specs) == 1:
            cpu0 = cpu_jiffies()
            blocks = [_run_block(specs[0])]
            cpu1 = cpu_jiffies()
        else:
            with ProcessPoolExecutor(max_workers=len(specs)) as px:
                futs = [px.submit(_run_block, spec) for spec in specs]
                # Sample CPU across the measured window only, not across process startup.
                time.sleep(max(0.0, start_at - time.time()))
                cpu0 = cpu_jiffies()
                blocks = [f.result() for f in futs]
                cpu1 = cpu_jiffies()

        row = summarise(label, len(qs), conc, args.seconds, args.tickets, len(payloads),
                        len(specs), blocks, cpu_busy_pct(cpu0, cpu1))
        print("[bench] {:<22} q={:<3} c={:<4} p={:<3} {:>6} tick/s  {:>6} dec/s  "
              "server p50 {:>7} ms  p95 {:>7} ms  e2e p50 {:>7} ms  "
              "loadgen cpu {}%  tok {:<5} err {}".format(
                  label, row["questions"], conc, len(specs), row["tickets_per_s"],
                  row["decisions_per_s"], row["server_p50_ms"], row["server_p95_ms"],
                  row["e2e_p50_ms"], row["loadgen_cpu_pct"],
                  row["input_tokens"], row["errors"]), flush=True)
        return row

    pool = ticket_pool(args.pool) if args.tickets == "distinct" else [PLAIN]
    docs = ([p + " Attached document follows. " + _DOC * 4 for p in pool]
            if args.tickets == "distinct" else [WITH_DOC])
    rows = []

    print(f"\n[bench] ticket mode: {args.tickets} (pool of {len(pool)}), "
          f"{args.processes} loadgen process(es)", flush=True)
    if "1" in sections:
        print("\n=== 1. concurrency sweep, 7 questions, plain ticket "
              "(the published table's shape)", flush=True)
        for conc in [int(c) for c in args.concurrency.split(",")]:
            rows.append(cell("plain/7q", pool, questions(7), conc))

    if "2" in sections:
        print("\n=== 2. question-count sweep at concurrency 1 "
              "(fixed per-request cost vs marginal per question)", flush=True)
        for nq in [int(q) for q in args.qcounts.split(",")]:
            rows.append(cell(f"plain/{nq}q", pool, questions(nq), 1))

    if "3" in sections:
        print("\n=== 3. ticket length, 7 questions "
              "(the '~400-token document costs ~4x throughput' claim)", flush=True)
        for name, tix in (("plain", pool), ("with_document", docs)):
            for conc in (1, 8):
                rows.append(cell(f"{name}/7q", tix, questions(7), conc))

    if "4" in sections:
        # The grid sections 1 and 2 between them cannot produce: section 1 is 7 questions
        # at every concurrency, section 2 is every question count at concurrency 1, and
        # neither gives a low question count UNDER LOAD.
        #
        # That cell is the one an accelerator comparison turns on. CUDA graphs remove a
        # per-pass CPU dispatch floor, which is most of a one-question request and almost
        # none of a seven-question one; fused kernels do the opposite, paying off only once
        # a pass is wide enough to be arithmetic-bound. Measuring a knob at 7q/c=1 and
        # 1q/c=1 only would miss both effects.
        print("\n=== 4. questions x concurrency grid "
              "(a knob's effect in the dispatch-bound and saturated regimes)", flush=True)
        for nq in [int(q) for q in args.qcounts.split(",")]:
            for conc in [int(c) for c in args.concurrency.split(",")]:
                rows.append(cell(f"plain/{nq}q", pool, questions(nq), conc))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"rows": rows, "target": args.endpoint or args.base,
                       "ticket_mode": args.tickets, "processes": args.processes,
                       "seconds_per_cell": args.seconds}, fh, indent=2)
        print(f"\n[bench] wrote {args.out}")
    print("[bench] BENCH_FINISHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
