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

Usage:
    bench_tickets.py --endpoint strands-decider-g6 --out bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import resource
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


def _read_proc_stat() -> tuple[float, float] | None:
    """(busy jiffies, total jiffies) from /proc/stat, or None where there is no /proc."""
    try:
        with open("/proc/stat") as fh:
            parts = fh.readline().split()
    except OSError:
        return None
    if not parts or parts[0] != "cpu":
        return None
    values = [float(v) for v in parts[1:]]
    total = sum(values)
    # fields: user nice system idle iowait irq softirq steal guest guest_nice
    idle = values[3] + (values[4] if len(values) > 4 else 0.0)
    return total - idle, total


class CpuSampler:
    """Loadgen CPU utilisation across a benchmark cell.

    Not decoration. Every throughput number from a load generator is really
    `min(what the server can serve, what the client can offer)`, and the two are
    indistinguishable in the output -- this project has already published one figure that
    turned out to measure a 4-vCPU client rather than the endpoint (see the g6e "LOWER
    BOUND" caveat in deploy/create_endpoint.py). A client pinned near 100% CPU invalidates
    the cell; well under 60% is the evidence that it does not.

    Two readings, because they answer different questions:

      * `system`  -- whole-box busy% from /proc/stat. The honest one: it includes the boto3
        TLS handshakes, the kernel's network stack and anything else sharing the box.
      * `harness` -- this process and its reaped children, from getrusage, as a percentage
        of one core times `nproc`. Portable (there is no /proc on macOS) and it attributes
        the cost to the harness specifically.

    Reported together; `loadgen_cpu_pct` is the system reading where available, because it
    is the one that can rule the client out as the limit.
    """

    def __init__(self) -> None:
        self.ncpu = os.cpu_count() or 1
        self._wall = time.monotonic()
        self._stat = _read_proc_stat()
        self._cpu = self._rusage()

    @staticmethod
    def _rusage() -> float:
        total = 0.0
        for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN):
            usage = resource.getrusage(who)
            total += usage.ru_utime + usage.ru_stime
        return total

    def read(self) -> dict:
        elapsed = max(time.monotonic() - self._wall, 1e-6)
        out: dict = {"loadgen_cpu_pct": None, "loadgen_harness_cpu_pct": None,
                     "loadgen_vcpu": self.ncpu}
        now_stat = _read_proc_stat()
        if self._stat and now_stat:
            busy = now_stat[0] - self._stat[0]
            total = now_stat[1] - self._stat[1]
            if total > 0:
                out["loadgen_cpu_pct"] = round(100.0 * busy / total, 1)
        harness = (self._rusage() - self._cpu) / (elapsed * self.ncpu)
        out["loadgen_harness_cpu_pct"] = round(100.0 * harness, 1)
        if out["loadgen_cpu_pct"] is None:
            # No /proc: the harness reading is the best available proxy, so report it as
            # the headline rather than leaving the field empty and un-judgeable.
            out["loadgen_cpu_pct"] = out["loadgen_harness_cpu_pct"]
        return out


def _run_shard(spec: dict) -> tuple[list[float], list[float], list[int], list[str]]:
    """One loadgen PROCESS: build a client, run `threads` threads until `deadline`.

    A separate process rather than more threads because the client is the thing being ruled
    out, and threads cannot be. `invoke_endpoint` spends its time in JSON encode/decode,
    SigV4 signing (HMAC in python) and TLS -- all of which hold the GIL -- so one process
    saturates around one core's worth of request issue no matter how many threads it runs.
    At 300+ decisions/s that ceiling is close enough to the endpoint's to be mistaken for
    it. `--processes` is what makes "doubling the client moves nothing" a real check.

    The boto3 client is constructed HERE, inside the child. botocore clients are not
    fork-safe -- a shared SSL context inherited across a fork produces sporadic handshake
    errors that look exactly like endpoint-side 5xx noise, which is a measurement bug that
    reads as a result.
    """
    lats: list[float] = []
    server: list[float] = []
    tokens: list[int] = []
    errs: list[str] = []

    # Build the client BEFORE the barrier. Constructing a botocore client loads and parses
    # the service JSON and costs ~0.5-1.5 s, and `fork` of a 16-way pool serialises some of
    # that -- so without the barrier the setup is charged to the measurement window.
    invoke = _make_invoke(spec["endpoint"], spec["base"], spec["region"])

    # THE BARRIER, and why it is load-bearing rather than tidy.
    #
    # Every shard waits for one absolute wall-clock instant and then runs for exactly
    # `seconds`, so throughput is `requests / seconds` for real. Without it each shard
    # started when it happened to finish forking, ran for LESS than `seconds`, and the
    # parent still divided by the full `seconds` -- so the cell under-reported by whatever
    # fraction of the window startup ate.
    #
    # MEASURED, and it is not small. On an L40S endpoint at 15 s per cell, against
    # Little's law (concurrency / end-to-end p50, which the server's own flat p50 makes a
    # reliable cross-check):
    #
    #     concurrency  processes  threads/proc  measured  N/e2e_p50  ratio
    #       1             1           1           9.87      9.73     1.01   <- no fork
    #       8             8           1          25.73     31.31     0.82
    #      16            16           1          17.67     38.85     0.45   <- 55% low
    #      32            16           2          40.07     38.78     1.03
    #      64            16           4          42.20     38.78     1.09
    #
    # The deficit lands exactly on the cells with ONE thread per process, because a shard
    # issuing one request at a time completes only tens of requests in 15 s, so a few
    # seconds of startup is tens of percent of its output. Cells with two or more threads
    # per process amortise it and agree with Little's law.
    #
    # This is almost certainly the "bistability" previously recorded as undiagnosed: two
    # cells at concurrency 17 reading 23.4-23.7 tickets/s where neighbours and repeats gave
    # 39-44, at UNCHANGED server p50 and zero errors. A real batching collapse would move
    # the server's own latency; a client that measured itself for part of the window does
    # not, which is exactly the signature reported.
    start = spec["start"]
    now = time.time()
    if start > now:
        time.sleep(start - now)
    else:
        # Setup overran the grace period. Say so rather than silently producing the short
        # window this barrier exists to prevent.
        errs.append(f"shard missed the start barrier by {now - start:.2f}s")

    deadline_mono = time.monotonic() + spec["seconds"]
    threads = max(1, spec["threads"])
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = [ex.submit(drive, invoke, spec["payloads"],
                          spec["offset"] + w, spec["stride"], deadline_mono,
                          lats, server, tokens, errs)
                for w in range(threads)]
        for fut in futs:
            fut.result()
    return lats, server, tokens, errs


def startup_grace(nproc: int) -> float:
    """Wall-clock budget for every shard to fork and build its boto3 client.

    Scales with the shard count because the cost is mostly contended CPU and page-cache
    work during `fork` plus service-model parsing, not a constant. Measured at ~1 s per
    client on a c7i; 0.4 s per shard plus 3 s of slack covers a 32-way pool with room, and
    a shard that still misses the barrier records an error rather than quietly measuring a
    short window.
    """
    return 3.0 + 0.4 * nproc


def split_threads(conc: int, processes: int) -> list[int]:
    """Thread counts per loadgen process, summing to EXACTLY `conc`.

    Exactly, because the sum IS the offered concurrency the cell reports. Rounding each
    shard up (the obvious `ceil(conc / nproc)` for all) offers more load than the label
    claims -- at `conc=10, processes=4` it would run 12 threads and attribute the resulting
    throughput to 10. Remainders are therefore spread one-per-shard instead.

    Never more processes than threads: an empty shard pays process startup, issues nothing,
    and makes `--processes` look like it changed the answer when all it changed was the
    number of idle children.
    """
    nproc = max(1, min(processes, max(1, conc)))
    base, extra = divmod(conc, nproc)
    return [base + (1 if i < extra else 0) for i in range(nproc)]


def _make_invoke(endpoint: str, base: str, region: str):
    """Build the one-request callable. Shared by the in-process and sharded paths."""
    if endpoint:
        # retries off, so a failure is an error rather than latency in disguise; pool well
        # above the concurrency under test, because botocore's default of 10 would
        # serialise everything past 10 and measure the client.
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
    ap.add_argument("--questions", type=int, default=7,
                    help="questions per request for the CONCURRENCY sweep (section 1). "
                         "7 is the production set and the default, so the published table "
                         "is unchanged. Raising it is how you vary ROWS PER PASS without "
                         "redeploying: rows = batch size x questions, so 8 requests x 14 "
                         "questions puts 112 rows through a pass that normally carries 56, "
                         "which is the same row count as max_batch_size 16 at 7 questions")
    ap.add_argument("--sections", default="1,2,3",
                    help="which sections to run. A full run is ~15 cells; when an endpoint "
                         "costs $3.76/hr and each configuration needs a fresh one, paying "
                         "for cells you will not read is real money")
    ap.add_argument("--tickets", default="distinct", choices=["distinct", "identical"],
                    help="distinct (default, realistic) sends a different ticket per "
                         "request; identical sends one ticket, which lets the backend's "
                         "same-state merge collapse the batch and inflates throughput")
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--processes", type=int, default=1,
                    help="split the offered concurrency across this many loadgen "
                         "PROCESSES. One python process saturates roughly one core "
                         "issuing signed HTTPS requests, so past ~150 decisions/s a "
                         "single-process harness measures itself. Use 16+ on a dedicated "
                         "box, and prove it by doubling this at fixed --concurrency: if "
                         "throughput moves more than noise, the client was the limit")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if bool(args.endpoint) == bool(args.base):
        ap.error("pass exactly one of --endpoint (SageMaker) or --base (direct HTTP)")
    if args.processes < 1:
        ap.error("--processes must be at least 1")

    def body(ticket: str, qs: dict) -> str:
        inner = json.dumps({"state": ticket, "questions": qs})
        return json.dumps({"inputs": [{"name": "REQUEST_JSON", "shape": [1, 1],
                                       "datatype": "BYTES", "data": [inner]}]})

    invoke = _make_invoke(args.endpoint, args.base, args.region)

    def cell(label: str, tickets: list[str], qs: dict, conc: int) -> dict:
        payloads = [body(t, qs) for t in tickets]
        invoke(payloads[0])  # warm this shape; a new shape compiles kernels on first call
        lats: list[float] = []
        server: list[float] = []
        tokens: list[int] = []
        errs: list[str] = []

        shards = split_threads(conc, args.processes)
        nproc = len(shards)
        cpu = CpuSampler()
        if nproc == 1:
            deadline = time.monotonic() + args.seconds
            with ThreadPoolExecutor(max_workers=conc) as ex:
                # stride by the worker count so concurrent in-flight requests are different
                # tickets, not the same one N times.
                futs = [ex.submit(drive, invoke, payloads, w, max(conc, 1), deadline,
                                  lats, server, tokens, errs) for w in range(conc)]
                for fut in futs:
                    fut.result()
        else:
            # A wall-clock START (not a deadline), because it has to mean the same instant
            # in every child; each child sleeps until it, then runs for exactly `seconds`
            # on its own monotonic clock. See the barrier comment in `_run_shard`.
            start = time.time() + startup_grace(nproc)
            specs, offset = [], 0
            for threads in shards:
                specs.append({"endpoint": args.endpoint, "base": args.base,
                              "region": args.region, "payloads": payloads,
                              "threads": threads, "start": start,
                              "seconds": args.seconds,
                              # Stride by the TOTAL thread count across all processes, so
                              # the pool walk stays interleaved rather than every shard
                              # replaying the same slice of tickets.
                              "offset": offset, "stride": conc})
                offset += threads
            with ProcessPoolExecutor(max_workers=nproc) as ex:
                for got_l, got_s, got_t, got_e in ex.map(_run_shard, specs):
                    lats.extend(got_l)
                    server.extend(got_s)
                    tokens.extend(got_t)
                    errs.extend(got_e)
        load = cpu.read()
        n_q = len(qs)
        tickets_s = len(lats) / args.seconds
        srt = sorted(lats)
        row = {"label": label, "ticket": label.split("/")[0], "questions": n_q,
               "ticket_mode": args.tickets, "pool": len(payloads),
               "concurrency": conc, "tickets": len(lats), "errors": len(errs),
               "error_sample": errs[:2],
               "tickets_per_s": round(tickets_s, 2),
               "decisions_per_s": round(tickets_s * n_q, 1),
               "ms_per_decision_gpu": round(1000.0 / (tickets_s * n_q), 2) if tickets_s else None,
               "server_p50_ms": round(statistics.median(server), 1) if server else None,
               "server_p95_ms": round(sorted(server)[min(len(server) - 1,
                                                         int(0.95 * len(server)))], 1)
                                 if server else None,
               "e2e_p50_ms": pct(srt, 0.50), "e2e_p95_ms": pct(srt, 0.95),
               "input_tokens": statistics.mode(tokens) if tokens else None,
               "processes": nproc, **load}
        print("[bench] {:<22} q={:<3} c={:<3} p={:<3} {:>6} tick/s  {:>6} dec/s  "
              "server p50 {:>6} ms  e2e p50 {:>7} ms  cpu {:>5}%  tok {:<5} err {}".format(
                  label, n_q, conc, nproc, row["tickets_per_s"], row["decisions_per_s"],
                  row["server_p50_ms"], row["e2e_p50_ms"], row["loadgen_cpu_pct"],
                  row["input_tokens"], row["errors"]), flush=True)
        return row

    pool = ticket_pool(args.pool) if args.tickets == "distinct" else [PLAIN]
    docs = ([p + " Attached document follows. " + _DOC * 4 for p in pool]
            if args.tickets == "distinct" else [WITH_DOC])
    rows = []

    want = {s.strip() for s in args.sections.split(",") if s.strip()}
    nq1 = args.questions

    print(f"\n[bench] ticket mode: {args.tickets} (pool of {len(pool)}), "
          f"processes {args.processes}", flush=True)
    if "1" in want:
        print(f"\n=== 1. concurrency sweep, {nq1} questions, plain ticket "
              "(the published table's shape)", flush=True)
        for conc in [int(c) for c in args.concurrency.split(",")]:
            rows.append(cell(f"plain/{nq1}q", pool, questions(nq1), conc))

    if "2" in want:
        print("\n=== 2. question-count sweep at concurrency 1 "
              "(fixed per-request cost vs marginal per question)", flush=True)
        for nq in [int(q) for q in args.qcounts.split(",")]:
            rows.append(cell(f"plain/{nq}q", pool, questions(nq), 1))

    if "3" in want:
        print("\n=== 3. ticket length, 7 questions "
              "(the '~400-token document costs ~4x throughput' claim)", flush=True)
        for name, tix in (("plain", pool), ("with_document", docs)):
            for conc in (1, 8):
                rows.append(cell(f"{name}/7q", tix, questions(7), conc))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"rows": rows, "target": args.endpoint or args.base,
                       "ticket_mode": args.tickets,
                       "processes": args.processes,
                       "seconds_per_cell": args.seconds}, fh, indent=2)
        print(f"\n[bench] wrote {args.out}")
    print("[bench] BENCH_FINISHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
