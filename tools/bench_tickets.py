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
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

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
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if bool(args.endpoint) == bool(args.base):
        ap.error("pass exactly one of --endpoint (SageMaker) or --base (direct HTTP)")

    rt = None
    session = None
    if args.endpoint:
        # retries off, so a failure is an error rather than latency in disguise; pool well
        # above the concurrency under test, because botocore's default of 10 would
        # serialise everything past 10 and measure the client.
        cfg = Config(max_pool_connections=256, retries={"max_attempts": 0},
                     read_timeout=300, connect_timeout=30)
        rt = boto3.client("sagemaker-runtime", region_name=args.region, config=cfg)
    else:
        import requests  # local-only dependency; not needed for the SageMaker path
        session = requests.Session()
        session.mount("http://", requests.adapters.HTTPAdapter(
            pool_connections=256, pool_maxsize=256, max_retries=0))

    def body(ticket: str, qs: dict) -> str:
        inner = json.dumps({"state": ticket, "questions": qs})
        return json.dumps({"inputs": [{"name": "REQUEST_JSON", "shape": [1, 1],
                                       "datatype": "BYTES", "data": [inner]}]})

    url = args.base.rstrip("/") + "/invocations" if args.base else ""

    def invoke(payload: str):
        t0 = time.monotonic()
        if rt is not None:
            r = rt.invoke_endpoint(EndpointName=args.endpoint,
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

    def cell(label: str, tickets: list[str], qs: dict, conc: int) -> dict:
        payloads = [body(t, qs) for t in tickets]
        invoke(payloads[0])  # warm this shape; a new shape compiles kernels on first call
        lats: list[float] = []
        server: list[float] = []
        tokens: list[int] = []
        errs: list[str] = []
        deadline = time.monotonic() + args.seconds
        with ThreadPoolExecutor(max_workers=conc) as ex:
            # stride by the worker count so concurrent in-flight requests are different
            # tickets, not the same one N times.
            futs = [ex.submit(drive, invoke, payloads, w, max(conc, 1), deadline,
                              lats, server, tokens, errs) for w in range(conc)]
            for fut in futs:
                fut.result()
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
               "input_tokens": statistics.mode(tokens) if tokens else None}
        print("[bench] {:<22} q={:<3} c={:<3} {:>6} tick/s  {:>6} dec/s  "
              "{:>6} ms/dec  server p50 {:>6} ms  tok {:<5} err {}".format(
                  label, n_q, conc, row["tickets_per_s"], row["decisions_per_s"],
                  row["ms_per_decision_gpu"], row["server_p50_ms"],
                  row["input_tokens"], row["errors"]), flush=True)
        return row

    pool = ticket_pool(args.pool) if args.tickets == "distinct" else [PLAIN]
    docs = ([p + " Attached document follows. " + _DOC * 4 for p in pool]
            if args.tickets == "distinct" else [WITH_DOC])
    rows = []

    print(f"\n[bench] ticket mode: {args.tickets} (pool of {len(pool)})", flush=True)
    print("\n=== 1. concurrency sweep, 7 questions, plain ticket "
          "(the published table's shape)", flush=True)
    for conc in [int(c) for c in args.concurrency.split(",")]:
        rows.append(cell("plain/7q", pool, questions(7), conc))

    print("\n=== 2. question-count sweep at concurrency 1 "
          "(fixed per-request cost vs marginal per question)", flush=True)
    for nq in [int(q) for q in args.qcounts.split(",")]:
        rows.append(cell(f"plain/{nq}q", pool, questions(nq), 1))

    print("\n=== 3. ticket length, 7 questions "
          "(the '~400-token document costs ~4x throughput' claim)", flush=True)
    for name, tix in (("plain", pool), ("with_document", docs)):
        for conc in (1, 8):
            rows.append(cell(f"{name}/7q", tix, questions(7), conc))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"rows": rows, "target": args.endpoint or args.base,
                       "ticket_mode": args.tickets,
                       "seconds_per_cell": args.seconds}, fh, indent=2)
        print(f"\n[bench] wrote {args.out}")
    print("[bench] BENCH_FINISHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
