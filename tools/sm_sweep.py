"""Concurrency sweep against a live SageMaker endpoint, via sagemaker-runtime.

The companion to `loadsweep_triton.py`, which hits the container's HTTP port directly. This
one goes through `InvokeEndpoint`, so it measures what a real caller gets: SigV4 signing,
the SageMaker front door, and the DLC's `/invocations` -> KServe v2 proxy. Running both and
comparing is how we know the platform's share of the cost -- measured at about 10 ms of
latency and no throughput loss (40.73 req/s here against 40.07 direct).

**Run it in-region.** From a laptop the round trip dominates: the same request that reports
44 ms of server-side `latency_ms` takes ~500-630 ms end to end from outside the region, so a
throughput number collected from a laptop measures the internet. The numbers in README.md
come from the EC2 box in us-west-2.

Threads, not asyncio, because boto3's client is synchronous. `max_pool_connections` is raised
to 256 for the same reason `loadsweep_triton.py` sets explicit httpx limits: botocore's
default pool is 10, and leaving it there silently serialises everything above concurrency 10
and measures the client.

`retries={"max_attempts": 0}` is deliberate. With retries on, a throttled or failed request
is retried inside `invoke_endpoint` and shows up as *latency* rather than as an error, which
hides exactly what a load test is for.

The caller needs `sagemaker:InvokeEndpoint` on the endpoint ARN.

Usage:
    sm_sweep.py --endpoint strands-decider-g6 --shapes short,long --concurrency 1,4,8,16,32
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "money", "technical": "bugs", "sales": "pricing"}},
    "frustration": {"type": "score", "instructions": "How frustrated is the writer?",
                    "criteria": ["calm", "frustrated", "depressed"]},
}

SHORT = "Help! My payouts have been failing for 3 days! "
PAD = ("The customer has contacted support four times about this. "
       "Each time they were told it would be escalated. ")
SHAPES = {"short": SHORT, "medium": SHORT + PAD * 12, "long": SHORT + PAD * 60}


def pct(srt: list[float], p: float) -> float:
    return round(srt[min(len(srt) - 1, int(p * len(srt)))] * 1000, 1) if srt else -1.0


def drive(invoke, payload: str, deadline: float, lats: list[float],
          server: list[float], errs: list[str]) -> None:
    """One loadgen thread: hammer until the deadline, recording outcomes.

    Defined at module level and given everything explicitly rather than closing over the
    sweep loop's variables. Closures over a loop variable are the classic way for a harness
    to silently measure the wrong cell once anything becomes concurrent across iterations.
    """
    while time.monotonic() < deadline:
        try:
            elapsed, server_ms = invoke(payload)
            lats.append(elapsed)
            if server_ms is not None:
                server.append(server_ms)
        except Exception as exc:
            # A failed request is a data point. Retries are off (see the module docstring),
            # so this is a real error rather than latency in disguise.
            errs.append(repr(exc)[:150])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--shapes", default="short,long")
    ap.add_argument("--concurrency", default="1,4,8,16,32")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = Config(max_pool_connections=256, retries={"max_attempts": 0},
                 read_timeout=300, connect_timeout=30)
    rt = boto3.client("sagemaker-runtime", region_name=args.region, config=cfg)

    def body_for(shape: str) -> str:
        inner = json.dumps({"state": SHAPES[shape], "questions": QUESTIONS})
        return json.dumps({"inputs": [{"name": "REQUEST_JSON", "shape": [1, 1],
                                       "datatype": "BYTES", "data": [inner]}]})

    def one(payload: str) -> tuple[float, float | None]:
        t0 = time.monotonic()
        r = rt.invoke_endpoint(EndpointName=args.endpoint,
                               ContentType="application/json", Body=payload)
        env = json.loads(r["Body"].read().decode())
        out = None
        for o in env.get("outputs") or []:
            if o.get("name") == "RESPONSE_JSON":
                out = json.loads(o["data"][0])
        if out is None or "error" in out:
            raise RuntimeError(str(out)[:120])
        return time.monotonic() - t0, out.get("latency_ms")

    rows = []
    for shape in args.shapes.split(","):
        if shape not in SHAPES:
            continue
        payload = body_for(shape)
        one(payload)  # warm this shape so the first measured call is not a kernel compile
        for conc in [int(c) for c in args.concurrency.split(",")]:
            lats: list[float] = []
            server: list[float] = []
            errs: list[str] = []
            deadline = time.monotonic() + args.seconds

            with ThreadPoolExecutor(max_workers=conc) as ex:
                futures = [ex.submit(drive, one, payload, deadline, lats, server, errs)
                           for _ in range(conc)]
                for fut in futures:
                    fut.result()

            srt = sorted(lats)
            row = {"shape": shape, "concurrency": conc, "requests": len(lats),
                   "errors": len(errs), "error_sample": errs[:2],
                   "req_per_s": round(len(lats) / args.seconds, 2),
                   "p50_ms": pct(srt, 0.50), "p95_ms": pct(srt, 0.95),
                   # The server's own number, so platform overhead is the difference.
                   "server_p50_ms": round(statistics.median(server), 1) if server else None}
            rows.append(row)
            print("[sm] {:<7} c={:<3} {:>7} req/s  e2e p50 {:>7} ms  p95 {:>8} ms  "
                  "server p50 {} ms  err {}".format(
                      row["shape"], conc, row["req_per_s"], row["p50_ms"],
                      row["p95_ms"], row["server_p50_ms"], row["errors"]), flush=True)
            if errs:
                print("   first error:", errs[0], flush=True)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"rows": rows, "endpoint": args.endpoint}, fh, indent=2)
        print(f"[sm] wrote {args.out}")
    print("[sm] SM_SWEEP_FINISHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
