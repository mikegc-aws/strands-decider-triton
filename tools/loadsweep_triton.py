"""Concurrency sweep against the Triton deployable, measuring what the batcher actually does.

Why a new harness rather than `evaluation/serving/loadsweep.py`: that one posts a bare
System One body to `/v1/systemone`, and this deployable speaks SageMaker's `/invocations`,
which on the Triton DLC is a proxy to KServe v2 `infer`. The payload has to travel inside a
v2 envelope, so the request builder is genuinely different. Everything else follows
loadsweep.py's rules deliberately:

  * one `httpx.AsyncClient` with EXPLICIT high pool limits. A shared *sync* client on
    default limits is what made `bench_jev_vs_sd.py` report 4.7 req/s where a plain urllib
    loadgen got 14.8 against the same server -- it measured the client, not the server.
  * the client is verified not to be the bottleneck, by reporting GPU utilisation alongside
    throughput. A flat throughput curve with a *starved* GPU means the loadgen is the limit;
    flat with a saturated GPU is a real ceiling. Without that you cannot tell.
  * every cell records the server's own measured token count, from the response `usage`, so
    cells are comparable by payload rather than by a shape label like "medium".

The loadgen runs on the same host as the server in the measurements recorded in README.md,
which is a 4 vCPU box. That is a real cap on the numbers and is reported, not hidden.

Usage:
    loadsweep_triton.py --base http://localhost:8100 --concurrency 1,4,8,16,32 --seconds 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import time

import httpx

# Three payload shapes. The state is what dominates cost, so it is what varies; the
# question set is held constant so a change in throughput is attributable.
QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "money", "technical": "bugs", "sales": "pricing"}},
    "frustration": {"type": "score", "instructions": "How frustrated is the writer?",
                    "criteria": ["calm", "frustrated", "depressed"]},
}

SHORT = "Help! My payouts have been failing for 3 days! "
# Padded by repetition rather than with lorem ipsum: the tokeniser sees realistic text and
# the measured token count is reported anyway, so the exact content does not need defending.
MEDIUM = SHORT + ("The customer has contacted support four times about this. "
                  "Each time they were told it would be escalated. ") * 12
LONG = SHORT + ("The customer has contacted support four times about this. "
                "Each time they were told it would be escalated. ") * 60

SHAPES = {"short": SHORT, "medium": MEDIUM, "long": LONG}


def envelope(state: str) -> dict:
    """The KServe v2 envelope. shape is [1, 1] because the model sets max_batch_size > 0
    and Triton prepends the batch dimension; [1] is rejected."""
    body = json.dumps({"state": state, "questions": QUESTIONS})
    return {"inputs": [{"name": "REQUEST_JSON", "shape": [1, 1],
                        "datatype": "BYTES", "data": [body]}]}


def unwrap(env: dict) -> dict:
    for o in env.get("outputs") or []:
        if o.get("name") == "RESPONSE_JSON":
            return json.loads(o["data"][0])
    raise ValueError(f"no RESPONSE_JSON in {sorted(env)}")


def gpu_util() -> tuple[float, float]:
    """(utilisation %, memory MiB). Returns (-1, -1) if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        u, m = out.split("\n")[0].split(",")
        return float(u), float(m)
    except Exception:
        return -1.0, -1.0


async def worker(client: httpx.AsyncClient, url: str, payload: dict,
                 deadline: float, lats: list[float], errs: list[str],
                 tokens: list[int]) -> None:
    while time.monotonic() < deadline:
        t0 = time.monotonic()
        try:
            r = await client.post(url, json=payload)
            body = unwrap(r.json())
            if "error" in body:
                errs.append(str(body["error"])[:120])
                continue
            lats.append(time.monotonic() - t0)
            usage = body.get("usage") or {}
            if "input_tokens" in usage:
                tokens.append(int(usage["input_tokens"]))
        except Exception as exc:
            # Any client or transport failure is a data point, not a reason to stop the
            # sweep -- an error rate is part of the result.
            errs.append(f"{type(exc).__name__}: {exc}"[:120])


async def cell(url: str, shape: str, conc: int, seconds: float) -> dict:
    payload = envelope(SHAPES[shape])
    lats: list[float] = []
    errs: list[str] = []
    tokens: list[int] = []
    # Explicit limits, well above the concurrency under test. This is the line that
    # `bench_jev_vs_sd.py` got wrong.
    limits = httpx.Limits(max_connections=512, max_keepalive_connections=512)
    utils: list[float] = []

    async def sampler(deadline: float) -> None:
        while time.monotonic() < deadline:
            u, _ = gpu_util()
            if u >= 0:
                utils.append(u)
            await asyncio.sleep(0.5)

    async with httpx.AsyncClient(timeout=300.0, limits=limits) as client:
        # Warm the path for this shape so the first measured request is not a kernel compile.
        await client.post(url, json=payload)
        deadline = time.monotonic() + seconds
        await asyncio.gather(
            sampler(deadline),
            *[worker(client, url, payload, deadline, lats, errs, tokens)
              for _ in range(conc)],
        )

    n = len(lats)
    srt = sorted(lats)
    def pct(p: float) -> float:
        return round(srt[min(len(srt) - 1, int(p * len(srt)))] * 1000, 1) if srt else -1.0
    return {
        "shape": shape,
        "concurrency": conc,
        "requests": n,
        "errors": len(errs),
        "error_sample": errs[:2],
        "req_per_s": round(n / seconds, 2),
        "p50_ms": pct(0.50),
        "p95_ms": pct(0.95),
        "mean_ms": round(statistics.fmean(lats) * 1000, 1) if lats else -1.0,
        # Comparability is a property of payloads: record what the server actually tokenised.
        "input_tokens": statistics.mode(tokens) if tokens else None,
        "gpu_util_mean": round(statistics.fmean(utils), 1) if utils else None,
        "gpu_util_max": max(utils) if utils else None,
    }


async def run(base: str, path: str, shapes: list[str], concs: list[int],
              seconds: float) -> list[dict]:
    url = base.rstrip("/") + path
    rows = []
    for shape in shapes:
        for conc in concs:
            row = await cell(url, shape, conc, seconds)
            rows.append(row)
            print(f"[sweep] {row['shape']:<7} c={row['concurrency']:<3} "
                  f"{row['req_per_s']:>7} req/s  p50 {row['p50_ms']:>7} ms  "
                  f"p95 {row['p95_ms']:>8} ms  tok {row['input_tokens']}  "
                  f"gpu {row['gpu_util_mean']}%  err {row['errors']}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://localhost:8100")
    ap.add_argument("--path", default="/invocations")
    ap.add_argument("--shapes", default="short,medium,long")
    ap.add_argument("--concurrency", default="1,4,8,16,32")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    shapes = [s for s in args.shapes.split(",") if s in SHAPES]
    concs = [int(c) for c in args.concurrency.split(",")]
    rows = asyncio.run(run(args.base, args.path, shapes, concs, args.seconds))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"rows": rows, "base": args.base}, fh, indent=2)
        print(f"[sweep] wrote {args.out}")
    print("[sweep] SWEEP_FINISHED")
    return 0


if __name__ == "__main__":
    # The guard is not boilerplate here. Without it, `import loadsweep_triton` -- which is
    # what a test collector, a `--help` wrapper or an editor's symbol indexer does -- starts
    # a real 5-cell load sweep against localhost:8100 and blocks for minutes. Every other
    # tool in this directory has the guard; this one did not, and importing it was a
    # 5-minute benchmark.
    raise SystemExit(main())
