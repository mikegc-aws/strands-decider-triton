"""The load generator's own arithmetic, with no AWS and no endpoint.

`tools/bench_tickets.py` produces the numbers this project publishes, so a quiet bug in it
is a published wrong answer rather than a failed test. Three parts are worth pinning, and
each corresponds to a way a throughput figure has actually been got wrong here:

  * `split_workers` -- `--concurrency` means TOTAL requests in flight. If spreading it over
    processes dropped a remainder, a c=32 row would silently offer less load than it says,
    and would not be comparable with the c=32 rows measured before `--processes` existed.
  * `cpu_busy_pct` -- the evidence column. `ml.g6e.xlarge` was published at 269.8
    decisions/s with a single-process generator whose server p50 was only 209 ms, and that
    number is a LOWER BOUND because nobody could show the client had headroom. A CPU figure
    that is wrong, or that silently reads 0 where it cannot be measured, is worse than no
    figure: it would be used as proof.
  * `summarise` -- merging per-process results. Dropping one process's samples would
    understate throughput by 1/N and nothing would complain.

The module is loaded by path rather than imported, because `tools/` is a directory of entry
points, not a package -- the same approach `test_deploy.py` uses.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bench():
    pytest.importorskip("boto3", reason="bench_tickets imports boto3 at module level")
    spec = importlib.util.spec_from_file_location(
        "tools_bench_tickets", ROOT / "tools" / "bench_tickets.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tools_bench_tickets"] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------- spreading the offered load


@pytest.mark.parametrize("total,procs", [(1, 1), (1, 8), (32, 1), (32, 4), (32, 5),
                                         (128, 16), (7, 16), (64, 3)])
def test_split_workers_offers_exactly_the_requested_concurrency(bench, total, procs):
    """The invariant that keeps a c=32 row a c=32 row: the per-process worker counts must
    sum to the total, whatever the remainder does."""
    workers = bench.split_workers(total, procs)
    assert sum(workers) == total
    assert all(w >= 1 for w in workers), "a process with no workers is a wasted fork"
    assert len(workers) <= procs


def test_split_workers_never_asks_for_more_processes_than_workers(bench):
    """At concurrency 4, 16 processes would mean 12 processes each building a boto3 client
    and then driving nothing -- startup cost charged to the measured window for no load."""
    assert bench.split_workers(4, 16) == [1, 1, 1, 1]


def test_split_workers_is_as_even_as_it_can_be(bench):
    """Uneven splits skew which tickets are in flight. One extra worker on the first few
    processes is unavoidable; two is a bug."""
    workers = bench.split_workers(34, 4)
    assert workers == [9, 9, 8, 8]
    assert max(workers) - min(workers) <= 1


# ------------------------------------------------------------ the evidence column


def test_cpu_busy_pct_is_the_non_idle_share(bench):
    # 1,000 jiffies passed, 250 of them idle -> 75% busy.
    assert bench.cpu_busy_pct((0, 0), (1000, 250)) == 75.0


def test_cpu_busy_pct_is_none_when_it_cannot_be_measured(bench):
    """None, never 0.0. A zero would read as "the client was idle, so the server was the
    limit" -- exactly the claim this column exists to support -- when in fact there is no
    /proc on the box and nothing was measured at all."""
    assert bench.cpu_busy_pct(None, (1000, 250)) is None
    assert bench.cpu_busy_pct((0, 0), None) is None
    # A window with no elapsed jiffies is also unmeasured rather than 0% or 100%.
    assert bench.cpu_busy_pct((1000, 250), (1000, 250)) is None


def test_cpu_jiffies_returns_a_pair_or_none(bench):
    """Must not raise on a box without /proc: the harness runs on macOS too, where the
    honest answer is "unknown"."""
    got = bench.cpu_jiffies()
    assert got is None or (len(got) == 2 and got[0] >= got[1] >= 0)


# --------------------------------------------------------- merging the processes


def _block(n: int, lat: float, server: float) -> dict:
    return {"lats": [lat] * n, "server": [server] * n, "tokens": [568] * n, "errs": []}


def test_summarise_adds_every_process_up(bench):
    """3 processes x 20 tickets over 10 s is 6 tickets/s, not 2. Losing a block would
    understate the endpoint by 1/N and look like a slower card."""
    blocks = [_block(20, 0.5, 400.0) for _ in range(3)]
    row = bench.summarise("plain/7q", 7, 96, 10.0, "distinct", 64, 3, blocks, 42.0)
    assert row["tickets"] == 60
    assert row["tickets_per_s"] == 6.0
    assert row["decisions_per_s"] == 42.0
    assert row["concurrency"] == 96 and row["processes"] == 3
    assert row["loadgen_cpu_pct"] == 42.0


def test_summarise_reports_server_and_end_to_end_separately(bench):
    """They answer different questions. A server p50 that is not climbing means the server
    is not saturated, whatever the end-to-end number does -- which is the whole reason the
    g6e figure in README.md is labelled a lower bound."""
    row = bench.summarise("plain/7q", 7, 32, 10.0, "distinct", 64, 2,
                          [_block(10, 0.8, 300.0), _block(10, 0.8, 300.0)], 30.0)
    assert row["server_p50_ms"] == 300.0
    assert row["e2e_p50_ms"] == 800.0


def test_summarise_counts_errors_without_counting_them_as_throughput(bench):
    """A failed request is a data point, not a ticket. Counting it would make an endpoint
    that is shedding load look fast."""
    blocks = [{"lats": [0.5] * 4, "server": [100.0] * 4, "tokens": [],
               "errs": ["RuntimeError(...)"] * 6}]
    row = bench.summarise("plain/7q", 7, 8, 2.0, "distinct", 64, 1, blocks, None)
    assert row["tickets"] == 4 and row["errors"] == 6
    assert row["tickets_per_s"] == 2.0
    assert row["error_sample"] == ["RuntimeError(...)"] * 2
    assert row["loadgen_cpu_pct"] is None


def test_summarise_survives_a_cell_that_returned_nothing(bench):
    """A cell can come back empty -- every request errored, or the endpoint was deleted
    mid-sweep. It must produce a row rather than a ZeroDivisionError that loses the rows
    already measured."""
    row = bench.summarise("plain/7q", 7, 8, 5.0, "distinct", 64, 1,
                          [{"lats": [], "server": [], "tokens": [], "errs": []}], None)
    assert row["tickets_per_s"] == 0.0
    assert row["ms_per_decision_gpu"] is None
    assert row["server_p50_ms"] is None


# ------------------------------------------------------------------ the envelope


def test_body_uses_the_batched_envelope_shape(bench):
    """`shape: [1, 1]`, because the model sets max_batch_size > 0 and Triton prepends the
    batch dimension. `[1]` is the most common way to get an opaque "Unable to parse
    'inputs'" out of this endpoint, and it would fail every request in the sweep."""
    import json

    sent = json.loads(bench.body("a ticket", {"q": {"type": "noul", "instructions": "?"}}))
    (tensor,) = sent["inputs"]
    assert tensor["name"] == "REQUEST_JSON"
    assert tensor["shape"] == [1, 1]
    assert tensor["datatype"] == "BYTES"
    assert json.loads(tensor["data"][0])["state"] == "a ticket"
