"""The Scheduler's `/health` counters, and the one that used to lose increments.

`refused_full` is the queue-full counter, and it was the only one of the five incremented
without `_counter_lock`. That matters more than the others would, because of *where* it is
incremented: `submit()` runs on the producer side, so its writers are the server's request
threads rather than the bounded set of model threads -- and it is only ever non-zero when
the queue is full, i.e. when the largest number of those threads are hitting `queue.Full`
simultaneously. `n += 1` is a load-add-store, so concurrent writers lose increments, and the
metric under-counted exactly when it was being read to decide whether `max_queue` was too
small.

These tests drive `submit` from many threads against a deliberately tiny queue, which is the
only way to make the race observable. They use a stub engine: the race is in the scheduler's
bookkeeping, not in anything that needs a GPU.
"""

from __future__ import annotations

import threading
from typing import ClassVar

import pytest

from strands_decider.scheduler import Overloaded, Scheduler, SchedulerConfig


class _BlockedEngine:
    """An engine whose `evaluate` blocks until released, so the queue can be filled.

    Without the block the model thread would drain the queue as fast as the producers fill
    it and `queue.Full` would never be reached.
    """

    def __init__(self) -> None:
        self.release = threading.Event()
        self.calls = 0

    def evaluate(self, request):
        self.calls += 1
        self.release.wait(timeout=10)
        return request


@pytest.fixture
def blocked():
    engine = _BlockedEngine()
    # max_queue=1 so the queue fills after a single admitted request, and every other
    # producer is refused. max_drain=1 keeps the model thread from emptying it in one go.
    scheduler = Scheduler(engine, SchedulerConfig(max_queue=1, max_drain=1, workers=1))
    yield scheduler, engine
    engine.release.set()
    scheduler.close()


def test_refused_full_counts_every_refusal_under_concurrency(blocked):
    """The regression test for the missing lock.

    32 threads race to submit against a queue of 1. However many are refused, the counter
    must equal the number of `Overloaded` exceptions actually raised -- a lost increment
    shows up as counter < refusals.
    """
    scheduler, _engine = blocked
    refusals: list[int] = []
    lock = threading.Lock()
    barrier = threading.Barrier(32)

    def producer() -> None:
        # Release all 32 at once, to maximise the overlap in `submit`.
        barrier.wait(timeout=10)
        try:
            scheduler.submit(_Request())
        except Overloaded:
            with lock:
                refusals.append(1)

    threads = [threading.Thread(target=producer) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert refusals, "the queue of 1 should have refused most of 32 concurrent submits"
    assert scheduler.refused_full == len(refusals), (
        f"refused_full={scheduler.refused_full} but {len(refusals)} submits were actually "
        "refused; an increment was lost, which is what the counter lock prevents")


def test_refused_full_is_reported_by_stats(blocked):
    """It is surfaced as `refused_queue_full`, which is the name /health exposes."""
    scheduler, _engine = blocked
    for _ in range(6):
        try:
            scheduler.submit(_Request())
        except Overloaded:
            pass
    stats = scheduler.stats()
    assert stats["refused_queue_full"] == scheduler.refused_full
    assert stats["refused_queue_full"] > 0


def test_every_counter_increment_takes_the_lock():
    """Source-level backstop, so a sixth counter cannot be added unlocked.

    Checked by inspection because the race is probabilistic: a bare `self.counter += 1` may
    well pass a behavioural test on a given run and lose increments in production under a
    different thread schedule. This asserts the invariant rather than sampling it.
    """
    import inspect

    from strands_decider import scheduler as module

    source = inspect.getsource(module)
    counters = ("served", "failed", "refused_full", "refused_timeout", "passes")
    offenders = []
    lines = source.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        for name in counters:
            if stripped == f"self.{name} += 1":
                # The immediately preceding non-blank line must be the lock acquisition.
                prev = next((lines[j].strip() for j in range(i - 1, -1, -1)
                             if lines[j].strip()), "")
                if prev != "with self._counter_lock:":
                    offenders.append(f"line {i + 1}: {stripped} (preceded by {prev!r})")
    assert not offenders, (
        "these counter increments are not under _counter_lock and can lose increments:\n"
        + "\n".join(offenders))


class _Request:
    """A stand-in for SystemOneRequest. The scheduler never inspects it on these paths."""

    state = "x"
    questions: ClassVar[dict] = {}
