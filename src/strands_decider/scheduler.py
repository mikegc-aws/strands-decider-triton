"""Request scheduling for the server: one device, many callers, bounded queueing.

`SystemOneEngine.evaluate` runs a forward pass on one device. Nothing about it is
re-entrant: on MPS two overlapping passes abort the process on a Metal command-buffer
assertion (issue #9), and on CUDA concurrent passes from several threads contend for the
same memory without any of them going faster. The server's handler used to be a plain
`def`, so FastAPI ran it on its 40-thread pool and overlapping requests did exactly that.

So the device is owned by exactly one thread, and HTTP concurrency is decoupled from
device concurrency:

    request  ->  async handler  ->  bounded queue  ->  model thread  ->  engine.evaluate
                 (no thread held)                      (one at a time)

The handler is `async def` and awaits a `concurrent.futures.Future`, so a request waiting
its turn holds no worker thread and a container accepts as many connections as its queue
allows. This is the shape `kev` uses (Apache-2.0); two things here are deliberately
different, both because an unbounded queue is not safe to put behind a load balancer:

* **The queue is bounded.** Full means HTTP 429 with `Retry-After`, immediately. An
  unbounded queue converts overload into unbounded latency, which looks like a hang to
  every caller instead of a clear refusal to one.
* **Queued work has a deadline.** A request that waited longer than `queue_timeout_s`
  gets HTTP 503 and is never handed to the device. Under a backlog the whole drained
  batch is checked before any of it runs, so a queue that built up during a slow pass is
  shed in one go rather than each entry waiting its turn to time out. Running work whose
  caller has already given up is the main way a queue turns a spike into a brownout.

Batching note: the model thread drains everything waiting and processes it as a batch,
but the entries are currently evaluated one after another. The drain is still load-bearing
-- it is what lets the deadline be applied to a whole backlog at once -- and it is the
seam a future cross-request batched forward pass slots into. It is NOT yet a shared
forward pass: two requests have different states, and the expensive part of this model is
the state, so merging them without a state cache would re-encode each one anyway.
Throughput per process is therefore bounded by one pass at a time; scale out with
processes (the model is ~5 GB, so several fit on one 24 GB card).
"""

from __future__ import annotations

import atexit
import queue
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .infer import SystemOneEngine
    from .schema import SystemOneRequest, SystemOneResponse


class Overloaded(RuntimeError):
    """The queue is full. The caller should retry later -- HTTP 429."""


class QueueTimeout(RuntimeError):
    """The request waited past its deadline and was never run -- HTTP 503."""


@dataclass
class SchedulerConfig:
    # Requests the model thread drains at once. Only the deadline sweep is batched today,
    # so this bounds how much backlog is shed per pass rather than GPU batch width.
    max_drain: int = 64
    # Queued requests allowed before 429. Roughly (target p99 / per-request latency):
    # at ~35 ms a pass, 256 is ~9 s of backlog, which is past most callers' patience,
    # so this is a ceiling rather than an operating point.
    max_queue: int = 256
    # A request that has waited this long is refused rather than run.
    queue_timeout_s: float = 30.0
    # Advertised in the 429 so a client backs off rather than spinning.
    retry_after_s: int = 1
    # The GIL hand-off below. The model thread releases the GIL at every device sync and
    # waits to get it back; at CPython's default 5 ms switch interval those waits
    # dominate while the event loop parses requests. `kev` measures ~2x on batch time.
    switch_interval_s: float = 0.0005
    # How many threads drain the queue.
    #
    # ONE is correct for the HF engine and is not a tuning knob there: that engine owns the
    # GPU at batch size 1, so a second thread would interleave device calls for no gain and
    # contend for the GIL. The queue exists precisely to serialise it.
    #
    # MANY is correct for the vLLM engine, where serialising would defeat the point. vLLM
    # batches whatever rows are in flight across its own scheduler, so throughput scales
    # with how many requests are concurrently submitted -- one thread would hand it one
    # request at a time and waste the batching entirely (measured: 5.2 req/s serialised
    # versus ~36 req/s batched on an L4). The engine-level `_last_offsets` race that used to
    # make concurrency unsafe is fixed, so this is now just a question of which engine is
    # underneath.
    workers: int = 1


@dataclass
class _Item:
    request: Any
    future: Future
    enqueued: float = field(default_factory=time.monotonic)


class Scheduler:
    """Serialises `engine.evaluate` onto one thread, with a bounded queue and a deadline."""

    def __init__(self, engine: SystemOneEngine, config: SchedulerConfig | None = None) -> None:
        self.engine = engine
        self.cfg = config or SchedulerConfig()
        self._q: queue.Queue[_Item] = queue.Queue(maxsize=self.cfg.max_queue)
        self._stopping = threading.Event()
        self._ready = threading.Event()
        self._closed = False
        # Counters are plain ints written only by the model thread and read by /health.
        # A torn read would misreport a statistic, never an answer, so they are lock-free.
        self.served = 0
        self.failed = 0
        self.refused_full = 0
        self.refused_timeout = 0
        self.passes = 0
        self.warmup_seconds: float | None = None
        # With more than one worker the counters above have concurrent writers, and
        # `n += 1` is a load-add-store that can lose an increment. A lock costs ~100 ns
        # against a 50 ms request, so take it rather than under-report.
        self._counter_lock = threading.Lock()
        self._prev_switch = sys.getswitchinterval()
        sys.setswitchinterval(self.cfg.switch_interval_s)
        n_workers = max(1, self.cfg.workers)
        # Keep the historic name when there is exactly one worker: that is the HF default,
        # and `test_passes_never_overlap_and_run_on_one_thread` asserts on it to prove the
        # engine is still being serialised onto a single named thread.
        self._threads = [
            threading.Thread(
                target=self._work,
                name="strands-decider-model" if n_workers == 1
                else f"strands-decider-model-{i}",
                daemon=True,
            )
            for i in range(n_workers)
        ]
        for thread in self._threads:
            thread.start()
        # A daemon thread killed inside a device call at interpreter exit can abort the
        # process, so stop it deliberately first.
        atexit.register(self.close)

    # ---- producer side ---------------------------------------------------------------

    def submit(self, request: SystemOneRequest) -> Future:
        """Enqueue a request. Raises `Overloaded` if the queue is full."""
        if self._stopping.is_set():
            raise Overloaded("the server is shutting down")
        item = _Item(request=request, future=Future())
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self.refused_full += 1
            raise Overloaded(
                f"{self.cfg.max_queue} requests are already queued; retry in "
                f"{self.cfg.retry_after_s}s"
            ) from None
        return item.future

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        """Blocking evaluate, for the CLI and tests. The server awaits the future instead."""
        out: SystemOneResponse = self.submit(request).result()
        return out

    # ---- consumer side ---------------------------------------------------------------

    def _work(self) -> None:
        while not self._stopping.is_set():
            try:
                # A short timeout rather than a blocking get, so `close()` is observed
                # promptly on an idle server. It adds no latency to a waiting request.
                batch = [self._q.get(timeout=0.05)]
            except queue.Empty:
                continue
            # With one worker, draining deep is free and lets the deadline sweep shed a
            # stale backlog in one go. With many workers it is actively harmful: the first
            # thread to wake would take the whole queue into its own batch and run it
            # sequentially while every other thread found the queue empty. That is how the
            # first vLLM measurement came out at 4.7 req/s -- identical to the serialised HF
            # engine -- despite 64 workers and vLLM's batching underneath.
            drain = 1 if self.cfg.workers > 1 else self.cfg.max_drain
            while len(batch) < drain:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break

            now = time.monotonic()
            deadline = self.cfg.queue_timeout_s
            # Shed the whole stale backlog before running any of it: if a slow pass let a
            # queue build, those callers are already gone and their work is pure waste.
            runnable = []
            for item in batch:
                waited = now - item.enqueued
                if item.future.cancelled():
                    self._q.task_done()
                elif waited > deadline:
                    with self._counter_lock:
                        self.refused_timeout += 1
                    item.future.set_exception(QueueTimeout(
                        f"request waited {waited:.1f}s, over the {deadline:.0f}s queue "
                        "deadline, and was not run"
                    ))
                    self._q.task_done()
                else:
                    runnable.append(item)

            if runnable:
                with self._counter_lock:
                    self.passes += 1
            for item in runnable:
                # Re-check against the clock, not the sweep above: entries behind a slow
                # pass go stale WHILE the earlier ones in this batch run. Without this
                # the sweep would only ever catch a backlog that was already stale when
                # the batch was drained, and the last entry of a 12-deep batch of 250 ms
                # passes would still run three seconds past its deadline.
                waited = time.monotonic() - item.enqueued
                if waited > deadline:
                    with self._counter_lock:
                        self.refused_timeout += 1
                    item.future.set_exception(QueueTimeout(
                        f"request waited {waited:.1f}s, over the {deadline:.0f}s queue "
                        "deadline, and was not run"
                    ))
                    self._q.task_done()
                    continue
                if not item.future.set_running_or_notify_cancel():
                    self._q.task_done()
                    continue
                try:
                    item.future.set_result(self.engine.evaluate(item.request))
                    with self._counter_lock:
                        self.served += 1
                except BaseException as exc:
                    with self._counter_lock:
                        self.failed += 1
                    item.future.set_exception(exc)
                finally:
                    self._q.task_done()

    # ---- lifecycle -------------------------------------------------------------------

    def warmup(self, state_tokens: Sequence[int] = (16, 256, 2048)) -> float:
        """Run a pass of every shape, so the first real caller does not pay for it.

        The first pass after process start compiles the Gated DeltaNet layers' Triton and
        flash-linear-attention kernels and builds the CUDA context: measured on an L4,
        44 s against ~100 ms warm (issue #22).

        Three dimensions have to be covered, because each compiles separately:

        * **Both readout paths.** One question takes the plain batched path and several
          take the shared-prefix path -- different code, different kernels.
        * **Several state lengths.** Triton caches per shape, so warming a 16-token state
          leaves a 2,000-token one to compile on a real caller. `state_tokens` is
          approximate (words, not tokenised) because it only has to land in the right
          bucket, not be exact.
        * **All three primitives**, since each renders a different option block.

        The compile is cached on disk under `TRITON_CACHE_DIR`, so this is a one-time
        cost per machine rather than per process: measured 44 s on a cold cache and 9 s
        on a warm one. Baking that cache into a container image removes it from start-up
        altogether.
        """
        from .schema import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion, SystemOneRequest

        noul = NoulQuestion(instructions="Is this a warm-up request?")
        choice = ChoiceQuestion(instructions="Which path is this?",
                                criteria={"warmup": "a warm-up pass", "real": "real traffic"})
        score = ScoreQuestion(instructions="How warm is the engine?",
                              criteria=["cold", "warming", "warm"])
        shapes: list[dict[str, Question]] = [
            {"only": noul},                                  # single question: batched path
            {"a": noul, "b": choice, "c": score},            # several: shared-prefix path
        ]
        started = time.perf_counter()
        for n in state_tokens:
            # ~1.3 tokens a word, so this is the right order of magnitude for the bucket.
            state = "warm up the kernels for this state length. " * max(1, n // 10)
            for questions in shapes:
                self.submit(SystemOneRequest(state=state, questions=questions)).result()
        self.warmup_seconds = time.perf_counter() - started
        self._ready.set()
        return self.warmup_seconds

    def mark_ready(self) -> None:
        """Declare readiness without warming (for `--no-warmup`)."""
        self._ready.set()

    @property
    def ready(self) -> bool:
        """True once the engine has actually run a pass.

        `/health` gates on this. Reporting healthy before the first pass tells an
        autoscaler an instance is ready while its next request will take 21 s.
        """
        return self._ready.is_set()

    def stats(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "warmup_seconds": (round(self.warmup_seconds, 2)
                               if self.warmup_seconds is not None else None),
            "queued": self._q.qsize(),
            "max_queue": self.cfg.max_queue,
            "served": self.served,
            "failed": self.failed,
            "refused_queue_full": self.refused_full,
            "refused_queue_timeout": self.refused_timeout,
            "device_passes": self.passes,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stopping.set()
        for thread in self._threads:
            thread.join(timeout=30.0)
        close_engine = getattr(self.engine, "close", None)
        if callable(close_engine):
            # The vLLM engine owns a background event loop and a child process; the HF
            # engine has no `close` and needs none.
            close_engine()
        sys.setswitchinterval(self._prev_switch)
        # Anything still queued never ran; tell its caller rather than leaving it hanging.
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if not item.future.done():
                item.future.set_exception(Overloaded("the server shut down before this ran"))
            self._q.task_done()
