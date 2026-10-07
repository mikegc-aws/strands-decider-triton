"""HTTP server exposing the System One API.

The path and the request and response shapes follow the public Jev API documentation.
JevBench's typesafe adapter runs against this server unchanged (evaluation/jevbench.md).
Compatibility with the Jev API itself is not verified.

Concurrency. The handler is `async def` and the device is owned by one thread behind a
bounded queue (`scheduler.py`). A request waiting its turn holds no worker thread, so the
server accepts many connections while running one pass at a time. The handler used to be
a plain `def`, which put every overlapping request on FastAPI's thread pool and into the
same engine concurrently: that aborted the process on MPS (issue #9) and returned other
requests' numbers at HTTP 200 on every device (issue #17).

Readiness. `/health` is liveness and answers while the model is still compiling kernels.
`/ready` is readiness and returns 503 until the engine has actually run a pass, which on
CUDA is ~21 s after start (issue #22). Point a load balancer or autoscaler at `/ready`;
pointing it at `/health` sends real traffic to an instance whose next request takes 21 s.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .infer import EngineConfig, SystemOneEngine, load_mlx
from .modeling import StrandsDeciderModel
from .scheduler import Overloaded, QueueTimeout, Scheduler, SchedulerConfig
from .schema import SystemOneRequest, SystemOneResponse

_engine: SystemOneEngine | None = None
_scheduler: Scheduler | None = None


def get_engine() -> SystemOneEngine:
    if _engine is None:  # pragma: no cover - guarded by create_app
        raise RuntimeError("engine not initialised")
    return _engine


def get_scheduler() -> Scheduler:
    if _scheduler is None:  # pragma: no cover - guarded by create_app
        raise RuntimeError("scheduler not initialised")
    return _scheduler


def create_app(
    checkpoint: str,
    *,
    device: str = "cuda",
    use_prefix_cache: bool = True,
    model_name: str | None = None,
    attn_implementation: str | None = None,
    strict_window: bool = False,
    max_batch: int = 32,
    vision: bool = False,
    warmup: bool = False,
    max_queue: int = 256,
    queue_timeout_s: float = 30.0,
    engine: Any | None = None,
    workers: int | None = None,
) -> FastAPI:
    """Build the app. `warmup=True` runs a pass before returning, so the app is ready
    the moment it is served; `serve()` turns it on and the tests leave it off."""
    global _engine, _scheduler

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        # Stop the model thread deliberately: a daemon thread killed inside a device
        # call at interpreter exit can abort the process instead of exiting.
        if _scheduler is not None:
            _scheduler.close()

    app = FastAPI(
        title="strands-decider System One",
        version="0.1.0",
        description="Typed, calibrated answers. Choice, Score and Noul over one state.",
        lifespan=lifespan,
    )

    # Identify the response's `model` by the checkpoint served, so multiple servers
    # on the same host cannot be confused. HF repo ids ("org/name") collapse to `name`.
    resolved_name = model_name or os.path.basename(checkpoint.rstrip("/")) or checkpoint

    config = EngineConfig(
        device=device, use_prefix_cache=use_prefix_cache, model_name=resolved_name,
        strict_window=strict_window, max_batch=max_batch,
    )
    if engine is not None:
        # A pre-built engine, so a caller that needs a different forward pass (the vLLM
        # engine, which must be constructed before this process touches CUDA) can supply
        # one without this function growing a branch per backend.
        _engine = engine
    elif vision:
        if device == "mlx":
            raise ValueError("--vision runs on torch devices (cuda, mps, cpu), not mlx")
        from .vision import load_vision_engine

        _engine = load_vision_engine(checkpoint, config, attn_implementation=attn_implementation)
    elif device == "mlx":
        _engine = load_mlx(checkpoint, config)
    else:
        model = StrandsDeciderModel.load(checkpoint, attn_implementation=attn_implementation)
        _engine = SystemOneEngine(model, config)

    # One worker unless told otherwise: correct for the HF engine, which owns the GPU at
    # batch size 1. The vLLM path raises this, because there the queue must feed many rows
    # in flight for vLLM's scheduler to batch them.
    _scheduler = Scheduler(
        _engine,
        SchedulerConfig(
            max_queue=max_queue,
            queue_timeout_s=queue_timeout_s,
            **({"workers": workers} if workers else {}),
        ),
    )
    if warmup:
        _scheduler.warmup()
    else:
        # Not warmed, but the queue is live: say so rather than reporting not-ready
        # forever and leaving `/ready` permanently red for a caller that never asked
        # for a warm-up.
        _scheduler.mark_ready()

    @app.get("/health")
    def health() -> dict:
        """Liveness. Answers while the engine is still compiling kernels."""
        eng = get_engine()
        return {
            "status": "ok",
            # Top level as well as under `scheduler`: /health is the endpoint an operator
            # reaches for first, and "ok" alone on a cold engine reads as "fast".
            "ready": get_scheduler().ready,
            "model": eng.cfg.model_name,
            "checkpoint": checkpoint,
            "base_model": eng.model.config.base_model,
            "num_slots": eng.model.config.num_slots,
            "max_length": eng.model.config.max_length,
            "temperature": eng.model.config.temperature,
            "device": eng.cfg.device,
            "prefix_cache": eng.cfg.use_prefix_cache,
            "vision": vision,
            "scheduler": get_scheduler().stats(),
        }

    @app.get("/ready")
    def ready() -> JSONResponse:
        """Readiness, for a load balancer or autoscaler. 503 until a pass has run."""
        sched = get_scheduler()
        stats = sched.stats()
        if not sched.ready:
            return JSONResponse(status_code=503, content={"status": "warming", **stats})
        return JSONResponse(status_code=200, content={"status": "ready", **stats})

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    async def systemone(request: SystemOneRequest) -> JSONResponse:
        sched = get_scheduler()
        started = time.perf_counter()
        try:
            future = sched.submit(request)
        except Overloaded as exc:
            # 429, not 503: the server is working, this caller should slow down.
            raise HTTPException(
                status_code=429, detail=str(exc),
                headers={"Retry-After": str(sched.cfg.retry_after_s)},
            ) from exc
        try:
            # Awaiting the future rather than blocking means this request occupies no
            # worker thread while it waits for the device.
            response = await asyncio.wrap_future(future)
        except ValueError as exc:
            # Option count over num_slots, over-long prompt under --strict-window,
            # undecodable image: caller error.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except QueueTimeout as exc:
            raise HTTPException(
                status_code=503, detail=str(exc),
                headers={"Retry-After": str(sched.cfg.retry_after_s)},
            ) from exc
        except Overloaded as exc:  # shutdown raced this request
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        payload = response.model_dump()
        # Includes queue wait, unlike a server that times only its own forward pass.
        payload["latency_ms"] = round(elapsed_ms, 2)
        return JSONResponse(content=payload)

    return app


def serve(
    checkpoint: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str = "cuda",
    use_prefix_cache: bool = True,
    model_name: str | None = None,
    strict_window: bool = False,
    max_batch: int = 32,
    vision: bool = False,
    warmup: bool = True,
    max_queue: int = 256,
    queue_timeout_s: float = 30.0,
) -> None:
    import uvicorn

    app = create_app(
        checkpoint,
        device=device,
        use_prefix_cache=use_prefix_cache,
        model_name=model_name,
        strict_window=strict_window,
        max_batch=max_batch,
        vision=vision,
        # On by default here and off in `create_app`: a served instance should be ready
        # before it is reachable, and the first pass costs ~21 s on CUDA.
        warmup=warmup,
        max_queue=max_queue,
        queue_timeout_s=queue_timeout_s,
    )
    # Single worker: the model owns the device, and forking more would duplicate it.
    # Scale out with separate processes (each ~5 GB, so several fit on one 24 GB card),
    # not with uvicorn workers sharing this one.
    uvicorn.run(app, host=host, port=port, workers=1)
