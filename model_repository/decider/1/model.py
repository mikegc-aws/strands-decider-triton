"""Triton python backend for the Decider's System One API.

This is the custom backend the brief asked for: it loads the model weights, accepts
batched requests, and returns structured probability outputs. What it is **not** is a
reimplementation of inference. The engine, the pointer readout, the prompt rendering, the
window fit and the calibrated confidence formulas all come from `strands_decider`
unchanged; this file is the adapter between Triton's batching and that engine, plus the
one thing Triton is actually here for.

**What Triton adds, precisely.** `strands_decider.scheduler.Scheduler` already provides a
bounded queue, deadlines and a worker pool, so Triton is not here for that. It is here for
cross-request batching, which the Scheduler explicitly does not do -- it drains a batch and
then evaluates it one entry at a time, and its docstring says so: the drain "is the seam a
future cross-request batched forward pass slots into. It is NOT yet a shared forward pass."
That gap is the measured ceiling on the torch path: 12 req/s at one worker, and *worse* with
more (5.2 at 4, 3.8 at 8), because concurrent passes contend without any going faster.

**Two paths through `execute()`, and which one you are actually on.**

1. The *batched* path, and the one production uses. If the engine offers `evaluate_many`
   -- `BatchedSystemOneEngine` does, so `SD_ENGINE=merged` does -- the whole Triton batch
   becomes ONE call, and the engine spends two forward passes on it rather than two per
   request. This is where the saving is.
2. The *per-request* path, `_group_by_state` and friends below. It is NOT dead code and
   NOT merely historical: `SD_ENGINE=hf` builds a plain `SystemOneEngine`, which has no
   `evaluate_many`, so that arm runs here. It is also the fallback if the batched path
   itself raises -- the single-request route shares `_fit`, the readout and the head, so it
   is the same answers by a slower route, which beats failing every caller at once.

Both are covered by `tests/test_triton_backend.py`, which parametrises over an engine with
`evaluate_many` and one without, because "the tests pass" meant the per-request path only
until that was fixed.

Only the merged HF engine belongs behind this. vLLM batches internally across its own
scheduler, so a second batching queue in front of it would add latency and coalesce
nothing.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import numpy as np
import triton_python_backend_utils as pb_utils

# The package and the wire helpers are installed into the image; see deploy/Dockerfile.
for _extra in ("/opt/strands-decider", "/opt/decider-triton"):
    if _extra not in sys.path and os.path.isdir(_extra):
        sys.path.insert(0, _extra)

from decider_triton.wire import (  # noqa: E402
    RequestDecodeError,
    decode_request,
    encode_error,
    encode_response,
)

OUTPUT = "RESPONSE_JSON"
INPUT = "REQUEST_JSON"


class TritonPythonModel:
    def initialize(self, args: dict) -> None:
        """Load the engine once per model instance.

        Triton starts one stub process per `instance_group` count, each with its own copy
        of the weights, and calls this before any request is served. Loading here rather
        than lazily is what keeps the Gated DeltaNet kernel compilation (54.2 s cold,
        20.7 s warm, against ~67 ms steady state) off the first caller.
        """
        self.logger = pb_utils.Logger
        model_config = json.loads(args["model_config"])
        params = {k: v["string_value"]
                  for k, v in (model_config.get("parameters") or {}).items()}

        def param(name: str, default: str) -> str:
            # Environment wins over config.pbtxt, so a SageMaker deployment can override
            # without rebuilding the model repository into the image.
            return os.environ.get(name) or params.get(name) or default

        checkpoint = param("SD_CHECKPOINT", "/opt/strands-decider/checkpoint")
        merged = param("SD_MERGED_TORSO", "/opt/strands-decider/merged")
        engine_kind = param("SD_ENGINE", "merged").strip().lower()
        prefix_cache = param("SD_PREFIX_CACHE", "1") not in ("0", "false", "no", "off")
        device = param("SD_DEVICE", "cuda")
        max_batch = int(param("SD_MAX_BATCH", "32"))

        # How many question ROWS share one forward pass -- `BatchedSystemOneEngine.max_rows`.
        # Distinct from SD_MAX_BATCH, which chunks ONE request's questions.
        #
        # Exposed as a parameter because it is half of a pair that has to move together, and
        # the other half (`max_batch_size`) already lives in config.pbtxt. The rule, measured
        # twice from opposite directions:
        #
        #     max_batch_size x typical questions per request <= max_rows
        #
        # Overshoot does not fail, it CHUNKS -- so you pay the queueing delay to assemble a
        # wide batch and then split it anyway, buying no amortisation. That is exactly how
        # `max_batch_size: 32` against max_rows 128 measured 56 decisions/s against 101.5
        # (server p50 611 ms -> 2,031 ms): Triton formed batches of 26 requests = 182 rows.
        # Leaving max_rows unreachable from configuration is what made that a rebuild to fix.
        max_rows = int(param("SD_MAX_ROWS", "128"))
        if max_rows < 1:
            raise pb_utils.TritonModelException(
                f"SD_MAX_ROWS={max_rows} is not a usable row budget; it must be >= 1.")
        # Two independent, opt-in accelerators, both OFF by default because each changes
        # the arithmetic slightly and the eager reference path is what every published
        # number in this repository was measured on. They serve opposite regimes: graphs
        # cut single-request latency, fusion raises saturated throughput. See
        # README.md, "Known limits".

        # Fused kernels (strands_decider/fused_layers.py). `load_merged_engine` refuses
        # rather than half-fusing, so a bad value here fails `initialize()` and Triton
        # never reports ready -- the correct outcome, not an inconvenience.
        fuse_layers = param("SD_FUSE_LAYERS", "0") not in ("0", "false", "no", "off", "")

        # CUDA graph capture (strands_decider/cuda_graphs.py). Removes the ~45 ms per-pass
        # CPU dispatch floor on the one-pass route.
        #
        #   0    eager everywhere (the default)
        #   1    graph the one-pass route -- measured 45.9 ms -> 21.5 ms server p50 for a
        #        one-question request on an L4
        #   all  additionally graph the state/row pair, which measured 0.73x-1.02x on the
        #        same card and so is deliberately NOT included in `1`
        graph_mode = param("SD_CUDA_GRAPHS", "0").strip().lower()
        cuda_graphs = graph_mode not in ("0", "false", "no", "off")
        cuda_graphs_two_pass = graph_mode in ("all", "2", "two-pass", "two_pass")


        self._assert_fla_on_gpu(device)

        started = time.perf_counter()
        if engine_kind == "merged":
            from strands_decider.merged_engine import load_merged_engine

            # Optional kwargs are filtered against the REAL signature instead of passed
            # blind, because this file and the installed `strands_decider` version
            # independently. A SageMaker `ModelDataUrl` overlay untars over
            # /opt/ml/model, so THIS model.py runs in front of whatever package the image
            # was built with -- and the newest published image (v23-triton-onepass,
            # 2026-10-07) predates `fuse_layers`, `cuda_graphs` and `max_rows`.
            #
            # Passing an unknown kwarg there does not degrade, it kills the load:
            #     TypeError: load_merged_engine() got an unexpected keyword argument
            # `initialize()` raises, Triton never reports ready, /ping never passes, and
            # SageMaker fails the endpoint ~31 MINUTES later on the health check. That has
            # happened and it cost a deploy, which is the whole reason this guard exists.
            #
            # Silence is not the fallback, though -- see `_engine_kwargs`: an option that
            # was explicitly turned ON and cannot be honoured RAISES, because serving the
            # unfused torso after being asked for the fused one would make a benchmark
            # comparison a lie. Only defaults are allowed to be dropped, with a warning.
            optional = {
                "fuse_layers": (fuse_layers, False),
                "cuda_graphs": (cuda_graphs, False),
                "cuda_graphs_two_pass": (cuda_graphs_two_pass, False),
                "max_rows": (max_rows, 128),
            }
            extra = self._engine_kwargs(load_merged_engine, optional)

            self.engine = load_merged_engine(
                checkpoint, merged, device=device,
                use_prefix_cache=prefix_cache, max_batch=max_batch,
                model_name="strands-decider-triton",
                **extra,
            )
        elif engine_kind == "hf":
            # NOTE: the shipped image does NOT carry `peft`, by design -- the LoRA is
            # folded in at build time and shipping peft would invite someone loading an
            # unmerged checkpoint at runtime. So this arm only works in an image built
            # with peft present. It exists for A/B against the merged torso, and it fails
            # with that explanation rather than a bare ImportError from three frames down.
            if fuse_layers:
                # Refused rather than ignored: silently serving the unfused torso after
                # being asked for the fused one would make a benchmark comparison a lie,
                # and the A/B between the two is the whole reason SD_ENGINE=hf exists.
                raise pb_utils.TritonModelException(
                    "SD_FUSE_LAYERS=1 is not supported with SD_ENGINE=hf: the fused "
                    "kernels replace the torso's layers in place and PEFT wraps them, so "
                    "the adapter would be bypassed. Use SD_ENGINE=merged."
                )
            try:
                from strands_decider.infer import load_engine
            except ImportError as exc:
                raise pb_utils.TritonModelException(
                    f"SD_ENGINE=hf needs peft, which this image does not ship ({exc}). "
                    "The LoRA is merged at build time; use SD_ENGINE=merged. To A/B the "
                    "two, build an image with peft in the runtime stage."
                ) from exc
            self.engine = load_engine(checkpoint, device=device,
                                      use_prefix_cache=prefix_cache)
        else:
            # Not a fallback: an unknown engine here would serve a different forward pass
            # than the one configured, which is the exact bug class this project fixed in
            # the SageMaker shim.
            raise pb_utils.TritonModelException(
                f"SD_ENGINE={engine_kind!r} is not supported by this backend. Use "
                "'merged' or 'hf'. vLLM batches internally and should be served by "
                "serving/vllm_server.py, not behind Triton's dynamic batcher."
            )
        self.logger.log_info(
            f"[decider] {engine_kind} engine loaded in {time.perf_counter() - started:.1f}s "
            f"(prefix_cache={prefix_cache}, device={device}, fused_layers={fuse_layers})"
        )
        if param("SD_WARMUP", "1") not in ("0", "false", "no", "off"):
            self._warmup()

    def _engine_kwargs(self, loader, optional: dict) -> dict:
        """Keep only the optional kwargs `loader` actually accepts.

        `optional` maps a kwarg name to `(requested value, default value)`.

        The split is deliberate and is the whole point:

          * requested == default  -> DROP it if unsupported. Nothing was asked for, so an
            older engine that cannot do it is already doing what the caller wanted.
          * requested != default  -> RAISE if unsupported. An operator who set
            `SD_FUSE_LAYERS=1` or `SD_MAX_ROWS=256` and silently got neither would read the
            resulting throughput number as evidence about a configuration that was never
            served. Failing the load says so in the Triton log in seconds instead.

        Signature introspection rather than try/except TypeError, because that exception is
        indistinguishable from the same error raised *inside* a supported code path, and
        retrying a partially-constructed 4.5 GB engine load to find out is not free.
        """
        import inspect

        accepted = inspect.signature(loader).parameters
        keep, dropped, refused = {}, [], []
        for name, (value, default) in optional.items():
            if name in accepted:
                keep[name] = value
            elif value == default:
                dropped.append(name)
            else:
                refused.append(f"{name}={value!r}")

        if refused:
            raise pb_utils.TritonModelException(
                f"{loader.__name__}() in the installed strands_decider does not accept "
                f"{', '.join(refused)}. This model repository is NEWER than the package in "
                "the image -- the usual cause is a SageMaker ModelDataUrl overlay in front "
                "of an older container. Rebuild the image from this commit, or unset the "
                "option. Refusing rather than serving a configuration you did not ask for."
            )
        if dropped:
            self.logger.log_warn(
                f"[decider] installed strands_decider does not accept {sorted(dropped)}; "
                "left at its built-in default (each was already off/default here, so "
                "nothing was silently changed)."
            )
        return keep

    def _warmup(self) -> None:
        """Compile every shape a caller might hit, before Triton reports the model ready.

        Not optional, and not one request. The Gated DeltaNet layers' Triton/fla kernels
        compile per shape, and the two readout paths are *different code*: one question
        takes the plain batched path, several take the shared-prefix path. Measured on this
        hardware, the first forward costs 54.2 s on a cold kernel cache and 20.7 s on a
        warm one, against ~67 ms steady state -- so any shape missed here is paid for by
        whichever caller reaches it first.

        Mirrors `strands_decider.scheduler.Scheduler.warmup`'s shape list, deliberately:
        three state lengths across both readout paths. `initialize()` blocking on this is
        what makes Triton's readiness honest.
        """
        from strands_decider.schema import (
            ChoiceQuestion,
            NoulQuestion,
            ScoreQuestion,
            SystemOneRequest,
        )

        noul = NoulQuestion(instructions="Is this a warm-up request?")
        choice = ChoiceQuestion(instructions="Which path is this?",
                                criteria={"warmup": "a warm-up pass",
                                          "real": "real traffic"})
        score = ScoreQuestion(instructions="How warm is the engine?",
                              criteria=["cold", "warming", "warm"])
        shapes = [
            {"only": noul},                        # 1 question -> batched path
            {"a": noul, "b": choice, "c": score},  # N questions -> shared-prefix path
        ]
        started = time.perf_counter()
        for approx_tokens in (16, 256, 2048):
            # ~1.3 tokens a word, so this only has to land in the right shape bucket.
            state = ("warm up the kernels for this state length. "
                     * max(1, approx_tokens // 10))
            for questions in shapes:
                try:
                    self.engine.evaluate(
                        SystemOneRequest(state=state, questions=questions))
                except Exception as exc:
                    # A warm-up failure is worth shouting about but not worth refusing to
                    # start over: the shape may simply not fit the window, and the server
                    # is still able to answer everything else.
                    self.logger.log_warn(
                        f"[decider] warm-up failed at ~{approx_tokens} tokens, "
                        f"{len(questions)} question(s): {exc}")
        self.logger.log_info(
            f"[decider] warm-up covered {3 * len(shapes)} shapes in "
            f"{time.perf_counter() - started:.1f}s")
        self._warmup_graphs(noul, choice, score)

    def _warmup_graphs(self, noul, choice, score) -> None:
        """Drive `evaluate_many` so the CUDA-graph buckets are captured before readiness.

        The sweep above goes through `evaluate`, which never touches `evaluate_many` and so
        never reaches a graph. Capturing costs ~0.4 s per shape, and a bucket is only
        captured after it has run eagerly `HOT_BUCKET` times -- so each shape is driven a
        few times here and then captured, rather than leaving the first real caller of each
        shape to pay for it. No-op when graphs are off: `capture_pending` returns 0.
        """
        from strands_decider.schema import SystemOneRequest

        engine = self.engine
        if not hasattr(engine, "capture_pending"):
            return
        started = time.perf_counter()
        # Both routes, across the state lengths the envelope covers: few questions take the
        # one-pass (combined) graph, several take the state+rows pair. 7 is the shape the
        # published latency numbers use.
        plans = [(16, 1), (16, 3), (256, 1), (256, 3), (256, 7), (600, 7), (900, 7)]
        questions = {"a": noul, "b": choice, "c": score,
                     "d": noul, "e": choice, "f": score, "g": noul}
        names = list(questions)
        for approx_tokens, n in plans:
            state = "warm up the kernels for this state length. " * max(
                1, approx_tokens // 10)
            request = SystemOneRequest(
                state=state, questions={k: questions[k] for k in names[:n]})
            for _ in range(3):  # > HOT_BUCKET, so the bucket is hot enough to capture
                try:
                    engine.evaluate_many([request])
                except Exception as exc:
                    self.logger.log_warn(
                        f"[decider] graph warm-up failed at ~{approx_tokens} tokens, "
                        f"{n} question(s): {exc}")
                    break
            engine.capture_pending()
        stats = getattr(engine, "graph_stats", dict)()
        if stats:
            self.logger.log_info(
                f"[decider] cuda graphs after warm-up: {stats} in "
                f"{time.perf_counter() - started:.1f}s")

    def _assert_fla_on_gpu(self, device: str) -> None:
        """Refuse to start if the Gated DeltaNet layers are on the CPU reference path.

        This is the project's nastiest known failure mode and it is silent. Without a C
        compiler at *runtime*, Triton (the GPU kernel compiler) cannot build its CUDA
        driver shim; `flash-linear-attention` **catches** that, warns, and returns 'cpu'.
        The server then starts, reports healthy, and answers *correctly* on the GPU with
        all 18 DeltaNet layers on the slow reference path. `serving/README.md` records that
        the first build of the HF image shipped in exactly that state and passed the whole
        smoke test.

        A Triton python backend is especially exposed: it runs under the container's own
        interpreter, so whether `gcc` and `python3-dev` are reachable is a property of the
        base image rather than of anything this repository controls.
        """
        if device != "cuda":
            return
        try:
            from fla.utils import _device as fla_device

            platform = getattr(fla_device, "device_platform", None)
        except Exception as exc:
            raise pb_utils.TritonModelException(
                f"could not import flash-linear-attention to verify its backend: {exc}. "
                "The Gated DeltaNet layers need it on CUDA."
            ) from exc

        if platform != "cuda":
            raise pb_utils.TritonModelException(
                f"flash-linear-attention reports device_platform={platform!r}, not 'cuda'. "
                "The Gated DeltaNet layers would run on the CPU reference path: answers "
                "stay correct but ~1.3x slower, and nothing else would report it. Usual "
                "cause is no C compiler in the image -- Triton builds its CUDA driver shim "
                "at runtime and needs gcc, libc6-dev and python3-dev."
            )

    # How many requests between CUDA-graph stats lines. Operationally this is the only way
    # to tell a server that is replaying graphs from one that quietly fell back to the
    # eager path on every shape -- the answers are identical either way, so nothing else
    # would report it, and "it got slower at some point" is a terrible bug report.
    GRAPH_STATS_EVERY = 200

    def _log_graph_stats(self, n: int) -> None:
        stats = getattr(self.engine, "graph_stats", dict)()
        if not stats:
            return
        self._seen = getattr(self, "_seen", 0) + n
        if self._seen >= getattr(self, "_next_stats", 0):
            self._next_stats = self._seen + self.GRAPH_STATS_EVERY
            self.logger.log_info(f"[decider] cuda graphs after {self._seen} "
                                 f"request(s): {stats}")

    def _group_by_state(self, parsed: list) -> dict:
        """Group a Triton batch by rendered state, preserving each request's slot.

        This is where the per-request path's batching pays. The expensive part of this model
        is encoding the *state*: the shared-prefix path encodes it once and forks its KV
        cache across the questions, so `state + N x question` tokens instead of
        `N x (state + question)`. Two requests that share a state can therefore be answered
        in one engine call for roughly the price of one, and agent traffic does share states
        -- a guardrail panel asks several questions about the same payload.

        Requests with distinct states still cost one call each. Triton's batching does not
        make them one forward pass, and this backend does not pretend otherwise: merging
        unrelated states into one padded batch would re-encode each of them anyway, which is
        the same reason `scheduler.py` never did it.

        The key is the RENDERED state, not the raw value, and that is a correctness fix
        rather than tidiness. Keying on `json.dumps(state)` collided a string state with a
        structurally equal dict one -- the string `'{"a": 1}'` against the object
        `{"a": 1}` -- which render DIFFERENTLY: `render_content` uses a string as-is and
        re-emits a dict with `indent=2`. `_merge_same_state` then keeps only the first
        member's state, so the other caller would receive answers computed against a prompt
        it never sent, at HTTP 200. Grouping on the rendered text makes "same group" mean
        "same prompt prefix", which is exactly what the merge requires.
        """
        from strands_decider.prompting import render_state

        groups: dict[str, list[int]] = {}
        for i, req in enumerate(parsed):
            if req is None:
                continue
            groups.setdefault(render_state(req.state), []).append(i)
        return groups

    def execute(self, requests: list) -> list:
        """One Triton batch -> one response per request, in the same order.

        Triton requires exactly one response per request, in order, so every path here
        produces one -- including the error paths. A missing response hangs the client.
        """
        started = time.perf_counter()
        n = len(requests)
        responses: list = [None] * n
        parsed: list = [None] * n
        self._log_graph_stats(n)

        # ---- decode. A bad payload is the caller's error and must not fail its
        # neighbours in the batch, so it is answered immediately and skipped below.
        for i, request in enumerate(requests):
            try:
                tensor = pb_utils.get_input_tensor_by_name(request, INPUT)
                if tensor is None:
                    raise RequestDecodeError(f"missing input tensor {INPUT!r}")
                raw = tensor.as_numpy().reshape(-1)[0]
                parsed[i] = decode_request(raw)
            except RequestDecodeError as exc:
                responses[i] = self._respond(encode_error(str(exc)))
            except Exception as exc:  # malformed tensor, not a model failure
                responses[i] = self._respond(
                    encode_error(f"could not read request: {exc}"))

        # ---- evaluate. ONE engine call for the whole batch when the engine can do it.
        #
        # This loop used to run one `evaluate` per distinct state, which meant Triton's
        # dynamic batcher coalesced the requests and then we walked them one at a time.
        # Measured on an L4: a forward pass costs max(45 ms, tokens x 0.0935 ms), and the
        # 45 ms floor is CPU dispatch (~5,676 kernel launches), not arithmetic. So 8
        # requests cost 16 passes and ~790 ms where two passes over the same token count
        # cost ~434 ms. `evaluate_many` does the latter: all distinct states in one pass,
        # then every question suffix against its own state's forked cache.
        #
        # It also subsumes `_merge_same_state` -- the engine de-duplicates identical
        # states itself, by token ids rather than by rendered text.
        live = [i for i in range(n) if parsed[i] is not None and responses[i] is None]
        batched = getattr(self.engine, "evaluate_many", None)
        if live and batched is not None:
            try:
                outs = batched([parsed[i] for i in live])
                for i, out in zip(live, outs, strict=True):
                    if isinstance(out, ValueError):
                        # Caller errors: option count over num_slots, a prompt truncated
                        # through its option list, images on a text-only engine. Same
                        # mapping the FastAPI server gives 422.
                        responses[i] = self._respond(encode_error(str(out)))
                    elif isinstance(out, Exception):
                        self.logger.log_error(f"[decider] request failure: {out}")
                        responses[i] = self._respond(
                            encode_error(f"inference failed: {out}", kind="internal"))
                    else:
                        responses[i] = self._respond(
                            encode_response(out, (time.perf_counter() - started) * 1000))
            except Exception as exc:
                # A failure of the batched path itself, not of one request. Fall through
                # to the per-request loop rather than failing every caller at once: the
                # single-request path shares `_fit`, the readout and the head, so it is
                # the same answers by a slower route.
                self.logger.log_error(
                    f"[decider] batched evaluate failed, falling back to per-request: "
                    f"{exc}\n{traceback.format_exc()}")

        # ---- the PER-REQUEST FALLBACK path. See "Two paths through execute()" at the top
        # of this module: it serves `SD_ENGINE=hf` (a plain `SystemOneEngine`, which has no
        # `evaluate_many`), and it catches a batched path that raised.
        #
        # Note what makes it a fallback, because it is not an `else`: this loop ALWAYS runs,
        # and the `continue` below is the only thing that makes it a no-op after a
        # successful batched pass. That shape is deliberate -- it means a batched call that
        # answered only *some* of the batch still has the rest picked up here, rather than
        # the gap being filled by the "no response produced" guard at the end. The cost of
        # the shape is that removing the `continue` as "dead" would silently re-run every
        # request a second time, at roughly 45 ms a pass.
        for _state, idxs in self._group_by_state(parsed).items():
            if all(responses[i] is not None for i in idxs):
                continue
            try:
                if len(idxs) == 1:
                    i = idxs[0]
                    out = self.engine.evaluate(parsed[i])
                    responses[i] = self._respond(
                        encode_response(out, (time.perf_counter() - started) * 1000))
                    continue
                merged, mapping = self._merge_same_state(parsed, idxs)
                out = self.engine.evaluate(merged)
                self._split_response(out, mapping, idxs, responses, started)
            except ValueError as exc:
                for i in idxs:
                    if responses[i] is None:
                        responses[i] = self._respond(encode_error(str(exc)))
            except Exception as exc:
                self.logger.log_error(
                    f"[decider] engine failure: {exc}\n{traceback.format_exc()}")
                for i in idxs:
                    if responses[i] is None:
                        responses[i] = self._respond(
                            encode_error(f"inference failed: {exc}", kind="internal"))

        # Nothing may be left unanswered.
        for i in range(n):
            if responses[i] is None:
                responses[i] = self._respond(
                    encode_error("no response produced", kind="internal"))
        return responses

    def _merge_same_state(self, parsed: list, idxs: list[int]):
        """Combine several same-state requests into one, and return the key mapping.

        Question names are caller-chosen, so two requests in one batch can both use
        "urgent" and the merged dict needs unique keys. The generated key is returned in an
        explicit `mapping` rather than being parsed back apart afterwards: a caller is
        free to name a question `__0__urgent`, and a prefix-stripping split would then hand
        that caller's answer to a different caller. Cross-request answer contamination at
        HTTP 200 is the worst failure this service can have, so the mapping is carried
        rather than inferred.

        Returns `(request, mapping)` where mapping is `[(slot, merged_key, original_key)]`.
        """
        from strands_decider.schema import SystemOneRequest

        questions: dict = {}
        mapping: list[tuple[int, str, str]] = []
        for i in idxs:
            for name, q in parsed[i].questions.items():
                key = f"q{len(questions)}"
                questions[key] = q
                mapping.append((i, key, name))
        return SystemOneRequest(state=parsed[idxs[0]].state, questions=questions), mapping

    def _split_response(self, out, mapping: list, idxs: list[int],
                        responses: list, started: float) -> None:
        """Undo `_merge_same_state` via the explicit mapping, never by parsing keys."""
        from strands_decider.schema import SystemOneResponse, Usage

        elapsed = (time.perf_counter() - started) * 1000
        per_slot: dict[int, dict] = {i: {} for i in idxs}
        for slot, merged_key, original in mapping:
            if merged_key in out.answers:
                per_slot[slot][original] = out.answers[merged_key]

        # Usage is per request, and the state was encoded ONCE for the whole group, so
        # charging every member for it would overcount badly. Divide it; the output token
        # count stays exact because it is one decision per question.
        share = max(1, len(idxs))
        for i in idxs:
            responses[i] = self._respond(encode_response(
                SystemOneResponse(
                    model=out.model,
                    answers=per_slot[i],
                    usage=Usage(
                        input_tokens=out.usage.input_tokens // share,
                        output_tokens=len(per_slot[i]),
                    ),
                ),
                elapsed,
            ))

    @staticmethod
    def _respond(body: bytes):
        return pb_utils.InferenceResponse(output_tensors=[
            pb_utils.Tensor(OUTPUT, np.array([body], dtype=object))
        ])

    def finalize(self) -> None:
        close = getattr(getattr(self, "engine", None), "close", None)
        if callable(close):
            close()
