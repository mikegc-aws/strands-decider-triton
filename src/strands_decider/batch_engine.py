"""Cross-request batching: `B states + sum(N questions)` in two forward passes.

The gap this closes
-------------------
`SystemOneEngine.evaluate` already does the right thing for **one** request. Its
shared-prefix path encodes the state once and forwards only the question suffixes against
a forked cache, so a request costs `state + N x question` tokens rather than
`N x (state + question)`. That is the model's defining serving optimisation and it is why a
longer state costs this deployable ~14% of throughput where duplicating it per question
costs ~75%.

The defect is one scope up. The Triton backend called `evaluate` once per request, in a
Python loop (`model.py`, "one engine call per distinct state"), so with B requests in
flight we paid the per-pass cost 2B times instead of twice. Measured on an L4, one forward
pass costs `max(45 ms, tokens x 0.0935 ms)` -- a ~45 ms floor that is CPU dispatch, not
arithmetic (~5,676 kernel launches per pass; streaming 3.8 GB of bf16 weights at the L4's
300 GB/s accounts for only 12.7 ms). So the floor dominates at realistic request sizes and
paying it 16 times rather than twice is the whole problem.

Predicted and then measured, 8 requests x 7 questions, ~90-token states:

    per-request (before)  16 passes, 4,640 tokens   ~824 ms   (measured 790)
    per-batch   (here)     2 passes, 4,640 tokens   ~434 ms

Same token count. The entire saving is passes.

Why left-padding the states is not a detail
-------------------------------------------
Pass 1 batches states of different lengths, so they have to be padded, and the padding
side changes the answers.

* **Right-padding is wrong here.** Real state occupies `[0, len_i)`, pads
  `[len_i, L_max)`, and the suffix continues at `L_max`. The distance from the suffix to
  the end of the real state is then `L_max - len_i`, which differs per row -- so RoPE sees
  the question as sitting a variable, fictitious distance after the state. Worse, a Gated
  DeltaNet layer is recurrent: it would consume the pad tokens *last*, corrupting the
  recurrent state at exactly the point pass 2 picks it up.
* **Left-padding is correct.** Every row's real state ends at `L_max - 1`, so the suffix
  continues from `L_max` for all rows and `cache_position` is uniform and right. Pads are
  consumed *before* the real tokens, so the recurrent state is still driven by the whole
  real state afterwards, and the attention mask marks them dead for the 6 attention layers.

`_pad` (right-padding) is still used for the question suffixes, unchanged -- that is what
the existing shared-prefix path already does, and `pool_last_token` finds the last real
token by mask.

This reasoning is why `tests/test_batch_engine.py` asserts **equivalence against the
per-request path** rather than merely that the shapes work: a padding or recurrence bug
here produces a confidently wrong probability at HTTP 200, not a crash. Mixed state
lengths in one batch is the case that detects it, so that is what the test uses.
"""

from __future__ import annotations

import copy
from typing import Any

import torch

from .infer import (
    _ROW_STATES,
    SystemOneEngine,
    UnforkableCache,
    _option_token_index,
    _to_answer,
)
from .modeling import (
    apply_temperature,
    gather_options,
    masked_log_softmax,
    pool_last_token,
)
from .prompting import RenderedQuestion, render_question, render_state
from .schema import Answer, SystemOneRequest, SystemOneResponse, Usage


def _gather_layered_cache(cache: Any, index: torch.Tensor) -> Any:
    """Select cache rows by index: row `r` of the result is row `index[r]` of `cache`.

    The gather counterpart of `infer._fork_layered_cache`, and deliberately the same
    shape of code: it walks `cache.layers`, touches only the tensors named in
    `_ROW_STATES` (`keys`/`values` for attention layers, `conv_states`/`recurrent_states`
    for Gated DeltaNet ones) and raises `UnforkableCache` for any other tensor, because a
    tensor we cannot name is a tensor whose batch dimension we cannot assume.

    `index_select` rather than `expand`: a fork repeats one state across N rows, this
    maps M distinct states onto N rows where several rows may share a state. It also
    returns new storage, which `expand` does not -- and new storage is required, not
    merely tidy, because a DeltaNet layer updates its states and `has_previous_state`
    flag in place during pass 2 and would otherwise corrupt the prefix it came from.
    """
    if not isinstance(getattr(cache, "layers", None), list):
        raise UnforkableCache(f"unsupported KV cache type: {type(cache)!r}")

    out = copy.copy(cache)
    out.layers = []
    for layer in cache.layers:
        nl = copy.copy(layer)
        for name, v in list(vars(nl).items()):
            is_state = name in _ROW_STATES
            if isinstance(v, torch.Tensor):
                if not is_state:
                    raise UnforkableCache(
                        f"cache layer {type(layer).__name__} holds tensor {name!r}")
                if v.numel():
                    setattr(nl, name, v.index_select(0, index).contiguous())
            elif isinstance(v, dict):
                if any(isinstance(t, torch.Tensor) for t in v.values()) and not is_state:
                    raise UnforkableCache(
                        f"cache layer {type(layer).__name__} holds tensors in {name!r}")
                setattr(nl, name, {
                    k: (t.index_select(0, index).contiguous()
                        if isinstance(t, torch.Tensor) and t.numel() else t)
                    for k, t in v.items()
                })
        out.layers.append(nl)
    return out


class _Prepared:
    """One request, rendered and tokenised, ready to become rows in a shared batch."""

    __slots__ = ("error", "names", "offsets", "q_ids", "rendered", "state_ids")

    def __init__(self, names: list[str], rendered: list[RenderedQuestion],
                 state_ids: list[int], q_ids: list[list[int]],
                 offsets: list[list[tuple[int, int]]]) -> None:
        self.names = names
        self.rendered = rendered
        self.state_ids = state_ids
        self.q_ids = q_ids
        self.offsets = offsets
        self.error: Exception | None = None


class BatchedSystemOneEngine(SystemOneEngine):
    """A `SystemOneEngine` that can also evaluate many requests in two passes.

    `evaluate` is inherited untouched, so every single-request caller, test and the
    FastAPI path behave exactly as before. `evaluate_many` is the new entry point and the
    only thing the Triton backend needs to call.

    `max_rows` caps how many question rows share one pass. It is separate from
    `cfg.max_batch` (which chunks the questions of a single request and whose value is
    tuned for that) because the useful limit here is activation memory for the whole
    in-flight batch, not one request's question count. 128 rows of short suffixes is a
    few hundred MB on a 24 GB card; lower it if a long-question workload runs tight.
    """

    # Above how many DUPLICATED state tokens the two-pass route starts to pay.
    #
    # The single-pass route forwards `state + question` per row, so a state shared by N
    # rows is encoded N times; the two-pass route encodes it once and spends an extra
    # forward pass instead. One pass costs ~45 ms of fixed CPU dispatch on an L4 and
    # tokens cost ~0.0935 ms, so the extra pass is worth it once duplication exceeds
    # 45 / 0.0935 ~= 480 tokens.
    #
    # Calibrated against four measurements on an L4 (single request, warm), and it
    # predicts all four:
    #
    #   questions  dup tokens  1 pass    2 passes   rule says   measured winner
    #   1              0        45.6 ms    94 ms     1 pass      1 pass  (2.1x)
    #   3            180        48.6 ms    93 ms     1 pass      1 pass  (1.9x)
    #   7            540        97.9 ms   101 ms     2 passes    ~tie
    #   14          1170       190.9 ms   158 ms     2 passes    2 passes (1.2x)
    #
    # This guard is why single-question latency is ~45 ms and not ~94. `evaluate` has
    # always had the equivalent check ("One question gains nothing from a shared prefix
    # and pays a second forward"); an earlier version of this class dropped it and
    # regressed one-question calls by 2.1x while the throughput numbers all still
    # improved -- the batch cases never exercise it.
    DUP_TOKEN_BUDGET = 480

    def __init__(self, *args: Any, max_rows: int = 128,
                 dup_token_budget: int | None = None,
                 cuda_graphs: bool = False, cuda_graphs_two_pass: bool = False,
                 **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.max_rows = max_rows
        self.dup_token_budget = (self.DUP_TOKEN_BUDGET if dup_token_budget is None
                                 else dup_token_budget)
        # Opt-in, and off by default: the graph path is new, and the eager path is the one
        # every published number was measured on. `SD_CUDA_GRAPHS=1` turns it on.
        self.want_cuda_graphs = bool(cuda_graphs)
        # The two-pass route is gated SEPARATELY and stays off even at
        # `SD_CUDA_GRAPHS=1`, because it was measured and it does not pay on an L4.
        #
        # Interleaved A/B in one process -- the two routes alternated per iteration over
        # one copy of the weights, because the measuring box was shared and a run that
        # timed all of one route and then all of the other attributed a neighbour's load to
        # whichever half it landed in. One request, seven questions, median of 9:
        #
        #   state tokens   145    335    505    725    945
        #   eager/graph   0.86x  1.02x  0.84x  0.78x  0.73x
        #
        # The one-pass route over the same model goes the other way -- 2.06x at one
        # question over a 75-token state -- so this is not capture being slow, it is the
        # shape of the two-pass route. The state pass has to carry an explicit `Sb x Sb`
        # mask (a graph cannot branch on "this batch needs no mask"), which puts attention
        # on the masked SDPA path instead of the causal flash one, and that costs more than
        # the kernel launches it saves as soon as a state passes a few hundred tokens. The
        # eager path gets the fast kernel for free whenever a batch's states happen to be
        # the same length.
        #
        # It is kept, correct and parity-gated rather than deleted, because it is the part
        # worth building on: see README "Known limits" for the variant to try next (eager
        # state pass, graphed row pass). `SD_CUDA_GRAPHS=all` turns it on.
        self.want_cuda_graphs_two_pass = bool(cuda_graphs_two_pass)
        self._graphs: Any = None
        self._graphs_tried = False

    # ---- CUDA graphs -----------------------------------------------------

    def graphs(self) -> Any:
        """The `CudaGraphs` helper, or None if graphs are off or unavailable here.

        Built on first use rather than in `__init__` so a server that never calls
        `evaluate_many` (the FastAPI path) allocates no buffers at all, and so the layout
        probe runs after the model is on its device. The footprint is 64 MiB for the
        one-pass route and 1,188 MiB with the state bank, measured on this model.
        """
        if self._graphs is not None or self._graphs_tried:
            return self._graphs
        self._graphs_tried = True
        if not (self.want_cuda_graphs or self.want_cuda_graphs_two_pass):
            return None
        try:
            from .cuda_graphs import CudaGraphs

            pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
            self._graphs = CudaGraphs(self.model.torso, pad_id,
                                      two_pass=self.want_cuda_graphs_two_pass)
            print(f"[strands-decider] cuda graphs enabled "
                  f"({self._graphs.bytes() / 2**20:.0f} MiB of buffers, "
                  f"two_pass={self.want_cuda_graphs_two_pass})")
        except Exception as exc:
            # Same instinct as `UnforkableCache` degrading to batched encoding: the eager
            # path answers identically, just slower, so an unavailable graph path must not
            # be a failed request.
            print(f"[strands-decider] cuda graphs unavailable ({type(exc).__name__}: "
                  f"{exc}); using the eager forward path")
            self._graphs = None
        return self._graphs

    def _disable_graphs(self, exc: BaseException) -> None:
        """Give up on graphs for the rest of this process, and say why.

        Deliberately permanent rather than per call. A pass that raised part-way through
        may have left the shared buffers half written, and while nothing reads them before
        the next pass refills them, a path that keeps failing and retrying would turn one
        bug into a latency cliff on every request.
        """
        import traceback

        print(f"[strands-decider] cuda graph pass failed, falling back to the eager "
              f"forward path for the rest of this process: {type(exc).__name__}: {exc}\n"
              f"{traceback.format_exc()}", flush=True)
        self._graphs = None
        self.want_cuda_graphs = False
        self.want_cuda_graphs_two_pass = False

    def capture_pending(self, limit: int | None = None) -> int:
        """Capture graphs for the shapes seen so far. Returns how many were captured.

        Exposed so warm-up can pay the ~0.4 s per capture before the server reports ready
        (`model.py::_warmup`), rather than leaving it to whichever caller arrives first.
        """
        graphs = self.graphs()
        if graphs is None:
            return 0
        with graphs.lock:
            return int(graphs.capture_pending(limit))

    def graph_stats(self) -> dict[str, int]:
        graphs = self.graphs()
        return graphs.stats() if graphs is not None else {}

    def _row_mask(self, lens: list[int], width: int) -> torch.Tensor:
        """`[rows, width]` 1/0 mask from real lengths, matching `_pad`'s right-padding.

        The graph path already knows every length exactly, but the readout finds the
        `<answer>` position through `pool_last_token(hidden, mask)` -- so it is handed the
        same mask the eager path would have built, and there is one pooling rule rather
        than two.
        """
        ar = torch.arange(width, device=self.device)
        return (ar[None, :] < torch.tensor(lens, device=self.device)[:, None]).long()

    def _readout(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
        row_rendered: list[RenderedQuestion],
        opt_idx: torch.Tensor | None,
    ) -> torch.Tensor:
        """Hidden states -> per-row probabilities. Extracted from `_rows_probs`, unchanged.

        It exists as a method because the CUDA-graph path produces `hidden` itself and so
        cannot go through `model.forward`, which would re-run the torso. Keeping it in one
        place is what stops the graph path and the eager two-pass path drifting apart --
        and `tools/batch_parity.py` gates that they have not.
        """
        pooled = pool_last_token(hidden, mask).to(torch.float32)
        n_slots = torch.tensor([rq.n_slots for rq in row_rendered], device=self.device)
        if self.model.config.head_type == "pointer":
            if opt_idx is None:
                raise ValueError("pointer head needs opt_idx (option token positions)")
            options = gather_options(hidden, opt_idx)
            raw = self.model.head(pooled, options.to(torch.float32))
        else:
            raw = self.model.head(pooled)
        logits = apply_temperature(
            raw, self._temperatures([rq.kind for rq in row_rendered]))
        return masked_log_softmax(logits, n_slots).exp()

    def _absolute_opt_idx(
        self,
        bases: list[int],
        row_rendered: list[RenderedQuestion],
        row_offsets: list[list[tuple[int, int]]],
    ) -> torch.Tensor | None:
        """Option token positions for the one-pass route: absolute, one base per row.

        `_option_idx` takes a single shared base, which is right for the two-pass route
        where every row's hidden states start at its suffix. Here each row carries its own
        state, so its options sit after **its** state length.
        """
        if self.model.config.head_type != "pointer":
            return None
        rows = [
            [base + i for i in _option_token_index(offs, rq.option_spans, 0)]
            for base, rq, offs in zip(bases, row_rendered, row_offsets, strict=True)
        ]
        width = max(len(r) for r in rows)
        return torch.tensor([r + [-1] * (width - len(r)) for r in rows],
                            dtype=torch.long, device=self.device)

    def _graph_combined(
        self,
        prompts: list[list[int]],
        row_rendered: list[RenderedQuestion],
        opt_idx: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int] | None:
        """The one-pass route through a captured graph, or None to run it eagerly."""
        graphs = self.graphs() if self.want_cuda_graphs else None
        if graphs is None:
            return None
        from .cuda_graphs import GraphsUnavailable, admits_combined

        lens = [len(p) for p in prompts]
        if not admits_combined(lens):
            return None
        try:
            with graphs.lock:
                hidden = graphs.combined(prompts)
                if graphs.capture_due():
                    graphs.capture_pending(limit=1)
            mask = self._row_mask(lens, hidden.size(1))
            return self._readout(hidden, mask, row_rendered, opt_idx), sum(lens)
        except GraphsUnavailable:
            # A shape that does not fit a buffer, not a broken graph path: decline this
            # call and let the eager route answer it. Keeping graphs on for the next
            # request is the point -- one awkward shape must not cost every later one.
            return None
        except Exception as exc:
            self._disable_graphs(exc)
            return None

    def _graph_two_pass(
        self,
        states: list[list[int]],
        row_state: list[int],
        row_q: list[list[int]],
        row_rendered: list[RenderedQuestion],
        row_offsets: list[list[tuple[int, int]]],
    ) -> tuple[torch.Tensor, int] | None:
        """The two-pass route through captured graphs, or None to run it eagerly.

        The state pass leaves state `i` in bank entry `i`, which is exactly what
        `row_state` already indexes -- so the cache gather `_gather_layered_cache` does by
        hand becomes an `index_select` inside the row graph.
        """
        graphs = self.graphs() if self.want_cuda_graphs_two_pass else None
        if graphs is None:
            return None
        from .cuda_graphs import GraphsUnavailable, admits_two_pass

        state_lens = [len(s) for s in states]
        row_lens = [len(q) for q in row_q]
        if not admits_two_pass(state_lens, row_lens):
            return None
        try:
            with graphs.lock:
                graphs.states(states)
                hidden = graphs.rows(row_q, row_state, state_lens)
                if graphs.capture_due():
                    graphs.capture_pending(limit=1)
            mask = self._row_mask(row_lens, hidden.size(1))
            probs = self._readout(
                hidden, mask, row_rendered,
                self._option_idx(row_rendered, 0, row_offsets)
                if self.model.config.head_type == "pointer" else None)
            return probs, sum(state_lens) + sum(row_lens)
        except GraphsUnavailable:
            return None   # see `_graph_combined`: decline the shape, keep the path
        except Exception as exc:
            self._disable_graphs(exc)
            return None

    # ---- preparation -----------------------------------------------------

    def _prepare(self, request: SystemOneRequest) -> _Prepared:
        if request.images:
            raise ValueError(
                "this engine has no vision tower; serve with --vision for images")
        names = list(request.questions.keys())
        rendered: list[RenderedQuestion] = []
        for name in names:
            rq = render_question(request.questions[name])
            if (self.model.config.head_type != "pointer"
                    and rq.n_slots > self.model.config.num_slots):
                raise ValueError(
                    f"question has {rq.n_slots} options but this model has "
                    f"{self.model.config.num_slots} slots; split the question or "
                    f"retrain with a larger num_slots")
            rendered.append(rq)
        # `_fit` gives the question first claim on the window and front-truncates it,
        # returning the offsets so a pointer readout survives truncation. Reused
        # unchanged -- the window policy must not differ between the one-request and
        # many-request paths, or the same payload would answer differently depending on
        # how busy the server happened to be.
        state_ids, q_ids, offsets = self._fit(
            render_state(request.state), [rq.text for rq in rendered])
        return _Prepared(names, rendered, state_ids, q_ids, offsets)

    # ---- the two passes --------------------------------------------------

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def _rows_probs(
        self,
        states: list[list[int]],
        row_state: list[int],
        row_q: list[list[int]],
        row_rendered: list[RenderedQuestion],
        row_offsets: list[list[tuple[int, int]]],
    ) -> tuple[torch.Tensor, int]:
        """Probabilities for every row, in two passes. Returns (probs, tokens forwarded).

        `states` are the DISTINCT state token sequences; `row_state[r]` indexes into them.
        De-duplication happens here rather than in the caller, so two requests that share
        a state (the same document asked two question sets) cost one state encode -- which
        is what `model.py::_merge_same_state` used to arrange by hand.
        """
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0

        # How many state tokens the single-pass route would re-encode. Zero when every
        # row has its own state (nothing is shared, so there is nothing to save).
        dup = sum(len(states[si]) for si in row_state) - sum(len(s) for s in states)
        if dup <= self.dup_token_budget:
            return self._rows_probs_one_pass(
                states, row_state, row_q, row_rendered, row_offsets)

        # ---- the graphed route, if it is available and these shapes fit it. Two passes
        # still, the same two, with the per-pass CPU dispatch replaced by one replay.
        graphed = self._graph_two_pass(states, row_state, row_q, row_rendered, row_offsets)
        if graphed is not None:
            return graphed

        # ---- pass 1: all distinct states, LEFT-padded (see module docstring)
        width = max(len(s) for s in states)
        state_ids = torch.tensor(
            [[pad_id] * (width - len(s)) + s for s in states], device=self.device)
        state_mask = torch.tensor(
            [[0] * (width - len(s)) + [1] * len(s) for s in states], device=self.device)
        prefix_out = self.model.torso(
            input_ids=state_ids,
            attention_mask=state_mask,
            use_cache=True,
            return_dict=True,
        )

        # ---- fan the per-state cache out to per-row
        index = torch.tensor(row_state, dtype=torch.long, device=self.device)
        cache = _gather_layered_cache(prefix_out.past_key_values, index)

        # ---- pass 2: every question suffix, right-padded, against its own state
        suffix_ids, suffix_mask = self._pad(row_q)
        full_mask = torch.cat([state_mask.index_select(0, index), suffix_mask], dim=1)
        hidden = self.model.encode(
            input_ids=suffix_ids,
            attention_mask=full_mask,
            past_key_values=cache,
        )

        # ---- readout. `hidden` is the suffix only, so option positions are
        # suffix-relative and take base=0: the cached state never enters the gather.
        probs = self._readout(
            hidden, full_mask, row_rendered,
            self._option_idx(row_rendered, 0, row_offsets)
            if self.model.config.head_type == "pointer" else None)

        n_tokens = int(state_mask.sum().item()) + int(suffix_mask.sum().item())
        return probs, n_tokens

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def _rows_probs_one_pass(
        self,
        states: list[list[int]],
        row_state: list[int],
        row_q: list[list[int]],
        row_rendered: list[RenderedQuestion],
        row_offsets: list[list[tuple[int, int]]],
    ) -> tuple[torch.Tensor, int]:
        """One forward pass over `state + question` per row. The model's native shape.

        Chosen when little or no state is shared (see `DUP_TOKEN_BUDGET`), where paying a
        second pass to avoid re-encoding a short state is a straight loss. Equivalent to
        `_slot_probs_batched`, generalised to rows whose states differ: option positions
        are absolute here, so each row's base is **its own** state length rather than one
        shared prefix length.
        """
        prompts = [states[si] + q for si, q in zip(row_state, row_q, strict=True)]
        bases = [len(states[si]) for si in row_state]
        opt_idx = self._absolute_opt_idx(bases, row_rendered, row_offsets)

        graphed = self._graph_combined(prompts, row_rendered, opt_idx)
        if graphed is not None:
            return graphed

        ids, mask = self._pad(prompts)
        n_slots = torch.tensor([rq.n_slots for rq in row_rendered], device=self.device)
        out = self.model(
            input_ids=ids,
            attention_mask=mask,
            n_slots=n_slots,
            temperature=self._temperatures([rq.kind for rq in row_rendered]),
            opt_idx=opt_idx,
        )
        return out["log_probs"].exp(), int(mask.sum().item())

    # ---- public ----------------------------------------------------------

    def evaluate_many(
        self, requests: list[SystemOneRequest]
    ) -> list[SystemOneResponse | Exception]:
        """Evaluate requests together. Result `i` is a response or the error for `i`.

        Errors are returned rather than raised so one malformed request cannot fail the
        batch around it. The Triton backend already maps `ValueError` to a caller error
        and anything else to a server error; returning them keeps that mapping, per
        request, which a single raise could not.
        """
        prepared: list[_Prepared | Exception] = []
        for r in requests:
            try:
                prepared.append(self._prepare(r))
            except Exception as exc:  # attributed to its own request, never the batch
                prepared.append(exc)

        # Flatten to rows, de-duplicating states by their token ids.
        states: list[list[int]] = []
        state_key: dict[tuple[int, ...], int] = {}
        rows: list[tuple[int, int]] = []  # (request index, question index)
        row_state: list[int] = []
        for ri, p in enumerate(prepared):
            if isinstance(p, Exception):
                continue
            key = tuple(p.state_ids)
            si = state_key.get(key)
            if si is None:
                si = len(states)
                state_key[key] = si
                states.append(p.state_ids)
            for qi in range(len(p.names)):
                rows.append((ri, qi))
                row_state.append(si)

        tokens_for: dict[int, int] = dict.fromkeys(range(len(requests)), 0)
        probs_for: dict[tuple[int, int], list[float]] = {}

        # Chunk by rows. A request's rows may straddle chunks: every row is independent
        # and answers are reassembled by name, so this is safe and keeps one long
        # request from forcing an oversized pass.
        for start in range(0, len(rows), self.max_rows):
            chunk = rows[start:start + self.max_rows]
            # Re-index the states this chunk actually touches, so a chunk never encodes
            # a state none of its rows uses.
            local: dict[int, int] = {}
            local_states: list[list[int]] = []
            local_row_state: list[int] = []
            for si in row_state[start:start + self.max_rows]:
                if si not in local:
                    local[si] = len(local_states)
                    local_states.append(states[si])
                local_row_state.append(local[si])

            row_rendered = [prepared[ri].rendered[qi] for ri, qi in chunk]  # type: ignore[union-attr]
            row_q = [prepared[ri].q_ids[qi] for ri, qi in chunk]  # type: ignore[union-attr]
            row_offsets = [prepared[ri].offsets[qi] for ri, qi in chunk]  # type: ignore[union-attr]

            probs, ntok = self._rows_probs(
                local_states, local_row_state, row_q, row_rendered, row_offsets)

            # Token accounting is split evenly across the chunk's requests. It is a
            # usage figure for a shared pass, so no per-request number is exactly
            # right; dividing keeps the batch total honest, which is the property that
            # matters for cost.
            touched = {ri for ri, _ in chunk}
            share, rem = divmod(ntok, len(touched))
            for k, ri in enumerate(sorted(touched)):
                tokens_for[ri] += share + (1 if k < rem else 0)

            for i, (ri, qi) in enumerate(chunk):
                rq = row_rendered[i]
                probs_for[(ri, qi)] = probs[i, : rq.n_slots].tolist()

        out: list[SystemOneResponse | Exception] = []
        for ri, p in enumerate(prepared):
            if isinstance(p, Exception):
                out.append(p)
                continue
            answers: dict[str, Answer] = {}
            try:
                for qi, name in enumerate(p.names):
                    answers[name] = _to_answer(
                        p.rendered[qi], probs_for[(ri, qi)],
                        ordinal_smoothing=self.model.config.ordinal_smoothing)
            except Exception as exc:  # attributed to its own request, never the batch
                out.append(exc)
                continue
            out.append(SystemOneResponse(
                model=self.cfg.model_name,
                answers=answers,
                usage=Usage(input_tokens=tokens_for[ri], output_tokens=len(p.names)),
            ))
        return out
