"""CUDA graph capture for the Decider's forward passes, against the hybrid Qwen3.5 torso.

Why this exists
---------------
One forward pass on an L4 costs `max(45 ms, tokens x 0.0935 ms)`. The 45 ms floor is not
arithmetic: it is the CPU issuing ~5,676 kernel launches and waiting, against a 12.7 ms
weight-streaming floor, with prefill at ~34% of the card's peak. A captured CUDA graph
issues the same kernels in one driver call, so it removes that floor.

Note what that does and does not promise, because the measurements below are blunt about
it. Removing a *fixed* cost helps a pass that is small enough for the fixed cost to
dominate, and does nothing for one the batcher has filled: `max(45, tokens x 0.0935)` is
only dispatch-bound while the left term wins, i.e. below about 480 tokens. So this is a
LATENCY lever at low concurrency, not a throughput lever at saturation.

Why README used to say this does not work -- and what changed
-------------------------------------------------------------
It said "the shapes that go fast return wrong values", and that was a true measurement of
**plain** capture. `/opt/probe/capture_probe.json` on the build box records it: every shape
captured, every shape reported `new_input_drift` of 1.0-2.0 against a fresh eager pass, and
every shape ALSO reported `called a synchronizing CUDA operation` in its steady state.

Those two findings are the same finding. transformers builds its attention masks inside the
forward, and that code reads device data back to the host and branches on it
(`_ignore_causal_mask_sdpa` is `torch.all(mask == 1)`). A host-side branch inside a capture
is not captured -- its *result* is baked in. The graph then replays one particular set of
sequence lengths forever, which is exactly the "fast but wrong" signature.

transformers 5 lets the caller pass `attention_mask` as a dict keyed by layer type, which
skips mask construction entirely. Hoisting every host-side decision out of the captured
region -- the masks, the cache's `has_previous_state` flags, the shapes -- makes replay
**bit-identical** to an eager pass over the same buffers. The go/no-go probe for this work
captured each of the three passes below, then changed the input buffers and compared a
replay against a fresh eager pass over the new inputs: `new-input drift 0.000e+00` for all
three. Not "within tolerance": zero.

So the old note was not wrong about what it tested. It was testing capture of a forward
that computes part of itself on the CPU.

The approach is `kev`'s (`kev/cuda_graphs.py`, Apache-2.0; see THIRD_PARTY_NOTICES.md),
ported to this engine. The borrowed ideas are the fixed-layout state bank, shape bucketing,
and the exact padding masks with the fully-masked-row trap handled explicitly.

What is graphed, and what is switched on
----------------------------------------
`BatchedSystemOneEngine` has two routes, and all three passes are implemented:

    combined  one pass over `state + question` per row, no cache   (_rows_probs_one_pass)
    states    every distinct state, LEFT-padded, into the bank     (_rows_probs, pass 1)
    rows      every question row against its state, from the bank  (_rows_probs, pass 2)

**`SD_CUDA_GRAPHS=1` enables `combined` only.** That is where the launch floor dominates.
`evaluate_many` timed in-process, routes alternated per iteration, median of 21 on an idle
L4 (IQR <= 1.6 ms):

    1 question,  75 state tokens   eager 40.9 ms   graph 19.9 ms   2.06x
    1 question, 145                      41.9           22.4       1.87x
    1 question, 245                      42.1           28.7       1.47x
    1 question, 335                      43.7           35.9       1.22x
    1 question, 725                      58.7           61.2       0.96x
    2 questions, 75                      44.0           27.9       1.58x

The floor is legible in the eager column: ~41 ms whatever the pass holds, up to a few
hundred tokens. The graph column is what the tokens actually cost. So this does not make
the GPU faster -- it removes a fixed CPU cost and leaves a variable GPU one, which is why
the win shrinks as the pass fills and is gone by 400-500 tokens.

**`SD_CUDA_GRAPHS=all` adds `states` + `rows`, and is not the default because it measured
SLOWER** -- 0.73x to 1.02x at seven questions, interleaved A/B in one process. The cause is
in the shape of that route rather than in capture: a graph cannot branch, so the state pass
must always carry an explicit `Sb x Sb` additive mask, which puts the 6 attention layers on
the masked SDPA path instead of the causal flash one. Past a few hundred state tokens that
costs more than the launches it saves, while the eager path gets the fast kernel for free
whenever a batch's states happen to be the same length. `BatchedSystemOneEngine.__init__`
carries the table.

It is kept rather than deleted because it is correct (`tools/batch_parity.py
--cuda-graphs --cuda-graphs-two-pass`, zero decision flips, 12 graphs captured, 14 replays)
and because it is the half of the port worth building on. The variant to try next, and the
reason it is not here:

    Keep the state pass EAGER -- it is compute-bound anyway, and it keeps the causal flash
    kernel -- and graph only the row pass. The row pass is ~45 ms of the 92 ms a
    one-request/seven-question call takes, so that is worth about 1.3x on exactly the case
    `combined` misses. What stops it is that a graphed row pass must read its states from
    FIXED addresses, which is what the bank is for, and the bank is the ~1.1 GB that made
    capture fail with CUDA OOM on a card shared with two other processes. The way out is
    visible but untried: move the per-row gather OUTSIDE the captured region and write
    straight into the row buffers before replaying. That is ~48 launches against the 5,676
    a pass costs, so hoisting it out gives up nothing measurable -- and it deletes the bank.

A graph has fixed shapes, so every pass is padded to a bucket and the padding is masked
exactly:

* **State passes are LEFT-padded.** 18 of the 24 layers are recurrent. Their padding mask
  zeroes the hidden states at the pads, so the Gated DeltaNet keys, values and queries are
  zero there and the recurrent state is still zero when the real tokens begin; the causal
  convolution sees the same zeros its own left padding would have supplied. The 6 attention
  layers mask the pads as keys. Right-padding would instead let the pads drive the
  recurrence *after* the real state, corrupting precisely the state the row pass picks up --
  the failure `batch_engine.py`'s docstring already describes, and it is silent.
* **Question rows are RIGHT-padded**, as the eager path already does. A row's pads come
  after its real tokens, so they can only corrupt a final recurrent state nobody reads, and
  under causal attention they cannot reach a real position.
* **A pad query attends to itself.** Not cosmetic. A fully-masked row of an additive mask is
  a row of `finfo.min`; with `-inf` it is NaN, and NaN survives being zeroed afterwards
  (NaN * 0 = NaN), so one unused pad row would poison the readout of the rows beside it.

Results therefore equal the eager passes up to floating-point reassociation (a different
chunking of the DeltaNet scan, other GEMM shapes), and a request's answers never depend on
what else shares its batch. `tools/batch_parity.py` is the gate: zero decision flips.

Capture cost and the fallbacks
------------------------------
Capturing costs ~0.4 s, so a bucket's first passes run **the same code, from the same
buffers, eagerly** and the bucket joins `pending`; `capture_pending` captures it once it has
been seen `HOT_BUCKET` times. Every other way this can fail degrades rather than breaks, in
the same spirit as `UnforkableCache` falling back to batched encoding:

    unsupported cache layout     -> GraphsUnavailable at construction, engine runs eager
    shape outside the envelope   -> that pass runs eager (see `admits_*`)
    a buffer too small for a shape -> that CALL runs eager, the path stays on
    capture raises (e.g. OOM)    -> no further captures at all; graphs already captured
                                    keep replaying, new shapes run eager (see
                                    `capture_pending` for why it is this blunt)
    anything else at run time    -> the engine disables graphs and re-runs the pass eager

Deliberately NOT done
---------------------
* The two-pass route is all-or-nothing: if the state pass is outside the envelope the row
  pass runs eager too, rather than loading an eagerly-computed cache into the bank. With a
  3,000-token state the state pass is ~280 ms of real compute and the row pass's ~45 ms
  floor is 14% of the request, so the extra machinery would buy little.
* No length grouping across state passes. At most `max_batch_size` (8) distinct states
  reach one call and all of them are under `GRAPH_STATE`, so one padded pass is enough.
"""

from __future__ import annotations

import functools
import threading
from collections import OrderedDict
from typing import Any, NamedTuple

import torch

# ---- the envelope ---------------------------------------------------------------------
#
# These bound GPU memory, not the model's context window (that is
# `StrandsDeciderConfig.max_length`, 3072). A pass outside them runs eagerly, which is the
# right answer rather than a regression: above about 1,000 tokens a pass stops being
# dispatch-bound and the padding plus the explicit mask would make the graph the slower
# route. Sized for one L4 (23 GB, ~5 GB of weights): 64 MiB for the one-pass route, and
# 1,188 MiB when the state bank is allocated too.
#
# Note what `admits_combined` and `admits_two_pass` do with these: a pass is graphed
# ALL-OR-NOTHING. One state over GRAPH_STATE in a Triton batch sends that whole batch to
# the eager path, rather than splitting it. That is a deliberate simplification -- the
# alternative is partitioning rows mid-route and merging two readouts of different widths,
# and the rows this would rescue are the long ones, which are compute-bound anyway.

GRAPH_STATES = 8      # distinct states one state pass may hold = bank entries. Triton
                      # coalesces at most `max_batch_size` (8) requests, so 8 covers a
                      # full batch even with no state shared between callers.
BANK_WIDTH = 3072     # positions per bank entry, states right-aligned. = max_length, so a
                      # state that fits the window fits the bank.
GRAPH_STATE = 1024    # longest state bucket graphed. Beyond it a state pass is
                      # compute-bound (1024 x 0.0935 ms ~ 96 ms against the 45 ms floor).
GRAPH_ROW = 256       # longest question-row bucket graphed. Questions are tens of tokens;
                      # `_fit` permits up to 0.75 x 3072, and those run eager.
GRAPH_COMBINED = 1024 # longest `state + question` bucket graphed on the one-pass route.
GRAPH_ROWS = 16       # question rows per graphed pass; more become several replays.
GRAPH_TOKENS = 32768  # rows x positions the row pass's attention buffers hold (~400 MB
                      # here: 6 attention layers x 2 tensors x 2 kv heads x 256 dim).
HIDDEN_TOKENS = GRAPH_ROWS * GRAPH_COMBINED   # rows x positions of hidden-state output
PASS_TOKENS = 256     # tokens a pass costs however few it holds: below this it is bound by
                      # reading the weights, not by compute. Used by `length_groups`.
GRAPHS_KEPT = 128     # captured graphs kept, least recently used evicted
HOT_BUCKET = 2        # eager passes after which a bucket's graph is captured


class GraphsUnavailable(RuntimeError):
    """This torso or device cannot be graphed; the caller runs the eager path.

    Deliberately the same shape of escape hatch as `infer.UnforkableCache`: a layout we
    cannot *name* is a layout whose buffers we cannot assume we own, and guessing would
    move probabilities silently.
    """


# ---- shape bucketing (pure; tests/test_cuda_graphs.py covers these on CPU) -------------


def bucket(n: int, steps: int = 8, floor: int = 16) -> int:
    """`n` rounded up to one of `steps` steps per power of two, in steps of at least `floor`.

    The padding adds less than `2 / steps` of `n` (just past a power of two, one step is
    almost that much), or fewer than `floor` tokens for small `n`. Coarse on purpose: a
    graph is per shape, so a bucket per token count would mean a capture per request.
    """
    if n <= 0:
        raise ValueError(f"bucket() needs a positive length, got {n}")
    step = max(floor, (1 << (n - 1).bit_length()) // steps)
    return -(-n // step) * step


def pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def count_bucket(n: int) -> int:
    """A row or state count rounded up to 1, 2, 3, 4, 6, 8, 12, 16, 24, 32.

    Empty entries padding a pass add less than half, and a traffic mix meets few distinct
    counts -- which is the property that keeps the number of captured graphs small.
    """
    return bucket(n, steps=4, floor=1)


def length_groups(lengths: list[int], cap: int) -> list[list[int]]:
    """Indices grouped into padded passes of at most `cap` items, computing the fewest tokens.

    A pass costs its *padded* size, `count_bucket(items) x bucket(longest)`, but at least
    `PASS_TOKENS`. Sorting by length first and then cutting is optimal among groupings of
    consecutive lengths (dynamic programming; ties keep the larger pass). This is what stops
    one 200-token question in a batch of 40-token ones from dragging every row into the
    200-token bucket.
    """
    order = sorted(range(len(lengths)), key=lengths.__getitem__)
    padded = [bucket(lengths[i]) for i in order]
    counts = [0] + [count_bucket(n) for n in range(1, cap + 1)]
    best: list[float] = [0.0] + [float("inf")] * len(order)
    cut = [0] * (len(order) + 1)
    for j in range(1, len(order) + 1):
        for i in range(max(0, j - cap), j):
            cost = best[i] + max(PASS_TOKENS, counts[j - i] * padded[j - 1])
            if cost < best[j]:
                best[j], cut[j] = cost, i
    groups: list[list[int]] = []
    j = len(order)
    while j:
        groups.append(order[cut[j]:j])
        j = cut[j]
    return groups[::-1]


# ---- the padding masks (pure; the CPU tests check them against a brute-force reference) --
#
# Every one of these ORs in `k == q` (the diagonal), which is not cosmetic. A pad row of an
# additive mask built from `-inf` softmaxes to NaN, and NaN survives being multiplied by
# zero afterwards, so one unused pad row would poison the rows beside it. Letting a pad
# query attend to itself means no row is ever fully masked. For a *real* query the diagonal
# is already allowed causally, so adding it changes nothing that is read.


def combined_allow(rowlen: torch.Tensor, Lb: int) -> torch.Tensor:
    """`[rows, Lb, Lb]`: causal over `state + question`, right-padded.

    `rowlen` is `[rows, 1]`. Pads sit after the real tokens, so under causal attention they
    cannot reach a real position; masking them as keys keeps them out of the real queries'
    denominators as well.
    """
    i = torch.arange(Lb, device=rowlen.device)
    q, k = i[None, :, None], i[None, None, :]
    return ((k <= q) & (k < rowlen[:, :, None])) | (k == q)


def state_allow(pad: torch.Tensor, Sb: int) -> torch.Tensor:
    """`[states, Sb, Sb]`: causal over a LEFT-padded state.

    `pad` is `[states, 1]`, how many leading slots of the bucket are padding. Real tokens
    occupy `[pad, Sb)`, so a key is valid exactly when `k >= pad`.
    """
    i = torch.arange(Sb, device=pad.device)
    q, k = i[None, :, None], i[None, None, :]
    return ((k <= q) & (k >= pad[:, :, None])) | (k == q)


def row_allow(plen: torch.Tensor, rowlen: torch.Tensor, Sr: int, Lb: int) -> torch.Tensor:
    """`[rows, Lb, Sr + Lb]`: a question row continuing its state out of the bank.

    Keys `[0, Sr)` are the bank window, in which the row's own state is RIGHT-aligned, so
    its real positions are `[Sr - plen, Sr)` and everything before them is another state's
    leftovers -- masked, which is what makes the shared bank safe. Keys `[Sr, Sr + Lb)` are
    the row's own tokens, causal and with the row's pads masked.
    """
    q = torch.arange(Lb, device=plen.device)[None, :, None]
    k = torch.arange(Sr + Lb, device=plen.device)[None, None, :]
    return (((k < Sr) & (k >= Sr - plen[:, :, None]))
            | ((k >= Sr) & (k - Sr <= q) & (k - Sr < rowlen[:, :, None]))
            | (k - Sr == q))


def admits_combined(prompt_lens: list[int]) -> bool:
    """Does the one-pass route's shape fit the graphed envelope?"""
    return bool(prompt_lens) and bucket(max(prompt_lens)) <= GRAPH_COMBINED


def admits_two_pass(state_lens: list[int], row_lens: list[int]) -> bool:
    """Does the two-pass route's shape fit? All-or-nothing, see the module docstring."""
    if not state_lens or not row_lens:
        return False
    return (len(state_lens) <= GRAPH_STATES
            and bucket(max(state_lens)) <= GRAPH_STATE
            and bucket(max(row_lens)) <= GRAPH_ROW
            and max(state_lens) <= BANK_WIDTH)


# ---- cache plumbing -------------------------------------------------------------------


def is_attention(layer: Any) -> bool:
    from transformers.cache_utils import DynamicLayer

    return isinstance(layer, DynamicLayer)


def layer_tensors(layer: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """The two tensors a cache layer holds: attention keys/values, or DeltaNet states."""
    if is_attention(layer):
        return layer.keys, layer.values
    return layer.conv_states[0], layer.recurrent_states[0]


@functools.cache
def _buffer_kv_class() -> type:
    """`BufferKV`, built lazily so this module imports without transformers.

    An attention cache layer over preallocated `[N, heads, T, dim]` views: the first
    `filled` positions hold the cached state and a pass writes its own keys and values
    *after* them instead of concatenating. No allocation and no Python state change, which
    is what lets one layer object serve the warm-up, the capture and every replay from the
    same addresses.

    Cached, so the class is defined once rather than per pass: `_cache` runs on every
    eager pass of a pending bucket, and a fresh class each time would make `isinstance`
    comparisons between two of them false for no reason.
    """
    from transformers.cache_utils import DynamicLayer

    class BufferKV(DynamicLayer):  # type: ignore[misc,valid-type]
        def __init__(self, keys: torch.Tensor, values: torch.Tensor, filled: int) -> None:
            super().__init__()
            self.buffers = (keys, values)
            self.filled = filled
            self.is_initialized = True
            self.keys, self.values = keys[..., :filled, :], values[..., :filled, :]

        def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
                   *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            end = self.filled + key_states.shape[-2]
            for buf, new in zip(self.buffers, (key_states, value_states), strict=True):
                buf[..., self.filled:end, :].copy_(new)
            return tuple(buf[..., :end, :] for buf in self.buffers)  # type: ignore[return-value]

    return BufferKV


def set_linear(layer: Any, conv: torch.Tensor, recurrent: torch.Tensor,
               previous: bool) -> None:
    """Point a Gated DeltaNet cache layer at these conv/recurrent tensors.

    `has_previous_state` is set explicitly rather than left to accumulate, and that is
    load-bearing: `LinearAttentionLayer.update_conv_state` flips it to True on the first
    call, so a warm-up pass would leave the *capture* taking the "concatenate the previous
    conv state" branch that the state pass must not take. Setting it per pass makes the
    captured graph match the pass it was captured for.
    """
    layer.conv_states[0], layer.recurrent_states[0] = conv, recurrent
    layer.is_conv_states_initialized[0] = True
    layer.is_recurrent_states_initialized[0] = True
    layer.conv_kernel_size[0] = conv.shape[-1]
    layer.has_previous_state[0] = previous
    layer.device, layer.dtype = conv.device, conv.dtype


class Buffers:
    """Per layer, two flat buffers and the per-entry shape they are viewed at.

    Flat and then viewed, rather than one tensor per shape, so that memory is fixed however
    many shapes turn up: `tokens` is a budget of rows x positions for the attention layers
    and `entries` a count for the DeltaNet states.

    `probe` is any object with a `.layers` list of cache layers -- a real `DynamicCache`, or
    a stub in the CPU tests.
    """

    def __init__(self, probe: Any, tokens: int, entries: int, device: Any) -> None:
        self.slots: list[tuple[bool, list[torch.Tensor], list[torch.Size]]] = []
        for layer in probe.layers:
            ts, attention = layer_tensors(layer), is_attention(layer)
            sizes = [tokens * t.shape[1] * t.shape[3] if attention else entries * t[0].numel()
                     for t in ts]
            self.slots.append((
                attention,
                [torch.zeros(n, dtype=t.dtype, device=device)
                 for n, t in zip(sizes, ts, strict=True)],
                [t.shape[1:] for t in ts],
            ))

    def views(self, n: int, length: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per layer its two buffers viewed for `n` entries: attention `[n, heads, length,
        dim]`, DeltaNet `[n, *state]`."""
        def view(flat: torch.Tensor, shape: torch.Size, attention: bool) -> torch.Tensor:
            want = (n, shape[0], length, shape[2]) if attention else (n, *shape)
            size = torch.Size(want)
            if size.numel() > flat.numel():
                raise GraphsUnavailable(
                    f"buffer of {flat.numel()} elements cannot hold {tuple(want)}")
            return flat[:size.numel()].view(size)
        return [tuple(view(f, s, attention) for f, s in zip(flats, shapes, strict=True))  # type: ignore[misc]
                for attention, flats, shapes in self.slots]

    def bytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for _attention, flats, _shapes in self.slots for t in flats)


class _Row(NamedTuple):
    """One question row of a graphed row pass."""

    ids: list[int]
    pos: list[int]
    entry: int        # which bank entry holds its state
    state_len: int    # how many real positions of that entry are its state


# ---- the graphs -----------------------------------------------------------------------


class CudaGraphs:
    """Captured graphs plus the fixed buffers they read and write.

    One instance owns every buffer, so passes must not overlap: `self.lock` is held for the
    whole of a pass and its copy-out. The Triton backend calls `execute()` sequentially on
    one stub thread and `Scheduler` keeps one thread on the device, so the lock is insurance
    rather than the mechanism -- but two threads sharing these buffers would answer one
    caller from another caller's hidden states, which is the worst failure this service has.
    """

    def __init__(self, torso: Any, pad_id: int, two_pass: bool = False) -> None:
        from transformers import DynamicCache
        from transformers.cache_utils import DynamicLayer, LinearAttentionLayer

        param = next(torso.parameters(), None)
        if param is None or param.device.type != "cuda":
            raise GraphsUnavailable("the torso is not on a CUDA device")
        if not torch.cuda.is_available():
            raise GraphsUnavailable("torch reports no CUDA device")

        self.torso = torso
        self.pad_id = int(pad_id)
        self.device, self.dtype = param.device, param.dtype
        self.hidden_size = int(torso.config.hidden_size)
        self.lock = threading.Lock()

        self.pool = torch.cuda.graph_pool_handle()
        self.stream = torch.cuda.Stream()
        self.graphs: OrderedDict[tuple, tuple[Any, torch.Tensor]] = OrderedDict()
        self.pending: dict[tuple, tuple[Any, torch.Tensor]] = {}
        self.eager_runs: dict[tuple, int] = {}
        self.failed: dict[tuple, tuple[str, torch.Tensor]] = {}
        self.capture_disabled = False
        self.captures = 0
        self.replays = 0

        # Learn the cache layout -- layer kinds, state shapes, dtypes -- from one tiny
        # eager pass rather than from the config, so a transformers change that moves a
        # state shape is caught here instead of corrupting a buffer.
        probe = DynamicCache(config=torso.config)
        with torch.inference_mode():
            torso(input_ids=torch.full((1, 16), self.pad_id, dtype=torch.long,
                                       device=self.device),
                  past_key_values=probe, use_cache=True)

        # `BufferKV` and `set_linear` rely on this layout: plain attention layers, and
        # single-state DeltaNet layers that update their states IN PLACE. `record_past`
        # assigns instead of copying, which would leave our buffers stale while every
        # shape still ran -- a confidently wrong probability at HTTP 200. Refuse it here.
        for layer in probe.layers:
            if type(layer) is DynamicLayer:
                continue
            if (type(layer) is LinearAttentionLayer
                    and layer.number_of_states == 1 and not layer.record_past):
                continue
            raise GraphsUnavailable(
                "CUDA graphs support attention and single-state in-place Gated DeltaNet "
                f"cache layers only; got {sorted({type(x).__name__ for x in probe.layers})}")

        # The state bank and the row buffers are ~1.1 GB and are needed ONLY by the
        # two-pass route, which is off by default. Allocating them anyway cost real
        # capture failures: on this 23 GB L4 with two other processes resident, capture of
        # a 640-token state pass went OOM with 12 MB free on the card. The one-pass route
        # needs the hidden-state buffer and nothing else, so that is all it gets.
        self.two_pass = bool(two_pass)
        self.bank = self.rowbuf = None
        self.bank_views: list[tuple[torch.Tensor, torch.Tensor]] = []
        if self.two_pass:
            self.bank = Buffers(probe, GRAPH_STATES * BANK_WIDTH, GRAPH_STATES,
                                self.device)
            # One bank layout for every pass, so the state and row graphs agree on it:
            # attention `[GRAPH_STATES, heads, BANK_WIDTH, dim]` with each state
            # right-aligned at the end, DeltaNet `[GRAPH_STATES, *state]`.
            self.bank_views = self.bank.views(GRAPH_STATES, BANK_WIDTH)
            self.rowbuf = Buffers(probe, GRAPH_TOKENS, GRAPH_ROWS, self.device)
        self.hidden = torch.zeros(HIDDEN_TOKENS * self.hidden_size,
                                  dtype=self.dtype, device=self.device)
        self.bank_filled = 0

    def bytes(self) -> int:
        extra = sum(b.bytes() for b in (self.bank, self.rowbuf) if b is not None)
        return extra + self.hidden.numel() * self.hidden.element_size()

    # ---- the pieces every pass shares -------------------------------------------------

    def _mask(self, allow: torch.Tensor) -> torch.Tensor:
        """Additive attention mask `[B, 1, Lq, Lk]` from a boolean `[B, Lq, Lk]`.

        `finfo.min` rather than `-inf`, matching `modeling.MASK_VALUE`'s reasoning: a row
        that ends up fully masked then softmaxes to uniform instead of to NaN.
        """
        return torch.zeros(allow.shape, dtype=self.dtype, device=self.device).masked_fill(
            ~allow, torch.finfo(self.dtype).min)[:, None]

    def _forward(self, ids: torch.Tensor, pos: torch.Tensor, full: torch.Tensor,
                 linear: torch.Tensor, cache: Any, use_cache: bool) -> torch.Tensor:
        """The torso, with both masks supplied.

        Passing `attention_mask` as a dict is the whole trick: transformers 5 then skips
        `create_causal_mask`/`create_recurrent_attention_mask`, which read device data back
        to the host and branch on it. See the module docstring for what that did to the
        earlier capture attempt.
        """
        return self.torso(
            input_ids=ids,
            position_ids=pos,
            attention_mask={"full_attention": full, "linear_attention": linear},
            past_key_values=cache,
            use_cache=use_cache,
            return_dict=True,
        ).last_hidden_state

    def _cache(self, views: list[tuple[torch.Tensor, torch.Tensor]], filled: int,
               previous: bool) -> Any:
        """A `DynamicCache` over buffer views, rebuilt per pass.

        Rebuilt rather than reused because the flags have to be set per pass, not
        accumulated -- see `set_linear`. It is Python-only work, so it costs nothing at
        replay (the graph replays kernels, not Python) and keeps the eager runs of a
        `pending` bucket doing exactly what the capture will do.
        """
        from transformers import DynamicCache

        buffer_kv = _buffer_kv_class()
        cache = DynamicCache(config=self.torso.config)
        for i, (layer, v) in enumerate(zip(cache.layers, views, strict=True)):
            if is_attention(layer):
                cache.layers[i] = buffer_kv(*v, filled)
            else:
                set_linear(layer, *v, previous)
        return cache

    def _replay(self, key: tuple, body: Any, rows: list[list[int]]) -> None:
        """Run `key`'s pass on `rows` (one int64 row per sequence): replay, or run eagerly.

        Every body for a key reads and writes the same buffer views, so the one kept in
        `pending` stands for all of them -- which is why a bucket can be captured later,
        from a pass it did not run.
        """
        if key in self.graphs:
            self.graphs.move_to_end(key)
            graph, buf = self.graphs[key]
        elif key in self.failed:
            graph, buf = None, self.failed[key][1]
        else:
            if key not in self.pending:
                self.pending[key] = (
                    body,
                    torch.zeros((len(rows), len(rows[0])), dtype=torch.long,
                                device=self.device),
                )
            graph, buf = None, self.pending[key][1]
            self.eager_runs[key] = self.eager_runs.get(key, 0) + 1
        # One H2D copy per pass carries ids, positions and every length the masks need, so
        # the graph has no host-side input left to bake in.
        buf.copy_(torch.tensor(rows, dtype=torch.long), non_blocking=True)
        if graph is not None:
            graph.replay()
            self.replays += 1
        else:
            body(buf)

    # ---- capture ----------------------------------------------------------------------

    def capture_due(self) -> bool:
        """Is a bucket hot enough to be worth ~0.4 s of capture?"""
        if self.capture_disabled:
            return False
        runs = list(self.eager_runs.values())
        return bool(runs) and max(runs) >= HOT_BUCKET

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def capture_pending(self, limit: int | None = None) -> int:
        """Capture graphs for up to `limit` buckets that have run eagerly, the hottest first.

        The caller must keep other passes out; `BatchedSystemOneEngine` holds `self.lock`.
        Each capture overwrites the shared buffers, which is fine *between* passes because a
        pass refills everything it reads.

        A warm-up pass runs first, on a side stream: Triton autotuning and cuBLAS workspace
        setup must not happen inside a capture. `capture_begin`/`capture_end` rather than
        `torch.cuda.graph`, whose `synchronize`, `gc.collect` and `empty_cache` on entry
        would stall a busy server once per capture. `capture_error_mode="thread_local"` so
        another thread's CUDA call (a request tokenising) cannot invalidate the capture.

        `inference_mode` and not `no_grad`, here and on every public method. The buffers are
        allocated on the engine's own path, which is already inside `inference_mode`, so they
        are *inference tensors* -- and writing to one from outside inference mode raises
        "Inplace update to inference tensor outside InferenceMode is not allowed". That is
        how this was found: warm-up called `capture_pending` from outside, and every row-pass
        capture failed and silently fell back to eager while the two passes captured from
        inside the engine succeeded. The failure was visible only in `stats()["failed"]`.
        """
        if self.capture_disabled:
            return 0
        done = 0
        count = len(self.pending) if limit is None else min(limit, len(self.pending))
        for _ in range(count):
            if not self.pending:
                break
            key = max(self.pending, key=lambda k: self.eager_runs.get(k, 0))
            body, buf = self.pending.pop(key)
            self.eager_runs.pop(key, None)
            current, graph = torch.cuda.current_stream(), torch.cuda.CUDAGraph()
            self.stream.wait_stream(current)
            try:
                with torch.cuda.stream(self.stream):
                    body(buf)
                    graph.capture_begin(pool=self.pool,
                                        capture_error_mode="thread_local")
                    try:
                        body(buf)
                    finally:
                        graph.capture_end()
            except Exception as exc:  # e.g. out of memory for a new shape
                # A failed `capture_end` leaves the allocator routing this thread's
                # allocations into the graph pool, which would fail every later capture AND
                # let ordinary eager passes allocate graph memory. End it explicitly.
                try:
                    # Private, hence the getattr default: there is no public way to undo a
                    # failed capture, and leaving it undone is worse than reaching in.
                    getattr(torch._C, "_cuda_endAllocateToPool",  # noqa: SLF001
                            lambda *a: None)(self.device.index, self.pool)
                except Exception:  # private API; best effort
                    pass
                self.failed[key] = (f"{type(exc).__name__}: {exc}", buf)
                # And stop capturing ANYTHING else, for good. Measured on the L4 with two
                # other processes on the card: the first capture failed with CUDA OOM (the
                # graph pool had grown to ~1.7 GB on top of 1.16 GB of buffers), and every
                # capture after it failed differently --
                #
                #   RuntimeError: it->second->use_count > 0 INTERNAL ASSERT FAILED at
                #   "/pytorch/c10/cuda/CUDACachingAllocator.cpp":2226
                #
                # -- so the recovery above is not enough to put the allocator back. A card
                # with no room for one graph has no room for the next one either, and
                # retrying turns a clean degradation into allocator asserts on every later
                # pass. The graphs already captured keep replaying; new shapes run eager.
                self.capture_disabled = True
                print(f"[strands-decider] cuda graph capture of {key} failed; no further "
                      f"shapes will be captured and they run eagerly: "
                      f"{self.failed[key][0]}", flush=True)
                break
            finally:
                current.wait_stream(self.stream)
            self.graphs[key] = (graph, buf)
            self.captures += 1
            done += 1
            while len(self.graphs) > GRAPHS_KEPT:
                self.graphs.popitem(last=False)
        return done

    def stats(self) -> dict[str, int]:
        return {"captured": self.captures, "kept": len(self.graphs),
                "pending": len(self.pending), "failed": len(self.failed),
                "replays": self.replays, "capture_off": int(self.capture_disabled)}

    # ---- pass 1 of the one-pass route: `state + question` per row, no cache -----------

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def combined(self, prompts: list[list[int]]) -> torch.Tensor:
        """One pass over each full prompt. Returns hidden states `[rows, max len, d]`.

        `use_cache=False` and no cache object, exactly as `_rows_probs_one_pass` runs it:
        with `cache_params=None` a Gated DeltaNet layer skips the conv-state update and
        feeds the causal convolution directly, so this is the same arithmetic rather than a
        cached impersonation of it.
        """
        width = max(map(len, prompts))
        # Zeros rather than `empty`: a row shorter than its group's longest leaves tail
        # positions unwritten, and while the readout never indexes past a row's own length,
        # a buffer that cannot hold a stale NaN is one less thing to have to prove.
        out = torch.zeros((len(prompts), width, self.hidden_size),
                          dtype=self.dtype, device=self.device)
        lens = [len(p) for p in prompts]
        for idx in length_groups(lens, GRAPH_ROWS):
            longest = max(lens[i] for i in idx)
            Lb = bucket(longest)
            group = max(1, min(GRAPH_ROWS, HIDDEN_TOKENS // Lb))
            group = max(n for n in range(1, group + 1) if count_bucket(n) <= group)
            for start in range(0, len(idx), group):
                part = idx[start:start + group]
                hidden = self._combined_pass([prompts[i] for i in part], Lb)
                take = max(lens[i] for i in part)
                out[torch.tensor(part, device=self.device), :take] = hidden[:, :take]
        return out

    def _combined_pass(self, prompts: list[list[int]], Lb: int) -> torch.Tensor:
        Nb = count_bucket(len(prompts))
        hidden = self.hidden[:Nb * Lb * self.hidden_size].view(Nb, Lb, self.hidden_size)

        def body(buf: torch.Tensor) -> None:
            rowlen = buf[:, 2 * Lb:2 * Lb + 1]
            i = torch.arange(Lb, device=self.device)
            hidden.copy_(self._forward(
                buf[:, :Lb], buf[:, Lb:2 * Lb], self._mask(combined_allow(rowlen, Lb)),
                (i[None] < rowlen).long(), None, False))

        pad = [self.pad_id] * Lb
        uploads = [list(p) + [self.pad_id] * (Lb - len(p))
                   + list(range(len(p))) + [0] * (Lb - len(p)) + [len(p)]
                   for p in prompts]
        uploads += [pad + [0] * Lb + [0]] * (Nb - len(prompts))
        self._replay(("combined", Nb, Lb), body, uploads)
        return hidden[:len(prompts)]

    # ---- pass 1 of the two-pass route: the states, into the bank ----------------------

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def states(self, states: list[list[int]]) -> None:
        """Encode every distinct state into bank entry `i`, LEFT-padded.

        Nothing is returned: the bank *is* the result, and `rows` reads it. One pass for all
        of them -- see the module docstring on why no length grouping here.
        """
        if self.bank is None:
            raise GraphsUnavailable(
                "the state bank was not allocated; this instance graphs the one-pass "
                "route only (construct with two_pass=True)")
        if len(states) > GRAPH_STATES:
            raise GraphsUnavailable(
                f"{len(states)} states exceeds GRAPH_STATES={GRAPH_STATES}")
        Sb = bucket(max(map(len, states)))
        Nb = count_bucket(len(states))

        def body(buf: torch.Tensor) -> None:
            # `pad` is how many leading slots of the bucket are padding, per state.
            pad = Sb - buf[:, 2 * Sb:]
            i = torch.arange(Sb, device=self.device)
            views = [(a[:Nb, :, BANK_WIDTH - Sb:], b[:Nb, :, BANK_WIDTH - Sb:])
                     if attention else (a[:Nb], b[:Nb])
                     for (attention, *_), (a, b) in zip(self.bank.slots, self.bank_views,
                                                        strict=True)]
            self._forward(buf[:, :Sb], buf[:, Sb:2 * Sb],
                          self._mask(state_allow(pad, Sb)),
                          (i[None] >= pad).long(), self._cache(views, 0, False), True)

        # Real tokens keep their own position ids; the pads are parked at 0, where the mask
        # makes them unreachable. RoPE here is relative, so what matters is that a question
        # row continues from `len(state)` -- which `rows` arranges.
        uploads = [[self.pad_id] * (Sb - len(s)) + list(s)
                   + [0] * (Sb - len(s)) + list(range(len(s))) + [len(s)]
                   for s in states]
        uploads += [[self.pad_id] * Sb + [0] * Sb + [0]] * (Nb - len(states))
        self.bank_filled = 0          # cleared first: a pass that raises must not leave the
        self._replay(("states", Nb, Sb), body, uploads)   # bank looking populated
        self.bank_filled = len(states)

    # ---- pass 2 of the two-pass route: the question rows, against the bank -----------

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def rows(self, row_q: list[list[int]], row_entry: list[int],
             state_lens: list[int]) -> torch.Tensor:
        """Every question suffix against its state's bank entry.

        Returns hidden states for the SUFFIX only, `[rows, max len, d]` -- the same shape
        and coordinates `_rows_probs` gets from `model.encode`, so the readout above is
        unchanged.

        Rows see the last `Sr` positions of the bank, `Sr` the longest state in the pass
        rounded up to a power of two. Rounding only lengthens attention (the extra slots are
        masked), and it keeps the number of distinct graphs down.
        """
        if self.rowbuf is None:
            raise GraphsUnavailable("the row buffers were not allocated; this instance "
                                    "graphs the one-pass route only")
        # A row pass reads bank entries the state pass wrote. Reading an entry nobody wrote
        # would answer a question against whatever the previous batch left there -- another
        # caller's document, at HTTP 200 -- so this is a guard and not an assertion of
        # tidiness.
        if row_entry and max(row_entry) >= self.bank_filled:
            raise GraphsUnavailable(
                f"row wants bank entry {max(row_entry)} but only {self.bank_filled} "
                "entries were written by the last state pass")
        lens = [len(q) for q in row_q]
        width = max(lens)
        out = torch.zeros((len(row_q), width, self.hidden_size),
                          dtype=self.dtype, device=self.device)
        for idx in length_groups(lens, GRAPH_ROWS):
            Lb = bucket(max(lens[i] for i in idx))
            Sr = pow2(max(16, max(state_lens[row_entry[i]] for i in idx)))
            group = min(GRAPH_ROWS, GRAPH_TOKENS // (Sr + Lb),
                        max(1, HIDDEN_TOKENS // Lb))
            if group < 1:
                raise GraphsUnavailable(
                    f"a row pass of Sr={Sr} + Lb={Lb} does not fit GRAPH_TOKENS")
            group = max(n for n in range(1, group + 1) if count_bucket(n) <= group)
            for start in range(0, len(idx), group):
                part = idx[start:start + group]
                rows = [_Row(row_q[i], list(range(state_lens[row_entry[i]],
                                                  state_lens[row_entry[i]] + lens[i])),
                             row_entry[i], state_lens[row_entry[i]])
                        for i in part]
                hidden = self._row_pass(rows, Sr, Lb)
                take = max(lens[i] for i in part)
                out[torch.tensor(part, device=self.device), :take] = hidden[:, :take]
        return out

    def _row_pass(self, rows: list[_Row], Sr: int, Lb: int) -> torch.Tensor:
        Nb, T = count_bucket(len(rows)), Sr + Lb
        hidden = self.hidden[:Nb * Lb * self.hidden_size].view(Nb, Lb, self.hidden_size)

        def body(buf: torch.Tensor) -> None:
            rowlen = buf[:, 2 * Lb:2 * Lb + 1]
            plen = buf[:, 2 * Lb + 1:2 * Lb + 2]
            entry = buf[:, 2 * Lb + 2]
            views = self.rowbuf.views(Nb, T)
            # Each row's state, copied out of the bank into this pass's own storage. New
            # storage is required rather than tidy: a DeltaNet layer updates its states in
            # place during this pass, so a row writing into the bank would corrupt the
            # state the rows beside it still need (the same reason
            # `_gather_layered_cache` uses `index_select` and not `expand`).
            for (attention, *_), dst, src in zip(self.bank.slots, views, self.bank_views,
                                                 strict=True):
                for d, s in zip(dst, src, strict=True):
                    if attention:
                        d[:, :, :Sr].copy_(
                            s[:, :, BANK_WIDTH - Sr:].index_select(0, entry))
                    else:
                        d.copy_(s.index_select(0, entry))
            linear = (torch.arange(Lb, device=self.device)[None] < rowlen).long()
            hidden.copy_(self._forward(
                buf[:, :Lb], buf[:, Lb:2 * Lb],
                self._mask(row_allow(plen, rowlen, Sr, Lb)),
                linear, self._cache(views, Sr, True), True))

        uploads = [list(r.ids) + [self.pad_id] * (Lb - len(r.ids))
                   + list(r.pos) + [0] * (Lb - len(r.pos))
                   + [len(r.ids), r.state_len, r.entry] for r in rows]
        uploads += [[self.pad_id] * Lb + [0] * Lb + [0, 0, 0]] * (Nb - len(rows))
        self._replay(("rows", Nb, Lb, Sr), body, uploads)
        return hidden[:len(rows)]
