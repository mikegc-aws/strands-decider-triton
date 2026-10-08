"""CUDA graphs, minus the CUDA: bucketing, buffer layout, the padding masks, and fallback.

A graph is fixed-shape, so every pass is padded to a bucket and the padding is masked. That
makes three things worth testing without a GPU, and one thing that cannot be:

Tested here. **The masks**, against a brute-force reference, because a mask that lets one
row see another row's state produces a confidently wrong probability at HTTP 200 rather
than a crash -- and the state bank means rows genuinely do share storage. **The bucketing
and grouping**, because a bucket that is too small silently drops tokens off a pass.
**The buffer layout**, because a view computed one stride wrong reads a neighbour's keys.
**The fallback**, because a graph path that fails loudly is fine and one that fails silently
is not.

NOT tested here: that a replay equals an eager pass. That needs the real 2B torso on a real
GPU and it is the whole point, so it has its own gate -- `tools/batch_parity.py
--cuda-graphs`, run on the box, gating on zero decision flips against the single-request
path. The module docstring of `cuda_graphs.py` records what that measured.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

torch = pytest.importorskip("torch")

from strands_decider.cuda_graphs import (  # noqa: E402
    BANK_WIDTH,
    GRAPH_COMBINED,
    GRAPH_ROW,
    GRAPH_ROWS,
    GRAPH_STATE,
    GRAPH_STATES,
    Buffers,
    GraphsUnavailable,
    admits_combined,
    admits_two_pass,
    bucket,
    combined_allow,
    count_bucket,
    length_groups,
    pow2,
    row_allow,
    state_allow,
)

# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------


def test_bucket_never_loses_tokens_and_stays_within_its_padding_bound():
    """A bucket below `n` would silently truncate a pass; one far above it wastes a pass.

    The contract in the docstring is "less than 2/steps of n, or fewer than `floor`
    tokens", which at the default steps=8 is 25%.
    """
    for n in range(1, 4100):
        b = bucket(n)
        assert b >= n, n
        assert b - n < max(16, 0.25 * n) + 1, n


def test_bucket_is_monotonic_so_a_longer_pass_never_lands_in_a_smaller_graph():
    values = [bucket(n) for n in range(1, 2000)]
    assert values == sorted(values)


def test_bucket_steps_are_coarse_enough_to_keep_the_graph_count_small():
    # The whole point of bucketing is that a traffic mix meets few shapes. Over the range
    # a question row can occupy, there should be tens of buckets, not hundreds.
    assert len({bucket(n) for n in range(1, GRAPH_ROW + 1)}) <= 24


def test_bucket_refuses_a_non_positive_length():
    # A zero-length pass would index `buf[:, :0]` and forward nothing, which is a bug in
    # the caller rather than a shape to pad.
    with pytest.raises(ValueError):
        bucket(0)


def test_count_bucket_matches_the_documented_ladder():
    assert [count_bucket(n) for n in range(1, 17)] == [
        1, 2, 3, 4, 6, 6, 8, 8, 12, 12, 12, 12, 16, 16, 16, 16]


def test_pow2_rounds_up_to_a_power_of_two():
    assert [pow2(n) for n in (1, 2, 3, 16, 17, 1024, 1025)] == [
        1, 2, 4, 16, 32, 1024, 2048]


# ---------------------------------------------------------------------------
# Grouping rows into passes
# ---------------------------------------------------------------------------


def test_length_groups_covers_every_row_exactly_once():
    lengths = [40, 37, 210, 41, 38, 39, 205, 36]
    groups = length_groups(lengths, GRAPH_ROWS)
    assert sorted(i for g in groups for i in g) == list(range(len(lengths)))


def test_length_groups_respects_the_cap():
    groups = length_groups([50] * 40, 8)
    assert all(len(g) <= 8 for g in groups)
    assert sum(len(g) for g in groups) == 40


def test_length_groups_keeps_a_long_row_out_of_the_short_rows_pass():
    """One 200-token question must not drag six 40-token ones into the 200 bucket -- that
    is five times the tokens, and `max_rows` already made the pass big enough to notice."""
    lengths = [40, 40, 40, 40, 40, 40, 200]
    groups = length_groups(lengths, GRAPH_ROWS)
    long_group = next(g for g in groups if 6 in g)
    assert long_group == [6]


def test_length_groups_keeps_rows_together_when_splitting_would_cost_more():
    # Eight identical short rows cost one padded pass; splitting them would pay the
    # PASS_TOKENS floor twice for no benefit.
    assert length_groups([40] * 8, GRAPH_ROWS) == [list(range(8))]


# ---------------------------------------------------------------------------
# The envelope: which shapes take a graph and which fall back
# ---------------------------------------------------------------------------


def test_admits_combined_at_the_boundary():
    assert admits_combined([GRAPH_COMBINED])
    assert not admits_combined([GRAPH_COMBINED + 1])
    assert not admits_combined([])


def test_admits_two_pass_at_each_boundary():
    ok = dict(state_lens=[GRAPH_STATE], row_lens=[GRAPH_ROW])
    assert admits_two_pass(**ok)
    # A state past the graphed length: the pass is compute-bound there, so eager is right.
    assert not admits_two_pass(state_lens=[GRAPH_STATE + 1], row_lens=[GRAPH_ROW])
    # A question longer than the row bucket (`_fit` allows up to 0.75 x max_length).
    assert not admits_two_pass(state_lens=[64], row_lens=[GRAPH_ROW + 1])
    # More distinct states than the bank has entries.
    assert not admits_two_pass(state_lens=[64] * (GRAPH_STATES + 1), row_lens=[32])
    assert admits_two_pass(state_lens=[64] * GRAPH_STATES, row_lens=[32])
    assert not admits_two_pass(state_lens=[], row_lens=[32])
    assert not admits_two_pass(state_lens=[64], row_lens=[])


def test_the_bank_can_hold_any_state_the_window_admits():
    """`BANK_WIDTH` below the model's context window would mean a legal request whose state
    cannot be stored, which is a shape bug rather than a tuning choice."""
    from strands_decider.modeling import StrandsDeciderConfig

    assert BANK_WIDTH >= StrandsDeciderConfig().max_length


# ---------------------------------------------------------------------------
# The padding masks
# ---------------------------------------------------------------------------


def _reference_combined(lens: list[int], Lb: int) -> torch.Tensor:
    """Brute force: a real query sees exactly the real tokens at or before it."""
    allow = torch.zeros(len(lens), Lb, Lb, dtype=torch.bool)
    for r, n in enumerate(lens):
        for q in range(Lb):
            for k in range(Lb):
                allow[r, q, k] = (k <= q and k < n) or k == q
    return allow


def test_combined_mask_matches_a_brute_force_causal_mask():
    lens = [8, 5, 1]
    got = combined_allow(torch.tensor(lens)[:, None], 8)
    assert torch.equal(got, _reference_combined(lens, 8))


def test_combined_mask_lets_a_real_query_see_every_real_token_up_to_itself_and_no_pad():
    Lb, lens = 12, [12, 7, 3]
    allow = combined_allow(torch.tensor(lens)[:, None], Lb)
    for r, n in enumerate(lens):
        for q in range(n):
            seen = allow[r, q].nonzero().flatten().tolist()
            assert seen == list(range(q + 1)), (r, q, seen)


def test_state_mask_hides_the_left_padding_and_nothing_else():
    """Left padding is the half of this that is not optional: the pads must be invisible as
    keys, and every real token must see every earlier real token."""
    Sb, lens = 10, [10, 6, 2]
    pad = torch.tensor([Sb - n for n in lens])[:, None]
    allow = state_allow(pad, Sb)
    for r, n in enumerate(lens):
        start = Sb - n
        for q in range(start, Sb):
            seen = allow[r, q].nonzero().flatten().tolist()
            assert seen == list(range(start, q + 1)), (r, q, seen)


def test_row_mask_shows_a_row_its_own_state_and_never_a_neighbours():
    """The bank is shared storage. A row whose state is 3 tokens must see exactly the last
    3 slots of the window, not the slots another, longer state left behind."""
    Sr, Lb = 8, 4
    plen = torch.tensor([8, 3, 0])[:, None]
    rowlen = torch.tensor([4, 2, 0])[:, None]
    allow = row_allow(plen, rowlen, Sr, Lb)
    for r in range(3):
        for q in range(int(rowlen[r])):
            seen = allow[r, q].nonzero().flatten().tolist()
            want = list(range(Sr - int(plen[r]), Sr)) + [Sr + j for j in range(q + 1)]
            assert seen == want, (r, q, seen, want)


def test_no_mask_ever_leaves_a_row_fully_masked():
    """A fully-masked row of an additive mask is NaN with `-inf`, and NaN * 0 = NaN -- so
    one unused pad row would poison the readout of the rows beside it."""
    assert combined_allow(torch.tensor([[0], [3]]), 6).any(-1).all()
    assert state_allow(torch.tensor([[6], [2]]), 6).any(-1).all()
    assert row_allow(torch.tensor([[0], [4]]), torch.tensor([[0], [2]]), 8, 4).any(-1).all()


def test_a_filler_row_cannot_reach_any_real_position():
    """The rows that pad a pass up to its count bucket carry length 0 and entry 0. They
    may read whatever they like -- their outputs are discarded -- but they must not appear
    as a *key* to a real row, or they would enter its softmax denominator."""
    Sr, Lb = 8, 4
    plen = torch.tensor([8, 0])[:, None]      # row 0 real, row 1 filler
    rowlen = torch.tensor([4, 0])[:, None]
    allow = row_allow(plen, rowlen, Sr, Lb)
    # Masks are per row, so a filler row is not a key for anyone; what has to hold is that
    # the real row's own pads are not keys for its real queries.
    for q in range(4):
        assert not allow[0, q, Sr + 4:].any()


# ---------------------------------------------------------------------------
# Buffer layout
# ---------------------------------------------------------------------------


class _StubProbe:
    """A cache layout like the merged Qwen3.5 torso's, at 1/100 the size.

    Built from the real `DynamicLayer` so `is_attention` classifies it the way it will in
    the image, rather than from a duck type that could diverge from transformers.
    """

    def __init__(self, heads=2, dim=4, n_attention=2, n_linear=3, conv=6, state=(3, 4, 4)):
        from transformers.cache_utils import DynamicLayer

        layers = []
        for _ in range(n_attention):
            layer = DynamicLayer()
            layer.keys = torch.zeros(1, heads, 7, dim)
            layer.values = torch.zeros(1, heads, 7, dim)
            layers.append(layer)
        for _ in range(n_linear):
            layers.append(_StubLinear(conv, state))
        self.layers = layers


class _StubLinear:
    def __init__(self, conv: int, state: tuple[int, ...]):
        self.conv_states = {0: torch.zeros(1, conv, 4)}
        self.recurrent_states = {0: torch.zeros(1, *state)}


def test_buffer_views_have_the_shape_each_pass_needs():
    probe = _StubProbe()
    buffers = Buffers(probe, tokens=64, entries=8, device="cpu")
    views = buffers.views(4, 16)
    assert len(views) == 5
    for keys, values in views[:2]:                 # attention layers
        assert tuple(keys.shape) == (4, 2, 16, 4)
        assert tuple(values.shape) == (4, 2, 16, 4)
    for conv, recurrent in views[2:]:              # Gated DeltaNet layers
        assert tuple(conv.shape) == (4, 6, 4)
        assert tuple(recurrent.shape) == (4, 3, 4, 4)


def test_buffer_views_of_one_flat_buffer_do_not_overlap_between_entries():
    """Entries share one flat allocation, so an off-by-one in the view arithmetic makes
    one row's keys another row's keys -- which is cross-request contamination."""
    probe = _StubProbe()
    buffers = Buffers(probe, tokens=64, entries=8, device="cpu")
    keys, _values = buffers.views(4, 16)[0]
    keys.zero_()
    keys[2] = 1.0
    assert keys[0].sum() == 0 and keys[1].sum() == 0 and keys[3].sum() == 0
    assert keys[2].sum() == keys[2].numel()


def test_buffer_views_are_a_window_on_the_same_storage_at_every_shape():
    """A graph replays into fixed addresses, so two shapes drawn from one buffer must start
    at the same place -- otherwise a captured graph writes where nothing reads."""
    probe = _StubProbe()
    buffers = Buffers(probe, tokens=64, entries=8, device="cpu")
    a = buffers.views(4, 16)[0][0]
    b = buffers.views(2, 8)[0][0]
    assert a.data_ptr() == b.data_ptr()


def test_buffers_refuse_a_view_larger_than_the_allocation():
    """Better a clear refusal than a `view()` error three frames down, or worse a silent
    reinterpretation of the neighbouring layer's memory."""
    probe = _StubProbe()
    buffers = Buffers(probe, tokens=32, entries=2, device="cpu")
    with pytest.raises(GraphsUnavailable):
        buffers.views(16, 64)


def test_buffers_keep_each_layers_dtype():
    """The Gated DeltaNet recurrent state is fp32 while the keys are bf16 in the image;
    allocating one dtype for both would quietly halve or double a state."""
    probe = _StubProbe()
    probe.layers[0].keys = probe.layers[0].keys.to(torch.bfloat16)
    probe.layers[0].values = probe.layers[0].values.to(torch.bfloat16)
    buffers = Buffers(probe, tokens=64, entries=8, device="cpu")
    keys, _ = buffers.views(2, 8)[0]
    conv, recurrent = buffers.views(2, 8)[2]
    assert keys.dtype is torch.bfloat16
    assert conv.dtype is torch.float32 and recurrent.dtype is torch.float32


# ---------------------------------------------------------------------------
# Fallback selection
# ---------------------------------------------------------------------------


def test_graphs_refuse_a_cpu_torso_rather_than_pretending():
    from strands_decider.cuda_graphs import CudaGraphs

    class _Torso(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(2))

    with pytest.raises(GraphsUnavailable, match="CUDA"):
        CudaGraphs(_Torso(), pad_id=0)


def _engine_with_graphs_requested():
    """A `BatchedSystemOneEngine` with graphs asked for, on a stub model, on the CPU."""
    from strands_decider.batch_engine import BatchedSystemOneEngine
    from strands_decider.infer import EngineConfig

    class _Cfg:
        ordinal_smoothing = 0.0
        head_type = "pointer"
        num_slots = 16
        max_length = 128
        temperature = 1.0
        temperature_by_kind: ClassVar[dict[str, float]] = {}

    class _Torso(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(2))

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = _Cfg()
            self.torso = _Torso()

    engine = BatchedSystemOneEngine.__new__(BatchedSystemOneEngine)
    engine.cfg = EngineConfig(device="cpu", model_name="stub")
    engine.device = "cpu"
    engine.max_rows = 128
    engine.dup_token_budget = 480
    engine.want_cuda_graphs = True
    engine.want_cuda_graphs_two_pass = True
    engine._graphs = None
    engine._graphs_tried = False
    engine._model = _Model()
    type(engine).model = property(lambda self: self._model)  # type: ignore[assignment]
    return engine


def test_an_engine_that_cannot_capture_degrades_instead_of_failing():
    """Same instinct as `UnforkableCache` falling back to batched encoding: the eager path
    gives the same answers, so an unavailable graph path must never be a failed request."""
    engine = _engine_with_graphs_requested()
    try:
        assert engine.graphs() is None
        assert engine.capture_pending() == 0
        assert engine.graph_stats() == {}
        # Asked twice, probed once: the refusal is remembered rather than re-raised per
        # request, which would put the construction cost on every call.
        assert engine.graphs() is None
        assert engine._graphs_tried is True
    finally:
        del type(engine).model


def test_the_two_pass_route_is_gated_separately_from_the_one_pass_route():
    """`SD_CUDA_GRAPHS=1` must not switch on the state/row pair, which measured slower on
    an L4 (0.73x-1.02x at 7 questions). Only `SD_CUDA_GRAPHS=all` does."""
    engine = _engine_with_graphs_requested()
    try:
        engine.want_cuda_graphs_two_pass = False
        # Declines before it ever asks for a graph, so this holds on a CPU-only box too.
        assert engine._graph_two_pass([[1] * 64], [0], [[1] * 32], [], []) is None
    finally:
        del type(engine).model


def test_a_runtime_failure_turns_graphs_off_for_good():
    """A pass that raised part-way may have left the shared buffers half written. Retrying
    per request would turn one bug into a latency cliff on every request."""
    engine = _engine_with_graphs_requested()
    try:
        engine.want_cuda_graphs = True
        engine._disable_graphs(RuntimeError("CUDA error: unspecified launch failure"))
        assert engine.want_cuda_graphs is False
        assert engine.want_cuda_graphs_two_pass is False
        assert engine._graphs is None
    finally:
        del type(engine).model


def test_the_graph_routes_decline_a_shape_outside_the_envelope_without_touching_cuda():
    """`_graph_combined` / `_graph_two_pass` return None, and the caller runs eager."""
    engine = _engine_with_graphs_requested()
    try:
        assert engine._graph_combined([[1] * (GRAPH_COMBINED + 1)], [], None) is None
        assert engine._graph_two_pass([[1] * 10], [0], [[1] * 4], [], []) is None
    finally:
        del type(engine).model


def test_row_mask_for_the_readout_matches_right_padding():
    """The graph path knows every length, but hands `pool_last_token` the same 1/0 mask the
    eager path would have built, so there is one pooling rule and not two."""
    engine = _engine_with_graphs_requested()
    try:
        mask = engine._row_mask([4, 2, 1], 4)
        assert mask.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]]
    finally:
        del type(engine).model
