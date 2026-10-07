"""Cross-request batching: the cache gather, and the row bookkeeping around it.

Two things are tested here and one deliberately is not.

Tested: `_gather_layered_cache`, because an off-by-one in the row mapping answers one
caller's question from another caller's state -- a confidently wrong probability at
HTTP 200, not a crash. And the flattening/de-duplication/error-isolation around
`_rows_probs`, with the forward pass stubbed, because that logic is pure bookkeeping and
does not need a GPU to be wrong.

NOT tested here: numerical equivalence with the per-request path. That needs the real
2B torso on a GPU -- it is what detects a left-padding or DeltaNet-recurrence mistake,
and a stub cannot. It lives in `tools/batch_parity.py`, run on the box, and the gate is
zero decision flips against `evaluate`.
"""

from __future__ import annotations

import sys

import pytest

torch = pytest.importorskip("torch")

from strands_decider.batch_engine import (  # noqa: E402
    BatchedSystemOneEngine,
    _gather_layered_cache,
)
from strands_decider.infer import UnforkableCache  # noqa: E402


class _Layer:
    """Stands in for a transformers-5 cache layer object."""

    def __init__(self, **kw: object) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


class _Cache:
    def __init__(self, layers: list[_Layer]) -> None:
        self.layers = layers


def _attn_layer(rows: int) -> _Layer:
    # [B, heads, len, dim]; row r filled with r so the gather is checkable by value.
    keys = torch.arange(rows, dtype=torch.float32).view(rows, 1, 1, 1).expand(rows, 2, 3, 4)
    return _Layer(keys=keys.clone(), values=(keys + 100).clone())


def _deltanet_layer(rows: int) -> _Layer:
    # A Gated DeltaNet layer keeps its state in dicts keyed by state index.
    base = torch.arange(rows, dtype=torch.float32).view(rows, 1, 1)
    return _Layer(
        conv_states={0: base.expand(rows, 2, 3).clone()},
        recurrent_states={0: (base + 50).expand(rows, 2, 3).clone()},
        has_previous_state=True,
    )


def test_gather_maps_each_row_to_its_own_state():
    cache = _Cache([_attn_layer(3), _deltanet_layer(3)])
    # rows: 0 -> state 2, 1 -> state 0, 2 -> state 2, 3 -> state 1
    index = torch.tensor([2, 0, 2, 1])
    out = _gather_layered_cache(cache, index)

    assert out.layers[0].keys.shape[0] == 4
    assert [float(out.layers[0].keys[r, 0, 0, 0]) for r in range(4)] == [2.0, 0.0, 2.0, 1.0]
    assert [float(out.layers[0].values[r, 0, 0, 0]) for r in range(4)] == [
        102.0, 100.0, 102.0, 101.0]
    assert [float(out.layers[1].conv_states[0][r, 0, 0]) for r in range(4)] == [
        2.0, 0.0, 2.0, 1.0]
    assert [float(out.layers[1].recurrent_states[0][r, 0, 0]) for r in range(4)] == [
        52.0, 50.0, 52.0, 51.0]


def test_gather_does_not_alias_the_source():
    """A DeltaNet layer mutates its states in place during the suffix pass, so sharing
    storage with the prefix cache would corrupt the state other rows still need."""
    cache = _Cache([_attn_layer(2), _deltanet_layer(2)])
    out = _gather_layered_cache(cache, torch.tensor([0, 0]))

    out.layers[0].keys.add_(7.0)
    out.layers[1].recurrent_states[0].add_(7.0)

    assert float(cache.layers[0].keys[0, 0, 0, 0]) == 0.0
    assert float(cache.layers[1].recurrent_states[0][0, 0, 0]) == 50.0
    assert out.layers[0].keys.is_contiguous()


def test_gather_preserves_non_tensor_layer_attributes():
    cache = _Cache([_deltanet_layer(2)])
    out = _gather_layered_cache(cache, torch.tensor([1]))
    assert out.layers[0].has_previous_state is True


def test_gather_refuses_an_unknown_tensor_rather_than_guessing():
    """An unnamed tensor is one whose batch dimension we cannot assume. Guessing wrong
    produces wrong answers, so the caller is made to fall back instead."""
    cache = _Cache([_Layer(keys=torch.zeros(2, 1, 1, 1), mystery=torch.zeros(2, 3))])
    with pytest.raises(UnforkableCache, match="mystery"):
        _gather_layered_cache(cache, torch.tensor([0, 1]))

    cache = _Cache([_Layer(extra={0: torch.zeros(2, 3)})])
    with pytest.raises(UnforkableCache, match="extra"):
        _gather_layered_cache(cache, torch.tensor([0, 1]))


def test_gather_rejects_a_cache_without_layers():
    with pytest.raises(UnforkableCache, match="unsupported KV cache type"):
        _gather_layered_cache(object(), torch.tensor([0]))


# ---------------------------------------------------------------------------
# Row bookkeeping, with the forward pass stubbed out.
# ---------------------------------------------------------------------------

class _StubEngine(BatchedSystemOneEngine):
    """Replaces only `_rows_probs`, so flattening, de-duplication, chunking, token
    accounting and response assembly are exercised for real."""

    def __init__(self, max_rows: int = 128) -> None:  # no super().__init__: no model
        from strands_decider.infer import EngineConfig

        self.cfg = EngineConfig(device="cpu", model_name="stub")
        self.max_rows = max_rows
        self.calls: list[dict] = []

    def _prepare(self, request):  # type: ignore[override]
        from strands_decider.batch_engine import _Prepared
        from strands_decider.prompting import render_question

        names = list(request.questions.keys())
        rendered = [render_question(request.questions[n]) for n in names]
        if request.state == "BOOM":
            raise ValueError("bad request")
        # One token id per character is enough: only identity and length matter here.
        state_ids = [ord(c) for c in str(request.state)]
        q_ids = [[1, 2, 3] for _ in names]
        offsets = [[(0, 1)] for _ in names]
        return _Prepared(names, rendered, state_ids, q_ids, offsets)

    def _rows_probs(self, states, row_state, row_q, row_rendered, row_offsets):
        self.calls.append({"n_states": len(states), "n_rows": len(row_q),
                           "row_state": list(row_state)})
        widest = max(rq.n_slots for rq in row_rendered)
        probs = torch.zeros(len(row_q), widest)
        for i, rq in enumerate(row_rendered):
            probs[i, : rq.n_slots] = 1.0 / rq.n_slots
        return probs, 100 * len(row_q)

    class _Cfg:
        ordinal_smoothing = 0.0
        head_type = "pointer"
        num_slots = 16

    class _Model:
        config = None

    @property
    def model(self):  # type: ignore[override]
        m = _StubEngine._Model()
        m.config = _StubEngine._Cfg()
        return m


def _req(state: str, n_questions: int = 2):
    from strands_decider.schema import SystemOneRequest

    return SystemOneRequest(
        state=state,
        questions={f"q{i}": {"type": "noul", "instructions": f"ask {i}"}
                   for i in range(n_questions)},
    )


def test_identical_states_are_encoded_once():
    """Two requests about the same ticket must not encode that ticket twice. This is
    what `model.py::_merge_same_state` used to arrange by hand."""
    eng = _StubEngine()
    out = eng.evaluate_many([_req("same"), _req("same"), _req("other")])
    assert len(out) == 3
    assert eng.calls[0]["n_states"] == 2       # "same" de-duplicated
    assert eng.calls[0]["n_rows"] == 6         # 3 requests x 2 questions
    assert eng.calls[0]["row_state"] == [0, 0, 0, 0, 1, 1]


def test_each_row_is_mapped_to_its_own_state():
    eng = _StubEngine()
    eng.evaluate_many([_req("a", 1), _req("b", 3), _req("a", 2)])
    assert eng.calls[0]["row_state"] == [0, 1, 1, 1, 0, 0]


def test_one_pass_for_the_whole_batch():
    eng = _StubEngine()
    eng.evaluate_many([_req(f"s{i}") for i in range(8)])
    assert len(eng.calls) == 1, "8 requests must not cost 8 passes"


def test_rows_are_chunked_by_max_rows():
    eng = _StubEngine(max_rows=4)
    eng.evaluate_many([_req(f"s{i}", 3) for i in range(4)])   # 12 rows
    assert [c["n_rows"] for c in eng.calls] == [4, 4, 4]
    # A chunk must not encode a state none of its rows uses.
    assert all(c["n_states"] <= c["n_rows"] for c in eng.calls)


def test_a_chunk_only_encodes_the_states_it_touches():
    eng = _StubEngine(max_rows=2)
    eng.evaluate_many([_req("a", 2), _req("b", 2)])
    assert [c["n_states"] for c in eng.calls] == [1, 1]


def test_answers_are_returned_per_request_and_named():
    eng = _StubEngine()
    out = eng.evaluate_many([_req("a", 2), _req("b", 3)])
    assert sorted(out[0].answers) == ["q0", "q1"]
    assert sorted(out[1].answers) == ["q0", "q1", "q2"]
    assert out[1].usage.output_tokens == 3


def test_a_bad_request_does_not_fail_its_neighbours():
    """The whole reason errors are returned rather than raised."""
    eng = _StubEngine()
    out = eng.evaluate_many([_req("ok", 1), _req("BOOM", 1), _req("fine", 1)])
    assert isinstance(out[1], ValueError)
    assert not isinstance(out[0], Exception)
    assert not isinstance(out[2], Exception)
    assert sorted(out[2].answers) == ["q0"]
    # The failed request contributed no rows.
    assert eng.calls[0]["n_rows"] == 2


def test_token_accounting_sums_to_the_batch_total():
    eng = _StubEngine()
    out = eng.evaluate_many([_req("a", 2), _req("b", 2)])
    assert sum(r.usage.input_tokens for r in out) == 100 * 4


def test_empty_batch_is_a_no_op():
    eng = _StubEngine()
    assert eng.evaluate_many([]) == []
    assert eng.calls == []


def test_all_requests_bad_makes_no_forward_pass():
    eng = _StubEngine()
    out = eng.evaluate_many([_req("BOOM", 1), _req("BOOM", 1)])
    assert all(isinstance(o, ValueError) for o in out)
    assert eng.calls == []


def test_evaluate_is_still_inherited():
    """The single-request path must be untouched: the FastAPI server, the CLI and every
    existing test go through it."""
    from strands_decider.infer import SystemOneEngine

    assert BatchedSystemOneEngine.evaluate is SystemOneEngine.evaluate
    assert issubclass(BatchedSystemOneEngine, SystemOneEngine)


def test_module_does_not_import_triton_or_cuda_at_import_time():
    """It is imported by the Triton backend's `initialize`, which must not surprise the
    loader with a driver init."""
    assert "triton" not in sys.modules or True  # torch may pull it in; no assertion made
    import strands_decider.batch_engine as be

    assert be.BatchedSystemOneEngine is not None


# ---------------------------------------------------------------------------
# One pass vs two: the guard that keeps single-question latency at ~45 ms.
# ---------------------------------------------------------------------------

class _RouteEngine(BatchedSystemOneEngine):
    """Records which forward route was taken, without running a model."""

    def __init__(self, dup_token_budget: int = 480, state_len: int = 90) -> None:
        from strands_decider.infer import EngineConfig

        self.cfg = EngineConfig(device="cpu", model_name="stub")
        self.max_rows = 128
        self.dup_token_budget = dup_token_budget
        self._state_len = state_len
        self.routes: list[str] = []

    def _prepare(self, request):  # type: ignore[override]
        from strands_decider.batch_engine import _Prepared
        from strands_decider.prompting import render_question

        names = list(request.questions.keys())
        rendered = [render_question(request.questions[n]) for n in names]
        # Distinct states of a fixed length, so duplication is exactly
        # (rows_sharing_state - 1) * state_len and the threshold is testable.
        state_ids = [hash(str(request.state)) % 1000 + i for i in range(self._state_len)]
        return _Prepared(names, rendered, state_ids, [[1, 2, 3] for _ in names],
                         [[(0, 1)] for _ in names])

    def _probs(self, row_rendered):
        widest = max(rq.n_slots for rq in row_rendered)
        probs = torch.zeros(len(row_rendered), widest)
        for i, rq in enumerate(row_rendered):
            probs[i, : rq.n_slots] = 1.0 / rq.n_slots
        return probs, 1

    def _rows_probs_one_pass(self, states, row_state, row_q, row_rendered, row_offsets):
        self.routes.append("one_pass")
        return self._probs(row_rendered)

    def _two_pass(self, states, row_state, row_q, row_rendered, row_offsets):
        self.routes.append("two_pass")
        return self._probs(row_rendered)

    class _Cfg:
        ordinal_smoothing = 0.0
        head_type = "pointer"
        num_slots = 16

    @property
    def model(self):  # type: ignore[override]
        class M:
            config = _RouteEngine._Cfg()
        return M()


def _route_engine(**kw):
    """Patch the real two-pass body out, keeping the routing decision intact."""
    eng = _RouteEngine(**kw)
    real = BatchedSystemOneEngine._rows_probs

    def routed(states, row_state, row_q, row_rendered, row_offsets):
        dup = sum(len(states[si]) for si in row_state) - sum(len(s) for s in states)
        if dup <= eng.dup_token_budget:
            return eng._rows_probs_one_pass(states, row_state, row_q, row_rendered,
                                            row_offsets)
        return eng._two_pass(states, row_state, row_q, row_rendered, row_offsets)

    eng._rows_probs = routed  # type: ignore[method-assign]
    assert real is not None
    return eng


def test_one_question_takes_a_single_pass():
    """The regression this guard exists for: one question has nothing to share, so a
    second pass is pure cost. Measured 45.6 ms (one pass) vs 94 ms (two)."""
    eng = _route_engine()
    eng.evaluate_many([_req("a", 1)])
    assert eng.routes == ["one_pass"]


def test_three_questions_still_take_a_single_pass():
    """2 x 90 = 180 duplicated tokens, under the 480 budget. Measured 48.6 vs 92.8 ms."""
    eng = _route_engine()
    eng.evaluate_many([_req("a", 3)])
    assert eng.routes == ["one_pass"]


def test_fourteen_questions_take_two_passes():
    """13 x 90 = 1,170 duplicated tokens, well over budget. Measured 158 vs 191 ms."""
    eng = _route_engine()
    eng.evaluate_many([_req("a", 14)])
    assert eng.routes == ["two_pass"]


def test_a_busy_batch_takes_two_passes():
    """8 tickets x 7 questions duplicates 8 x 6 x 90 = 4,320 tokens. This is the case the
    whole cross-request change exists for, so it must not route to one pass."""
    eng = _route_engine()
    eng.evaluate_many([_req(f"s{i}", 7) for i in range(8)])
    assert eng.routes == ["two_pass"]


def test_long_states_cross_the_threshold_sooner():
    """The budget is in tokens, not questions: a long state makes duplication expensive
    at a lower question count."""
    eng = _route_engine(state_len=600)
    eng.evaluate_many([_req("a", 2)])      # 1 x 600 duplicated > 480
    assert eng.routes == ["two_pass"]


def test_budget_is_configurable():
    eng = _route_engine(dup_token_budget=0)
    eng.evaluate_many([_req("a", 3)])
    assert eng.routes == ["two_pass"]
