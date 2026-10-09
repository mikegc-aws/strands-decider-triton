"""The Triton backend's batching logic, with Triton and the engine both stubbed.

The real backend only runs inside a `tritonserver` process on a GPU. What is testable here
is the part most likely to be wrong and most expensive to get wrong: how one Triton batch
is split into engine calls, and how the answers are routed back.

The failure this file mostly exists to prevent is **cross-request answer contamination** --
caller A receiving caller B's probability distribution, at HTTP 200, with a well-formed
body. There is no way to notice that from outside.

BOTH of `execute`'s paths are covered, and that is the point of the parametrised `model`
fixture. `execute` prefers the engine's `evaluate_many` when it has one and falls back to a
per-request loop otherwise:

    _BatchedStubEngine  has evaluate_many  -> the batched path. What SD_ENGINE=merged runs,
                                             i.e. production.
    _StubEngine         has not            -> the per-request path. What SD_ENGINE=hf runs,
                                             and the fallback when the batched path raises.

This file previously defined only the second one, so every test exercised the path
production does *not* take, and the contamination properties above were unverified on the
path that actually serves traffic.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_PY = ROOT / "model_repository" / "decider" / "1" / "model.py"


# --------------------------------------------------------------------- the Triton stub


class _Tensor:
    def __init__(self, name, value):
        self._name = name
        self._value = value

    def as_numpy(self):
        import numpy as np

        return np.array([self._value], dtype=object)


class _InferenceResponse:
    def __init__(self, output_tensors):
        self.output_tensors = output_tensors

    def body(self):
        import numpy as np

        arr = self.output_tensors[0]._value
        if isinstance(arr, np.ndarray):
            arr = arr.reshape(-1)[0]
        return json.loads(bytes(arr).decode())


def _install_triton_stub():
    pb = types.ModuleType("triton_python_backend_utils")

    class TritonModelException(Exception):
        pass

    class Logger:
        @staticmethod
        def log_info(*a):
            pass

        @staticmethod
        def log_warn(*a):
            pass

        @staticmethod
        def log_error(*a):
            pass

    def get_input_tensor_by_name(request, name):
        return request.get(name)

    class Tensor:
        def __init__(self, name, value):
            self._name = name
            self._value = value

    pb.TritonModelException = TritonModelException
    pb.Logger = Logger
    pb.get_input_tensor_by_name = get_input_tensor_by_name
    pb.Tensor = Tensor
    pb.InferenceResponse = _InferenceResponse
    sys.modules["triton_python_backend_utils"] = pb
    return pb


def _load_model_module():
    _install_triton_stub()
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("decider_triton_model", MODEL_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------- engine stubs


class _StubEngine:
    """Per-request engine, with no `evaluate_many`. What `SD_ENGINE=hf` builds.

    Records the requests it was handed and answers every question deterministically.
    """

    def __init__(self):
        self.calls: list = []
        self.many_calls: list = []

    def evaluate(self, request):
        from strands_decider.schema import NoulAnswer, SystemOneResponse, Usage

        self.calls.append(request)
        answers = {}
        for i, name in enumerate(request.questions):
            # A distinct value per question, so a misrouted answer is detectable.
            answers[name] = NoulAnswer(noul=round(0.1 * (i + 1), 4))
        return SystemOneResponse(model="stub", answers=answers,
                                 usage=Usage(input_tokens=300, output_tokens=len(answers)))


class _BatchedStubEngine(_StubEngine):
    """Engine offering `evaluate_many`, like `BatchedSystemOneEngine`. Production's path.

    Mirrors the real contract: one call for the whole batch, and a per-request error is
    RETURNED in place rather than raised, so one bad request cannot fail its neighbours.
    """

    def evaluate_many(self, requests):
        self.many_calls.append(list(requests))
        out: list = []
        for r in requests:
            try:
                out.append(self.evaluate(r))
            except Exception as exc:
                out.append(exc)
        return out


def _build(engine):
    mod = _load_model_module()
    m = mod.TritonPythonModel()
    m.logger = sys.modules["triton_python_backend_utils"].Logger
    m.engine = engine
    return m


@pytest.fixture(params=["batched", "per_request"])
def model(request):
    """Both paths through `execute`. See the module docstring."""
    return _build(_BatchedStubEngine() if request.param == "batched" else _StubEngine())


@pytest.fixture
def per_request_model():
    """Only the per-request path, for assertions about `_group_by_state` itself."""
    return _build(_StubEngine())


@pytest.fixture
def batched_model():
    """Only the batched path, for assertions about `evaluate_many` dispatch."""
    return _build(_BatchedStubEngine())


def _req(state, questions: dict) -> dict:
    body = json.dumps({"state": state, "questions": questions}).encode()
    return {"REQUEST_JSON": _Tensor("REQUEST_JSON", body)}


def _noul(instr="urgent?"):
    return {"type": "noul", "instructions": instr}


# ------------------------------------------------------- routing, on BOTH paths
#
# These are the properties a caller can actually observe, so they must hold whichever
# path `execute` took.


def test_answers_are_never_routed_to_the_wrong_caller(model):
    """Each caller gets exactly its own questions back, and no others."""
    reqs = [_req("s", {"alpha": _noul(), "beta": _noul()}),
            _req("s", {"gamma": _noul()})]
    responses = model.execute(reqs)

    assert set(responses[0].body()["answers"]) == {"alpha", "beta"}
    assert set(responses[1].body()["answers"]) == {"gamma"}


def test_duplicate_question_names_across_callers_do_not_collide(model):
    """Names are caller-chosen; two callers may both use "urgent"."""
    reqs = [_req("s", {"urgent": _noul("A")}), _req("s", {"urgent": _noul("B")})]
    responses = model.execute(reqs)

    assert set(responses[0].body()["answers"]) == {"urgent"}
    assert set(responses[1].body()["answers"]) == {"urgent"}


def test_question_names_resembling_internal_keys_are_handled(model):
    """Caller-chosen names that look like the backend's own bookkeeping.

    Checked rather than assumed. The earlier `__{slot}__{name}` prefix scheme was in fact
    collision-free here too -- I simulated this exact input against it and it routed
    correctly -- so this is not a regression test for a live bug. The backend carries an
    explicit mapping anyway, because "no string parsing, so no collision to reason about"
    is a cheaper property to hold than "the prefix is unambiguous for all caller input".
    """
    reqs = [_req("s", {"__1__x": _noul(), "mine": _noul()}),
            _req("s", {"x": _noul()})]
    responses = model.execute(reqs)

    assert set(responses[0].body()["answers"]) == {"__1__x", "mine"}
    assert set(responses[1].body()["answers"]) == {"x"}


def test_every_request_gets_exactly_one_response_in_order(model):
    """Triton requires it; a missing response hangs that client."""
    reqs = [_req("a", {"q": _noul()}), _req("b", {"q": _noul()}),
            _req("a", {"q": _noul()})]
    responses = model.execute(reqs)
    assert len(responses) == 3
    assert all(r is not None for r in responses)


def test_a_malformed_request_does_not_fail_its_neighbours(model):
    """One caller's bad JSON must not take down the batch it happened to land in."""
    good = _req("s", {"q": _noul()})
    bad = {"REQUEST_JSON": _Tensor("REQUEST_JSON", b"{not json")}
    responses = model.execute([bad, good, bad])

    assert "error" in responses[0].body()
    assert "answers" in responses[1].body()
    assert "error" in responses[2].body()


def test_a_missing_input_tensor_is_a_caller_error(model):
    responses = model.execute([{}])
    assert "error" in responses[0].body()


def test_engine_valueerror_becomes_a_caller_error_not_a_server_error(model):
    """The engine raises ValueError for option-count and truncation problems, which the
    FastAPI server maps to 422. The two deployables must agree."""
    def boom(_request):
        raise ValueError("question has 300 options but this model has 24 slots")

    model.engine.evaluate = boom
    body = model.execute([_req("s", {"q": _noul()})])[0].body()
    assert body["error"]["type"] == "invalid_request"
    assert "300 options" in body["error"]["message"]


def test_an_unexpected_engine_failure_is_tagged_internal(model):
    def boom(_request):
        raise RuntimeError("CUDA out of memory")

    model.engine.evaluate = boom
    body = model.execute([_req("s", {"q": _noul()})])[0].body()
    assert body["error"]["type"] == "internal"


def test_response_shape_matches_the_http_server(model):
    body = model.execute([_req("s", {"q": _noul()})])[0].body()
    assert set(body) == {"model", "answers", "usage", "latency_ms"}


def test_an_empty_batch_is_a_no_op(model):
    assert model.execute([]) == []


# ------------------------------------------------- the batched path specifically


def test_the_whole_batch_becomes_one_engine_call(batched_model):
    """The reason Triton is here. 8 requests must not cost 8 engine calls."""
    reqs = [_req(f"state {i}", {"q": _noul()}) for i in range(8)]
    batched_model.execute(reqs)

    assert len(batched_model.engine.many_calls) == 1, "one evaluate_many for the batch"
    assert len(batched_model.engine.many_calls[0]) == 8


def test_the_per_request_path_is_not_used_when_evaluate_many_exists(batched_model):
    """Belt and braces: distinct states must not quietly fall through to the old loop."""
    reqs = [_req("one", {"a": _noul()}), _req("two", {"b": _noul()})]
    responses = batched_model.execute(reqs)

    assert len(batched_model.engine.many_calls) == 1
    assert batched_model.engine.calls == [] or len(batched_model.engine.calls) == 2, (
        "the stub's evaluate_many delegates per request; what matters is that execute "
        "itself did not run the per-request loop")
    assert set(responses[0].body()["answers"]) == {"a"}
    assert set(responses[1].body()["answers"]) == {"b"}


def test_a_batched_failure_falls_back_to_the_per_request_path(batched_model):
    """A failure OF the batched path, not of one request, must not fail every caller.

    The single-request route shares `_fit`, the readout and the head, so it is the same
    answers by a slower route -- which beats returning an error to everyone.
    """
    def explode(_requests):
        raise RuntimeError("batched forward blew up")

    batched_model.engine.evaluate_many = explode
    reqs = [_req("a", {"q": _noul()}), _req("b", {"q": _noul()})]
    responses = batched_model.execute(reqs)

    assert len(batched_model.engine.calls) == 2, "fell back to per-request evaluate"
    assert all("answers" in r.body() for r in responses)


def test_a_per_request_error_from_evaluate_many_is_attributed_to_that_request(batched_model):
    """`evaluate_many` RETURNS errors per request. A ValueError in slot 1 must land on
    slot 1 and nowhere else."""
    real = batched_model.engine.evaluate_many

    def selective(requests):
        out = real(requests)
        out[1] = ValueError("question has 300 options but this model has 24 slots")
        return out

    batched_model.engine.evaluate_many = selective
    responses = batched_model.execute([_req("a", {"q": _noul()}),
                                       _req("b", {"q": _noul()}),
                                       _req("c", {"q": _noul()})])

    assert "answers" in responses[0].body()
    assert responses[1].body()["error"]["type"] == "invalid_request"
    assert "answers" in responses[2].body()


# ------------------------------------------- the per-request path specifically


def test_same_state_requests_become_one_engine_call(per_request_model):
    """The point of grouping here: the state is encoded once for the whole group."""
    reqs = [_req("same payload", {"a": _noul()}),
            _req("same payload", {"b": _noul()}),
            _req("same payload", {"c": _noul()})]
    responses = per_request_model.execute(reqs)

    assert len(per_request_model.engine.calls) == 1, "same-state requests share one forward"
    assert len(responses) == 3
    assert set(responses[0].body()["answers"]) == {"a"}
    assert set(responses[1].body()["answers"]) == {"b"}
    assert set(responses[2].body()["answers"]) == {"c"}


def test_distinct_states_get_their_own_calls(per_request_model):
    """Merging unrelated states would re-encode each one anyway, so it is not done."""
    reqs = [_req("one", {"a": _noul()}), _req("two", {"b": _noul()})]
    per_request_model.execute(reqs)
    assert len(per_request_model.engine.calls) == 2


def test_structured_state_groups_by_value(per_request_model):
    """A dict state is rendered deterministically, so equal dicts share a forward."""
    reqs = [_req({"x": 1, "y": 2}, {"a": _noul()}),
            _req({"x": 1, "y": 2}, {"b": _noul()})]
    per_request_model.execute(reqs)
    assert len(per_request_model.engine.calls) == 1


def test_a_string_state_is_not_grouped_with_an_equal_looking_dict(per_request_model):
    """Regression. `json.dumps({"a": 1})` is exactly the string `'{"a": 1}'`, so keying
    the groups on `json.dumps` merged these two -- but they RENDER differently
    (`render_content` uses a string as-is and re-emits a dict with `indent=2`). Since
    `_merge_same_state` keeps only the first member's state, the second caller would have
    received answers computed against a prompt it never sent, at HTTP 200. The key is the
    rendered state now, so the two stay apart.
    """
    as_dict = _req({"a": 1}, {"q": _noul()})
    as_text = _req(json.dumps({"a": 1}), {"q": _noul()})
    assert json.dumps({"a": 1}) == '{"a": 1}'  # the collision this guards

    responses = per_request_model.execute([as_dict, as_text])

    assert len(per_request_model.engine.calls) == 2, (
        "a string state and a dict state render differently and must not be merged")
    states = [c.state for c in per_request_model.engine.calls]
    assert {"a": 1} in states and '{"a": 1}' in states
    assert all("answers" in r.body() for r in responses)


def test_shared_state_tokens_are_not_charged_to_every_caller(per_request_model):
    """The state was encoded once; billing each member for all of it would overcount."""
    reqs = [_req("s", {"a": _noul()}), _req("s", {"b": _noul()})]
    responses = per_request_model.execute(reqs)
    total = sum(r.body()["usage"]["input_tokens"] for r in responses)
    assert total <= 300, f"group encoded 300 tokens but reported {total}"


# ------------------------------------------------- version tolerance of the engine loader
#
# These cover `_engine_kwargs`, which exists because of a specific, expensive failure: a
# SageMaker `ModelDataUrl` overlay puts this model.py in front of whatever
# `strands_decider` the IMAGE was built with, and the newest published image
# (v23-triton-onepass) predates `fuse_layers`, `cuda_graphs` and `max_rows`. Passing one of
# those blind raises `TypeError: load_merged_engine() got an unexpected keyword argument`
# inside `initialize()`, after which Triton never reports ready and SageMaker fails the
# endpoint ~31 MINUTES later on the ping health check. That happened once and cost a deploy.
#
# The asymmetry is the whole design and each half is asserted below: an option still at its
# default may be DROPPED (nothing was asked for), but an option explicitly turned on must
# RAISE (serving the un-fused torso after being asked for the fused one would make a
# benchmark comparison a lie).


def _old_loader(checkpoint, merged, *, device="cuda", use_prefix_cache=True,
                max_batch=32, model_name="x"):
    """A `load_merged_engine` from before any of the optional kwargs existed."""


def _new_loader(checkpoint, merged, *, device="cuda", use_prefix_cache=True,
                max_batch=32, model_name="x", fuse_layers=False, cuda_graphs=False,
                cuda_graphs_two_pass=False, max_rows=128):
    """Today's signature."""


def test_supported_kwargs_are_passed_through(batched_model):
    keep = batched_model._engine_kwargs(_new_loader, {
        "fuse_layers": (True, False), "max_rows": (256, 128)})
    assert keep == {"fuse_layers": True, "max_rows": 256}


def test_defaulted_kwargs_are_dropped_against_an_older_engine(batched_model):
    """The no-op case. Nothing was requested, so an engine that cannot do it is already
    doing what the caller wanted -- and dropping is what keeps an overlay deployable."""
    keep = batched_model._engine_kwargs(_old_loader, {
        "fuse_layers": (False, False), "cuda_graphs": (False, False),
        "max_rows": (128, 128)})
    assert keep == {}


def test_explicitly_enabled_kwarg_is_refused_rather_than_ignored(batched_model):
    """The load-bearing half. `SD_MAX_ROWS=256` against an engine pinned at 128 must fail
    the load, not serve 128 quietly -- otherwise the throughput number gets attributed to a
    row budget that was never in effect."""
    pb = sys.modules["triton_python_backend_utils"]
    with pytest.raises(pb.TritonModelException) as exc:
        batched_model._engine_kwargs(_old_loader, {"max_rows": (256, 128)})
    assert "max_rows=256" in str(exc.value)
    # The message must name the cause, because the reader is looking at a failed endpoint.
    assert "overlay" in str(exc.value).lower()


def test_refusal_mentions_every_unsupported_request(batched_model):
    pb = sys.modules["triton_python_backend_utils"]
    with pytest.raises(pb.TritonModelException) as exc:
        batched_model._engine_kwargs(_old_loader, {
            "fuse_layers": (True, False), "max_rows": (256, 128)})
    assert "fuse_layers=True" in str(exc.value)
    assert "max_rows=256" in str(exc.value)
