"""The Triton backend's batching logic, with Triton and the engine both stubbed.

The real backend only runs inside a `tritonserver` process on a GPU. What is testable here
is the part most likely to be wrong and most expensive to get wrong: how one Triton batch
is split into engine calls, and how the answers are routed back.

The failure this file mostly exists to prevent is **cross-request answer contamination** --
caller A receiving caller B's probability distribution, at HTTP 200, with a well-formed
body. There is no way to notice that from outside.
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


# ----------------------------------------------------------------------- the engine stub


class _StubEngine:
    """Records the requests it was handed and answers every question deterministically."""

    def __init__(self):
        self.calls: list = []

    def evaluate(self, request):
        from strands_decider.schema import NoulAnswer, SystemOneResponse, Usage

        self.calls.append(request)
        answers = {}
        for i, name in enumerate(request.questions):
            # A distinct value per question, so a misrouted answer is detectable.
            answers[name] = NoulAnswer(noul=round(0.1 * (i + 1), 4))
        return SystemOneResponse(model="stub", answers=answers,
                                 usage=Usage(input_tokens=300, output_tokens=len(answers)))


@pytest.fixture
def model():
    mod = _load_model_module()
    m = mod.TritonPythonModel()
    m.logger = sys.modules["triton_python_backend_utils"].Logger
    m.engine = _StubEngine()
    return m


def _req(state: str, questions: dict) -> dict:
    body = json.dumps({"state": state, "questions": questions}).encode()
    return {"REQUEST_JSON": _Tensor("REQUEST_JSON", body)}


def _noul(instr="urgent?"):
    return {"type": "noul", "instructions": instr}


# -------------------------------------------------------------------------- grouping


def test_same_state_requests_become_one_engine_call(model):
    """The point of batching here: the state is encoded once for the whole group."""
    reqs = [_req("same payload", {"a": _noul()}),
            _req("same payload", {"b": _noul()}),
            _req("same payload", {"c": _noul()})]
    responses = model.execute(reqs)

    assert len(model.engine.calls) == 1, "same-state requests should share one forward"
    assert len(responses) == 3
    assert set(responses[0].body()["answers"]) == {"a"}
    assert set(responses[1].body()["answers"]) == {"b"}
    assert set(responses[2].body()["answers"]) == {"c"}


def test_distinct_states_get_their_own_calls(model):
    """Merging unrelated states would re-encode each one anyway, so it is not done."""
    reqs = [_req("one", {"a": _noul()}), _req("two", {"b": _noul()})]
    model.execute(reqs)
    assert len(model.engine.calls) == 2


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

    assert len(model.engine.calls) == 1
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


def test_structured_state_groups_by_value(model):
    """A dict state is rendered deterministically, so equal dicts share a forward."""
    a = {"REQUEST_JSON": _Tensor("REQUEST_JSON", json.dumps(
        {"state": {"x": 1, "y": 2}, "questions": {"a": _noul()}}).encode())}
    b = {"REQUEST_JSON": _Tensor("REQUEST_JSON", json.dumps(
        {"state": {"x": 1, "y": 2}, "questions": {"b": _noul()}}).encode())}
    model.execute([a, b])
    assert len(model.engine.calls) == 1


# ------------------------------------------------------------------- one response each


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


# ------------------------------------------------------------------------------ usage


def test_shared_state_tokens_are_not_charged_to_every_caller(model):
    """The state was encoded once; billing each member for all of it would overcount."""
    reqs = [_req("s", {"a": _noul()}), _req("s", {"b": _noul()})]
    responses = model.execute(reqs)
    total = sum(r.body()["usage"]["input_tokens"] for r in responses)
    assert total <= 300, f"group encoded 300 tokens but reported {total}"


def test_response_shape_matches_the_http_server(model):
    body = model.execute([_req("s", {"q": _noul()})])[0].body()
    assert set(body) == {"model", "answers", "usage", "latency_ms"}
