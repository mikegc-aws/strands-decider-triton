"""The Triton wire contract, tested without Triton.

The python backend can only be exercised inside a `tritonserver` process, so a format bug
found there costs a multi-GB image round trip. Everything in `wire.py` is therefore free of
Triton and torch imports and tested here instead.

The contract that matters: a request body accepted by `POST /v1/systemone` must be accepted
here, byte for byte, and the response must carry the same keys. Otherwise clients and
JevBench's `typesafe` adapter would need a second code path for the Triton deployable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decider_triton.wire import (  # noqa: E402
    RequestDecodeError,
    decode_request,
    encode_error,
    encode_response,
    is_error,
)

# The exact body serving/smoke_test.sh posts, so the two cannot drift.
SMOKE_BODY = {
    "state": "My payouts have been failing for three days and nobody has replied.",
    "questions": {
        "needs_escalation": {"type": "noul",
                             "instructions": "Should this be escalated to a human?"},
        "sentiment": {"type": "choice",
                      "instructions": "What is the writer's tone?",
                      "criteria": {"calm": "measured", "frustrated": "annoyed",
                                   "angry": "hostile"}},
        "severity": {"type": "score",
                     "instructions": "How severe is this?",
                     "criteria": ["none", "low", "medium", "high", "critical"]},
    },
}


# ------------------------------------------------------------------------------- decoding


def test_decodes_the_smoke_test_body():
    req = decode_request(json.dumps(SMOKE_BODY).encode())
    assert set(req.questions) == {"needs_escalation", "sentiment", "severity"}
    assert req.questions["severity"].criteria == ["none", "low", "medium", "high", "critical"]


def test_accepts_bytes_and_str_alike():
    """Triton hands TYPE_STRING through as bytes; tests find str more convenient."""
    body = json.dumps(SMOKE_BODY)
    assert decode_request(body).state == decode_request(body.encode()).state


def test_accepts_a_structured_state():
    """`state` may be a dict or list, rendered deterministically by prompting.py."""
    req = decode_request(json.dumps({
        "state": {"ticket": 7, "body": "help"},
        "questions": {"q": {"type": "noul", "instructions": "urgent?"}},
    }))
    assert req.state == {"ticket": 7, "body": "help"}


def test_accepts_an_empty_state():
    """Documented as valid: the question may carry the whole task."""
    req = decode_request(json.dumps({
        "state": "", "questions": {"q": {"type": "noul", "instructions": "2+2=4?"}},
    }))
    assert req.state == ""


@pytest.mark.parametrize("raw,expect", [
    (b"not json at all", "valid JSON"),
    (b'"a bare string"', "must be a JSON object"),
    (b"[1,2,3]", "must be a JSON object"),
    (b'{"state": "x"}', "did not validate"),            # questions is required
    (b'{"state": "x", "questions": {}}', "did not validate"),  # min_length=1
    (b"\xff\xfe invalid utf8", "valid UTF-8"),
])
def test_caller_errors_raise_requestdecodeerror(raw, expect):
    with pytest.raises(RequestDecodeError, match=expect):
        decode_request(raw)


def test_a_choice_with_one_option_is_rejected():
    """schema.py requires >= 2; a single option would make confidence 1.0 by definition."""
    with pytest.raises(RequestDecodeError):
        decode_request(json.dumps({
            "state": "x",
            "questions": {"q": {"type": "choice", "instructions": "pick",
                                "criteria": {"only": "one"}}},
        }))


def test_a_score_with_eleven_levels_is_rejected():
    """MAX_SCORE_LEVELS is 10."""
    with pytest.raises(RequestDecodeError):
        decode_request(json.dumps({
            "state": "x",
            "questions": {"q": {"type": "score", "instructions": "rate",
                                "criteria": [str(i) for i in range(11)]}},
        }))


# ------------------------------------------------------------------------------- encoding


def _response():
    from strands_decider.schema import ChoiceAnswer, NoulAnswer, SystemOneResponse, Usage

    return SystemOneResponse(
        model="strands-decider-triton",
        answers={
            "needs_escalation": NoulAnswer(noul=0.8751),
            "sentiment": ChoiceAnswer(choice="frustrated",
                                      probabilities={"calm": 0.1, "frustrated": 0.8,
                                                     "angry": 0.1},
                                      confidence=0.7),
        },
        usage=Usage(input_tokens=86, output_tokens=2),
    )


def test_response_carries_the_same_keys_as_the_http_server():
    payload = json.loads(encode_response(_response(), latency_ms=140.0345))
    assert set(payload) == {"model", "answers", "usage", "latency_ms"}
    assert payload["latency_ms"] == 140.03, "latency is rounded to 2dp like the FastAPI server"
    assert payload["answers"]["needs_escalation"] == {"type": "noul", "noul": 0.8751}


def test_latency_is_omitted_when_not_supplied():
    assert "latency_ms" not in json.loads(encode_response(_response()))


def test_structured_probability_output_survives_the_round_trip():
    """The point of the whole system: probabilities and confidence, not text."""
    payload = json.loads(encode_response(_response()))
    sentiment = payload["answers"]["sentiment"]
    assert sentiment["choice"] == "frustrated"
    assert sentiment["probabilities"]["frustrated"] == 0.8
    assert sentiment["confidence"] == 0.7
    assert abs(sum(sentiment["probabilities"].values()) - 1.0) < 1e-9


# --------------------------------------------------------------------------------- errors


def test_error_body_is_distinguishable_from_a_response():
    assert is_error(encode_error("bad option count"))
    assert not is_error(encode_response(_response()))


def test_error_body_names_the_problem():
    payload = json.loads(encode_error("choice requires at least 2 options"))
    assert payload["error"]["type"] == "invalid_request"
    assert "at least 2" in payload["error"]["message"]


def test_internal_errors_are_tagged_differently_from_caller_errors():
    """A broken deployment and a malformed payload must not look the same to a client."""
    caller = json.loads(encode_error("bad json"))
    internal = json.loads(encode_error("cuda oom", kind="internal"))
    assert caller["error"]["type"] != internal["error"]["type"]


def test_is_error_tolerates_non_json():
    assert not is_error(b"\x00\x01 not json")


# ------------------------------------------------------------------- KServe v2 envelope
#
# SageMaker's /invocations on the Triton DLC proxies to Triton's v2 `infer`, so the System
# One body has to be wrapped. These tests pin the wrapping, because getting it wrong is an
# opaque HTTP 500 from Triton's JSON parser rather than anything the backend can report.


def test_v2_request_wraps_the_payload():
    from decider_triton.wire import encode_v2_request

    env = encode_v2_request(SMOKE_BODY)
    assert list(env) == ["inputs"]
    inp = env["inputs"][0]
    assert inp["name"] == "REQUEST_JSON"
    assert inp["datatype"] == "BYTES"
    # [batch, 1]. The model sets max_batch_size > 0, so Triton prepends the batch
    # dimension to `dims: [1]`; sending [1] is rejected.
    assert inp["shape"] == [1, 1]
    assert json.loads(inp["data"][0])["state"] == SMOKE_BODY["state"]


def test_v2_request_accepts_a_prebuilt_string():
    from decider_triton.wire import encode_v2_request

    env = encode_v2_request(json.dumps(SMOKE_BODY))
    assert json.loads(env["inputs"][0]["data"][0]) == SMOKE_BODY


def test_v2_response_unwraps_to_the_answer_body():
    from decider_triton.wire import decode_v2_response, encode_response

    inner = encode_response(_response(), latency_ms=41.0).decode()
    env = {"model_name": "decider",
           "outputs": [{"name": "RESPONSE_JSON", "datatype": "BYTES",
                        "shape": [1, 1], "data": [inner]}]}
    body = decode_v2_response(env)
    assert body["answers"]["needs_escalation"]["noul"] == 0.8751
    assert body["latency_ms"] == 41.0


def test_v2_response_surfaces_a_triton_error():
    from decider_triton.wire import decode_v2_response

    with pytest.raises(ValueError, match="Triton returned an error"):
        decode_v2_response({"error": "Unable to parse 'inputs'"})


@pytest.mark.parametrize("env,expect", [
    ({}, "no 'outputs'"),
    ({"outputs": []}, "no 'outputs'"),
    ({"outputs": [{"name": "WRONG", "data": ["{}"]}]}, "no RESPONSE_JSON"),
    ({"outputs": [{"name": "RESPONSE_JSON", "data": []}]}, "carried no data"),
])
def test_v2_response_refuses_a_malformed_envelope(env, expect):
    """Raises rather than returning {} -- an empty dict reads as 'no answers'."""
    from decider_triton.wire import decode_v2_response

    with pytest.raises(ValueError, match=expect):
        decode_v2_response(env)


def test_v2_round_trip_through_the_backend_output_shape():
    """The envelope the backend's np.array([body], dtype=object) becomes over the wire."""
    from decider_triton.wire import decode_v2_response, encode_response

    inner = encode_response(_response())
    env = {"outputs": [{"name": "RESPONSE_JSON", "datatype": "BYTES",
                        "shape": [1, 1], "data": [inner.decode()]}]}
    assert set(decode_v2_response(env)["answers"]) == {"needs_escalation", "sentiment"}
