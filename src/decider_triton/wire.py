"""The wire format between Triton and the Decider engine.

One JSON string in, one JSON string out, byte-compatible with `POST /v1/systemone`. Kept
in its own module, with no Triton and no torch imports, so the contract is unit-testable
on a laptop -- the Triton python backend can only be exercised inside a `tritonserver`
process, and a format bug found there costs a 10 GB image round trip to fix.

Why JSON rather than a tensor per field. A System One request is a state plus an arbitrary
dict of named questions, each carrying its own option list of 2-255 entries whose labels
are defined *by the request* rather than by the model (that genericity is the architecture's
whole point -- see `prompting.py`). There is no fixed tensor schema for that. Encoding it as
one opaque string also keeps every request the same shape, `dims: [1]`, which is what lets
Triton's dynamic batcher coalesce requests at all.

Error contract, chosen to match the FastAPI server so a client cannot tell the two apart:

    bad JSON / schema violation / option count   -> 400-equivalent, `error` in the body
    anything else                                -> re-raised for Triton to report

A caller error must not look like a server error, because the thing that produces one is a
malformed request and the thing that produces the other is a broken deployment.
"""

from __future__ import annotations

import json
from typing import Any

# Matches the FastAPI handler, which adds this to the response body rather than a header.
LATENCY_KEY = "latency_ms"


class RequestDecodeError(ValueError):
    """The bytes were not a valid SystemOneRequest. A caller error, not a server fault."""


def decode_request(raw: bytes | bytearray | memoryview | str) -> Any:
    """Bytes from a Triton TYPE_STRING tensor -> a validated `SystemOneRequest`.

    Triton hands TYPE_STRING through as `bytes` (numpy object arrays of bytes), so this
    accepts both and leaves `str` working for tests.
    """
    from strands_decider.schema import SystemOneRequest

    if isinstance(raw, (bytes, bytearray, memoryview)):
        try:
            text = bytes(raw).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RequestDecodeError(f"request was not valid UTF-8: {exc}") from exc
    else:
        text = raw

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RequestDecodeError(f"request was not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise RequestDecodeError(
            f"request must be a JSON object, got {type(payload).__name__}"
        )

    try:
        return SystemOneRequest.model_validate(payload)
    except Exception as exc:
        # pydantic's ValidationError message names the offending field, which is the
        # useful part for a caller fixing their payload.
        raise RequestDecodeError(f"request did not validate: {exc}") from exc


def encode_response(response: Any, latency_ms: float | None = None) -> bytes:
    """A `SystemOneResponse` -> JSON bytes, shaped exactly like `/v1/systemone`'s body."""
    payload = response.model_dump()
    if latency_ms is not None:
        payload[LATENCY_KEY] = round(latency_ms, 2)
    return json.dumps(payload).encode("utf-8")


def encode_error(message: str, *, kind: str = "invalid_request") -> bytes:
    """A caller error, as a body rather than a Triton-level failure.

    Triton reports an exception from `execute()` as a server error for the whole request,
    which would make a client's own malformed payload indistinguishable from the model
    being down. Returning a body keeps that distinction, matching the FastAPI server's
    HTTP 422 for the same inputs.
    """
    return json.dumps({"error": {"type": kind, "message": message}}).encode("utf-8")


def is_error(raw: bytes | str) -> bool:
    """True if `raw` is an error body from `encode_error`. For tests and clients."""
    text = bytes(raw).decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and "error" in payload


# --------------------------------------------------------------------------------------
# KServe v2 envelope.
#
# CORRECTION, found by running it: SageMaker's `/invocations` on the Triton DLC is a thin
# proxy to Triton's KServe v2 `POST /v2/models/<name>/infer`. It does NOT accept a raw
# System One body -- that returns
#
#     {"error": "Unable to parse 'inputs': attempt to access non-existing object member
#                'inputs'"}
#
# So the Triton deployable is NOT drop-in for `/v1/systemone` clients, which is what this
# project claimed until it was tested. The payload still travels as the same JSON, but it
# has to be wrapped in a tensor envelope. These two helpers are that wrapping, kept here so
# there is exactly one implementation of it shared by the client, the tests and the
# benchmark harness.
# --------------------------------------------------------------------------------------

INPUT_NAME = "REQUEST_JSON"
OUTPUT_NAME = "RESPONSE_JSON"


def encode_v2_request(payload: dict | str) -> dict:
    """A System One request -> a KServe v2 infer body.

    `shape` is `[1, 1]`: the model declares `max_batch_size > 0`, so Triton prepends a
    batch dimension to the configured `dims: [1]`. Sending `[1]` is rejected.
    """
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return {
        "inputs": [{
            "name": INPUT_NAME,
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [body],
        }]
    }


def decode_v2_response(response: dict) -> dict:
    """A KServe v2 infer response -> the System One answer body.

    Raises `ValueError` when the envelope is not shaped as expected, rather than returning
    a partial answer: a caller that silently got `{}` back would treat it as "no answers".
    """
    if "error" in response and "outputs" not in response:
        raise ValueError(f"Triton returned an error: {response['error']}")
    outputs = response.get("outputs")
    if not outputs:
        raise ValueError(f"no 'outputs' in the Triton response: {sorted(response)}")
    for out in outputs:
        if out.get("name") == OUTPUT_NAME:
            data = out.get("data") or []
            if not data:
                raise ValueError(f"{OUTPUT_NAME} carried no data")
            item = data[0]
            if isinstance(item, (bytes, bytearray)):
                item = bytes(item).decode("utf-8")
            return json.loads(item)
    raise ValueError(
        f"no {OUTPUT_NAME} output; got {[o.get('name') for o in outputs]}"
    )
