"""The correctness gate must be able to tell the published checkpoints apart.

`tools/reference_check.py` is the project's governing rule in executable form: zero
decision mismatches before any throughput number is trusted. With one published checkpoint
a decision-only gate was sufficient. With six it is not, and the failure is silent in the
worst direction -- it reports PASS.

Measured, not hypothesised. The numbers in `SERVED_QWEN35` are the actual answers from a
`v25-qwen3.5-v1` endpoint on `ml.g6.4xlarge`, 2026-10-10, and the old gate printed
`PASS: 0 decision mismatch(es) against the published v21 values` against them.

These tests need no network and no model: they drive `check()` and the identification
scoring directly over recorded responses.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ref():
    return _load(ROOT / "tools" / "reference_check.py", "ref_check")


# Observed from the live endpoint. Keyed the way `body["answers"]` arrives.
SERVED_QWEN35 = {
    "noul_urgency": {
        "urgency": {"type": "noul", "noul": 0.8563},
    },
    "choice_team": {
        "team": {"type": "choice", "choice": "billing", "confidence": 0.6610,
                 "probabilities": {"billing": 0.7740, "sales": 0.1198, "retail": 0.1062}},
    },
    "score_frustration": {
        "frustration": {"type": "score", "score": 1.0325, "confidence": 0.5106,
                        "probabilities": {"0": 0.1961, "1": 0.5753, "2": 0.2286}},
    },
}

TOLERANCE = 0.02


def _worst(ref, served, model):
    """Worst delta of `served` against `model`'s published values -- what --identify ranks."""
    worst = 0.0
    for case_name, expectations in ref.REFERENCES[model].items():
        for qname, expect in expectations.items():
            answer = served[case_name][qname]
            worst = max(worst, max(ref.deltas(answer, expect), default=0.0))
    return worst


def test_all_six_checkpoints_share_the_same_decisions(ref):
    """The premise of the whole problem, pinned.

    If this ever fails, the decision gate HAS become discriminating and the probability
    gate could in principle be relaxed back to an advisory. Until then it cannot.
    """
    decisions = set()
    for references in ref.REFERENCES.values():
        noul = references["noul_urgency"]["urgency"]["noul"]
        choice = references["choice_team"]["team"]["choice"]
        score = references["score_frustration"]["frustration"]["score"]
        decisions.add((noul >= 0.5, choice, round(score)))
    assert len(decisions) == 1, (
        f"checkpoints no longer agree on all three decisions: {decisions}")
    assert decisions == {(True, "billing", 1)}


def test_serving_qwen35_passes_against_its_own_values(ref):
    """The gate must not cry wolf: the right model against its own numbers passes."""
    for case_name, expectations in ref.REFERENCES["qwen3.5-v1"].items():
        for qname, expect in expectations.items():
            answer = SERVED_QWEN35[case_name][qname]
            decision_ok, probs_ok, _ = ref.check(answer, expect, TOLERANCE)
            assert decision_ok, f"{case_name}/{qname} decision"
            assert probs_ok, f"{case_name}/{qname} probabilities"
    assert _worst(ref, SERVED_QWEN35, "qwen3.5-v1") <= 0.005, (
        "served values should match their own published figures far inside tolerance")


def test_serving_qwen35_fails_against_v21_values(ref):
    """The regression this file exists for.

    The old gate reported PASS here, because the decisions matched. The decisions STILL
    match -- that is the point -- so the probability gate has to be the thing that fails.
    """
    decisions_all_matched = True
    probability_failures = 0
    for case_name, expectations in ref.REFERENCES["v21"].items():
        for qname, expect in expectations.items():
            answer = SERVED_QWEN35[case_name][qname]
            decision_ok, probs_ok, _ = ref.check(answer, expect, TOLERANCE)
            decisions_all_matched &= decision_ok
            probability_failures += 0 if probs_ok else 1

    assert decisions_all_matched, (
        "the premise changed: v21's decisions no longer match what qwen3.5-v1 serves, so "
        "this no longer reproduces the silent-pass bug")
    assert probability_failures > 0, (
        "serving qwen3.5-v1 while checking against v21 MUST fail. This is the exact "
        "condition that printed PASS on 2026-10-10.")


def test_identify_picks_the_served_checkpoint(ref):
    """--identify must name the right one, and beat the runner-up by a clear margin."""
    scored = sorted((_worst(ref, SERVED_QWEN35, m), m) for m in ref.REFERENCES)
    best_delta, best_model = scored[0]
    runner_up_delta = scored[1][0]

    assert best_model == "qwen3.5-v1"
    assert best_delta <= TOLERANCE
    # The margin must exceed the best match's own error, which is the condition
    # `identify` uses to distinguish a real identification from a coin flip.
    assert runner_up_delta - best_delta > best_delta


@pytest.mark.parametrize("model", ["v21", "qwen3.5-v1", "E2B-gemma4-v1",
                                   "E4B-gemma4-v1", "12B-gemma4-v1",
                                   "26B-A4B-gemma4-v1"])
def test_every_checkpoint_is_separable_from_every_other(model, ref):
    """No two published checkpoints may be within tolerance of each other.

    If a future release lands inside 0.02 of an existing one, `--identify` silently stops
    discriminating and the gate stops being able to catch a mis-deploy. Better to fail
    here, when the reference values are added, than at 3am against an endpoint.
    """
    mine = ref.REFERENCES[model]
    # Reconstruct what this checkpoint would serve, exactly, from its own published values.
    served = {}
    for case_name, expectations in mine.items():
        served[case_name] = {}
        for qname, expect in expectations.items():
            answer = {"type": expect["kind"]}
            if expect["kind"] == "noul":
                answer["noul"] = expect["noul"]
            else:
                answer["confidence"] = expect["confidence"]
                if expect["kind"] == "score":
                    answer["score"] = expect["score"]
            if expect.get("probabilities"):
                answer["probabilities"] = dict(expect["probabilities"])
            served[case_name][qname] = answer

    assert _worst(ref, served, model) == pytest.approx(0.0, abs=1e-9)
    for other in ref.REFERENCES:
        if other == model:
            continue
        assert _worst(ref, served, other) > TOLERANCE, (
            f"{model} and {other} are within tolerance of each other, so the gate can no "
            f"longer tell a {model} deployment from a {other} one")


def test_reference_values_cover_every_case(ref):
    """Every checkpoint must answer every case, or the gate quietly checks less."""
    case_names = {c["name"] for c in ref.CASES}
    for model, references in ref.REFERENCES.items():
        assert set(references) == case_names, f"{model} does not cover every case"
        assert model in ref.HUB_IDS, f"{model} has no Hub id for the banner"
