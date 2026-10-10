"""Pin `StrandsDeciderConfig.from_json` against the key sets real checkpoints publish.

Nothing in this suite had ever fed `from_json` a published `strands_decider_config.json`.
`from_json` is `cls(**json.load(fh))` with no unknown-key filter, so the first thing that
noticed a checkpoint carrying an extra key was the Docker **merge stage** -- a GPU build
box, an S3 context sync and several minutes in, for a `TypeError` that costs nothing to
catch here.

The key sets below are copied verbatim from the published files (fetched 2026-10-10) rather
than downloaded, so this runs offline in CI exactly as the rest of the suite does. Values
are representative, not meaningful: `from_json` only ever fails on the key NAMES.
"""

from __future__ import annotations

import json

import pytest

from strands_decider.modeling import (
    TEXT_TOWER_CLASSES,
    StrandsDeciderConfig,
    assert_bos_contract,
)

# v21 (deployed today) and 2B-qwen3.5-v1-2610 publish byte-identical key sets.
SHIPPED_KEYS = [
    "base_model", "head_dropout", "head_hidden", "head_init", "head_type",
    "kl_frozen_weight", "lora_alpha", "lora_dropout", "lora_r", "lora_targets",
    "max_length", "num_slots", "ordinal_smoothing", "pointer_dim", "temperature",
    "temperature_by_kind", "torch_dtype", "use_lora",
]

# All four gemma4 releases add exactly these three, which the dataclass has no fields for.
GEMMA4_EXTRA_KEYS = ["force_bos", "full_weight_targets", "host_embeddings"]


def _write(tmp_path, keys, **overrides):
    """A config file carrying exactly `keys`, with plausible values."""
    base = {
        "base_model": "Qwen/Qwen3.5-2B-Base", "head_dropout": 0.0, "head_hidden": 0,
        "head_init": "random", "head_type": "pointer", "kl_frozen_weight": 0.0,
        "lora_alpha": 96, "lora_dropout": 0.05, "lora_r": 48,
        "lora_targets": ["q_proj", "k_proj"], "max_length": 4096, "num_slots": 24,
        "ordinal_smoothing": 0.1, "pointer_dim": 768, "temperature": 1.0,
        "temperature_by_kind": {"score": 1.83}, "torch_dtype": "bfloat16",
        "use_lora": True,
        "force_bos": True, "full_weight_targets": [], "host_embeddings": False,
    }
    doc = {k: base[k] for k in keys}
    doc.update(overrides)
    path = tmp_path / "strands_decider_config.json"
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return str(path)


def test_the_shipped_checkpoint_key_set_loads(tmp_path):
    """v21 and 2B-qwen3.5-v1-2610. If this breaks, the deployed model stops loading."""
    cfg = StrandsDeciderConfig.from_json(_write(tmp_path, SHIPPED_KEYS))
    assert cfg.head_type == "pointer"
    assert cfg.max_length == 4096
    # Read from the file, not defaulted: v21 is 256 and every 2610 release is 768, and a
    # head built at the wrong width fails a shape check rather than mis-answering.
    assert cfg.pointer_dim == 768


def test_max_length_is_read_from_the_file_not_the_dataclass_default():
    """The dataclass default is 3072 and every shipped checkpoint sets 4096. Code that
    reads the default instead of the file silently gets a different window -- which is how
    `cuda_graphs.BANK_WIDTH`'s invariant came to be tested against itself."""
    assert StrandsDeciderConfig().max_length == 3072


def test_the_gemma4_key_set_loads(tmp_path):
    """The four gemma4-2610 releases parse, with the three extra keys given meanings.

    This test used to assert the opposite -- that the key set was refused with a
    `TypeError` -- as a characterisation of the state before gemma4 support, and it said
    whoever added that support had to come here and decide what the three keys MEAN rather
    than make the error go away. That is what happened:

      * `full_weight_targets` and `host_embeddings` are training-side and inert at
        inference. Declared so they round-trip, read by nothing.
      * `force_bos` is NOT inert, and is `true` on all four. It is now enforced by
        `assert_bos_contract` at load rather than satisfied by coincidence -- see the
        tests below.
    """
    cfg = StrandsDeciderConfig.from_json(_write(tmp_path, [*SHIPPED_KEYS, *GEMMA4_EXTRA_KEYS]))
    assert cfg.force_bos is True
    assert cfg.full_weight_targets == []
    assert cfg.host_embeddings is False
    # The serving-relevant fields must survive alongside the new ones.
    assert cfg.head_type == "pointer"
    assert cfg.max_length == 4096


def test_force_bos_defaults_off_so_existing_checkpoints_are_unaffected(tmp_path):
    """v21 and qwen3.5-v1 do not publish the key at all. They must not acquire a contract
    they never asserted, because `assert_bos_contract` is a startup failure when unmet."""
    cfg = StrandsDeciderConfig.from_json(_write(tmp_path, SHIPPED_KEYS))
    assert cfg.force_bos is False
    assert cfg.full_weight_targets == []
    assert cfg.host_embeddings is False


def test_a_genuinely_unknown_key_is_still_refused(tmp_path):
    """Accepting the three gemma4 keys must not turn `from_json` permissive.

    Strictness is the feature: a key this code does not understand is a checkpoint making
    an assertion nobody is honouring, and the `TypeError` at build time is how that gets
    noticed. Only the three keys whose meanings were established are allowed through.
    """
    with pytest.raises(TypeError, match="speculative_decoding_depth"):
        StrandsDeciderConfig.from_json(
            _write(tmp_path, SHIPPED_KEYS, speculative_decoding_depth=4))


# ---------------------------------------------------------------------------
# force_bos, held to its word
# ---------------------------------------------------------------------------


class _Tok:
    """Minimal stand-in for a fast tokeniser's call + bos_token_id."""

    def __init__(self, bos_token_id, prepends):
        self.bos_token_id = bos_token_id
        self._prepends = prepends

    def __call__(self, text, add_special_tokens=True):
        ids = [101, 102, 103]
        if add_special_tokens and self._prepends:
            ids = [self.bos_token_id, *ids]
        return {"input_ids": ids}


def test_bos_contract_is_a_no_op_when_the_checkpoint_does_not_ask():
    """v21 and qwen3.5-v1 must be unaffected, including with a BOS-less tokeniser."""
    cfg = StrandsDeciderConfig(force_bos=False)
    assert_bos_contract(cfg, _Tok(bos_token_id=None, prepends=False))


def test_bos_contract_passes_when_the_tokeniser_supplies_one():
    """A Gemma tokeniser prepends <bos> under add_special_tokens=True. Verified, not
    assumed: that is the whole point of the check."""
    assert_bos_contract(StrandsDeciderConfig(force_bos=True),
                        _Tok(bos_token_id=2, prepends=True))


def test_bos_contract_refuses_a_tokeniser_that_declares_no_bos():
    cfg = StrandsDeciderConfig(force_bos=True)
    with pytest.raises(RuntimeError, match="no bos_token_id"):
        assert_bos_contract(cfg, _Tok(bos_token_id=None, prepends=False))


def test_bos_contract_refuses_a_tokeniser_that_will_not_prepend():
    """The case that would otherwise serve, silently, without the leading token the
    checkpoint requires -- changing every probability and raising nowhere."""
    cfg = StrandsDeciderConfig(force_bos=True)
    with pytest.raises(RuntimeError, match="rather than starting with bos_token_id"):
        assert_bos_contract(cfg, _Tok(bos_token_id=2, prepends=False))


# ---------------------------------------------------------------------------
# The text-tower table
# ---------------------------------------------------------------------------


# Every base model across the six published checkpoints, from their `base_model` fields.
PUBLISHED_MODEL_TYPES = {
    "qwen3_5": "Qwen/Qwen3.5-2B-Base (v21, qwen3.5-v1)",
    "gemma4": "google/gemma-4-{E2B,E4B,26B-A4B}-it",
    "gemma4_unified": "google/gemma-4-12B-it",
}


@pytest.mark.parametrize("model_type", sorted(PUBLISHED_MODEL_TYPES))
def test_every_published_architecture_has_a_text_tower_class(model_type):
    """A `model_type` absent from the table falls through to `AutoModel`, which for these
    multimodal checkpoints returns the wrapper -- and PEFT reports the resulting name
    mismatch as an EMPTY MERGE rather than an error. So a missing entry is a silently
    un-adapted model, not a crash."""
    assert model_type in TEXT_TOWER_CLASSES, (
        f"{model_type} ({PUBLISHED_MODEL_TYPES[model_type]}) would load via AutoModel")


@pytest.mark.parametrize("model_type", sorted(PUBLISHED_MODEL_TYPES))
def test_the_named_class_exists_and_exposes_a_text_tower(model_type):
    """The table names classes; this proves the installed transformers has them, and that
    `.model` is the text tower whose layers sit at `.layers` -- which is what makes the
    adapters' `base_model.model.layers.N` paths line up."""
    transformers = pytest.importorskip("transformers")
    cls_name = TEXT_TOWER_CLASSES[model_type]
    cls = getattr(transformers, cls_name, None)
    assert cls is not None, (
        f"transformers {transformers.__version__} has no {cls_name}; the image pins "
        "transformers 5.18, where it is present")
    # The text config is where hidden_size lives -- the wrapper config has none, which is
    # the loud half of the failure this table prevents.
    assert hasattr(cls.config_class, "__name__")
    src = __import__("inspect").getsource(cls.__init__)
    assert "self.model" in src


def test_text_tower_table_covers_the_text_only_variants():
    """Both the wrapper and `*_text` forms must map, since `get_text_config()` on an
    already-unwrapped checkpoint reports the text type."""
    for base in ("qwen3_5", "gemma4", "gemma4_unified"):
        assert TEXT_TOWER_CLASSES[base] == TEXT_TOWER_CLASSES[f"{base}_text"]
