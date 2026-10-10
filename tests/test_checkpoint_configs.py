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

from strands_decider.modeling import StrandsDeciderConfig

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


def test_the_gemma4_key_set_is_refused_loudly(tmp_path):
    """The four gemma4-2610 releases cannot be loaded, and that refusal is correct.

    This is a *characterisation* test, not an endorsement: it exists so that whoever adds
    gemma4 support has to come here and decide what the three keys MEAN, rather than
    making the error go away. In particular:

      * `full_weight_targets` and `host_embeddings` are training-side and inert at
        inference -- safe to accept and ignore.
      * `force_bos` is NOT. It is `true` on all four gemma4 configs, and nothing in this
        repo reads it. The state happens to get a BOS because `infer.py` tokenises it with
        `add_special_tokens=True`, but the question suffix is deliberately tokenised with
        `add_special_tokens=False`. So adding these as ignored fields would make the
        configs load while quietly dropping a flag the checkpoint asserts.

    The 2B-qwen3.5-v1-2610 publisher stripped all three *because each held its off value*
    and said so in that repo's README. For gemma4 one of them does not, so the same move
    is not safe. Confirm the flag's semantics against the training code first.
    """
    path = _write(tmp_path, SHIPPED_KEYS + GEMMA4_EXTRA_KEYS)
    with pytest.raises(TypeError, match=r"full_weight_targets|force_bos|host_embeddings"):
        StrandsDeciderConfig.from_json(path)


@pytest.mark.parametrize("extra", GEMMA4_EXTRA_KEYS)
def test_each_unknown_key_is_refused_individually(tmp_path, extra):
    """Pins which keys are unknown one at a time, so adding support for one does not
    silently appear to add support for all three."""
    with pytest.raises(TypeError, match=extra):
        StrandsDeciderConfig.from_json(_write(tmp_path, [*SHIPPED_KEYS, extra]))
