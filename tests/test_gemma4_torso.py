"""The gemma4 text tower must load through `load_text_tower` and actually forward.

`test_checkpoint_configs.py` pins the TABLE -- that each published architecture maps to a
class, and that the class exists. This file pins the BEHAVIOUR: that a tower obtained the
way `load_text_tower` obtains it runs a forward pass and produces the hidden state the
pointer head reads.

Everything here is CPU-only, needs no network and no checkpoint. The models are a few
thousand parameters, shaped to mirror the real architectures rather than to be
representative of them: 4-sliding+1-full layer types, per-layer embeddings, and a
KV-shared tail, because those three features are what make gemma4 different from the
Qwen3.5 hybrid this engine was built for.

Two questions these answer, both of which were open after the port landed:

  * does the tower self-supply `per_layer_inputs`? E2B and E4B set
    `hidden_size_per_layer_input: 256`, and `Gemma4TextModel.forward` takes
    `per_layer_inputs` as an OPTIONAL argument. If it had to be passed, the loader would
    be returning a tower that cannot be driven. It does not: it recomputes them from
    `input_ids`.
  * is `shared_kv_states` a silent hazard? It is a second cache channel living entirely
    outside the `Cache` object, read by 20 of E2B's 35 layers, and nothing in this repo
    knows it exists. The concern was that a wrong layer arrangement would read the wrong
    keys and answer plausibly. It does not: it raises `KeyError`.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from strands_decider.modeling import load_text_tower  # noqa: E402

# Real E2B: 35 layers, num_kv_shared_layers 20, so layers 0-14 are NOT shared and the
# 4-sliding+1-full cycle puts full_attention at 4, 9 and 14 -- inside that prefix. This
# mirrors the shape at 10 layers: full at 4 and 9, shared tail of 4.
E2B_SHAPED = dict(
    vocab_size=256, hidden_size=64, num_hidden_layers=10,
    num_attention_heads=2, num_key_value_heads=1, head_dim=32,
    intermediate_size=128, sliding_window=8,
    layer_types=["sliding_attention"] * 4 + ["full_attention"]
    + ["sliding_attention"] * 4 + ["full_attention"],
    vocab_size_per_layer_input=256, hidden_size_per_layer_input=16,
    num_kv_shared_layers=4,
)


def _tiny_lm(**overrides):
    cfg = transformers.Gemma4TextConfig(**{**E2B_SHAPED, **overrides})
    torch.manual_seed(0)
    return transformers.Gemma4ForCausalLM(cfg)


def test_the_text_tower_forwards_with_per_layer_embeddings():
    """The tower must be drivable as returned, with PLE and KV sharing both on."""
    torso = _tiny_lm().model
    assert type(torso).__name__ == "Gemma4TextModel"
    assert torso.config.hidden_size_per_layer_input == 16
    assert torso.config.num_kv_shared_layers == 4

    ids = torch.randint(0, 256, (2, 12))
    with torch.no_grad():
        out = torso(input_ids=ids, attention_mask=torch.ones_like(ids))
    h = out.last_hidden_state
    assert h.shape == (2, 12, 64)
    assert torch.isfinite(h).all(), "non-finite hidden states"
    assert float(h.abs().mean()) > 0, "all-zero hidden states would pass a shape check"


def test_the_last_token_hidden_state_is_usable_by_the_pointer_head():
    """`pool_last_token` reads the final unmasked position; that is the whole readout."""
    torso = _tiny_lm().model
    ids = torch.randint(0, 256, (3, 9))
    with torch.no_grad():
        h = torso(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
    pooled = h[:, -1, :]
    assert pooled.shape == (3, torso.config.hidden_size)
    assert torch.isfinite(pooled).all()


def test_hidden_size_is_readable_from_the_towers_own_config():
    """The loud half of the wrapper bug: `AutoModel` returns a config with no top-level
    `hidden_size`, so `StrandsDeciderModel.hidden_size` raises and no head can be built.
    The tower's config must carry it."""
    torso = _tiny_lm().model
    assert int(torso.config.hidden_size) == 64
    for attr in ("hidden_size",):
        assert getattr(torso.config, attr, None) is not None


def test_shared_kv_states_raises_rather_than_reading_the_wrong_keys():
    """Pins that the second cache channel fails LOUDLY on a bad layer arrangement.

    A KV-shared layer reads `shared_kv_states[self.layer_type]`, which an earlier
    non-shared layer of the same type must have populated. Here the shared tail contains a
    `full_attention` layer while the non-shared prefix is all sliding, so there is nothing
    to read.

    The failure being a `KeyError` is the point. The migration's worst case was that this
    channel would quietly serve keys computed from the question suffix alone -- a plausible
    distribution at HTTP 200, for a fifth to a half of the network. It does not do that.
    """
    lm = _tiny_lm(
        num_hidden_layers=4,
        layer_types=["sliding_attention", "sliding_attention",
                     "sliding_attention", "full_attention"],
        num_kv_shared_layers=2,
    )
    ids = torch.randint(0, 256, (1, 8))
    with pytest.raises(KeyError, match="full_attention"), torch.no_grad():
        lm.model(input_ids=ids, attention_mask=torch.ones_like(ids))


def test_load_text_tower_round_trips_a_saved_gemma4_checkpoint(tmp_path):
    """The real loader path, end to end, without the network.

    `save_pretrained` writes `model_type: gemma4_text`, which the table maps to
    `Gemma4ForCausalLM`; `load_text_tower` must therefore return the TEXT tower -- layers
    at `.layers`, config carrying `hidden_size` -- and not an `AutoModel` wrapper.
    """
    from transformers import AutoConfig, AutoModel

    _tiny_lm().save_pretrained(tmp_path, safe_serialization=True)
    base_cfg = AutoConfig.from_pretrained(tmp_path)
    assert base_cfg.model_type == "gemma4_text"

    torso = load_text_tower(str(tmp_path), base_cfg, AutoModel, dtype=torch.float32)
    assert type(torso).__name__ == "Gemma4TextModel"
    assert int(torso.config.hidden_size) == 64
    # The adapters target `base_model.model.layers.N.…`; this is the `.layers.N` half.
    names = {n for n, _ in torso.named_modules()}
    assert "layers.0.self_attn.q_proj" in names
    assert "layers.0.mlp.down_proj" in names

    ids = torch.randint(0, 256, (1, 10))
    with torch.no_grad():
        h = torso(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
    assert h.shape == (1, 10, 64)
    assert torch.isfinite(h).all()
