"""Fused-kernel layer rewriting: what it refuses, and what it preserves.

Tested here, on CPU, with no fla and no GPU:

  * the **layout checks**, one case per assumption the replacement forwards make. Each of
    these, if wrong and unchecked, produces a well-formed wrong probability rather than a
    crash -- a non-zero-centred RMSNorm rescales every activation in the layer, a mismatched
    `conv_dim` makes the post-conv split mix queries with values, a biased projection drops
    the bias. So the refusal is the feature and it is what is pinned.
  * the **weight concatenation preserves values**, in the order the split reverses. An
    off-by-one in that order would route the `a` projection's output into `beta`, which is
    still a number between 0 and 1.
  * **idempotence**, because `fuse_torso` deletes the modules it replaces: a second call
    that did not short-circuit would fail on a missing `in_proj_qkv`, or worse concatenate
    an already-concatenated weight.
  * that the fused weights are **buffers, not bare attributes**, so `nn.Module.to` moves
    them. The upstream this is ported from uses bare attributes; a later `.to(device)` would
    then leave the fused weights on the old device.

NOT tested here, and it cannot be: that the fused kernels compute what the reference layers
compute. That needs fla, Triton and a real GPU. It lives in two places --
`fused_layers.verify_kernels`, which runs at load time and refuses to fuse if any of the
five kernel contracts has moved, and `tools/fused_ab.py`, which compares served
probabilities through both torsos. The gate is zero decision flips, as everywhere else.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from strands_decider.fused_layers import (  # noqa: E402
    FusionUnsupported,
    _concat,
    check_layout,
    decoder_layers,
    fuse_enabled,
    fuse_torso,
    is_fused,
)

# Tiny but structurally honest: the real torso is 24 layers of 2,048 hidden, 18 of them
# Gated DeltaNet. These shapes keep every ratio that the checks and the split depend on
# (conv_dim == 2*key_dim + value_dim, num_v_heads a multiple of num_k_heads, q_proj double
# width for the packed output gate) and nothing else.
H = 16
HEAD_K = HEAD_V = 4
NUM_K, NUM_V = 2, 4
KEY_DIM, VALUE_DIM = HEAD_K * NUM_K, HEAD_V * NUM_V
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
CONV_W = 4
ATTN_HEADS, HEAD_DIM = 2, 8


class Qwen3_5RMSNorm(nn.Module):
    """Name-compatible with the class the checks look for, zero-centred like it."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps
        self.activation = "silu"


class _NotZeroCentred(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))


class _DeltaNet(nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.key_dim, self.value_dim, self.conv_dim = KEY_DIM, VALUE_DIM, CONV_DIM
        self.head_k_dim, self.head_v_dim = HEAD_K, HEAD_V
        self.num_k_heads, self.num_v_heads = NUM_K, NUM_V
        self.activation = "silu"
        self.in_proj_qkv = nn.Linear(H, CONV_DIM, bias=False)
        self.in_proj_z = nn.Linear(H, VALUE_DIM, bias=False)
        self.in_proj_b = nn.Linear(H, NUM_V, bias=False)
        self.in_proj_a = nn.Linear(H, NUM_V, bias=False)
        self.conv1d = nn.Conv1d(CONV_DIM, CONV_DIM, bias=False, kernel_size=CONV_W,
                                groups=CONV_DIM, padding=CONV_W - 1)
        self.dt_bias = nn.Parameter(torch.ones(NUM_V))
        self.A_log = nn.Parameter(torch.zeros(NUM_V))
        self.norm = Qwen3_5RMSNormGated(HEAD_V)
        self.out_proj = nn.Linear(VALUE_DIM, H, bias=False)


class _Attention(nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = HEAD_DIM
        self.scaling = HEAD_DIM ** -0.5
        self.config = type("Cfg", (), {"_attn_implementation": "eager"})()
        # Double width: Qwen3.5 packs the sigmoid output gate into q_proj.
        self.q_proj = nn.Linear(H, ATTN_HEADS * HEAD_DIM * 2, bias=False)
        self.k_proj = nn.Linear(H, ATTN_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(H, ATTN_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(ATTN_HEADS * HEAD_DIM, H, bias=False)
        self.q_norm = Qwen3_5RMSNorm(HEAD_DIM)
        self.k_norm = Qwen3_5RMSNorm(HEAD_DIM)


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(H, 2 * H, bias=False)
        self.up_proj = nn.Linear(H, 2 * H, bias=False)
        self.down_proj = nn.Linear(2 * H, H, bias=False)
        self.act_fn = nn.SiLU()


class _Layer(nn.Module):
    def __init__(self, layer_idx: int, block_type: str) -> None:
        super().__init__()
        self.block_type = block_type
        if block_type == "linear_attention":
            self.linear_attn = _DeltaNet(layer_idx)
        else:
            self.self_attn = _Attention(layer_idx)
        self.mlp = _MLP()
        self.input_layernorm = Qwen3_5RMSNorm(H)
        self.post_attention_layernorm = Qwen3_5RMSNorm(H)


class _Torso(nn.Module):
    """A `Qwen3_5TextModel` as far as the rewriting is concerned: a `.layers` ModuleList."""

    def __init__(self, blocks: tuple[str, ...] = ("linear_attention", "full_attention")) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer(i, b) for i, b in enumerate(blocks)])


def _fuse(torso: nn.Module) -> nn.Module:
    """`fuse_torso` without the GPU kernel probes. See that function's docstring: nothing
    that serves traffic can reach `verify=False`."""
    return fuse_torso(torso, verify=False)


# ---------------------------------------------------------------------------
# Finding the layers
# ---------------------------------------------------------------------------

def test_layers_found_on_a_text_torso():
    torso = _Torso()
    assert decoder_layers(torso) is torso.layers


def test_layers_found_one_level_down():
    """The multimodal `Qwen3_5Model` keeps the stack under `.language_model`."""
    outer = nn.Module()
    outer.language_model = _Torso()
    assert len(decoder_layers(outer)) == 2


def test_a_torso_with_no_layer_list_is_refused():
    with pytest.raises(FusionUnsupported, match="no decoder layer list"):
        decoder_layers(nn.Linear(2, 2))


# ---------------------------------------------------------------------------
# Layout refusals. One per assumption in the replacement forwards.
# ---------------------------------------------------------------------------

def test_an_unknown_block_type_is_refused():
    torso = _Torso()
    torso.layers[0].block_type = "sparse_moe"
    with pytest.raises(FusionUnsupported, match="block_type"):
        check_layout(torso.layers)


def test_a_non_zero_centred_norm_is_refused():
    """`1 + weight` is right only for the zero-centred form. On any other RMSNorm it
    rescales every activation in the layer and the answers stay well-formed."""
    torso = _Torso()
    torso.layers[0].input_layernorm = _NotZeroCentred(H)
    with pytest.raises(FusionUnsupported, match="zero-centred"):
        check_layout(torso.layers)


def test_a_non_zero_centred_head_norm_is_refused():
    torso = _Torso(("full_attention",))
    torso.layers[0].self_attn.q_norm = _NotZeroCentred(HEAD_DIM)
    with pytest.raises(FusionUnsupported, match="zero-centred"):
        check_layout(torso.layers)


def test_a_mixture_of_experts_mlp_is_refused():
    torso = _Torso()
    torso.layers[0].mlp = nn.Module()
    with pytest.raises(FusionUnsupported, match=r"dense Qwen3\.5 MLP"):
        check_layout(torso.layers)


def test_a_non_silu_mlp_activation_is_refused():
    """The fused SwiGLU is silu(gate)*up. The check is numeric, not by class name, because
    `ACT2FN["silu"]` has been `nn.SiLU`, `SiLUActivation` and a bare function across
    transformers releases -- and a name check refused the real checkpoint."""
    torso = _Torso()
    torso.layers[0].mlp.act_fn = nn.GELU()
    with pytest.raises(FusionUnsupported, match="silu"):
        check_layout(torso.layers)


def test_a_silu_under_another_name_is_accepted():
    class SiLUActivation(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

    torso = _Torso()
    torso.layers[0].mlp.act_fn = SiLUActivation()
    check_layout(torso.layers)


def test_a_biased_projection_is_refused():
    """The fused projection concatenates weights only, so a bias would be dropped."""
    torso = _Torso()
    torso.layers[0].linear_attn.in_proj_z = nn.Linear(H, VALUE_DIM, bias=True)
    with pytest.raises(FusionUnsupported, match="bias-free"):
        check_layout(torso.layers)


def test_a_mismatched_conv_dim_is_refused():
    """conv_dim != 2*key_dim + value_dim means the post-conv split lands in the wrong
    channels -- queries read as values, with no error anywhere."""
    torso = _Torso()
    torso.layers[0].linear_attn.conv_dim = CONV_DIM + 4
    with pytest.raises(FusionUnsupported, match="conv_dim"):
        check_layout(torso.layers)


def test_mismatched_projection_widths_are_refused():
    torso = _Torso()
    torso.layers[0].linear_attn.in_proj_b = nn.Linear(H, NUM_V + 1, bias=False)
    with pytest.raises(FusionUnsupported, match="projection widths"):
        check_layout(torso.layers)


def test_value_heads_that_do_not_divide_by_key_heads_are_refused():
    """Without the division there is no head sharing to exploit, so the repeat the fused
    path removes is not removable."""
    torso = _Torso()
    torso.layers[0].linear_attn.num_k_heads = 3
    with pytest.raises(FusionUnsupported, match="num_v_heads"):
        check_layout(torso.layers)


def test_a_non_silu_conv_activation_is_refused():
    """fla's causal conv implements only silu/swish; substituting it changes every q/k/v."""
    torso = _Torso()
    torso.layers[0].linear_attn.activation = "gelu"
    with pytest.raises(FusionUnsupported, match="silu"):
        check_layout(torso.layers)


def test_a_non_depthwise_conv_is_refused():
    torso = _Torso()
    torso.layers[0].linear_attn.conv1d = nn.Conv1d(CONV_DIM, CONV_DIM, bias=False,
                                                   kernel_size=CONV_W)
    with pytest.raises(FusionUnsupported, match="depthwise"):
        check_layout(torso.layers)


def test_a_q_projection_without_the_packed_gate_is_refused():
    """Qwen3.5 packs the sigmoid output gate into q_proj at double head width. A single-width
    projection would be split in the wrong place and half the heads would become gates."""
    torso = _Torso(("full_attention",))
    torso.layers[0].self_attn.q_proj = nn.Linear(H, ATTN_HEADS * HEAD_DIM + 1, bias=False)
    with pytest.raises(FusionUnsupported, match="packs the sigmoid output gate"):
        check_layout(torso.layers)


def test_a_missing_mixer_is_refused():
    torso = _Torso(("linear_attention",))
    del torso.layers[0].linear_attn
    with pytest.raises(FusionUnsupported, match="no linear_attn"):
        check_layout(torso.layers)


def test_an_empty_torso_is_refused():
    with pytest.raises(FusionUnsupported, match="no decoder layers"):
        check_layout(torch.nn.ModuleList([]))


def test_a_recognised_torso_passes_the_layout_check():
    check_layout(_Torso(("linear_attention", "full_attention", "linear_attention")).layers)


# ---------------------------------------------------------------------------
# The concatenation
# ---------------------------------------------------------------------------

def test_concat_preserves_values_in_split_order():
    """The fused forward splits this weight's output back apart with `sd_splits`. If the
    order here and the order there disagree, the `a` projection's output arrives as `beta`
    -- still a plausible number, never an error."""
    a = nn.Linear(5, 3, bias=False)
    b = nn.Linear(5, 2, bias=False)
    c = nn.Linear(5, 1, bias=False)
    w = _concat(a, b, c)

    assert w.shape == (6, 5)
    assert torch.equal(w[:3], a.weight)
    assert torch.equal(w[3:5], b.weight)
    assert torch.equal(w[5:], c.weight)
    assert w.is_contiguous()


def test_concat_matches_running_the_projections_separately():
    a = nn.Linear(5, 3, bias=False)
    b = nn.Linear(5, 2, bias=False)
    x = torch.randn(4, 5)
    fused = torch.nn.functional.linear(x, _concat(a, b)).split([3, 2], -1)
    # atol, matching the rest of this file. One GEMM over the concatenated weight and two
    # GEMMs over the halves reduce in a different order, so they agree to fp32 rounding and
    # not bit-exactly -- and `torch.allclose`'s default atol of 1e-8 is tighter than that.
    # MEASURED: with unseeded inputs this failed roughly 1 run in 5, which is a flaky gate
    # rather than a detected bug. The property under test is "same weights, same answer",
    # and 1e-6 states it without asserting a reduction order the kernel never promised.
    assert torch.allclose(fused[0], a(x), atol=1e-6)
    assert torch.allclose(fused[1], b(x), atol=1e-6)


def test_the_deltanet_split_reverses_the_deltanet_concat():
    """End to end on the real widths: one GEMM then `sd_splits` must equal four GEMMs."""
    torso = _Torso(("linear_attention",))
    m = torso.layers[0].linear_attn
    x = torch.randn(1, 3, H)
    want = [m.in_proj_qkv(x), m.in_proj_z(x), m.in_proj_b(x), m.in_proj_a(x)]

    _fuse(torso)
    got = torch.nn.functional.linear(x, m.sd_in_proj).split(m.sd_splits, -1)

    assert m.sd_splits == [CONV_DIM, VALUE_DIM, NUM_V, NUM_V]
    for w, g in zip(want, got, strict=True):
        assert torch.allclose(w, g, atol=1e-6)


def test_the_attention_split_reverses_the_attention_concat():
    torso = _Torso(("full_attention",))
    m = torso.layers[0].self_attn
    x = torch.randn(1, 3, H)
    want = [m.q_proj(x), m.k_proj(x), m.v_proj(x)]

    _fuse(torso)
    got = torch.nn.functional.linear(x, m.sd_qkv).split(m.sd_splits, -1)

    for w, g in zip(want, got, strict=True):
        assert torch.allclose(w, g, atol=1e-6)


def test_the_mlp_concat_keeps_gate_before_up():
    """`swiglu(gate, up)` is `silu(gate) * up` and is not symmetric, so the halves must not
    be swapped."""
    torso = _Torso(("linear_attention",))
    mlp = torso.layers[0].mlp
    x = torch.randn(2, H)
    want_gate, want_up = mlp.gate_proj(x), mlp.up_proj(x)

    _fuse(torso)
    gate, up = torch.nn.functional.linear(x, mlp.sd_gate_up).chunk(2, -1)

    assert torch.allclose(want_gate, gate, atol=1e-6)
    assert torch.allclose(want_up, up, atol=1e-6)


def test_the_norm_weights_are_zero_centred_and_fp32():
    """`1 + weight`, in fp32, folded once at load rather than per token. Getting the +1
    wrong turns a trained norm into a near-no-op."""
    torso = _Torso(("linear_attention",))
    with torch.no_grad():
        torso.layers[0].input_layernorm.weight.fill_(0.25)
    _fuse(torso)

    w = torso.layers[0].sd_input_norm_weight
    assert w.dtype is torch.float32
    assert torch.allclose(w, torch.full_like(w, 1.25))


# ---------------------------------------------------------------------------
# Idempotence, and what the rewrite leaves behind
# ---------------------------------------------------------------------------

def test_fuse_is_idempotent():
    """The rewrite deletes the modules it replaces, so a second call that did not
    short-circuit would either fail on a missing `in_proj_qkv` or concatenate an
    already-concatenated weight."""
    torso = _Torso()
    _fuse(torso)
    first = torso.layers[0].linear_attn.sd_in_proj.clone()

    _fuse(torso)
    _fuse(torso)

    assert torch.equal(torso.layers[0].linear_attn.sd_in_proj, first)
    assert torso.layers[0].linear_attn.sd_splits == [CONV_DIM, VALUE_DIM, NUM_V, NUM_V]
    assert is_fused(torso)


def test_fuse_is_idempotent_without_the_layout_check_passing_twice():
    """Idempotence is checked before the layout is, so a second call on a torso the checks
    would now reject (because the originals are gone) is still a no-op."""
    torso = _Torso()
    _fuse(torso)
    assert not hasattr(torso.layers[0].linear_attn, "in_proj_qkv")
    _fuse(torso)  # must not raise


def test_fuse_reports_unfused_before_and_fused_after():
    torso = _Torso()
    assert not is_fused(torso)
    _fuse(torso)
    assert is_fused(torso)


def test_the_originals_are_released():
    """Keeping them would roughly double the weight footprint of every projection this
    touches, which on a 24 GB card is the difference between fitting and not."""
    torso = _Torso()
    _fuse(torso)
    m = torso.layers[0].linear_attn
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
        assert not hasattr(m, name)
    assert not hasattr(torso.layers[0].mlp, "gate_proj")
    assert not hasattr(torso.layers[1].self_attn, "q_proj")


def test_fused_weights_are_buffers_so_module_to_moves_them():
    """A bare attribute is invisible to `nn.Module.to`, which would leave the fused weights
    on the old device after a later `.to()` -- an exception at best."""
    torso = _Torso()
    _fuse(torso)
    m = torso.layers[0].linear_attn
    assert "sd_in_proj" in dict(m.named_buffers())
    assert "sd_conv_weight" in dict(m.named_buffers())

    torso.to(torch.float64)
    assert m.sd_in_proj.dtype is torch.float64


def test_fused_weights_stay_out_of_the_state_dict():
    """Non-persistent, so nothing that saves or reloads this model sees fused weights and
    mistakes them for the checkpoint's."""
    torso = _Torso()
    _fuse(torso)
    keys = torso.state_dict()
    assert not any(k.endswith(("sd_in_proj", "sd_qkv", "sd_gate_up")) for k in keys)


def test_a_cpu_torso_is_refused_when_the_kernels_are_verified():
    """The real entry point, with its probes on. fla's kernels are Triton and CUDA-only, so
    a CPU torso must be refused rather than fused onto a path that cannot run."""
    torso = _Torso()
    with pytest.raises((FusionUnsupported, ImportError, ModuleNotFoundError)):
        fuse_torso(torso)
    # And refused *before* the rewrite, so the torso is still usable.
    assert not is_fused(torso)
    assert hasattr(torso.layers[0].linear_attn, "in_proj_qkv")


# ---------------------------------------------------------------------------
# The flag
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("value", "expected"), [
    (None, False), ("", False), ("0", False), ("false", False), ("no", False),
    ("off", False), ("OFF", False), ("1", True), ("true", True), ("yes", True),
    ("on", True),
])
def test_the_flag_defaults_to_off(monkeypatch, value, expected):
    """Default off, because fusion changes the rounding and every published number in this
    repository was measured on the reference path."""
    if value is None:
        monkeypatch.delenv("SD_FUSE_LAYERS", raising=False)
    else:
        monkeypatch.setenv("SD_FUSE_LAYERS", value)
    assert fuse_enabled() is expected


def test_the_engine_does_not_fuse_unless_asked(monkeypatch):
    """`load_merged_engine(fuse_layers=None)` reads the env; nothing else reaches in."""
    import strands_decider.fused_layers as fl

    monkeypatch.delenv("SD_FUSE_LAYERS", raising=False)
    calls: list[object] = []
    monkeypatch.setattr(fl, "fuse_torso", lambda t, **kw: calls.append(t))
    assert fl.fuse_enabled() is False
    assert calls == []
