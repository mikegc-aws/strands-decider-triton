"""Fused inference kernels for the Qwen3.5 torso, behind `SD_FUSE_LAYERS=1`.

Why. A *small* pass on this deployable is dispatch-bound -- `max(45 ms, tokens x 0.0935 ms)`
on an L4, where the 45 ms floor is ~5,676 CPU kernel launches against a 12.7 ms
weight-streaming floor (see `batch_engine.py`). A *full* pass is not: once the batcher fills
one with ~56 question rows it is memory-bandwidth-bound, and prefill runs at ~34% of the
card's peak. Cross-request batching already spread the per-pass floor across every request
in flight, so what is left to win is the GPU work itself -- fewer kernels, less arithmetic,
less traffic -- which is what this module goes after. The LoRA merge took 17,574 CUDA ops
down to 9,411 (`merged_engine.py`); this takes aim at what remains.

That distinction is not a footnote: it is why the numbers below go in opposite directions at
batch 1 and at batch 56, and why this is a throughput lever rather than a latency one.

What the reference spends the launches on, and what replaces it, per decoder layer:

| reference (transformers 5.18)                                   | fused                                    |
| --- | --- |
| 4 separate GEMMs for q/k/v, z, b, a                             | 1 GEMM, weights concatenated             |
| `causal_conv1d_fn` = a PyTorch depthwise `F.conv1d`, behind a    | fla's Triton causal conv, started from   |
| `torch.cat` of the cached conv state (`Cache.update_conv_state`) | the cached state, no concatenation       |
| gate `-exp(A_log)*softplus(a+dt_bias)`, `beta.sigmoid()` and the | computed inside the chunk kernel         |
| q/k `repeat_interleave` as fp32 elementwise ops (not on THIS     | key heads shared by value heads (GVA)    |
| checkpoint: 16 key heads to 16 value heads, so it never runs)    |                                          |
| `Qwen3_5RMSNormGated` as ~6 elementwise ops                     | fla's fused gated RMSNorm                |
| 2 GEMMs for the MLP gate and up, then `silu(g)*u`               | 1 GEMM and a fused SwiGLU                |
| 2 zero-centred `Qwen3_5RMSNorm`s plus a residual add            | fla's fused RMSNorm, the second adding   |
|                                                                 | the residual                             |
| attention: 3 GEMMs, 2 head RMSNorms, `out * sigmoid(gate)`      | 1 GEMM, fused norms, fused sigmoid gate  |

What it bought, measured on this deployable's L4 (`tools/fused_ab.py`), and why it is still
**off by default**:

    full batch, 8 tickets x 7 questions (56 rows)   540 ms -> 419 ms   1.29x
      (103.8 -> 133.7 decisions/s; fused faster in 7 of 7 interleaved rounds)
    torso forward, batch 1,   128 tokens             42 ms ->  49 ms   0.85x  SLOWER
    torso forward, batch 1, 1,024 tokens             84 ms ->  73 ms   1.14x

The win is in the saturated regime and the loss is at batch 1, which is the two regimes
behaving differently rather than a surprise: one small pass is CPU-dispatch-bound, and fla's
ops carry *more* Python per call (`input_guard`, autotune lookups, an autograd `Function`)
even while launching fewer kernels, whereas a 56-row pass is memory-bandwidth-bound -- the
deploy measurements put an L40S at 2.7x an L4's throughput for 2.88x its bandwidth -- and
there doing less arithmetic and moving less data is exactly what helps. On a 4-vCPU
g6.xlarge the Python side is not cheap. So this is a throughput lever, not a latency one.

The *math* is the reference's. The *rounding* is not: the fused kernels keep fp32 where the
reference rounds to bf16 in between (`Qwen3_5RMSNormGated` does
`self.weight * hidden_states.to(input_dtype)` mid-formula, for instance). Measured over 80
answers across both readout routes: **zero decision flips**, mean |dp| 0.0011-0.0019, max
0.0102. The max is above this project's 7e-3 advisory band, and `--fp32-reference` is what
settles whether that matters: against the same torso in fp32 the reference bf16 path sits
0.00105 away on average and the fused path 0.00124, a 1.18x difference with the
per-primitive maxima not ordering consistently. Both bf16 paths are about equally close to
the exact answer and simply not close to each other, which is rounding rather than a kernel
error. `kev` records the same magnitude for its own bf16 path (max 0.0133, mean 0.0014 from
fp32, zero argmax flips, L40S). `tools/batch_parity.py` and `tools/reference_check.py` both
pass with fusion on, at zero flips and zero mismatches.

Provenance. The approach, the choice of fla ops and the structure of the four replacement
forwards are a port of `kev/fused_qwen35.py` (Apache-2.0); see THIRD_PARTY_NOTICES.md for
what was taken and what changed. The two substantive differences:

  * **The cache contract is the reference's, unchanged.** kev skips writing back the
    DeltaNet state on a pass that *continues* a cached one, because its question rows never
    continue from each other and the write-back costs as much as the read. The same is true
    here -- `batch_engine._rows_probs` pass 2 and `infer._slot_probs_shared_prefix` both
    throw their cache away afterwards -- but a fused path with a *different* cache contract
    than the reference is a trap for the next person to add a decode step, and it would make
    `_fork_layered_cache`'s invariants depend on which kernels happen to be installed. So
    this writes the state back exactly where the reference does. The cost is measured and
    reported rather than assumed away.
  * **The assumptions are verified at load, on the GPU, before the first caller.**
    `verify_kernels` is this module's `_assert_fla_on_gpu` (see
    `model_repository/decider/1/model.py`): five probes, each comparing a fused kernel
    against the reference expression it replaces, each naming the wrong answer it prevents.
    They exist because every one of those assumptions fails *silently* -- an fla release
    that swallowed `use_gate_in_kernel` into `**kwargs`, or grouped GVA heads by `%` instead
    of `//`, still returns a well-formed probability distribution. A confidently wrong
    probability at HTTP 200 is this project's worst failure, so the kernels are made to
    prove themselves rather than trusted.

Scope and limits, stated rather than discovered later:

  * **Inference only.** There is no backward pass: the fused projections replace the
    originals and the originals are freed (keeping them would roughly double the weight
    footprint of every layer this touches).
  * **One-way within a process.** `fuse_torso` cannot be undone, because un-fusing would
    need the weights it freed. Reversibility is the flag: `SD_FUSE_LAYERS=0` (the default)
    loads the reference torso, and nothing in this module runs.
  * **Pinned to flash-linear-attention 0.5.2**, the version these kernel contracts were
    verified against. A different version refuses to fuse rather than fusing on an
    unverified contract.
  * The torso's final `self.norm` is left alone: it is one kernel per pass, not per layer.
  * **This checkpoint has 16 linear key heads and 16 value heads**, so the reference's
    `repeat_interleave` of q/k never runs and the "no head repeat" part of the port buys
    nothing here. The GVA path and its probe are kept because they cost nothing and the next
    checkpoint may not be 1:1, but do not credit the speed-up above to them.
"""

from __future__ import annotations

import os
import types
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# The flash-linear-attention these kernel contracts were verified against (see
# `verify_kernels`). Pinned exactly: the ops this module leans on take their most important
# arguments through `**kwargs` (`use_gate_in_kernel`, `A_log`, `dt_bias`), so a version that
# renamed or dropped one would ignore it in silence and compute a different gate.
FLA_VERSION = "0.5.2"

# Env flag. Default OFF: fusion changes the arithmetic, and the reference path is the one
# every published number in this repository was measured on.
FUSE_ENV = "SD_FUSE_LAYERS"

# Marker attribute, so `fuse_torso` is idempotent. Checked before any kernel probe runs, so
# a second call is free as well as harmless.
_FUSED_FLAG = "_sd_fused_layers"

# Relative tolerance for the load-time kernel probes. Each probe compares two routes through
# the *same* kernel family in the serving dtype (bf16), differing only in where the gate /
# the head grouping / the conv state is handled, so agreement is expected to be near
# bit-exact and 2e-2 relative is a wide net for a gross contract break -- not a calibration
# tolerance. The calibration tolerance is `tools/fused_ab.py`'s 7e-3 on probabilities.
_PROBE_RTOL = 2e-2


class FusionUnsupported(RuntimeError):
    """This torso, or this flash-linear-attention, is not one the fused path recognises.

    Raised rather than falling back to the reference layer-by-layer, because a *partially*
    fused torso is the failure this repository is built to refuse: it still answers, with a
    well-formed distribution, at HTTP 200. Same reasoning as `merge_lora.py`'s watched-weight
    guard and `model.py::_assert_fla_on_gpu` -- verify loudly, refuse rather than degrade.

    The caller's correct response is to load without fusion (`SD_FUSE_LAYERS=0`), which is
    the default, not to retry or to patch around the message.
    """


def fuse_enabled(default: bool = False) -> bool:
    """Whether `SD_FUSE_LAYERS` asks for fusion. Opt-in; anything unset means `default`."""
    raw = os.environ.get(FUSE_ENV)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def is_fused(torso: nn.Module) -> bool:
    return bool(getattr(torso, _FUSED_FLAG, False))


# ---------------------------------------------------------------------------
# Layout recognition
# ---------------------------------------------------------------------------

def decoder_layers(torso: nn.Module) -> nn.ModuleList:
    """The decoder layer list of a Qwen3.5 text torso.

    `merged_engine.load_merged_torso` hands back `Qwen3_5ForCausalLM(text_config).model`,
    i.e. a `Qwen3_5TextModel`, which carries `.layers` directly. The multimodal
    `Qwen3_5Model` keeps the same stack one level down under `.language_model`, and
    `StrandsDeciderModel._load_torso` could in principle produce either, so both are
    resolved -- by looking for the attribute rather than by class name, because the class
    names moved between transformers 5.17 and 5.18.
    """
    for candidate in (torso, getattr(torso, "language_model", None), getattr(torso, "model", None)):
        layers = getattr(candidate, "layers", None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return layers
    raise FusionUnsupported(
        f"{type(torso).__name__} exposes no decoder layer list (.layers, "
        ".language_model.layers or .model.layers); the fused path cannot tell which modules "
        "to rewrite, so it refuses rather than rewriting the wrong ones"
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FusionUnsupported(message)


def _bias_free(where: str, *linears: Any) -> None:
    for lin in linears:
        _require(
            getattr(lin, "bias", None) is None,
            f"{where}: expected a bias-free nn.Linear, found a bias. The fused projection "
            "concatenates weights only, so a bias would be dropped and every token would "
            "get a silently shifted projection.",
        )


def check_layout(layers: nn.ModuleList) -> None:
    """Refuse any torso whose layer layout is not the one the replacement forwards assume.

    Every check here corresponds to one line of the fused forwards. If a check is missing and
    the assumption is wrong, the result is not a crash: it is a plausible, wrong probability.

    The two that matter most and are easiest to get wrong:

    * **Zero-centred RMSNorm.** `Qwen3_5RMSNorm` holds `weight` initialised to *zeros* and
      applies `1.0 + weight`. Hand fla's `rms_norm` the bare weight and a well-trained layer
      becomes a near-no-op; hand a *non* zero-centred norm `1 + weight` and every activation
      is scaled wrong. The class is checked by name here and the identity is checked
      numerically in `verify_kernels`.
    * **`silu` convolution activation.** fla's causal conv accepts only `silu`/`swish`. The
      reference takes it from `config.hidden_act`, so a checkpoint with a different
      activation must be refused, not quietly given silu.
    """
    _require(len(layers) > 0, "torso has no decoder layers")

    for i, layer in enumerate(layers):
        at = f"layer {i}"
        block = getattr(layer, "block_type", None)
        _require(
            block in ("linear_attention", "full_attention"),
            f"{at}: block_type={block!r} is not one the fused path implements "
            "('linear_attention' or 'full_attention')",
        )

        for name in ("input_layernorm", "post_attention_layernorm"):
            norm = getattr(layer, name, None)
            _require(norm is not None, f"{at}: no {name}")
            _require(
                type(norm).__name__ == "Qwen3_5RMSNorm",
                f"{at}.{name}: expected the zero-centred Qwen3_5RMSNorm, found "
                f"{type(norm).__name__}. The fused norm passes `1 + weight`, which is only "
                "right for the zero-centred form; on any other RMSNorm it rescales every "
                "activation in the layer and the answers stay well-formed while being wrong.",
            )
            _require(hasattr(norm, "eps"), f"{at}.{name}: no .eps")

        mlp = getattr(layer, "mlp", None)
        _require(mlp is not None, f"{at}: no mlp")
        _require(
            all(hasattr(mlp, n) for n in ("gate_proj", "up_proj", "down_proj")),
            f"{at}.mlp: {type(mlp).__name__} is not the dense Qwen3.5 MLP "
            "(gate_proj/up_proj/down_proj). Mixture-of-experts layers are not covered; load "
            "with SD_FUSE_LAYERS=0.",
        )
        _bias_free(f"{at}.mlp", mlp.gate_proj, mlp.up_proj)
        _require(
            getattr(mlp, "act_fn", None) is None or _acts_like_silu(mlp.act_fn),
            f"{at}.mlp.act_fn: the fused SwiGLU computes silu(gate)*up, but "
            f"{type(mlp.act_fn).__name__} does not agree with silu on a test sweep",
        )

        if block == "linear_attention":
            _check_deltanet_layout(getattr(layer, "linear_attn", None), at)
        else:
            _check_attention_layout(getattr(layer, "self_attn", None), at)


def _acts_like_silu(fn: Any) -> bool:
    """Does this activation agree with silu? Measured, not matched by name.

    `ACT2FN["silu"]` has been `nn.SiLU`, `SiLUActivation` and a bare function across
    transformers releases -- a name check refused the real checkpoint on the first run of
    this module. What the fused SwiGLU actually needs is that the function *is* silu, which
    is cheap to establish directly and does not go stale.
    """
    if isinstance(fn, nn.SiLU):
        return True
    if not callable(fn):
        return False
    try:
        x = torch.linspace(-6.0, 6.0, 49, dtype=torch.float32)
        return bool(torch.allclose(fn(x), F.silu(x), atol=1e-6))
    except Exception:
        # Anything that will not take a plain fp32 vector is not something the fused SwiGLU
        # can stand in for, whatever it is called.
        return False


def _check_deltanet_layout(m: Any, at: str) -> None:
    _require(m is not None, f"{at}: block_type is linear_attention but there is no linear_attn")
    needed = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "conv1d", "norm",
              "out_proj", "A_log", "dt_bias", "key_dim", "value_dim", "conv_dim",
              "head_k_dim", "head_v_dim", "num_k_heads", "num_v_heads", "layer_idx")
    missing = [n for n in needed if not hasattr(m, n)]
    _require(not missing, f"{at}.linear_attn ({type(m).__name__}): missing {missing}")
    _bias_free(f"{at}.linear_attn", m.in_proj_qkv, m.in_proj_z, m.in_proj_b, m.in_proj_a)

    _require(
        m.conv_dim == 2 * m.key_dim + m.value_dim,
        f"{at}.linear_attn: conv_dim {m.conv_dim} != 2*key_dim + value_dim "
        f"{2 * m.key_dim + m.value_dim}; the post-conv q/k/v split would land in the wrong "
        "channels and silently mix queries with values",
    )
    _require(
        m.in_proj_qkv.out_features == m.conv_dim
        and m.in_proj_z.out_features == m.value_dim
        and m.in_proj_b.out_features == m.num_v_heads
        and m.in_proj_a.out_features == m.num_v_heads,
        f"{at}.linear_attn: projection widths "
        f"({m.in_proj_qkv.out_features}, {m.in_proj_z.out_features}, "
        f"{m.in_proj_b.out_features}, {m.in_proj_a.out_features}) do not match the split "
        f"({m.conv_dim}, {m.value_dim}, {m.num_v_heads}, {m.num_v_heads}) the fused "
        "projection would cut them at",
    )
    # GVA: fla shares key head h with value heads [h*n, (h+1)*n), which is only equivalent to
    # the reference's `repeat_interleave` when the counts divide. `verify_kernels` checks the
    # grouping direction as well, because `%` instead of `//` is a plausible upstream change
    # that no shape check can see.
    _require(
        m.num_k_heads > 0 and m.num_v_heads % m.num_k_heads == 0,
        f"{at}.linear_attn: num_v_heads {m.num_v_heads} is not a multiple of num_k_heads "
        f"{m.num_k_heads}, so value heads cannot share key heads and the repeat the fused "
        "path removes is not removable",
    )
    _require(
        getattr(m.conv1d, "bias", None) is None,
        f"{at}.linear_attn.conv1d: has a bias; the fused conv is called with bias=None",
    )
    _require(
        m.conv1d.groups == m.conv_dim and m.conv1d.weight.shape[1] == 1,
        f"{at}.linear_attn.conv1d: expected a depthwise conv (groups == conv_dim, weight "
        f"[D, 1, W]), found groups={m.conv1d.groups} weight={tuple(m.conv1d.weight.shape)}; "
        "fla's conv takes a [D, W] weight and squeezing a non-depthwise one would drop "
        "channels",
    )
    _require(
        _is_silu_name(getattr(m, "activation", None)),
        f"{at}.linear_attn.activation={getattr(m, 'activation', None)!r}: fla's causal conv "
        "implements only silu/swish, and substituting silu for something else changes every "
        "q/k/v the layer produces",
    )
    norm = m.norm
    _require(
        type(norm).__name__ == "Qwen3_5RMSNormGated" and hasattr(norm, "variance_epsilon"),
        f"{at}.linear_attn.norm: expected Qwen3_5RMSNormGated, found {type(norm).__name__}",
    )
    _require(
        getattr(norm, "activation", "silu") in ("silu", "swish"),
        f"{at}.linear_attn.norm.activation={getattr(norm, 'activation', None)!r}: the fused "
        "gated norm is called with activation='swish'",
    )
    _require(
        norm.weight.shape == (m.head_v_dim,),
        f"{at}.linear_attn.norm.weight {tuple(norm.weight.shape)} is not [head_v_dim] "
        f"[{m.head_v_dim}]; the fused norm normalises over the last axis of a "
        "[B, T, heads, head_v_dim] view",
    )


def _is_silu_name(value: Any) -> bool:
    return isinstance(value, str) and value in ("silu", "swish")


def _check_attention_layout(m: Any, at: str) -> None:
    _require(m is not None, f"{at}: block_type is full_attention but there is no self_attn")
    needed = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm", "head_dim",
              "scaling", "config", "layer_idx")
    missing = [n for n in needed if not hasattr(m, n)]
    _require(not missing, f"{at}.self_attn ({type(m).__name__}): missing {missing}")
    _bias_free(f"{at}.self_attn", m.q_proj, m.k_proj, m.v_proj)
    # Qwen3.5 packs the output gate into q_proj: its width is heads * head_dim * 2, and the
    # per-head second half is the gate. Getting this wrong swaps queries for gates.
    _require(
        m.q_proj.out_features % (2 * m.head_dim) == 0,
        f"{at}.self_attn.q_proj: out_features {m.q_proj.out_features} is not a multiple of "
        f"2*head_dim {2 * m.head_dim}; Qwen3.5 packs the sigmoid output gate alongside q in "
        "this projection and the fused split would cut it in the wrong place",
    )
    for name in ("q_norm", "k_norm"):
        norm = getattr(m, name)
        _require(
            type(norm).__name__ == "Qwen3_5RMSNorm" and hasattr(norm, "eps"),
            f"{at}.self_attn.{name}: expected the zero-centred Qwen3_5RMSNorm, found "
            f"{type(norm).__name__}",
        )
        _require(
            norm.weight.shape == (m.head_dim,),
            f"{at}.self_attn.{name}.weight {tuple(norm.weight.shape)} is not [head_dim] "
            f"[{m.head_dim}]",
        )


# ---------------------------------------------------------------------------
# Kernel contract probes
# ---------------------------------------------------------------------------

def verify_kernels(layers: nn.ModuleList) -> dict[str, float]:
    """Make the fused kernels prove they compute what the reference layers compute.

    This module's equivalent of `model.py::_assert_fla_on_gpu`, and for the same reason:
    every assumption below fails *silently*. fla takes `use_gate_in_kernel`, `A_log` and
    `dt_bias` through `**kwargs`, so a release that renamed one would ignore it and compute
    the decay from the raw projection instead of `-exp(A_log)*softplus(a + dt_bias)`; the
    server would start, report healthy, and answer confidently wrong. The version pin alone
    is not enough, because a wheel can be patched in place.

    Five probes, each against the reference expression it replaces:

      1. zero-centred RMSNorm   -- `rms_norm(x, 1 + w)` vs the module's own forward
      2. gated RMSNorm          -- fla's fused gate vs `Qwen3_5RMSNormGated.forward`
      3. in-kernel gate and beta -- vs `-exp(A_log)*softplus(a+dt_bias)` and `b.sigmoid()`
      4. GVA head sharing       -- vs the reference's explicit `repeat_interleave`
      5. conv state continuation -- a split pass with the carried state vs one whole pass

    Returns the measured relative deviations, so the log records how much headroom there was
    rather than only that it passed. Raises `FusionUnsupported` on any failure.
    """
    from fla.modules.conv import causal_conv1d
    from fla.modules.fused_norm_gate import rms_norm_gated
    from fla.modules.layernorm import rms_norm
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    linear = next((lay.linear_attn for lay in layers
                   if getattr(lay, "block_type", None) == "linear_attention"), None)
    _require(linear is not None,
             "no linear_attention layer to verify the DeltaNet kernels against")
    dev = linear.A_log.device
    dtype = linear.in_proj_qkv.weight.dtype
    _require(
        dev.type == "cuda",
        f"the fused kernels are fla Triton kernels and run on CUDA only; this torso is on "
        f"{dev}. Load with {FUSE_ENV}=0 for the reference path.",
    )

    measured: dict[str, float] = {}
    gen = torch.Generator(device=dev).manual_seed(20261008)

    def randn(*shape: int, dt: torch.dtype | None = None) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, device=dev, dtype=dt or dtype)

    def agree(name: str, ref: torch.Tensor, got: torch.Tensor, why: str) -> None:
        ref32, got32 = ref.float(), got.float()
        _require(
            torch.isfinite(got32).all(),
            f"kernel probe {name!r} produced non-finite values. {why}",
        )
        scale = ref32.abs().max().clamp_min(1e-6)
        rel = float((ref32 - got32).abs().max() / scale)
        measured[name] = rel
        _require(
            rel <= _PROBE_RTOL,
            f"kernel probe {name!r} disagrees with the reference by {rel:.4g} relative "
            f"(limit {_PROBE_RTOL:g}). {why} Refusing to fuse; load with {FUSE_ENV}=0.",
        )

    # ---- 1. the zero-centred RMSNorm identity.
    norm0 = layers[0].input_layernorm
    x = randn(8, norm0.weight.numel())
    agree("rms_norm_zero_centred", norm0(x),
          rms_norm(x, 1.0 + norm0.weight.float(), None, eps=norm0.eps),
          "Qwen3_5RMSNorm applies `1 + weight` to a zero-initialised weight; if that is no "
          "longer true, every activation in every layer is rescaled and the answers stay "
          "well-formed while being wrong.")

    # ---- 2. the gated RMSNorm. Shapes as the fused forward uses them: the reference
    # flattens to [rows, head_v_dim], the fused call keeps the [B, T, heads, V] view.
    hv, V = int(linear.num_v_heads), int(linear.head_v_dim)
    core, z = randn(1, 4, hv, V), randn(1, 4, hv, V)
    agree("rms_norm_gated",
          linear.norm(core.reshape(-1, V), z.reshape(-1, V)).reshape(1, 4, hv, V),
          rms_norm_gated(core, z, linear.norm.weight, None, activation="swish",
                         eps=linear.norm.variance_epsilon),
          "fla's fused gated norm must match Qwen3_5RMSNormGated: norm, then scale by "
          "weight, then multiply by silu(gate).")

    # ---- 3/4. the chunk rule. T=64 is one full chunk, which exercises the intra-chunk path
    # and the carried state at once.
    T, hk, K = 64, int(linear.num_k_heads), int(linear.head_k_dim)
    rep = hv // hk
    q, k = randn(1, T, hk, K), randn(1, T, hk, K)
    v = randn(1, T, hv, V)
    raw_a, raw_b = randn(1, T, hv, dt=torch.float32), randn(1, T, hv, dt=torch.float32)
    g_ref = -linear.A_log.float().exp() * F.softplus(raw_a + linear.dt_bias.float())
    beta_ref = raw_b.sigmoid()

    out_ref, _ = chunk_gated_delta_rule(
        q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2), v,
        g=g_ref, beta=beta_ref, use_qk_l2norm_in_kernel=True)
    out_gva, _ = chunk_gated_delta_rule(
        q, k, v, g=g_ref, beta=beta_ref, use_qk_l2norm_in_kernel=True)
    agree("gva_head_sharing", out_ref, out_gva,
          "fla must share key head h with value heads [h*n, (h+1)*n), the grouping the "
          "reference's repeat_interleave produces. Grouped the other way (h = j % H) the "
          "layer reads every value head against the wrong query.")

    out_fused, _ = chunk_gated_delta_rule(
        q, k, v, g=raw_a, beta=raw_b, use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True, A_log=linear.A_log, dt_bias=linear.dt_bias,
        use_beta_sigmoid_in_kernel=True)
    agree("gate_and_beta_in_kernel", out_gva, out_fused,
          "`use_gate_in_kernel`, `A_log`, `dt_bias` and `use_beta_sigmoid_in_kernel` travel "
          "through fla's `**kwargs`. If any were renamed or dropped the kernel would read "
          "the raw projection as a log-decay, silently, and still return a distribution.")

    # ---- 5. the conv state convention. Pass 1 fills the cached conv state and pass 2
    # continues from it (batch_engine's two-pass route, and infer's shared-prefix route), so
    # a split pass must equal one whole pass. The reference reaches the same place by
    # concatenating the cached state onto the input; this carries it as a state instead.
    W = int(linear.conv1d.weight.shape[-1])
    cw = linear.conv1d.weight.squeeze(1).contiguous()
    cx = randn(1, 4 * W, int(linear.conv_dim))
    whole, _ = causal_conv1d(cx, cw, None, activation=linear.activation)
    head, state = causal_conv1d(cx[:, :2 * W], cw, None, activation=linear.activation,
                                output_final_state=True)
    tail, _ = causal_conv1d(cx[:, 2 * W:], cw, None, initial_state=state,
                            activation=linear.activation)
    agree("conv_state_continuation", whole, torch.cat([head, tail], dim=1),
          "a question suffix is convolved against the state the ticket left behind. If the "
          "state convention differs, the first W tokens of every suffix see the wrong left "
          "context -- which reads as a slightly-off probability, never as an error.")

    return measured


def _check_fla_version() -> str:
    try:
        import fla
    except Exception as exc:   # any import failure has the same answer: do not fuse
        raise FusionUnsupported(
            f"could not import flash-linear-attention, which the fused kernels are: {exc}"
        ) from exc
    version = str(getattr(fla, "__version__", "unknown"))
    _require(
        version == FLA_VERSION,
        f"the fused kernels were verified against flash-linear-attention=={FLA_VERSION}, "
        f"found {version}. The ops this module calls take `use_gate_in_kernel`, `A_log` and "
        f"`dt_bias` through `**kwargs`, so a version that renamed one would ignore it "
        f"without complaint. Install {FLA_VERSION} or load with {FUSE_ENV}=0.",
    )
    return version


# ---------------------------------------------------------------------------
# The replacement forwards
# ---------------------------------------------------------------------------

def _concat(*linears: nn.Linear) -> torch.Tensor:
    """One `[sum(out), in]` weight for several bias-free projections of the same input.

    Contiguous, because `F.linear` on a view of three stacked weights is measurably slower
    than on one block and the whole point of the concatenation is a single GEMM.
    """
    return torch.cat([lin.weight for lin in linears], 0).contiguous()


def _deltanet_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    """`Qwen3_5GatedDeltaNet.forward`, fused.

    One GEMM for q/k/v/z/b/a; fla's Triton conv started from the cached state rather than
    from a concatenation of it; the gate, the beta sigmoid, the q/k L2 norm and the key-head
    sharing all inside the chunk kernel; fla's fused gated norm.

    The cache contract is the reference's: a pass with a cache fills or advances it, in
    place, at the same points `Cache.update_conv_state` / `update_recurrent_state` would.
    `_fork_layered_cache` and `_gather_layered_cache` give every row its own storage
    precisely because this happens in place, and that invariant must not depend on which
    kernels are installed.
    """
    # Variable-length packing would need `cu_seqlens` threaded into both fla ops. Nothing in
    # this deployable produces it, and ignoring it would mix one row's tokens into the next
    # row's recurrence, so it is refused rather than dropped.
    if kwargs.get("cu_seq_lens_q") is not None:
        raise FusionUnsupported(
            "the fused DeltaNet does not implement variable-length packing (cu_seq_lens_q); "
            f"load with {FUSE_ENV}=0"
        )

    from fla.modules.conv import causal_conv1d
    from fla.modules.fused_norm_gate import rms_norm_gated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

    # Kept from the reference, and load-bearing here: `batch_engine` LEFT-pads the states in
    # pass 1, and a recurrent layer that consumed non-zero pad embeddings would carry them
    # into the state the question suffixes continue from.
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    B, T, _ = hidden_states.shape

    mixed, z, b, a = F.linear(hidden_states, self.sd_in_proj).split(self.sd_splits, -1)

    layer = cache_params.layers[self.layer_idx] if cache_params is not None else None
    previous = layer is not None and layer.has_previous_state[0]

    mixed, conv_state = causal_conv1d(
        mixed, self.sd_conv_weight, None,
        initial_state=layer.conv_states[0] if previous else None,
        output_final_state=layer is not None,
        activation=self.activation,
    )
    q, k, v = mixed.split([self.key_dim, self.key_dim, self.value_dim], -1)

    core, recurrent = chunk_gated_delta_rule(
        q.reshape(B, T, -1, self.head_k_dim),
        k.reshape(B, T, -1, self.head_k_dim),
        v.reshape(B, T, -1, self.head_v_dim),
        g=a, beta=b,
        initial_state=layer.recurrent_states[0] if previous else None,
        output_final_state=layer is not None,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True, A_log=self.A_log, dt_bias=self.dt_bias,
        use_beta_sigmoid_in_kernel=True,
    )

    if layer is not None:
        if not layer.is_conv_states_initialized[0]:
            # fla returns the state already trimmed to the kernel width, so the buffer is
            # allocated at exactly that width -- the reference reaches the same shape by
            # padding or slicing a concatenation.
            layer.lazy_initialization(conv_states=conv_state,
                                      conv_kernel_size=conv_state.shape[-1])
        # copy_, not assignment: the cache layer marks these as static addresses, and the
        # forked caches in infer.py/batch_engine.py are expected to own their storage.
        layer.conv_states[0].copy_(conv_state)
        layer.has_previous_state[0] = True
        cache_params.update_recurrent_state(recurrent, self.layer_idx)

    core = rms_norm_gated(
        core, z.reshape(B, T, -1, self.head_v_dim), self.norm.weight, None,
        activation="swish", eps=self.norm.variance_epsilon,
    )
    return self.out_proj(core.reshape(B, T, -1))


def _attention_forward(self, hidden_states, position_embeddings, attention_mask,
                       past_key_values=None, **kwargs):
    """`Qwen3_5Attention.forward`, fused: one projection GEMM, fused q/k norms, fused gate.

    Qwen3.5 packs the sigmoid output gate into `q_proj`, two head-widths per head, so the
    single GEMM produces q, its gate, k and v together and the gate costs nothing extra.
    """
    from fla.modules.activations import sigmoidglu
    from fla.modules.layernorm import rms_norm
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    shape = hidden_states.shape[:-1]
    q_gate, k, v = F.linear(hidden_states, self.sd_qkv).split(self.sd_splits, -1)
    q, gate = q_gate.reshape(*shape, -1, 2 * self.head_dim).chunk(2, -1)

    q = rms_norm(q, self.sd_q_norm_weight, None, eps=self.q_norm.eps).transpose(1, 2)
    k = rms_norm(k.reshape(*shape, -1, self.head_dim), self.sd_k_norm_weight, None,
                 eps=self.k_norm.eps).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
    v = v.reshape(*shape, -1, self.head_dim).transpose(1, 2)

    if past_key_values is not None:
        k, v = past_key_values.update(k, v, self.layer_idx)

    # `_attn_implementation` is private but it is how the reference layer selects its
    # attention kernel; reading it anywhere else would risk fusing sdpa over a torso
    # loaded for eager.
    attend = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward)  # noqa: SLF001
    # The interface transposes back for us: [B, T, heads, head_dim].
    out, weights = attend(self, q, k, v, attention_mask, dropout=0.0,
                          scaling=self.scaling, **kwargs)
    return self.o_proj(sigmoidglu(gate.reshape(*shape, -1), out.reshape(*shape, -1))), weights


def _mlp_forward(self, x):
    """One GEMM for gate and up, then fla's fused SwiGLU (`silu(gate) * up`)."""
    from fla.modules.activations import swiglu

    gate, up = F.linear(x, self.sd_gate_up).chunk(2, -1)
    return self.down_proj(swiglu(gate, up))


def _decoder_forward(self, hidden_states, position_embeddings, attention_mask=None,
                     position_ids=None, past_key_values=None, **kwargs):
    """`Qwen3_5DecoderLayer.forward` with both RMSNorms fused; the second adds the residual.

    `rms_norm(..., residual=r, prenorm=True)` returns `(norm(x + r), x + r)`, which is
    exactly the reference's `residual + mixer_out` followed by the post-mixer norm -- one
    kernel instead of an add and a norm.
    """
    from fla.modules.layernorm import rms_norm

    residual = hidden_states
    h = rms_norm(hidden_states, self.sd_input_norm_weight, None,
                 eps=self.input_layernorm.eps)
    if self.block_type == "linear_attention":
        h = self.linear_attn(hidden_states=h, cache_params=past_key_values,
                             attention_mask=attention_mask, **kwargs)
    else:
        h, _ = self.self_attn(hidden_states=h, attention_mask=attention_mask,
                              position_ids=position_ids, past_key_values=past_key_values,
                              position_embeddings=position_embeddings, **kwargs)
    h, residual = rms_norm(h, self.sd_post_norm_weight, None, residual=residual,
                           eps=self.post_attention_layernorm.eps, prenorm=True)
    return residual + self.mlp(h)


# ---------------------------------------------------------------------------
# The rewrite
# ---------------------------------------------------------------------------

def _buffer(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    """Attach a fused weight as a non-persistent buffer rather than a bare attribute.

    A bare attribute (which is what the upstream this is ported from uses) is invisible to
    `nn.Module.to`, so a later `model.to(device)` would move the originals and leave the
    fused weights behind on the old device -- an exception at best, and on a multi-GPU host
    the kind of silent cross-device read that is hard to attribute. `persistent=False` keeps
    them out of `state_dict`, so nothing that saves or loads this model sees fused weights.
    """
    module.register_buffer(name, tensor, persistent=False)


def _rewrite(layers: nn.ModuleList) -> None:
    """Replace each layer's projections, norms and forwards. In place, and not reversible.

    The original `nn.Linear`s are deleted, which is what keeps this from roughly doubling the
    weight footprint of every projection it concatenates. That is also why there is no
    `unfuse`: the weights it would need are gone. `SD_FUSE_LAYERS=0` is the way back.
    """
    for layer in layers:
        if layer.block_type == "linear_attention":
            m = layer.linear_attn
            _buffer(m, "sd_in_proj",
                    _concat(m.in_proj_qkv, m.in_proj_z, m.in_proj_b, m.in_proj_a))
            m.sd_splits = [m.conv_dim, m.value_dim, m.num_v_heads, m.num_v_heads]
            _buffer(m, "sd_conv_weight", m.conv1d.weight.squeeze(1).contiguous())
            del m.in_proj_qkv, m.in_proj_z, m.in_proj_b, m.in_proj_a
            m.forward = types.MethodType(_deltanet_forward, m)
        else:
            m = layer.self_attn
            _buffer(m, "sd_qkv", _concat(m.q_proj, m.k_proj, m.v_proj))
            m.sd_splits = [m.q_proj.out_features, m.k_proj.out_features,
                           m.v_proj.out_features]
            # fp32, and `1 + w`: Qwen3_5RMSNorm is zero-centred and computes in fp32 before
            # casting back. Folding the +1 here keeps it out of the per-token path.
            _buffer(m, "sd_q_norm_weight", 1.0 + m.q_norm.weight.float())
            _buffer(m, "sd_k_norm_weight", 1.0 + m.k_norm.weight.float())
            del m.q_proj, m.k_proj, m.v_proj
            m.forward = types.MethodType(_attention_forward, m)

        _buffer(layer.mlp, "sd_gate_up", _concat(layer.mlp.gate_proj, layer.mlp.up_proj))
        del layer.mlp.gate_proj, layer.mlp.up_proj
        layer.mlp.forward = types.MethodType(_mlp_forward, layer.mlp)

        _buffer(layer, "sd_input_norm_weight", 1.0 + layer.input_layernorm.weight.float())
        _buffer(layer, "sd_post_norm_weight",
                1.0 + layer.post_attention_layernorm.weight.float())
        layer.forward = types.MethodType(_decoder_forward, layer)


@torch.no_grad()  # type: ignore[untyped-decorator]
def fuse_torso(torso: nn.Module, *, verify: bool = True) -> nn.Module:
    """Rewrite a Qwen3.5 text torso in place for fused-kernel serving. Returns it.

    Call it with the torso already on its serving device: the kernel probes run on the real
    weights, on the real device, and fla's Triton kernels are CUDA-only.

    Order matters and is deliberate. Idempotence is checked first, so a second call costs
    nothing. The layout is checked next, so an unrecognised torso is refused before anything
    is touched. The kernel probes come last of the checks, because they are the only part
    that needs a GPU -- and they run *before* the rewrite, so a failing kernel contract
    leaves the torso exactly as it was.

    `verify=False` exists for the CPU unit tests, which rewrite a toy torso that has no fla
    kernels behind it. Nothing that serves traffic passes it: `merged_engine` does not expose
    it and `model_repository/decider/1/model.py` cannot set it.
    """
    layers = decoder_layers(torso)
    if is_fused(torso):
        return torso
    check_layout(layers)
    if verify:
        _check_fla_version()
        measured = verify_kernels(layers)
        print("[strands-decider] fused-kernel probes passed: "
              + ", ".join(f"{k} {v:.2e}" for k, v in sorted(measured.items())))
    _rewrite(layers)
    setattr(torso, _FUSED_FLAG, True)
    return torso
