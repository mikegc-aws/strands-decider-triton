# Third-party notices

This project includes and derives from the work below. `src/strands_decider/mps_kernels.py`
points here for its attribution, which is why this file exists at the repository root.

Everything listed is Apache-2.0, the same licence as this project, so no additional
restriction applies to a user of this repository.

---

## 1. Hugging Face Transformers — Qwen3.5 Gated DeltaNet chunk rule

**Used in:** `src/strands_decider/mps_kernels.py`
**Upstream:** `transformers` 5.17, `src/transformers/models/qwen3_5/modeling_qwen3_5.py`,
function `torch_chunk_gated_delta_rule`
**Licence:** Apache License 2.0
**Copyright:** Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights
reserved.
**Project:** https://github.com/huggingface/transformers

`chunk_gated_delta_rule_mps` is a modification of that function. The algorithm is
unchanged; the two `torch.linalg.solve_triangular` calls are replaced by one recursive
block inversion applied to both right-hand sides, and the body was restructured around
that change. The reason is specific to Apple silicon: those two solves cost ~20 ms each on
MPS for the shape this model uses, and MPS has no fused Gated DeltaNet kernel because
neither `flash-linear-attention` (Triton, Linux only) nor `causal_conv1d` (CUDA) can be
installed there.

The full Apache-2.0 notice is reproduced at the top of `mps_kernels.py`, as the licence
requires for a modified file.

---

## 2. `kev` — the async-handler / bounded-queue / single-device-thread pattern

**Used in:** `src/strands_decider/scheduler.py`
**Upstream:** https://github.com/jaredpalmer/kev
**Licence:** Apache License 2.0

`Scheduler` follows the shape `kev` uses for serving a single non-re-entrant device behind
an async HTTP handler: the request handler is `async def` and awaits a future, while one
worker thread owns the device, so a queued request holds no worker thread. No code is
copied; the structure and the GIL switch-interval tuning are the borrowed ideas, and
`scheduler.py`'s own docstring records the two places this deliberately differs (the queue
is bounded, and queued work carries a deadline).

---

## 3. `decider-2b` — shared-prefix cache forking for hybrid torsos

**Used in:** `src/strands_decider/infer.py` (`_fork_layered_cache`)
**Upstream:** `decider/shared_prefix.py`
**Licence:** Apache License 2.0

The approach for repeating a batch-1 cache across N rows on a hybrid (mixed attention +
linear-attention) torso is `decider-2b`'s: fork per-layer objects, dicts and tensors rather
than sharing them, because a Gated DeltaNet layer updates its states and its
`has_previous_state` flag in place and a shared fork would corrupt the prefix it came from.

---

## 4. `kev` — the fused Qwen3.5 inference layers

**Used in:** `src/strands_decider/fused_layers.py`
**Upstream:** `kev/fused_qwen35.py` (https://github.com/jaredpalmer/kev)
**Licence:** Apache License 2.0

`fused_layers.py` is a **port**, not a copy: the same approach applied to the torso
`merged_engine.load_merged_torso` builds, which is a different wrapper from `kev`'s.

**What was taken.** The diagnosis — that `transformers`' reference Qwen3.5 layers spend over
a third of their GPU time outside the matrix multiplies, on a PyTorch depthwise conv behind
a concatenation of the cached conv state, on a dozen fp32 elementwise kernels per DeltaNet
layer for the gating, the head repeat and the gated norm, and on several more per RMSNorm
and SwiGLU. The remedy, which is the choice of `flash-linear-attention` ops and how they map
onto the reference layers: one projection GEMM for q/k/v/z/b/a with the weights concatenated;
fla's Triton causal conv started from the cached conv state rather than from a concatenation
of it; fla's chunked gated delta rule with the gate `-exp(A_log)*softplus(a + dt_bias)`, the
beta sigmoid and the q/k L2 norm computed inside the kernel and key heads shared by value
heads; fla's fused gated RMSNorm; one GEMM for the MLP gate and up with a fused SwiGLU; the
decoder layer's two zero-centred RMSNorms as fla's fused RMSNorm with weight `1 + w` in fp32,
the second also adding the residual; and for the attention layers one GEMM for q (with its
packed output gate), k and v, with the q/k RMSNorms and the sigmoid output gate fused. The
structure of the four replacement forwards follows `kev`'s, and so does the observation that
this is a serving-only rewrite with no backward pass.

**What changed.**

- **The cache contract is the reference's.** `kev`'s DeltaNet forward does not advance a
  cached state on a pass that continues one, because its question rows never continue from
  each other. The same holds here, but this port writes the conv and recurrent states back
  exactly where `Cache.update_conv_state` / `update_recurrent_state` would, so the fused and
  reference paths cannot diverge in cache behaviour. `infer._fork_layered_cache` and
  `batch_engine._gather_layered_cache` depend on in-place state updates, and that invariant
  must not depend on which kernels happen to be installed.
- **The assumptions are verified on the GPU before the first request.**
  `fused_layers.verify_kernels` adds five probes with no counterpart upstream, each comparing
  a fused kernel against the reference expression it replaces: the zero-centred RMSNorm
  identity, the gated RMSNorm, the in-kernel gate and beta, the GVA head grouping, and the
  conv-state continuation. They exist because each of those assumptions fails silently — an
  fla release that swallowed `use_gate_in_kernel` into `**kwargs` would still return a
  well-formed probability distribution. `check_layout` likewise refuses an unrecognised
  layer layout rather than fusing part of a torso.
- **The fused weights are non-persistent buffers, not bare attributes**, so
  `nn.Module.to` moves them and `state_dict` does not carry them.
- **`_fix_nb` is not ported.** Upstream forces fla's `NB` launch constant to 1 to stop a busy
  server re-autotuning on every new batch shape. In the pinned 0.5.2, `NB` is
  `cdiv(T, 65536)` rather than `cdiv(T, ~2000)`, so it changes far less often, and patching a
  dependency's kernel-launch internals is not something this repository can verify. It is
  recorded as a possible future win instead.
- The torso's final `norm` is left unfused (one kernel per pass, not per layer), the
  variable-length (`cu_seqlens`) path is refused rather than ignored, and fusion is opt-in
  behind `SD_FUSE_LAYERS` with the reference path as the default.

No file is copied verbatim, so no modified-file notice is carried in the source; this entry
is the attribution and `fused_layers.py`'s docstring points here.

---

## A note on the entry above without a URL

`decider-2b` is named as it is named in the source comments, and is recorded there as
Apache-2.0. If it is publicly hosted, a link belongs here — open an issue or a PR and it
will be added. The attribution is given on the strength of the source's own record rather
than omitted for want of a URL. (`kev` was in the same position until
`fused_layers.py` was ported from it; the link added to entries 2 and 4 is the same
repository.)

---

## Model weights

This repository contains **no model weights**. The deployable downloads
`strands-decider-2B-hobson-v21` at image-build time (see `deploy/Dockerfile.triton`, which
takes the repository and revision as build arguments). The checkpoint carries its own
licence and terms, which are not altered or restated by this repository; check the model
repository before redistributing an image built from it.

The base model is a Qwen3.5 checkpoint, likewise subject to its own licence.
