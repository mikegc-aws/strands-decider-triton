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

## A note on the two entries above without a URL

`kev` and `decider-2b` are named as they are named in the source comments, and both are
recorded there as Apache-2.0. If either is publicly hosted, a link belongs here — open an
issue or a PR and it will be added. The attribution is given on the strength of the
source's own record rather than omitted for want of a URL.

---

## Model weights

This repository contains **no model weights**. The deployable downloads
`strands-decider-2B-hobson-v21` at image-build time (see `deploy/Dockerfile.triton`, which
takes the repository and revision as build arguments). The checkpoint carries its own
licence and terms, which are not altered or restated by this repository; check the model
repository before redistributing an image built from it.

The base model is a Qwen3.5 checkpoint, likewise subject to its own licence.
