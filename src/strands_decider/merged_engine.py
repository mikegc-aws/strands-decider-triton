"""The HF engine with the LoRA adapter already folded into the torso weights.

Why this exists. `StrandsDeciderModel.load` wraps the torso in PEFT and leaves the rank-16
adapter unmerged, so every adapted projection runs as `base(x) + B(A(x))` -- three matmuls
where one would do. Measured on an L4 (`/opt/prof/merge.py`, `/var/log/sd-merge.log`):

    aten::mm per forward   558 -> 186
    total CUDA ops      17,574 -> 9,411
    torso forward, bf16:   128 tok  76.5 -> 47.3 ms   (1.62x)
                         1,024 tok 132.0 -> 81.0 ms   (1.63x)
                         4,096 tok 660.3 -> 354.5 ms  (1.86x)

The vLLM image already merges at build time (`serving/merge_lora.py`). The HF path never
did, which left the torch engine paying for PEFT at every request. This closes that gap
without changing anything above the forward pass.

Measured, with a correctness check behind it rather than a throughput number alone:
0 of 6 decision flips against the unmerged torso, max `|Δnoul|` 0.0078 and max
`|Δscore|` 0.0033.

This docstring used to end "CUDA graph capture over this torso was measured on the same
hardware and does **not** work -- it returns wrong values at every shape where it is fast.
Do not reach for it here." That was a true measurement of *plain* capture and it is no
longer the whole story: the wrong values came from transformers building its attention
masks inside the captured region, where a host-side branch is baked in rather than
recorded. Hand the masks in precomputed and replay is bit-identical to eager. See
`cuda_graphs.py`, which is opt-in through `SD_CUDA_GRAPHS` and off by default.

Unlike the vLLM engine this keeps the **shared-prefix path**, which is the torch engine's
real advantage: it encodes the state once and forks its cache across the questions, where
vLLM must send each question as a separate row carrying the whole state again. That is why
the torch path still wins at long states (519 ms against vLLM's 638 ms at a ~3,000-token
state) and it is the reason to keep this engine at all.

Nothing here reimplements inference. `SystemOneEngine` is used unchanged -- prompt
rendering, the question-first window fit, front truncation, option spans, both readout
paths, temperature, calibration and answer decoding all come from it. This module only
builds a `StrandsDeciderModel` whose torso is pre-merged.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from .infer import EngineConfig, SystemOneEngine
from .modeling import (
    StrandsDeciderConfig,
    StrandsDeciderModel,
    assert_bos_contract,
    base_revision,
    build_head,
    checkpoint_dir,
    config_path,
    load_head_state,
    load_text_tower,
)


class MergedTorsoMismatch(RuntimeError):
    """The merged directory is not the torso this checkpoint's head was trained against.

    Raised rather than worked around because the failure it prevents is silent. The pointer
    head is a 1M-parameter readout fitted to one torso's hidden states; point it at a
    different model (or a different base revision) and it still returns a well-formed
    probability distribution -- just a meaningless one, at HTTP 200. Same failure class as
    `VllmSystemOneEngine._check_hidden_len`.
    """


def load_merged_torso(
    merged_torso: str,
    config: StrandsDeciderConfig,
    *,
    attn_implementation: str | None = None,
) -> nn.Module:
    """Load a plain (already-merged) directory as a bare torso.

    Shares `modeling.load_text_tower` with `StrandsDeciderModel._load_torso` and
    `serving/merge_lora.py` rather than mirroring them by hand: every base this serves is
    a multimodal checkpoint, so `AutoModel` would hand back the wrapper with its vision
    tower attached. `serving/merge_lora.py` writes the same shape, and the two must
    agree -- a name mismatch there makes PEFT report an *empty* merge rather than an error.
    """
    from transformers import AutoConfig, AutoModel

    kwargs: dict[str, Any] = {"dtype": getattr(torch, config.torch_dtype)}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    base_cfg = AutoConfig.from_pretrained(merged_torso)
    torso = load_text_tower(merged_torso, base_cfg, AutoModel, **kwargs)
    torso.config.use_cache = True
    return torso


def build_merged_model(
    checkpoint: str,
    merged_torso: str,
    *,
    attn_implementation: str | None = None,
    torch_dtype: str | None = None,
) -> StrandsDeciderModel:
    """A `StrandsDeciderModel` with a pre-merged torso and no PEFT wrapper.

    The head, tokenizer, window and fitted temperatures still come from the Decider
    checkpoint -- `merge_lora.py` writes only the torso and the tokenizer, by design, so
    the calibration lives in exactly one place.

    `torch_dtype` overrides the checkpoint's own (`bfloat16`). It exists for one job:
    `tools/fused_ab.py --fp32-reference` loads the torso in fp32 to decide whether a
    probability difference between two kernel paths is rounding or a bug -- the bf16
    reference and the fused path can then both be measured against the same higher-precision
    answer instead of only against each other. It is **not** a serving knob: fp32 doubles the
    weight footprint and roughly halves throughput for a difference no caller can see.
    """
    ckpt = checkpoint_dir(checkpoint)
    config = StrandsDeciderConfig.from_json(config_path(ckpt))
    config.base_revision = base_revision(ckpt, config)
    if torch_dtype:
        config.torch_dtype = torch_dtype

    if not os.path.isdir(merged_torso):
        raise FileNotFoundError(
            f"{merged_torso}: not a directory. Build it with "
            f"`python serving/merge_lora.py {checkpoint} {merged_torso}`."
        )
    # A merged directory must NOT carry an adapter: if it does, someone pointed this at a
    # raw checkpoint and the adapter would be silently ignored, serving the un-adapted base.
    if os.path.isdir(os.path.join(merged_torso, "lora")):
        raise MergedTorsoMismatch(
            f"{merged_torso} contains a lora/ directory, so it is a Decider checkpoint "
            "rather than a merged torso. Serving it here would silently ignore the "
            "adapter and answer as the un-adapted base model."
        )

    head_state = load_head_state(ckpt)
    torso = load_merged_torso(merged_torso, config,
                             attn_implementation=attn_implementation)

    hidden = StrandsDeciderModel.hidden_size(torso)
    head = build_head(config, hidden)
    # Check the head actually fits this torso before loading, so a wrong merged directory
    # fails here with a clear message instead of at the first request with a bad answer.
    try:
        head.load_state_dict(head_state)
    except RuntimeError as exc:
        raise MergedTorsoMismatch(
            f"the head in {ckpt} does not fit the torso in {merged_torso} "
            f"(hidden_size={hidden}): {exc}"
        ) from exc

    # Build the module directly: __init__ would attach a fresh LoRA on top of weights that
    # already have it folded in, which is the one thing that must not happen here.
    obj = StrandsDeciderModel.__new__(StrandsDeciderModel)
    nn.Module.__init__(obj)
    obj.config = config
    obj.torso = torso
    obj.head = head.to(torch.float32)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Before serving, not on the first request: a checkpoint that asserts force_bos and a
    # tokeniser that will not supply one is a startup failure, not a per-request one.
    assert_bos_contract(config, tok)
    obj.tokenizer = tok
    return obj


@torch.inference_mode()  # type: ignore[untyped-decorator]
def load_merged_engine(
    checkpoint: str,
    merged_torso: str,
    *,
    device: str = "cuda",
    use_prefix_cache: bool = True,
    max_batch: int = 32,
    strict_window: bool = False,
    model_name: str = "strands-decider-merged",
    attn_implementation: str | None = None,
    max_rows: int = 128,
    fuse_layers: bool | None = None,
    cuda_graphs: bool = False,
    cuda_graphs_two_pass: bool = False,
) -> SystemOneEngine:
    """A `BatchedSystemOneEngine` over a pre-merged torso.

    `use_prefix_cache` defaults to **True**, unlike the vLLM engine which must refuse it.
    That is the point of this engine: the shared-prefix path needs an HF torso, and it is
    what makes many questions over one state nearly free.

    Returns the batched subclass rather than a plain `SystemOneEngine`. It *is* a
    `SystemOneEngine` -- `evaluate` is inherited untouched, so every existing caller is
    unaffected -- and it additionally offers `evaluate_many`, which spreads the ~45 ms
    per-pass floor across every request in flight instead of paying it per request. A
    server that does not know about `evaluate_many` simply never calls it, so this is a
    safe default rather than a behaviour change.

    `fuse_layers` replaces the torso's decoder layers with fla's Triton kernels (see
    `fused_layers.py`). `None` reads `SD_FUSE_LAYERS`, which defaults to off: fusion changes
    the rounding, and the reference path is what every published number here was measured
    on. It is applied *after* the engine has moved the model to its device, because the
    kernel-contract probes run on the real weights on the real GPU, and it raises rather
    than falling back -- a half-fused torso still answers, which is the failure this project
    exists to refuse.

    `cuda_graphs` is off by default and opt-in through `SD_CUDA_GRAPHS=1`. It replaces the
    per-pass CPU dispatch with a graph replay on the ONE-pass route, which is where the
    ~45 ms launch floor dominates: measured 45.9 ms -> 21.5 ms server-side for a
    one-question request on an L4. `cuda_graphs_two_pass` (`SD_CUDA_GRAPHS=all`) extends it
    to the state/row pair and is off even then, because it was measured at 0.73x-1.02x
    there; `BatchedSystemOneEngine.__init__` records the numbers and the cause. Both fall
    back to the eager path whenever capture is unavailable or a shape sits outside the
    graphed envelope; see `cuda_graphs.py`.

    The two are independent and compose, but their ORDER here is load-bearing -- see the
    comment at the bottom of this function.

    """
    model = build_merged_model(checkpoint, merged_torso,
                              attn_implementation=attn_implementation)
    cfg = EngineConfig(
        device=device,
        use_prefix_cache=use_prefix_cache,
        max_batch=max_batch,
        strict_window=strict_window,
        model_name=model_name,
    )
    from .batch_engine import BatchedSystemOneEngine

    engine = BatchedSystemOneEngine(model, cfg, max_rows=max_rows,
                                    cuda_graphs=cuda_graphs,
                                    cuda_graphs_two_pass=cuda_graphs_two_pass)

    from .fused_layers import fuse_enabled, fuse_torso

    # Fuse BEFORE anything is captured. Graph capture is lazy -- the first pass of a given
    # shape records it -- so fusing here means a graph records the FUSED kernels. Fusing
    # after a capture would leave the replay running reference kernels while the eager
    # fallback ran fused ones: two paths through the same engine doing different arithmetic,
    # disagreeing by ~1.2e-3 depending only on whether a shape happened to be graphed. That
    # is unreproducible-answer territory, so the order is not a style choice.
    if fuse_enabled() if fuse_layers is None else fuse_layers:
        fuse_torso(engine.model.torso)
    return engine

