"""Fold the Decider's LoRA adapter into its base torso and save a plain checkpoint.

Run at IMAGE BUILD time, on CPU. vLLM then serves the result as an ordinary Qwen3.5 model
with no PEFT at runtime, and the pointer head stays outside, in `vllm_engine.py`.

Why bother, in numbers. Served un-merged, the torso loads as
`PeftModelForFeatureExtraction` and every adapted projection runs its base GEMM plus two
skinny LoRA GEMMs. Measured on an L4: **558 `aten::mm` calls per forward where the merged
model needs 186**, costing 1.59x at 256 tokens and 1.86x at 4,096 (660 ms -> 354 ms;
prefill 6,204 -> 11,553 tok/s). Those extra GEMMs are far too small to occupy the GPU, so
they buy launch overhead and nothing else.

It is also safe. Merging folds `B @ A * scale` into the base weight, the same function in
exact arithmetic; in bf16 it moves the last digits. Checked over six probe cases spanning
calm to furious and 130 to 2,900 tokens: **zero decision flips**, max |delta noul| 0.0078,
max |delta score| 0.0033 on a 0-4 scale.

The base model and revision come from the checkpoint's own config and provenance, not from
a constant here, so a checkpoint trained on a different base cannot be silently merged into
the wrong torso.

Usage: python merge_lora.py <decider-checkpoint> <out-dir>
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch
import transformers
from peft import PeftModel
from transformers import AutoConfig, AutoModel

from strands_decider.modeling import (
    StrandsDeciderConfig,
    base_revision,
    config_path,
)


def main(checkpoint: str, out: str) -> int:
    config = StrandsDeciderConfig.from_json(config_path(checkpoint))
    revision = base_revision(checkpoint, config) or config.base_revision
    print(f"[merge] checkpoint {checkpoint}")
    print(f"[merge] base       {config.base_model} @ {revision or 'default branch'}")

    kwargs = {"dtype": getattr(torch, config.torch_dtype), "revision": revision}
    base_cfg = AutoConfig.from_pretrained(config.base_model, revision=revision)

    # Mirror StrandsDeciderModel._load_torso exactly. If the torso were loaded any other
    # way the adapter's module names would not line up, and PEFT reports that as an empty
    # merge rather than an error -- which would silently produce an un-adapted model that
    # still answers, i.e. the worst possible outcome.
    if base_cfg.model_type in {"qwen3_5", "qwen3_5_text"}:
        lm = transformers.Qwen3_5ForCausalLM.from_pretrained(
            config.base_model, config=base_cfg.get_text_config(), **kwargs
        )
        torso = lm.model
    else:
        lm = None
        torso = AutoModel.from_pretrained(config.base_model, **kwargs)

    adapter = Path(checkpoint) / "lora"
    if not adapter.is_dir():
        raise SystemExit(f"no LoRA adapter at {adapter}; nothing to merge")

    # Which weights should the merge actually move? Read it from the adapter rather than
    # guessing. The first version of this guard snapshotted the first state-dict entry,
    # which is `embed_tokens.weight` -- a weight LoRA never targets -- so a perfectly good
    # merge looked like a no-op and failed the build. A check has to watch the thing it is
    # checking.
    adapter_cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    targets = adapter_cfg.get("target_modules") or []
    if isinstance(targets, str):
        targets = [targets]
    if not targets:
        raise SystemExit(f"{adapter}/adapter_config.json lists no target_modules")
    print(f"[merge] adapter targets: {sorted(targets)}")

    watched = [
        k for k in torso.state_dict()
        if k.endswith(".weight") and any(f".{t}." in f".{k}" for t in targets)
    ]
    if not watched:
        raise SystemExit(
            f"no weight in this torso matches the adapter's target_modules {sorted(targets)}. "
            "The adapter was trained against a different module layout, so merging would "
            "silently produce the un-adapted base."
        )
    before = {k: torso.state_dict()[k].clone() for k in watched[:8]}
    print(f"[merge] watching {len(before)} of {len(watched)} targeted weights")

    peft_model = PeftModel.from_pretrained(torso, str(adapter), is_trainable=False)
    merged = peft_model.merge_and_unload()

    # Guard against a silent no-op: PEFT reports "matched no layers" by changing nothing,
    # not by raising, and an un-adapted base still answers every request plausibly.
    merged_state = merged.state_dict()
    moved = sum(1 for k, old in before.items() if not torch.equal(old, merged_state[k]))
    if moved == 0:
        raise SystemExit(
            f"merge changed none of the {len(before)} targeted weights it should have "
            f"(e.g. {next(iter(before))}). PEFT matched no layers. Refusing to ship a model "
            "that would answer as the un-adapted base."
        )
    print(f"[merge] {moved}/{len(before)} watched weights moved -> adapter applied")

    if lm is not None:
        lm.model = merged
        to_save = lm
    else:
        to_save = merged
    to_save.save_pretrained(out, safe_serialization=True)

    for name in ("tokenizer.json", "tokenizer_config.json"):
        src = Path(checkpoint) / name
        if src.is_file():
            shutil.copy(src, Path(out) / name)
    print(f"[merge] wrote {out}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
