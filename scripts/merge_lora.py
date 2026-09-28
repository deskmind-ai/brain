"""Merge a trained LoRA into its base model, so serving needs no PEFT wrapper.

The training run writes an adapter plus `deskmind.json` (the prompt-format flags the checkpoint was trained with:
round_size, compact_targets). Both the MLX and the local backends load a plain model directory, so the merged
directory carries the adapter's tokenizer and deskmind.json alongside the merged weights.

    python scripts/merge_lora.py runs/brain-0.8b/Qwen3.5-0.8B-lora     # -> runs/brain-0.8b/Qwen3.5-0.8B-merged
    python scripts/merge_lora.py runs/brain-4b/Qwen3.5-4B-lora --base Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter", type=Path)
    ap.add_argument("--base", help="base model id (default: the adapter's base_model_name_or_path)")
    ap.add_argument("--out", type=Path, help="default: <adapter dir without -lora>-merged")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    out = args.out or args.adapter.with_name(args.adapter.name.replace("-lora", "") + "-merged")

    import torch
    from modelscope import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = args.base or json.loads((args.adapter / "adapter_config.json").read_text())["base_model_name_or_path"]
    if not Path(base).exists() and "/" in base and base.count("/") > 1:
        # trained elsewhere: the config records that machine's cache path (…/models/Qwen--Qwen3.5-0.8B/snapshots/master)
        marker = next((part for part in Path(base).parts if "--" in part), None)
        if marker:
            base = marker.replace("--", "/", 1)
    path = base if Path(base).exists() else snapshot_download(base)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, args.dtype))
    model = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(path).save_pretrained(out)
    # save_pretrained writes the text submodel's config (model_type qwen3_5_text), which the MLX backend does not
    # know; the weights are laid out exactly like the base checkpoint, so keep the base's own config.
    shutil.copy(Path(path) / "config.json", out / "config.json")
    for name in ("deskmind.json", "chat_template.jinja"):
        src = args.adapter / name
        if src.exists():
            shutil.copy(src, out / name)
    print(json.dumps({"base": base, "adapter": str(args.adapter), "out": str(out),
                      "deskmind-brain": json.loads((out / "deskmind.json").read_text()) if (out / "deskmind.json").exists() else None}))


if __name__ == "__main__":
    main()
