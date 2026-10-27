"""Phase 6: merge a LoRA adapter into the base model's weight files.

For every adapted layer: W_merged = W + (alpha / r) * B @ A, computed in float32
and stored back in the base dtype (bfloat16). Every other file (config, tokenizer,
custom model code, weight index) is copied from the base snapshot unchanged, so the
merged model loads anywhere the base model does, including vLLM v0.10.1 on the 3090.

Why not AutoModel's tools/merge_lora.py: it merges correctly, but saving through the
newer transformers fails on Nemotron Nano's custom code (_tied_weights_keys is a list).

Run (inside a container with torch and safetensors):
    python training/merge_lora.py --base <snapshot dir> --adapter <dir with adapter_model.safetensors> \
        --out checkpoints/router-sft
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "base_model.model."


def load_adapter(adapter_dir: Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]:
    """Return {base weight name: (A, B)} and the LoRA scale alpha / r."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(adapter_dir / "adapter_model.safetensors", "pt") as f:
        for key in f.keys():
            module, kind = key.removeprefix(PREFIX).rsplit(".lora_", 1)  # kind: "A.weight" or "B.weight"
            pairs.setdefault(f"{module}.weight", {})[kind[0]] = f.get_tensor(key)
    return {name: (p["A"], p["B"]) for name, p in pairs.items()}, scale


def merge(base: Path, adapter_dir: Path, out: Path) -> int:
    lora, scale = load_adapter(adapter_dir)
    weight_map = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    missing = set(lora) - set(weight_map)
    if missing:
        raise SystemExit(f"Adapter layers not in base model: {sorted(missing)[:5]}")

    out.mkdir(parents=True, exist_ok=True)
    for item in base.iterdir():  # config, tokenizer, custom code, index: copied unchanged
        if not item.name.endswith(".safetensors") and item.is_file():
            shutil.copy(item.resolve(), out / item.name)

    merged = 0
    for shard in sorted(set(weight_map.values())):
        tensors = {}
        with safe_open(base / shard, "pt") as f:
            metadata = f.metadata()
            for name in f.keys():
                w = f.get_tensor(name)
                if name in lora:
                    a, b = lora[name]
                    w = (w.float() + scale * (b.float() @ a.float())).to(w.dtype)
                    merged += 1
                tensors[name] = w.contiguous()
        save_file(tensors, out / shard, metadata=metadata)
        print(f"wrote {shard}")
    if merged != len(lora):
        raise SystemExit(f"Merged {merged} of {len(lora)} adapter layers")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True, help="base model snapshot directory")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    n = merge(args.base, args.adapter, args.out)
    print(f"Merged {n} LoRA layers into {args.out}")


if __name__ == "__main__":
    main()
