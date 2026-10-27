"""Unit test for training/merge_lora.py on tiny fake weights. Needs torch and safetensors,
so it runs in the training container, not poc-dev:
    python tests/test_merge_lora.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.merge_lora import merge  # noqa: E402

tmp = Path(tempfile.mkdtemp())
base, adapter, out = tmp / "base", tmp / "adapter", tmp / "out"
base.mkdir(), adapter.mkdir()

torch.manual_seed(0)
w1, w2, keep = (torch.randn(6, 4, dtype=torch.bfloat16) for _ in range(3))
save_file({"layers.0.proj.weight": w1, "layers.0.norm.weight": keep}, base / "model-1.safetensors")
save_file({"lm_head.weight": w2}, base / "model-2.safetensors")
(base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
    "layers.0.proj.weight": "model-1.safetensors", "layers.0.norm.weight": "model-1.safetensors",
    "lm_head.weight": "model-2.safetensors"}}))
(base / "config.json").write_text('{"model": "fake"}')

a1, b1, a2, b2 = torch.randn(2, 4), torch.randn(6, 2), torch.randn(2, 4), torch.randn(6, 2)
save_file({f"base_model.model.{n}.lora_{k}.weight": t.to(torch.bfloat16) for n, k, t in [
    ("layers.0.proj", "A", a1), ("layers.0.proj", "B", b1), ("lm_head", "A", a2), ("lm_head", "B", b2)]},
    adapter / "adapter_model.safetensors")
(adapter / "adapter_config.json").write_text(json.dumps({"lora_alpha": 32, "r": 16}))

assert merge(base, adapter, out) == 2
m1, m2 = load_file(out / "model-1.safetensors"), load_file(out / "model-2.safetensors")
bf = lambda t: t.to(torch.bfloat16).float()  # noqa: E731  adapter tensors were stored in bf16
expect1 = (w1.float() + 2.0 * bf(b1) @ bf(a1)).to(torch.bfloat16)
assert torch.equal(m1["layers.0.proj.weight"], expect1)
assert torch.equal(m2["lm_head.weight"], (w2.float() + 2.0 * bf(b2) @ bf(a2)).to(torch.bfloat16))
assert torch.equal(m1["layers.0.norm.weight"], keep), "non-LoRA weights must be unchanged"
assert m1["layers.0.proj.weight"].dtype == torch.bfloat16
assert (out / "config.json").read_text() == '{"model": "fake"}' and (out / "model.safetensors.index.json").exists()
print("ok    merge applies W + (alpha/r) * B @ A, keeps other weights and files unchanged")
print("\nAll 1 checks passed.")
