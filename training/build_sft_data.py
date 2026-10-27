"""Phase 6, step 3: turn labeled router data into chat examples for LoRA SFT.

Each example is the exact conversation the router sees at inference time:
    system:    the short router prompt (agents/router.py SHORT_PROMPT)
    user:      router_input (the message, plus the focus note for multi-order messages)
    assistant: the labeled route as JSON, keys in schema order

NeMo AutoModel's ChatDataset reads these files directly and trains only on the
assistant turn. Train comes from router_train.jsonl, validation from router_dev.jsonl.
The stress test is never used here; it is the held-out score.

Run from /workspace:
    python training/build_sft_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agents.router import SCHEMA, SHORT_PROMPT  # noqa: E402

DATA = ROOT / "training" / "data"
OUT = ROOT / "training" / "output" / "sft"
KEYS = list(SCHEMA["properties"])  # intent, order_id, damaged, photo_provided, new_product


def to_example(row: dict) -> dict:
    route = {k: row["expected_route"][k] for k in KEYS}
    return {"messages": [
        {"role": "system", "content": SHORT_PROMPT},
        {"role": "user", "content": row["router_input"]},
        {"role": "assistant", "content": json.dumps(route)},
    ]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for src, dst in [("router_train.jsonl", "train.jsonl"), ("router_dev.jsonl", "val.jsonl")]:
        rows = [json.loads(line) for line in (DATA / src).read_text().splitlines() if line.strip()]
        (OUT / dst).write_text("".join(json.dumps(to_example(r)) + "\n" for r in rows))
        print(f"{dst}: {len(rows)} examples")


if __name__ == "__main__":
    main()
