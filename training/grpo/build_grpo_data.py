"""Phase 6, step 4: write GRPO prompts for NeMo RL.

Each row: {"input": router_input, "output": ground truth for the reward}. NeMo RL adds
the system prompt (training/grpo/system_prompt.txt, the same short prompt SFT used),
samples routes from the model, and the reward environment scores them.

Run from /workspace:
    python training/grpo/build_grpo_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from agents.router import SHORT_PROMPT  # noqa: E402

DATA = ROOT / "training" / "data"
OUT = ROOT / "training" / "output" / "grpo"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "system_prompt.txt").write_text(SHORT_PROMPT)
    for src, dst in [("router_train.jsonl", "train.jsonl"), ("router_dev.jsonl", "val.jsonl")]:
        rows = [json.loads(line) for line in (DATA / src).read_text().splitlines() if line.strip()]
        out = [{"input": r["router_input"],
                "output": json.dumps({k: r[k] for k in ("id", "router_input", "expected_route", "group")})} for r in rows]
        (OUT / dst).write_text("".join(json.dumps(o) + "\n" for o in out))
        print(f"{dst}: {len(out)} prompts")


if __name__ == "__main__":
    main()
