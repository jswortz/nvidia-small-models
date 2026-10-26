"""Unit tests for Phase 4 NeMo Agent Toolkit profiling. No model needed: a fake client stands in for Nano.

Run from /workspace (needs nvidia-nat[profiler]):
    python tests/test_nat_profiling.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(ROOT / "data" / "generate.py"), "--db", str(tmp / "store.db"),
                "--policy", str(tmp / "policy.md")], check=True, capture_output=True)
os.environ["STORE_DB"] = str(tmp / "store.db")
os.environ["STORE_POLICY"] = str(tmp / "policy.md")

from agents import nat_profiling, router, routed_agent  # noqa: E402
from agents.routed_agent import run_routed  # noqa: E402

if not nat_profiling.NAT_AVAILABLE:
    print("skip  nvidia-nat is not installed")
    sys.exit(0)

routed_agent.PLANNER_BASE_URL = None  # the planner uses the fake local client
passed = 0


def ok(label: str) -> None:
    global passed
    passed += 1
    print(f"ok    {label}")


class FakeClient:
    """Router calls use 100 prompt tokens, writer calls 200, planner calls 300, so each stage is identifiable."""

    base_url = "http://nim:8000/v1"

    def __init__(self, intent: str):
        self.intent = intent
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        if system == router.ROUTER_PROMPT:
            content, prompt = json.dumps({"intent": self.intent, "order_id": "1012", "damaged": False,
                                          "photo_provided": False, "new_product": None}), 100
        elif system == routed_agent.WRITER_PROMPT:
            content, prompt = "Your refund is issued (policy 1.1).", 200
        else:
            content, prompt = "We do not sell gift cards.", 300
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))],
                               usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=7))


# Case 1 acts (router, lookup, engine, action, writer). Case 2 escalates (router, lookup, engine, planner).
all_steps = []
with nat_profiling.Collector("refund") as c:
    first = run_routed("Please refund order 1012.", client=FakeClient("refund"))
all_steps.append(c.steps)
with nat_profiling.Collector("other") as c:
    second = run_routed("Do you sell gift cards?", client=FakeClient("other"))
all_steps.append(c.steps)

assert first["answer"].startswith("Your refund is issued") and second["escalated"]
ok("instrumented agent returns the same results")

events = [s.event_type.value for s in all_steps[0]]
assert events[0] == "WORKFLOW_START" and events[-1] == "WORKFLOW_END"
assert events.count("LLM_START") == events.count("LLM_END") == 2
assert events.count("FUNCTION_START") == events.count("FUNCTION_END")
ok("each case is one workflow with paired function and LLM events")

llm_stage = {s.function_ancestry.function_name for s in all_steps[0] if s.event_type.value == "LLM_END"}
assert llm_stage == {"router", "writer"}, llm_stage
ok("LLM calls are recorded under the stage that made them")

out = tmp / "nat"
stages = {r["stage"]: r for r in nat_profiling.profile(all_steps, out)}
assert set(stages) == {"router", "lookup", "engine", "action", "planner", "writer"}, set(stages)
assert stages["router"]["cases"] == 2 and stages["router"]["avg_prompt_tokens"] == 100
assert stages["writer"]["cases"] == 1 and stages["writer"]["avg_prompt_tokens"] == 200
assert stages["planner"]["cases"] == 1 and stages["planner"]["avg_prompt_tokens"] == 300
assert stages["action"]["cases"] == 1 and stages["action"]["llm_calls"] == 0
assert stages["engine"]["llm_calls"] == 0 and stages["engine"]["avg_ms"] >= 0
ok("per-stage summary attributes tokens to router, writer, and planner, and none to code stages")

for name in ("standardized_data_all.csv", "workflow_profiling_metrics.json", "workflow_profiling_report.txt",
             "per_stage_summary.json"):
    assert (out / name).exists(), name
ok("NAT profiler outputs are written")

print(f"\nAll {passed} checks passed.")
