"""Unit tests for the hosted planner timeout fallback and model call timing. No model needed.

Run from /workspace:
    python tests/test_planner_fallback.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(ROOT / "data" / "generate.py"), "--db", str(tmp / "store.db"),
                "--policy", str(tmp / "policy.md")], check=True, capture_output=True)
os.environ["STORE_DB"] = str(tmp / "store.db")
os.environ["STORE_POLICY"] = str(tmp / "policy.md")

from agents import router, routed_agent  # noqa: E402
from agents.routed_agent import TimedClient, run_routed  # noqa: E402
from tools.store_tools import lookup_order  # noqa: E402

routed_agent.PLANNER_BASE_URL = None  # tests pass planner clients directly; ignore the shell's setting
passed = 0


def ok(label: str) -> None:
    global passed
    passed += 1
    print(f"ok    {label}")


def reply(content: str | None = None, tool_calls: list | None = None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def timeout_error() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=httpx.Request("POST", "https://hosted.example/v1/chat/completions"))


class LocalClient:
    """Stands in for local Nano: routes every message to 'other' and answers planner calls directly."""

    base_url = "http://nim:8000/v1"

    def __init__(self, fail_planner: bool = False):
        self.fail_planner, self.planner_calls = fail_planner, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        if kwargs["messages"][0]["content"] == router.ROUTER_PROMPT:
            return reply(json.dumps({"intent": "other", "order_id": None, "damaged": False,
                                     "photo_provided": False, "new_product": None}))
        self.planner_calls += 1
        if self.fail_planner:
            raise timeout_error()
        return reply("Local planner answer.")


class HostedClient:
    """Stands in for hosted Super: plays a fixed script, where an exception in the script is raised."""

    base_url = "https://hosted.example/v1"

    def __init__(self, script: list):
        self.script, self.calls = list(script), 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


# Hosted planner times out on its first call: the local planner answers, and the run says so.
local, hosted = LocalClient(), HostedClient([timeout_error()])
result = run_routed("Do you sell gift cards?", client=local, planner_client=hosted)
assert result["planner_fallback"] and result["answer"] == "Local planner answer."
assert local.planner_calls == 1 and hosted.calls == 1
assert any(t["tool"] == "_planner_fallback" for t in result["trace"])
ok("hosted timeout falls back to the local planner")

servers = [(c["server"], c["ok"]) for c in result["model_calls"]]
assert servers == [("nim", True), ("hosted.example", False), ("nim", True)], servers
assert "APITimeoutError" in result["model_calls"][1]["error"]
assert all(isinstance(c["seconds"], float) for c in result["model_calls"])
ok("model_calls logs router, failed hosted call, and local planner with timings")

# Hosted planner works: no fallback, and the local planner is never called.
local, hosted = LocalClient(), HostedClient([reply("Hosted planner answer.")])
result = run_routed("Do you sell gift cards?", client=local, planner_client=hosted)
assert not result["planner_fallback"] and result["answer"] == "Hosted planner answer."
assert local.planner_calls == 0
ok("healthy hosted planner is used with no fallback")

# Hosted planner refunds and then fails: no rerun, so nothing can be done twice.
refund_call = SimpleNamespace(id="c1", type="function", function=SimpleNamespace(
    name="issue_refund", arguments=json.dumps({"order_id": "1012", "reason": "test"})))
local, hosted = LocalClient(), HostedClient([reply(None, [refund_call]), timeout_error()])
result = run_routed("Do you sell gift cards?", client=local, planner_client=hosted)
assert not result["planner_fallback"] and result["stopped"] and "APITimeoutError" in result["error"]
assert local.planner_calls == 0 and lookup_order("1012")["status"] == "refunded"
assert any(t["tool"] == "issue_refund" for t in result["trace"])
ok("no fallback after the hosted planner changed the store; the refund stays in the trace")

# No hosted planner configured and the local model fails: stopped with the error, nothing raised.
local = LocalClient(fail_planner=True)
result = run_routed("Do you sell gift cards?", client=local)
assert result["stopped"] and not result["planner_fallback"] and "APITimeoutError" in result["error"]
ok("local planner failure returns a stopped result instead of raising")

# TimedClient re-raises errors after logging them.
log: list[dict] = []
try:
    TimedClient(HostedClient([timeout_error()]), log).chat.completions.create(model="m", messages=[])
    raise AssertionError("expected an error")
except openai.APITimeoutError:
    pass
assert log[0]["server"] == "hosted.example" and not log[0]["ok"]
ok("TimedClient logs a failed call and re-raises it")

print(f"\nAll {passed} checks passed.")
