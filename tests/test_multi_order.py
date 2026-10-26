"""Unit tests for messages that name several orders. No model needed: a fake client stands in for Nano.

Run from /workspace:
    python tests/test_multi_order.py
"""
from __future__ import annotations

import json
import os
import re
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

from agents import router  # noqa: E402
from agents.routed_agent import find_orders, run_routed  # noqa: E402
from tools.store_tools import lookup_order  # noqa: E402

passed = 0


def ok(label: str) -> None:
    global passed
    passed += 1
    print(f"ok    {label}")


class FakeClient:
    """Answers router calls from a fixed intent per order, and writer calls with a fixed reply."""

    def __init__(self, intents: dict[str, str], reply: str = "Done (policy 1.1, 1.4)."):
        self.intents, self.reply = intents, reply
        self.router_calls, self.writer_calls = [], []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        system, user = kwargs["messages"][0]["content"], kwargs["messages"][1]["content"]
        if system == router.ROUTER_PROMPT:
            self.router_calls.append(user)
            focus = re.search(r"about order (\d{4})", user)
            oid = focus.group(1) if focus else re.search(r"\d{4}", user).group(0)
            content = json.dumps({"intent": self.intents[oid], "order_id": oid, "damaged": False,
                                  "photo_provided": False, "new_product": None})
        else:
            self.writer_calls.append(json.loads(user))
            content = self.reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


# Detection: only real orders count, prices are ignored, and each order counts once.
assert find_orders("Refund order 1012 and also cancel order 1080.") == ["1012", "1080"]
ok("finds two real orders in order of appearance")
assert find_orders("Where is order 5555? Also order 1012.") == ["1012"]
ok("ignores a number that is not a real order")
assert find_orders("Refund order 1012, the $1299 TV, order 1012 again") == ["1012"]
ok("ignores prices and counts a repeated order once")

# A one-order message takes the original single-order path.
client = FakeClient({"1012": "refund"})
result = run_routed("Refund order 1012 please.", client=client)
assert "multi_order" not in result and len(client.router_calls) == 1
ok("one order stays on the single-order path")

# x003: both requests are routed, decided, and executed, and one reply covers both.
client = FakeClient({"1012": "refund", "1080": "cancel"})
result = run_routed("Refund order 1012 and also cancel order 1080.", client=client)
assert result["multi_order"] and [r["intent"] for r in result["routes"]] == ["refund", "cancel"]
assert lookup_order("1012")["status"] == "refunded" and lookup_order("1080")["status"] == "refunded"
assert len(client.writer_calls) == 1 and len(client.writer_calls[0]["requests"]) == 2
assert all("Classify only the request about order" in m for m in client.router_calls)
ok("x003: refund and cancel both executed, one reply for both")

# An allowed action next to a declined one: only the allowed one changes the store.
client = FakeClient({"1118": "cancel", "1107": "refund"})
result = run_routed("Please cancel order 1118, and return the knife set from order 1107.", client=client)
assert lookup_order("1118")["status"] == "refunded" and lookup_order("1107")["status"] == "delivered"
assert [d["outcome"] for d in result["decisions"]] == ["act", "decline"]
ok("mixed: the allowed cancel runs, the late return is declined")

# A request the engine would escalate is turned into a question, and the planner is never called.
client = FakeClient({"1090": "other", "1104": "refund"})
result = run_routed("Add a toaster to order 1090 and return order 1104.", client=client)
assert [d["outcome"] for d in result["decisions"]] == ["need_info", "act"] and not result["escalated"]
assert lookup_order("1104")["status"] == "refunded"
ok("an unplaceable request asks the customer instead of escalating")

print(f"\nAll {passed} checks passed.")
