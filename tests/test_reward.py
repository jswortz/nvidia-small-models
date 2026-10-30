"""Unit tests for the Phase 6 store-state reward. No model needed.

Run from /workspace:
    python tests/test_reward.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training import reward as R  # noqa: E402

passed = 0


def ok(label: str) -> None:
    global passed
    passed += 1
    print(f"ok    {label}")


def route(intent, oid, damaged=False, photo=False, new_product=None) -> dict:
    return {"intent": intent, "order_id": oid, "damaged": damaged, "photo_provided": photo, "new_product": new_product}


def case(cid: str, expected: dict, message: str = "test message", group: str = "test") -> dict:
    return {"id": cid, "group": group, "router_input": message, "expected_route": expected}


# Pick orders by their computed policy facts, so the test does not depend on hard-coded ids.
os.environ["STORE_DB"] = str(R._template())
from tools.store_tools import lookup_order  # noqa: E402

eligible = next(o for o in (lookup_order(str(n)) for n in range(1001, 1201))
                if o.get("status") == "delivered" and not o["final_sale"]
                and o["policy_facts"]["within_30_day_return_window"]
                and not o["policy_facts"]["refund_needs_supervisor_approval"])["order_id"]
other = next(o for o in (lookup_order(str(n)) for n in range(1001, 1201))
             if o.get("status") == "delivered" and o["order_id"] != eligible)["order_id"]

refund_case = case("refund", route("refund", eligible), f"Please refund order {eligible}, I changed my mind.")
status_case = case("status", route("order_status", eligible), f"Where is order {eligible}?")

assert R.reward(route("refund", eligible), refund_case) == 1.35
ok("correct route: state match + route + JSON = 1.35")

assert R.reward(json.dumps(route("refund", eligible)), refund_case) == 1.35
ok("raw JSON text from the model scores the same as a route dict")

assert R.reward(route("order_status", eligible), refund_case) == 0.1
assert not R.is_harmful(route("order_status", eligible), refund_case)
ok("harmless wrong route (missed refund): JSON credit only, 0.1")

assert R.reward(route("refund", eligible), status_case) == round(0.1 - 2.0, 4)
assert R.is_harmful(route("refund", eligible), status_case)
ok("harmful wrong route (refund when only status was asked): 0.1 - 2.0 = -1.9")

assert R.reward("Sure, I'll refund that for you!", refund_case) == 0.0
assert R.reward('{"intent": "refund"}', refund_case) == 0.0
ok("unparseable or incomplete output scores 0, with no state credit")

assert R.reward(route("refund", eligible, photo=True), refund_case) == 1.1
ok("a wrong field that does not change the outcome keeps state credit but loses the route bonus")

multi = case("multi", route("refund", eligible), f"Refund order {eligible} and where is {other}?", group="multi")
assert R.reward(route("refund", other), multi) == 1.35
ok("multi-order: the agent's order override is applied before scoring")

start = time.perf_counter()
for _ in range(50):
    R.reward(route("refund", eligible), refund_case)
per_call_ms = 1000 * (time.perf_counter() - start) / 50
assert per_call_ms < 100, per_call_ms
ok(f"reward is fast: {per_call_ms:.1f} ms per call")

# The reward mirrors steps 2 to 4 of the routed agent. Check both leave the same store for several routes.
from agents import router  # noqa: E402
from agents.routed_agent import run_routed  # noqa: E402


class FakeClient:
    def __init__(self, r: dict):
        self.r = r
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        content = json.dumps(self.r) if kwargs["messages"][0]["content"] == router.ROUTER_PROMPT else "Done."
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))],
                               usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))


for r in [route("refund", eligible), route("cancel", eligible), route("replacement", eligible, True, True),
          route("replacement", eligible, False, False, "AeroBrew Coffee Maker"), route("refund", "1150"),
          route("refund", "1042", True), route("order_status", eligible)]:
    db = Path(tempfile.mkdtemp()) / "store.db"
    shutil.copy(R._template(), db)
    os.environ["STORE_DB"] = str(db)
    run_routed(f"message about order {r['order_id']}", client=FakeClient(r))
    agent_state = R._snapshot(db)
    assert R.final_state(r) == agent_state, r
ok("reward's decide-and-act matches the routed agent's store changes on 7 routes")

# Regression: product search after a state check used to point at a deleted temp store.
swap = {"id": "swap-1", "router_input": f"swap order {eligible} for the AeroBrew Coffee Maker", "group": "swap",
        "expected_route": route("replacement", eligible, False, False, "AeroBrew Coffee Maker")}
R.final_state(route("refund", eligible))
assert R.reward(swap["expected_route"], swap) in (1.35, 0.35), R.reward(swap["expected_route"], swap)
ok("new_product scoring still works after a state check")

print(f"\nAll {passed} checks passed.")
