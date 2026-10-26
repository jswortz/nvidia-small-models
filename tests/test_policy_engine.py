"""Unit tests for the policy engine and router parsing. No model needed.

Run from /workspace:
    python tests/test_policy_engine.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(ROOT / "data" / "generate.py"), "--db", str(tmp / "store.db"),
                "--policy", str(tmp / "policy.md")], check=True, capture_output=True)
os.environ["STORE_DB"] = str(tmp / "store.db")
os.environ["STORE_POLICY"] = str(tmp / "policy.md")

from agents.policy_engine import decide  # noqa: E402
from agents.router import parse  # noqa: E402
from tools.store_tools import lookup_order, search_products  # noqa: E402

passed = 0


def route(intent, order_id=None, damaged=False, photo=False, new_product=None):
    return {"intent": intent, "order_id": order_id, "damaged": damaged,
            "photo_provided": photo, "new_product": new_product}


def check(label, decision, outcome, clauses=None, action=None):
    global passed
    ok = decision.outcome == outcome and (clauses is None or decision.clauses == clauses) \
        and decision.action == action
    if not ok:
        print(f"FAIL  {label}\n      got {decision}")
        sys.exit(1)
    passed += 1
    print(f"ok    {label}")


blender = search_products("1200W blender")["results"][0]
tv55 = search_products("55in TV")["results"][0]

# The 10 eval cases, decided by rules alone
check("c004 cancel a processing order", decide(route("cancel", "1199"), lookup_order("1199")),
      "act", ["1.4"], "issue_refund")
check("c005 refund over $500 inside 30 days", decide(route("refund", "1042"), lookup_order("1042")),
      "act", ["1.1", "6.1"], "issue_refund")
check("c006 refund at 45 days, not damaged", decide(route("refund", "1100"), lookup_order("1100")),
      "decline", ["1.1"])
check("c007 final sale, not damaged", decide(route("refund", "1150"), lookup_order("1150")),
      "decline", ["1.3"])
check("c008 already refunded", decide(route("refund", "1023"), lookup_order("1023")), "decline", [])
check("c009 cracked, day 32, photo, swap to 1200W",
      decide(route("replacement", "1187", damaged=True, photo=True, new_product="1200W"), lookup_order("1187"), blender),
      "act", ["4.2"], "create_replacement")
check("c010 order not found", decide(route("order_status", "5555"), lookup_order("5555")), "info")

# Branches the eval cases do not reach yet
check("damage claim after 7 days with no photo asks for one",
      decide(route("replacement", "1187", damaged=True), lookup_order("1187")), "need_info", ["4.3"])
check("damage refund after 7 days offers a replacement instead",
      decide(route("refund", "1187", damaged=True), lookup_order("1187")), "decline", ["4.1", "4.2"])
check("cross-category swap is declined",
      decide(route("replacement", "1187", damaged=True, photo=True, new_product="TV"), lookup_order("1187"), tv55),
      "decline", ["4.2"])
earbuds_order = lookup_order("1026")  # electronics, delivered 10 days ago, not final sale
vacuum = search_products("robot vacuum")["results"][0]
check("non-damaged cross-category exchange cites the exchange clause",
      decide(route("replacement", "1026", new_product="robot vacuum"), earbuds_order, vacuum), "decline", ["3.1"])
check("final sale exchange is declined before asking which model",
      decide(route("replacement", "1047", new_product="a different one"), lookup_order("1047"), None),
      "decline", ["1.3"])
check("final sale item that arrived damaged can still be refunded within 7 days",
      decide(route("refund", "1150", damaged=True), lookup_order("1150")), "act", ["4.1"], "issue_refund")
check("missing order number asks for it", decide(route("refund"), None), "need_info")
check("unclassifiable request escalates", decide(route("other"), None), "escalate")

# Router output parsing
parsed = parse('Sure! ```json\n{"intent": "refund", "order_id": 1042, "damaged": false}\n```')
assert parsed["intent"] == "refund" and parsed["order_id"] == "1042", parsed
passed += 1
print("ok    router parses JSON wrapped in prose and code fences")
assert parse("I am not sure what you mean")["intent"] == "other"
passed += 1
print("ok    router turns unparseable output into intent 'other'")
assert parse('{"intent": "teleport"}')["intent"] == "other"
passed += 1
print("ok    router rejects intents outside the allowed list")

print(f"\nAll {passed} checks passed.")
