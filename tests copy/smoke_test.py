"""Phase 1 smoke test. Builds a throwaway copy of the store and exercises every tool.

Run from /workspace:
    python tests/smoke_test.py
Your real data/store.db is never touched.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(ROOT / "data" / "generate.py"),
                "--db", str(tmp / "store.db"), "--policy", str(tmp / "policy.md")], check=True)
os.environ["STORE_DB"] = str(tmp / "store.db")
os.environ["STORE_POLICY"] = str(tmp / "policy.md")

from tools import policy_search, store_tools  # noqa: E402
from tools.registry import OPENAI_TOOLS, call_tool  # noqa: E402

passed = 0


def check(label: str, condition: bool, detail: object = "") -> None:
    global passed
    if not condition:
        print(f"FAIL  {label}\n      {detail}")
        sys.exit(1)
    passed += 1
    print(f"ok    {label}")


order = store_tools.lookup_order("1187")
check("lookup_order finds 1187, delivered 32 days ago",
      order["found"] and order["product_id"] == "P012" and order["days_since_delivery"] == 32, order)
check("lookup_order accepts '#1187'", store_tools.lookup_order("#1187")["found"])
check("lookup_order handles a missing order", not store_tools.lookup_order("9999")["found"])

hits = policy_search.search_policy("my blender arrived cracked, can I swap it for a different model?")
ids = [r["clause_id"] for r in hits["results"]]
check("search_policy returns clause 4.2 for a cracked item swap", "4.2" in ids, ids)

hits = policy_search.search_policy("how many days do I have to return something")
ids = [r["clause_id"] for r in hits["results"]]
check("search_policy returns clause 1.1 for the return window", "1.1" in ids, ids)

products = store_tools.search_products("1200W blender")["results"]
check("search_products finds the 1200W blender first", products and products[0]["product_id"] == "P013", products)

swap = store_tools.create_replacement("1187", "Arrived cracked, clause 4.2", "P013")
check("create_replacement swaps 1187 to P013 with a $50 difference",
      swap["ok"] and swap["price_difference"] == 50.0, swap)
check("order 1187 is now replacement_issued",
      store_tools.lookup_order("1187")["status"] == "replacement_issued")

swap = store_tools.create_replacement("1100", "test", "P023")
check("create_replacement rejects a cross-category swap", not swap["ok"], swap)

refund = store_tools.issue_refund("1042", "Customer return, clause 1.1")
check("issue_refund queues the $1,299 refund for approval",
      refund["ok"] and refund["status"] == "refund_pending_approval", refund)

refund = store_tools.issue_refund("1023", "test")
check("issue_refund rejects an already refunded order", not refund["ok"], refund)

refund = store_tools.issue_refund("1199", "Cancelled before shipping, clause 1.4")
check("issue_refund cancels a processing order", refund["ok"] and refund["status"] == "refunded", refund)

result = json.loads(call_tool("lookup_order", '{"order_id": "1150"}'))
check("call_tool dispatches JSON arguments", result["found"] and result["final_sale"], result)
check("call_tool reports unknown tools", not json.loads(call_tool("nope", {}))["ok"])
check("registry exposes 5 tool schemas", len(OPENAI_TOOLS) == 5)

print(f"\nAll {passed} checks passed.")
