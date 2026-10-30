"""Unit tests for eval scoring helpers: grounded citations. No model or store needed.

Run from /workspace:
    python tests/test_eval_scoring.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evals.run_eval import cited_clauses, false_promise, ungrounded_citations  # noqa: E402

passed = 0


def ok(label: str) -> None:
    global passed
    passed += 1
    print(f"ok    {label}")


assert cited_clauses("Refunded (policy 4.1, 4.2).") == ["4.1", "4.2"]
assert cited_clauses("Per clause 60.2 and section 1.3.") == ["60.2", "1.3"]
ok("finds clause numbers after policy, clause, or section, including lists")

assert cited_clauses("Under policy 3.1 you owe $39.99 for the 4.5 star model.") == ["3.1"]
assert cited_clauses("Your order total is 39.99 and it ships in 2.5 days.") == []
ok("ignores prices and numbers that are not near a policy word")

get = lambda cid: {"tool": "get_clause", "result": {"found": True, "clause_id": cid}}  # noqa: E731
search = {"tool": "search_policy", "result": {"results": [{"clause_id": "2.1"}, {"clause_id": "1.1"}]}}
assert ungrounded_citations({"answer": "Declined (policy 3.1).", "trace": [get("3.1")]}) == []
assert ungrounded_citations({"answer": "See policy 2.1 and 1.1.", "trace": [search]}) == []
ok("clauses fetched by get_clause or returned by search_policy count as grounded")

# Held-out h009: the planner cited 2.1 for gift wrapping without retrieving any policy.
h009 = {"answer": "Our policy (clause 2.1) requires gift wrapping at checkout.", "trace": [{"tool": "_route", "result": {}}]}
assert ungrounded_citations(h009) == ["2.1"]
assert ungrounded_citations({"answer": "Within the window (policy 60.2).", "trace": [get("4.2")]}) == ["60.2"]
ok("a cited clause that was never retrieved is flagged (h009, invented 60.2)")

no_action = lambda answer: {"answer": answer, "trace": []}  # noqa: E731
assert false_promise(no_action("Your refund has been processed."))
assert false_promise(no_action("I will issue a replacement today."))
assert not false_promise(no_action("Policy 1.2 applies. No refund, replacement, or cancellation was processed."))
assert not false_promise(no_action("I can't confirm the refund was processed until we have the order number."))
assert false_promise(no_action("No problem! Your refund has been processed."))  # negation in an earlier sentence
ok("false promise check skips negated sentences (p005, x001) but still catches real claims")

print(f"\nAll {passed} checks passed.")
