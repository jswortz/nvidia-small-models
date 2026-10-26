"""Phase 6, step 2: a verifiable reward for the router, computed from the store's final state.

reward(output, case) runs what the routed agent does after routing (look up the
order, let the policy engine decide, run the action) on a fresh copy of the store,
and compares the final store with the one the correct route produces. No writer
call is needed, so one reward takes milliseconds.

Scoring:
    +1.0   final store state matches the correct route's state
    -2.0   harmful: the route makes a store change the correct route would not make
    +0.25  the route itself is fully correct (same scoring as evals/run_router_eval.py)
    +0.1   the output is valid route JSON
A wrongful refund costs more than a missed one, matching the eval's priorities.

One change from the suggested weights: output that is not valid route JSON scores
0 in total. Otherwise unparseable output would earn the +1.0 state credit on every
case where the right answer changes nothing, and RL would learn to emit garbage there.

Messages the engine escalates go to the planner model in the real agent. The reward
does not run a model, so it treats escalation as "code changed nothing".

Not thread-safe: the store path is an environment variable. Call it from one thread.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import policy_engine, router  # noqa: E402
from evals.run_router_eval import effective_route, field_scores  # noqa: E402
from tools.registry import call_tool  # noqa: E402

W_STATE, W_HARMFUL, W_ROUTE, W_JSON = 1.0, -2.0, 0.25, 0.1

_TEMPLATE: Path | None = None
_CORRECT_STATE: dict[str, tuple] = {}


def _template() -> Path:
    """Build the store once per process; every reward starts from a copy of it."""
    global _TEMPLATE
    if _TEMPLATE is None:
        tmp = Path(tempfile.mkdtemp())
        subprocess.run([sys.executable, str(ROOT / "data" / "generate.py"), "--db", str(tmp / "store.db"),
                        "--policy", str(tmp / "policy.md")], check=True, capture_output=True)
        os.environ["STORE_POLICY"] = str(tmp / "policy.md")
        _TEMPLATE = tmp / "store.db"
    return _TEMPLATE


def _snapshot(db: Path) -> frozenset:
    """The parts of the store an action can change, without ids, timestamps, or free text."""
    con = sqlite3.connect(db)
    try:
        rows = {("order", oid, status) for oid, status in con.execute("SELECT order_id, status FROM orders")}
        rows |= {("refund", oid, round(amount, 2)) for oid, amount in con.execute("SELECT order_id, amount FROM refunds")}
        rows |= {("replacement", oid, pid) for oid, pid in
                 con.execute("SELECT order_id, new_product_id FROM replacements")}
    finally:
        con.close()
    return frozenset(rows)


def final_state(route: dict) -> frozenset:
    """Run lookup, policy engine, and action for one route on a fresh store; return the final state.

    This mirrors steps 2 to 4 of agents/routed_agent.py. tests/test_reward.py checks they agree.
    """
    db = Path(tempfile.mkdtemp()) / "store.db"
    shutil.copy(_template(), db)
    os.environ["STORE_DB"] = str(db)
    order = product = None
    if route["intent"] not in ("policy_question", "other") and route["order_id"]:
        order = json.loads(call_tool("lookup_order", {"order_id": route["order_id"]}))
    if route["intent"] == "replacement" and route["new_product"] and order and order.get("found"):
        hits = json.loads(call_tool("search_products", {"query": route["new_product"]}))["results"]
        product = hits[0] if hits else None
    decision = policy_engine.decide(route, order, product)
    if decision.outcome == "act":
        call_tool(decision.action, decision.action_args)
    state = _snapshot(db)
    shutil.rmtree(db.parent, ignore_errors=True)
    return state


def valid_json(text: str) -> bool:
    """True when the output is one JSON object with every route field of the right type."""
    try:
        data = json.loads(router._THINK.sub("", text).strip())
    except (json.JSONDecodeError, TypeError):
        return False
    return (isinstance(data, dict) and data.get("intent") in router.INTENTS
            and all(k in data for k in router.SCHEMA["required"])
            and isinstance(data["damaged"], bool) and isinstance(data["photo_provided"], bool))


def to_route(output: str | dict, case: dict) -> dict | None:
    """The route the agent would act on, or None when the output is not valid route JSON."""
    if isinstance(output, str):
        if not valid_json(output):
            return None
        route = router.parse(output)
    else:
        route = router.parse(json.dumps(output))
    route = router.backstop(route, case["router_input"])
    return effective_route(route, case)


def reward(output: str | dict, case: dict) -> float:
    """Score one router output (raw text or a route dict) for one labeled case."""
    route = to_route(output, case)
    if route is None:
        return 0.0
    if case["id"] not in _CORRECT_STATE:
        _CORRECT_STATE[case["id"]] = final_state(case["expected_route"])
    correct = _CORRECT_STATE[case["id"]]
    initial = _snapshot(_template())
    got = final_state(route)

    score = W_JSON
    if got == correct:
        score += W_STATE
    if (got - initial) - (correct - initial):  # a change the correct route would not make
        score += W_HARMFUL
    if all(field_scores(route, case["expected_route"]).values()):
        score += W_ROUTE
    return round(score, 4)


def is_harmful(output: str | dict, case: dict) -> bool:
    route = to_route(output, case)
    if route is None:
        return False
    initial = _snapshot(_template())
    correct = _CORRECT_STATE.setdefault(case["id"], final_state(case["expected_route"]))
    return bool((final_state(route) - initial) - (correct - initial))
