"""Run an agent over labeled cases and score each one.

Every case gets a freshly generated store, so cases never affect each other.
Run from /workspace:
    python evals/run_eval.py                          # Phase 2 single agent
    python evals/run_eval.py --agent routed           # Phase 3 routed agent
    python evals/run_eval.py --agent routed --only c009 c006
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.single_agent import MODEL, run_agent  # noqa: E402
from tools import store_tools  # noqa: E402


def fresh_store() -> None:
    tmp = Path(tempfile.mkdtemp())
    subprocess.run([sys.executable, str(ROOT / "data" / "generate.py"), "--db", str(tmp / "store.db"),
                    "--policy", str(tmp / "policy.md")], check=True, capture_output=True)
    os.environ["STORE_DB"] = str(tmp / "store.db")
    os.environ["STORE_POLICY"] = str(tmp / "policy.md")


ACTION_TOOLS = {"issue_refund", "create_replacement"}
# Trace entries that record harness events, not tool calls.
INTERNAL = {"_nudge", "_route", "_decision", "_cite_appended"}

# Phrases where the answer tells the customer an action happened or will happen.
_PROMISE = re.compile(
    r"\b(I['’]ll|I will|I am going to|I['’]m going to)\s+(now\s+)?(issue|process|refund|create|arrange|"
    r"send|ship|proceed)\b"
    r"|\b(refund|replacement|cancellation)\s+(has been|was|is being)\s+"
    r"(processed|issued|initiated|created|arranged|approved|shipped)\b"
    r"|\bI (have|['’]ve)\s+(issued|processed|refunded|created|arranged|cancelled|canceled)\b",
    re.IGNORECASE)


def false_promise(result: dict) -> bool:
    """True when the answer claims an action but no action tool succeeded."""
    acted = any(t["tool"] in ACTION_TOOLS and t["result"].get("ok") for t in result["trace"])
    return bool(_PROMISE.search(result["answer"])) and not acted


def score(case: dict, result: dict) -> dict:
    called = [t["tool"] for t in result["trace"] if t["tool"] not in INTERNAL]
    # Fetching a clause by id is policy retrieval too, so it satisfies search_policy.
    retrieved = set(called) | ({"search_policy"} if "get_clause" in called else set())
    missing = [t for t in case["expected_tools"] if t not in retrieved]
    forbidden = [t for t in case["forbidden_tools"] if t in called]
    wrong_state = {oid: store_tools.lookup_order(oid).get("status")
                   for oid, want in case["expected_db_state"].items()
                   if store_tools.lookup_order(oid).get("status") != want}
    missing_clauses = [c for c in case["expected_clauses"] if c not in result["answer"]]
    checks = {
        "tools_ok": not missing and not forbidden,
        "state_ok": not wrong_state,
        "clauses_ok": not missing_clauses,
        "honest": not false_promise(result),
        "finished": not result["stopped"],
    }
    route = result.get("route")
    return {**checks, "success": all(checks.values()), "called": called,
            "missing_tools": missing, "forbidden_called": forbidden,
            "wrong_state": wrong_state, "missing_clauses": missing_clauses,
            # Router accuracy is reported, not part of success: a wrong route can still end well.
            "route_ok": None if route is None else route["intent"] == case.get("expected_intent"),
            "cite_appended": any(t["tool"] == "_cite_appended" for t in result["trace"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["single", "routed"], default="single")
    parser.add_argument("--cases", type=Path, default=ROOT / "evals" / "cases_phase2.jsonl")
    parser.add_argument("--only", nargs="*", help="case ids to run")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    if args.agent == "routed":
        from agents.routed_agent import run_routed as agent_fn
    else:
        agent_fn = run_agent

    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]

    out_dir = ROOT / "evals" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.agent}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

    rows = []
    mark = lambda ok: "-" if ok is None else ("ok" if ok else "FAIL")  # noqa: E731
    print(f"{'case':<6} {'type':<8} {'route':<6} {'tools':<6} {'state':<6} {'cite':<6} {'honest':<7} "
          f"{'pass':<5} {'nudge':>5} {'secs':>6}")
    with out_path.open("w") as out:
        for case in cases:
            fresh_store()
            try:
                result = agent_fn(case["message"], model=args.model)
            except Exception as exc:  # keep the run going if one case errors
                result = {"answer": f"ERROR: {exc}", "trace": [], "steps": 0,
                          "tokens": {}, "latency_s": 0, "stopped": True}
            s = score(case, result)
            nudges = result.get("nudges", 0)
            rows.append(s | {"latency_s": result["latency_s"], "nudges": nudges,
                             "escalated": result.get("escalated", False)})
            print(f"{case['id']:<6} {case['type']:<8} {mark(s['route_ok']):<6} {mark(s['tools_ok']):<6} "
                  f"{mark(s['state_ok']):<6} {mark(s['clauses_ok']):<6} {mark(s['honest']):<7} "
                  f"{mark(s['success']):<5} {nudges:>5} {result['latency_s']:>6}")
            out.write(json.dumps({"case": case, "result": result, "score": s}) + "\n")

    n = max(len(rows), 1)
    passed = sum(r["success"] for r in rows)
    avg = sum(r["latency_s"] for r in rows) / n
    print(f"\nAgent: {args.agent}   Task success: {passed}/{len(rows)}   "
          f"False promises: {sum(1 for r in rows if not r['honest'])}   Avg latency: {avg:.1f}s")
    if args.agent == "routed":
        routed = [r for r in rows if r["route_ok"] is not None]
        print(f"Router accuracy: {sum(r['route_ok'] for r in routed)}/{len(routed)}   "
              f"Escalated to planner: {sum(r['escalated'] for r in rows)}   "
              f"Citations added by harness: {sum(r['cite_appended'] for r in rows)}")
    else:
        print(f"Cases nudged: {sum(1 for r in rows if r['nudges'])}")
    print(f"Full traces: {out_path}")


if __name__ == "__main__":
    main()
