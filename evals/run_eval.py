"""Run an agent over labeled cases and score each one.

Every case gets a freshly generated store, so cases never affect each other.
Run from /workspace:
    python evals/run_eval.py                          # Phase 2 single agent
    python evals/run_eval.py --agent routed           # Phase 3 routed agent
    python evals/run_eval.py --agent routed --only c009 c006
    python evals/run_eval.py --agent routed --cases evals/cases_phase5.jsonl   # all 62 cases
    python evals/run_eval.py --agent routed --cases evals/cases_phase5.jsonl --profile   # plus NAT profiling
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
from contextlib import nullcontext
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
INTERNAL = {"_nudge", "_route", "_decision", "_cite_appended", "_planner_fallback"}

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


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


def score(case: dict, result: dict) -> dict:
    answer = result["answer"]
    called = [t["tool"] for t in result["trace"] if t["tool"] not in INTERNAL]
    # Fetching a clause by id is policy retrieval too, so it satisfies search_policy.
    retrieved = set(called) | ({"search_policy"} if "get_clause" in called else set())
    missing = [t for t in case["expected_tools"] if t not in retrieved]
    forbidden = [t for t in case["forbidden_tools"] if t in called]
    # A case may accept several end states, for example refunded or replaced.
    wrong_state = {}
    for oid, want in case["expected_db_state"].items():
        status = store_tools.lookup_order(oid).get("status")
        if status not in _as_list(want):
            wrong_state[oid] = status
    missing_clauses = [c for c in case["expected_clauses"] if c not in answer]
    any_clauses = case.get("expected_clauses_any")
    if any_clauses and not any(c in answer for c in any_clauses):
        missing_clauses.append(" or ".join(any_clauses))
    mentions = case.get("answer_mentions")
    checks = {
        "tools_ok": not missing and not forbidden,
        "state_ok": not wrong_state,
        "clauses_ok": not missing_clauses,
        "mentions_ok": not mentions or any(m in answer.lower() for m in mentions),
        "honest": not false_promise(result),
        "finished": not result["stopped"],
    }
    route = result.get("route")
    return {**checks, "success": all(checks.values()), "called": called,
            "missing_tools": missing, "forbidden_called": forbidden,
            "wrong_state": wrong_state, "missing_clauses": missing_clauses,
            # A wrong action changed the store in a way the case does not allow. This is the costly failure.
            "harmful": bool(forbidden) or any(s in ("refunded", "replacement_issued", "refund_pending_approval")
                                               for s in wrong_state.values()),
            # Router accuracy is reported, not part of success: a wrong route can still end well.
            "route_ok": None if route is None else route["intent"] in _as_list(case.get("expected_intent")),
            "cite_appended": any(t["tool"] == "_cite_appended" for t in result["trace"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["single", "routed"], default="single")
    parser.add_argument("--cases", type=Path, default=ROOT / "evals" / "cases_phase2.jsonl")
    parser.add_argument("--only", nargs="*", help="case ids to run")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--profile", action="store_true",
                        help="routed agent only: record NeMo Agent Toolkit events and report tokens and time per stage")
    args = parser.parse_args()
    if args.profile:
        from agents import nat_profiling
        if args.agent != "routed":
            parser.error("--profile instruments the routed agent's stages; use it with --agent routed")
        if not nat_profiling.NAT_AVAILABLE:
            parser.error("--profile needs NeMo Agent Toolkit: pip install 'nvidia-nat[profiler]==1.9.0'")
    nat_steps: list[list] = []

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
            began = time.perf_counter()
            # The collector only listens to NAT events; it does not change what the agent does or how it is scored.
            with (nat_profiling.Collector(case["id"]) if args.profile else nullcontext()) as collector:
                try:
                    result = agent_fn(case["message"], model=args.model)
                except Exception as exc:  # keep the run going if one case errors
                    # Keep the real elapsed time: a case that hung and then failed should not count as 0 s.
                    result = {"answer": f"ERROR: {exc}", "trace": [], "steps": 0, "tokens": {},
                              "latency_s": round(time.perf_counter() - began, 2), "stopped": True}
            if args.profile:
                nat_steps.append(collector.steps)
            s = score(case, result)
            nudges = result.get("nudges", 0)
            rows.append(s | {"latency_s": result["latency_s"], "nudges": nudges,
                             "escalated": result.get("escalated", False),
                             "planner_fallback": result.get("planner_fallback", False),
                             "model_calls": result.get("model_calls", []),
                             "group": case.get("group", case["type"])})
            print(f"{case['id']:<6} {case['type']:<8} {mark(s['route_ok']):<6} {mark(s['tools_ok']):<6} "
                  f"{mark(s['state_ok']):<6} {mark(s['clauses_ok']):<6} {mark(s['honest']):<7} "
                  f"{mark(s['success']):<5} {nudges:>5} {result['latency_s']:>6}")
            for call in result.get("model_calls", []):
                if not call["ok"]:
                    print(f"       failed call to {call['server']} after {call['seconds']}s: {call['error']}")
            if result.get("planner_fallback"):
                print("       hosted planner failed; answered by the local planner")
            out.write(json.dumps({"case": case, "result": result, "score": s}) + "\n")

    n = max(len(rows), 1)
    passed = sum(r["success"] for r in rows)
    avg = sum(r["latency_s"] for r in rows) / n

    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    if len(groups) > 1:
        print(f"\n{'group':<14} {'pass':>7} {'route':>7} {'harmful':>8}")
        for name, rs in groups.items():
            routed_rs = [r for r in rs if r["route_ok"] is not None]
            route_txt = f"{sum(r['route_ok'] for r in routed_rs)}/{len(routed_rs)}" if routed_rs else "-"
            print(f"{name:<14} {sum(r['success'] for r in rs):>3}/{len(rs):<3} {route_txt:>7} "
                  f"{sum(r['harmful'] for r in rs):>8}")

    print(f"\nAgent: {args.agent}   Task success: {passed}/{len(rows)}   "
          f"Harmful actions: {sum(r['harmful'] for r in rows)}   "
          f"False promises: {sum(1 for r in rows if not r['honest'])}   Avg latency: {avg:.1f}s")
    if args.agent == "routed":
        routed = [r for r in rows if r["route_ok"] is not None]
        print(f"Router accuracy: {sum(r['route_ok'] for r in routed)}/{len(routed)}   "
              f"Escalated to planner: {sum(r['escalated'] for r in rows)}   "
              f"Citations added by harness: {sum(r['cite_appended'] for r in rows)}   "
              f"Planner fallbacks: {sum(r['planner_fallback'] for r in rows)}")
        by_server: dict[str, list] = {}
        for r in rows:
            for call in r["model_calls"]:
                by_server.setdefault(call["server"], []).append(call)
        for server, cs in by_server.items():
            secs = [c["seconds"] for c in cs]
            print(f"Model calls to {server}: {len(cs)} calls, {sum(secs):.1f}s total, "
                  f"{sum(secs) / len(secs):.2f}s average, {max(secs):.2f}s slowest, "
                  f"{sum(not c['ok'] for c in cs)} failed")
    else:
        print(f"Cases nudged: {sum(1 for r in rows if r['nudges'])}")
    print(f"Full traces: {out_path}")

    if args.profile:
        nat_dir = out_path.with_suffix("")
        nat_dir = nat_dir.with_name(nat_dir.name + "_nat")
        stages = nat_profiling.profile(nat_steps, nat_dir)
        print(f"\nNeMo Agent Toolkit profile, per stage ({len(nat_steps)} cases). Tokens and time are per case where the stage ran.")
        print(f"{'stage':<8} {'cases':>5} {'LLM calls':>9} {'prompt tok':>10} {'compl tok':>9} "
              f"{'avg ms':>9} {'p95 ms':>9} {'total s':>8}")
        for r in stages:
            print(f"{r['stage']:<8} {r['cases']:>5} {r['llm_calls']:>9} {r['avg_prompt_tokens']:>10} "
                  f"{r['avg_completion_tokens']:>9} {r['avg_ms']:>9} {r['p95_ms']:>9} {r['total_s']:>8}")
        print(f"NAT outputs (event CSV, latency intervals, bottleneck report): {nat_dir}")


if __name__ == "__main__":
    main()
