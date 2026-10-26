"""Run the agent over labeled cases and score each one.

Every case gets a freshly generated store, so cases never affect each other.
Run from /workspace:
    python evals/run_eval.py                     # all Phase 2 cases
    python evals/run_eval.py --only c009 c006    # selected cases
"""
from __future__ import annotations

import argparse
import json
import os
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


def score(case: dict, result: dict) -> dict:
    called = [t["tool"] for t in result["trace"] if t["tool"] != "_nudge"]
    missing = [t for t in case["expected_tools"] if t not in called]
    forbidden = [t for t in case["forbidden_tools"] if t in called]
    wrong_state = {oid: store_tools.lookup_order(oid).get("status")
                   for oid, want in case["expected_db_state"].items()
                   if store_tools.lookup_order(oid).get("status") != want}
    missing_clauses = [c for c in case["expected_clauses"] if c not in result["answer"]]
    checks = {
        "tools_ok": not missing and not forbidden,
        "state_ok": not wrong_state,
        "clauses_ok": not missing_clauses,
        "finished": not result["stopped"],
    }
    return {**checks, "success": all(checks.values()), "called": called,
            "missing_tools": missing, "forbidden_called": forbidden,
            "wrong_state": wrong_state, "missing_clauses": missing_clauses}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=ROOT / "evals" / "cases_phase2.jsonl")
    parser.add_argument("--only", nargs="*", help="case ids to run")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]

    out_dir = ROOT / "evals" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase2_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

    rows = []
    print(f"{'case':<6} {'type':<8} {'tools':<6} {'state':<6} {'cite':<6} {'pass':<5} {'steps':>5} "
          f"{'nudge':>5} {'secs':>6}")
    with out_path.open("w") as out:
        for case in cases:
            fresh_store()
            try:
                result = run_agent(case["message"], model=args.model)
            except Exception as exc:  # keep the run going if one case errors
                result = {"answer": f"ERROR: {exc}", "trace": [], "steps": 0,
                          "tokens": {}, "latency_s": 0, "stopped": True}
            s = score(case, result)
            nudges = result.get("nudges", 0)
            rows.append(s | {"latency_s": result["latency_s"], "nudges": nudges})
            mark = lambda ok: "ok" if ok else "FAIL"  # noqa: E731
            print(f"{case['id']:<6} {case['type']:<8} {mark(s['tools_ok']):<6} {mark(s['state_ok']):<6} "
                  f"{mark(s['clauses_ok']):<6} {mark(s['success']):<5} {result['steps']:>5} "
                  f"{nudges:>5} {result['latency_s']:>6}")
            out.write(json.dumps({"case": case, "result": result, "score": s}) + "\n")

    passed = sum(r["success"] for r in rows)
    avg = sum(r["latency_s"] for r in rows) / max(len(rows), 1)
    nudged = sum(1 for r in rows if r["nudges"])
    print(f"\nTask success: {passed}/{len(rows)}   Cases nudged: {nudged}   Avg latency: {avg:.1f}s   "
          f"Model: {args.model}")
    print(f"Full traces: {out_path}")


if __name__ == "__main__":
    main()