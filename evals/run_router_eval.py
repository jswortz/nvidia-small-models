"""Phase 6: score the router alone against labeled routes.

Each line of the input file needs "router_input" (the text the router sees) and
"expected_route" (intent, order_id, damaged, photo_provided, new_product).

Scoring follows what the agent does with each field:
- new_product counts as correct when it leads to the same product search result as
  the label (both empty also counts), because the agent only uses it to search.
- For multi-order messages the agent replaces order_id with the order code found,
  so the effective route is scored the same way.

Run from /workspace:
    python evals/run_router_eval.py --cases training/data/router_stress_test.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import router  # noqa: E402

FIELDS = ["intent", "order_id", "damaged", "photo_provided", "new_product"]


def _store() -> None:
    """A fresh read-only store, so product search works without touching data/store.db."""
    tmp = Path(tempfile.mkdtemp())
    subprocess.run([sys.executable, str(ROOT / "data" / "generate.py"), "--db", str(tmp / "store.db"),
                    "--policy", str(tmp / "policy.md")], check=True, capture_output=True)
    os.environ["STORE_DB"], os.environ["STORE_POLICY"] = str(tmp / "store.db"), str(tmp / "policy.md")


def _top_product(name: str | None) -> str | None:
    from tools.store_tools import search_products
    if not name:
        return None
    hits = search_products(name)["results"]
    return hits[0]["product_id"] if hits else f"unmatched:{name}"


def field_scores(route: dict, expected: dict) -> dict[str, bool]:
    scores = {f: route.get(f) == expected.get(f) for f in FIELDS if f != "new_product"}
    scores["new_product"] = _top_product(route.get("new_product")) == _top_product(expected.get("new_product"))
    return scores


def effective_route(route: dict, case: dict) -> dict:
    """The route the agent acts on: multi-order messages get the order code found."""
    focus = case["expected_route"]["order_id"] if case.get("group") == "multi" else None
    return route | {"order_id": focus} if focus else route


def main() -> None:
    # Imported here so training/reward.py can use field_scores without the openai package.
    from openai import OpenAI

    from agents.single_agent import MODEL

    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("ROUTER_MODEL", MODEL))
    parser.add_argument("--base-url", default=os.getenv("ROUTER_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://nim:8000/v1")))
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel requests; keep 1 when latency matters, since batching changes it")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prompt", choices=["full", "short"], default=os.getenv("ROUTER_PROMPT", "full"),
                        help="router prompt: full for the base model, short for a fine-tuned router")
    args = parser.parse_args()

    _store()
    client = OpenAI(base_url=args.base_url, api_key=os.getenv("OPENAI_API_KEY", "local"), timeout=120)
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()][:args.limit]

    def run(case: dict) -> dict:
        start = time.perf_counter()
        route, usage = router.classify(case["router_input"], client, args.model, args.prompt)
        seconds = time.perf_counter() - start
        route = effective_route(route, case)
        scores = field_scores(route, case["expected_route"])
        return {"id": case["id"], "group": case.get("group", "-"), "style": case.get("style", "-"),
                "route": route, "expected": case["expected_route"], "fields": scores,
                "exact": all(scores.values()), "seconds": round(seconds, 3), "prompt_tokens": usage["prompt"],
                "completion_tokens": usage["completion"]}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run, cases))

    out = ROOT / "evals" / "results" / f"router_{args.cases.stem}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def line(name: str, rs: list[dict]) -> str:
        cells = [f"{sum(r['fields'][f] for r in rs) / len(rs):>7.0%}" for f in FIELDS]
        return f"{name:<16} {len(rs):>5} {sum(r['exact'] for r in rs) / len(rs):>7.0%} " + " ".join(cells)

    header = f"{'group':<16} {'cases':>5} {'exact':>7} " + " ".join(f"{f[:7]:>7}" for f in FIELDS)
    for key in ("group", "style"):
        buckets: dict[str, list] = defaultdict(list)
        for r in rows:
            buckets[r[key]].append(r)
        print(f"\nBy {key}\n" + header.replace("group", key, 1))
        for name in sorted(buckets):
            print(line(name, buckets[name]))
    print("\n" + line("ALL", rows))
    print(f"\nModel: {args.model}   Prompt: {args.prompt}   Exact route: {sum(r['exact'] for r in rows)}/{len(rows)}   "
          f"Avg latency: {sum(r['seconds'] for r in rows) / len(rows):.2f}s (workers={args.workers})   "
          f"Avg prompt tokens: {sum(r['prompt_tokens'] for r in rows) / len(rows):.0f}")
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
