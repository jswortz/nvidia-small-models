"""Phase 3 routed agent: the model understands and writes; code decides and acts.

    1. Router (Nano) classifies the message into a structured request.
    2. Code looks up the order and, for swaps, the requested product.
    3. The policy engine (plain rules) decides: act, decline, inform, or ask.
    4. Code runs the action, if any.
    5. Writer (Nano) explains the decision to the customer and cites the clauses.
    Requests the router cannot classify go to the planner (the Phase 2 agent).
    A message naming two or more real orders runs steps 1 to 4 once per order, then one reply covers all.

Run one message from /workspace:
    python -m agents.routed_agent "My blender from order 1187 arrived cracked..."
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import asdict
from types import SimpleNamespace
from urllib.parse import urlparse

from openai import OpenAI

from agents import policy_engine, router
from agents.nat_profiling import NatLLMClient, step
from agents.single_agent import run_agent as run_planner
from tools import policy_search
from tools.registry import call_tool

MODEL = os.getenv("AGENT_MODEL", "nvidia/NVIDIA-Nemotron-Nano-9B-v2")
# Optional: send only router calls to another served model, such as the LoRA adapter "router-sft"
# (served next to the base model with vLLM --lora-modules). Pair it with ROUTER_PROMPT=short, the
# prompt the adapter was trained on. Unset means the router uses MODEL like every other call.
ROUTER_MODEL = os.getenv("ROUTER_MODEL")
# Optional: point the planner at hosted Nemotron Super. Unset means the local model plans too.
PLANNER_MODEL = os.getenv("PLANNER_MODEL", MODEL)
PLANNER_BASE_URL = os.getenv("PLANNER_BASE_URL")
# The hosted free tier sometimes hangs or returns 500s. The SDK default is a 600 s timeout with 2 retries,
# so cap it and fall back to the local model instead.
PLANNER_TIMEOUT_S = float(os.getenv("PLANNER_TIMEOUT_S", "30"))
ACTION_TOOLS = ("issue_refund", "create_replacement")

WRITER_PROMPT = """/no_think You write the final reply to a retail customer. The decision has already been made by store systems. Do not change it.

Rules:
1. Two to four sentences, friendly and direct.
2. Describe what happened using only "action_result". If "action_result" is null, nothing was refunded, replaced, or cancelled: do not say it was, and do not say you will do it.
3. Cite every clause listed in "clauses" as (policy X.Y). Cite no other clause numbers; if "clauses" is empty, cite none.
4. If "outcome" is "need_info", ask for exactly what is missing.
5. If "outcome" is "info" and "retrieved_policy" is given, answer from it and cite the clause that answers the question."""

MULTI_WRITER_PROMPT = WRITER_PROMPT.replace("1. Two to four sentences", "1. Two to six sentences") + """
6. The message covers several orders. "requests" holds one decision per order. Cover every order, in the order given, and name each order number. Each request's rules 2 to 5 apply only to that order.
7. Say nothing the requests and clauses do not state: no claims about whether an action can be undone, how long it takes, or what happens next."""

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
# A four digit number not preceded by "$", so prices such as $1299 are not read as orders.
_ORDER_NUMBER = re.compile(r"(?<!\$)\b(\d{4})\b")


def find_orders(message: str) -> list[str]:
    """Distinct order numbers in the message that exist in the store, in the order they appear."""
    seen: list[str] = []
    for number in _ORDER_NUMBER.findall(message):
        if number not in seen and json.loads(call_tool("lookup_order", {"order_id": number})).get("found"):
            seen.append(number)
    return seen


def _client(base_url: str | None = None) -> OpenAI:
    if base_url:  # hosted planner
        return OpenAI(base_url=base_url, api_key=os.getenv("NVIDIA_API_KEY"),
                      timeout=PLANNER_TIMEOUT_S, max_retries=1)
    return OpenAI(base_url=os.getenv("OPENAI_BASE_URL", "http://nim:8000/v1"),
                  api_key=os.getenv("OPENAI_API_KEY", "local"))


class TimedClient:
    """Wraps a client and logs every chat call: server, model, seconds, and any error.

    The log shows where a slow case spent its time, for example a hosted call that hung.
    """

    def __init__(self, client, log: list[dict]):
        self._client, self._log = client, log
        base_url = getattr(client, "base_url", None)
        self.server = urlparse(str(base_url)).hostname if base_url else "unknown"
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        entry = {"server": self.server, "model": kwargs.get("model"), "ok": False}
        start = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**kwargs)
            entry["ok"] = True
            return resp
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"[:200]
            raise
        finally:
            entry["seconds"] = round(time.perf_counter() - start, 2)
            self._log.append(entry)


def _add(tokens: dict, usage: dict) -> None:
    tokens["prompt"] += usage.get("prompt", 0)
    tokens["completion"] += usage.get("completion", 0)


def write_reply(client, model: str, context: dict) -> tuple[str, dict]:
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=300,
        messages=[{"role": "system", "content": WRITER_PROMPT},
                  {"role": "user", "content": json.dumps(context, indent=2)}])
    usage = {"prompt": resp.usage.prompt_tokens, "completion": resp.usage.completion_tokens} if resp.usage \
        else {"prompt": 0, "completion": 0}
    return _THINK.sub("", resp.choices[0].message.content or "").strip(), usage


def run_multi(user_message: str, order_ids: list[str], client: OpenAI, model: str) -> dict:
    """One message, several orders: route, decide, and act once per order, then write one reply.

    The planner is never called here. It sees the whole message and could act on every order
    again, so a request the engine cannot place is answered by asking the customer instead.
    """
    start = time.perf_counter()
    tokens = {"prompt": 0, "completion": 0}
    trace: list[dict] = []

    def tool(name: str, args: dict) -> dict:
        result = json.loads(call_tool(name, args))
        trace.append({"step": len(trace) + 1, "tool": name, "arguments": json.dumps(args), "result": result})
        return result

    routes, decisions, requests, all_clauses = [], [], [], []
    for order_id in order_ids:
        # Same router, same prompt: the message is passed whole with a note naming the order to classify.
        with step("router"):
            route, usage = router.classify(f"{user_message}\n\n(Classify only the request about order {order_id}.)",
                                           client, ROUTER_MODEL or model)
        _add(tokens, usage)
        route["order_id"] = order_id  # Code found this order; the model does not get to change it.
        trace.append({"step": len(trace) + 1, "tool": "_route", "arguments": order_id, "result": route})

        with step("lookup"):
            order = tool("lookup_order", {"order_id": order_id})
            product = None
            if route["intent"] == "replacement" and route["new_product"]:
                hits = tool("search_products", {"query": route["new_product"]})["results"]
                product = hits[0] if hits else None

        with step("engine"):
            decision = policy_engine.decide(route, order, product)
        if decision.outcome == "escalate":
            decision = policy_engine.Decision(
                "need_info", f"The request about order {order_id} is not one the store systems can handle. "
                             "Ask the customer what they need for this order.")
        trace.append({"step": len(trace) + 1, "tool": "_decision", "arguments": order_id, "result": asdict(decision)})

        action_result = None
        if decision.outcome == "act":
            with step("action"):
                action_result = tool(decision.action, decision.action_args)
        clause_text = []
        with step("lookup"):
            for clause_id in decision.clauses:
                clause = policy_search.get_clause(clause_id)
                trace.append({"step": len(trace) + 1, "tool": "get_clause", "arguments": clause_id, "result": clause})
                clause_text.append({k: clause.get(k) for k in ("clause_id", "title", "text")})
                if clause_id not in all_clauses:
                    all_clauses.append(clause_id)

        routes.append(route)
        decisions.append(asdict(decision))
        requests.append({
            "order": {k: order.get(k) for k in ("order_id", "product_name", "status", "delivered_date")},
            "request": route["intent"],
            "outcome": decision.outcome,
            "decision": decision.reason,
            "clauses": clause_text,
            "action_result": action_result,
        })

    with step("writer"):
        resp = client.chat.completions.create(
            model=model, temperature=0, max_tokens=400,
            messages=[{"role": "system", "content": MULTI_WRITER_PROMPT},
                      {"role": "user", "content": json.dumps({"customer_message": user_message,
                                                              "requests": requests}, indent=2)}])
        if resp.usage:
            _add(tokens, {"prompt": resp.usage.prompt_tokens, "completion": resp.usage.completion_tokens})
        answer = _THINK.sub("", resp.choices[0].message.content or "").strip()

        missing = [c for c in all_clauses if c not in answer]
        if missing:
            answer = f"{answer} (policy {', '.join(missing)})"
            trace.append({"step": len(trace) + 1, "tool": "_cite_appended", "arguments": "", "result": {"clauses": missing}})

    # "route" and "decision" keep the single-order shape for the eval; the full lists sit beside them.
    return {"answer": answer, "trace": trace, "steps": 2, "tokens": tokens, "nudges": 0,
            "latency_s": round(time.perf_counter() - start, 2), "stopped": False,
            "route": routes[0], "decision": decisions[0], "routes": routes, "decisions": decisions,
            "escalated": False, "multi_order": True}


def run_routed(user_message: str, client: OpenAI | None = None, model: str = MODEL,
               planner_client: OpenAI | None = None) -> dict:
    calls: list[dict] = []
    # NatLLMClient emits NAT profiler events; TimedClient keeps the per-call log in result["model_calls"].
    result = _run_routed(user_message, TimedClient(NatLLMClient(client or _client()), calls), model,
                         planner_client, calls)
    result["model_calls"] = calls
    return result


def _run_routed(user_message: str, client, model: str, planner_client, calls: list[dict]) -> dict:
    with step("lookup"):
        order_ids = find_orders(user_message)
    if len(order_ids) >= 2:
        return run_multi(user_message, order_ids, client, model)
    start = time.perf_counter()
    tokens = {"prompt": 0, "completion": 0}
    trace: list[dict] = []

    def tool(name: str, args: dict) -> dict:
        result = json.loads(call_tool(name, args))
        trace.append({"step": len(trace) + 1, "tool": name, "arguments": json.dumps(args), "result": result})
        return result

    # 1. Understand
    with step("router"):
        route, usage = router.classify(user_message, client, ROUTER_MODEL or model)
    _add(tokens, usage)
    trace.append({"step": 1, "tool": "_route", "arguments": "", "result": route})

    # 2. Gather facts
    order = product = None
    with step("lookup"):
        if route["intent"] not in ("policy_question", "other") and route["order_id"]:
            order = tool("lookup_order", {"order_id": route["order_id"]})
        if route["intent"] == "replacement" and route["new_product"] and order and order.get("found"):
            # Search every category: the engine, not the search, decides whether a cross-category swap is allowed.
            hits = tool("search_products", {"query": route["new_product"]})["results"]
            product = hits[0] if hits else None

    # 3. Decide
    with step("engine"):
        decision = policy_engine.decide(route, order, product)
    trace.append({"step": len(trace) + 1, "tool": "_decision", "arguments": "", "result": asdict(decision)})

    if decision.outcome == "escalate":
        with step("planner"):
            return _escalate(user_message, client, model, planner_client, calls, trace, route, start)

    # 4. Act
    action_result = None
    if decision.outcome == "act":
        with step("action"):
            action_result = tool(decision.action, decision.action_args)

    # 5. Explain, grounded in the actual policy text
    retrieved = None
    clause_text = []
    with step("lookup"):
        if decision.outcome == "info" and route["intent"] == "policy_question":
            retrieved = tool("search_policy", {"query": user_message})["results"]
        for clause_id in decision.clauses:
            clause = policy_search.get_clause(clause_id)
            trace.append({"step": len(trace) + 1, "tool": "get_clause", "arguments": clause_id, "result": clause})
            clause_text.append({k: clause.get(k) for k in ("clause_id", "title", "text")})

    context = {
        "customer_message": user_message,
        "outcome": decision.outcome,
        "decision": decision.reason,
        "clauses": clause_text,
        "action_result": action_result,
        "order": {k: order.get(k) for k in ("order_id", "product_name", "status", "delivered_date")}
        if order and order.get("found") else None,
        "retrieved_policy": retrieved,
    }
    with step("writer"):
        answer, usage = write_reply(client, model, context)
        _add(tokens, usage)

        # Guarantee every decided clause is cited, and record when the harness had to add one.
        missing = [c for c in decision.clauses if c not in answer]
        if missing:
            answer = f"{answer} (policy {', '.join(missing)})"
            trace.append({"step": len(trace) + 1, "tool": "_cite_appended", "arguments": "", "result": {"clauses": missing}})

    return {"answer": answer, "trace": trace, "steps": 2, "tokens": tokens, "nudges": 0,
            "latency_s": round(time.perf_counter() - start, 2), "stopped": False,
            "route": route, "decision": asdict(decision), "escalated": False}


def _escalate(user_message: str, client, model: str, planner_client, calls: list[dict],
              trace: list[dict], route: dict, start: float) -> dict:
    """Hand the message to the planner (the Phase 2 agent), falling back to local Nano if a hosted call fails."""
    hosted = planner_client or (_client(PLANNER_BASE_URL) if PLANNER_BASE_URL else None)
    if hosted is None:
        result = run_planner(user_message, client=client, model=PLANNER_MODEL)
    else:
        result = run_planner(user_message, client=TimedClient(NatLLMClient(hosted), calls), model=PLANNER_MODEL)
    fallback = False
    if hosted is not None and result.get("error"):
        acted = any(t["tool"] in ACTION_TOOLS and t["result"].get("ok") for t in result["trace"])
        # Retry on the local model only if the hosted planner changed nothing in the store.
        if not acted:
            fallback = True
            first = result
            result = run_planner(user_message, client=client, model=model)
            result["trace"] = first["trace"] + [{"step": 0, "tool": "_planner_fallback", "arguments": "",
                                                 "result": {"error": first["error"]}}] + result["trace"]
            result["nudges"] += first["nudges"]
            for k in ("prompt", "completion"):
                result["tokens"][k] += first["tokens"][k]
    result["trace"] = trace + result["trace"]
    result["route"] = route
    result["escalated"] = True
    result["planner_fallback"] = fallback
    result["latency_s"] = round(time.perf_counter() - start, 2)
    return result


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "Where is my order 1187?"
    result = run_routed(text)
    print("ROUTE:", json.dumps(result["route"]))
    for t in result["trace"]:
        if t["tool"] != "_route":
            print(f"  {t['tool']}({t['arguments']})\n    -> {json.dumps(t['result'])[:200]}")
    print(f"\nANSWER: {result['answer']}")
    print(f"latency={result['latency_s']}s tokens={result['tokens']} escalated={result['escalated']}")
    for call in result["model_calls"]:
        print(f"  model call: {call['server']} {call['seconds']}s" + ("" if call["ok"] else f" FAILED {call['error']}"))
