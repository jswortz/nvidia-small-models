"""OpenAI-style tool schemas plus a dispatcher.

Use this when an agent calls tools directly through the chat completions API
(Phase 2 baseline, quick tests). mcp_server.py exposes the same functions over MCP.
"""
from __future__ import annotations

import json

from tools import policy_search, store_tools

FUNCTIONS = {
    "lookup_order": store_tools.lookup_order,
    "issue_refund": store_tools.issue_refund,
    "create_replacement": store_tools.create_replacement,
    "search_products": store_tools.search_products,
    "search_policy": policy_search.search_policy,
}


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


OPENAI_TOOLS = [
    _fn("lookup_order",
        "Look up an order by id. Returns product, status, delivery date, days since delivery, and total.",
        {"order_id": {"type": "string", "description": "Order id, for example '1187'"}},
        ["order_id"]),
    _fn("issue_refund",
        "Refund an order. Only call after checking the policy allows it. "
        "Refunds over $500 are queued for supervisor approval.",
        {"order_id": {"type": "string"},
         "reason": {"type": "string", "description": "Short reason, citing the policy clause"},
         "amount": {"type": "number", "description": "Partial amount. Omit for a full refund"}},
        ["order_id", "reason"]),
    _fn("create_replacement",
        "Ship a replacement for a delivered order. Optionally swap to a different product "
        "in the same category; returns the price difference.",
        {"order_id": {"type": "string"},
         "reason": {"type": "string", "description": "Short reason, citing the policy clause"},
         "new_product_id": {"type": "string", "description": "Product id to swap to. Omit for the same product"}},
        ["order_id", "reason"]),
    _fn("search_products",
        "Find products by name keywords, for example '1200W blender'. Use to get a product id for a swap.",
        {"query": {"type": "string"},
         "category": {"type": "string", "enum": ["kitchen", "home", "electronics"]}},
        ["query"]),
    _fn("search_policy",
        "Search the returns, refunds, and warranty policy. Returns numbered clauses to cite.",
        {"query": {"type": "string"}},
        ["query"]),
]


def call_tool(name: str, arguments: str | dict) -> str:
    """Run a tool by name and return its result as a JSON string for the tool message."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return json.dumps({"ok": False, "error": f"Arguments were not valid JSON: {exc}"})
    fn = FUNCTIONS.get(name)
    if fn is None:
        return json.dumps({"ok": False, "error": f"Unknown tool '{name}'"})
    try:
        result = fn(**arguments)
    except TypeError as exc:
        result = {"ok": False, "error": f"Bad arguments for {name}: {exc}"}
    return json.dumps(result)
