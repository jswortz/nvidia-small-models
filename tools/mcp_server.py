"""Expose the store tools over MCP.

Run from /workspace:
    python -m tools.mcp_server                    # stdio, for local MCP clients
    python -m tools.mcp_server streamable-http    # HTTP at http://<host>:8100/mcp
"""
from __future__ import annotations

import sys

from mcp.server.fastmcp import FastMCP

from tools import policy_search, store_tools

mcp = FastMCP("retail-store", host="0.0.0.0", port=8100)


@mcp.tool()
def lookup_order(order_id: str) -> dict:
    """Look up an order by id. Returns product, status, delivery date, days since delivery, and total."""
    return store_tools.lookup_order(order_id)


@mcp.tool()
def issue_refund(order_id: str, reason: str, amount: float | None = None) -> dict:
    """Refund an order after checking the policy allows it. Refunds over $500 are queued for approval."""
    return store_tools.issue_refund(order_id, reason, amount)


@mcp.tool()
def create_replacement(order_id: str, reason: str, new_product_id: str | None = None) -> dict:
    """Ship a replacement for a delivered order, optionally a different product in the same category."""
    return store_tools.create_replacement(order_id, reason, new_product_id)


@mcp.tool()
def search_products(query: str, category: str | None = None) -> dict:
    """Find products by name keywords, for example '1200W blender'."""
    return store_tools.search_products(query, category)


@mcp.tool()
def search_policy(query: str) -> dict:
    """Search the returns, refunds, and warranty policy. Returns numbered clauses to cite."""
    return policy_search.search_policy(query)


if __name__ == "__main__":
    mcp.run(transport=sys.argv[1] if len(sys.argv) > 1 else "stdio")
