"""Store tools the action agent calls. Plain Python, no framework.

Design rule: tools enforce hard business limits (no double refunds, the $500
approval threshold, same-category swaps). Policy judgment, such as whether a
request is inside the return window, is the agent's job. The eval checks that
judgment through the final database state.
"""
from __future__ import annotations

import os
import re
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "store.db"
APPROVAL_THRESHOLD = 500.00  # policy clause 6.1


def _db_path() -> Path:
    return Path(os.getenv("STORE_DB", DEFAULT_DB))


def _connect() -> sqlite3.Connection:
    path = _db_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python data/generate.py")
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def _today(con: sqlite3.Connection) -> date:
    return date.fromisoformat(con.execute("SELECT value FROM meta WHERE key='today'").fetchone()[0])


def _clean_id(order_id: str) -> str:
    return str(order_id).lower().replace("order", "").replace("#", "").strip()


def _get_order(con: sqlite3.Connection, order_id: str):
    return con.execute(
        """SELECT o.order_id, o.customer_id, o.product_id, o.quantity, o.total,
                  o.order_date, o.delivered_date, o.status,
                  p.name AS product_name, p.category, p.price, p.final_sale
           FROM orders o JOIN products p ON p.product_id = o.product_id
           WHERE o.order_id = ?""",
        (_clean_id(order_id),),
    ).fetchone()


def lookup_order(order_id: str) -> dict:
    """Look up an order: product, status, dates, total, and days since delivery."""
    with closing(_connect()) as con:
        row = _get_order(con, order_id)
        if row is None:
            return {"found": False, "error": f"Order {order_id} not found"}
        today = _today(con)
    delivered = row["delivered_date"]
    days = (today - date.fromisoformat(delivered)).days if delivered else None
    is_delivered = row["status"] == "delivered" and days is not None
    return {
        "found": True,
        "order_id": row["order_id"],
        "product_id": row["product_id"],
        "product_name": row["product_name"],
        "category": row["category"],
        "quantity": row["quantity"],
        "total": row["total"],
        "order_date": row["order_date"],
        "delivered_date": delivered,
        "days_since_delivery": days,
        "status": row["status"],
        "final_sale": bool(row["final_sale"]),
        # Eligibility computed in code so the model never does date math.
        "policy_facts": {
            "can_cancel_before_shipping": row["status"] == "processing",           # clause 1.4
            "within_30_day_return_window": is_delivered and days <= 30,            # clauses 1.1, 3.1
            "within_7_day_damage_report": is_delivered and days <= 7,              # clause 4.1
            "within_60_day_damage_replacement": is_delivered and days <= 60,       # clause 4.2
            "refund_needs_supervisor_approval": row["total"] > APPROVAL_THRESHOLD,  # clause 6.1
        },
    }


def issue_refund(order_id: str, reason: str, amount: float | None = None) -> dict:
    """Refund an order. Refunds over $500 are queued for human approval (clause 6.1)."""
    with closing(_connect()) as con:
        row = _get_order(con, order_id)
        if row is None:
            return {"ok": False, "error": f"Order {order_id} not found"}
        if row["status"] not in ("delivered", "processing"):
            return {"ok": False, "error": f"Order is '{row['status']}' and cannot be refunded"}

        amount = row["total"] if amount is None else round(float(amount), 2)
        if amount <= 0 or amount > row["total"]:
            return {"ok": False, "error": f"Amount must be between 0 and the order total {row['total']}"}

        if amount > APPROVAL_THRESHOLD:
            con.execute("UPDATE orders SET status='refund_pending_approval' WHERE order_id=?",
                        (row["order_id"],))
            con.commit()
            return {"ok": True, "status": "refund_pending_approval", "amount": amount,
                    "policy_clause": "6.1",
                    "message": "Refunds over $500 need supervisor approval. The request is queued."}

        con.execute("INSERT INTO refunds (order_id, amount, reason, created_at) VALUES (?,?,?,?)",
                    (row["order_id"], amount, reason, _today(con).isoformat()))
        con.execute("UPDATE orders SET status='refunded' WHERE order_id=?", (row["order_id"],))
        con.commit()
    return {"ok": True, "status": "refunded", "order_id": row["order_id"], "amount": amount}


def create_replacement(order_id: str, reason: str, new_product_id: str | None = None) -> dict:
    """Ship a replacement for a delivered order. new_product_id must be in the same category."""
    with closing(_connect()) as con:
        row = _get_order(con, order_id)
        if row is None:
            return {"ok": False, "error": f"Order {order_id} not found"}
        if row["status"] != "delivered":
            return {"ok": False, "error": f"Order is '{row['status']}'. Only delivered orders can be replaced"}

        new_id = (new_product_id or row["product_id"]).strip().upper()
        new = con.execute("SELECT product_id, name, category, price FROM products WHERE product_id=?",
                          (new_id,)).fetchone()
        if new is None:
            return {"ok": False, "error": f"Product {new_id} not found"}
        if new["category"] != row["category"]:
            return {"ok": False, "error": "Replacement must be in the same category (clause 4.2)"}

        diff = round((new["price"] - row["price"]) * row["quantity"], 2)
        con.execute(
            """INSERT INTO replacements (order_id, new_product_id, price_difference, reason, created_at)
               VALUES (?,?,?,?,?)""",
            (row["order_id"], new_id, diff, reason, _today(con).isoformat()),
        )
        con.execute("UPDATE orders SET status='replacement_issued' WHERE order_id=?", (row["order_id"],))
        con.commit()
    return {
        "ok": True,
        "status": "replacement_issued",
        "order_id": row["order_id"],
        "new_product_id": new_id,
        "new_product_name": new["name"],
        "price_difference": diff,
        "customer_owes": max(diff, 0.0),
        "refund_to_customer": max(-diff, 0.0),
    }


_FILLER = {"the", "a", "an", "one", "model", "for", "to", "my", "instead", "different", "new", "other"}


def search_products(query: str, category: str | None = None, limit: int = 5) -> dict:
    """Find products by name keywords, for example '1200W blender'."""
    terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if t not in _FILLER]
    sql, params = "SELECT * FROM products", ()
    if category:
        sql, params = sql + " WHERE category = ?", (category.lower(),)
    with closing(_connect()) as con:
        rows = con.execute(sql, params).fetchall()
    scored = []
    for r in rows:
        name = r["name"].lower()
        score = sum(1 for t in terms if t in name)
        if score:
            scored.append((score, r))
    scored.sort(key=lambda x: (-x[0], x[1]["price"]))
    return {"results": [
        {"product_id": r["product_id"], "name": r["name"], "category": r["category"],
         "price": r["price"], "final_sale": bool(r["final_sale"])}
        for _, r in scored[:limit]
    ]}
