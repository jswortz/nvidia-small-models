"""Phase 3 policy engine: plain rules decide what the store does. No model involved.

Input: the router's structured request, the order from lookup_order, and the
product the customer asked for (if any). Output: a Decision the code executes
and the reply writer explains. Every branch names the policy clause it applies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DONE_STATUSES = ("refunded", "refund_pending_approval", "replacement_issued")


@dataclass
class Decision:
    outcome: str                      # "act", "decline", "info", "need_info", or "escalate"
    reason: str                       # one plain sentence the reply writer explains
    clauses: list[str] = field(default_factory=list)
    action: str | None = None         # "issue_refund" or "create_replacement"
    action_args: dict = field(default_factory=dict)


def _refund(order: dict, reason: str, clauses: list[str]) -> Decision:
    if order["policy_facts"]["refund_needs_supervisor_approval"]:
        clauses = clauses + ["6.1"]
        reason += " The amount is over $500, so the refund is queued for supervisor approval."
    return Decision("act", reason, clauses, "issue_refund",
                    {"order_id": order["order_id"], "reason": f"{reason} (policy {', '.join(clauses)})"})


def decide(route: dict, order: dict | None, product: dict | None = None) -> Decision:
    intent = route["intent"]

    if intent == "policy_question":
        return Decision("info", "General policy question. Answer from the retrieved clauses.")
    if intent == "other":
        return Decision("escalate", "Request does not fit a known type. Hand it to the planner.")
    if not route.get("order_id"):
        return Decision("need_info", "No order number was given. Ask the customer for it.")
    if order is None or not order.get("found"):
        return Decision("info", f"Order {route['order_id']} was not found. Ask the customer to check the number.")
    if intent == "order_status":
        return Decision("info", f"Order {order['order_id']} ({order['product_name']}) is {order['status']}.")

    facts, status = order["policy_facts"], order["status"]
    if status in DONE_STATUSES:
        return Decision("decline", f"This order is already {status.replace('_', ' ')}, so there is nothing more to do on it.")

    if intent == "cancel":
        if facts["can_cancel_before_shipping"]:
            return _refund(order, "The order has not shipped yet, so it is cancelled for a full refund.", ["1.4"])
        return Decision("decline", "The order has already shipped, so it can no longer be cancelled. "
                        "A return may be possible after delivery.", ["1.4"])

    if intent == "refund":
        if facts["can_cancel_before_shipping"]:
            return _refund(order, "The order has not shipped yet, so it is cancelled for a full refund.", ["1.4"])
        if status != "delivered":
            return Decision("decline", "The order is still in transit. A return can start after it is delivered.", ["1.1"])
        if route["damaged"]:
            if facts["within_7_day_damage_report"]:
                return _refund(order, "The item arrived damaged and was reported within 7 days.", ["4.1"])
            if facts["within_60_day_damage_replacement"]:
                return Decision("decline", "A damage refund is only available within 7 days of delivery. "
                                "A free replacement is still available.", ["4.1", "4.2"])
            return Decision("decline", "It has been more than 60 days, so this goes through the manufacturer warranty.", ["5.1"])
        if order["final_sale"]:
            return Decision("decline", "This item is final sale and arrived undamaged, so it cannot be returned.", ["1.3"])
        if facts["within_30_day_return_window"]:
            return _refund(order, "The item is within the 30 day return window.", ["1.1"])
        return Decision("decline", "The item is past the 30 day return window and was not damaged.", ["1.1"])

    if intent == "replacement":
        if status != "delivered":
            return Decision("decline", "Only delivered orders can be replaced.")
        # Eligibility first: no point asking which model if the swap is not allowed at all.
        if route["damaged"]:
            if not facts["within_60_day_damage_replacement"]:
                return Decision("decline", "It has been more than 60 days, so this goes through the manufacturer warranty.", ["5.1"])
            if not facts["within_7_day_damage_report"] and not route["photo_provided"]:
                return Decision("need_info", "Damage claims after 7 days need a photo. Ask for one.", ["4.3"])
            clause, reason = "4.2", "The item arrived damaged and is within the 60 day replacement window."
        else:
            if order["final_sale"]:
                return Decision("decline", "This item is final sale and arrived undamaged, so it cannot be exchanged.", ["1.3"])
            if not facts["within_30_day_return_window"]:
                return Decision("decline", "The item is past the 30 day exchange window and was not damaged.", ["3.1"])
            clause, reason = "3.1", "The item is within the 30 day exchange window."
        # Then the product they asked for.
        if route.get("new_product") and product is None:
            return Decision("need_info", "The requested model could not be found. Ask which product they want.")
        if product is not None and product["category"] != order["category"]:
            return Decision("decline", "A swap must be for a product in the same category.", [clause])
        args = {"order_id": order["order_id"], "reason": f"{reason} (policy {clause})"}
        if product is not None:
            args["new_product_id"] = product["product_id"]
        return Decision("act", reason, [clause], "create_replacement", args)

    return Decision("escalate", "Unhandled request. Hand it to the planner.")
