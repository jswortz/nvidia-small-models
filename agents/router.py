"""Phase 3 router: Nano turns a customer message into a structured request.

The router has one job: understand what the customer wants. It never decides
whether the store allows it. That is the policy engine's job.
"""
from __future__ import annotations

import json
import os
import re

INTENTS = ["order_status", "policy_question", "cancel", "refund", "replacement", "other"]

# vLLM's guided decoding forces the output to match this schema.
SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "order_id": {"type": ["string", "null"]},
        "damaged": {"type": "boolean"},
        "photo_provided": {"type": "boolean"},
        "new_product": {"type": ["string", "null"]},
    },
    "required": ["intent", "order_id", "damaged", "photo_provided", "new_product"],
}

ROUTER_PROMPT = """/no_think Classify the customer support message. Reply with JSON only.

intent, pick one:
- order_status: asks where an order is or what its status is
- policy_question: asks how something works (return windows, refund timing, shipping costs, warranty) without asking you to act on a specific order. "How long until a refund shows up?" is a policy_question.
- cancel: wants to stop an order before it is delivered, or uses the word cancel
- refund: wants their money back, or wants to return or send back an item. "Return order 2033" is a refund, not a cancel.
- replacement: wants a replacement, an exchange, or a swap to a different model
- other: anything else

order_id: the order number as a string, or null if none is given
damaged: true if the item arrived damaged, broken, cracked, or defective, or is not working properly (stopped working, will not turn on, dead pixels, strange noises or smells). false if the customer says it works.
photo_provided: true only if the customer says they attached or sent a photo
new_product: the different product or model they want instead, in their words, or null

Examples (message -> intent, damaged):
- "I'd like to send back the kettle from order 2041, it's fine but we don't use it" -> refund, false
- "Cancel order 2077, I ordered it by mistake" -> cancel, false
- "The lamp from order 2090 won't turn on anymore" -> refund, true
- "My headphones from order 2012 crackle and cut out" -> replacement, true
- "When will my refund hit my account?" -> policy_question, false
- "Do you price match?" -> other, false"""

# Example order numbers above (2012 to 2090) are outside the store's 1000s range on purpose,
# so the prompt never hints at a real test order. tests/test_router.py checks this.

# Phase 6: a fine-tuned router learns the rules from its training examples, so it gets
# this short prompt instead (fewer prompt tokens per call). Training data uses the same text.
SHORT_PROMPT = ("/no_think Classify the customer support message as JSON with intent "
                "(order_status, policy_question, cancel, refund, replacement, other), "
                "order_id, damaged, photo_provided, new_product.")


def prompt_for(style: str | None = None) -> str:
    """The router prompt: 'short' for a fine-tuned router, otherwise the full prompt.

    Set ROUTER_PROMPT=short in the environment to use the short one everywhere."""
    style = style or os.getenv("ROUTER_PROMPT", "full")
    return SHORT_PROMPT if style == "short" else ROUTER_PROMPT

_ORDER_ID = re.compile(r"\b(\d{4})\b")
_JSON = re.compile(r"\{.*\}", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _text(value) -> str | None:
    """A non-empty string or None. A model can emit true or 0 here; only text is a product name."""
    return (value.strip() or None) if isinstance(value, str) else None


def parse(text: str) -> dict:
    """Turn model output into a clean route. Anything unparseable becomes intent 'other'."""
    match = _JSON.search(_THINK.sub("", text or ""))
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    intent = data.get("intent") if data.get("intent") in INTENTS else "other"
    order_id = data.get("order_id")
    return {
        "intent": intent,
        "order_id": str(order_id).strip("# ") if isinstance(order_id, (str, int)) and order_id else None,
        "damaged": data.get("damaged") is True,
        "photo_provided": data.get("photo_provided") is True,
        "new_product": _text(data.get("new_product")),
    }


def classify(message: str, client, model: str, prompt: str | None = None) -> tuple[dict, dict]:
    """Return (route, token usage) for one customer message. prompt: 'full' or 'short' (see prompt_for)."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt_for(prompt)},
                  {"role": "user", "content": message}],
        temperature=0,
        max_tokens=200,
        extra_body={"guided_json": SCHEMA},
    )
    usage = {"prompt": resp.usage.prompt_tokens, "completion": resp.usage.completion_tokens} if resp.usage \
        else {"prompt": 0, "completion": 0}
    return backstop(parse(resp.choices[0].message.content or ""), message), usage


def backstop(route: dict, message: str) -> dict:
    """Deterministic backstop: if the model missed an order number that is plainly there, take it."""
    if route["order_id"] is None and route["intent"] not in ("policy_question", "other"):
        found = _ORDER_ID.search(message)
        route["order_id"] = found.group(1) if found else None
    return route
