"""Unit tests for the router. No model needed: a fake client stands in for Nano.

Run from /workspace:
    python tests/test_router.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import router  # noqa: E402

passed = 0


def ok(label: str) -> None:
    global passed
    passed += 1
    print(f"ok    {label}")


class FakeClient:
    """Returns a fixed model reply and records the request, shaped like the OpenAI client."""

    def __init__(self, reply: dict):
        self.reply = json.dumps(reply)
        self.request = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.request = kwargs
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def reply(intent, order_id=None, damaged=False):
    return {"intent": intent, "order_id": order_id, "damaged": damaged,
            "photo_provided": False, "new_product": None}


# The prompt's examples must not reuse order numbers or messages from the eval sets.
case_files = sorted((ROOT / "evals").glob("cases_*.jsonl"))
cases = [json.loads(line) for f in case_files for line in f.open() if line.strip()]
eval_ids = {m for c in cases for m in re.findall(r"\b\d{4}\b", c["message"])}
prompt_ids = set(re.findall(r"\b\d{4}\b", router.ROUTER_PROMPT))
leaked = prompt_ids & eval_ids
assert not leaked, f"router prompt mentions eval order numbers: {sorted(leaked)}"
ok(f"prompt order numbers {sorted(prompt_ids)} do not appear in {len(cases)} eval cases")

leaked_msgs = [c["id"] for c in cases if c["message"] in router.ROUTER_PROMPT]
assert not leaked_msgs, f"router prompt contains eval messages: {leaked_msgs}"
ok("prompt contains no eval message verbatim")

# Every intent the prompt defines is one the schema allows, and the reverse.
defined = set(re.findall(r"^- (\w+):", router.ROUTER_PROMPT, re.MULTILINE))
assert defined == set(router.INTENTS), defined ^ set(router.INTENTS)
ok("prompt defines exactly the schema's intents")

# Every example label is a valid intent and a valid damaged value.
examples = re.findall(r'-> (\w+), (true|false)$', router.ROUTER_PROMPT, re.MULTILINE)
assert examples and all(i in router.INTENTS for i, _ in examples), examples
ok(f"all {len(examples)} prompt examples use valid labels")

# classify sends the prompt and schema, at temperature 0.
client = FakeClient(reply("refund", "1042"))
route, usage = router.classify("Refund order 1042", client, "test-model")
assert client.request["messages"][0]["content"] == router.ROUTER_PROMPT
assert client.request["extra_body"] == {"guided_json": router.SCHEMA}
assert client.request["temperature"] == 0
assert route["intent"] == "refund" and usage == {"prompt": 10, "completion": 5}
ok("classify sends the prompt, schema, and temperature 0")

# Backstop: an order number the model missed is taken from the message for order intents...
route, _ = router.classify("Please return order 1032", FakeClient(reply("refund")), "m")
assert route["order_id"] == "1032", route
ok("backstop fills a missed order number for an order intent")

# ...but not for policy questions, which should not trigger an order lookup.
route, _ = router.classify("Is 2025 the year returns changed?", FakeClient(reply("policy_question")), "m")
assert route["order_id"] is None, route
ok("backstop leaves policy questions without an order number")

print(f"\nAll {passed} checks passed.")
