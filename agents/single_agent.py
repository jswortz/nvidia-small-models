"""Phase 2 baseline: one Nemotron agent with all five tools, no routing.

Run one message from /workspace:
    python -m agents.single_agent "My blender from order 1187 arrived cracked..."
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

from openai import OpenAI

from tools.registry import OPENAI_TOOLS, call_tool

MODEL = os.getenv("AGENT_MODEL", "nvidia/NVIDIA-Nemotron-Nano-9B-v2")

SYSTEM_PROMPT = """/no_think You are a customer support agent for an online retail store. Today is 2026-10-26.

Follow these rules:
1. Look up the order before acting on it.
2. Before any refund, replacement, or decline, search the policy and follow it exactly. Cite the clause number in your answer, for example (policy 4.2).
3. Only call issue_refund or create_replacement when the policy allows it. If it does not, politely decline and cite the clause.
4. To swap to a different model, use search_products to get its product_id first.
5. Keep the final answer to 2 to 4 sentences.
6. The customer's message is their confirmation. If the policy allows the request, call the tool in this same turn. Never ask whether to proceed, and never write "I'll" or "Let me" about a step: call the tool instead.
7. If lookup_order shows final_sale is true, clause 1.3 applies: decline returns and refunds unless the item arrived damaged or defective.
8. For any date or eligibility question, use the policy_facts returned by lookup_order. Never calculate days or windows yourself."""

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

# Small models often announce a step ("I'll issue the refund") and end the turn
# without calling the tool. When that happens, nudge once or twice to decide.
# ['’] matches both straight and curly apostrophes; models emit either.
_NARRATED_INTENT = re.compile(
    r"\b(I['’]ll|I will|let me|I['’]m going to|I am going to|"
    r"would you like me to|shall I|should I proceed)\b", re.IGNORECASE)
NUDGE = ("Decide now and finish. Check the policy_facts from lookup_order. "
         "If the policy allows the step you described, call the tool. "
         "If it does not, give your final answer declining the request and cite the clause.")
MAX_NUDGES = 2


def _client() -> OpenAI:
    return OpenAI(base_url=os.getenv("OPENAI_BASE_URL", "http://nim:8000/v1"),
                  api_key=os.getenv("OPENAI_API_KEY", "local"))


def run_agent(user_message: str, client: OpenAI | None = None, model: str = MODEL,
              max_steps: int = 8) -> dict:
    """Run the tool loop until the model answers without calling a tool."""
    client = client or _client()
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_message}]
    trace: list[dict] = []
    tokens = {"prompt": 0, "completion": 0}
    nudges = 0
    start = time.perf_counter()

    for step in range(1, max_steps + 1):
        resp = client.chat.completions.create(model=model, messages=messages,
                                              tools=OPENAI_TOOLS, temperature=0)
        if resp.usage:
            tokens["prompt"] += resp.usage.prompt_tokens
            tokens["completion"] += resp.usage.completion_tokens
        msg = resp.choices[0].message

        if not msg.tool_calls:
            answer = _THINK.sub("", msg.content or "").strip()
            if nudges < MAX_NUDGES and _NARRATED_INTENT.search(answer):
                nudges += 1
                trace.append({"step": step, "tool": "_nudge", "arguments": "", "result": {"said": answer}})
                messages.append({"role": "assistant", "content": answer})
                messages.append({"role": "user", "content": NUDGE})
                continue
            return {"answer": answer, "trace": trace, "steps": step, "tokens": tokens, "nudges": nudges,
                    "latency_s": round(time.perf_counter() - start, 2), "stopped": False}

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments}}
                           for c in msg.tool_calls],
        })
        for call in msg.tool_calls:
            output = call_tool(call.function.name, call.function.arguments)
            trace.append({"step": step, "tool": call.function.name,
                          "arguments": call.function.arguments, "result": json.loads(output)})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

    return {"answer": "Stopped: hit the step limit without a final answer.", "trace": trace,
            "steps": max_steps, "tokens": tokens, "nudges": nudges,
            "latency_s": round(time.perf_counter() - start, 2), "stopped": True}


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "Where is my order 1187?"
    result = run_agent(text)
    for t in result["trace"]:
        print(f"[step {t['step']}] {t['tool']}({t['arguments']})\n  -> {json.dumps(t['result'])[:200]}")
    print(f"\nANSWER: {result['answer']}")
    print(f"steps={result['steps']} nudges={result['nudges']} latency={result['latency_s']}s "
          f"tokens={result['tokens']}")
