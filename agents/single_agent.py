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
5. Keep the final answer to 2 to 4 sentences."""

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


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
            return {"answer": answer, "trace": trace, "steps": step, "tokens": tokens,
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
            "steps": max_steps, "tokens": tokens,
            "latency_s": round(time.perf_counter() - start, 2), "stopped": True}


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "Where is my order 1187?"
    result = run_agent(text)
    for t in result["trace"]:
        print(f"[step {t['step']}] {t['tool']}({t['arguments']})\n  -> {json.dumps(t['result'])[:200]}")
    print(f"\nANSWER: {result['answer']}")
    print(f"steps={result['steps']} latency={result['latency_s']}s tokens={result['tokens']}")
