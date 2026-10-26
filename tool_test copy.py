"""Phase 0 check: one full tool-call round trip against the local Nemotron server."""
import json
import os

from openai import OpenAI

client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL", "http://nim:8000/v1"), api_key="local")
MODEL = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"

tools = [{
    "type": "function",
    "function": {
        "name": "lookup_order",
        "description": "Look up an order's status by order id",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
}]


def lookup_order(order_id):
    return {"order_id": order_id, "status": "shipped", "eta": "2026-10-28"}


messages = [
    {"role": "system", "content": "/no_think You are a retail support agent. Use tools when needed."},
    {"role": "user", "content": "Where is my order 1187?"},
]

# Turn 1: the model should ask for a tool call
resp = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
msg = resp.choices[0].message
print("TOOL CALLS:", msg.tool_calls)
assert msg.tool_calls, "No structured tool call. Check the parser flags in docker logs."

# Run the tool and send the result back
messages.append(msg)
for call in msg.tool_calls:
    args = json.loads(call.function.arguments)
    result = lookup_order(**args)
    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})

# Turn 2: the model should answer using the tool result
final = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
print("ANSWER:", final.choices[0].message.content)
