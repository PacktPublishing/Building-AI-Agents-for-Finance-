"""Bake-off: Mistral (function calling via la Plateforme).

Requires MISTRAL_API_KEY. Mistral's Agents API (May 2025) adds hosted
agents with built-in connectors (code execution, web search, document
library) and handoffs; this implementation uses plain function calling,
which is the portable subset every Mistral model supports.
"""

import json
import os

from mistralai.client import Mistral

from common import (
    compute_pe_tool_func, get_stock_data_tool_func, input_message,
    persist_framework_output, system_message,
)

MODEL = os.getenv("BAKEOFF_MISTRAL_MODEL", "mistral-medium-latest")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_data",
            "description": "Get the latest price and EPS for a ticker",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_pe_ratio",
            "description": "Compute P/E ratio from price and EPS",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number"},
                    "eps": {"type": "number"},
                },
                "required": ["price", "eps"],
            },
        },
    },
]


def _dispatch(name: str, args: dict) -> str:
    if name == "get_stock_data":
        d = get_stock_data_tool_func(args["ticker"])
        return f"{d.ticker}: price={d.price} eps={d.eps}"
    if name == "compute_pe_ratio":
        return str(compute_pe_tool_func(args["price"], args["eps"]))
    raise ValueError(name)


def main() -> None:
    if not os.getenv("MISTRAL_API_KEY"):
        raise SystemExit("MISTRAL_API_KEY is not set; skipping live run.")
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": input_message},
    ]
    in_tok = out_tok = 0
    while True:
        resp = client.chat.complete(
            model=MODEL, messages=messages, tools=TOOLS)
        usage = resp.usage
        in_tok += usage.prompt_tokens
        out_tok += usage.completion_tokens
        msg = resp.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            break
        for call in msg.tool_calls:
            result = _dispatch(
                call.function.name, json.loads(call.function.arguments))
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.function.name,
                "content": result,
            })
    final = msg.content
    usage_s = f"input={in_tok} output={out_tok} total={in_tok + out_tok}"
    print(final)
    print(f"\n[token usage] {usage_s}")
    persist_framework_output("mistral", str(final), usage_s)


if __name__ == "__main__":
    main()
