"""Bake-off: Claude Agent SDK (safe-by-design agentic workflows).

The Claude Agent SDK (renamed from the Claude Code SDK in September
2025) drives Claude models with sessions, subagents, hooks, and native
MCP support. Custom tools are exposed to the agent as an in-process MCP
server. Requires ANTHROPIC_API_KEY with available credits.
"""

import asyncio
import os

from claude_agent_sdk import (
    ClaudeAgentOptions, create_sdk_mcp_server, query, tool,
)

from common import (
    compute_pe_tool_func, get_stock_data_tool_func, input_message,
    persist_framework_output, system_message,
)


@tool("get_stock_data", "Get the latest price and EPS for a ticker",
      {"ticker": str})
async def get_stock_data(args):
    d = get_stock_data_tool_func(args["ticker"])
    return {"content": [
        {"type": "text", "text": f"{d.ticker}: price={d.price} eps={d.eps}"}]}


@tool("compute_pe_ratio", "Compute P/E ratio from price and EPS",
      {"price": float, "eps": float})
async def compute_pe_ratio(args):
    pe = compute_pe_tool_func(args["price"], args["eps"])
    return {"content": [{"type": "text", "text": str(pe)}]}


async def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set; skipping live run.")
    finance_tools = create_sdk_mcp_server(
        name="finance", tools=[get_stock_data, compute_pe_ratio])
    options = ClaudeAgentOptions(
        system_prompt=system_message,
        mcp_servers={"finance": finance_tools},
        allowed_tools=[
            "mcp__finance__get_stock_data",
            "mcp__finance__compute_pe_ratio",
        ],
        max_turns=8,
    )
    final = []
    async for message in query(prompt=input_message, options=options):
        text = getattr(message, "result", None)
        if text:
            final.append(text)
    output = "\n".join(final)
    print(output)
    persist_framework_output("claude_agent_sdk", output)


if __name__ == "__main__":
    asyncio.run(main())
