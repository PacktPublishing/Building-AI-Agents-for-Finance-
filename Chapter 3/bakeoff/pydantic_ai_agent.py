"""Bake-off: PydanticAI (schema-first validation)."""

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from common import (
    OPENAI_MODEL, check_api_key, compute_pe_tool_func,
    get_stock_data_tool_func, input_message, persist_framework_output,
    system_message,
)


class InvestmentMemo(BaseModel):
    """The validated output schema every run must satisfy."""

    aapl_pe: float = Field(description="Computed P/E ratio for AAPL")
    jpm_pe: float = Field(description="Computed P/E ratio for JPM")
    memo: str = Field(description="Short comparative investment memo")


agent = Agent(
    f"openai:{OPENAI_MODEL}",
    system_prompt=system_message,
    output_type=InvestmentMemo,
)


@agent.tool
def get_stock_data(ctx: RunContext, ticker: str) -> dict:
    """Get stock data for a given ticker symbol."""
    d = get_stock_data_tool_func(ticker)
    return {"ticker": d.ticker, "price": d.price, "eps": d.eps}


@agent.tool
def compute_pe_ratio(ctx: RunContext, price: float, eps: float) -> float:
    """Compute P/E ratio given stock price and earnings per share."""
    return compute_pe_tool_func(price, eps)


def main() -> None:
    check_api_key()
    result = agent.run_sync(input_message)
    memo = result.output
    u = result.usage() if callable(result.usage) else result.usage
    usage = (f"input={u.input_tokens} output={u.output_tokens} "
             f"total={u.total_tokens}")
    print(memo.model_dump_json(indent=2))
    print(f"\n[token usage] {usage}")
    persist_framework_output(
        "pydantic_ai", memo.model_dump_json(indent=2), usage)


if __name__ == "__main__":
    main()
