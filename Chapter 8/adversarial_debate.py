"""
Adversarial Debate layer — implemented as a LangGraph state graph that runs
ON TOP of the Investment Committee from `investment_committee.py`.

The committee's thesis becomes the seed for a structured debate:
  - A BULL agent argues to enter the position.
  - A BEAR agent argues against.
  - A Devil's Advocate identifies the weakest claim on each side.
  - A neutral Judge issues a final verdict that may differ from the committee.

Why heterogeneous models matter:
  We deliberately use TWO different model families (Anthropic + OpenAI) for
  the bull and bear roles, because two instances of the same model produce
  correlated errors and the debate collapses into agreement.

Run:
    export ANTHROPIC_API_KEY=...
    export OPENAI_API_KEY=...
    python adversarial_debate.py
"""
from __future__ import annotations

from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from investment_committee import run_committee


# ---------------------------------------------------------------------------
# Heterogeneous model line-up. The bull and bear are intentionally from
# different families so that they bring different priors to the debate — that
# family diversity, not a temperature setting, is what drives the disagreement.
# We pass no `temperature`: Opus 4.7 (the bull) rejects it with a 400, and
# Claude models released after Sonnet 4.6 drop the parameter, so omitting it
# keeps the debate runnable as models are upgraded.
# ---------------------------------------------------------------------------
bull_model = ChatAnthropic(model="claude-opus-4-7", max_tokens=1500)
bear_model = ChatOpenAI(model="gpt-4o", max_tokens=1500)
devil_model = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1000)
judge_model = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1500)


class DebateState(TypedDict):
    ticker: str
    committee_thesis: str
    committee_decision: str
    bull_case: str
    bear_case: str
    devil_critique: str
    judge_verdict: str
    final_decision: str


def bull_node(state: DebateState) -> dict:
    sys = SystemMessage(content=(
        "You are a BULL analyst. Construct the strongest possible case to ENTER a long "
        "position. Cite the strongest evidence from the committee thesis: catalysts, "
        "valuation support, technical confirmation, sentiment tailwinds. Be concrete and "
        "do NOT hedge. Max 200 words. End with: BULL_CONVICTION=<0-100>."
    ))
    msg = HumanMessage(content=(
        f"Ticker: {state['ticker']}\n"
        f"Committee thesis:\n{state['committee_thesis']}\n"
        f"Committee decision: {state['committee_decision']}"
    ))
    out = bull_model.invoke([sys, msg])
    return {"bull_case": out.content}


def bear_node(state: DebateState) -> dict:
    sys = SystemMessage(content=(
        "You are a BEAR analyst. Construct the strongest possible case AGAINST entering "
        "a long position. Surface risks the committee may have under-weighted: tail "
        "risks, macro headwinds, valuation stretch, sentiment crowding, technical "
        "exhaustion. Be concrete and do NOT hedge. Max 200 words. End with: "
        "BEAR_CONVICTION=<0-100>."
    ))
    msg = HumanMessage(content=(
        f"Ticker: {state['ticker']}\n"
        f"Committee thesis:\n{state['committee_thesis']}\n"
        f"Committee decision: {state['committee_decision']}"
    ))
    out = bear_model.invoke([sys, msg])
    return {"bear_case": out.content}


def devil_advocate(state: DebateState) -> dict:
    sys = SystemMessage(content=(
        "You are a Devil's Advocate. Identify the SINGLE weakest claim in the BULL case "
        "and the SINGLE weakest claim in the BEAR case, and explain in one sentence each "
        "why those claims are weak. Max 150 words."
    ))
    msg = HumanMessage(content=(
        f"BULL case:\n{state['bull_case']}\n\nBEAR case:\n{state['bear_case']}"
    ))
    out = devil_model.invoke([sys, msg])
    return {"devil_critique": out.content}


def _parse_verdict(text: str) -> str:
    upper = text.upper().replace(" ", "")
    for action in ("BUY", "SELL", "HOLD"):
        if f"VERDICT={action}" in upper:
            return action
    return "HOLD"


def judge_node(state: DebateState) -> dict:
    sys = SystemMessage(content=(
        "You are a NEUTRAL Judge. Weigh the bull case, the bear case, and the devil's "
        "critique. Issue a verdict that may differ from the committee's prior decision. "
        "Reply STRICTLY in the form:\n"
        "VERDICT=<BUY|HOLD|SELL>\nCONFIDENCE=<0-100>\nRATIONALE=<2-3 sentences>"
    ))
    msg = HumanMessage(content=(
        f"Committee prior: {state['committee_decision']}\n\n"
        f"BULL case:\n{state['bull_case']}\n\n"
        f"BEAR case:\n{state['bear_case']}\n\n"
        f"Devil's critique:\n{state['devil_critique']}"
    ))
    out = judge_model.invoke([sys, msg])
    return {"judge_verdict": out.content, "final_decision": _parse_verdict(out.content)}


def build_debate():
    g = StateGraph(DebateState)
    g.add_node("bull", bull_node)
    g.add_node("bear", bear_node)
    g.add_node("devil", devil_advocate)
    g.add_node("judge", judge_node)

    # Bull and bear in parallel; both feed the devil; devil feeds the judge.
    g.add_edge(START, "bull")
    g.add_edge(START, "bear")
    g.add_edge("bull", "devil")
    g.add_edge("bear", "devil")
    g.add_edge("devil", "judge")
    g.add_edge("judge", END)
    return g.compile()


def run_pipeline(ticker: str) -> dict:
    """Run the full pipeline: committee first, then debate on the committee output."""
    committee_result = run_committee(ticker)
    initial: DebateState = {
        "ticker": ticker,
        "committee_thesis": committee_result["pm_thesis"],
        "committee_decision": committee_result["final_decision"],
        "bull_case": "",
        "bear_case": "",
        "devil_critique": "",
        "judge_verdict": "",
        "final_decision": "",
    }
    debate_result = build_debate().invoke(initial)
    return {"committee": committee_result, "debate": debate_result}


if __name__ == "__main__":
    out = run_pipeline("TSLA")
    print("=" * 80)
    print(f"COMMITTEE said       : {out['committee']['final_decision']}")
    print(f"DEBATE  judge says   : {out['debate']['final_decision']}")
    print("=" * 80)
    print("\n--- Bull case ---\n",        out["debate"]["bull_case"])
    print("\n--- Bear case ---\n",        out["debate"]["bear_case"])
    print("\n--- Devil's critique ---\n", out["debate"]["devil_critique"])
    print("\n--- Judge verdict ---\n",    out["debate"]["judge_verdict"])
