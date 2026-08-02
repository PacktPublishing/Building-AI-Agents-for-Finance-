"""
evaluator.py
=============
Lab extension ("Extension: Adding the evaluator-optimizer pattern"):
adds an EvaluatorAgent to the LlamaIndex pipeline, transforming the
architecture from pure sequential to a sequential + evaluator-optimizer
hybrid (see the chapter's "Choosing and combining architectural styles"
section).

The EvaluatorAgent reviews the SupervisorAgent's output and sends it
back for revision if quality criteria are not met. To wire it in:
point the SupervisorAgent's can_handoff_to at ["EvaluatorAgent"], add
this agent to the workflow's agent list, and persist the synthesized
report into shared state so the evaluator has real content to review.
"""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o", temperature=0)


async def evaluate_assessment(ctx: Context) -> str:
    """Review the health assessment for quality.

    Checks:
    1. Completeness — Are all profitability and liquidity ratios covered?
    2. Consistency — Does the overall score align with individual scores?
    3. Justification — Are scores backed by specific data points?

    Returns feedback or approval.
    """
    current_state = await ctx.store.get("state")

    profitability = current_state.get("threshold_profitability_comments")
    liquidity = current_state.get("threshold_liquidity_comments")
    overall = current_state.get("overall_comments")

    issues = []

    # Check completeness
    if profitability is None:
        issues.append("Missing profitability analysis.")
    if liquidity is None:
        issues.append("Missing liquidity analysis.")
    if overall is None or overall == "":
        issues.append("Missing overall assessment.")

    # Check that ratios were actually scored
    prof_ratios = current_state.get("profitability_ratios", {})
    liq_ratios = current_state.get("liquidity_ratios", {})

    if len(prof_ratios) < 4:
        issues.append(
            f"Only {len(prof_ratios)}/4 profitability metrics scored."
        )
    if len(liq_ratios) < 4:
        issues.append(
            f"Only {len(liq_ratios)}/4 liquidity metrics scored."
        )

    if issues:
        feedback = (
            "REVISION NEEDED. The following issues were found:\n"
            + "\n".join(f"- {issue}" for issue in issues)
            + "\nPlease address these issues and re-submit."
        )
        return feedback
    else:
        return "APPROVED. The assessment is complete and consistent."


# The EvaluatorAgent — add this to the workflow after SupervisorAgent
evaluator_agent = FunctionAgent(
    name="EvaluatorAgent",
    description=(
        "Review the health assessment for completeness, consistency, "
        "and justification quality."
    ),
    system_prompt=(
        "You are a quality reviewer. Call evaluate_assessment exactly "
        "once. It checks the health assessment for:\n"
        "1. Completeness: Are all ratios covered?\n"
        "2. Consistency: Does the overall score match individual scores?\n"
        "3. Justification: Are scores backed by data?\n"
        "If it returns REVISION NEEDED, hand off to the SupervisorAgent "
        "with the specific feedback -- but allow at most one revision "
        "round before approving.\n"
        "If it returns APPROVED, you are the last agent: write the final "
        "approved assessment yourself and do not hand off.\n"
    ),
    llm=llm,
    tools=[evaluate_assessment],
    can_handoff_to=["SupervisorAgent"],  # Can send back for revision
)
