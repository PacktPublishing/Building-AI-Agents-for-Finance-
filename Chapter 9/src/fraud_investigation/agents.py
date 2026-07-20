"""
agents.py
=========
Fraud investigation agents using the adversarial debate pattern.

The Fraud Advocate argues FOR fraud.
The Legitimacy Advocate argues AGAINST fraud.
The Investigation Decision Agent weighs both arguments.
"""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o", temperature=0)


fraud_advocate = FunctionAgent(
    name="FraudAdvocate",
    description="Argue the case for fraud.",
    system_prompt=(
        "You are an insurance fraud investigator. Review "
        "all evidence and build the strongest possible case "
        "that this claim is fraudulent. Cite specific "
        "evidence for each point. Also identify the "
        "weaknesses in your own argument — where is your "
        "case thin? Be thorough but honest."
    ),
    llm=llm,
    tools=[],
    can_handoff_to=["InvestigationDecision"],
)

legitimacy_advocate = FunctionAgent(
    name="LegitimacyAdvocate",
    description="Argue the claim is legitimate.",
    system_prompt=(
        "You are a consumer protection advocate. Review "
        "all evidence and build the strongest possible case "
        "that this claim is legitimate. For every red flag, "
        "provide an innocent alternative explanation. Also "
        "identify the weaknesses in your own argument. "
        "Your role is to ensure no legitimate claim is "
        "wrongly denied."
    ),
    llm=llm,
    tools=[],
    can_handoff_to=["InvestigationDecision"],
)

investigation_decision = FunctionAgent(
    name="InvestigationDecision",
    description="Weigh fraud and legitimacy arguments.",
    system_prompt=(
        "You are the senior fraud investigator. You have "
        "received arguments from both a Fraud Advocate and "
        "a Legitimacy Advocate. Evaluate:\n"
        "1. Evidence quality (specific vs. circumstantial)\n"
        "2. Reasoning strength (logical vs. speculative)\n"
        "3. Acknowledged weaknesses on each side\n\n"
        "Produce a determination:\n"
        "- strong_fraud_indicators: Strong evidence of deliberate fraud\n"
        "- likely_fraud: Preponderance of evidence suggests fraud\n"
        "- inconclusive: Insufficient evidence either way\n"
        "- likely_legitimate: Preponderance of evidence supports legitimacy\n\n"
        "Include a confidence score (0.0-1.0) and a "
        "recommended action."
    ),
    llm=llm,
    tools=[],
    can_handoff_to=[],
)
