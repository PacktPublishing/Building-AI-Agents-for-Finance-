"""
AutoGen implementation of the Multi-Agent Financial Analyst.

Architecture: Conversational multi-agent with code execution.
Pattern: Section 6.4 (conversational two-agent pair; GroupChat is the scaling path) and conversation
programming (Section 4.4) from Chapter 7.

Agents:
  AssistantAgent (LLM-powered) + UserProxyAgent (code executor)
"""
