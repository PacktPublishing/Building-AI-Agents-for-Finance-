"""
LlamaIndex AgentWorkflow implementation of the Multi-Agent Financial
Health Analyzer.

Architecture: Sequential pipeline with explicit handoffs and shared state.
Pattern: Section 2.1 (Sequential) from Chapter 7.

Agents:
  FundamentalAgent → ProfitabilityAgent → LiquidityAgent → SupervisorAgent
"""
