"""
Claims Processing Pipeline — Multi-Agent Sequential Architecture.

Architecture: Sequential pipeline with guardrails (Ch 7, Section 2.1).
Pattern: FNOL → Parser → Coverage → Fraud → Assessment → Decision → Compliance

Each agent reads from and writes to a shared claim context.
The Compliance Agent has veto power over all decisions.
"""
