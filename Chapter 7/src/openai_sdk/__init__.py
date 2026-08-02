"""
OpenAI Agents SDK implementation of the Multi-Agent Financial Analyst.

Architecture: Leader-Follower (Orchestrator-Worker) with dynamic routing.
Pattern: Section 2.2 (Orchestrator-Worker) from Chapter 7.

Agents:
  Manager Agent → Stock Price Agent
                → Company Basic Information Agent
                → Income Statement Agent
"""
