# Multi-Agent Trading System

Source code and chapter prose for a chapter of the book *Building AI Agents in Finance*. The chapter develops two multi-agent architectures with [LangGraph](https://langchain-ai.github.io/langgraph/) that compose into a single pipeline:

1. **Hierarchical Investment Committee** — `investment_committee.py` (chapter : Building a Multi-Agent Investment Committee with LangGraph). Four specialist analysts run in parallel, a Portfolio Manager synthesises a thesis, and a Risk Officer applies hard rules + an LLM check.
2. **Adversarial Debate** built on top of the committee's output — `adversarial_debate.py` (chapter : Stress-Testing the Committee with an Adversarial Debate Layer). Bull / Bear / Devil's Advocate / Judge stress-test the committee's verdict, with the Bull and Bear running on **different model families** on purpose.

## Requirements

- Python **3.10 or later**
- An **Anthropic** API key (mandatory)
- An **OpenAI** API key (only needed to run the debate stage)

## Setting up a virtual environment

A virtual environment keeps this project's dependencies isolated from the rest of your system and makes the install reproducible.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked by execution policy, run this once for your user account:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### macOS / Linux (bash or zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Leaving the venv

When you're done, run `deactivate` to return to your system Python.

## Configuring API keys

Copy the template and fill in your real keys:

```powershell
# Windows
copy .env.example .env
```

```bash
# macOS / Linux
cp .env.example .env
```

Open `.env` in your editor and replace each placeholder with the corresponding key. There are two ways to make the keys visible to the scripts:

**Option A — set them in your shell (no code changes).**

Windows (PowerShell):

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:OPENAI_API_KEY    = "sk-..."
```

macOS / Linux:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

**Option B — load `.env` automatically with `python-dotenv`.**

`python-dotenv` is already in `requirements.txt`. Add these two lines at the very top of either script (before the other imports) and the `.env` file is read on every run:

```python
from dotenv import load_dotenv
load_dotenv()
```

## Running

With the venv activated and keys configured:

```bash
python investment_committee.py     # Stage 1 only
python adversarial_debate.py       # Full pipeline (committee → debate)
python backtest.py                 # Walk-forward backtest of Stage 1
```

The committee and debate scripts evaluate `TSLA` by default. Edit the `__main__` block of either script to change the ticker. 
`backtest.py` runs a weekly walk-forward over a small universe and prints Sharpe, max drawdown, and hit rate; see the *Backtesting the committee* section of the chapter for what the harness is and is not — in particular, it is a framework backtest, not a research-grade one (not aligning historical prices for multi-agent runs).

## Project layout

```
.
├── investment_committee.py      # Stage 1 — LangGraph: 4 specialists → PM → Risk
├── adversarial_debate.py        # Stage 2 — LangGraph: Bull / Bear / Devil / Judge
├── backtest.py                  # Walk-forward backtest harness for Stage 1 (Simple backtest - not research grade - no historical data replay for multi-agent)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
