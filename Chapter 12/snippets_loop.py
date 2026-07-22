"""snippets_loop.py: Runnable illustration for the loop section's guard-and-resume prose.

Checkpointing a batch run so that a crash on name k resumes from name k, not name 1;
the mechanism the chapter describes in prose but did not otherwise show as code.

Offline and deterministic: the "agent" is a scripted stand-in so the loop mechanics are
the whole show. 

Run it:  .venv/Scripts/python.exe snippets_loop.py
"""

from __future__ import annotations

import json
import os

# --- Checkpointing a batch run --------------------------------------------------------------
CHECKPOINT_PATH = os.path.join("results", "checkpoint_demo.json")


def run_batch_with_checkpoint(tickers: list[str], analyse_one, checkpoint_path: str) -> dict:
    """Run a batch, persisting which names are done after EACH name."""
    done: dict = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as fh:
            done = json.load(fh)

    for ticker in tickers:
        if ticker in done:
            continue                                   # already paid for: do not re-spend
        done[ticker] = analyse_one(ticker)             # may raise: that is the crash
        with open(checkpoint_path, "w", encoding="utf-8") as fh:
            json.dump(done, fh)                        # persist at the safe boundary
    return done


# --- Demo ------------------------------------------------------------------------------------
def main() -> None:
    print("checkpointed batch: crash on the 3rd name, resume without re-running 1-2")
    os.makedirs("results", exist_ok=True)
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
    tickers = ["MSFT", "AAPL", "GOOGL", "JPM", "KO"]
    crash_once = {"armed": True}

    def analyse_one(ticker: str) -> str:
        if ticker == "GOOGL" and crash_once["armed"]:
            crash_once["armed"] = False
            raise RuntimeError("simulated crash mid-batch")
        print(f"    analysing {ticker} (paid: one full agent run)")
        return "FAIRLY_VALUED"

    try:
        run_batch_with_checkpoint(tickers, analyse_one, CHECKPOINT_PATH)
    except RuntimeError as exc:
        print(f"    !! batch died on GOOGL: {exc}")

    print("    --- restart ---")
    with open(CHECKPOINT_PATH, encoding="utf-8") as fh:
        already = json.load(fh)
    print(f"    resuming: {len(already)} names already done -> {sorted(already)}")
    done = run_batch_with_checkpoint(tickers, analyse_one, CHECKPOINT_PATH)
    print(f"    batch complete: {len(done)}/{len(tickers)} names; "
          f"MSFT and AAPL were not re-run (their cost was not re-spent)")
    # os.remove(CHECKPOINT_PATH)                         # tidy up the demo artefact


if __name__ == "__main__":
    main()
