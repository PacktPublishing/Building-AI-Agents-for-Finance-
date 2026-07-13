# Framework decision-matrix template (Chapter 3)

A ready-to-adapt version of the scoring rubric and decision matrix from
Chapter 3, *AI Agent Frameworks in Finance*.

- `decision_matrix.csv` — the nine decision dimensions (D1-D9), example
  weights, and the chapter's illustrative 0-5 scores for each framework.
  Replace the scores with your own pilot results; adjust the weights to
  your institution's priorities (they must sum to 1.0).
- `score_frameworks.py` — computes weighted scores and prints a ranking:

```
python score_frameworks.py
```

Tip from the chapter: if you are in a highly regulated unit (trading,
compliance), up-weight D3 and D7. If you are a quant R&D shop, up-weight
D1, D4, D5, and D6.
