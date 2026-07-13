"""Figure 3.2 - Agent workflow sequence diagram for the framework bake-off.

Renders a sequence diagram showing how the manager agent coordinates the
data, compute, and analyst agents on the P/E comparison task.

Usage: python fig_3_2_bakeoff_sequence.py  ->  svg/fig_3_2_bakeoff_sequence.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ACTORS = ["User", "Manager Agent", "Data Agent", "Compute Agent", "Analyst Agent"]
X = {a: i * 2.2 for i, a in enumerate(ACTORS)}

# (from, to, label, dashed) or ("note", actor, text) or ("banner", None, text)
STEPS = [
    ("User", "Manager Agent", '"Compare AAPL and JPM P/E ratios"', False),
    ("note", "Manager Agent", "Planning phase\n(LangGraph, AutoGen, CrewAI)"),
    ("Manager Agent", "Data Agent", "Get AAPL data", False),
    ("Data Agent", "Manager Agent", "Price=\\$195.3, EPS=\\$6.67", True),
    ("Manager Agent", "Data Agent", "Get JPM data", False),
    ("Data Agent", "Manager Agent", "Price=\\$148.7, EPS=\\$12.61", True),
    ("Manager Agent", "Compute Agent", "Calculate P/E ratios", False),
    ("Compute Agent", "Manager Agent", "AAPL: 29.28, JPM: 11.79", True),
    ("Manager Agent", "Analyst Agent", "Generate investment memo", False),
    ("note", "Analyst Agent", "Synthesis phase\n(OpenAI Agents SDK, Claude SDK)"),
    ("Analyst Agent", "Manager Agent", "Comparative memo", True),
    ("Manager Agent", "User", "Final report with P/E analysis", True),
]
BANNER = ("Frameworks differ in:  control flow (sequential vs. parallel)  |  "
          "agent handoffs (explicit vs. implicit)\nvalidation (PydanticAI adds "
          "type safety)  |  tracing (OpenAI Agents SDK and Google ADK add audit logs)")

ROW_H = 0.72
TOP = len(STEPS) * ROW_H + 1.6

fig, ax = plt.subplots(figsize=(10.4, 6.6))
ax.set_xlim(-1.1, X[ACTORS[-1]] + 1.1)
ax.set_ylim(-1.7, TOP + 0.9)
ax.axis("off")

for a in ACTORS:  # lifelines + actor boxes (top and bottom)
    ax.plot([X[a], X[a]], [-0.6, TOP - 0.4], color="#b0b8c4", lw=1.1, ls=":", zorder=1)
    for y in (TOP - 0.4, -0.6):
        ax.add_patch(FancyBboxPatch((X[a] - 0.85, y - 0.22), 1.7, 0.5,
                     boxstyle="round,pad=0.06", fc="#dbeafe", ec="#34699c", lw=1.2, zorder=3))
        ax.text(X[a], y + 0.03, a, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="#1e3a5f", zorder=4)

y = TOP - 1.15
for step in STEPS:
    if step[0] == "note":
        _, actor, text = step
        cx = min(X[actor], X[ACTORS[-1]] - 1.6)
        ax.add_patch(FancyBboxPatch((cx - 1.3, y - 0.3), 2.9, 0.58,
                     boxstyle="round,pad=0.05", fc="#fff8dc", ec="#c4841c", lw=1.1, zorder=5))
        ax.text(cx + 0.15, y - 0.01, text, ha="center", va="center",
                fontsize=8, color="#7a5a12", style="italic", zorder=6)
    else:
        src, dst, label, dashed = step
        x1, x2 = X[src], X[dst]
        ax.annotate("", xy=(x2, y), xytext=(x1, y), zorder=5,
                    arrowprops=dict(arrowstyle="-|>", lw=1.4,
                                    color="#7b68ee" if dashed else "#34699c",
                                    linestyle=(0, (5, 3)) if dashed else "solid"))
        ax.text((x1 + x2) / 2, y + 0.13, label, ha="center", va="bottom",
                fontsize=8.4, color="#333", zorder=6)
    y -= ROW_H

ax.add_patch(FancyBboxPatch((-0.6, -1.62), X[ACTORS[-1]] + 1.2, 0.72,
             boxstyle="round,pad=0.06", fc="#fff8dc", ec="#c4841c", lw=1.2, zorder=5))
ax.text((X[ACTORS[-1]]) / 2, -1.26, BANNER, ha="center", va="center",
        fontsize=8.4, color="#7a5a12", zorder=6)

out = Path(__file__).parent / "svg"
out.mkdir(exist_ok=True)
for ext, dpi in (("png", 300), ("svg", 300)):
    fig.savefig(out / f"fig_3_2_bakeoff_sequence.{ext}", dpi=dpi, bbox_inches="tight",
                facecolor="white")
print("rendered", out / "fig_3_2_bakeoff_sequence.png")
