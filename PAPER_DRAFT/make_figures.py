"""Generate the paper's four figures from the repository's own generated data.

No number here is typed by hand except where a table in `RESULTS.md` or a numbered Finding is the
only source, and each of those is cited in `SOURCES` below so a reviewer can check it.

Run with **system Python**, not the pinned competition venv: that venv exists to serialise
submission artifacts at the competition's library versions, and matplotlib is not one of them.

    python PAPER_DRAFT/make_figures.py        # -> PAPER_DRAFT/figures/*.svg and *.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"

SOURCES = {
    "fig1": "runs/e1b-stagea-deepseek/features/stagea-deepseek/layer_curve.csv (generated)",
    "fig2": "experiments.md, the leaderboard series; PROGRESS.md Findings 49, 50, 60, 61",
    "fig3": "train/eval_batch_composition.py, RESULTS.md 'E40: batch composition'",
    "fig4": "RESULTS.md 'E28: familiarity rank candidates' (offline) + the leaderboard series",
}

# Validated with the skill's validator: 3-slot all-pairs light mode PASSes every gate; the aqua
# slot WARNs on contrast (2.74:1), so every series carries a visible direct label.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
SURFACE, GRID = "#fcfcfb", "#e6e5e1"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2.0, "figure.dpi": 140,
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.svg / .png")


def fig1_layer_curve() -> None:
    """Every layer beats the honest nested result. That gap is the selection artifact."""
    path = ROOT / "runs/e1b-stagea-deepseek/features/stagea-deepseek/layer_curve.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    series = {}
    for r in rows:
        series.setdefault(r["pooling"], []).append((int(r["layer"]), float(r["balanced_accuracy"])))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for pooling, colour in (("mean", BLUE), ("last", ORANGE), ("max", AQUA)):
        pts = sorted(series[pooling])
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, color=colour, zorder=3)
        # Direct labels: the relief rule for the sub-3:1 aqua slot, and better than a legend box.
        ax.annotate(f"{pooling} pooling", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    color=colour, fontsize=8.5, va="center", fontweight="bold")

    ax.axhline(0.6732, color=INK2, lw=1.4, ls=(0, (5, 3)), zorder=2)
    ax.annotate("text-only floor  0.673", (0.6, 0.6732), xytext=(0, 5), textcoords="offset points",
                color=INK2, fontsize=8.5)
    ax.axhline(0.5515, color=INK, lw=1.4, ls=(0, (2, 2)), zorder=2)
    ax.annotate("honest nested-selected probe  0.552", (0.6, 0.5515), xytext=(0, -13),
                textcoords="offset points", color=INK, fontsize=8.5, fontweight="bold")
    ax.axhline(0.5, color=MUTED, lw=1.0, zorder=1)
    ax.annotate("chance", (0.6, 0.5), xytext=(0, -12), textcoords="offset points", color=MUTED, fontsize=8)

    ax.set_xlim(-0.5, 42)
    ax.set_ylim(0.44, 0.72)
    ax.set_xlabel("hidden-state layer")
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Picking a layer after seeing the folds inflates the score", pad=12, fontsize=11,
                 fontweight="bold", loc="left")
    fig.text(0.0, -0.06, "DeepSeek-R1-0528-Qwen3-8B, 138 instances, 37 layers. Per-layer values are "
             "optimistic by construction:\nthey are read off the same data that would select the layer. "
             "Nested selection scores 0.552.", fontsize=8, color=INK2, ha="left")
    save(fig, "fig1_layer_curve")


def fig2_leaderboard_ladder() -> None:
    """Twenty uploads, each answering one question. The ladder is 1/14 per rung."""
    rows = [
        ("e27k  ranking + derived counts", 12, True),
        ("e27m/n/o/p  each count neighbour", 11, False),
        ("e27e  ranking, median split", 10, False),
        ("e27h  DeepSeek's split alone", 10, False),
        ("e28a  suffix NLL", 10, False),
        ("e27f  best constants", 9, False),
        ("e27g/i/j  one other model's split", 9, False),
        ("E11 v2  fused text + states", 9, False),
        ("e27a/b/c  fitted on official labels", 8, False),
        ("e29a  Min-K gap, DeepSeek only", 8, False),
        ("all-False", 7, False),
        ("e27d  ranking, inverted", 4, False),
        ("e28d  Min-K gap, all models", 4, False),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ys = range(len(rows))[::-1]
    for y, (label, score, is_best) in zip(ys, rows):
        colour = ORANGE if is_best else BLUE
        ax.barh(y, score, height=0.62, color=colour, zorder=3,
                path_effects=None, edgecolor=SURFACE, linewidth=1.0)
        ax.annotate(f"{score}/14", (score, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK, fontweight="bold" if is_best else "normal")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.axvline(7, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.annotate("all-False = 7", (7, len(rows) - 0.3), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=MUTED)
    ax.axvline(9, color=INK2, lw=1.2, ls=(0, (2, 2)), zorder=2)
    ax.annotate("constants ceiling = 9", (9, -0.8), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=INK2)
    ax.set_xlim(0, 14.6)
    ax.set_xlabel("cases correct, of 14")
    ax.grid(axis="y", visible=False)
    ax.set_title("The leaderboard series: each upload answers one question", pad=12, fontsize=11,
                 fontweight="bold", loc="left")
    fig.text(0.0, -0.05, "e27d at 4/14 is the hinge: far below chance means the ranking carries "
             "information and points the wrong way.\nShipping its exact opposite scored 10/14.",
             fontsize=8, color=INK2, ha="left")
    save(fig, "fig2_leaderboard_ladder")


def fig3_null_distribution() -> None:
    """Given the derived counts, how surprising is 12/14 if the ranking is noise?"""
    dist = {4: 0.044, 6: 0.247, 8: 0.408, 10: 0.245, 12: 0.052, 14: 0.003}
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for score, p in dist.items():
        observed = score == 12
        ax.bar(score, p, width=1.4, color=ORANGE if observed else BLUE, zorder=3,
               edgecolor=SURFACE, linewidth=1.0)
        ax.annotate(f"{p:.3f}", (score, p), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK, fontweight="bold" if observed else "normal")
    ax.annotate("observed\n12/14", (12.6, 0.060), xytext=(26, 28), textcoords="offset points",
                ha="left", fontsize=9, color=ORANGE, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.5,
                                connectionstyle="angle,angleA=0,angleB=90,rad=3"))
    # The null mean falls inside the tallest bar. Cut the bar with a surface-coloured spacer so
    # the rule reads as a gap, and park the label in the empty upper right with a leader.
    ax.plot([8.05, 8.05], [0, 0.408], color=SURFACE, lw=3.0, zorder=5)  # stops at the bar top,
    ax.plot([8.05, 8.05], [0, 0.408], color=INK2, lw=1.3, ls=(0, (4, 3)), zorder=6)  # not the axis
    ax.annotate("null mean\n8.05", xy=(8.05, 0.335), xytext=(13.9, 0.355), textcoords="data",
                ha="center", fontsize=8.5, color=INK2, zorder=7,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1.0, shrinkA=2, shrinkB=4))
    ax.set_xticks(list(dist))
    ax.set_xlim(2.4, 15.6)
    ax.set_ylim(0, 0.47)
    ax.set_xlabel("cases correct, of 14")
    ax.set_ylabel("probability")
    ax.grid(axis="x", visible=False)
    ax.set_title("With the counts right, a ranking that knows nothing still scores 8", pad=12,
                 fontsize=11, fontweight="bold", loc="left")
    fig.text(0.0, -0.08, "Exact distribution under a random ranking with e27k's counts, averaged over "
             "all 56 batch compositions\nconsistent with the measured balances. "
             "P(score >= 12) = 0.055. Totals are spaced two apart: parity is fixed by the counts.",
             fontsize=8, color=INK2, ha="left")
    save(fig, "fig3_null_distribution")


def fig4_inversion() -> None:
    """The offline winner scored 4/14; the offline loser scored 12/14."""
    pts = [
        # label, offline problem-level AUC, leaderboard cases, note
        ("e27e\nmean NLL", 0.312, 10, "left"),  # left, or it collides with e28a at the same y
        ("e27k\nmean NLL + counts", 0.312, 12, "right"),
        ("e28a\nsuffix NLL", 0.388, 10, "right"),
        ("e29a\nMin-K gap, DeepSeek only", 0.867, 8, "left"),
        ("e28d\nMin-K gap", 0.867, 4, "left"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label, auc, score, side in pts:
        best = score == 12
        ax.scatter(auc, score, s=132, color=ORANGE if best else BLUE, zorder=4,
                   edgecolor=SURFACE, linewidth=1.8)
        dx = 12 if side == "right" else -12
        ax.annotate(label, (auc, score), xytext=(dx, 0), textcoords="offset points",
                    ha="left" if side == "right" else "right", va="center", fontsize=8.5,
                    color=INK, fontweight="bold" if best else "normal")
    ax.axhline(7, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.annotate("chance (all-False)", (0.17, 7), xytext=(0, 5), textcoords="offset points",
                fontsize=8, color=MUTED)
    ax.axvline(0.5, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.annotate("offline chance", (0.5, 13.2), xytext=(5, 0), textcoords="offset points",
                fontsize=8, color=MUTED)
    ax.set_xlim(0.14, 0.98)
    ax.set_ylim(2.5, 13.6)
    ax.set_xlabel("offline problem-level AUC  (E28 harness, template-grouped)")
    ax.set_ylabel("leaderboard cases correct, of 14")
    ax.set_title("On this task, offline validation ran against the leaderboard", pad=12,
                 fontsize=11, fontweight="bold", loc="left")
    fig.text(0.0, -0.07, "Min-K gap was the best offline candidate of twelve — AUC 0.867, "
             "template-bootstrap CI [0.728, 0.973],\nmaxT-adjusted p = 0.0010 — and scored 4/14. "
             "Mean NLL was the worst and scored 10, then 12 with derived counts.\n"
             "e27e and e27k share an offline coordinate: same statistic, different counts.",
             fontsize=8, color=INK2, ha="left")
    save(fig, "fig4_inversion")


if __name__ == "__main__":
    print("writing figures to", OUT)
    fig1_layer_curve()
    fig2_leaderboard_ladder()
    fig3_null_distribution()
    fig4_inversion()
    print("\nsources:")
    for k, v in SOURCES.items():
        print(f"  {k}: {v}")
