"""Generate the paper figures (vector PDF) from the recorded result JSONs.
CPU-only (matplotlib). Colorblind-safe Okabe-Ito palette (Okabe & Ito 2008, the
reference qualitative CVD-safe set). Clean paper style: thin marks, recessive
axes, selective direct labels, no chartjunk, single y-axis.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "results", "figs")
os.makedirs(FIGS, exist_ok=True)

# Okabe-Ito (CVD-safe)
BLUE, ORANGE, GREEN, VERM, SKY, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7"
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.7, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False,
})


def style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)


# ---------------------------------------------------------------- Fig 1: collapse
def fig_collapse():
    ops = ["first", "before", "last", "after"]
    tvr = [98.8, 98.4, 1.9, 2.0]
    earliest = [100, 100, 98, 94]           # picks-earliest (MASTER §1)
    # colour by operator DIRECTION (an entity property, not rank):
    # wants-earliest (accidentally correct) vs wants-latest (collapsed)
    col = [BLUE, BLUE, ORANGE, ORANGE]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.5))
    x = range(4)
    for ax, vals, ttl in [(a1, tvr, "(a) Temporal validity (TVR)"),
                          (a2, earliest, "(b) Picks the earliest candidate")]:
        bars = ax.bar(x, vals, width=0.62, color=col, zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 2, f"{v:.0f}" if v == int(v) else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(list(x)); ax.set_xticklabels(ops)
        ax.set_ylim(0, 112); ax.set_ylabel("%"); ax.set_title(ttl, fontsize=9)
        ax.grid(axis="x", visible=False); style(ax)
    leg = [Patch(fc=BLUE, label="wants earliest (first/before)"),
           Patch(fc=ORANGE, label="wants latest (last/after)")]
    a1.legend(handles=leg, loc="center left", fontsize=7.2, bbox_to_anchor=(0.0, 0.62))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "figure1_tvr_collapse.pdf"))
    plt.close(fig)
    print("wrote figure1_tvr_collapse.pdf")


# ---------------------------------------------------------------- Fig: fusion curve
def fig_fusion():
    d = json.load(open(os.path.join(ROOT, "results", "phase2_fusion_tvr.json")))
    alphas = d["alphas"]; per = d["per_op"]
    xs = list(range(len(alphas)))
    xt = [f"{a:g}" if a < 1e6 else r"$\infty$" for a in alphas]
    colmap = {"first": BLUE, "before": SKY, "last": ORANGE, "after": VERM}
    fig, ax = plt.subplots(figsize=(4.3, 2.9))
    for op in ("first", "before", "last", "after"):
        ax.plot(xs, per[op], marker="o", ms=3.2, lw=1.4, color=colmap[op], label=op, zorder=3)
    macro = [sum(per[o][i] for o in per) / len(per) for i in range(len(alphas))]
    ax.plot(xs, macro, marker="s", ms=4, lw=2.2, color=INK, label="macro", zorder=4)
    ax.set_xticks(xs); ax.set_xticklabels(xt, fontsize=7.5)
    ax.set_xlabel(r"fusion weight $\alpha$"); ax.set_ylabel("realized TVR (%)")
    ax.set_ylim(-4, 106); style(ax)
    ax.axvline(0, color=MUTED, lw=0.6, ls=":")
    ax.text(0.05, 8, "collapse\n(LLM only)", fontsize=7, color=MUTED, va="bottom")
    ax.legend(ncol=3, fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fusion_curve.pdf")); plt.close(fig)
    print("wrote fusion_curve.pdf")


# ---------------------------------------------------------------- Fig: injection locus
def fig_locus():
    d = json.load(open(os.path.join(ROOT, "results", "phase2_trielocal_endtoend.json")))
    alphas = sorted({float(a) for a in d["global"].keys()})
    def macro(loc, a):
        v = d[loc][str(a) if str(a) in d[loc] else (f"{a:g}")]
        vals = [100 * v[op]["hit"] / v[op]["n"] for op in v if v[op]["n"]]
        return sum(vals) / len(vals)
    # keys may be '0'/'0.2'... normalize
    def get(loc, a):
        for k in d[loc]:
            if abs(float(k) - a) < 1e-9:
                vv = d[loc][k]
                vals = [100 * vv[op]["hit"] / vv[op]["n"] for op in vv if vv[op]["n"]]
                return sum(vals) / len(vals)
        return None
    xs = list(range(len(alphas)))
    g = [get("global", a) for a in alphas]; t = [get("trielocal", a) for a in alphas]
    fig, ax = plt.subplots(figsize=(4.3, 2.9))
    ax.plot(xs, t, marker="o", ms=4, lw=2.0, color=GREEN, label="trie-local (ours)", zorder=4)
    ax.plot(xs, g, marker="s", ms=4, lw=2.0, color=VERM, label="global", zorder=3)
    for xi, v in [(xs[-1], t[-1]), (xs[-1], g[-1])]:
        ax.text(xi + 0.04, v, f"{v:.0f}", fontsize=7.5, va="center",
                color=GREEN if v == t[-1] else VERM)
    ax.set_xticks(xs); ax.set_xticklabels([f"{a:g}" for a in alphas], fontsize=8)
    ax.set_xlabel(r"fusion weight $\alpha$"); ax.set_ylabel("answer accuracy Hits@1 (%)")
    ax.set_ylim(0, 106); style(ax)
    ax.legend(fontsize=8, loc="center right")
    ax.annotate("same discriminator,\nsame LLM scores", xy=(1.0, 55), fontsize=7, color=MUTED)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "injection_locus.pdf")); plt.close(fig)
    print("wrote injection_locus.pdf")


def fig_graphical_abstract():
    """Problem -> Fix summary: collapse bars, arrow, recovered bars."""
    ops = ["first", "before", "last", "after"]
    before = [98.8, 98.4, 1.9, 2.0]        # LLM alone (collapse)
    after = [100, 100, 100, 100]           # + trie-local discriminator
    col = [BLUE, BLUE, ORANGE, ORANGE]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.3))
    x = range(4)
    a1.bar(x, before, width=0.62, color=col, zorder=3)
    a1.set_title("KG-constrained LLM decoder", fontsize=9)
    a1.text(2.5, 12, "temporal\ncollapse", color=ORANGE, fontsize=8.5, ha="center", fontweight="bold")
    a2.bar(x, after, width=0.62, color=col, zorder=3)
    a2.set_title("+ trie-local temporal discriminator", fontsize=9)
    a2.text(2.5, 60, "restored", color=GREEN, fontsize=8.5, ha="center", fontweight="bold")
    for ax in (a1, a2):
        ax.set_xticks(list(x)); ax.set_xticklabels(ops, fontsize=8)
        ax.set_ylim(0, 112); ax.set_ylabel("TVR (%)"); ax.grid(axis="x", visible=False); style(ax)
    # arrow between panels
    fig.text(0.505, 0.5, r"$\Rightarrow$", fontsize=22, color=INK, ha="center", va="center")
    leg = [Patch(fc=BLUE, label="wants earliest"), Patch(fc=ORANGE, label="wants latest")]
    a1.legend(handles=leg, loc="upper right", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 1)); fig.subplots_adjust(wspace=0.35)
    fig.savefig(os.path.join(FIGS, "graphical_abstract.pdf")); plt.close(fig)
    print("wrote graphical_abstract.pdf")


if __name__ == "__main__":
    fig_collapse(); fig_fusion(); fig_locus(); fig_graphical_abstract()
    print("all figures ->", FIGS)
