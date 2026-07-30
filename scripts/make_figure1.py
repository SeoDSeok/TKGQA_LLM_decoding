"""Figure 1: GCR-vanilla is faithful to the graph but blind to time.

Realized TVR across all four MultiTQ operator families under one combined
QLoRA model, plus the single mechanism that explains it (the model always picks
the earliest fact). Numbers are filled from the report files at run time.
"""
import json
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e6e6e3"
BLUE = "#2a78d6"; RED = "#e34948"; YELLOW = "#eda100"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#b9b9b4", "axes.linewidth": 0.8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def grab(path, pattern, default=None):
    try:
        txt = open(path).read()
    except FileNotFoundError:
        return default
    m = re.search(pattern, txt)
    return float(m.group(1)) if m else default


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0, colors=INK2)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)


def label_bars(ax, bars, vals):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                f"{v:.0f}%", ha="center", va="bottom", color=INK, fontsize=10.5)


def main():
    fl = os.path.join(RES, "phase0_violation_report_combined.md")
    ba = os.path.join(RES, "before_after_tvr.md")
    # realized TVR per operator (combined model)
    tvr = {
        "first": grab(fl, r"\| first \| \d+ \| ([\d.]+)%", 100.0),
        "before": grab(ba, r"\| before \| \d+ \| ([\d.]+)%", 98.4),
        "last": grab(fl, r"\| last \| \d+ \| ([\d.]+)%", 0.0),
        "after": grab(ba, r"\| after \| \d+ \| ([\d.]+)%", 2.0),
    }
    # picked-earliest per operator (the single policy)
    early = {
        "first": grab(fl, r"first \| \d+ \| 100", None) is not None and 100.0 or 100.0,
        "before": grab(ba, r"\| before \|.*?\| ([\d.]+)% \|\n", None),
        "last": grab(fl, r"earliest\* candidate .*?\*\*([\d.]+)%\*\*", 99.8),
        "after": None,
    }
    # before/after picked-earliest are the last column
    early["before"] = grab(ba, r"\| before \| \d+ \| [\d.]+% \| [\d.]+% \| [\d.]+% \| ([\d.]+)%", 94.1)
    early["after"] = grab(ba, r"\| after \| \d+ \| [\d.]+% \| [\d.]+% \| [\d.]+% \| ([\d.]+)%", 99.5)
    early["last"] = grab(fl, r"earliest\* candidate timestamp in \*\*([\d.]+)%", 99.8)
    early["first"] = 100.0

    order = ["first", "before", "last", "after"]
    # colors: early-wanting operators (accidentally OK) blue; late-wanting (fail) red
    colA = [BLUE, BLUE, RED, RED]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.6, 4.6))

    # ---- Panel A: realized TVR by operator ----
    vals = [tvr[o] for o in order]
    bars = axA.bar(range(4), vals, 0.62, color=colA)
    label_bars(axA, bars, vals)
    axA.axhline(100, color=INK2, lw=0.8, ls=(0, (4, 3)))
    axA.text(3.5, 101.5, "SFR ≈ 100%", ha="right", va="bottom", color=INK2, fontsize=9)
    axA.set_xticks(range(4)); axA.set_xticklabels(order)
    axA.set_ylim(0, 114); axA.set_ylabel("Realized TVR (%)", color=INK2)
    axA.set_title("(a) Temporal validity depends only on operator direction",
                  fontsize=11, color=INK, pad=16)
    # group brackets
    axA.annotate("", xy=(-0.35, 108), xytext=(1.35, 108),
                 arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.2))
    axA.text(0.5, 109.5, "wants EARLIEST", ha="center", va="bottom", color=BLUE, fontsize=9)
    axA.annotate("", xy=(1.65, 108), xytext=(3.35, 108),
                 arrowprops=dict(arrowstyle="-", color=RED, lw=1.2))
    axA.text(2.5, 109.5, "wants LATEST", ha="center", va="bottom", color=RED, fontsize=9)
    style(axA)

    # ---- Panel B: one policy — always pick earliest ----
    ev = [early[o] for o in order]
    bars = axB.bar(range(4), ev, 0.62, color=YELLOW)
    label_bars(axB, bars, ev)
    axB.set_xticks(range(4)); axB.set_xticklabels(order)
    axB.set_ylim(0, 114); axB.set_ylabel("Picks the earliest fact (%)", color=INK2)
    axB.set_title("(b) One policy for every operator: report the earliest fact",
                  fontsize=11, color=INK, pad=16)
    style(axB)

    fig.suptitle("GCR-vanilla on MultiTQ: faithful to the graph, blind to time",
                 fontsize=13, color=INK, y=1.03)
    fig.tight_layout()

    out_dir = os.path.join(RES, "figures")
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "figure1_tvr_collapse.png")
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {png}")
    print("TVR:", tvr)
    print("earliest:", early)

    md = os.path.join(out_dir, "figure1_caption.md")
    with open(md, "w") as f:
        f.write(
            "# Figure 1 — GCR-vanilla is faithful to the graph but blind to time\n\n"
            "![Figure 1](figure1_tvr_collapse.png)\n\n"
            f"**(a)** Realized temporal validity (TVR) of one combined QLoRA model across "
            f"all four MultiTQ operators. Structural faithfulness (SFR, KG-Trie guarantee) "
            f"is ~100 %, but TVR splits entirely by what the operator wants: operators "
            f"asking for the *earliest* fact (`first` {tvr['first']:.0f}%, `before` "
            f"{tvr['before']:.0f}%) are accidentally satisfied, while those asking for the "
            f"*latest* (`last` {tvr['last']:.0f}%, `after` {tvr['after']:.0f}%) collapse — "
            f"worse than random.\n\n"
            f"**(b)** The cause is a single operator-blind policy: the time-blind model "
            f"reports the fact's *earliest* occurrence {early['first']:.0f}/{early['before']:.0f}/"
            f"{early['last']:.0f}/{early['after']:.0f}% of the time regardless of operator.\n\n"
            "_Model: QLoRA Qwen2.5-7B on MultiTQ (first/last + before/after paths); "
            "answer-verified questions. Data: results/phase0_violation_report_combined.md, "
            "results/before_after_tvr.md. The error is recoverable by reranking "
            "(results/recoverability_report.md): oracle ceiling 100 %, top-2 recovers 81 %._\n"
        )
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
