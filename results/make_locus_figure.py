"""Injection-locus figure from the single factorial run (results/locus_factorial.json).

Three arms, one run: A = per-group scoring + per-group normalization (deployed),
B = per-group scoring + global normalization, C = mixed-set scoring + global
normalization. Answer accuracy vs fusion weight, macro over first/last.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open(os.path.join(ROOT, "results", "locus_factorial.json")))
alphas = D["alphas"]; M = D["M"]; OPS = ("first", "last")

def macro(arm, a, key="hit"):
    v = [100 * M[arm][str(a)][o][key] / M[arm][str(a)][o]["n"] for o in OPS]
    return sum(v) / len(v)

BLUE, ORANGE, RED, INK, GRID = "#0072B2", "#E69F00", "#D55E00", "#222222", "#DDDDDD"
ARMS = [("A_group_group", "A  per-group scoring + per-group norm", BLUE, "o", "-"),
        ("B_group_global", "B  per-group scoring + global norm",  ORANGE, "s", "--"),
        ("C_mixed_global", "C  mixed-set scoring + global norm",  RED, "^", ":")]

fig, ax = plt.subplots(figsize=(3.5, 2.7))
x = range(len(alphas))
for arm, lab, c, mk, ls in ARMS:
    ax.plot(x, [macro(arm, a) for a in alphas], marker=mk, linestyle=ls, color=c,
            lw=2.0, ms=5, label=lab, clip_on=False, zorder=3)

ax.set_xticks(list(x)); ax.set_xticklabels([("0" if a == 0 else f"{a:g}") for a in alphas], fontsize=8)
ax.set_xlabel(r"fusion weight $\alpha$", fontsize=9, color=INK)
ax.set_ylabel("answer accuracy (%)", fontsize=9, color=INK)
ax.set_ylim(0, 105); ax.set_yticks([0, 25, 50, 75, 100])
ax.tick_params(labelsize=8, colors=INK, length=3)
ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(GRID)

i02 = alphas.index(0.2)
ax.axvline(i02, color="#BBBBBB", lw=1.0, ls=(0,(4,3)), zorder=1)

ax.legend(fontsize=6.6, frameon=False, loc="center left", bbox_to_anchor=(0.20, 0.40),
          handlelength=2.0, labelspacing=0.35)
fig.tight_layout()
for ext in ("pdf", "png"):
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", f"injection_locus_v2.{ext}")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
