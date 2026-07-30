"""Architecture schematic for the value-space temporal set scorer (fig:arch).
CPU-only (matplotlib). Okabe-Ito CVD-safe palette, matching make_paper_figures.py.
Horizontal staged block diagram (two-column-spanning figure* in the paper).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "results", "figs")
os.makedirs(FIGS, exist_ok=True)

BLUE, ORANGE, GREEN, VERM, SKY, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7"
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "text.color": INK})


def box(ax, x, y, w, h, text, fc="white", ec=MUTED, tc=INK, fs=8.5, lw=1.0, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
                       fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=4, fontweight="bold" if bold else "normal")
    return (x, y, w, h)


def arrow(ax, p0, p1, color=MUTED, lw=1.2, style="-|>"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=10,
                        color=color, lw=lw, zorder=2, shrinkA=1, shrinkB=1)
    ax.add_patch(a)


def right(b):  # right-mid of a box
    x, y, w, h = b; return (x + w, y + h / 2)


def left(b):
    x, y, w, h = b; return (x, y + h / 2)


def top(b):
    x, y, w, h = b; return (x + w / 2, y + h)


def bot(b):
    x, y, w, h = b; return (x + w / 2, y)


fig, ax = plt.subplots(figsize=(7.1, 2.55))
ax.set_xlim(0, 10.4); ax.set_ylim(0, 3.7); ax.axis("off")

# --- inputs (left) ---
q = box(ax, 0.05, 2.55, 1.35, 0.6, "question $q$", fc="#F2F7FB", ec=BLUE)
cand = box(ax, 0.05, 1.35, 1.35, 0.7, "candidate set\n$C=\\{t_i\\}$, anchor $a$", fc="#FBF4E9", ec=ORANGE)

# --- LLM polarity (grounds operator) ---
llm = box(ax, 1.95, 2.5, 1.75, 0.7,
          "LLM zero-shot\npolarity 99.8%\n(wants-later, thr)", fc="#F2F7FB", ec=BLUE, fs=8)
# --- sign folding ---
sign = box(ax, 4.15, 2.5, 1.7, 0.7,
           "sign folding\n$st,\\;sd$", fc="#EAF6F1", ec=GREEN, fs=8.5)
# --- Bochner Phi ---
phi = box(ax, 4.15, 1.35, 1.7, 0.72,
          "Bochner $\\Phi(t)$\nrel $\\oplus$ abs", fc="#EAF6F1", ec=GREEN, fs=8.5)
# --- MiniLM ---
mini = box(ax, 1.95, 1.35, 1.75, 0.72,
           "frozen MiniLM\ntext emb (384)", fc="#F6F6F6", ec=MUTED, fs=8)

# --- set encoder ---
enc = box(ax, 6.25, 1.55, 1.95, 1.4,
          "set encoder ($\\times2$)\n\ncand$\\leftrightarrow$cand\nself-attn\n+\ncand$\\rightarrow q$\ncross-attn", fc="#F7EFF4", ec=PURPLE, fs=8)
# --- score ---
score = box(ax, 8.55, 2.0, 1.75, 0.62, "per-cand.\nscore $s_\\theta$", fc="white", ec=INK, bold=True, fs=8.5)
# --- fusion ---
fuse = box(ax, 8.55, 0.7, 1.75, 0.78,
           "trie-local fusion\n$\\log P_{LLM}$\n$+\\,\\alpha\\log\\mathrm{softmax}\\,s_\\theta$",
           fc="#FBECE4", ec=VERM, fs=7.6)

# arrows
arrow(ax, right(q), left(llm), color=BLUE)
arrow(ax, right(llm), left(sign), color=BLUE)
arrow(ax, right(cand), (4.15, 1.71), color=ORANGE)          # cand -> Phi
arrow(ax, right(cand), (1.95, 1.71), color=ORANGE)          # cand -> minilm (features)
arrow(ax, bot(sign), top(phi), color=GREEN)                 # sign -> Phi (signed features)
arrow(ax, right(sign), (6.25, 2.7), color=GREEN)            # sign -> enc
arrow(ax, right(phi), (6.25, 2.05), color=GREEN)            # Phi -> enc
arrow(ax, right(mini), (6.25, 1.75), color=MUTED)           # minilm -> enc
arrow(ax, top(enc), (8.55, 2.3), color=PURPLE)              # enc -> score
arrow(ax, bot(score), top(fuse), color=INK)                 # score -> fusion
arrow(ax, right(q), (7.2, 2.95), color=BLUE, lw=0.9, style="-|>")  # q -> enc (cross-attn cond)

ax.text(5.2, 3.5, "value space (the representation the LLM lacks)",
        ha="center", fontsize=8, color=GREEN, style="italic")
ax.text(9.42, 0.28, "picked $t$", ha="center", fontsize=8, color=VERM)
arrow(ax, bot(fuse), (9.42, 0.42), color=VERM, lw=1.0)

out = os.path.join(FIGS, "architecture.pdf")
fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=150)
print("wrote", out)
