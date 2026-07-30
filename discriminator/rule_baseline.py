"""Deterministic rule-based temporal comparator over the value-space features.

Purpose (reviewer request): show that once time is in value space, the operator is a
trivial hand-written comparison — so the paper's contribution is the value-space
*representation*, not the learned network. This scorer uses NO trained parameters: it
applies the textbook rule per operator to the candidate's fractional year (and the
question anchor for before/after), then reports the same per-operator top-1 validity
(TVR proxy) metric as eval_offline, on the same answer-verified test candidate sets.

Rules (argmax of a score built only from t and the operator):
  first  -> pick earliest   : s = -t
  last   -> pick latest      : s = +t
  before -> a candidate strictly before the anchor (closest from below on top)
  after  -> a candidate strictly after  the anchor (closest from above on top)
  composites reuse the interval primitives (bound filter + superlative direction).

Run: PYTHONPATH=. python discriminator/rule_baseline.py --out results/rule_baseline.md
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np

from discriminator.dataset import build_examples

NEG = -1e18
# interval primitives: (dir, uses_lower, uses_upper); dir -1 = min/earliest, +1 = max/latest
OP_RULE = {
    "first": (-1, 0, 0), "last": (1, 0, 0),
    "before": (0, 0, 1), "after": (0, 1, 0),
    "after_first": (-1, 1, 0), "before_last": (1, 0, 1),
    "equal": (0, 1, 1), "equal_multi": (0, 1, 1),
}


def score(cands, op, anchor):
    dr, ul, uu = OP_RULE.get(op, (0, 0, 0))
    t = np.array([c["frac"] for c in cands], dtype=float)
    s = np.zeros_like(t)
    # bound filter: candidates violating the anchor bound are pushed below any satisfier
    if (ul or uu) and anchor is not None:
        ok = np.ones_like(t, dtype=bool)
        if ul:                      # 'after': must be >= anchor
            ok &= t >= anchor
        if uu:                      # 'before': must be <= anchor
            ok &= t <= anchor
        s = np.where(ok, 0.0, NEG)
        # among satisfiers prefer the one closest to the anchor (textbook 'the ... before/after')
        prox = -np.abs(t - anchor)
        s = s + prox
    # superlative direction dominates (scaled so it outranks the proximity tiebreak)
    if dr != 0:
        s = s + dr * t * 1e6
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_ops", nargs="+", default=["first", "last", "before", "after"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ex = build_examples(args.split, tuple(args.eval_ops))
    per = defaultdict(lambda: {"n": 0, "top1": 0})
    for e in ex:
        s = score(e["cands"], e["op"], e.get("anchor_frac"))
        pred = int(np.argmax(s))
        per[e["op"]]["n"] += 1
        per[e["op"]]["top1"] += int(e["cands"][pred]["valid"] > 0)

    L = ["### Rule-based comparator (no trained parameters)", "",
         "| operator | rule top-1 (TVR) | n |", "|---|--:|--:|"]
    for op in args.eval_ops:
        if op in per:
            d = per[op]
            L.append(f"| {op} | {100*d['top1']/d['n']:.1f}% | {d['n']} |")
    macro = np.mean([100 * per[o]["top1"] / per[o]["n"] for o in per])
    L.append(f"\n**macro top-1 = {macro:.1f}%**")
    md = "\n".join(L)
    print(md)
    if args.out:
        open(args.out, "w").write(md + "\n")


if __name__ == "__main__":
    main()
