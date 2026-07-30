"""D1: SFT data audit — is the earliest-collapse a label-prior artifact (T1)?

Measures, from the signatures the combined model was trained on:
  1. operator counts;
  2. per-operator gold-position (earliest / middle / latest) in the candidate set
     — the key metric: if gold is earliest across most data, "always pick
     earliest" minimizes loss without learning the operator;
  3. timestamp-token exposure histogram (long-tail);
  4. operator x gold-position cross-table + mutual information — if operator
     strongly predicts gold-position (high MI) yet the model still collapsed,
     the *signal was present* → weakens T1 (data prior), points to T2
     (representation can't compare timestamps).

Output: results/d1_data_audit.md
"""
import json
import math
import os
import pickle
from collections import Counter, defaultdict

from checker.timepoint import TimePoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIG = os.path.join(ROOT, "data", "multitq", "signatures")
IDX = os.path.join(ROOT, "data", "multitq", "index", "kg_index.pkl")


def gold_position(gold_ts, cand_ts):
    """earliest / latest / middle / single, by chronological rank of gold."""
    cand = sorted({TimePoint.parse(t) for t in cand_ts})
    g = TimePoint.parse(gold_ts)
    if len(cand) < 2:
        return "single"
    if g == cand[0]:
        return "earliest"
    if g == cand[-1]:
        return "latest"
    return "middle"


def load_examples():
    """Return list of {op, gold_ts, candidates(list of str)} for train signatures."""
    idx = pickle.load(open(IDX, "rb"))
    sr2ots = idx["sr2ots"]
    exs = []
    # first/last: candidates stored directly
    fl = os.path.join(SIG, "first_last_time_train.jsonl")
    for l in open(fl):
        s = json.loads(l)
        exs.append({"op": s["op"], "gold_ts": s["answer"],
                    "candidates": list(s["candidate_timestamps"])})
    # before/after: candidates = (s,r) group timestamps from KG
    ba = os.path.join(SIG, "before_after_train.jsonl")
    for l in open(ba):
        s = json.loads(l)
        facts = sr2ots.get((s["s"], s["r"]), [])
        cands = [t for t, _ in facts]
        exs.append({"op": s["op"], "gold_ts": s["gold_ts"], "candidates": cands})
    return exs


def mutual_information(pairs):
    """MI between operator and gold-position over a list of (op, pos)."""
    n = len(pairs)
    joint = Counter(pairs)
    po = Counter(o for o, _ in pairs)
    pp = Counter(p for _, p in pairs)
    mi = 0.0
    for (o, p), c in joint.items():
        pxy = c / n
        mi += pxy * math.log2(pxy / ((po[o] / n) * (pp[p] / n)))
    # normalize by min entropy for interpretability
    ho = -sum((c / n) * math.log2(c / n) for c in po.values())
    hp = -sum((c / n) * math.log2(c / n) for c in pp.values())
    nmi = mi / min(ho, hp) if min(ho, hp) > 0 else 0.0
    return mi, nmi


def main():
    exs = load_examples()
    ops = ["first", "before", "last", "after"]

    # 1) operator counts
    op_counts = Counter(e["op"] for e in exs)

    # 2) per-operator gold position (only examples with >=2 candidates)
    pos_by_op = defaultdict(Counter)
    pairs = []  # (op, pos) for MI, multi-candidate only
    for e in exs:
        pos = gold_position(e["gold_ts"], e["candidates"])
        pos_by_op[e["op"]][pos] += 1
        if pos != "single":
            pairs.append((e["op"], pos))

    # 3) timestamp exposure histogram (gold ts frequency)
    ts_exposure = Counter(e["gold_ts"] for e in exs)
    exp_hist = Counter(ts_exposure.values())  # #times-seen -> #distinct-timestamps

    # 4) MI
    mi, nmi = mutual_information(pairs)

    # overall earliest-gold fraction among multi-candidate examples
    multi = [e for e in exs if gold_position(e["gold_ts"], e["candidates"]) != "single"]
    earliest = sum(1 for e in multi if gold_position(e["gold_ts"], e["candidates"]) == "earliest")
    earliest_frac = earliest / len(multi) if multi else 0.0

    # ---- write report ----
    L = ["# D1 — SFT data audit (is the collapse a label prior? T1)", "",
         f"Audited {len(exs)} train signatures "
         f"({len(multi)} with ≥2 candidate timestamps).", "",
         "## 1. Operator counts", "", "| operator | n |", "|---|--:|"]
    for o in ops:
        L.append(f"| {o} | {op_counts.get(o,0)} |")

    L += ["", "## 2. Gold-position within the candidate set (per operator)", "",
          "| operator | earliest | middle | latest | (single) |", "|---|--:|--:|--:|--:|"]
    for o in ops:
        c = pos_by_op[o]
        tot = sum(c.values()) or 1
        L.append(f"| {o} | {100*c['earliest']/tot:.0f}% | {100*c['middle']/tot:.0f}% | "
                 f"{100*c['latest']/tot:.0f}% | {c['single']} |")
    L += ["",
          f"- **Overall earliest-gold fraction (multi-candidate examples): "
          f"{100*earliest_frac:.1f}%**",
          f"- Decision threshold (design doc): ≥60% ⇒ T1 (data prior) plausible.",
          ""]

    L += ["## 3. Operator ↔ gold-position dependence", "",
          f"- Mutual information: **{mi:.3f} bits**  (normalized {nmi:.3f})",
          "- High MI ⇒ the operator *does* determine whether gold is earliest/latest,",
          "  i.e. the training signal to learn the operator **is present**. If the model",
          "  still collapsed, that points away from T1 (prior) toward T2 (representation).",
          ""]

    L += ["## 4. Timestamp-token exposure (gold frequency)", "",
          f"- distinct timestamps used as gold: {len(ts_exposure)} / 4017 registered",
          f"- mean exposures per used timestamp: {sum(ts_exposure.values())/max(len(ts_exposure),1):.2f}",
          "", "| #times seen as gold | #distinct timestamps |", "|--:|--:|"]
    for k in sorted(exp_hist)[:8]:
        L.append(f"| {k} | {exp_hist[k]} |")
    tail = sum(v for k, v in exp_hist.items() if k > 8)
    if tail:
        L.append(f"| 9+ | {tail} |")
    L += ["",
          "Long-tail / low exposure supports T2: 4017 special tokens seen ~twice each",
          "cannot acquire ordinal (chronological) structure from SFT alone.", ""]

    # interpretation
    L += ["## Interpretation → D2", ""]
    if earliest_frac >= 0.60:
        L.append(f"- earliest-gold {100*earliest_frac:.0f}% ≥ 60% ⇒ **T1 (data prior) is "
                 "plausible**; D2 (operator+position balanced SFT) is the decisive test.")
    else:
        L.append(f"- earliest-gold {100*earliest_frac:.0f}% < 60%: the data is NOT globally "
                 "earliest-biased; combined with high operator↔position MI this suggests the "
                 "signal is present but unused ⇒ leans T2 (representation). D2 still needed to "
                 "confirm, D3 becomes central.")
    L.append("- Either way, D2's balanced model — not the Phase 0 collapse model — is the "
             "baseline the Phase 1 discriminator must beat.")

    out = os.path.join(ROOT, "results", "d1_data_audit.md")
    open(out, "w").write("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
