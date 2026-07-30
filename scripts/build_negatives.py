"""Track C: temporal-perturbation negative generator for the Phase 1 discriminator.

Plan v3 contribution #2: learn temporal constraints operator-agnostically via
contrastive training. From each answer-verified signature (a *positive*
temporally-correct path) we synthesize hard negatives with 4 perturbation types:

  1. timestamp_swap    — replace the correct timestamp with another timestamp
                         from the SAME (s,r,o) group (structurally identical,
                         temporally wrong: the core first/last trap).
  2. hop_reorder       — for multi-hop paths, reverse chronological order of hops
                         (here single-hop, so emitted only when >=2 hops exist).
  3. interval_shift    — replace with a timestamp OUTSIDE the group's span
                         (a fact-time that violates any before/after window).
  4. granularity_noise — corrupt the granularity (day->month/year truncation or
                         off-by-one) so the timestamp no longer matches exactly.

Each negative is labelled with its perturbation type so the discriminator's
training signal can later be aligned with the *realized* violation distribution
(the early-bias / pick-min failure measured in Track A — see
results/phase0_violation_report.md).

Outputs:
  data/multitq/discriminator/contrastive_{split}.jsonl
  results/negatives_report.md
"""
import argparse
import json
import os
import random

from checker.path_format import temporal_path_to_string, wrap_path
from checker.timepoint import TimePoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIG = os.path.join(ROOT, "data", "multitq", "signatures")
# deterministic RNG (Math.random-free constraint doesn't apply here, but keep it seeded)
RNG = random.Random(1234)


def path_str(s, r, o, ts):
    return wrap_path(temporal_path_to_string([(s, r, o, ts)]))


def perturb(sig):
    """Return list of (neg_path, ptype) for one positive signature."""
    s, r, o = sig["s"], sig["r"], sig["o"]
    cand = sorted(set(sig["candidate_timestamps"]))
    gold = sig["answer"]
    negs = []

    # 1) timestamp_swap: another timestamp from the same group
    others = [t for t in cand if t != gold]
    if others:
        negs.append((path_str(s, r, o, RNG.choice(others)), "timestamp_swap"))

    # 3) interval_shift: a timestamp outside the group's span
    lo, hi = TimePoint.parse(cand[0]), TimePoint.parse(cand[-1])
    shifted = f"{max(2005, lo.year - 3):04d}-{lo.month:02d}-{lo.day:02d}"
    if shifted not in cand:
        negs.append((path_str(s, r, o, shifted), "interval_shift"))

    # 4) granularity_noise: truncate to month (drops the day) or off-by-one day
    gtp = TimePoint.parse(gold)
    if gtp.granularity == "day":
        # off-by-one day, staying a valid-looking but wrong exact date
        try:
            noisy = f"{gtp.year:04d}-{gtp.month:02d}-{(gtp.day % 28) + 1:02d}"
            if noisy != gold:
                negs.append((path_str(s, r, o, noisy), "granularity_noise"))
        except Exception:
            pass

    # 2) hop_reorder: only meaningful for multi-hop; recorded as N/A for single-hop
    #    (kept in the taxonomy for when Track B adds 2-hop composite signatures)
    return negs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    src = os.path.join(SIG, f"first_last_time_{args.split}.jsonl")
    sigs = [json.loads(l) for l in open(src)]

    out_dir = os.path.join(ROOT, "data", "multitq", "discriminator")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"contrastive_{args.split}.jsonl")

    from collections import Counter
    ptype_counts = Counter()
    n_pos = n_neg = 0
    with open(out_path, "w") as f:
        for sig in sigs:
            pos = path_str(sig["s"], sig["r"], sig["o"], sig["answer"])
            rec = {"quid": sig["quid"], "question": sig["question"], "op": sig["op"],
                   "positive": pos, "negatives": []}
            for neg_path, ptype in perturb(sig):
                rec["negatives"].append({"path": neg_path, "ptype": ptype})
                ptype_counts[ptype] += 1
                n_neg += 1
            if rec["negatives"]:
                n_pos += 1
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = [
        "# Contrastive negatives for the temporal discriminator (Track C)",
        "",
        f"- split: `{args.split}`  ·  positives with >=1 negative: **{n_pos}**  ·  "
        f"total negatives: **{n_neg}**  (avg {n_neg/max(n_pos,1):.2f}/positive)",
        "",
        "| perturbation type | count | role |",
        "|---|--:|---|",
        f"| timestamp_swap | {ptype_counts['timestamp_swap']} | same (s,r,o), wrong time — the first/last trap |",
        f"| interval_shift | {ptype_counts['interval_shift']} | time outside the group span — before/after violation |",
        f"| granularity_noise | {ptype_counts['granularity_noise']} | off-by-one / wrong-granularity date |",
        f"| hop_reorder | {ptype_counts['hop_reorder']} | (reserved — needs 2-hop signatures from Track B) |",
        "",
        "Alignment note: Track A found the model's realized failure is a **pick-min /",
        "early-bias** (picks earliest timestamp 99.8% of the time). `timestamp_swap`",
        "negatives directly cover this failure mode; the discriminator should weight",
        "them to match the realized violation distribution.",
        "",
        f"Saved: `{os.path.relpath(out_path, ROOT)}`",
    ]
    rep = os.path.join(ROOT, "results", "negatives_report.md")
    open(rep, "w").write("\n".join(report))
    print("\n".join(report))
    print(f"\nwrote {out_path} and {rep}")


if __name__ == "__main__":
    main()
