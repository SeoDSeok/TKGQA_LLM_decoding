"""D2: build operator + gold-position balanced SFT data.

Decisive test of T1 (data prior). The cleanest isolation uses first/last
multi-candidate signatures: equal counts of `first` (gold = earliest) and `last`
(gold = latest) → earliest-gold = 50 %, latest-gold = 50 %, and the operator
perfectly predicts gold position (MI = 1 bit). If a model trained on this STILL
always picks earliest, the collapse is not a label prior (T1 ruled out) → T2
(representation).

Optionally folds in before/after with gold re-picked to balance the correct-side
position (earliest-correct vs latest-correct, alternating) for scale.

Output: data/multitq/train_paths_balanced.jsonl
"""
import argparse
import json
import os
import pickle
import random

from checker.path_format import temporal_path_to_string, wrap_path
from checker.timepoint import TimePoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIG = os.path.join(ROOT, "data", "multitq", "signatures")
IDX = os.path.join(ROOT, "data", "multitq", "index", "kg_index.pkl")
RNG = random.Random(11)

PROMPT = (
    "Reasoning path is a sequence of temporal triples in the KG that connects the "
    "topic entity to the answer, where each relation carries the fact's timestamp "
    "in brackets. It starts with <PATH> and ends with </PATH>.\n"
    "# Question:\n{question}\n# Topic entity:\n{topic}\n")


def ex(s, r, o, ts, q, ans, op):
    path = wrap_path(temporal_path_to_string([(s, r, o, ts)]))
    return {"prompt": PROMPT.format(question=q, topic=s),
            "completion": f"{path}\nAnswer: {ans}", "op": op}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include_ba", action="store_true", help="also add balanced before/after")
    ap.add_argument("--out", default=os.path.join(ROOT, "data/multitq/train_paths_balanced.jsonl"))
    args = ap.parse_args()

    # first/last: multi-candidate only
    fl = [json.loads(l) for l in open(os.path.join(SIG, "first_last_time_train.jsonl"))]
    firsts = [s for s in fl if s["op"] == "first" and s["n_candidates"] >= 2]
    lasts = [s for s in fl if s["op"] == "last" and s["n_candidates"] >= 2]
    k = min(len(firsts), len(lasts))
    RNG.shuffle(firsts); RNG.shuffle(lasts)
    firsts, lasts = firsts[:k], lasts[:k]
    exs = []
    for s in firsts + lasts:
        exs.append(ex(s["s"], s["r"], s["o"], s["answer"], s["question"], s["answer"], s["op"]))

    n_ba = 0
    if args.include_ba:
        idx = pickle.load(open(IDX, "rb")); sr2ots = idx["sr2ots"]
        ba = [json.loads(l) for l in open(os.path.join(SIG, "before_after_train.jsonl"))]
        RNG.shuffle(ba)
        for i, s in enumerate(ba):
            anchor = TimePoint.parse(s["anchor"])
            facts = sr2ots.get((s["s"], s["r"]), [])
            side = [(TimePoint.parse(t), o) for t, o in facts
                    if (TimePoint.parse(t).strictly_before(anchor) if s["op"] == "before"
                        else TimePoint.parse(t).strictly_after(anchor))]
            if len(side) < 2:
                continue
            side.sort()
            tp, o = side[0] if i % 2 == 0 else side[-1]   # alternate earliest/latest-correct
            exs.append(ex(s["s"], s["r"], o, str(tp), s["question"], o, s["op"]))
            n_ba += 1
            if n_ba >= 2 * k:  # keep before/after from dominating
                break

    RNG.shuffle(exs)
    with open(args.out, "w") as f:
        for e in exs:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"{len(exs)} balanced examples -> {args.out}")
    print("op mix:", dict(Counter(e["op"] for e in exs)))
    print(f"first/last multi-candidate per class: {k}")


if __name__ == "__main__":
    main()
