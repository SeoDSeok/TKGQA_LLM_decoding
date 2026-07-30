"""Combine first/last + before/after signatures into one SFT set (combined model).

Both operator families use the same GCR path format; the completion is the
temporally-correct fact. before/after is subsampled toward the first/last count
so no single operator dominates.
"""
import argparse
import json
import os
import random

from checker.path_format import temporal_path_to_string, wrap_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIG = os.path.join(ROOT, "data", "multitq", "signatures")
RNG = random.Random(7)

PROMPT = (
    "Reasoning path is a sequence of temporal triples in the KG that connects the "
    "topic entity to the answer, where each relation carries the fact's timestamp "
    "in brackets. It starts with <PATH> and ends with </PATH>.\n"
    "# Question:\n{question}\n# Topic entity:\n{topic}\n"
)


def example(s, r, o, ts, question, answer_str, op):
    path = wrap_path(temporal_path_to_string([(s, r, o, ts)]))
    return {"prompt": PROMPT.format(question=question, topic=s),
            "completion": f"{path}\nAnswer: {answer_str}", "op": op}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=os.path.join(ROOT, "data/multitq/train_paths_combined.jsonl"))
    args = ap.parse_args()

    fl = [json.loads(l) for l in open(os.path.join(SIG, f"first_last_time_{args.split}.jsonl"))]
    ba = [json.loads(l) for l in open(os.path.join(SIG, f"before_after_{args.split}.jsonl"))]

    exs = []
    for s in fl:  # first/last: answer is the timestamp
        exs.append(example(s["s"], s["r"], s["o"], s["answer"], s["question"], s["answer"], s["op"]))
    # subsample before/after toward the first/last count for balance
    RNG.shuffle(ba)
    ba = ba[:min(len(ba), len(fl))]
    for s in ba:  # before/after: answer is the entity
        exs.append(example(s["s"], s["r"], s["o"], s["gold_ts"], s["question"], s["o"], s["op"]))

    RNG.shuffle(exs)
    with open(args.out, "w") as f:
        for e in exs:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"{len(exs)} combined examples -> {args.out}")
    print("op mix:", dict(Counter(e["op"] for e in exs)))


if __name__ == "__main__":
    main()
