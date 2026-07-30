"""Convert answer-verified signatures into GCR-format temporal SFT data.

Each verified first/last signature (s, r, o, answer_ts) becomes an SFT example
whose target is the temporally-correct reasoning path in the timestamp-augmented
format, using the bracketed timestamp special token `[YYYY-MM-DD]`.

Output: data/multitq/train_paths_{split}.jsonl  with {prompt, completion}.
"""
import argparse
import json
import os

from checker.path_format import temporal_path_to_string, wrap_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIG = os.path.join(ROOT, "data", "multitq", "signatures")

PROMPT = (
    "Reasoning path is a sequence of temporal triples in the KG that connects the "
    "topic entity to the answer, where each relation carries the fact's timestamp "
    "in brackets. It starts with <PATH> and ends with </PATH>.\n"
    "# Question:\n{question}\n# Topic entity:\n{topic}\n"
)


def to_example(sig):
    path = wrap_path(temporal_path_to_string([(sig["s"], sig["r"], sig["o"], sig["answer"])]))
    prompt = PROMPT.format(question=sig["question"], topic=sig["s"])
    completion = f"{path}\nAnswer: {sig['answer']}"
    return {"prompt": prompt, "completion": completion,
            "quid": sig["quid"], "op": sig["op"], "n_candidates": sig["n_candidates"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    src = os.path.join(SIG, f"first_last_time_{args.split}.jsonl")
    sigs = [json.loads(l) for l in open(src)]
    out = os.path.join(ROOT, "data", "multitq", f"train_paths_{args.split}.jsonl")
    with open(out, "w") as f:
        for s in sigs:
            f.write(json.dumps(to_example(s), ensure_ascii=False) + "\n")
    print(f"{len(sigs)} examples -> {out}")
    ex = to_example(sigs[0])
    print("\n--- sample ---")
    print("PROMPT:\n" + ex["prompt"])
    print("COMPLETION:\n" + ex["completion"])


if __name__ == "__main__":
    main()
