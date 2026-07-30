"""S1 data — answer-relevance reranker training sets.

Reads a reconstructed-signatures jsonl (from build_fulltest_signatures.py, any split),
joins each instance to its question, and emits candidate-set examples whose per-candidate
`valid = (candidate's answer in gold_answers)` — i.e. an ANSWER-membership label (not the
temporal-validity label the discriminator trains on). Training TemporalSetScorer(cond=text)
on these = a learned answer-relevance reranker, complementary to the temporal disc.

Output schema matches discriminator.dataset.CandidateSetDataset:
  {quid, op, question, s, cands:[{o,r,t_str,frac,valid}], anchor_frac}
Only sets with >=1 valid and >=1 invalid (a real ranking decision) are kept.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from checker.timepoint import TimePoint
from discriminator.encode_time import date_to_fracyear
from discriminator.phase2_system_e1 import fine_op, answer_of
from scripts.build_before_after_signatures import parse_anchor, strict_norm

QDIR = os.path.join(ROOT, "data", "multitq", "questions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigs", required=True, help="reconstructed signatures jsonl")
    ap.add_argument("--split", default="train", help="questions split to join (train/dev/test)")
    ap.add_argument("--k_max", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    questions = {q["quid"]: q["question"] for q in json.load(open(f"{QDIR}/{args.split}.json"))}
    kept = 0
    with open(args.out, "w") as fout:
        for line in open(args.sigs):
            d = json.loads(line)
            if not d["resolvable"] or not d["candidates"]:
                continue
            q = questions.get(d["quid"])
            if q is None:
                continue
            atype, tlevel = d["answer_type"], d["time_level"]
            gold = {strict_norm(a) for a in d["gold_answers"]}
            cands = []
            for o, t in d["candidates"][: args.k_max * 4]:
                tp = TimePoint.parse(t)
                ans = answer_of(o, t, atype, tlevel)
                cands.append({"o": o, "r": d["relation"], "t_str": str(tp),
                              "frac": date_to_fracyear(tp),
                              "valid": int(strict_norm(ans) in gold)})
            nv = sum(c["valid"] for c in cands)
            if nv == 0 or nv == len(cands) or len(cands) < 2:
                continue                                   # need a real ranking decision
            if len(cands) > args.k_max:
                v = [c for c in cands if c["valid"]]
                iv = [c for c in cands if not c["valid"]]
                cands = (v + iv)[: args.k_max]
            anchor_frac = None
            if d.get("anchor"):
                anchor_frac = date_to_fracyear(TimePoint.parse(d["anchor"]))
            ex = {"quid": d["quid"], "op": fine_op(d["qtype"], q), "question": q,
                  "s": d["topic"], "cands": cands, "anchor_frac": anchor_frac}
            fout.write(json.dumps(ex) + "\n")
            kept += 1
    print(f"wrote {args.out}  ({kept} reranker training sets)")


if __name__ == "__main__":
    main()
