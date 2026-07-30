"""End-to-end zero-shot evaluation: LLM-supplied polarity -> signed discriminator.

No gold operator label is used at inference. For each held-out-operator question
the base LLM classifies the temporal polarity (llm_polarity.py, 99.8 % zero-shot),
that prediction is fed to the sign-folded discriminator as its only conditioning,
and we measure top-1. This is the honest zero-shot pipeline: the LLM grounds the
operator's meaning, the discriminator does the value comparison.
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from discriminator.dataset import (build_examples, build_text_embeddings,
                                   CandidateSetDataset, collate)
from discriminator.set_encoder import TemporalSetScorer
from discriminator.train import NEG
from discriminator.llm_polarity import build_model, classify, OP2LABEL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL2OP = {"earliest": "first", "latest": "last", "before": "before", "after": "after"}


@torch.no_grad()
def run(ckpt, eval_ops, use_llm, dev="cuda"):
    state = torch.load(ckpt, map_location=dev)
    a = state["args"]
    te = build_examples("test", tuple(eval_ops))
    tr = build_examples("train", tuple(a["train_ops"]))
    emb = build_text_embeddings(tr + te, device=dev)

    override, pol_acc = {}, None
    if use_llm:
        # dedup questions, classify once, map label->op, measure polarity accuracy
        qs, quids, gold = [], [], []
        seen = {}
        for e in te:
            if e["question"] not in seen:
                seen[e["question"]] = True
                qs.append(e["question"])
        lm, tok = build_model()
        preds = classify(lm, tok, qs)
        qpred = dict(zip(qs, preds))
        del lm; torch.cuda.empty_cache()
        correct = 0
        for e in te:
            lbl = qpred[e["question"]]
            override[e["quid"]] = LABEL2OP[lbl]
            correct += int(lbl == OP2LABEL[e["op"]])
        pol_acc = correct / len(te)

    ds = CandidateSetDataset(te, emb, polarity_override=override)
    ld = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=collate)
    model = TemporalSetScorer(edim=ds.edim, h=a["h"], layers=a["layers"],
                              time_mode=a["time_mode"], pointwise=a["pointwise"],
                              use_leak=a.get("use_leak", False), cond=a.get("cond", "text")).to(dev)
    model.load_state_dict(state["model"]); model.eval()
    per = defaultdict(lambda: {"n": 0, "top1": 0})
    for b in ld:
        bb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        s = model(bb).masked_fill(bb["mask"], NEG)
        pred = s.argmax(1)
        for i, op in enumerate(b["op"]):
            per[op]["n"] += 1
            per[op]["top1"] += int(bb["valid"][i, pred[i]] > 0)
    return per, pol_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval_ops", nargs="+", required=True)
    ap.add_argument("--source", default="llm", choices=["llm", "gold"])
    args = ap.parse_args()
    per, pol_acc = run(args.ckpt, args.eval_ops, use_llm=(args.source == "llm"))
    src = "LLM zero-shot polarity" if args.source == "llm" else "gold polarity"
    print(f"### End-to-end zero-shot — {os.path.basename(args.ckpt)}  ({src})")
    if pol_acc is not None:
        print(f"LLM polarity accuracy on eval questions = {pol_acc:.3f}")
    for op in ("first", "last", "before", "after"):
        if op in per:
            d = per[op]
            print(f"  {op:7s} top-1 {100*d['top1']/d['n']:.1f}%  (n={d['n']})")


if __name__ == "__main__":
    main()
