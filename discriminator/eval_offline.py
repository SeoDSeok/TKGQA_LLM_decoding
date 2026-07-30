"""Offline evaluation for the temporal discriminator (Phase 1 gate).

Reports, on the held-out test candidate sets:
  * per-operator top-1 accuracy (the TVR proxy) — the gate is last/after ≥ 90 %;
  * a k-stratified breakdown (guards against the "mostly-2-candidate looks easy"
    artifact, design §6.2) with a separate k≥4 column;
  * valid/invalid AUC;
  * the standing baselines the discriminator must beat, scored on the *same* sets:
      - raw D3-full model     (span-logprob ranking, last 46.2 %)
      - earliest-heuristic     (always pick min timestamp — the collapse policy)
      - oracle                 (rule checker, 100 % ceiling).
The raw-model column is read from the existing D3-full result rather than re-run.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from discriminator.dataset import build_examples, build_text_embeddings, CandidateSetDataset, collate
from discriminator.set_encoder import TemporalSetScorer
from discriminator.train import roc_auc, NEG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def kbin(k):
    return "k=2" if k == 2 else ("k=3" if k == 3 else "k>=4")


@torch.no_grad()
def eval_model(ckpt, eval_ops, dev="cuda"):
    state = torch.load(ckpt, map_location=dev)
    a = state["args"]
    te = build_examples("test", tuple(eval_ops))
    tr = build_examples("train", tuple(a["train_ops"]))
    emb = build_text_embeddings(tr + te, device=dev)
    ds = CandidateSetDataset(te, emb, use_rank_leak=a.get("use_leak", False))
    ld = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=collate)
    model = TemporalSetScorer(edim=ds.edim, h=a["h"], layers=a["layers"],
                              time_mode=a["time_mode"], pointwise=a["pointwise"],
                              use_leak=a.get("use_leak", False)).to(dev)
    model.load_state_dict(state["model"]); model.eval()

    per = defaultdict(lambda: defaultdict(lambda: {"n": 0, "top1": 0}))
    earliest = defaultdict(lambda: {"n": 0, "top1": 0})
    all_s, all_y = [], []
    for b in ld:
        bb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        s = model(bb).masked_fill(bb["mask"], NEG)
        pred = s.argmax(1)
        valid, mask, ops = bb["valid"], bb["mask"], b["op"]
        t_abs = bb["t_abs"].masked_fill(mask, float("inf"))
        emin = t_abs.argmin(1)  # earliest-timestamp heuristic
        for i, op in enumerate(ops):
            k = int((~mask[i]).sum().item())
            per[op][kbin(k)]["n"] += 1
            per[op][kbin(k)]["top1"] += int(valid[i, pred[i]] > 0)
            per[op]["all"]["n"] += 1
            per[op]["all"]["top1"] += int(valid[i, pred[i]] > 0)
            earliest[op]["n"] += 1
            earliest[op]["top1"] += int(valid[i, emin[i]] > 0)
        real = ~mask
        all_s.append(torch.sigmoid(s[real]).cpu()); all_y.append(valid[real].cpu())
    auc = roc_auc(torch.cat(all_y).numpy(), torch.cat(all_s).numpy())
    return per, earliest, auc


def pct(d):
    return 100 * d["top1"] / d["n"] if d["n"] else float("nan")


def report(per, earliest, auc, title):
    L = [f"### {title}", "",
         "| operator | all top-1 | k=2 | k=3 | k>=4 | earliest-heuristic |",
         "|---|--:|--:|--:|--:|--:|"]
    for op in ("first", "last", "before", "after"):
        if op not in per:
            continue
        p = per[op]
        L.append(f"| {op} | {pct(p['all']):.1f}% ({p['all']['n']}) | "
                 f"{pct(p['k=2']):.1f}% | {pct(p['k=3']):.1f}% | {pct(p['k>=4']):.1f}% | "
                 f"{pct(earliest[op]):.1f}% |")
    L.append("")
    L.append(f"**valid/invalid AUC (in-set) = {auc:.3f}**")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "save_models/discriminator/v0.pt"))
    ap.add_argument("--eval_ops", nargs="+", default=["first", "last", "before", "after"])
    ap.add_argument("--title", default="Discriminator v0 (all ops)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    per, earliest, auc = eval_model(args.ckpt, args.eval_ops)
    md = report(per, earliest, auc, args.title)
    print(md)
    if args.out:
        open(args.out, "a").write("\n\n" + md + "\n")


if __name__ == "__main__":
    main()
