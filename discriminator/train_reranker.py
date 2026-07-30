"""S1 — train the answer-relevance reranker.

Reuses TemporalSetScorer(cond=text) and train.infonce_bce, but on answer-membership
candidate sets (scripts/build_rerank_data.py): the listwise InfoNCE pushes the gold-answer
candidate above the distractors, learning "which candidate is the answer" — complementary
to the temporal disc's "which timestamp is valid". Saves a checkpoint in the same format
phase2_system_* loaders expect (dict with model/log_tau/args).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from discriminator.dataset import build_text_embeddings, CandidateSetDataset, collate
from discriminator.set_encoder import TemporalSetScorer
from discriminator.train import infonce_bce


def load_examples(path):
    return [json.loads(l) for l in open(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="rerank_train.jsonl from build_rerank_data.py")
    ap.add_argument("--val", default="", help="optional held-out jsonl for top-1 logging")
    ap.add_argument("--cond", default="text")
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap #train sets (debug)")
    ap.add_argument("--out", default=os.path.join(ROOT, "save_models/discriminator/rerank_ans.pt"))
    args = ap.parse_args()
    dev = args.device

    ex = load_examples(args.data)
    if args.limit:
        ex = ex[: args.limit]
    emb = build_text_embeddings(ex, device=dev)
    ds = CandidateSetDataset(ex, emb)
    ld = DataLoader(ds, batch_size=args.bs, shuffle=True, collate_fn=collate, drop_last=False)
    print(f"reranker train: {len(ex)} sets, edim={ds.edim}, cond={args.cond}")

    model = TemporalSetScorer(edim=ds.edim, h=args.h, layers=args.layers,
                              time_mode="bochner", pointwise=False, use_leak=False,
                              cond=args.cond).to(dev)
    log_tau = torch.nn.Parameter(torch.zeros((), device=dev))
    opt = torch.optim.AdamW(list(model.parameters()) + [log_tau], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * max(1, len(ld)))

    for ep in range(args.epochs):
        model.train(); tot = nb = 0.0
        for b in ld:
            bb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            s = model(bb)
            loss, lr_, lb_ = infonce_bce(s, bb["valid"], bb["mask"], log_tau, args.lam)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item(); nb += 1
        # train top-1 (gold-answer candidate ranked first within its set)
        model.eval(); hit = n = 0
        with torch.inference_mode():
            for b in ld:
                bb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
                s = model(bb).masked_fill(bb["mask"], -1e4)
                pred = s.argmax(1)
                for i in range(s.shape[0]):
                    hit += int(bb["valid"][i, pred[i]].item() > 0); n += 1
        print(f"ep{ep:2d} loss {tot/nb:.3f} tau {log_tau.exp().item():.2f} | train top-1 {100*hit/max(n,1):.1f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": model.state_dict(), "log_tau": log_tau.detach(),
                "args": {"h": args.h, "layers": args.layers, "time_mode": "bochner",
                         "pointwise": False, "use_leak": False, "cond": args.cond}}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
