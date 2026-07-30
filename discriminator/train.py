"""Train the temporal set scorer (InfoNCE + BCE) and report per-operator metrics.

Primary loss — multi-positive set-wise InfoNCE (listwise; the form used at
inference): push the valid candidate(s) above the in-set distractors.
Auxiliary — per-candidate BCE (valid/invalid), λ-weighted, tied to the AUC gate.

Pre-registered zero-shot splits (design §3.3) select which operators appear in
train vs eval via --train_ops / --eval_ops; the default trains and evaluates on
all four.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from discriminator.dataset import (build_examples, build_khop_examples,
                                   build_during_examples, build_text_embeddings,
                                   CandidateSetDataset, collate)
from discriminator.set_encoder import TemporalSetScorer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def to_dev(b, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


NEG = -1e4  # finite mask constant (avoids -inf NaN in logsumexp backward)


def infonce_bce(scores, valid, mask, log_tau, lam=0.3):
    tau = log_tau.exp().clamp(0.05, 5.0)
    s = scores / tau
    real = ~mask
    all_s = s.masked_fill(mask, NEG)                     # drop pads
    pos_s = s.masked_fill(mask | (valid <= 0), NEG)      # keep only valid, real
    l_rank = (torch.logsumexp(all_s, dim=1) - torch.logsumexp(pos_s, dim=1)).mean()
    l_bin = F.binary_cross_entropy_with_logits(scores[real], valid[real])
    return l_rank + lam * l_bin, l_rank.item(), l_bin.item()


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    per = defaultdict(lambda: {"n": 0, "top1": 0})
    all_scores, all_labels = [], []
    for b in loader:
        b = to_dev(b, dev)
        s = model(b)
        valid, mask, ops = b["valid"], b["mask"], b["op"]
        pred = s.masked_fill(mask, NEG).argmax(dim=1)
        for i, op in enumerate(ops):
            per[op]["n"] += 1
            per[op]["top1"] += int(valid[i, pred[i]].item() > 0)
        real = ~mask
        all_scores.append(torch.sigmoid(s[real]).cpu())
        all_labels.append(valid[real].cpu())
    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    auc = roc_auc(labels, scores)
    return per, auc


def roc_auc(y, s):
    # rank-based AUC (no sklearn dependency)
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    npos = y.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def fmt(per):
    return " ".join(f"{op} {100*d['top1']/max(d['n'],1):.1f}%(n{d['n']})"
                    for op, d in sorted(per.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_ops", nargs="+", default=["first", "last", "before", "after"])
    ap.add_argument("--eval_ops", nargs="+", default=["first", "last", "before", "after"])
    ap.add_argument("--fewshot_ops", nargs="+", default=[],
                    help="held-out operators to inject k-shot labeled sets into training")
    ap.add_argument("--kshot", type=int, default=0, help="#labeled sets per few-shot operator")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--time_mode", default="bochner", choices=["bochner", "zero"])
    ap.add_argument("--cond", default="text",
                    choices=["text", "polarity", "polarity_signed", "polarity_interval"],
                    help="condition on question text, LLM abstract polarity, sign-folded "
                         "polarity (breaks 2x2 XOR), or interval predicate (unifies during/equal)")
    ap.add_argument("--pointwise", action="store_true")
    ap.add_argument("--use_leak", action="store_true")
    ap.add_argument("--data_mode", default="isolated", choices=["isolated", "khop", "interval"],
                    help="isolated (s,r,o) sets, k-hop full-subgraph (T3), or interval "
                         "(first/last/before/after + synthetic during) for polarity_interval")
    ap.add_argument("--split_mode", default="question", choices=["question", "sr_disjoint", "entity_disjoint"],
                    help="S1 audit: re-split combined data so train/test share no (s,r) group "
                         "or no subject entity — tests for group memorization")
    ap.add_argument("--drop_during", action="store_true",
                    help="interval mode: exclude during from TRAIN (zero-shot during test)")
    ap.add_argument("--k_max", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0, help="overfit debug: cap #train sets")
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--out", default=os.path.join(ROOT, "save_models", "discriminator"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    ap.add_argument("--abs_only", action="store_true",
                    help="ablation: drop group-min/max rescaling of the relative channel")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = args.device if getattr(args, "device", None) else ("cuda" if torch.cuda.is_available() else "cpu")

    if args.data_mode == "khop":
        tr = build_khop_examples("train", k_max=args.k_max, seed=args.seed)
        te = build_khop_examples("test", k_max=args.k_max, seed=args.seed + 1)
        tr = [e for e in tr if e["op"] in args.train_ops]
        te = [e for e in te if e["op"] in args.eval_ops]
    elif args.data_mode == "interval":
        tr = build_examples("train", ("first", "last", "before", "after"))
        te = build_examples("test", ("first", "last", "before", "after"))
        dtr = build_during_examples("train"); dte = build_during_examples("test", seed=1)
        if not args.drop_during:
            tr = tr + dtr
        te = te + dte  # always evaluate during (zero-shot if --drop_during)
    else:
        tr = build_examples("train", tuple(args.train_ops))
        te = build_examples("test", tuple(args.eval_ops))
    if args.split_mode != "question":
        # S1 audit: merge both splits and re-partition by GROUP so train/test are
        # disjoint on (s,r) or subject entity — a 99% that survives this is not
        # memorizing groups. Deterministic hash assignment, ~80/20.
        import hashlib as _h
        allex = tr + te
        def _key(e):
            r = e["cands"][0]["r"] if e["cands"] else ""
            return e["s"] if args.split_mode == "entity_disjoint" else f"{e['s']}|||{r}"
        def _bucket(k):
            return int(_h.md5(k.encode()).hexdigest(), 16) % 10
        tr = [e for e in allex if _bucket(_key(e)) >= 2 and e["op"] in args.train_ops]
        te = [e for e in allex if _bucket(_key(e)) < 2 and e["op"] in args.eval_ops]
        tr_groups = {_key(e) for e in tr}; te_groups = {_key(e) for e in te}
        overlap = tr_groups & te_groups
        print(f"[{args.split_mode}] train={len(tr)} test={len(te)}  "
              f"groups tr={len(tr_groups)} te={len(te_groups)} OVERLAP={len(overlap)} (must be 0)")
    if args.fewshot_ops and args.kshot > 0:
        rng = np.random.RandomState(args.seed)
        pool = build_examples("train", tuple(args.fewshot_ops))
        by_op = defaultdict(list)
        for e in pool:
            by_op[e["op"]].append(e)
        added = {}
        for op in args.fewshot_ops:
            idx = rng.permutation(len(by_op[op]))[: args.kshot]
            shots = [by_op[op][i] for i in idx]
            tr += shots; added[op] = len(shots)
        print(f"few-shot inject: {added}")
    emb = build_text_embeddings(tr + te, device=dev)
    if args.limit:
        tr = tr[: args.limit]
    dtr = CandidateSetDataset(tr, emb, use_rank_leak=args.use_leak, abs_only=args.abs_only)
    dte = CandidateSetDataset(te, emb, use_rank_leak=args.use_leak, abs_only=args.abs_only)
    ltr = DataLoader(dtr, batch_size=args.bs, shuffle=True, collate_fn=collate, drop_last=False)
    lte = DataLoader(dte, batch_size=128, shuffle=False, collate_fn=collate)
    print(f"train sets={len(tr)}  test sets={len(te)}  edim={dtr.edim}")

    model = TemporalSetScorer(edim=dtr.edim, h=args.h, layers=args.layers,
                              time_mode=args.time_mode, pointwise=args.pointwise,
                              use_leak=args.use_leak, cond=args.cond).to(dev)
    log_tau = torch.nn.Parameter(torch.zeros((), device=dev))
    print(f"params={model.n_params()/1e6:.2f}M  time_mode={args.time_mode} "
          f"pointwise={args.pointwise} use_leak={args.use_leak}")
    opt = torch.optim.AdamW(list(model.parameters()) + [log_tau], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * max(1, len(ltr)))

    best = -1; hist = []
    for ep in range(args.epochs):
        model.train(); tot = lr = lb = 0.0; nb = 0
        for b in ltr:
            b = to_dev(b, dev)
            s = model(b)
            loss, lrk, lbn = infonce_bce(s, b["valid"], b["mask"], log_tau, args.lam)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item(); lr += lrk; lb += lbn; nb += 1
        per, auc = evaluate(model, lte, dev)
        macro = np.mean([100 * d["top1"] / max(d["n"], 1) for d in per.values()])
        late = np.mean([100 * per[o]["top1"] / max(per[o]["n"], 1)
                        for o in ("last", "after") if o in per]) if any(
                        o in per for o in ("last", "after")) else float("nan")
        hist.append({"epoch": ep, "loss": tot/nb, "l_rank": lr/nb, "l_bin": lb/nb,
                     "auc": auc, "macro_top1": macro, "late_top1": late,
                     "per_op": {o: round(100*d["top1"]/max(d["n"],1), 1) for o, d in per.items()}})
        print(f"ep{ep:2d} loss {tot/nb:.3f} (rank {lr/nb:.3f} bin {lb/nb:.3f}) "
              f"tau {log_tau.exp().item():.2f} | AUC {auc:.3f} macro {macro:.1f} "
              f"late {late:.1f} | {fmt(per)}")
        # select on macro (balanced across all ops); late-only selection is
        # degenerate — a "later-is-better" model aces last/after but tanks before.
        score = macro
        if score > best:
            best = score
            os.makedirs(args.out, exist_ok=True)
            torch.save({"model": model.state_dict(), "log_tau": log_tau.detach(),
                        "args": vars(args)}, os.path.join(args.out, f"{args.tag}.pt"))
    # persist metrics
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump({"args": vars(args), "history": hist, "best_late_or_macro": best},
              open(os.path.join(ROOT, "results", f"discriminator_{args.tag}_train.json"), "w"), indent=2)
    print(f"\nBEST macro top-1 = {best:.1f}  (saved {args.tag}.pt)")


if __name__ == "__main__":
    main()
