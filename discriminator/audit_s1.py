"""S1 audit finalizer — n / k-distribution / bootstrap CI / per-k random baseline
and a 50-sample manual audit of the 100 % claims, for any discriminator checkpoint
under any split (question / sr_disjoint / entity_disjoint).

One GPU inference pass; all statistics are CPU. Confirms the headline accuracies are
not group-memorization or small-n / easy-k artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from discriminator.dataset import build_examples, build_text_embeddings, CandidateSetDataset, collate
from discriminator.set_encoder import TemporalSetScorer
from discriminator.train import NEG


def group_key(e, mode):
    r = e["cands"][0]["r"] if e["cands"] else ""
    return e["s"] if mode == "entity_disjoint" else f"{e['s']}|||{r}"


def bucket(k):
    return int(hashlib.md5(k.encode()).hexdigest(), 16) % 10


def eval_split(ckpt, split_mode, eval_ops, dev="cuda", seed=0):
    state = torch.load(ckpt, map_location=dev); a = state["args"]
    tr = build_examples("train", ("first", "last", "before", "after"))
    te = build_examples("test", ("first", "last", "before", "after"))
    if split_mode == "question":
        ex = [e for e in te if e["op"] in eval_ops]
    else:
        allex = tr + te
        ex = [e for e in allex if bucket(group_key(e, split_mode)) < 2 and e["op"] in eval_ops]
    emb = build_text_embeddings(tr + te, device=dev)
    ds = CandidateSetDataset(ex, emb, use_rank_leak=a.get("use_leak", False))
    ld = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=collate)
    model = TemporalSetScorer(edim=ds.edim, h=a["h"], layers=a["layers"], time_mode=a["time_mode"],
                              pointwise=a["pointwise"], use_leak=a.get("use_leak", False),
                              cond=a.get("cond", "text")).to(dev)
    model.load_state_dict(state["model"]); model.eval()
    rows = []  # (op, k, correct, gold_idx, pick_idx)
    with torch.inference_mode():
        bi = 0
        for b in ld:
            bb = {k_: (v.to(dev) if torch.is_tensor(v) else v) for k_, v in b.items()}
            s = model(bb).masked_fill(bb["mask"], NEG)
            pred = s.argmax(1)
            for i, op in enumerate(b["op"]):
                k = int((~bb["mask"][i]).sum())
                correct = int(bb["valid"][i, pred[i]] > 0)
                rows.append((op, k, correct, int(pred[i]), ex[bi]))
                bi += 1
    return rows


def bootstrap_ci(correct, B=2000, seed=0):
    rng = np.random.RandomState(seed)
    c = np.array(correct)
    if len(c) == 0:
        return (float("nan"), float("nan"))
    means = [c[rng.randint(0, len(c), len(c))].mean() for _ in range(B)]
    return (100 * np.percentile(means, 2.5), 100 * np.percentile(means, 97.5))


def kbin(k):
    return "k=2" if k == 2 else ("k=3" if k == 3 else "k>=4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split_mode", default="question", choices=["question", "sr_disjoint", "entity_disjoint"])
    ap.add_argument("--eval_ops", nargs="+", default=["first", "last", "before", "after"])
    ap.add_argument("--audit_n", type=int, default=50)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rows = eval_split(args.ckpt, args.split_mode, args.eval_ops)
    by_op = defaultdict(list)
    for op, k, c, pk, e in rows:
        by_op[op].append((k, c, pk, e))

    L = [f"### {os.path.basename(args.ckpt)}  ·  split=`{args.split_mode}`", "",
         "| operator | n | top-1 | 95% CI | k=2 | k=3 | k>=4 | mean random 1/k |",
         "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for op in ("first", "last", "before", "after"):
        if op not in by_op:
            continue
        items = by_op[op]
        n = len(items); acc = 100 * sum(c for _, c, _, _ in items) / n
        lo, hi = bootstrap_ci([c for _, c, _, _ in items])
        kacc = defaultdict(list)
        for k, c, _, _ in items:
            kacc[kbin(k)].append(c)
        def kp(kb):
            v = kacc.get(kb, [])
            return f"{100*sum(v)/len(v):.1f}%" if v else "—"
        rnd = 100 * np.mean([1.0 / k for k, _, _, _ in items])
        L.append(f"| {op} | {n} | {acc:.1f}% | [{lo:.1f}, {hi:.1f}] | {kp('k=2')} | {kp('k=3')} | "
                 f"{kp('k>=4')} | {rnd:.1f}% |")

    # manual-audit sample (mix of correct/incorrect), for the 100% claims
    rng = np.random.RandomState(0)
    allitems = [(op, k, c, pk, e) for op, k, c, pk, e in rows]
    idx = rng.permutation(len(allitems))[: args.audit_n]
    L += ["", f"## Manual-audit sample ({args.audit_n} random)", ""]
    for j in idx[:12]:  # show first 12 in the md; full set in json
        op, k, c, pk, e = allitems[j]
        cand = e["cands"][pk]
        gold = next((cc for cc in e["cands"] if cc["valid"] > 0), None)
        L.append(f"- [{'✓' if c else '✗'}] ({op}, k={k}) pick t=`{cand.get('t_str','?')}` "
                 f"gold t=`{gold.get('t_str','?') if gold else '?'}` — {e['question'][:70]}")
    md = "\n".join(L)
    print(md)
    if args.out:
        open(args.out, "a").write("\n\n" + md + "\n")
        audit = [{"op": op, "k": k, "correct": c,
                  "pick_t": e["cands"][pk].get("t_str"),
                  "gold_t": next((cc.get("t_str") for cc in e["cands"] if cc["valid"] > 0), None),
                  "q": e["question"]} for j in idx for (op, k, c, pk, e) in [allitems[j]]]
        json.dump(audit, open(args.out.replace(".md", f"_{args.split_mode}_audit.json"), "w"),
                  indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
