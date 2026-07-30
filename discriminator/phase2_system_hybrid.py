"""E1-hybrid — dense-retrieve the (s,r), KG-expand its sibling timestamps, then
trie-local discriminator fusion. The front-end ablation (phase2_system_dense)
showed dense retrieval reaches high candidate recall (75.5% @80) but the disc
cannot help because dense sets lack the (s,r)-sibling groups it re-ranks. This
runner restores that structure:

  1. dense-retrieve top-N facts for the question (100% coverage front-end);
  2. from each retrieved fact, take the (topic, r, direction) whose topic entity
     appears in the question -> a candidate group;
  3. KG-EXPAND each group to ALL its (o,t) siblings (sr2ots / or2sts) -> the
     sibling-timestamp structure the trie-local scorer needs;
  4. fuse logP_LLM + alpha * within-group log-softmax(s_theta) (trie-local).

Recall is now "is the gold's (s,r) group among the top-M retrieved groups",
which can exceed the fact-level recall because KG expansion guarantees the
sibling timestamps once the group is found. `--recall_only` reports that group
recall + disc-only Hits@1 (minutes on GPU) before the long decode.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from torch.utils.data import DataLoader
from discriminator.dataset import build_text_embeddings, CandidateSetDataset, collate
from discriminator.set_encoder import TemporalSetScorer
from discriminator.train import NEG
from discriminator.encode_time import date_to_fracyear
from discriminator.phase2_system_e1 import fine_op, answer_of, rank_of_gold, PROMPT
from checker.timepoint import TimePoint
from checker.path_format import wrap_path, temporal_path_to_string
from scripts.build_before_after_signatures import parse_anchor, strict_norm
from scripts.multitq_linker import norm

QFILE = os.path.join(ROOT, "data", "multitq", "questions", "test.json")
DENSE = os.path.join(ROOT, "data", "multitq", "index", "fact_dense.pkl")
KGIDX = os.path.join(ROOT, "data", "multitq", "index", "kg_index.pkl")
NEEDS_ANCHOR = {"equal", "equal_multi", "before_after", "after_first", "before_last"}


def mentioned(ent, qnorm):
    n = norm(ent)
    return len(n) >= 3 and (" " + n + " ") in qnorm


def build_examples(topn=80, groups_m=8, per_group=15, k_max=40, n_sample=0,
                   seed=0, device="cuda", dense_path=DENSE, anchor_mode="entity"):
    rng = np.random.RandomState(seed)
    qs = json.load(open(QFILE))
    if n_sample and n_sample < len(qs):
        idx = rng.choice(len(qs), size=n_sample, replace=False)
        qs = [qs[i] for i in sorted(idx)]
    kg = pickle.load(open(KGIDX, "rb"))
    sr2ots, or2sts = kg["sr2ots"], kg["or2sts"]

    pack = pickle.load(open(dense_path, "rb"))
    facts, emb = pack["facts"], pack["emb"].to(device).float()
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    qemb = enc.encode([q["question"] for q in qs], batch_size=512, convert_to_tensor=True,
                      normalize_embeddings=True, show_progress_bar=True).to(device).float()

    out = []
    B = 256
    for i0 in range(0, len(qs), B):
        top = (qemb[i0:i0 + B] @ emb.T).topk(topn, dim=1).indices.cpu().numpy()
        for j, q in enumerate(qs[i0:i0 + B]):
            qnorm = " " + norm(q["question"]) + " "
            gold = {strict_norm(a) for a in q["answers"]}
            atype, tlevel = q["answer_type"], q["time_level"]
            op = fine_op(q["qtype"], q["question"])
            # 1-2. candidate (topic, r, direction) groups from retrieved facts,
            # ordered by retrieval rank, topic = the fact entity that is in the question.
            groups, seen_g = [], set()
            for k in top[j]:
                s, r, o, t = facts[k]
                s_in, o_in = mentioned(s, qnorm), mentioned(o, qnorm)
                if s_in:
                    key = (s, r, "subj")
                elif o_in:
                    key = (o, r, "obj")
                else:
                    continue
                if key not in seen_g:
                    seen_g.add(key); groups.append(key)
                if len(groups) >= groups_m:
                    break
            if not groups:  # nothing mentioned: fall back to top fact's subject
                s, r, o, t = facts[top[j][0]]
                groups = [(s, r, "subj")]
            # 3. KG-expand each group to its sibling (o,t); cap per group.
            cands, topic = [], groups[0][0]
            for gi, (e, r, d) in enumerate(groups):
                sib = sr2ots.get((e, r), []) if d == "subj" else or2sts.get((e, r), [])
                sib = [(o2, t2) for (t2, o2) in sib]           # index stores (t, o)
                if len(sib) > per_group:
                    sib = [sib[x] for x in rng.choice(len(sib), per_group, replace=False)]
                for (o2, t2) in sib:
                    tp = TimePoint.parse(t2)
                    ans = answer_of(o2, t2, atype, tlevel)
                    cands.append({"o": o2, "r": r, "t_str": str(tp),
                                  "frac": date_to_fracyear(tp),
                                  "valid": int(strict_norm(ans) in gold),
                                  "answer": ans, "direction": d, "group": gi,
                                  "fact": ((e, r, o2, t2) if d == "subj" else (o2, r, e, t2))})
            if len(cands) > k_max:
                goldc = [c for c in cands if c["valid"]]
                rest = [c for c in cands if not c["valid"]]
                rng.shuffle(goldc); rng.shuffle(rest)
                goldc = goldc[: max(1, k_max // 2)] if goldc else []
                cands = goldc + rest[: max(0, k_max - len(goldc))]
            if not cands:
                continue
            anchor_frac = None
            if q["qtype"] in NEEDS_ANCHOR:
                a = parse_anchor(q["question"])
                if a is None:                                  # event anchor (entity)
                    if anchor_mode == "entity":
                        # S2: the anchor entity is the mentioned entity that is NOT the topic;
                        # take the retrieved fact connecting topic<->anchor via the topic's
                        # relation, else any fact linking them, else the top fact.
                        trel = groups[0][1]
                        conn = [facts[k] for k in top[j]
                                if mentioned(facts[k][0], qnorm) and mentioned(facts[k][2], qnorm)
                                and norm(facts[k][0]) != norm(facts[k][2])]
                        same_rel = [f for f in conn if f[1] == trel]
                        pick = (same_rel or conn or [facts[top[j][0]]])[0]
                        a = TimePoint.parse(pick[3])
                    else:                                      # 'topfact' = original behaviour
                        both = [facts[k] for k in top[j]
                                if mentioned(facts[k][0], qnorm) and mentioned(facts[k][2], qnorm)]
                        a = TimePoint.parse((both[0] if both else facts[top[j][0]])[3])
                anchor_frac = date_to_fracyear(a)
            ex = {"quid": q["quid"], "op": op, "question": q["question"], "s": topic,
                  "cands": cands, "anchor_frac": anchor_frac, "qtype": q["qtype"],
                  "answer_type": atype, "qlabel": q.get("qlabel", "?"),
                  "n_gold_in_set": sum(c["valid"] for c in cands)}
            if q["qtype"] in ("equal", "equal_multi"):
                ex["anchor2_frac"] = anchor_frac
            out.append(ex)
    return out


def disc_group_local(disc, examples, dev):
    """Per-example: within-(s,r)-group log-softmax of disc scores (trie-local)."""
    emb = build_text_embeddings(examples, device=dev)
    ds = CandidateSetDataset(examples, emb)
    ld = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)
    raw = []
    with torch.inference_mode():
        for b in ld:
            bb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            s = disc(bb).masked_fill(bb["mask"], NEG)
            for i in range(s.shape[0]):
                k = int((~bb["mask"][i]).sum()); raw.append(s[i, :k].cpu())
    out = []
    for e, sc in zip(examples, raw):
        loc = torch.zeros(len(e["cands"]))
        by = defaultdict(list)
        for i, c in enumerate(e["cands"]):
            by[c["group"]].append(i)
        for idxs in by.values():
            if len(idxs) > 1:
                loc[idxs] = F.log_softmax(sc[idxs], dim=0)
        out.append(loc.numpy())
    return out


@torch.inference_mode()
def llm_fact_scores(model, tok, ex, bs=32):
    prompt = PROMPT.format(question=ex["question"], topic=ex["s"])
    pids = tok(prompt, add_special_tokens=False).input_ids
    seqs, spans = [], []
    for c in ex["cands"]:
        path = wrap_path(temporal_path_to_string([c["fact"]]))
        ids = tok(prompt + path, add_special_tokens=False).input_ids
        seqs.append(ids); spans.append((len(pids), len(ids)))
    scores = []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]; sp = spans[i:i + bs]
        m = max(len(s) for s in chunk)
        inp = torch.full((len(chunk), m), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(chunk), m), dtype=torch.long)
        for jj, s in enumerate(chunk):
            inp[jj, :len(s)] = torch.tensor(s); att[jj, :len(s)] = 1
        inp = inp.to(model.device); att = att.to(model.device)
        logp = F.log_softmax(model(input_ids=inp, attention_mask=att).logits, dim=-1)
        for jj, (a, b) in enumerate(sp):
            tgt = inp[jj, 1:b]
            lp = logp[jj, :b - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            scores.append(lp[a - 1:].mean().item())
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_adapter", default=os.path.join(ROOT, "save_models/gcr-multitq-combined"))
    ap.add_argument("--llm_tok", default=os.path.join(ROOT, "data/multitq/tokenizers/Qwen2.5-7B-Instruct-ts"))
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--disc", default=os.path.join(ROOT, "save_models/discriminator/iv_all.pt"))
    ap.add_argument("--dense", default=DENSE)
    ap.add_argument("--llm", default="qwen", help="'none' = CPU dry-run")
    ap.add_argument("--topn", type=int, default=80)
    ap.add_argument("--groups_m", type=int, default=8)
    ap.add_argument("--per_group", type=int, default=15)
    ap.add_argument("--k_max", type=int, default=40)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0, 0.2, 0.5, 1, 2])
    ap.add_argument("--anchor_mode", default="entity", choices=["entity", "topfact"],
                    help="S2 event-anchor grounding: 'entity' (relation-matched) or 'topfact'")
    ap.add_argument("--rerank", default="", help="S1 answer-relevance reranker ckpt (optional)")
    ap.add_argument("--betas", nargs="+", type=float, default=[0.5, 1, 2],
                    help="rerank fusion weights to sweep (at the best alpha)")
    ap.add_argument("--rerank_gate", default="entity", choices=["none", "entity"],
                    help="'entity' = apply rerank only to entity-answer questions (it hurts "
                         "time answers by overriding the disc); 'none' = all questions")
    ap.add_argument("--dump_scores", default="", help="jsonl of per-instance (lp,disc,rr,valid,"
                    "meta) so future fusion sweeps are pure-CPU (see scripts/fuse_sweep.py)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--recall_only", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "e1_hybrid_system.md"))
    args = ap.parse_args()
    dev = args.device

    ex = build_examples(args.topn, args.groups_m, args.per_group, args.k_max,
                        args.n, device=dev, dense_path=args.dense, anchor_mode=args.anchor_mode)
    ksz = np.array([len(e["cands"]) for e in ex])
    got = np.array([e["n_gold_in_set"] > 0 for e in ex])
    print(f"questions: {len(ex)}  mean k={ksz.mean():.1f}  anchor_mode={args.anchor_mode}  "
          f"group-recall (gold (s,r) expanded): {100*got.mean():.1f}%")

    dstate = torch.load(args.disc, map_location=dev); da = dstate["args"]
    disc = TemporalSetScorer(edim=384, h=da["h"], layers=da["layers"], time_mode=da["time_mode"],
                             pointwise=da["pointwise"], use_leak=da.get("use_leak", False),
                             cond=da.get("cond", "text")).to(dev)
    disc.load_state_dict(dstate["model"]); disc.eval()
    dloc = disc_group_local(disc, ex, dev)

    # S1: optional answer-relevance reranker -> global log-softmax over each set
    rr = None
    if args.rerank:
        rstate = torch.load(args.rerank, map_location=dev); ra = rstate["args"]
        rmodel = TemporalSetScorer(edim=384, h=ra["h"], layers=ra["layers"], time_mode=ra["time_mode"],
                                   pointwise=ra["pointwise"], use_leak=ra.get("use_leak", False),
                                   cond=ra.get("cond", "text")).to(dev)
        rmodel.load_state_dict(rstate["model"]); rmodel.eval()
        emb = build_text_embeddings(ex, device=dev)
        ds = CandidateSetDataset(ex, emb)
        ld = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)
        rr = []
        with torch.inference_mode():
            for b in ld:
                bb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
                s = rmodel(bb).masked_fill(bb["mask"], NEG)
                for i in range(s.shape[0]):
                    k = int((~bb["mask"][i]).sum())
                    rr.append(F.log_softmax(s[i, :k], dim=0).cpu().numpy())

    if args.recall_only:
        by, h1 = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
        for i, e in enumerate(ex):
            by[e["qtype"]][0] += int(e["n_gold_in_set"] > 0); by[e["qtype"]][1] += 1
            r = rank_of_gold(dloc[i], e["cands"])
            h1[e["qtype"]][0] += int(r == 1); h1[e["qtype"]][1] += 1
        L = [f"# E1-hybrid recall check (topn={args.topn}, groups_m={args.groups_m}, n={len(ex)})",
             f"- group-recall (gold (s,r) found & expanded): **{100*got.mean():.1f}%**", ""]
        for qt in sorted(by):
            L.append(f"  - {qt}: recall {100*by[qt][0]/by[qt][1]:.1f}%  "
                     f"disc-only H@1 {100*h1[qt][0]/h1[qt][1]:.1f}%  (n={by[qt][1]})")
        open(args.out, "w").write("\n".join(L)); print("\n".join(L)); return

    if args.llm == "none":
        model = tok = None
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(args.llm_tok)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb,
                device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa")
        model.resize_token_embeddings(len(tok))
        model = PeftModel.from_pretrained(model, args.llm_adapter).eval()

    def newcell():
        return {"n": 0, "h1": 0, "h10": 0, "mrr": 0.0}
    metrics = {a: {"overall": newcell(), "qtype": defaultdict(newcell),
                   "cross": defaultdict(newcell)} for a in args.alphas}
    lp_all = []
    for i, e in enumerate(ex):
        lp = (np.zeros(len(e["cands"])) if model is None
              else np.asarray(llm_fact_scores(model, tok, e)))
        lp_all.append(lp)
        ck = f"{e['qlabel']}/{e['answer_type']}"
        for a in args.alphas:
            fused = lp + a * dloc[i]
            r = rank_of_gold(fused, e["cands"])
            h1 = int(r == 1); h10 = int(r is not None and r <= 10); mrr = (1.0 / r) if r else 0.0
            for cell in (metrics[a]["overall"], metrics[a]["qtype"][e["qtype"]],
                         metrics[a]["cross"][ck]):
                cell["n"] += 1; cell["h1"] += h1; cell["h10"] += h10; cell["mrr"] += mrr

    def fmt(c):
        n = max(c["n"], 1)
        return 100 * c["h1"] / n, 100 * c["h10"] / n, c["mrr"] / n

    def marg(a, pred):
        cs = [c for k, c in metrics[a]["cross"].items() if pred(*k.split("/"))]
        n = sum(c["n"] for c in cs)
        return 100 * sum(c["h1"] for c in cs) / max(n, 1)
    best = max(args.alphas, key=lambda a: fmt(metrics[a]["overall"])[0])

    def row(a):
        return (fmt(metrics[a]["overall"])[0], marg(a, lambda q, t: q == "Single"),
                marg(a, lambda q, t: q == "Multiple"), marg(a, lambda q, t: t == "entity"),
                marg(a, lambda q, t: t == "time"))
    r0, rb = row(0), row(best)
    L = ["# E1-hybrid — dense-retrieve (s,r) + KG-expand siblings + trie-local disc", "",
         f"n={len(ex)}, topn={args.topn}, groups_m={args.groups_m}, mean k={ksz.mean():.1f}, "
         f"group-recall={100*got.mean():.1f}%. LLM `{os.path.basename(args.llm_adapter)}`, "
         f"disc `{os.path.basename(args.disc)}`.", "",
         "| alpha | Hits@1 | Hits@10 | MRR |", "|--:|--:|--:|--:|"]
    for a in args.alphas:
        h1, h10, mrr = fmt(metrics[a]["overall"])
        L.append(f"| {a:g}{' **(best)**' if a==best else ''} | {h1:.1f}% | {h10:.1f}% | {mrr:.3f} |")
    L += ["", "## STANDARD PROTOCOL (TimeR4 Table-3 format)", "",
          "| method | Overall | Single | Multiple | Entity | Time |", "|---|--:|--:|--:|--:|--:|",
          f"| hybrid+vanilla | {r0[0]:.1f} | {r0[1]:.1f} | {r0[2]:.1f} | {r0[3]:.1f} | {r0[4]:.1f} |",
          f"| hybrid+disc (a={best:g}) | {rb[0]:.1f} | {rb[1]:.1f} | {rb[2]:.1f} | {rb[3]:.1f} | {rb[4]:.1f} |"]

    # optional per-instance score dump -> future fusion sweeps are pure CPU
    if args.dump_scores:
        with open(args.dump_scores, "w") as f:
            for i, e in enumerate(ex):
                f.write(json.dumps({
                    "quid": e["quid"], "qtype": e["qtype"], "answer_type": e["answer_type"],
                    "qlabel": e["qlabel"],
                    "lp": [float(x) for x in lp_all[i]],
                    "disc": [float(x) for x in dloc[i]],
                    "rr": ([float(x) for x in rr[i]] if rr is not None else None),
                    "valid": [int(c["valid"]) for c in e["cands"]]}) + "\n")
        print(f"wrote per-instance scores: {args.dump_scores}")

    # S1: reranker fusion sweep at the best alpha (score = lp + best*disc + beta*rerank),
    # optionally GATED to entity-answer questions (rerank hurts time answers).
    if rr is not None:
        def eval_beta(beta):
            m = {"overall": newcell(), "cross": defaultdict(newcell)}
            for i, e in enumerate(ex):
                b = 0.0 if (args.rerank_gate == "entity" and e["answer_type"] == "time") else beta
                fused = lp_all[i] + best * dloc[i] + b * rr[i]
                r = rank_of_gold(fused, e["cands"])
                h1 = int(r == 1); h10 = int(r is not None and r <= 10); mrr = (1.0 / r) if r else 0.0
                ck = f"{e['qlabel']}/{e['answer_type']}"
                for cell in (m["overall"], m["cross"][ck]):
                    cell["n"] += 1; cell["h1"] += h1; cell["h10"] += h10; cell["mrr"] += mrr
            def mg(pred):
                cs = [c for k, c in m["cross"].items() if pred(*k.split("/"))]
                n = sum(c["n"] for c in cs)
                return 100 * sum(c["h1"] for c in cs) / max(n, 1)
            o = fmt(m["overall"])
            return (o[0], mg(lambda q, t: q == "Single"), mg(lambda q, t: q == "Multiple"),
                    mg(lambda q, t: t == "entity"), mg(lambda q, t: t == "time"))
        L += ["", f"## + answer reranker (S1), fused at disc a={best:g}, gate={args.rerank_gate}", "",
              "| method | Overall | Single | Multiple | Entity | Time |", "|---|--:|--:|--:|--:|--:|"]
        bestb = max(args.betas, key=lambda bb: eval_beta(bb)[0])
        for beta in args.betas:
            rb2 = eval_beta(beta)
            tag = " **(best)**" if beta == bestb else ""
            L.append(f"| +rerank (b={beta:g}){tag} | {rb2[0]:.1f} | {rb2[1]:.1f} | {rb2[2]:.1f} | "
                     f"{rb2[3]:.1f} | {rb2[4]:.1f} |")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(L))
    json.dump({a: {"overall": metrics[a]["overall"],
                   "qtype": {k: dict(v) for k, v in metrics[a]["qtype"].items()}}
               for a in args.alphas}, open(args.out.replace(".md", ".json"), "w"),
              indent=2, default=float)
    print("\n".join(L)); print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
