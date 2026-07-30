"""E1-dense — system-level comparison with a DENSE-RETRIEVAL front-end (TimeR4-style).

Replaces the string-match linker with dense fact retrieval (build_fact_index.py):
every test question gets top-N candidate facts by cosine similarity, so coverage is
100% by construction — the standard full-test protocol applies with no unresolvable
bucket. The back-end is unchanged: GCR path log-likelihood + trie-local discriminator
fusion. This isolates how much of the absolute gap to agentic systems (TimeR4/RTQA)
was the front-end.

Candidate construction per retrieved fact (s,r,o,t):
  * entity answers: the fact entity NOT mentioned in the question (tie -> object);
  * time answers: t formatted at the question's granularity;
  * dedupe by (answer, t), keep highest retrieval rank.
Anchor: explicit date (parse_anchor) OR event anchor = timestamp of the highest-
ranked retrieved fact BOTH of whose entities appear in the question, falling back to
the top-1 retrieved fact.

--recall_only: skip the LLM; report retrieval oracle-recall@{10,20,40,N} + disc-only
Hits@1 (minutes on GPU) — run this FIRST to see the ceiling before the long decode.
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

from discriminator.dataset import build_text_embeddings, CandidateSetDataset, collate
from discriminator.set_encoder import TemporalSetScorer
from discriminator.train import NEG
from discriminator.encode_time import date_to_fracyear
from discriminator.phase2_system_e1 import (fine_op, answer_of, disc_all_scores,
                                            rank_of_gold, PROMPT)
from checker.timepoint import TimePoint
from checker.path_format import wrap_path, temporal_path_to_string
from scripts.build_before_after_signatures import parse_anchor, strict_norm
from scripts.multitq_linker import norm

QFILE = os.path.join(ROOT, "data", "multitq", "questions", "test.json")
IDX = os.path.join(ROOT, "data", "multitq", "index", "fact_dense.pkl")
NEEDS_ANCHOR = {"equal", "equal_multi", "before_after", "after_first", "before_last"}


def mentioned(ent, qnorm):
    n = norm(ent)
    return len(n) >= 3 and n in qnorm


def build_examples(topn=40, n_sample=0, seed=0, device="cuda", index_path=IDX):
    rng = np.random.RandomState(seed)
    qs = json.load(open(QFILE))
    if n_sample and n_sample < len(qs):
        idx = rng.choice(len(qs), size=n_sample, replace=False)
        qs = [qs[i] for i in sorted(idx)]

    with open(index_path, "rb") as f:
        pack = pickle.load(f)
    facts, emb = pack["facts"], pack["emb"].to(device).float()
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    qtexts = [q["question"] for q in qs]
    qemb = enc.encode(qtexts, batch_size=512, convert_to_tensor=True,
                      normalize_embeddings=True, show_progress_bar=True).to(device).float()

    out = []
    B = 256
    for i0 in range(0, len(qs), B):
        sims = qemb[i0:i0 + B] @ emb.T                      # (b, N)
        top = sims.topk(topn, dim=1).indices.cpu().numpy()
        for j, q in enumerate(qs[i0:i0 + B]):
            qnorm = " " + norm(q["question"]) + " "
            gold = {strict_norm(a) for a in q["answers"]}
            atype, tlevel = q["answer_type"], q["time_level"]
            op = fine_op(q["qtype"], q["question"])
            ranked = [facts[k] for k in top[j]]
            # anchor
            anchor_frac = None
            if q["qtype"] in NEEDS_ANCHOR:
                a = parse_anchor(q["question"])
                if a is None:
                    both = [f for f in ranked
                            if mentioned(f[0], qnorm) and mentioned(f[2], qnorm)]
                    src = both[0] if both else ranked[0]
                    a = TimePoint.parse(src[3])
                anchor_frac = date_to_fracyear(a)
            # candidates
            seen, cands, topic = set(), [], None
            for (s, r, o, t) in ranked:
                s_in, o_in = mentioned(s, qnorm), mentioned(o, qnorm)
                if topic is None and (s_in or o_in):
                    topic = s if s_in else o
                ans_ent = s if (o_in and not s_in) else o
                tp = TimePoint.parse(t)
                ans = answer_of(ans_ent, t, atype, tlevel)
                key = (strict_norm(ans), str(tp))
                if key in seen:
                    continue
                seen.add(key)
                cands.append({"o": ans_ent, "r": r, "t_str": str(tp),
                              "frac": date_to_fracyear(tp),
                              "valid": int(strict_norm(ans) in gold),
                              "answer": ans, "fact": (s, r, o, t)})
            ex = {"quid": q["quid"], "op": op, "question": q["question"],
                  "s": topic or ranked[0][0], "cands": cands,
                  "anchor_frac": anchor_frac, "qtype": q["qtype"],
                  "answer_type": atype, "qlabel": q.get("qlabel", "?"),
                  "n_gold_in_set": sum(c["valid"] for c in cands)}
            if q["qtype"] in ("equal", "equal_multi"):
                ex["anchor2_frac"] = anchor_frac
            out.append(ex)
    return out


@torch.inference_mode()
def llm_fact_scores(model, tok, ex, bs=32):
    """Length-normalized log-likelihood of each candidate's stored fact as a path."""
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
        for j, s in enumerate(chunk):
            inp[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
        inp = inp.to(model.device); att = att.to(model.device)
        logp = F.log_softmax(model(input_ids=inp, attention_mask=att).logits, dim=-1)
        for j, (a, b) in enumerate(sp):
            tgt = inp[j, 1:b]
            lp = logp[j, :b - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            scores.append(lp[a - 1:].mean().item())
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_adapter", default=os.path.join(ROOT, "save_models/gcr-multitq-combined"))
    ap.add_argument("--llm_tok", default=os.path.join(ROOT, "data/multitq/tokenizers/Qwen2.5-7B-Instruct-ts"))
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--disc", default=os.path.join(ROOT, "save_models/discriminator/iv_all.pt"))
    ap.add_argument("--index", default=IDX)
    ap.add_argument("--llm", default="qwen", help="'none' = CPU dry-run (skip the 8B LLM)")
    ap.add_argument("--topn", type=int, default=40)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0, 0.2, 0.5, 1, 2])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=0, help="proportional sample; 0 = full test")
    ap.add_argument("--recall_only", action="store_true",
                    help="no LLM: retrieval recall@k + disc-only Hits@1 (ceiling check)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "e1_dense_system.md"))
    ap.add_argument("--dump", default="")
    args = ap.parse_args()
    dev = args.device

    ex = build_examples(args.topn, args.n, device=dev, index_path=args.index)
    ksz = np.array([len(e["cands"]) for e in ex])
    got = np.array([e["n_gold_in_set"] > 0 for e in ex])
    print(f"questions: {len(ex)} (coverage 100% by construction)  mean k={ksz.mean():.1f}  "
          f"retrieval oracle-recall@{args.topn}: {100*got.mean():.1f}%")

    # discriminator scores (all qtypes via the unified interval scorer)
    dstate = torch.load(args.disc, map_location=dev); da = dstate["args"]
    disc = TemporalSetScorer(edim=384, h=da["h"], layers=da["layers"], time_mode=da["time_mode"],
                             pointwise=da["pointwise"], use_leak=da.get("use_leak", False),
                             cond=da.get("cond", "text")).to(dev)
    disc.load_state_dict(dstate["model"]); disc.eval()
    disc_scores, _ = disc_all_scores(disc, ex, dev)

    if args.recall_only:
        L = [f"# E1-dense recall check (topn={args.topn}, n={len(ex)})", "",
             f"- retrieval oracle-recall@{args.topn}: **{100*got.mean():.1f}%** (the ceiling)",
             "- per-qtype recall:"]
        by = defaultdict(lambda: [0, 0])
        h1 = defaultdict(lambda: [0, 0])
        for i, e in enumerate(ex):
            by[e["qtype"]][0] += int(e["n_gold_in_set"] > 0); by[e["qtype"]][1] += 1
            r = rank_of_gold(F.log_softmax(disc_scores[i], dim=0).numpy(), e["cands"])
            h1[e["qtype"]][0] += int(r == 1); h1[e["qtype"]][1] += 1
        for qt in sorted(by):
            L.append(f"  - {qt}: recall {100*by[qt][0]/by[qt][1]:.1f}%  "
                     f"disc-only H@1 {100*h1[qt][0]/h1[qt][1]:.1f}%  (n={by[qt][1]})")
        open(args.out, "w").write("\n".join(L)); print("\n".join(L)); return

    if args.llm == "none":                     # CPU dry-run: validate metric assembly
        model = tok = None
        print("[dry-run] LLM skipped; alpha=0 row is uninformative.")
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
    dump_rows = []
    for i, e in enumerate(ex):
        lp = (np.zeros(len(e["cands"])) if model is None
              else np.asarray(llm_fact_scores(model, tok, e)))
        ds_ls = F.log_softmax(disc_scores[i], dim=0).numpy()
        ck = f"{e['qlabel']}/{e['answer_type']}"
        rec = {"quid": e["quid"], "qtype": e["qtype"], "qlabel": e["qlabel"],
               "answer_type": e["answer_type"], "k": len(e["cands"]), "rank": {}}
        for a in args.alphas:
            fused = lp if a == 0 else lp + a * ds_ls
            r = rank_of_gold(fused, e["cands"])
            rec["rank"][str(a)] = r
            h1 = int(r == 1); h10 = int(r is not None and r <= 10); mrr = (1.0 / r) if r else 0.0
            for cell in (metrics[a]["overall"], metrics[a]["qtype"][e["qtype"]],
                         metrics[a]["cross"][ck]):
                cell["n"] += 1; cell["h1"] += h1; cell["h10"] += h10; cell["mrr"] += mrr
        dump_rows.append(rec)

    def fmt(c):
        n = max(c["n"], 1)
        return 100 * c["h1"] / n, 100 * c["h10"] / n, c["mrr"] / n

    def marginal(a, pred):
        cells = [c for k, c in metrics[a]["cross"].items() if pred(*k.split("/"))]
        n = sum(c["n"] for c in cells)
        return 100 * sum(c["h1"] for c in cells) / max(n, 1)

    best_a = max(args.alphas, key=lambda a: fmt(metrics[a]["overall"])[0])
    def std_row(a):
        return (fmt(metrics[a]["overall"])[0],
                marginal(a, lambda ql, at: ql == "Single"),
                marginal(a, lambda ql, at: ql == "Multiple"),
                marginal(a, lambda ql, at: at == "entity"),
                marginal(a, lambda ql, at: at == "time"))
    s0, sb = std_row(0), std_row(best_a)
    L = ["# E1-dense — dense-retrieval front-end + trie-local repair (standard protocol)", "",
         f"n = {len(ex)} test questions (100% coverage), topn = {args.topn}, mean k = "
         f"{ksz.mean():.1f}, retrieval oracle-recall = {100*got.mean():.1f}%. "
         f"LLM `{os.path.basename(args.llm_adapter)}`, disc `{os.path.basename(args.disc)}`.", "",
         "| alpha | Hits@1 | Hits@10 | MRR |", "|--:|--:|--:|--:|"]
    for a in args.alphas:
        h1, h10, mrr = fmt(metrics[a]["overall"])
        tag = " (vanilla)" if a == 0 else (" **(best)**" if a == best_a else "")
        L.append(f"| {a:g}{tag} | {h1:.1f}% | {h10:.1f}% | {mrr:.3f} |")
    L += ["", "## STANDARD PROTOCOL (TimeR4 Table-3 format)", "",
          "| method | Overall | Single | Multiple | Entity | Time |", "|---|--:|--:|--:|--:|--:|",
          f"| dense+vanilla | {s0[0]:.1f} | {s0[1]:.1f} | {s0[2]:.1f} | {s0[3]:.1f} | {s0[4]:.1f} |",
          f"| dense+disc (a={best_a:g}) | {sb[0]:.1f} | {sb[1]:.1f} | {sb[2]:.1f} | {sb[3]:.1f} | {sb[4]:.1f} |",
          "", f"## Per qtype (@ alpha={best_a:g})", "",
          "| qtype | n | vanilla H@1 | +disc H@1 |", "|---|--:|--:|--:|"]
    for qt in ("equal", "equal_multi", "first_last", "before_after", "after_first", "before_last"):
        c0, cb = metrics[0]["qtype"].get(qt), metrics[best_a]["qtype"].get(qt)
        if not c0 or not c0["n"]:
            continue
        L.append(f"| {qt} | {c0['n']} | {fmt(c0)[0]:.1f}% | {fmt(cb)[0]:.1f}% |")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(L))
    json.dump({a: {"overall": metrics[a]["overall"],
                   "qtype": {k: dict(v) for k, v in metrics[a]["qtype"].items()},
                   "cross": {k: dict(v) for k, v in metrics[a]["cross"].items()}}
               for a in args.alphas}, open(args.out.replace(".md", ".json"), "w"),
              indent=2, default=float)
    if args.dump:
        with open(args.dump, "w") as f:
            for r in dump_rows:
                f.write(json.dumps(r) + "\n")
    print("\n".join(L)); print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
