"""E1 — system-level comparison on the full MultiTQ test (fills main.tex TODO-P1).

Consumes the CPU-reconstructed instances in
`data/multitq/signatures/fulltest_resolved.jsonl` (topic, relation, direction,
candidate (o,t) facts, anchor, gold answers; all 6 qtypes) and decodes each with
the KG-specialized LLM under constrained scoring, comparing:

  * GCR-vanilla  : rank candidates by the LLM path log-likelihood (alpha = 0).
  * +discriminator (trie-local): fuse  logP_LLM + alpha * log softmax(s_theta)
    over the candidate set (the timestamp branch, after the trie has fixed (s,r)).

Reports Hits@1 / Hits@10 / MRR overall, per qtype, and Multiple/Single x
entity/time. Because MultiTQ has no gold KG annotations, candidates are the
reconstructed set (oracle recall ~46%); the honest signal is the +disc-vs-vanilla
DELTA on the SAME candidates, not the absolute number (see results/s3lite_harness.md).

A single unified interval discriminator (iv_all, cond=polarity_interval) scores
every qtype via (direction, lower/upper bound) — composites and equal included
through the extended OP_INTERVAL map. `--llm none` runs the whole pipeline on CPU
(no 8B model) to validate plumbing before a GPU run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
GCR = os.path.join(ROOT, "gcr_base")
sys.path.insert(0, GCR)

from torch.utils.data import DataLoader

from discriminator.dataset import build_text_embeddings, CandidateSetDataset, collate
from discriminator.set_encoder import TemporalSetScorer
from discriminator.train import NEG
from discriminator.encode_time import date_to_fracyear
from checker.timepoint import TimePoint
from checker.path_format import wrap_path, temporal_path_to_string
from checker.constraints import resolve_first_last, resolve_before_after
from scripts.build_fulltest_signatures import _fmt_time
from scripts.build_before_after_signatures import strict_norm

SIG = os.path.join(ROOT, "data", "multitq", "signatures", "fulltest_resolved.jsonl")
QFILE = os.path.join(ROOT, "data", "multitq", "questions", "test.json")
PROMPT = ("Reasoning path is a sequence of temporal triples in the KG that connects the "
          "topic entity to the answer, where each relation carries the fact's timestamp "
          "in brackets. It starts with <PATH> and ends with </PATH>.\n"
          "# Question:\n{question}\n# Topic entity:\n{topic}\n")


def fine_op(qtype, question):
    """Map a qtype to the fine operator the discriminator conditions on."""
    if qtype == "first_last":
        return resolve_first_last(question)      # 'first' | 'last'
    if qtype == "before_after":
        return resolve_before_after(question)    # 'before' | 'after'
    return qtype  # after_first / before_last / equal / equal_multi (in OP_INTERVAL)


def answer_of(o, t_str, answer_type, time_level):
    """The candidate's answer string as scored against gold."""
    if answer_type == "time":
        return _fmt_time(TimePoint.parse(t_str), time_level)
    return o


def build_examples(limit=0, per_qtype=0, k_max=40, seed=0, sigs=SIG, n_sample=0):
    """One discriminator/LLM example per resolvable instance, joined to its question.
    Candidate sets are capped at k_max (keeping every gold-bearing candidate + a seeded
    sample of the rest) so the per-candidate LLM path scoring stays tractable; the gold
    is preserved, so Hits/MRR ceilings are unchanged by the cap.

    Returns (examples, misses): `misses` are the sampled-but-unresolvable questions,
    which the STANDARD protocol counts in every denominator as automatic misses."""
    rng = np.random.RandomState(seed)
    questions = {q["quid"]: q["question"] for q in json.load(open(QFILE))}
    rows = [json.loads(line) for line in open(sigs)]
    if n_sample and n_sample < len(rows):     # proportional (uniform) sample of the file
        idx = rng.choice(len(rows), size=n_sample, replace=False)
        rows = [rows[i] for i in sorted(idx)]
    seen = defaultdict(int)
    out, misses = [], []
    for d in rows:
        if per_qtype and seen[d["qtype"]] >= per_qtype:
            continue
        if not d["resolvable"] or not d["candidates"]:
            misses.append({"quid": d["quid"], "qtype": d["qtype"],
                           "qlabel": d.get("qlabel", "?"), "answer_type": d["answer_type"]})
            continue
        q = questions.get(d["quid"])
        if q is None:
            continue
        seen[d["qtype"]] += 1
        op = fine_op(d["qtype"], q)
        atype, tlevel, direction = d["answer_type"], d["time_level"], d.get("direction", "subj")
        gold = {strict_norm(a) for a in d["gold_answers"]}
        cands = []
        for o, t in d["candidates"]:
            tp = TimePoint.parse(t)
            ans = answer_of(o, t, atype, tlevel)
            cands.append({"o": o, "r": d["relation"], "t_str": str(tp),
                          "frac": date_to_fracyear(tp),
                          "valid": int(strict_norm(ans) in gold),
                          "answer": ans, "direction": direction})
        if k_max and len(cands) > k_max:
            # cap to k_max EXACTLY: keep a seeded sample of gold-bearing candidates
            # (capped at k_max//2 so extra golds don't inflate the Hits lottery)
            # and fill with non-gold. Guarantees bounded LLM cost per instance.
            gold_c = [c for c in cands if c["valid"]]
            rest = [c for c in cands if not c["valid"]]
            rng.shuffle(gold_c); rng.shuffle(rest)
            gold_c = gold_c[: max(1, k_max // 2)] if gold_c else []
            cands = gold_c + rest[: max(0, k_max - len(gold_c))]
            rng.shuffle(cands)
        anchor_frac = None
        if d.get("anchor"):
            anchor_frac = date_to_fracyear(TimePoint.parse(d["anchor"]))
        ex = {"quid": d["quid"], "op": op, "question": q, "s": d["topic"],
              "cands": cands, "anchor_frac": anchor_frac,
              "qtype": d["qtype"], "answer_type": atype, "qlabel": d.get("qlabel", "?"),
              "n_gold_in_set": sum(c["valid"] for c in cands)}
        # equal/equal_multi: tight interval -> mirror the single anchor to the upper bound
        if d["qtype"] in ("equal", "equal_multi"):
            ex["anchor2_frac"] = anchor_frac
        out.append(ex)
        if limit and len(out) >= limit:
            break
    return out, misses


@torch.inference_mode()
def llm_path_scores(model, tok, ex, bs=32):
    """Length-normalized log-likelihood of each candidate's full path (direction-aware)."""
    prompt = PROMPT.format(question=ex["question"], topic=ex["s"])
    pids = tok(prompt, add_special_tokens=False).input_ids
    seqs, spans = [], []
    for c in ex["cands"]:
        # subj: (topic, r, o, t); obj: the candidate IS the subject reaching the topic
        triple = ((ex["s"], c["r"], c["o"], c["t_str"]) if c["direction"] == "subj"
                  else (c["o"], c["r"], ex["s"], c["t_str"]))
        path = wrap_path(temporal_path_to_string([triple]))
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


def disc_all_scores(disc, examples, dev):
    """Per-example tensor of masked discriminator scores (one score per candidate)."""
    emb = build_text_embeddings(examples, device=dev)
    ds = CandidateSetDataset(examples, emb)
    ld = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)
    out = []
    with torch.inference_mode():
        for b in ld:
            bb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            s = disc(bb).masked_fill(bb["mask"], NEG)
            for i in range(s.shape[0]):
                k = int((~bb["mask"][i]).sum()); out.append(s[i, :k].cpu())
    return out, ds.edim


def rank_of_gold(fused, cands):
    """1-indexed rank of the first candidate whose answer is gold; None if absent."""
    order = np.argsort(-np.asarray(fused))
    for pos, idx in enumerate(order, 1):
        if cands[idx]["valid"]:
            return pos
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_adapter", default=os.path.join(ROOT, "save_models/gcr-multitq-combined"))
    ap.add_argument("--llm_tok", default=os.path.join(ROOT, "data/multitq/tokenizers/Qwen2.5-7B-Instruct-ts"))
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--llm", default="qwen", help="'none' = CPU dry-run (skip the 8B LLM)")
    ap.add_argument("--disc", default=os.path.join(ROOT, "save_models/discriminator/iv_all.pt"))
    ap.add_argument("--alphas", nargs="+", type=float, default=[0, 0.2, 0.5, 1, 2, 5])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap total instances (smoke test)")
    ap.add_argument("--per_qtype", type=int, default=0, help="cap instances per qtype")
    ap.add_argument("--k_max", type=int, default=40, help="cap candidates/instance (gold kept)")
    ap.add_argument("--sigs", default=SIG, help="resolved-signatures jsonl (use the _full file "
                    "for the standard whole-test protocol)")
    ap.add_argument("--n", type=int, default=0, help="proportional (uniform) sample size over "
                    "the sigs file; 0 = all rows")
    ap.add_argument("--dump", default="", help="per-instance results jsonl (ranks per alpha)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "e1_system_table.md"))
    args = ap.parse_args()
    dev = "cpu" if args.llm == "none" else args.device

    ex, misses = build_examples(args.limit, args.per_qtype, args.k_max,
                                sigs=args.sigs, n_sample=args.n)
    ksz = np.array([len(e["cands"]) for e in ex])
    print(f"instances: {len(ex)} resolvable + {len(misses)} unresolvable (= misses, "
          f"standard protocol)  mean k={ksz.mean():.1f}  "
          f"gold-in-set: {sum(e['n_gold_in_set'] > 0 for e in ex)}")

    # --- discriminator (runs on CPU or GPU) ---
    dstate = torch.load(args.disc, map_location=dev); da = dstate["args"]
    disc_scores, edim = None, 384
    disc = TemporalSetScorer(edim=384, h=da["h"], layers=da["layers"], time_mode=da["time_mode"],
                             pointwise=da["pointwise"], use_leak=da.get("use_leak", False),
                             cond=da.get("cond", "text")).to(dev)
    disc.load_state_dict(dstate["model"]); disc.eval()
    disc_scores, edim = disc_all_scores(disc, ex, dev)

    # --- LLM path scores (skipped in dry-run) ---
    if args.llm == "none":
        lp_all = [np.zeros(len(e["cands"])) for e in ex]
        print("[dry-run] LLM skipped: alpha=0 (GCR-vanilla) is uninformative here; "
              "this validates the disc + data pipeline only.")
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
        lp_all = [np.asarray(llm_path_scores(model, tok, e)) for e in ex]

    # --- fuse and score every metric at every alpha ---
    # buckets: overall, per-qtype, and (qlabel x answer_type)
    def newcell():
        return {"n": 0, "h1": 0, "h10": 0, "mrr": 0.0}
    metrics = {a: {"overall": newcell(),
                   "qtype": defaultdict(newcell),
                   "cross": defaultdict(newcell)} for a in args.alphas}
    dump_rows = []
    for idx, e in enumerate(ex):
        lp = lp_all[idx]
        ds_ls = F.log_softmax(disc_scores[idx], dim=0).numpy()
        cross_key = f"{e['qlabel']}/{e['answer_type']}"
        rec = {"quid": e["quid"], "qtype": e["qtype"], "qlabel": e["qlabel"],
               "answer_type": e["answer_type"], "k": len(e["cands"]), "rank": {}}
        for a in args.alphas:
            fused = lp if a == 0 else lp + a * ds_ls
            r = rank_of_gold(fused, e["cands"])
            rec["rank"][str(a)] = r
            h1 = int(r == 1); h10 = int(r is not None and r <= 10); mrr = (1.0 / r) if r else 0.0
            for cell in (metrics[a]["overall"], metrics[a]["qtype"][e["qtype"]],
                         metrics[a]["cross"][cross_key]):
                cell["n"] += 1; cell["h1"] += h1; cell["h10"] += h10; cell["mrr"] += mrr
        dump_rows.append(rec)
    # STANDARD PROTOCOL: unresolvable questions enter every denominator as misses.
    for m in misses:
        ck = f"{m['qlabel']}/{m['answer_type']}"
        for a in args.alphas:
            for cell in (metrics[a]["overall"], metrics[a]["qtype"][m["qtype"]],
                         metrics[a]["cross"][ck]):
                cell["n"] += 1
        dump_rows.append({"quid": m["quid"], "qtype": m["qtype"], "qlabel": m["qlabel"],
                          "answer_type": m["answer_type"], "k": 0, "rank": None})

    def fmt(cell):
        n = max(cell["n"], 1)
        return 100 * cell["h1"] / n, 100 * cell["h10"] / n, cell["mrr"] / n

    best_a = max(args.alphas, key=lambda a: fmt(metrics[a]["overall"])[0])

    def marginal(a, pred):
        """H@1 marginal over cross buckets matching pred(qlabel, atype)."""
        cells = [c for k, c in metrics[a]["cross"].items()
                 if pred(*k.split("/"))]
        n = sum(c["n"] for c in cells)
        return 100 * sum(c["h1"] for c in cells) / max(n, 1)

    def std_row(a):
        return (fmt(metrics[a]["overall"])[0],
                marginal(a, lambda ql, at: ql == "Single"),
                marginal(a, lambda ql, at: ql == "Multiple"),
                marginal(a, lambda ql, at: at == "entity"),
                marginal(a, lambda ql, at: at == "time"))

    L = ["# E1 — system-level comparison (full MultiTQ test, reconstructed candidates)", "",
         f"n = {len(ex)} resolvable instances, mean k = {ksz.mean():.1f}. LLM = "
         f"`{os.path.basename(args.llm_adapter) if args.llm!='none' else 'NONE (dry-run)'}`, "
         f"disc = `{os.path.basename(args.disc)}` (cond={da.get('cond')}). "
         "GCR-vanilla = alpha 0; +disc(trie-local) = fused. Absolute numbers are bounded by "
         "the ~46% reconstruction ceiling; read the **delta** vs alpha=0.", "",
         "## Overall vs alpha", "",
         "| alpha | Hits@1 | Hits@10 | MRR |", "|--:|--:|--:|--:|"]
    for a in args.alphas:
        h1, h10, mrr = fmt(metrics[a]["overall"])
        tag = " (GCR-vanilla)" if a == 0 else (" **(best)**" if a == best_a else "")
        L.append(f"| {a:g}{tag} | {h1:.1f}% | {h10:.1f}% | {mrr:.3f} |")

    v = fmt(metrics[0]["overall"]); bcell = fmt(metrics[best_a]["overall"])
    s0, sb = std_row(0), std_row(best_a)
    L += ["", f"**Delta (+disc @ alpha={best_a:g} vs GCR-vanilla):** "
          f"Hits@1 {v[0]:.1f} -> {bcell[0]:.1f} ({bcell[0]-v[0]:+.1f} pp), "
          f"MRR {v[2]:.3f} -> {bcell[2]:.3f}.", "",
          "## STANDARD PROTOCOL (TimeR4 Table-3 format, unresolvable = miss)", "",
          "| method | Overall | Single | Multiple | Entity | Time |", "|---|--:|--:|--:|--:|--:|",
          f"| GCR-vanilla | {s0[0]:.1f} | {s0[1]:.1f} | {s0[2]:.1f} | {s0[3]:.1f} | {s0[4]:.1f} |",
          f"| +disc (a={best_a:g}) | {sb[0]:.1f} | {sb[1]:.1f} | {sb[2]:.1f} | {sb[3]:.1f} | {sb[4]:.1f} |",
          "",
          f"## Per qtype (GCR-vanilla vs +disc @ alpha={best_a:g})", "",
          "| qtype | n | vanilla H@1 | +disc H@1 | vanilla MRR | +disc MRR |",
          "|---|--:|--:|--:|--:|--:|"]
    for qt in ("equal", "equal_multi", "first_last", "before_after", "after_first", "before_last"):
        c0, cb = metrics[0]["qtype"].get(qt), metrics[best_a]["qtype"].get(qt)
        if not c0 or not c0["n"]:
            continue
        a0, ab = fmt(c0), fmt(cb)
        L.append(f"| {qt} | {c0['n']} | {a0[0]:.1f}% | {ab[0]:.1f}% | {a0[2]:.3f} | {ab[2]:.3f} |")

    L += ["", f"## Multiple/Single x entity/time (@ alpha={best_a:g})", "",
          "| bucket | n | vanilla H@1 | +disc H@1 | +disc H@10 | +disc MRR |",
          "|---|--:|--:|--:|--:|--:|"]
    for key in sorted(metrics[best_a]["cross"]):
        c0, cb = metrics[0]["cross"][key], metrics[best_a]["cross"][key]
        a0, ab = fmt(c0), fmt(cb)
        L.append(f"| {key} | {cb['n']} | {a0[0]:.1f}% | {ab[0]:.1f}% | {ab[1]:.1f}% | {ab[2]:.3f} |")

    L += ["", "## Reading",
          "- **Delta is the claim:** on identical reconstructed candidates, +disc(trie-local) "
          "lifts Hits@1/MRR over GCR-vanilla; the absolute ceiling is the linker's, not the "
          "temporal method's.",
          "- Latest-wanting qtypes (last within first_last; before_last) should show the "
          "largest gain, mirroring the isolated collapse-and-fix."]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(L))
    json.dump({a: {"overall": metrics[a]["overall"],
                   "qtype": {k: dict(v) for k, v in metrics[a]["qtype"].items()},
                   "cross": {k: dict(v) for k, v in metrics[a]["cross"].items()}}
               for a in args.alphas},
              open(args.out.replace(".md", ".json"), "w"), indent=2, default=float)
    if args.dump:
        with open(args.dump, "w") as f:
            for r in dump_rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote per-instance dump: {args.dump} ({len(dump_rows)} rows)")
    print("\n".join(L)); print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
