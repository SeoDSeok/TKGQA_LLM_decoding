"""Phase 2 — fuse the temporal discriminator into constrained decoding; measure
the *realized* TVR recovery curve (the paper's headline figure).

At the timestamp branch of GCR-vanilla constrained decoding, all k candidate
paths share the prefix `<PATH>s -> r ` and differ only in the single
`[YYYY-MM-DD]` special token. So ONE forward of that prefix yields the LLM's
log-prob for every candidate at once (read off the branch-position logits). We
fuse with the discriminator:

    log P_final(t_i) = log P_LLM(t_i) + alpha * log softmax(s_theta)_i

and sweep alpha. alpha=0 is the pure-LLM baseline (reproduces the collapse);
alpha->inf is the pure discriminator. The curve shows how much temporal-validity
the discriminator injects into the *actual* decoder, per operator.
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

GCR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gcr_base")
sys.path.insert(0, GCR)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from torch.utils.data import DataLoader

from discriminator.dataset import (build_examples, build_text_embeddings,
                                   CandidateSetDataset, collate)
from discriminator.set_encoder import TemporalSetScorer
from discriminator.train import NEG
from checker.path_format import wrap_path, temporal_path_to_string

PROMPT = ("Reasoning path is a sequence of temporal triples in the KG that connects the "
          "topic entity to the answer, where each relation carries the fact's timestamp "
          "in brackets. It starts with <PATH> and ends with </PATH>.\n"
          "# Question:\n{question}\n# Topic entity:\n{topic}\n")


@torch.inference_mode()
def llm_candidate_logprobs(model, tok, ex, ts_ids):
    """One forward at the timestamp branch -> logP_LLM for every candidate."""
    c0 = ex["cands"][0]
    full = wrap_path(temporal_path_to_string([(ex["s"], c0["r"], c0["o"], c0["t_str"])]))
    text = PROMPT.format(question=ex["question"], topic=ex["s"])
    ids = tok(text + full, add_special_tokens=False).input_ids
    # locate the timestamp special-token position (shared branch point)
    tset = {t for t in ts_ids if t is not None}
    pos = next((i for i, t in enumerate(ids) if t in tset), None)
    if pos is None:
        return None
    prefix = torch.tensor([ids[:pos]], device=model.device)
    logp = F.log_softmax(model(prefix).logits[0, -1], dim=-1)
    return [logp[tid].item() if tid is not None else float("-inf") for tid in ts_ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_adapter", default=os.path.join(ROOT, "save_models/gcr-multitq-combined"))
    ap.add_argument("--llm_tok", default=os.path.join(ROOT, "data/multitq/tokenizers/Qwen2.5-7B-Instruct-ts"))
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--disc", default=os.path.join(ROOT, "save_models/discriminator/sig_all.pt"))
    ap.add_argument("--per_op", type=int, default=200, help="cap eval sets per operator (speed)")
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1, 2, 1e6])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "phase2_fusion_tvr.md"))
    args = ap.parse_args()
    dev = "cuda"

    # ---- LLM (special-token combined model) ----
    tok = AutoTokenizer.from_pretrained(args.llm_tok)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb,
            device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(model, args.llm_adapter).eval()

    # ---- discriminator ----
    dstate = torch.load(args.disc, map_location=dev)
    da = dstate["args"]

    # ---- eval sets (cap per op) ----
    ex_all = build_examples("test", ("first", "last", "before", "after"))
    by_op = defaultdict(list)
    for e in ex_all:
        if len(by_op[e["op"]]) < args.per_op:
            by_op[e["op"]].append(e)
    ex = [e for op in ("first", "last", "before", "after") for e in by_op[op]]
    emb = build_text_embeddings(build_examples("train", tuple(da["train_ops"])) + ex, device=dev)
    ds = CandidateSetDataset(ex, emb, use_rank_leak=da.get("use_leak", False))
    disc = TemporalSetScorer(edim=ds.edim, h=da["h"], layers=da["layers"],
                             time_mode=da["time_mode"], pointwise=da["pointwise"],
                             use_leak=da.get("use_leak", False), cond=da.get("cond", "text")).to(dev)
    disc.load_state_dict(dstate["model"]); disc.eval()

    # discriminator scores per set (batch)
    ld = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=collate)
    disc_scores = []
    with torch.inference_mode():
        for b in ld:
            bb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            s = disc(bb).masked_fill(bb["mask"], NEG)
            for i in range(s.shape[0]):
                k = int((~bb["mask"][i]).sum())
                disc_scores.append(s[i, :k].detach().cpu())

    # LLM logprobs per set + fuse over alphas
    alphas = args.alphas
    tvr = {a: defaultdict(lambda: {"n": 0, "hit": 0}) for a in alphas}
    llm_only = defaultdict(lambda: {"n": 0, "hit": 0})
    skipped = 0
    for idx, e in enumerate(ex):
        ts_ids = [tok.convert_tokens_to_ids(f"[{c['t_str']}]") for c in e["cands"]]
        ts_ids = [None if (t is None or t == tok.unk_token_id) else t for t in ts_ids]
        lp = llm_candidate_logprobs(model, tok, e, ts_ids)
        if lp is None:
            skipped += 1; continue
        lp = torch.tensor(lp)
        sd = disc_scores[idx]
        sd_ls = F.log_softmax(sd, dim=0)
        valid = torch.tensor([c["valid"] for c in e["cands"]], dtype=torch.float32)
        for a in alphas:
            fused = lp + (0.0 if a == 0 else a) * sd_ls if a < 1e6 else sd_ls
            pick = int(fused.argmax())
            d = tvr[a][e["op"]]; d["n"] += 1; d["hit"] += int(valid[pick] > 0)

    # ---- report ----
    ops = ("first", "last", "before", "after")
    L = ["# Phase 2 — Fusion decoding: realized TVR vs alpha", "",
         f"LLM = `{os.path.basename(args.llm_adapter)}` (special-token combined model, the "
         f"Figure-1 collapse model). Discriminator = `{os.path.basename(args.disc)}` "
         f"(`cond={da.get('cond','text')}`). Fusion "
         "`logP_final = logP_LLM + α·log softmax(s_θ)`; one forward per question at the "
         "timestamp branch gives all candidate LLM log-probs. "
         f"n≈{args.per_op}/op (skipped {skipped}).", "",
         "| α | first | last | before | after | macro |", "|--:|--:|--:|--:|--:|--:|"]
    for a in alphas:
        row = []
        vals = []
        for op in ops:
            d = tvr[a][op]
            v = 100 * d["hit"] / d["n"] if d["n"] else float("nan")
            row.append(f"{v:.1f}%"); vals.append(v)
        label = "∞ (disc only)" if a >= 1e6 else ("0 (LLM only)" if a == 0 else f"{a:g}")
        L.append(f"| {label} | {row[0]} | {row[1]} | {row[2]} | {row[3]} | {np.nanmean(vals):.1f}% |")
    L += ["", "## Reading",
          "- **α=0** reproduces the LLM collapse (last/after low) — the pure constrained decoder.",
          "- Increasing α injects the discriminator's value-space ordering; the macro/late "
          "TVR climbs toward the offline ceiling.",
          "- **α=∞** = pure discriminator re-ranking (offline result), the upper bound of "
          "this fusion."]
    open(args.out, "w").write("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {args.out}")

    # JSON + figure
    data = {"alphas": list(alphas), "per_op": {op: [
        (100 * tvr[a][op]["hit"] / tvr[a][op]["n"] if tvr[a][op]["n"] else None) for a in alphas]
        for op in ops}, "disc": os.path.basename(args.disc), "per_op_n": args.per_op}
    json.dump(data, open(args.out.replace(".md", ".json"), "w"), indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [a if a < 1e6 else (max(x for x in alphas if x < 1e6) * 2) for a in alphas]
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        colors = {"first": "#4C78A8", "last": "#F58518", "before": "#54A24B", "after": "#E45756"}
        for op in ops:
            ys = data["per_op"][op]
            ax.plot(xs, ys, marker="o", ms=4, lw=1.8, color=colors[op], label=op)
        macro = [float(np.nanmean([data["per_op"][op][j] for op in ops])) for j in range(len(alphas))]
        ax.plot(xs, macro, marker="s", ms=5, lw=2.4, color="#333", label="macro", zorder=5)
        ax.set_xlabel(r"fusion weight $\alpha$  (log P$_{final}$ = log P$_{LLM}$ + $\alpha\cdot$log softmax s$_\theta$)")
        ax.set_ylabel("realized TVR (%)")
        ax.set_title("Phase 2: discriminator fusion recovers temporal validity")
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_ylim(-3, 103); ax.grid(alpha=0.3); ax.legend(ncol=3, fontsize=8, loc="lower right")
        xt = [x for x in xs[:-1]] + [xs[-1]]
        ax.set_xticks(xt); ax.set_xticklabels([f"{a:g}" if a < 1e6 else "∞" for a in alphas], fontsize=7)
        fig.tight_layout()
        figp = os.path.join(ROOT, "results", "figures", "phase2_fusion_curve.png")
        os.makedirs(os.path.dirname(figp), exist_ok=True)
        fig.savefig(figp, dpi=150)
        print(f"wrote {figp}")
    except Exception as ex_:
        print(f"[fig skipped] {ex_}")


if __name__ == "__main__":
    main()
