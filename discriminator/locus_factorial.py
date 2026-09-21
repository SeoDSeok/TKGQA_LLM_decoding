"""N1b - factorial injection-locus ablation: scoring context x normalization scope.

The two-arm N1 experiment (phase2_trielocal.py) changes BOTH the discriminator's
scoring context (whole mixed candidate set vs each (r,o) sibling group as its own
set) AND the normalization scope (global log-softmax vs per-group log-softmax), so
it cannot attribute the effect to either factor alone. This script runs the 2x2
(three realizable cells) to separate them:

  A) group score  + group norm   = trie-local, the deployed form
  B) group score  + global norm  = same raw scores as A, one global denominator
  C) mixed score  + global norm  = the global control of N1

  A vs B  isolates NORMALIZATION SCOPE (scoring context held fixed)
  B vs C  isolates SCORING CONTEXT   (normalization held fixed)

Same LLM path scores, same candidate set, same discriminator parameters throughout.
Metrics mirror N1: answer accuracy (picked fact == gold) and (r,o) branch fidelity.
"""
from __future__ import annotations

import argparse, json, os, sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gcr_base"))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from torch.utils.data import DataLoader

from discriminator.dataset import build_text_embeddings, CandidateSetDataset, collate
from discriminator.set_encoder import TemporalSetScorer
from discriminator.train import NEG
from discriminator.phase2_khop import load_s2facts, build_khop, llm_path_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_adapter", default=os.path.join(ROOT, "save_models/gcr-multitq-combined"))
    ap.add_argument("--llm_tok", default=os.path.join(ROOT, "data/multitq/tokenizers/Qwen2.5-7B-Instruct-ts"))
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--disc", default=os.path.join(ROOT, "save_models/discriminator/sig_all.pt"))
    ap.add_argument("--k_max", type=int, default=40)
    ap.add_argument("--per_op", type=int, default=150)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0, 0.2, 0.5, 1, 2, 5])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "locus_factorial.md"))
    args = ap.parse_args()
    dev = "cuda"; rng = np.random.RandomState(args.seed)

    s2 = load_s2facts()
    ex = build_khop("test", s2, args.k_max, args.per_op, rng)
    print(f"k-hop sets: {len(ex)}  mean k={np.mean([len(e['cands']) for e in ex]):.1f}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.llm_tok)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb,
            device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(model, args.llm_adapter).eval()

    dstate = torch.load(args.disc, map_location=dev); da = dstate["args"]
    disc = TemporalSetScorer(edim=384, h=da["h"], layers=da["layers"], time_mode=da["time_mode"],
                             pointwise=da["pointwise"], use_leak=da.get("use_leak", False),
                             cond=da.get("cond", "text")).to(dev)
    disc.load_state_dict(dstate["model"]); disc.eval()

    def disc_scores_for(examples):
        e2 = build_text_embeddings(examples, device=dev)
        d2 = CandidateSetDataset(examples, e2)
        l2 = DataLoader(d2, batch_size=64, shuffle=False, collate_fn=collate)
        out = []
        with torch.inference_mode():
            for b in l2:
                bb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
                s = disc(bb).masked_fill(bb["mask"], NEG)
                for i in range(s.shape[0]):
                    k = int((~bb["mask"][i]).sum()); out.append(s[i, :k].cpu())
        return out

    # ---- raw scores under the two scoring contexts -------------------------
    mixed_raw = disc_scores_for(ex)                      # context: whole set

    group_examples, group_map = [], []
    for pi, e in enumerate(ex):
        by = defaultdict(list)
        for i, c in enumerate(e["cands"]):
            by[(c["r"], c["o"])].append(i)
        for _key, idxs in by.items():
            if len(idxs) >= 2:
                group_examples.append({"quid": e["quid"], "op": e["op"], "question": e["question"],
                                       "s": e["s"], "anchor_frac": e.get("anchor_frac"),
                                       "cands": [e["cands"][i] for i in idxs]})
                group_map.append((pi, idxs))
    grp_raw_scores = disc_scores_for(group_examples) if group_examples else []

    # per-parent RAW scores under the group scoring context.
    # Singleton groups are never scored by the discriminator in the deployed form;
    # they contribute 0 to the local arm. For the global-norm arm we must place them
    # on a common scale, so we use the same 0 (their group log-softmax value), which
    # is exactly what arm A would use. This keeps A and B differing ONLY in denominator.
    grp_raw = [torch.zeros(len(e["cands"])) for e in ex]
    grp_seen = [torch.zeros(len(e["cands"]), dtype=torch.bool) for e in ex]
    for gs, (pi, idxs) in zip(grp_raw_scores, group_map):
        for pos, ci in enumerate(idxs):
            grp_raw[pi][ci] = gs[pos]; grp_seen[pi][ci] = True

    loci = ["A_group_group", "B_group_global", "C_mixed_global"]
    alphas = args.alphas
    M = {loc: {a: defaultdict(lambda: {"n": 0, "hit": 0, "struct": 0}) for a in alphas} for loc in loci}

    for idx, e in enumerate(ex):
        lp = torch.tensor(llm_path_scores(model, tok, e))
        # A: group scoring context, per-group normalization
        gA = torch.zeros(len(e["cands"]))
        by = defaultdict(list)
        for i, c in enumerate(e["cands"]):
            by[(c["r"], c["o"])].append(i)
        for idxs in by.values():
            if len(idxs) >= 2:
                gA[idxs] = F.log_softmax(grp_raw[idx][idxs], dim=0)
        # B: SAME raw scores as A, one global denominator
        gB = F.log_softmax(grp_raw[idx], dim=0)
        # C: mixed scoring context, one global denominator
        gC = F.log_softmax(mixed_raw[idx], dim=0)

        gr, go, gt = e["gold"]
        for a in alphas:
            for name, g in (("A_group_group", gA), ("B_group_global", gB), ("C_mixed_global", gC)):
                fused = lp if a == 0 else lp + a * g
                c = e["cands"][int(fused.argmax())]
                d = M[name][a][e["op"]]; d["n"] += 1
                d["hit"] += int(c["valid"] > 0)
                d["struct"] += int(c["r"] == gr and c["o"] == go)
        if (idx + 1) % 25 == 0:
            print(f"  {idx+1}/{len(ex)}", flush=True)

    ops = ("first", "last")
    def acc(loc, a, key="hit"):
        v = [100 * M[loc][a][o][key] / M[loc][a][o]["n"] for o in ops if M[loc][a][o]["n"]]
        return float(np.mean(v)) if v else float("nan")

    L = ["# N1b - factorial injection locus: scoring context x normalization scope", "",
         f"k-hop full-subgraph, n={args.per_op}/op, mean k="
         f"{np.mean([len(e['cands']) for e in ex]):.1f}. Same LLM path scores, same candidate "
         f"set, same discriminator parameters in every arm. Macro over first/last.", "",
         "| alpha | A hit | A BFR | B hit | B BFR | C hit | C BFR |",
         "|--:|--:|--:|--:|--:|--:|--:|"]
    for a in alphas:
        L.append("| %g | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |" % (
            a, acc("A_group_group", a), acc("A_group_group", a, "struct"),
            acc("B_group_global", a), acc("B_group_global", a, "struct"),
            acc("C_mixed_global", a), acc("C_mixed_global", a, "struct")))
    L += ["", "A = group score + group norm (trie-local, deployed)",
          "B = group score + GLOBAL norm  (isolates normalization scope vs A)",
          "C = mixed score + global norm  (the N1 global control)", "",
          "A vs B -> effect of normalization scope with scoring context held fixed.",
          "B vs C -> effect of scoring context with normalization held fixed."]
    open(args.out, "w").write("\n".join(L))
    json.dump({"alphas": list(alphas),
               "M": {loc: {str(a): {o: dict(M[loc][a][o]) for o in ops} for a in alphas} for loc in loci}},
              open(args.out.replace(".md", ".json"), "w"), indent=2, default=float)
    print("\n".join(L)); print("wrote", args.out)


if __name__ == "__main__":
    main()
