"""Reliable recoverability analysis: score each candidate path by model likelihood.

The beam approach was buggy (constrained beams share a prefix and collapse to
duplicates, so beam cannot enumerate the k candidate timestamps). Since the
candidate set is small and known, we instead score EVERY candidate path
`<PATH>s -> r [t] -> o</PATH>` by the model's length-normalized log-likelihood
of the path tokens given the prompt, and rank them.

This gives, bug-free:
  * top-1 TVR (argmax likelihood) — reproduces the greedy pick-min bias;
  * **gold rank distribution** — where the correct timestamp sits in the model's
    ranking. The correct answer is in the candidate set by construction (recall
    = 100%), so gold's rank tells us whether a temporal reranker / discriminator
    can recover it, and how hard the fix is (rank-2 with small margin = easy).
  * top-1 vs top-2 recoverability = the headline for "reranking suffices".
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

GCR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gcr_base")
sys.path.insert(0, GCR)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from checker.path_format import wrap_path, temporal_path_to_string
from checker.timepoint import TimePoint

PROMPT = ("Reasoning path is a sequence of temporal triples in the KG that connects the "
          "topic entity to the answer, where each relation carries the fact's timestamp "
          "in brackets. It starts with <PATH> and ends with </PATH>.\n"
          "# Question:\n{question}\n# Topic entity:\n{topic}\n")


@torch.inference_mode()
def score_paths(model, tok, prompt, paths, ts_token_ids):
    """Return the model's log-prob of the *timestamp token* (the decision point).

    All candidate paths share an identical prefix up to the timestamp; the only
    differing token is the single `[YYYY-MM-DD]` special token. Scoring that
    token's conditional log-prob is the pure temporal decision and matches what
    greedy constrained decoding argmaxes over.
    """
    scores = []
    for p, ts_id in zip(paths, ts_token_ids):
        full = tok(prompt + p, add_special_tokens=False).input_ids
        ids = torch.tensor([full], device=model.device)
        logits = model(ids).logits[0]
        logp = F.log_softmax(logits[:-1], dim=-1)
        tgt = ids[0, 1:]
        # locate the timestamp token position in the target sequence
        pos = [i for i, t in enumerate(tgt.tolist()) if t == ts_id]
        if not pos:
            scores.append(float("-inf"))
            continue
        scores.append(logp[pos[0], ts_id].item())
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=os.path.join(ROOT, "save_models/gcr-multitq-qwen7b"))
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--sigs", default=os.path.join(ROOT, "data/multitq/signatures/first_last_time_test.jsonl"))
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "recoverability_report.md"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapter)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb,
            device_map={"": 0}, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(model, args.adapter).eval()

    sigs = [json.loads(l) for l in open(args.sigs)]
    traps = [s for s in sigs if s["n_candidates"] >= 2][:args.limit]

    from collections import defaultdict, Counter
    per_op = defaultdict(lambda: {"n": 0, "top1": 0, "top2": 0, "rank_sum": 0, "margin_sum": 0.0})
    rank_hist = Counter()
    for sig in traps:
        cand = sorted({TimePoint.parse(t) for t in sig["candidate_timestamps"]})
        gold = cand[0] if sig["op"] == "first" else cand[-1]
        paths = [wrap_path(temporal_path_to_string([(sig["s"], sig["r"], sig["o"], str(t))])) for t in cand]
        ts_ids = [tok.convert_tokens_to_ids(f"[{t}]") for t in cand]
        scores = score_paths(model, tok, PROMPT.format(question=sig["question"], topic=sig["s"]), paths, ts_ids)
        order = sorted(range(len(cand)), key=lambda i: -scores[i])   # best first
        gold_idx = cand.index(gold)
        gold_rank = order.index(gold_idx) + 1                        # 1 = model's top pick
        op = sig["op"]; d = per_op[op]
        d["n"] += 1
        d["top1"] += int(gold_rank == 1)
        d["top2"] += int(gold_rank <= 2)
        d["rank_sum"] += gold_rank
        # margin between model's top pick and gold (log-prob units)
        d["margin_sum"] += scores[order[0]] - scores[gold_idx]
        rank_hist[min(gold_rank, 5)] += 1

    lines = ["# Recoverability — can a reranker fix the temporal error?", "",
             f"Model: QLoRA `{os.path.basename(args.adapter)}`. Candidates ranked by the "
             "model's log-prob of the **timestamp token** (the decision point — the only "
             "token that differs across candidates; matches greedy constrained decoding). "
             "Gold is always in the candidate set (recall = 100% by construction).", "",
             f"Evaluated {sum(d['n'] for d in per_op.values())} multi-timestamp questions.", "",
             "| operator | n | top-1 TVR (model pick) | gold-in-top-2 | mean gold rank | mean top1−gold margin |",
             "|---|--:|--:|--:|--:|--:|"]
    tot = defaultdict(float)
    for op in ("first", "last"):
        d = per_op[op]
        if not d["n"]:
            continue
        lines.append(f"| {op} | {d['n']} | {100*d['top1']/d['n']:.1f}% | {100*d['top2']/d['n']:.1f}% | "
                     f"{d['rank_sum']/d['n']:.2f} | {d['margin_sum']/d['n']:.3f} |")
        for k in ("n", "top1", "top2", "rank_sum", "margin_sum"):
            tot[k] += d[k]
    if tot["n"]:
        lines.append(f"| **all** | {int(tot['n'])} | **{100*tot['top1']/tot['n']:.1f}%** | "
                     f"**{100*tot['top2']/tot['n']:.1f}%** | {tot['rank_sum']/tot['n']:.2f} | "
                     f"{tot['margin_sum']/tot['n']:.3f} |")
    lines += ["", "**Gold-rank histogram (1 = model's top choice):**", ""]
    for r in sorted(rank_hist):
        lab = f"{r}" if r < 5 else "5+"
        lines.append(f"- rank {lab}: {rank_hist[r]}")
    lines += ["", "## Interpretation",
              "- **top-1 TVR** = the model's own pick is correct (reproduces greedy).",
              "- **gold-in-top-2** = a reranker choosing among the model's top-2 would "
              "recover the correct timestamp. High gap between top-2 and top-1 ⇒ the "
              "error is a *ranking* problem a temporal discriminator can fix.",
              "- **mean gold rank / margin** quantify how far off, and how confidently, "
              "the time-blind model mis-ranks the correct timestamp."]
    open(args.out, "w").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
