"""D3-full: last-TVR for the RAW-tokenization model (multi-token timestamps).

The raw model was trained on the same balanced data but with the base tokenizer,
so `[YYYY-MM-DD]` is many subword tokens (carrying pretrained digit/ordinal
structure) instead of one special token. We rank candidates by the model's
length-normalized log-prob of the *timestamp token span* (located via offset
mapping), then check first/last top-1 correctness — the same decision measured
for the special-token model (D2). If raw recovers `last`, tokenization is the
cause (T2 via representation); if raw also collapses, the failure is deeper.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

GCR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gcr_base")
sys.path.insert(0, GCR)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from checker.path_format import wrap_path, temporal_path_to_string
from checker.timepoint import TimePoint

SIG = os.path.join(ROOT, "data", "multitq", "signatures")
PROMPT = ("Reasoning path is a sequence of temporal triples in the KG that connects the "
          "topic entity to the answer, where each relation carries the fact's timestamp "
          "in brackets. It starts with <PATH> and ends with </PATH>.\n"
          "# Question:\n{question}\n# Topic entity:\n{topic}\n")


@torch.inference_mode()
def ts_span_logprob(model, tok, prompt, path, ts_str):
    """Length-normalized log-prob of the `[ts]` token span, located by char offsets."""
    text = prompt + path
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc.input_ids
    offs = enc.offset_mapping
    marker = f"[{ts_str}]"
    ci = text.rfind(marker)
    if ci < 0:
        return None
    lo, hi = ci, ci + len(marker)
    span = [k for k, (a, b) in enumerate(offs) if a < hi and b > lo and b > a]
    if not span:
        return None
    inp = torch.tensor([ids], device=model.device)
    logp = F.log_softmax(model(inp).logits[0][:-1], dim=-1)
    tgt = inp[0, 1:]
    lps = []
    for k in span:
        if k >= 1:
            lps.append(logp[k - 1, tgt[k - 1]].item())
    return float(np.mean(lps)) if lps else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=os.path.join(ROOT, "save_models/gcr-multitq-balanced-raw"))
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "d3full_raw_tvr.md"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)  # base tokenizer, no special ts tokens
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb,
            device_map={"": 0}, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    # raw training resized the padded base embedding (152064) down to the base
    # tokenizer size (151665); match that before loading the adapter.
    model.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(model, args.adapter).eval()

    sigs = [json.loads(l) for l in open(os.path.join(SIG, "first_last_time_test.jsonl"))]
    traps = [s for s in sigs if s["n_candidates"] >= 2][:args.limit]

    from collections import defaultdict
    per_op = defaultdict(lambda: {"n": 0, "top1": 0, "earliest": 0})
    for sig in traps:
        cand = sorted({TimePoint.parse(t) for t in sig["candidate_timestamps"]})
        gold = cand[0] if sig["op"] == "first" else cand[-1]
        prompt = PROMPT.format(question=sig["question"], topic=sig["s"])
        scores = []
        for t in cand:
            p = wrap_path(temporal_path_to_string([(sig["s"], sig["r"], sig["o"], str(t))]))
            scores.append(ts_span_logprob(model, tok, prompt, p, str(t)))
        if any(s is None for s in scores):
            continue
        pick = cand[int(np.argmax(scores))]
        d = per_op[sig["op"]]; d["n"] += 1
        d["top1"] += int(pick == gold)
        d["earliest"] += int(pick == cand[0])

    L = ["# D3-full — RAW tokenization model: first/last TVR", "",
         f"Model: `{os.path.basename(args.adapter)}` (base tokenizer, multi-token timestamps, "
         "LoRA only, pretrained digit embeddings frozen). Candidate ranking by "
         "length-normalized timestamp-span log-prob.", "",
         "| operator | n | top-1 TVR | picks earliest | (special-token D2) |",
         "|---|--:|--:|--:|--:|"]
    ref = {"first": 99.2, "last": 1.9}
    for op in ("first", "last"):
        d = per_op[op]
        if not d["n"]:
            continue
        L.append(f"| {op} | {d['n']} | {100*d['top1']/d['n']:.1f}% | "
                 f"{100*d['earliest']/d['n']:.1f}% | {ref[op]:.1f}% |")
    L += ["", "## Interpretation",
          "- **`last` TVR recovers vs D2 (1.9%)** ⇒ special-token registration was the "
          "cause; tokenization is a sub-contribution and the discriminator re-aims at "
          "implicit/composite/zero-shot generalization (raw model becomes a baseline).",
          "- **`last` still ~0%** ⇒ even pretrained digit tokenization can't do the "
          "comparison here → the failure is deeper than tokenization; the external "
          "functional-time-encoding discriminator is the needed fix (strongest T2 story)."]
    open(args.out, "w").write("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
