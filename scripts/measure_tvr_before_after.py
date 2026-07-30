"""Realized before/after TVR under GCR-vanilla constrained decoding.

For each answer-verified before/after signature, the time-blind trie = every
(o', t') fact of (topic, relation) rendered as <PATH>s -> r [t'] -> o'</PATH>.
The combined model decodes one path; TVR = the picked timestamp satisfies
before/after(anchor). Compared against the per-question opportunity (fraction of
candidate facts on the wrong side) and an early-bias diagnostic.
"""
import argparse
import json
import os
import pickle
import sys

import torch

GCR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gcr_base")
sys.path.insert(0, GCR)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from src.trie import MarisaTrie
from src.graph_constrained_decoding import GraphConstrainedDecoding
from checker.path_format import wrap_path, temporal_path_to_string, extract_timestamps
from checker.timepoint import TimePoint

IDX = os.path.join(ROOT, "data", "multitq", "index", "kg_index.pkl")
PROMPT = ("Reasoning path is a sequence of temporal triples in the KG that connects the "
          "topic entity to the answer, where each relation carries the fact's timestamp "
          "in brackets. It starts with <PATH> and ends with </PATH>.\n"
          "# Question:\n{question}\n# Topic entity:\n{topic}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=os.path.join(ROOT, "save_models/gcr-multitq-combined"))
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--sigs", default=os.path.join(ROOT, "data/multitq/signatures/before_after_test.jsonl"))
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--max_cand", type=int, default=200, help="cap trie size per question")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "before_after_tvr.md"))
    args = ap.parse_args()

    idx = pickle.load(open(IDX, "rb"))
    sr2ots = idx["sr2ots"]
    tok = AutoTokenizer.from_pretrained(args.adapter)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb,
            device_map={"": 0}, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(model, args.adapter).eval()

    sigs = [json.loads(l) for l in open(args.sigs)][:args.limit]
    from collections import defaultdict
    per_op = defaultdict(lambda: {"n": 0, "valid": 0, "opp_sum": 0.0, "earliest": 0})
    n_parse = 0
    examples = []
    for si, sig in enumerate(sigs):
        facts = sr2ots.get((sig["s"], sig["r"]), [])
        if len(facts) < 2:
            continue
        facts = facts[:args.max_cand]
        anchor = TimePoint.parse(sig["anchor"])
        direction = sig["op"]
        paths = [wrap_path(temporal_path_to_string([(sig["s"], sig["r"], o, t)])) for t, o in facts]
        trie = MarisaTrie(tok(paths, add_special_tokens=False).input_ids, max_token_id=len(tok) + 1)
        prompt = PROMPT.format(question=sig["question"], topic=sig["s"])
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        gcr = GraphConstrainedDecoding(tok, trie, None, None, True)
        with torch.inference_mode():
            out = model.generate(input_ids=enc.input_ids.to(model.device),
                                 attention_mask=enc.attention_mask.to(model.device),
                                 max_new_tokens=64, do_sample=False, num_beams=1,
                                 prefix_allowed_tokens_fn=gcr.allowed_tokens_fn,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=False)
        picked = extract_timestamps(gen)
        d = per_op[direction]
        d["n"] += 1
        d["opp_sum"] += sig["opportunity"]
        all_tps = sorted({TimePoint.parse(t) for t, _ in facts})
        if picked:
            n_parse += 1
            p = picked[0]
            ok = p.strictly_before(anchor) if direction == "before" else p.strictly_after(anchor)
            d["valid"] += int(ok)
            if p == all_tps[0]:
                d["earliest"] += 1
            if si < 6:
                examples.append((direction, str(anchor), str(p), ok, sig["question"]))

    lines = ["# Realized before/after TVR (combined model, GCR-vanilla decoding)", "",
             f"Model: QLoRA `{os.path.basename(args.adapter)}` (first/last + before/after). "
             "Trie = all (topic, relation) facts; TVR = picked timestamp respects the "
             "before/after anchor.", "",
             f"Evaluated {sum(d['n'] for d in per_op.values())} questions; {n_parse} parseable.", "",
             "| direction | n | model TVR | violation | opportunity (random) | picked-earliest |",
             "|---|--:|--:|--:|--:|--:|"]
    tot = defaultdict(float)
    for op in ("before", "after"):
        d = per_op[op]
        if not d["n"]:
            continue
        tvr = d["valid"] / d["n"]; opp = d["opp_sum"] / d["n"]; early = d["earliest"] / d["n"]
        lines.append(f"| {op} | {d['n']} | {100*tvr:.1f}% | {100*(1-tvr):.1f}% | {100*opp:.1f}% | {100*early:.1f}% |")
        for k in ("n", "valid", "opp_sum", "earliest"):
            tot[k] += d[k]
    if tot["n"]:
        lines.append(f"| **all** | {int(tot['n'])} | **{100*tot['valid']/tot['n']:.1f}%** | "
                     f"**{100*(1-tot['valid']/tot['n']):.1f}%** | {100*tot['opp_sum']/tot['n']:.1f}% | "
                     f"{100*tot['earliest']/tot['n']:.1f}% |")
    lines += ["", "## Notes",
              "- **opportunity** = fraction of time-blind candidate facts on the wrong side "
              "of the anchor (a random picker's expected violation).",
              "- **model TVR < (1 − opportunity)** ⇒ the model does worse than random on time; "
              "**> ** ⇒ it has some temporal preference.",
              "- **picked-earliest** exposes the same early-bias seen for first/last: for "
              "`before` questions picking early tends to satisfy the constraint, for `after` "
              "it tends to violate it.", "",
              "## Sample decisions", ""]
    for direction, anc, p, ok, q in examples:
        lines.append(f"- [{'✓' if ok else '✗'}] ({direction} {anc}) picked `{p}` — {q[:80]}")
    open(args.out, "w").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
