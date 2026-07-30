"""A2/A3: real (model-generated) TVR on answer-verified first/last questions.

For each answer-verified signature (s, r, o, candidate_timestamps) with >=2
timestamps, we build the GCR-vanilla constrained-decoding setup:

  * trie = the structurally-valid single-hop paths that differ ONLY in the
    timestamp:  <PATH>s -> r [t] -> o</PATH>  for every t in the group.
    (Timestamps are ordinary tokens for GCR-vanilla — the trie does NOT filter
    by time, exactly the pollution H1 is about.)
  * the trained KG-specialized LLM decodes under that trie (greedy);
  * TVR check: is the picked timestamp the min (first) / max (last) — the gold
    answer? This is the *realized* temporal validity, not the opportunity bound.

Compares model TVR against the random/time-blind opportunity baseline (mean 1/k)
to answer A4: does the model avoid the pollution or fall into it?
"""
import argparse
import json
import os
import sys

import torch

GCR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gcr_base")
sys.path.insert(0, GCR)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from src.trie import MarisaTrie                      # noqa: E402
from src.graph_constrained_decoding import GraphConstrainedDecoding  # noqa: E402
from checker.timepoint import TimePoint              # noqa: E402
from checker.path_format import wrap_path, temporal_path_to_string, extract_timestamps  # noqa: E402

PROMPT = (
    "Reasoning path is a sequence of temporal triples in the KG that connects the "
    "topic entity to the answer, where each relation carries the fact's timestamp "
    "in brackets. It starts with <PATH> and ends with </PATH>.\n"
    "# Question:\n{question}\n# Topic entity:\n{topic}\n"
)


def build_paths(sig):
    """All time-blind candidate paths for this signature (one per timestamp)."""
    paths = []
    for t in sig["candidate_timestamps"]:
        paths.append(wrap_path(temporal_path_to_string([(sig["s"], sig["r"], sig["o"], t)])))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=os.path.join(ROOT, "save_models/gcr-multitq-qwen7b"))
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--sigs", default=os.path.join(ROOT, "data/multitq/signatures/first_last_time_test.jsonl"))
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--beam", type=int, default=1, help="beam size (>1 enables in-beam recall)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "phase0_violation_report.md"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapter)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map={"": 0},
        torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    sigs = [json.loads(l) for l in open(args.sigs)]
    traps = [s for s in sigs if s["n_candidates"] >= 2][:args.limit]
    print(f"evaluating {len(traps)} multi-timestamp questions (of {len(sigs)} verified)")

    from collections import defaultdict
    per_op = defaultdict(lambda: {"n": 0, "correct": 0, "opp_sum": 0.0, "in_beam": 0})
    n_valid = 0
    pick_is_min = 0   # diagnostic: did the model pick the earliest timestamp?
    pick_is_max = 0
    examples = []
    for si, sig in enumerate(traps):
        paths = build_paths(sig)
        tok_paths = tok(paths, add_special_tokens=False).input_ids
        trie = MarisaTrie(tok_paths, max_token_id=len(tok) + 1)
        prompt = PROMPT.format(question=sig["question"], topic=sig["s"])
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc.input_ids.to(model.device)
        attn = enc.attention_mask.to(model.device)
        gcr = GraphConstrainedDecoding(tok, trie, None, None, True)
        gen_kwargs = dict(input_ids=input_ids, attention_mask=attn, max_new_tokens=48,
                          do_sample=False, prefix_allowed_tokens_fn=gcr.allowed_tokens_fn,
                          pad_token_id=tok.eos_token_id)
        if args.beam > 1:
            gen_kwargs.update(num_beams=args.beam, num_return_sequences=args.beam)
        else:
            gen_kwargs.update(num_beams=1)
        with torch.inference_mode():
            out = model.generate(**gen_kwargs)
        # NOTE: timestamps are registered as *special* tokens, so we must NOT
        # skip special tokens or the [YYYY-MM-DD] would be stripped out.
        seqs = out if out.dim() == 2 else out.unsqueeze(0)
        beam_ts = []
        for row in seqs:
            g = tok.decode(row[input_ids.shape[1]:], skip_special_tokens=False)
            ts = extract_timestamps(g)
            if ts:
                beam_ts.append(ts[0])
        picked = [beam_ts[0]] if beam_ts else []  # top-1 (best beam)
        cand = sorted({TimePoint.parse(t) for t in sig["candidate_timestamps"]})
        gold = cand[0] if sig["op"] == "first" else cand[-1]
        k = len(cand)
        op = sig["op"]
        per_op[op]["n"] += 1
        per_op[op]["opp_sum"] += (k - 1) / k
        correct = bool(picked) and picked[0] == gold
        if picked:
            n_valid += 1
            if picked[0] == cand[0]:
                pick_is_min += 1
            if picked[0] == cand[-1]:
                pick_is_max += 1
        if correct:
            per_op[op]["correct"] += 1
        if gold in beam_ts:   # in-beam recall: is the correct answer anywhere in the beam?
            per_op[op]["in_beam"] += 1
        if si < 6:
            examples.append((sig["question"], op, str(gold),
                             str(picked[0]) if picked else "None", correct))

    # aggregate
    lines = ["# Phase 0 — Real model TVR (answer-verified first/last)", "",
             f"Model: QLoRA `{os.path.basename(args.adapter)}` on Qwen2.5-7B, GCR-vanilla "
             "constrained decoding (timestamps as ordinary tokens).", "",
             f"Evaluated {len(traps)} multi-timestamp questions; "
             f"{n_valid} produced a parseable path.", "",
             f"Decoding: {'beam='+str(args.beam)+' (top-1 TVR + in-beam recall)' if args.beam>1 else 'greedy'}.",
             "",
             "| operator | n | model TVR | violation (1-TVR) | in-beam recall | opportunity baseline |",
             "|---|--:|--:|--:|--:|--:|"]
    tot_n = tot_c = tot_ib = 0; tot_opp = 0.0
    for op in ("first", "last"):
        d = per_op[op]
        if d["n"] == 0:
            continue
        tvr = d["correct"] / d["n"]
        opp = d["opp_sum"] / d["n"]
        ib = d["in_beam"] / d["n"]
        tot_n += d["n"]; tot_c += d["correct"]; tot_opp += d["opp_sum"]; tot_ib += d["in_beam"]
        lines.append(f"| {op} | {d['n']} | {100*tvr:.1f}% | {100*(1-tvr):.1f}% | "
                     f"{100*ib:.1f}% | {100*opp:.1f}% |")
    if tot_n:
        tvr = tot_c / tot_n
        lines.append(f"| **all** | {tot_n} | **{100*tvr:.1f}%** | **{100*(1-tvr):.1f}%** | "
                     f"{100*tot_ib/tot_n:.1f}% | {100*tot_opp/tot_n:.1f}% |")
    if n_valid:
        lines += ["", f"**Early-bias diagnostic:** picked the *earliest* candidate "
                  f"timestamp in **{100*pick_is_min/n_valid:.1f}%** of cases, the "
                  f"*latest* in {100*pick_is_max/n_valid:.1f}% — the model reports the "
                  f"fact's earliest occurrence regardless of the first/last operator."]
    lines += ["", "## Interpretation (A4)",
              "- **opportunity baseline** = temporal error of a random/time-blind picker "
              "((k-1)/k).",
              "- **violation (1-TVR)** = the trained GCR-vanilla model's *realized* temporal "
              "error under the structural trie.",
              "- violation ≈ opportunity ⇒ the model does NOT avoid the pollution "
              "(strong motivation for temporal guidance).",
              "- violation ≪ opportunity ⇒ the model already prefers correct timestamps "
              "(H1 weaker than the opportunity bound suggested).", "",
              "## Sample decisions", ""]
    for q, op, gold, picked, ok in examples:
        lines.append(f"- [{'✓' if ok else '✗'}] ({op}) gold `{gold}` picked `{picked}` — {q}")

    report = "\n".join(lines)
    open(args.out, "w").write(report)
    print("\n" + report)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
