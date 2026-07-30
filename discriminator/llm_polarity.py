"""Zero-shot operator-polarity classification with the base LLM.

The zero-shot-operator failure is a *language-grounding* problem, not a
value-comparison one: "…before 2010?" and "…after 2010?" are near-identical
surface strings with opposite temporal meaning, so no holistic question embedding
(MiniLM or 7B mean-pool) separates a held-out one. But the operator's meaning is
exactly what a pretrained LLM knows. Here we ask the base Qwen, zero-shot, to
name the temporal direction each question wants:

    latest / earliest  (superlative: last / first)
    after  / before     (threshold vs an anchor date)

If accuracy is high, an operator-agnostic discriminator conditioned on this
*polarity* (not on operator words) generalizes to any new operator the LLM can
describe — the genuine zero-shot fix.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from discriminator.dataset import build_examples

LABELS = ["earliest", "latest", "before", "after"]
OP2LABEL = {"first": "earliest", "last": "latest", "before": "before", "after": "after"}

INSTR = (
    "You classify the temporal intent of a question about dated events.\n"
    "Choose exactly ONE label:\n"
    "- earliest : wants the FIRST / earliest time something happened\n"
    "- latest   : wants the LAST / most recent time something happened\n"
    "- before   : wants events strictly BEFORE a given date/anchor\n"
    "- after    : wants events strictly AFTER a given date/anchor\n"
    "Answer with only the single label word.\n"
)


def build_model(base="Qwen/Qwen2.5-7B-Instruct"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(base)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb,
            device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    return model, tok


@torch.inference_mode()
def classify(model, tok, questions):
    """Return predicted label per question by comparing the 4 label-word logprobs."""
    label_ids = [tok(" " + l, add_special_tokens=False).input_ids[0] for l in LABELS]
    preds = []
    for q in questions:
        msgs = [{"role": "user", "content": INSTR + f"\nQuestion: {q}\nLabel:"}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids):
            ids = ids["input_ids"]
        ids = ids.to(model.device)
        logits = model(ids).logits[0, -1]
        lp = torch.log_softmax(logits, dim=-1)
        preds.append(LABELS[int(np.argmax([lp[i].item() for i in label_ids]))])
    return preds


def main():
    import argparse
    from collections import Counter, defaultdict
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_op", type=int, default=150)
    args = ap.parse_args()
    ex = build_examples("test", ("first", "last", "before", "after"))
    by_op = defaultdict(list)
    for e in ex:
        if e["question"] not in {x["question"] for x in by_op[e["op"]]}:
            by_op[e["op"]].append(e)
    model, tok = build_model()
    print("### Zero-shot LLM polarity classification (base Qwen2.5-7B-Instruct)")
    conf = defaultdict(Counter); acc = {}
    for op in ("first", "last", "before", "after"):
        qs = [e["question"] for e in by_op[op][: args.per_op]]
        preds = classify(model, tok, qs)
        gold = OP2LABEL[op]
        acc[op] = np.mean([p == gold for p in preds])
        for p in preds:
            conf[op][p] += 1
        print(f"  {op:7s} (gold={gold:8s}) acc {acc[op]:.3f}  preds={dict(conf[op])}")
    print(f"\nMACRO polarity acc = {np.mean(list(acc.values())):.3f}")


if __name__ == "__main__":
    main()
