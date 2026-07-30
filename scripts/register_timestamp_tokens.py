"""Register MultiTQ's 4017 timestamps as special tokens (Phase 0 checklist).

The tokenization study (results/tokenization_report.md) showed 0% of MultiTQ
dates are atomic (Qwen 10 tok, Llama 6 tok). Since the timestamp set is small
and closed, we register each bracketed date `[YYYY-MM-DD]` as a dedicated special
token so every timestamp is a single, consistent token. The resulting tokenizer
is saved for LoRA training / constrained decoding.

Output: data/multitq/tokenizers/<model>-ts/ (a saved AutoTokenizer).
"""
import argparse
import json
import os

from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS_FILE = os.path.join(ROOT, "data", "multitq", "MultiTQ", "kg", "ts2id.json")


def build(model_name: str, out_dir: str):
    timestamps = sorted(json.load(open(TS_FILE)).keys())
    tok = AutoTokenizer.from_pretrained(model_name)
    before = len(tok)
    tokens = [f"[{ts}]" for ts in timestamps]
    added = tok.add_special_tokens({"additional_special_tokens": tokens})
    after = len(tok)

    # sanity: every date now atomic
    bad = [ts for ts in timestamps
           if len(tok.encode(f"[{ts}]", add_special_tokens=False)) != 1]
    os.makedirs(out_dir, exist_ok=True)
    tok.save_pretrained(out_dir)

    print(f"model={model_name}")
    print(f"  added {added} timestamp tokens; vocab {before} -> {after}")
    print(f"  non-atomic after registration: {len(bad)} (expect 0)")
    print(f"  saved -> {out_dir}")
    print(f"  NOTE: call model.resize_token_embeddings({after}) after loading the "
          f"base model for training.")
    return {"model": model_name, "added": added, "vocab_before": before,
            "vocab_after": after, "non_atomic": len(bad), "out_dir": out_dir}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tag", default=None, help="output subdir tag")
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1] + "-ts"
    out = os.path.join(ROOT, "data", "multitq", "tokenizers", tag)
    info = build(args.model, out)
    meta = os.path.join(ROOT, "data", "multitq", "tokenizers", "registration_info.json")
    existing = json.load(open(meta)) if os.path.exists(meta) else {}
    existing[tag] = info
    json.dump(existing, open(meta, "w"), indent=2)
