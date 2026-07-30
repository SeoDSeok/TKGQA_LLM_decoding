"""QLoRA training-pipeline validation for the KG-specialized LLM (GPU fit-check).

Validates the *real* training path on one RTX 4090 (24 GB):
  * load Qwen2.5-7B-Instruct in 4-bit (nf4),
  * swap in the timestamp-special-token tokenizer and resize embeddings,
  * LoRA on attention projections + train the new timestamp embeddings,
  * run a short SFT loop on the answer-verified temporal path data,
  * report peak VRAM and the loss trajectory.

This is a pipeline/VRAM validation, NOT a converged model.
"""
import argparse
import json
import os
import time

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data(path, tokenizer, max_len=160):
    exs = [json.loads(l) for l in open(path)]
    batches = []
    for e in exs:
        full = e["prompt"] + e["completion"] + tokenizer.eos_token
        prompt_ids = tokenizer(e["prompt"], add_special_tokens=False).input_ids
        ids = tokenizer(full, add_special_tokens=False).input_ids[:max_len]
        labels = list(ids)
        # mask prompt tokens
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        batches.append((ids, labels))
    return batches


def collate(items, pad_id):
    maxlen = max(len(x[0]) for x in items)
    input_ids, labels, attn = [], [], []
    for ids, lab in items:
        p = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * p)
        labels.append(lab + [-100] * p)
        attn.append([1] * len(ids) + [0] * p)
    return (torch.tensor(input_ids), torch.tensor(attn), torch.tensor(labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data/multitq/tokenizers/Qwen2.5-7B-Instruct-ts"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data/multitq/train_paths_train.jsonl"))
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    dev = "cuda"
    torch.cuda.reset_peak_memory_stats()
    print(f"loading tokenizer (with timestamp tokens) from {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"loading {args.base} in 4-bit ...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map={"": 0},
        torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.resize_token_embeddings(len(tok))
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      modules_to_save=["embed_tokens"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    data = load_data(args.data, tok)
    print(f"training examples: {len(data)}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()

    losses = []
    t0 = time.time()
    step = 0
    i = 0
    while step < args.steps:
        batch = data[i % len(data): i % len(data) + args.bs]
        if len(batch) < args.bs:
            batch = data[:args.bs]
        i += args.bs
        input_ids, attn, labels = collate(batch, tok.pad_token_id)
        input_ids, attn, labels = input_ids.to(dev), attn.to(dev), labels.to(dev)
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = out.loss
        loss.backward()
        opt.step(); opt.zero_grad()
        losses.append(loss.item())
        step += 1
        if step == 1 or step % 10 == 0:
            print(f"  step {step:3d}  loss {loss.item():.4f}  "
                  f"peakVRAM {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.max_memory_reserved() / 1e9
    summary = {
        "base": args.base, "steps": args.steps, "bs": args.bs,
        "loss_start": round(sum(losses[:3]) / min(3, len(losses)), 4),
        "loss_end": round(sum(losses[-3:]) / min(3, len(losses)), 4),
        "peak_alloc_GB": round(peak, 2), "peak_reserved_GB": round(reserved, 2),
        "sec_per_step": round(dt / args.steps, 3),
        "n_examples": len(data),
    }
    print("\n=== QLoRA fit-check summary ===")
    print(json.dumps(summary, indent=2))
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(summary, open(os.path.join(ROOT, "results", "qlora_fit_check.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
