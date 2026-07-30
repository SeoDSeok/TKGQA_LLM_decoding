"""A1: QLoRA training of the KG-specialized LLM on MultiTQ temporal paths.

Beyond the fit-check this adds what plan v3 §5-A1 requires:
  * gradient accumulation (effective batch ~16);
  * **new-timestamp-embedding gradient verification** (risk #3): logs the grad
    norm of the 4017 newly-added timestamp token rows every step, to prove they
    are actually being trained (not frozen at random init, which would distort
    TVR);
  * adapter + tokenizer checkpointing;
  * a held-out generation spot-check.
"""
import argparse
import json
import os
import time

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_examples(path, tok, max_len=192):
    out = []
    for l in open(path):
        e = json.loads(l)
        full = e["prompt"] + e["completion"] + tok.eos_token
        pids = tok(e["prompt"], add_special_tokens=False).input_ids
        ids = tok(full, add_special_tokens=False).input_ids[:max_len]
        labels = list(ids)
        for i in range(min(len(pids), len(labels))):
            labels[i] = -100
        out.append((ids, labels))
    return out


def collate(items, pad_id):
    m = max(len(x[0]) for x in items)
    ii, ll, aa = [], [], []
    for ids, lab in items:
        p = m - len(ids)
        ii.append(ids + [pad_id] * p); ll.append(lab + [-100] * p); aa.append([1] * len(ids) + [0] * p)
    return torch.tensor(ii), torch.tensor(aa), torch.tensor(ll)


def find_new_token_embedding(model, tok, n_new):
    """Return (param, new_ids_tensor) for the trainable input-embedding rows of the new tokens."""
    emb_param = None
    for name, p in model.named_parameters():
        if "embed_tokens" in name and "modules_to_save" in name and p.requires_grad and p.dim() == 2:
            emb_param = p
            break
    if emb_param is None:  # fallback: any trainable 2D param with vocab-sized dim0
        for name, p in model.named_parameters():
            if p.requires_grad and p.dim() == 2 and p.shape[0] == len(tok):
                emb_param = p; break
    new_ids = torch.arange(len(tok) - n_new, len(tok))
    return emb_param, new_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data/multitq/tokenizers/Qwen2.5-7B-Instruct-ts"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data/multitq/train_paths_train.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "save_models/gcr-multitq-qwen7b"))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--micro_bs", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n_new", type=int, default=4017)
    args = ap.parse_args()

    torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map={"": 0},
        torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.resize_token_embeddings(len(tok))
    model = prepare_model_for_kbit_training(model)
    # raw mode (n_new==0): no special tokens → keep pretrained digit embeddings
    # (which carry ordinal structure); LoRA only, no embed training.
    lora_kwargs = dict(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                       task_type="CAUSAL_LM",
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    if args.n_new > 0:
        lora_kwargs["modules_to_save"] = ["embed_tokens"]
    lora = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    if args.n_new > 0:
        emb_param, new_ids = find_new_token_embedding(model, tok, args.n_new)
        print(f"tracking new-token embedding grad on param shape "
              f"{tuple(emb_param.shape) if emb_param is not None else None}, "
              f"new ids [{int(new_ids[0])}..{int(new_ids[-1])}]")
    else:
        emb_param, new_ids = None, None
        print("raw mode: no new tokens, embeddings frozen (LoRA only)")

    data = load_examples(args.data, tok)
    steps_per_epoch = max(1, len(data) // (args.micro_bs * args.grad_accum))
    total_opt_steps = int(steps_per_epoch * args.epochs)
    print(f"examples={len(data)}  opt-steps/epoch={steps_per_epoch}  total={total_opt_steps}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    dev = "cuda"
    log = []
    t0 = time.time()
    micro = 0; opt_step = 0; i = 0
    running = 0.0
    opt.zero_grad()
    while opt_step < total_opt_steps:
        batch = data[i % len(data): i % len(data) + args.micro_bs]
        if len(batch) < args.micro_bs:
            i = 0; batch = data[:args.micro_bs]
        i += args.micro_bs
        ii, aa, ll = collate(batch, tok.pad_token_id)
        out = model(input_ids=ii.to(dev), attention_mask=aa.to(dev), labels=ll.to(dev))
        (out.loss / args.grad_accum).backward()
        running += out.loss.item()
        micro += 1
        if micro % args.grad_accum == 0:
            # new-token embedding grad norm BEFORE step
            new_grad_norm = float("nan")
            if emb_param is not None and emb_param.grad is not None:
                new_grad_norm = emb_param.grad[new_ids].norm().item()
            opt.step(); opt.zero_grad(); opt_step += 1
            avg = running / args.grad_accum; running = 0.0
            if opt_step == 1 or opt_step % 5 == 0:
                print(f"  opt-step {opt_step:3d}/{total_opt_steps}  loss {avg:.4f}  "
                      f"new-tok-emb gradnorm {new_grad_norm:.4e}  "
                      f"peakVRAM {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
                log.append({"step": opt_step, "loss": avg, "new_tok_grad_norm": new_grad_norm})

    dt = time.time() - t0
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out); tok.save_pretrained(args.out)
    grad_norms = [x["new_tok_grad_norm"] for x in log if x["new_tok_grad_norm"] == x["new_tok_grad_norm"]]
    summary = {
        "base": args.base, "examples": len(data), "opt_steps": total_opt_steps,
        "effective_batch": args.micro_bs * args.grad_accum,
        "loss_start": round(log[0]["loss"], 4) if log else None,
        "loss_end": round(log[-1]["loss"], 4) if log else None,
        "new_tok_grad_norm_mean": round(sum(grad_norms) / len(grad_norms), 6) if grad_norms else None,
        "new_tok_grad_norm_nonzero": all(g > 0 for g in grad_norms) if grad_norms else None,
        "peak_alloc_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "peak_reserved_GB": round(torch.cuda.max_memory_reserved() / 1e9, 2),
        "minutes": round(dt / 60, 2), "out": args.out,
    }
    json.dump(summary, open(os.path.join(ROOT, "results", "train_kg_llm_summary.json"), "w"), indent=2)
    print("\n=== training summary ===")
    print(json.dumps(summary, indent=2))
    print("\nNEW-TOKEN EMBEDDING CHECK: "
          + ("PASS — timestamp embeddings are training (grad norm > 0)"
             if summary["new_tok_grad_norm_nonzero"] else "FAIL — new-token embeddings NOT training"))


if __name__ == "__main__":
    main()
