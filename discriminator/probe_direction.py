"""Diagnose the zero-shot-operator failure: is the 'wants-later' direction
linearly recoverable from the frozen question embedding, and does it generalize
to *held-out* operators?

Label each question by whether its operator prefers LATER timestamps:
  first -> 0, last -> 1, before -> 0, after -> 1.
Train a logistic probe on {first, after} and test on {last, before} (Split-A
operators). If the probe transfers (test acc high), the encoder already carries a
compositional direction axis and a direction-structured head can fix zero-shot;
if it inverts (acc < 0.5), MiniLM lacks the signal and we need the KG-LLM hidden
state (design §6.3).
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from discriminator.dataset import build_examples

WANTS_LATER = {"first": 0, "last": 1, "before": 0, "after": 1}


_LLM = {}


def encode(questions, device="cuda", encoder="minilm"):
    if encoder == "minilm":
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        return m.encode(questions, batch_size=512, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)
    # encoder == 'llm': mean-pooled last-hidden of a frozen Qwen2.5-7B-Instruct
    return encode_llm(questions, device)


def encode_llm(questions, device="cuda", base="Qwen/Qwen2.5-7B-Instruct", bs=32):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if "m" not in _LLM:
        tok = AutoTokenizer.from_pretrained(base)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        m = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb,
                device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa",
                output_hidden_states=True)
        m.eval()
        _LLM["m"], _LLM["tok"] = m, tok
    m, tok = _LLM["m"], _LLM["tok"]
    out = []
    import numpy as np
    with torch.inference_mode():
        for i in range(0, len(questions), bs):
            chunk = questions[i:i + bs]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
            hs = m(**enc).hidden_states[-1]              # (B,T,H)
            mask = enc.attention_mask.unsqueeze(-1).float()
            pooled = (hs * mask).sum(1) / mask.sum(1)    # mean-pool over real tokens
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            out.append(pooled.float().cpu().numpy())
    return np.concatenate(out)


def dedup_questions(split, ops):
    seen = {}
    for e in build_examples(split, tuple(ops)):
        seen[e["question"]] = e["op"]
    return list(seen.keys()), [seen[q] for q in seen]


def probe(train_ops, test_ops, device="cuda", encoder="minilm"):
    from sklearn.linear_model import LogisticRegression
    trq, tro = dedup_questions("train", train_ops)
    teq, teo = dedup_questions("test", test_ops)
    if len(set(WANTS_LATER[o] for o in tro)) < 2:
        print(f"  [skip] train ops {train_ops} are single-direction "
              f"(all wants_later={WANTS_LATER[tro[0]]}) — degenerate for a direction probe")
        return None, None, None
    Xtr = encode(trq, device, encoder); Xte = encode(teq, device, encoder)
    ytr = np.array([WANTS_LATER[o] for o in tro])
    yte = np.array([WANTS_LATER[o] for o in teo])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    acc_in = clf.score(Xtr, ytr)
    acc_out = clf.score(Xte, yte)
    # per-op held-out accuracy
    per = {}
    for op in test_ops:
        m = np.array([o == op for o in teo])
        per[op] = float((clf.predict(Xte[m]) == yte[m]).mean())
    return acc_in, acc_out, per


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="minilm", choices=["minilm", "llm"])
    args = ap.parse_args()
    print(f"### direction probe — encoder={args.encoder}")
    print("== Split-A: train {first, after} -> test {last, before} ==")
    ai, ao, per = probe(["first", "after"], ["last", "before"], encoder=args.encoder)
    if ai is not None:
        print(f"train acc {ai:.3f} | held-out acc {ao:.3f} | per-op {per}")
    print("\n== cross-family: train {first, last} -> test {before, after} ==")
    ai, ao, per = probe(["first", "last"], ["before", "after"], encoder=args.encoder)
    if ai is not None:
        print(f"train acc {ai:.3f} | held-out acc {ao:.3f} | per-op {per}")
    print("\n== sanity: all-ops in-distribution ==")
    ai, ao, per = probe(["first", "last", "before", "after"],
                        ["first", "last", "before", "after"], encoder=args.encoder)
    if ai is not None:
        print(f"train acc {ai:.3f} | test acc {ao:.3f} | per-op {per}")
