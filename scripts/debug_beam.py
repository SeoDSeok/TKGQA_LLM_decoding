"""Debug the beam x constrained-decoding interaction (in-beam recall bug)."""
import json, os, sys
import torch

GCR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gcr_base")
sys.path.insert(0, GCR)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from src.trie import MarisaTrie
from src.graph_constrained_decoding import GraphConstrainedDecoding
from checker.path_format import wrap_path, temporal_path_to_string, extract_timestamps

ADAPTER = os.path.join(ROOT, "save_models/gcr-multitq-qwen7b")
PROMPT = ("Reasoning path is a sequence of temporal triples in the KG that connects the "
          "topic entity to the answer, where each relation carries the fact's timestamp "
          "in brackets. It starts with <PATH> and ends with </PATH>.\n"
          "# Question:\n{question}\n# Topic entity:\n{topic}\n")

tok = AutoTokenizer.from_pretrained(ADAPTER)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", quantization_config=bnb,
        device_map={"": 0}, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
model.resize_token_embeddings(len(tok))
model = PeftModel.from_pretrained(model, ADAPTER).eval()

sigs = [json.loads(l) for l in open(os.path.join(ROOT, "data/multitq/signatures/first_last_time_test.jsonl"))]
# pick a few "last" questions with >=3 candidates
cases = [s for s in sigs if s["op"] == "last" and s["n_candidates"] >= 3][:3]

for sig in cases:
    paths = [wrap_path(temporal_path_to_string([(sig["s"], sig["r"], sig["o"], t)]))
             for t in sig["candidate_timestamps"]]
    tok_paths = tok(paths, add_special_tokens=False).input_ids
    trie = MarisaTrie(tok_paths, max_token_id=len(tok) + 1)
    prompt = PROMPT.format(question=sig["question"], topic=sig["s"])
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    ii = enc.input_ids.to(model.device); am = enc.attention_mask.to(model.device)
    print("=" * 80)
    print("Q:", sig["question"])
    print("candidates:", sig["candidate_timestamps"], " gold(last)=", sorted(sig["candidate_timestamps"])[-1])
    for beam in (1, 10):
        gcr = GraphConstrainedDecoding(tok, trie, None, None, True)
        kw = dict(input_ids=ii, attention_mask=am, max_new_tokens=48, do_sample=False,
                  prefix_allowed_tokens_fn=gcr.allowed_tokens_fn, pad_token_id=tok.eos_token_id,
                  num_beams=beam)
        if beam > 1:
            kw.update(num_return_sequences=beam)
        with torch.inference_mode():
            out = model.generate(**kw)
        seqs = out if out.dim() == 2 else out.unsqueeze(0)
        ts_seen = []
        for row in seqs:
            g = tok.decode(row[ii.shape[1]:], skip_special_tokens=False)
            ts = extract_timestamps(g)
            ts_seen.append(str(ts[0]) if ts else f"NONE::{g[:40]!r}")
        print(f"  beam={beam}: returned {len(seqs)} seqs; timestamps={ts_seen}")
