"""Timestamp tokenization verification (Phase 0 checklist).

Risk #3 in the plan: dates may be shattered into subwords, which hurts both the
DFA-ablation constraint and the GNN discriminator's timestamp features.  We
quantify, per tokenizer:

  * how the timestamp formats we emit ("2005-01-01", "[2005-01-01]", the full
    edge fragment) are split;
  * the subword-count distribution over ALL 4017 distinct MultiTQ timestamps;
  * whether registering those 4017 timestamps as special tokens makes each one
    atomic (feasible because the set is small and closed).

Llama-3.1's official tokenizer repo is gated; we use the byte-identical
un-gated mirror ``NousResearch/Meta-Llama-3.1-8B-Instruct`` as a stand-in.

Writes results/tokenization_report.md.
"""
import json
import os
from collections import Counter

from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS_FILE = os.path.join(ROOT, "data", "multitq", "MultiTQ", "kg", "ts2id.json")

TOKENIZERS = [
    ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
    ("Llama-3.1-8B-Instruct (NousResearch mirror)", "NousResearch/Meta-Llama-3.1-8B-Instruct"),
]

PROBE_STRINGS = [
    "2005-01-01",
    "2013-03-28",
    "2016",
    "2010-06",
    "[2013-03-28]",
    " [2013-03-28] ",
    "-> visit [2013-03-28] -> Cape_Verde",
]


def load_timestamps():
    d = json.load(open(TS_FILE))
    return sorted(d.keys())


def probe(tok, s):
    ids = tok.encode(s, add_special_tokens=False)
    pieces = [tok.decode([i]) for i in ids]
    return len(ids), pieces


def analyze_tokenizer(display, name, timestamps):
    lines = [f"### {display}", ""]
    try:
        tok = AutoTokenizer.from_pretrained(name)
    except Exception as e:
        lines += [f"_load failed: {str(e)[:200]}_", ""]
        return lines, None

    lines += [f"- base vocab size: {tok.vocab_size}", "", "**Probe strings:**", ""]
    lines += ["| string | #tokens | pieces |", "|---|--:|---|"]
    for s in PROBE_STRINGS:
        n, pieces = probe(tok, s)
        piece_repr = " · ".join(repr(p) for p in pieces)
        lines.append(f"| `{s}` | {n} | {piece_repr} |")
    lines.append("")

    # Distribution over all distinct timestamps.
    dist = Counter()
    for ts in timestamps:
        n, _ = probe(tok, ts)
        dist[n] += 1
    lines += ["**Subword-count distribution over all "
              f"{len(timestamps)} distinct MultiTQ timestamps (`YYYY-MM-DD`):**", ""]
    lines += ["| #tokens/date | count |", "|--:|--:|"]
    for k in sorted(dist):
        lines.append(f"| {k} | {dist[k]} |")
    atomic = dist.get(1, 0)
    lines += ["", f"- atomic (1-token) dates: {atomic}/{len(timestamps)} "
                  f"({100*atomic/len(timestamps):.1f}%)", ""]

    # Special-token registration experiment.
    before = len(tok)
    added = tok.add_special_tokens(
        {"additional_special_tokens": [f"[{ts}]" for ts in timestamps]})
    after = len(tok)
    # Re-probe a bracketed date now that it is a registered token.
    n_after, _ = probe(tok, "[2013-03-28]")
    lines += ["**Registering the 4017 bracketed timestamps as special tokens:**", "",
              f"- added {added} tokens; vocab {before} -> {after}",
              f"- `[2013-03-28]` now encodes to {n_after} token(s) "
              f"{'✅ atomic' if n_after == 1 else '❌ still split'}", ""]
    return lines, dist


def main():
    timestamps = load_timestamps()
    out = ["# Timestamp Tokenization Verification", "",
           f"MultiTQ has {len(timestamps)} distinct timestamps "
           f"(`{timestamps[0]}` … `{timestamps[-1]}`), all `YYYY-MM-DD`.", ""]
    for display, name in TOKENIZERS:
        lines, _ = analyze_tokenizer(display, name, timestamps)
        out += lines
    out += [
        "## Recommendation", "",
        "The distinct-timestamp set is small (4017) and closed, so **registering",
        "each timestamp as a dedicated special token** makes every date a single,",
        "consistent token — ideal for the GNN discriminator's timestamp feature",
        "and for the DFA-ablation alphabet. This is preferred over relying on",
        "digit-level subword splits (which vary by date and tokenizer).",
        "The bracket delimiters `[` `]` keep timestamps lexically separable in the",
        "path string so the checker can still recover them from raw text.",
        "",
    ]
    path = os.path.join(ROOT, "results", "tokenization_report.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(out))
    print(f"wrote {path}")
    print("\n".join(out[:4]))


if __name__ == "__main__":
    main()
