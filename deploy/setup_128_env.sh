#!/usr/bin/env bash
# Recreate the GCR inference environment on server 128 (141.223.44.128).
# Run this ON 128 after the repo has been rsynced to ~/personal_research4.
# The gcr_base patch (BitsAndBytesConfig for transformers>=5) travels with the
# repo, so no separate patch step is needed.
set -euo pipefail

REPO="${HOME}/personal_research4"

# 1) conda env: clone base if it has the right torch/transformers, else pin.
#    Memory recipe: GCR = clone of base (torch 2.6 cu124, transformers 5.12.1).
if conda env list | grep -q '^GCR '; then
  echo "[env] GCR already exists — skipping create"
else
  # Prefer cloning base (fastest, matches origin server); fall back to a fresh env.
  if conda env list | grep -q '^base '; then
    conda create -y -n GCR --clone base
  else
    conda create -y -n GCR python=3.12
    conda run -n GCR pip install "torch==2.6.*" --index-url https://download.pytorch.org/whl/cu124
    conda run -n GCR pip install "transformers==5.12.1" accelerate bitsandbytes safetensors
  fi
fi

# 2) inference deps (do NOT downgrade transformers to 4.44 — tokenizers 0.19 has
#    no py3.13 wheel and the Rust build fails).
conda run -n GCR pip install marisa-trie peft python-dotenv tiktoken openai

# 3) sanity: GPU visible, versions correct, one free 4090.
conda run -n GCR python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "| transformers", transformers.__version__,
      "| GPUs", torch.cuda.device_count())
PY

echo
echo "[next] base model Qwen2.5-7B-Instruct (15G):"
echo "  - if 128 has internet, it auto-downloads on first run (no action)."
echo "  - if offline, rsync the HF cache from the origin server:"
echo "      rsync -avz ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct \\"
echo "        dsseo@141.223.44.128:~/.cache/huggingface/hub/"
echo
echo "[run] the E1 main experiment (single 4090):"
echo "  cd $REPO && CUDA_VISIBLE_DEVICES=<free_gpu> PYTHONPATH=. \\"
echo "    conda run -n GCR python discriminator/phase2_trielocal.py \\"
echo "      --llm_adapter save_models/gcr-multitq-combined \\"
echo "      --disc save_models/discriminator/sig_all.pt"
