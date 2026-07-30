#!/usr/bin/env bash
# Create the TimeR4 backbone environment on server 128 (141.223.44.128).
# Run this ON 128 AFTER deploy/rsync_timer4_128.sh has landed timer4/ under
# ~/personal_research4. TimeR4 pins an OLD stack (transformers 4.32.0, deepspeed
# 0.10.0) that is INCOMPATIBLE with the GCR env (transformers 5.x) — so it gets
# its own conda env `timer4`. Do NOT reuse the GCR env.
#
# NOTE on GPUs: GPU0 on 128 is permanently dead ("fell off the bus"). Always run
# with CUDA_VISIBLE_DEVICES=1 (or another live GPU), never 0.
set -euo pipefail

REPO="${HOME}/personal_research4"
T4="${REPO}/timer4"
[ -d "$T4" ] || { echo "missing $T4 — run deploy/rsync_timer4_128.sh on 129 first"; exit 1; }

# 1) conda env with TimeR4's pinned stack (py3.10 so transformers 4.32 + tokenizers build).
if conda env list | grep -q '^timer4 '; then
  echo "[env] timer4 already exists — skipping create"
else
  conda create -y -n timer4 python=3.10
  # torch matched to CUDA on 128 (4090 = cu121+). Adjust index-url if the box differs.
  conda run -n timer4 pip install "torch==2.1.*" --index-url https://download.pytorch.org/whl/cu121
  conda run -n timer4 pip install -r "${T4}/requirements.txt"
  # sentence-transformers is used by the time-aware retriever (retrival.py); pin an
  # older line compatible with transformers 4.32.
  conda run -n timer4 pip install "sentence-transformers==2.2.2"
fi

# 2) sanity: versions + GPU visibility.
conda run -n timer4 python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "| transformers", transformers.__version__,
      "| GPUs", torch.cuda.device_count())
PY

# 3) CPU smoke test — the Phase-0 headroom probe needs NO weights and NO GPU.
echo "[smoke] Phase-0 oracle probe (expect GATE: GO, headroom ~9.9pp) ..."
cd "$REPO"
PYTHONPATH=. conda run -n timer4 python scripts/timer4_oracle_probe.py \
  --preds timer4/results/MultiTQ/finetuned_llama2/predictions.jsonl \
  --test  data/multitq/questions/test.json || \
  echo "[warn] probe failed — check that data/multitq/questions/test.json is present on 128"

cat <<'EOF'

=================  NO BAIDU NEEDED — use a cached reader  =======================
TimeR4's reader is model-agnostic: predict_answer.py feeds each test_prompt.json item's
`text` (retrieved facts + question) to a text-generation pipeline over --model_path, which
accepts ANY HF model. We already have strong readers cached, and TimeR4's exact 72.8
predictions are already in timer4/results/. So we do NOT download the Baidu weights.

  Reader we use:  NousResearch/Meta-Llama-3.1-8B-Instruct   (already in HF cache)
  (The retriever is unneeded: contexts are pre-released in test_prompt.json.)

=================  PHASE 1 — controlled baseline with OUR reader (GPU1)  ========
  cd ~/personal_research4/timer4
  CUDA_VISIBLE_DEVICES=1 conda run -n timer4 python predict_answer.py \
    --model_name llama \
    --model_path NousResearch/Meta-Llama-3.1-8B-Instruct \
    -d datasets/MultiTQ/prompt/test_prompt.json --debug \
    --predict_path datasets/MultiTQ/result
  # then score with our (TimeR4-comparable) protocol:
  cd ~/personal_research4 && PYTHONPATH=. python scripts/timer4_oracle_probe.py \
    --preds timer4/datasets/MultiTQ/result/predictions.jsonl \
    --test  data/multitq/questions/test.json
  # Reference: TimeR4's own fine-tuned-Llama2 run scored 72.5 (their reported 72.8) and is
  # already at timer4/results/MultiTQ/finetuned_llama2/predictions.jsonl.
EOF
