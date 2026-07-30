#!/usr/bin/env bash
# Push the Llama adapter (Exp 2 dependency, excluded from the minimal bundle) to 128.
# Run this ON 129 (where the file lives). Safe to run in parallel with the E1 run on
# 128 — it is a 129<->128 network transfer, not GPU work. Prompts for the 128 password
# once (run `ssh-copy-id dsseo@141.223.44.128` first to make it passwordless).
set -euo pipefail

SRC="/home/dsseo/personal_research4/save_models/gcr-multitq-llama"
DST="dsseo@141.223.44.128:~/personal_research4/save_models/"

[ -d "$SRC" ] || { echo "missing $SRC on this host"; exit 1; }
echo "[rsync] gcr-multitq-llama (2.1 GB) -> 128 ..."
rsync -avz --progress "$SRC" "$DST"
echo "[done] Exp 2 adapter is on 128. Base rmanluo/GCR-Meta-Llama-3.1-8B-Instruct (15 GB)"
echo "       auto-downloads on 128 if online; else also rsync its HF cache dir."
