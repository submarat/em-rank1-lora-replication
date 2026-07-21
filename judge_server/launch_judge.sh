#!/bin/bash
# Launch the local open-weight judge (Qwen2.5-32B-Instruct) as an OpenAI-compatible
# server on GPU 1, so it can run alongside a target model on GPU 0.
# Matches LOCAL_JUDGE_BASE_URL / LOCAL_JUDGE_MODEL in global_variables.py.
set -euo pipefail
cd /mnt/localssd/em/model-organisms-for-EM

CUDA_VISIBLE_DEVICES=1 uv run vllm serve Qwen/Qwen2.5-32B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.93 \
    --max-model-len 4096 \
    --dtype bfloat16
