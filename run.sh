#!/bin/bash
export WANDB_MODE=offline

python trainer/sft_pisco.py \
    --out_dir  out/pisco/ \
    --use_wandb \
    --learning_rate 2e-5 \
    --accumulation_steps 256 \
    --warmup_iters 200 \
    --wandb_project pisco-sft-train  \
    --data_path dataset/toy_data.jsonl \
    > log/output.log 2>&1
