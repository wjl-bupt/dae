#!/bin/bash

# ----------------------------
# 配置参数
# ----------------------------
ALGO="CustomPPO"
HPARAM_FILE="/root/dae/params/CustomPPO_atari.yml"
ENVS="ALE/Breakout-v5"
THREADS=32
LOGGING="--logging"
USE_WANDB="--use_wandb"
PROJECT="call-back"

# 你想跑的种子列表
SEEDS=(0 1 2 3 4)
# SEEDS=(0)

# ----------------------------
# 循环启动每个种子实验
# ----------------------------
for SEED in "${SEEDS[@]}"; do
    RUN_ID=$SEED  # 可用 seed 作为 run_id
    echo "Launching experiment: seed=$SEED, run_id=$RUN_ID"
    
    CUDA_VISIBLE_DEVICES=1 \
    uv run python train.py \
        --algo $ALGO \
        --hparam_file $HPARAM_FILE \
        --envs $ENVS \
        --threads $THREADS \
        $LOGGING \
        $USE_WANDB \
        --project $PROJECT \
        --seed $SEED \
        --run_id $RUN_ID \
        &
done

# 等待所有后台进程完成
wait

echo "All experiments finished."
