#!/bin/bash

# ----------------------------
# 配置参数
# ----------------------------
ALGO="PPO"
HPARAM_FILE="/root/dae/params/PPO_mujoco.yml"
THREADS=32
LOGGING="--logging"
USE_WANDB="--use_wandb"
PROJECT="customppo-con"

# Mujoco 环境列表
ENVS=(
    "Ant-v5"
    "Hopper-v5"
    "Walker2d-v5"
    "HalfCheetah-v5"
    "Humanoid-v5"
    "HumanoidStandup-v5"
    "Swimmer-v5"
    "Reacher-v5"
    "Pusher-v5"
    "InvertedPendulum-v5"
    "InvertedDoublePendulum-v5"
)

# 运行的最大 seed 数量
MAX_SEED=5

# activate proxy if we use wandb
# source /etc/profile.d/clash.sh
# proxy_on

# ----------------------------
# 双循环：每个环境 × 多个种子
# ----------------------------
for ENV_ID in "${ENVS[@]}"; do
    for (( SEED=0; SEED<MAX_SEED; SEED++ )); do
        
        RUN_ID="${ENV_ID}_seed${SEED}"
        echo "Launching experiment: env=$ENV_ID seed=$SEED run_id=$RUN_ID"

        CUDA_VISIBLE_DEVICES=1 \
        uv run python train.py \
            --algo $ALGO \
            --hparam_file $HPARAM_FILE \
            --envs $ENV_ID \
            --threads $THREADS \
            $LOGGING \
            $USE_WANDB \
            --continous \
            --project $PROJECT \
            --seed $SEED \
            --run_id $SEED  \
            &
    done
    wait
    echo "All Mujoco experiments finished."
done


