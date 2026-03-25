#!/bin/bash

# ----------------------------
# 配置参数
# ----------------------------

COMMIT_ID=$1

ALGO="CustomPPO"
HPARAM_FILE="/root/dae/params/CustomPPO_mujoco.yml"
THREADS=1
LOGGING="--logging"
USE_WANDB="--use_wandb"
PROJECT="mujoco-sb3-dae14"

# Mujoco 环境列表
ENVS=(
    "HalfCheetah-v5"
    "Ant-v5"
    # "HumanoidStandup-v5"
    "Swimmer-v5"
    "Walker2d-v5"
    "Hopper-v5"
    "InvertedPendulum-v5"
    "InvertedDoublePendulum-v5"
    "Reacher-v5"
    "Pusher-v5"
    "Humanoid-v5"
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
    for (( SEED=1; SEED<=MAX_SEED; SEED++ )); do
        
        RUN_ID="${ENV_ID}_seed${SEED}"
        echo "Launching experiment: env=$ENV_ID seed=$SEED run_id=$RUN_ID"

        CUDA_VISIBLE_DEVICES=0 \
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
            --commit_id $COMMIT_ID \
            &
            # 
    done
    wait
    echo "All Mujoco experiments finished."
done


