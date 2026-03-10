# #!/bin/bash

# COMMIT=$1
# SCRIPT=$2

# if [ -z "$COMMIT" ]; then
#   echo "Usage: ./run_commit.sh <commit_id> <script>"
#   exit 1
# fi

# # 创建logs目录
# mkdir -p logs

# # commit代码目录
# TMP_DIR=logs/code_$COMMIT

# echo "Export commit $COMMIT to $TMP_DIR"

# rm -rf $TMP_DIR
# mkdir -p $TMP_DIR

# # 导出commit代码
# git archive $COMMIT | tar -x -C $TMP_DIR

# # 让python优先加载这个commit的代码
# export PYTHONPATH=$TMP_DIR:$PYTHONPATH

# echo "Using code from: $TMP_DIR"
# echo "Running script: $SCRIPT"

# bash $SCRIPT


#!/bin/bash

COMMIT=$1
SCRIPT=$2

if [ -z "$COMMIT" ] || [ -z "$SCRIPT" ]; then
  echo "Usage: ./run_commit.sh <commit_id> <script>"
  exit 1
fi

# 创建 logs 目录
mkdir -p logs

# commit 代码目录
TMP_DIR=$(pwd)/logs/code_$COMMIT

echo "Export commit $COMMIT to $TMP_DIR"

rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

# 导出 commit 代码
git archive $COMMIT | tar -x -C "$TMP_DIR"

echo "-----------------------------------"
echo "Using commit: $COMMIT"
echo "Code directory: $TMP_DIR"
echo "Script: $SCRIPT"
echo "-----------------------------------"

# 进入 commit 目录
cd "$TMP_DIR" || exit 1

# 设置 pythonpath（保险）
export PYTHONPATH="$TMP_DIR:$PYTHONPATH"

# 执行脚本（用原始路径）
bash "$(git rev-parse --show-toplevel)/$SCRIPT"