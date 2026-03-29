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

# bash $SCRIPT $COMMIT


#!/bin/bash

#!/bin/bash

COMMIT=$1
SCRIPT=$2

if [ -z "$COMMIT" ] || [ -z "$SCRIPT" ]; then
  echo "Usage: ./run_commit.sh <commit_id> <script>"
  exit 1
fi

ROOT=$(git rev-parse --show-toplevel)

mkdir -p "$ROOT/logs"

# 创建唯一目录
TMP_DIR=$(mktemp -d "$ROOT/logs/code_${COMMIT}_XXXX")
trap "rm -rf '$TMP_DIR'" EXIT

echo "Export commit $COMMIT to $TMP_DIR"

# 导出 commit
git archive "$COMMIT" | tar -x -C "$TMP_DIR"

echo "-----------------------------------"
echo "Using commit: $COMMIT"
echo "Code directory: $TMP_DIR"
echo "Script: $SCRIPT"
echo "-----------------------------------"

cd "$TMP_DIR" || exit 1

sleep 5

export PYTHONPATH="$TMP_DIR:$PYTHONPATH"

bash "$ROOT/$SCRIPT" "$COMMIT"