#!/bin/bash

COMMIT=$1
SCRIPT=$2

TMP=/tmp/git_exp_$COMMIT

mkdir -p $TMP

git archive $COMMIT | tar -x -C $TMP

export PYTHONPATH=$TMP

echo "Using commit: $COMMIT"
echo "Code path: $TMP"

bash $SCRIPT