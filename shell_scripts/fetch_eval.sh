#!/bin/bash
# Fetch eval results and logs from remote machine.
#
# Usage:
#   bash fetch_eval.sh

SSH_PORT=36142
SSH_USER=root
SSH_HOST=209.146.116.50
REMOTE_DIR=/workspace/SemanticAwareScheduling

rm -rf evaluation/test_results/*
rm -rf logs/*

RUN_DIR="./evaluation/test_results/run_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RUN_DIR"

echo "Fetching into $RUN_DIR ..."

echo "Fetching eval output log..."
scp -P "$SSH_PORT" "$SSH_USER@$SSH_HOST:$REMOTE_DIR/eval_output.log" "$RUN_DIR/"

echo "Fetching eval results..."
scp -P "$SSH_PORT" -r "$SSH_USER@$SSH_HOST:$REMOTE_DIR/evaluation/test_results/*" "$RUN_DIR/"

echo "Fetching all logs..."
mkdir -p logs
scp -r -P "$SSH_PORT" "$SSH_USER@$SSH_HOST:$REMOTE_DIR/logs/" "."

echo "Done. Results in $RUN_DIR"
