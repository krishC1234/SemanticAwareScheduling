#!/bin/bash
# Run all evaluation scripts sequentially with the same seed.
# Baselines run first (no server needed), then the scheduler eval last.
#
# Usage:
#   bash run_eval.sh                  # defaults: seed=42, max_delay=600
#   bash run_eval.sh 42 120           # seed=42, max_delay=120
#   bash run_eval.sh 42 60 9321       # seed=42, max_delay=60, port=9321

SEED=${1:-42}
MAX_DELAY=${2:-600}
PORT=${3:-9321}

# Ensure we're in the project root (so .env and modules resolve correctly)
cd "$(dirname "$0")/.."

# Load .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "============================================================"
echo "  FULL EVALUATION SUITE"
echo "  seed=$SEED  max_delay=${MAX_DELAY}s  port=$PORT"
echo "============================================================"

# --- 1. Baselines (no server needed) ---
echo ""
echo "############################################################"
echo "  Running: greedy baseline"
echo "############################################################"
python3 -m evaluation.test_scripts.greedy_baseline \
    --seed "$SEED" --max-delay "$MAX_DELAY"

echo ""
echo "############################################################"
echo "  Running: polite baseline"
echo "############################################################"
python3 -m evaluation.test_scripts.polite_baseline \
    --seed "$SEED" --max-delay "$MAX_DELAY"

echo ""
echo "############################################################"
echo "  Running: FCFS split baseline"
echo "############################################################"
python3 -m evaluation.test_scripts.fcfs_split_baseline \
    --seed "$SEED" --max-delay "$MAX_DELAY"

# --- 2. Scheduler eval (needs the server) ---
echo ""
echo "############################################################"
echo "  Starting scheduler server on port $PORT..."
echo "############################################################"

mkdir -p logs
# Kill any leftover scheduler processes and free the port
pkill -9 -f "scheduler.main" 2>/dev/null || true
fuser -k "$PORT"/tcp 2>/dev/null || true
fuser -k 9321/tcp 2>/dev/null || true
sleep 5
python3 -m scheduler.main --port "$PORT" > logs/scheduler_stdout.log 2>&1 &
SCHEDULER_PID=$!
sleep 5

if ! kill -0 "$SCHEDULER_PID" 2>/dev/null; then
    echo "ERROR: scheduler failed to start. Server output:"
    cat logs/scheduler_stdout.log
    exit 1
fi
echo "Scheduler server started (PID $SCHEDULER_PID)"

echo ""
echo "############################################################"
echo "  Running: scheduler eval"
echo "############################################################"
python3 -m evaluation.test_scripts.scheulder_eval \
    --seed "$SEED" --max-delay "$MAX_DELAY" --port "$PORT"

echo "Stopping scheduler server (PID $SCHEDULER_PID)..."
kill -TERM "$SCHEDULER_PID" 2>/dev/null || true
sleep 2
kill -9 "$SCHEDULER_PID" 2>/dev/null || true
wait "$SCHEDULER_PID" 2>/dev/null || true
echo "Scheduler server stopped."

echo ""
echo "============================================================"
echo "  ALL EVALUATIONS COMPLETE"
echo "  Results saved to evaluation/test_results/"
echo "============================================================"
