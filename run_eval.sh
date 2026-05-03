#!/bin/bash
# Run all evaluation scripts sequentially with the same seed.
# Starts the scheduler server automatically for the scheduler eval.
#
# Usage:
#   bash run_eval.sh                  # defaults: seed=42, max_delay=600
#   bash run_eval.sh 42 120           # seed=42, max_delay=120
#   bash run_eval.sh 42 60 9321       # seed=42, max_delay=60, port=9321

SEED=${1:-42}
MAX_DELAY=${2:-600}
PORT=${3:-9321}

# Ensure we're in the project root (so .env and modules resolve correctly)
cd "$(dirname "$0")"

# Load .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "============================================================"
echo "  FULL EVALUATION SUITE"
echo "  seed=$SEED  max_delay=${MAX_DELAY}s  port=$PORT"
echo "============================================================"

# --- 1. Scheduler eval (needs the server) ---
echo ""
echo "############################################################"
echo "  Starting scheduler server on port $PORT..."
echo "############################################################"

python3 -m scheduler.main --port "$PORT" &
SCHEDULER_PID=$!
sleep 3

if ! kill -0 "$SCHEDULER_PID" 2>/dev/null; then
    echo "ERROR: scheduler failed to start"
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
kill -INT "$SCHEDULER_PID" 2>/dev/null
wait "$SCHEDULER_PID" 2>/dev/null
echo "Scheduler server stopped."

# --- 2. Baselines (no server needed) ---
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
echo "  Running: equal share baseline"
echo "############################################################"
python3 -m evaluation.test_scripts.equal_share_baseline \
    --seed "$SEED" --max-delay "$MAX_DELAY"

echo ""
echo "============================================================"
echo "  ALL EVALUATIONS COMPLETE"
echo "  Results saved to evaluation/test_results/"
echo "============================================================"