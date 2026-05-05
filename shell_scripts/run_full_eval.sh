#!/bin/bash
# Run 3 iterations of the full evaluation suite with different delays.
# Each iteration uses a random seed (shared across all 4 scripts).
#
# Delays: 5s, 30s, 90s
# Results go to evaluation/test_results/<delay>s_<timestamp>/
#
# Usage:
#   bash run_full_eval.sh              # default port 9321
#   bash run_full_eval.sh 9100         # custom port

PORT=${1:-9321}
DELAYS=(0 15 60)

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Load .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Verify scheduler starts before running anything
echo "Verifying scheduler can start..."
mkdir -p logs
pkill -9 -f "scheduler.main" 2>/dev/null || true
fuser -k "$PORT"/tcp 2>/dev/null || true
fuser -k 9321/tcp 2>/dev/null || true
sleep 3
python3 -m scheduler.main --port "$PORT" > logs/scheduler_stdout.log 2>&1 &
TEST_PID=$!
sleep 5
if ! kill -0 "$TEST_PID" 2>/dev/null; then
    echo "ERROR: scheduler failed to start. Server output:"
    cat logs/scheduler_stdout.log
    exit 1
fi
echo "Scheduler verified. Stopping test instance..."
kill -TERM "$TEST_PID" 2>/dev/null || true
sleep 2
kill -9 "$TEST_PID" 2>/dev/null || true
wait "$TEST_PID" 2>/dev/null || true

echo "============================================================"
echo "  FULL MULTI-DELAY EVALUATION"
echo "  Delays: ${DELAYS[*]}s"
echo "  Port: $PORT"
echo "============================================================"

for DELAY in "${DELAYS[@]}"; do
    SEED=$RANDOM
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_DIR="evaluation/test_results/${DELAY}s_${TIMESTAMP}"
    mkdir -p "$RUN_DIR"

    echo ""
    echo "============================================================"
    echo "  ITERATION: delay=${DELAY}s  seed=${SEED}"
    echo "  Output: $RUN_DIR"
    echo "============================================================"

    # --- Start scheduler for this iteration ---
    pkill -9 -f "scheduler.main" 2>/dev/null || true
    fuser -k "$PORT"/tcp 2>/dev/null || true
    sleep 3
    python3 -m scheduler.main --port "$PORT" > logs/scheduler_stdout.log 2>&1 &
    SCHEDULER_PID=$!
    sleep 5

    if ! kill -0 "$SCHEDULER_PID" 2>/dev/null; then
        echo "ERROR: scheduler failed to start for delay=${DELAY}s. Skipping."
        cat logs/scheduler_stdout.log
        continue
    fi
    echo "Scheduler started (PID $SCHEDULER_PID)"

    # --- Baselines ---
    # --- Scheduler eval ---
    echo ""
    echo "############################################################"
    echo "  [${DELAY}s] Running: scheduler eval"
    echo "############################################################"
    python3 -m evaluation.test_scripts.scheulder_eval \
        --seed "$SEED" --max-delay "$DELAY" --port "$PORT" --run-dir "$RUN_DIR"

    echo "Stopping scheduler server (PID $SCHEDULER_PID)..."
    kill -TERM "$SCHEDULER_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$SCHEDULER_PID" 2>/dev/null || true
    wait "$SCHEDULER_PID" 2>/dev/null || true
    echo "Scheduler server stopped."
    
    echo ""
    echo "############################################################"
    echo "  [${DELAY}s] Running: greedy baseline"
    echo "############################################################"
    python3 -m evaluation.test_scripts.greedy_baseline \
        --seed "$SEED" --max-delay "$DELAY" --run-dir "$RUN_DIR"

    echo ""
    echo "############################################################"
    echo "  [${DELAY}s] Running: polite baseline"
    echo "############################################################"
    python3 -m evaluation.test_scripts.polite_baseline \
        --seed "$SEED" --max-delay "$DELAY" --run-dir "$RUN_DIR"

    echo ""
    echo "############################################################"
    echo "  [${DELAY}s] Running: FCFS split baseline"
    echo "############################################################"
    python3 -m evaluation.test_scripts.fcfs_split_baseline \
        --seed "$SEED" --max-delay "$DELAY" --run-dir "$RUN_DIR"

    echo ""
    echo "############################################################"
    echo "  [${DELAY}s] Running: size-aware baseline"
    echo "############################################################"
    python3 -m evaluation.test_scripts.size_aware_baseline \
        --seed "$SEED" --max-delay "$DELAY" --run-dir "$RUN_DIR"

    echo ""
    echo "  Iteration delay=${DELAY}s complete."
done

echo ""
echo "============================================================"
echo "  ALL ITERATIONS COMPLETE"
echo "  Results saved to evaluation/test_results/"
echo "============================================================"