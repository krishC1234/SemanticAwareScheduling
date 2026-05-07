#!/bin/bash
# Evaluate scheduling at different contention levels (jobs/GPUs ratio).
# Runs 3 iterations with 3, 5, and 7 jobs competing for 8 GPUs.
# Delay=0 so all jobs arrive together — scheduler must split GPUs.
#
# Usage:
#   bash run_contention_eval.sh              # default port 9321
#   bash run_contention_eval.sh 9100         # custom port

PORT=${1:-9321}
JOB_COUNTS=(3 5 7)
DELAYS=(0 15 30)

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
echo "  CONTENTION EVALUATION"
echo "  Job counts: ${JOB_COUNTS[*]}"
echo "  Delays: ${DELAYS[*]}s"
echo "  Port: $PORT"
echo "============================================================"

for N_JOBS in "${JOB_COUNTS[@]}"; do
  for DELAY in "${DELAYS[@]}"; do
    SEED=$RANDOM
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_DIR="evaluation/test_results/${N_JOBS}jobs_${DELAY}s_${TIMESTAMP}"
    mkdir -p "$RUN_DIR"

    echo ""
    echo "============================================================"
    echo "  ITERATION: ${N_JOBS} jobs, delay=${DELAY}s, seed=${SEED}"
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
        echo "ERROR: scheduler failed to start for ${N_JOBS} jobs/${DELAY}s. Skipping."
        cat logs/scheduler_stdout.log
        continue
    fi
    echo "Scheduler started (PID $SCHEDULER_PID)"

    # --- Scheduler eval ---
    echo ""
    echo "############################################################"
    echo "  [${N_JOBS} jobs, ${DELAY}s] Running: scheduler eval"
    echo "############################################################"
    python3 -m evaluation.test_scripts.scheulder_eval \
        --seed "$SEED" --max-delay "$DELAY" --port "$PORT" \
        --run-dir "$RUN_DIR" --num-jobs "$N_JOBS"

    echo "Stopping scheduler server (PID $SCHEDULER_PID)..."
    kill -TERM "$SCHEDULER_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$SCHEDULER_PID" 2>/dev/null || true
    wait "$SCHEDULER_PID" 2>/dev/null || true
    echo "Scheduler server stopped."

    # --- Greedy ---
    echo ""
    echo "############################################################"
    echo "  [${N_JOBS} jobs, ${DELAY}s] Running: greedy baseline"
    echo "############################################################"
    python3 -m evaluation.test_scripts.greedy_baseline \
        --seed "$SEED" --max-delay "$DELAY" \
        --run-dir "$RUN_DIR" --num-jobs "$N_JOBS"

    # --- Polite ---
    echo ""
    echo "############################################################"
    echo "  [${N_JOBS} jobs, ${DELAY}s] Running: polite baseline"
    echo "############################################################"
    python3 -m evaluation.test_scripts.polite_baseline \
        --seed "$SEED" --max-delay "$DELAY" \
        --run-dir "$RUN_DIR" --num-jobs "$N_JOBS"

    # --- FCFS Split ---
    echo ""
    echo "############################################################"
    echo "  [${N_JOBS} jobs, ${DELAY}s] Running: FCFS split baseline"
    echo "############################################################"
    python3 -m evaluation.test_scripts.fcfs_split_baseline \
        --seed "$SEED" --max-delay "$DELAY" \
        --run-dir "$RUN_DIR" --num-jobs "$N_JOBS"

    # --- Size-Aware ---
    echo ""
    echo "############################################################"
    echo "  [${N_JOBS} jobs, ${DELAY}s] Running: size-aware baseline"
    echo "############################################################"
    python3 -m evaluation.test_scripts.size_aware_baseline \
        --seed "$SEED" --max-delay "$DELAY" \
        --run-dir "$RUN_DIR" --num-jobs "$N_JOBS"

    echo ""
    echo "  Iteration ${N_JOBS} jobs / ${DELAY}s delay complete."
  done
done

echo ""
echo "============================================================"
echo "  ALL ITERATIONS COMPLETE"
echo "  Results saved to evaluation/test_results/"
echo "============================================================"
