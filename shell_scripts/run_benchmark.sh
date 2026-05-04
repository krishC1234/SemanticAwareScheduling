#!/bin/bash
set -e

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Configuration
JOBS_DIR="train_data/jobs"
RESULTS_DIR="train_data/results"
CSV_FILE="train_data/benchmark.csv"
GPU_COUNTS=(1 2 4 8)

mkdir -p "$RESULTS_DIR"

# Check if CSV exists - create with header if not, otherwise append
if [ -f "$CSV_FILE" ]; then
    echo "Found existing $CSV_FILE - will append results"
    EXISTING_ROWS=$(wc -l < "$CSV_FILE")
    echo "  Current rows: $EXISTING_ROWS"
else
    echo "Creating new $CSV_FILE with header"
    echo "model,config,batch_size,param_count,gpu_count,total_time_sec,avg_throughput,peak_vram_mb,avg_sm_util_pct,avg_mem_bw_pct" > "$CSV_FILE"
fi

# Get list of models (top-level directories)
MODELS=$(find "$JOBS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

for MODEL_DIR in $MODELS; do
    MODEL=$(basename "$MODEL_DIR")
    echo ""
    echo "========================================"
    echo "MODEL: $MODEL"
    echo "========================================"
    
    # Get all configs for this model
    CONFIGS=$(find "$MODEL_DIR" -name "*.py" | sort)
    
    for CONFIG_PATH in $CONFIGS; do
        CONFIG=$(basename "$CONFIG_PATH" .py)
        
        for N_GPU in "${GPU_COUNTS[@]}"; do
            LOG_DIR="$RESULTS_DIR/$MODEL/$CONFIG"
            mkdir -p "$LOG_DIR"
            LOG_FILE="$LOG_DIR/gpu${N_GPU}.log"
            
            echo ""
            echo "--- $MODEL / $CONFIG / $N_GPU GPUs ---"
            
            # Start GPU monitoring in background
            nvidia-smi --query-gpu=memory.used,utilization.gpu,utilization.memory \
                --format=csv,noheader,nounits -l 1 > "$LOG_DIR/gpu_stats_${N_GPU}.csv" 2>/dev/null &
            MONITOR_PID=$!
            
            # Run training
            if torchrun --standalone --nproc_per_node=$N_GPU "$CONFIG_PATH" > "$LOG_FILE" 2>&1; then
                echo "  ✓ Completed successfully"
            else
                echo "  ✗ Failed (see $LOG_FILE)"
                kill $MONITOR_PID 2>/dev/null || true
                continue
            fi
            
            # Stop monitoring
            kill $MONITOR_PID 2>/dev/null || true
            sleep 1
            
            # Parse results from JSON block
            if grep -q "###RESULTS###" "$LOG_FILE"; then
                RESULTS_JSON=$(sed -n '/###RESULTS###/,/###END_RESULTS###/p' "$LOG_FILE" | grep -v "###" | head -1)
                BATCH_SIZE=$(echo "$RESULTS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['batch_size'])" 2>/dev/null || echo "0")
                PARAM_COUNT=$(echo "$RESULTS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['param_count'])" 2>/dev/null || echo "0")
                TOTAL_TIME=$(echo "$RESULTS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['total_time_sec'])" 2>/dev/null || echo "0")
                AVG_THROUGHPUT=$(echo "$RESULTS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['avg_throughput'])" 2>/dev/null || echo "0")
            else
                # Fallback to grep parsing
                BATCH_SIZE=$(grep -oP 'Batch:\K[0-9]+' "$LOG_FILE" | head -1 || echo "0")
                PARAM_COUNT=$(grep -oP 'Params:\K[0-9,]+' "$LOG_FILE" | head -1 | tr -d ',' || echo "0")
                TOTAL_TIME=$(grep -oP 'Total time:\K[0-9.]+' "$LOG_FILE" | head -1 || echo "0")
                AVG_THROUGHPUT=$(grep -oP 'Avg throughput:\K[0-9.]+' "$LOG_FILE" | head -1 || echo "0")
            fi
            
            # Parse GPU stats
            if [ -f "$LOG_DIR/gpu_stats_${N_GPU}.csv" ] && [ -s "$LOG_DIR/gpu_stats_${N_GPU}.csv" ]; then
                PEAK_VRAM=$(awk -F',' '{print $1}' "$LOG_DIR/gpu_stats_${N_GPU}.csv" | sort -n | tail -1 || echo "0")
                AVG_SM=$(awk -F',' '{sum+=$2; n++} END {if(n>0) printf "%.1f", sum/n; else print "0"}' "$LOG_DIR/gpu_stats_${N_GPU}.csv")
                AVG_MEM_BW=$(awk -F',' '{sum+=$3; n++} END {if(n>0) printf "%.1f", sum/n; else print "0"}' "$LOG_DIR/gpu_stats_${N_GPU}.csv")
            else
                PEAK_VRAM="0"
                AVG_SM="0"
                AVG_MEM_BW="0"
            fi
            
            # Append to CSV
            echo "$MODEL,$CONFIG,$BATCH_SIZE,$PARAM_COUNT,$N_GPU,$TOTAL_TIME,$AVG_THROUGHPUT,$PEAK_VRAM,$AVG_SM,$AVG_MEM_BW" >> "$CSV_FILE"
            
            echo "  Time: ${TOTAL_TIME}s | Throughput: ${AVG_THROUGHPUT} samples/sec | VRAM: ${PEAK_VRAM}MB"
        done
    done
    
    echo "✓ $MODEL complete"
done

echo ""
echo "========================================"
echo "BENCHMARK COMPLETE"
echo "========================================"
echo "Results: $CSV_FILE"
FINAL_ROWS=$(wc -l < "$CSV_FILE")
echo "Total rows: $FINAL_ROWS"
