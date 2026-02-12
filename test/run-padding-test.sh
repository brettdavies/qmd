#!/usr/bin/env bash
# run-padding-test.sh — Run a single vramPadding calibration test and log everything
#
# Usage:
#   ./test/run-padding-test.sh <percentage> <maxReservedGB> [label]
#
# Examples:
#   ./test/run-padding-test.sh 0.06 1 "step0-baseline"
#   ./test/run-padding-test.sh 0.2 5 "step1-high-padding"
#   ./test/run-padding-test.sh 0.12 5 "step2-pct-0.12-run1"
#
# Outputs structured log entry to stdout.
# Redirect to build your log:
#   ./test/run-padding-test.sh 0.06 1 "step0" >> test/padding-results.log
#
# Or use tee for live viewing + log:
#   ./test/run-padding-test.sh 0.06 1 "step0" 2>&1 | tee -a test/padding-results.log

PCT="${1:?Usage: $0 <percentage> <maxReservedGB> [label]}"
MAX="${2:?Usage: $0 <percentage> <maxReservedGB> [label]}"
LABEL="${3:-unlabeled}"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# --- Pre-test GPU snapshot ---
GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.free,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>&1 || echo "NVIDIA-SMI FAILED")
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,name,used_gpu_memory --format=csv,noheader 2>&1 || echo "none")

# --- Run test, capture output AND exit code ---
set +e
TEST_OUTPUT=$(bun test/vram-padding-calibration.ts "$PCT" "$MAX" 2>&1)
EXIT_CODE=$?
set -e

# --- Decode exit code ---
case $EXIT_CODE in
  0)   EXIT_MEANING="SUCCESS" ;;
  1)   EXIT_MEANING="JS_ERROR" ;;
  2)   EXIT_MEANING="BAD_ARGS" ;;
  3)   EXIT_MEANING="TIMEOUT" ;;
  134) EXIT_MEANING="SIGABRT" ;;
  137) EXIT_MEANING="OOM_KILL" ;;
  *)   EXIT_MEANING="UNKNOWN" ;;
esac

# --- Emit structured log entry ---
cat <<ENTRY
==== TEST_RUN ====
timestamp: $TIMESTAMP
label: $LABEL
config: percentage=$PCT maxReservedGB=$MAX
exit_code: $EXIT_CODE ($EXIT_MEANING)
gpu_pre: $GPU_INFO
gpu_procs: $GPU_PROCS
---- output ----
$TEST_OUTPUT
---- end ----

ENTRY
