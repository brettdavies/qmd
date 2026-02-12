#!/usr/bin/env bash
# run-step.sh — Run a batch of tests for one step with automatic 3-run validation
#
# Usage:
#   ./test/run-step.sh <step-label> <percentage> <maxReservedGB> [runs=3]
#
# Examples:
#   ./test/run-step.sh "step0-baseline" 0.06 1 1        # single run (crash test)
#   ./test/run-step.sh "step1-high" 0.2 5               # 3 runs (default)
#   ./test/run-step.sh "step2-pct-0.08" 0.08 5          # 3 runs
#   ./test/run-step.sh "step3-max-1.5" 0.4 1.5          # 3 runs
#
# Runs N iterations, waits 5s between runs if previous was SIGABRT,
# prints a summary verdict at the end.
#
# Redirect to build your log:
#   ./test/run-step.sh "step1" 0.2 5 >> test/padding-results.log 2>&1

LABEL="${1:?Usage: $0 <label> <percentage> <maxReservedGB> [runs=3]}"
PCT="${2:?Usage: $0 <label> <percentage> <maxReservedGB> [runs=3]}"
MAX="${3:?Usage: $0 <label> <percentage> <maxReservedGB> [runs=3]}"
RUNS="${4:-3}"

PASS=0
FAIL=0
FLAKY=0
CODES=""

echo "==== STEP: $LABEL ===="
echo "config: percentage=$PCT maxReservedGB=$MAX runs=$RUNS"
echo "started: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

for i in $(seq 1 "$RUNS"); do
  echo "---- run $i/$RUNS ----"

  # Pre-run GPU state
  echo "gpu_free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo '?') MiB"

  # Run test
  set +e
  OUTPUT=$(bun test/vram-padding-calibration.ts "$PCT" "$MAX" 2>&1)
  CODE=$?
  set -e

  # Decode
  case $CODE in
    0)   MEANING="SUCCESS"; PASS=$((PASS + 1)) ;;
    134) MEANING="SIGABRT"; FAIL=$((FAIL + 1)) ;;
    137) MEANING="OOM_KILL"; FAIL=$((FAIL + 1)) ;;
    *)   MEANING="ERROR"; FAIL=$((FAIL + 1)) ;;
  esac

  CODES="$CODES $CODE"

  # Print condensed output (just the important lines)
  echo "$OUTPUT" | grep -E '(totalVram|padding|gpuLayers|modelSize|Success|Error|TIMEOUT)' || true
  echo "exit: $CODE ($MEANING)"

  # After SIGABRT, wait for GPU to recover
  if [ "$CODE" -eq 134 ] || [ "$CODE" -eq 137 ]; then
    echo "  (waiting 5s for GPU recovery...)"
    sleep 5
    echo "  gpu_free_after_recovery: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo '?') MiB"
  fi

  echo ""
done

# --- Verdict ---
if [ "$PASS" -eq "$RUNS" ]; then
  VERDICT="PASS ($PASS/$RUNS)"
elif [ "$PASS" -gt 0 ]; then
  VERDICT="FLAKY ($PASS/$RUNS pass)"
else
  VERDICT="FAIL (0/$RUNS)"
fi

echo "---- verdict ----"
echo "label: $LABEL"
echo "config: percentage=$PCT maxReservedGB=$MAX"
echo "exit_codes:$CODES"
echo "verdict: $VERDICT"
echo "ended: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==== END STEP ===="
echo ""
