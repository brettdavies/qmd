#!/usr/bin/env bash
# sweep-ballast-levels.sh — Find the crash window by testing at different ballast levels
#
# Usage: ./test/sweep-ballast-levels.sh [percentage] [maxReservedGB]
# Defaults to 0.06 and 1 (library defaults)

PCT="${1:-0.06}"
MAX="${2:-1}"

echo "==== BALLAST LEVEL SWEEP ===="
echo "timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "padding config: percentage=$PCT maxReservedGB=$MAX"
echo ""

# Ballast levels in MiB — targeting different free VRAM amounts
# Total VRAM ~23856 MiB free on clean GPU (after CUDA context overhead)
for BALLAST_MB in 21856 22356 22856 23056 23256 23456 23556; do
  TARGET_FREE=$((23856 - BALLAST_MB))
  echo "---- ballast=${BALLAST_MB} MiB (target ~${TARGET_FREE} MiB free) ----"

  # Start ballast, capture output to temp file
  TMPOUT=$(mktemp)
  ./test/vram-ballast "$BALLAST_MB" > "$TMPOUT" 2>&1 &
  BPID=$!

  # Wait up to 10 seconds for BALLAST_READY
  READY=0
  for i in $(seq 1 10); do
    if grep -q "BALLAST_READY=1" "$TMPOUT" 2>/dev/null; then
      READY=1
      break
    fi
    if ! kill -0 "$BPID" 2>/dev/null; then
      break  # process died
    fi
    sleep 1
  done

  if [ "$READY" -ne 1 ]; then
    echo "  BALLAST FAILED (could not allocate)"
    cat "$TMPOUT" 2>/dev/null | head -5
    kill -9 "$BPID" 2>/dev/null
    wait "$BPID" 2>/dev/null
    rm -f "$TMPOUT"
    # Wait for VRAM recovery
    sleep 3
    continue
  fi

  # Read actual free VRAM from ballast output
  ACTUAL_FREE=$(grep "VRAM after" "$TMPOUT" | grep -oP '[\d.]+(?= MiB free)' || echo "?")
  echo "  actual_free: ${ACTUAL_FREE} MiB"
  echo "  nvidia_smi_free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null) MiB"

  # Run test
  set +e
  OUTPUT=$(bun test/vram-padding-calibration.ts "$PCT" "$MAX" 2>&1)
  CODE=$?
  set -e

  # Extract key values
  GPU_LAYERS=$(echo "$OUTPUT" | grep -oP 'gpuLayers: \K\d+' || echo "?")
  PADDING=$(echo "$OUTPUT" | grep -oP 'padding:\s+\K[\d.]+(?= GB)' || echo "?")

  case $CODE in
    0)   MEANING="SUCCESS" ;;
    134) MEANING="SIGABRT" ;;
    137) MEANING="OOM_KILL" ;;
    *)   MEANING="ERROR" ;;
  esac

  echo "  gpuLayers=$GPU_LAYERS padding=${PADDING}GB exit=$CODE ($MEANING)"

  # Kill ballast cleanly
  kill "$BPID" 2>/dev/null
  wait "$BPID" 2>/dev/null
  rm -f "$TMPOUT"

  # Wait for VRAM recovery — poll until free > 23000 MiB
  for i in $(seq 1 15); do
    FREE_NOW=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
    if [ "$FREE_NOW" -gt 23000 ] 2>/dev/null; then
      break
    fi
    sleep 1
  done

  FREE_AFTER=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
  if [ "$FREE_AFTER" -lt 23000 ] 2>/dev/null; then
    echo "  WARNING: VRAM not recovered (${FREE_AFTER} MiB free). Stopping."
    exit 1
  fi

  # After SIGABRT, extra recovery time
  if [ "$CODE" -eq 134 ] || [ "$CODE" -eq 137 ]; then
    echo "  (post-crash recovery wait 5s)"
    sleep 5
  fi

  echo ""
done

echo "==== END SWEEP ===="
