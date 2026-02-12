# vramPadding Calibration Runbook

> Find minimum `percentage` and `maxReserved` that prevent CUDA SIGABRT on RTX 3090 Ti.
> For [node-llama-cpp#551](https://github.com/withcatai/node-llama-cpp/issues/551#issuecomment-3887426735).

## Scripts

| Script | Purpose |
|--------|---------|
| `test/env-dump.sh` | Captures all environment metadata in one shot |
| `test/run-padding-test.sh` | Runs one test, logs GPU state + output + exit code |
| `test/run-step.sh` | Runs N iterations of one config, prints verdict (PASS/FAIL/FLAKY) |

All output goes to stdout. Append `>> test/padding-results.log 2>&1` to build your log.

---

## Setup (Terminal 1 — Ballast)

```bash
cd ~/dev/qmd

# Kill anything on the GPU
pkill ollama 2>/dev/null; sleep 2

# Compile ballast
nvcc -o test/vram-ballast test/vram-ballast.cu

# Smoke test (Ctrl+C after BALLAST_READY)
./test/vram-ballast 100
```

## Setup (Terminal 2 — Tests)

```bash
cd ~/dev/qmd

# Dump environment (run once)
./test/env-dump.sh | tee test/padding-results.log

# Make sure the model is cached (no ballast yet)
bun test/vram-padding-calibration.ts 0.06 1
```

---

## Step 0: Baseline Crash

Confirm the crash still happens with defaults. **Single run — expect SIGABRT.**

**Terminal 1:**
```bash
./test/vram-ballast 24064
# Wait for BALLAST_READY=1
```

**Terminal 2:**
```bash
./test/run-step.sh "step0-baseline" 0.06 1 1 >> test/padding-results.log 2>&1
tail -20 test/padding-results.log
```

**Expected:** `verdict: FAIL (0/1)`, exit code 134.

If it passes: crash window has shifted. Stop and re-characterize with these ballast levels first:
```bash
# Re-run at different ballast levels to find the new crash point
for MB in 22528 23040 23552 24064; do
  # Restart ballast in Terminal 1 at each level
  ./test/run-step.sh "rechar-${MB}MB" 0.06 1 1 >> test/padding-results.log 2>&1
done
```

---

## Step 1: Verify High Padding Works

**Terminal 1:** Keep ballast at 24064 MB.

**Terminal 2:**
```bash
./test/run-step.sh "step1-high-padding" 0.2 5 >> test/padding-results.log 2>&1
tail -20 test/padding-results.log
```

**Expected:** `verdict: PASS (3/3)`.

If any run fails: padding can't fix this. Stop and post to the issue.

---

## Step 2: Find Minimum Percentage

`maxReserved=5 GB` throughout. Effective padding = `24 * percentage` (cap never kicks in below 0.208).

**Terminal 1:** Keep ballast at 24064 MB.

**Terminal 2 — Coarse scan (1 run each, find the transition):**
```bash
for PCT in 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20; do
  ./test/run-step.sh "step2-pct-$PCT" "$PCT" 5 1 >> test/padding-results.log 2>&1
done
tail -40 test/padding-results.log
```

**Then validate the first passing value with 3 runs:**
```bash
# Replace 0.XX with whatever passed first in the scan above
./test/run-step.sh "step2-pct-0.XX-validate" 0.XX 5 >> test/padding-results.log 2>&1
tail -20 test/padding-results.log
```

**Record:** First passing percentage = `__________`

---

## Step 3: Find Minimum maxReserved

`percentage=0.4` throughout. Effective padding = maxReserved exactly (since `24*0.4=9.6 > maxReserved`).

**Terminal 1:** Keep ballast at 24064 MB.

**Terminal 2 — Coarse scan (0.5 GB steps):**
```bash
for MAX in 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0; do
  ./test/run-step.sh "step3-max-$MAX" 0.4 "$MAX" 1 >> test/padding-results.log 2>&1
done
tail -60 test/padding-results.log
```

**Fine scan between the transition (0.1 GB steps):**
```bash
# Replace X.X and Y.Y with last-fail and first-pass from coarse scan
# Example: if 1.5 failed and 2.0 passed, scan 1.6-1.9
for MAX in X.6 X.7 X.8 X.9; do
  ./test/run-step.sh "step3-fine-$MAX" 0.4 "$MAX" 1 >> test/padding-results.log 2>&1
done
tail -30 test/padding-results.log
```

**Validate first passing value with 3 runs:**
```bash
./test/run-step.sh "step3-max-X.X-validate" 0.4 X.X >> test/padding-results.log 2>&1
tail -20 test/padding-results.log
```

**Record:** First passing maxReserved = `__________` GB

---

## Step 4: Combined Validation

Use your results from Steps 2 and 3. Test across multiple ballast levels.

**Terminal 2 (restart ballast in Terminal 1 for each level):**
```bash
# Replace PCT and MAX with your values from Steps 2 and 3

# 4a: Crash point (restart ballast to 24064 in Terminal 1)
./test/run-step.sh "step4-ballast-23.5G" PCT MAX >> test/padding-results.log 2>&1

# 4b: Moderate pressure (restart ballast to 23552 in Terminal 1)
./test/run-step.sh "step4-ballast-23G" PCT MAX >> test/padding-results.log 2>&1

# 4c: Light pressure (restart ballast to 22528 in Terminal 1)
./test/run-step.sh "step4-ballast-22G" PCT MAX >> test/padding-results.log 2>&1

# 4d: Medium pressure (restart ballast to 16384 in Terminal 1)
./test/run-step.sh "step4-ballast-16G" PCT MAX >> test/padding-results.log 2>&1

# 4e: No pressure (kill ballast in Terminal 1)
./test/run-step.sh "step4-no-ballast" PCT MAX >> test/padding-results.log 2>&1

tail -60 test/padding-results.log
```

**Expected:** All PASS. If the crash point fails, bump the failing param by one increment and re-test.

---

## Step 5: Additional Models (Optional)

This requires modifying `test/vram-padding-calibration.ts` to load different model types.
If skipping, note it in the issue comment. The embed model is the one that originally crashed.

---

## After Testing

Your full log is at `test/padding-results.log`. Hand it to Claude:

> "Here's my raw log from the vramPadding calibration tests. Turn this into a report for the node-llama-cpp#551 issue comment."

---

## Quick Reference

**Exit codes:**

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | JS error |
| 2 | Bad CLI args |
| 3 | Timeout (>60s) |
| 134 | SIGABRT (the crash) |
| 137 | OOM kill |

**Ballast sizes:**

| GB | MB | Command | Approx free VRAM |
|----|----|---------|-----------------|
| 23.5 | 24064 | `./test/vram-ballast 24064` | ~356 MiB |
| 23.0 | 23552 | `./test/vram-ballast 23552` | ~856 MiB |
| 22.0 | 22528 | `./test/vram-ballast 22528` | ~1.9 GB |
| 16.0 | 16384 | `./test/vram-ballast 16384` | ~8 GB |

**Effective padding math:**

| Step | Percentage | maxReserved | Effective padding |
|------|-----------|-------------|-------------------|
| Step 2 | varies | 5 GB | `24 * percentage` (cap never triggers) |
| Step 3 | 0.4 | varies | `maxReserved` exactly (% never caps) |
| Step 4 | combined | combined | `min(24 * percentage, maxReserved)` |
