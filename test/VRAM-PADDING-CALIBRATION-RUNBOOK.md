# vramPadding Calibration Runbook

> **Purpose:** Find minimum `percentage` and `maxReserved` values that prevent CUDA SIGABRT
> **GPU:** RTX 3090 Ti (24 GB)
> **Crash point:** 23.5 GB ballast (~356 MiB free) with default padding (6% / 1 GB cap)
> **Reference:** [node-llama-cpp#551](https://github.com/withcatai/node-llama-cpp/issues/551#issuecomment-3887426735)

---

## Pre-Flight Checklist

Complete every item before starting tests.

- [ ] **1. Kill GPU consumers:** Stop ollama, other CUDA processes
  ```bash
  # Check what's using the GPU
  nvidia-smi

  # Kill ollama if running
  systemctl stop ollama  # or: pkill ollama
  ```

- [ ] **2. Verify clean GPU state** (should show ~24 GB free, no processes)
  ```bash
  nvidia-smi --query-gpu=memory.free,memory.total,memory.used --format=csv,noheader
  ```
  Write result here: `__________ MiB free / __________ MiB total`

- [ ] **3. Compile the ballast tool**
  ```bash
  cd ~/dev/qmd
  nvcc -o test/vram-ballast test/vram-ballast.cu
  ```
  Confirm: compiled without errors? [ yes / no ]

- [ ] **4. Smoke-test the ballast** (allocate 100 MB, then kill it)
  ```bash
  ./test/vram-ballast 100
  # Should print VRAM before/after, then BALLAST_READY=1
  # Press Ctrl+C to release
  ```
  Confirm: BALLAST_READY printed and Ctrl+C released cleanly? [ yes / no ]

- [ ] **5. Verify model is cached** (avoid download during test)
  ```bash
  ls -lh ~/.cache/qmd/models/ggml-org--embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf
  ```
  If missing, run once without ballast:
  ```bash
  bun test/vram-padding-calibration.ts 0.06 1
  ```

- [ ] **6. Record environment metadata** (fill in once)

  ```bash
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  nvcc --version | grep release
  bun --version
  ```

  | Field | Value |
  |-------|-------|
  | GPU | |
  | Driver version | |
  | CUDA toolkit | |
  | Bun version | |
  | node-llama-cpp version | _(logged by test script)_ |
  | Date/time started | |

---

## How to Run Each Test

Every test follows this exact sequence. **Do not skip steps.**

### Starting the Ballast (Terminal 1)

```bash
./test/vram-ballast <MB>
```

Wait for **all three lines** before proceeding:
```
VRAM before: XXXX.X MiB free / XXXX.X MiB total
VRAM after:  XXXX.X MiB free / XXXX.X MiB total
Allocated XXXX MB VRAM. PID XXXXX.
BALLAST_READY=1
```

Record the "VRAM after" free value — this is your actual free VRAM (from `cudaMemGetInfo`).

### Running the Test (Terminal 2)

```bash
bun test/vram-padding-calibration.ts <percentage> <maxReservedGB>
echo "Exit code: $?"
```

Record these values from the output:
- `totalVram` (from vramPadding callback)
- `padding` (from vramPadding callback)
- `gpuLayers` (from model load)
- `modelSize` (from model load)
- Final result: `Success! Embedding dim: XXX` or error
- Exit code: `$?`

### After a SIGABRT (Exit Code 134)

1. Wait 5 seconds (GPU driver needs time to reclaim VRAM)
2. Verify GPU state recovered:
   ```bash
   nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader
   ```
3. If free VRAM hasn't returned to the ballast-only baseline after 10 seconds:
   - Kill the ballast (Ctrl+C in Terminal 1)
   - Run `nvidia-smi` to verify full VRAM is free
   - Restart the ballast
   - Wait for `BALLAST_READY=1`

### Between Different Configurations

- If the ballast MB is changing: Ctrl+C the old ballast, start new one
- If only percentage/maxReserved is changing: keep ballast running, just re-run the test
- Always verify `nvidia-smi` shows expected free VRAM before each run

### The 3-Run Rule

A configuration is:
- **PASS** = 3/3 runs succeed (exit code 0 + embedding completes)
- **FAIL** = any run exits non-zero (134, 137, 1, or 3)
- **FLAKY** = 2/3 pass, 1/3 fails — record as flaky, treat as fail for calibration

---

## Step 0: Baseline Crash Reproduction

**Goal:** Confirm the SIGABRT still happens with default padding. If this passes, the crash window has shifted and we need to re-characterize before proceeding.

### Terminal 1: Start ballast
```bash
./test/vram-ballast 24064
```
_(24064 MB = 23.5 GB)_

### Terminal 2: Run test with defaults
```bash
bun test/vram-padding-calibration.ts 0.06 1
echo "Exit code: $?"
```

### Expected Result
Exit code **134** (SIGABRT). The default 1 GB padding is insufficient at ~356 MiB free VRAM.

### Record

| Run | Free VRAM (cudaMemGetInfo) | Padding | gpuLayers | Exit Code | Notes |
|-----|---------------------------|---------|-----------|-----------|-------|
| 1 | | 1 GB | | | |

### Decision
- **Exit 134?** Crash confirmed. Proceed to Step 1.
- **Exit 0 (success)?** Crash window shifted. STOP. Re-run the full VRAM pressure matrix (16/22/23/23.5 GB ballast) to find the new crash point. Update ballast level for all subsequent steps.
- **Exit 137?** OOM kill (not SIGABRT). Try 23 GB ballast instead.
- **Exit 1?** JavaScript error. Read the error message — likely a different problem.

---

## Step 1: Verify High Padding Works

**Goal:** Confirm that generous padding prevents the crash. If this fails, padding alone can't fix the issue — escalate to the maintainer immediately.

### Terminal 1: Keep ballast running (24064 MB)
_(If you restarted it, wait for BALLAST_READY=1)_

### Terminal 2: Run with high padding
```bash
bun test/vram-padding-calibration.ts 0.2 5
echo "Exit code: $?"
```

### Expected Result
Exit code **0**. With 5 GB max padding, "auto" should choose 0 or very few GPU layers, keeping everything safe on CPU.

### Record

| Run | Free VRAM | Padding (effective) | gpuLayers | Exit Code | Notes |
|-----|-----------|-------------------|-----------|-----------|-------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |

### Decision
- **3/3 pass?** Padding works. Proceed to Step 2.
- **Any failure?** STOP. Padding alone doesn't fix this. Post to the GitHub issue:
  > "Even with percentage=0.2, maxReserved=5 GB, the crash persists at 23.5 GB ballast. This suggests the issue isn't padding-related."

---

## Step 2: Find Minimum Percentage

**Goal:** With `maxReserved=5 GB` (so the cap won't interfere), find the lowest percentage that still prevents the crash.

**How this works:** On a 24 GB card, `maxReserved=5 GB` becomes the cap for any percentage >= `5/24 = 0.208`. Below 0.208, the percentage is the binding constraint and effective padding = `24 * percentage`. So the real search space is **0.06 to 0.20**.

### Terminal 1: Keep ballast running (24064 MB)

### Terminal 2: Run each config (3 runs each, stop at first failure)

**Strategy:** Start from 0.06 (known failure) and work UP until you find the first passing value. Then confirm with 3 runs.

```bash
# Start low (expect failure), work up
bun test/vram-padding-calibration.ts 0.06 5 && echo "PASS" || echo "FAIL: $?"
bun test/vram-padding-calibration.ts 0.08 5 && echo "PASS" || echo "FAIL: $?"
bun test/vram-padding-calibration.ts 0.10 5 && echo "PASS" || echo "FAIL: $?"
bun test/vram-padding-calibration.ts 0.12 5 && echo "PASS" || echo "FAIL: $?"
bun test/vram-padding-calibration.ts 0.14 5 && echo "PASS" || echo "FAIL: $?"
bun test/vram-padding-calibration.ts 0.16 5 && echo "PASS" || echo "FAIL: $?"
bun test/vram-padding-calibration.ts 0.18 5 && echo "PASS" || echo "FAIL: $?"
bun test/vram-padding-calibration.ts 0.20 5 && echo "PASS" || echo "FAIL: $?"
```

> **Shortcut:** If 0.06 passes (unlikely), 0.06 is already sufficient — skip to Step 3.
> **If 0.20 fails:** The minimum is > 0.20, meaning the cap is always dominant. Skip to Step 3 (maxReserved is what matters).

Once you find the transition (last FAIL → first PASS), do 3 full runs on the first passing value:

```bash
# Example: if 0.12 was first PASS
bun test/vram-padding-calibration.ts 0.12 5; echo "Exit: $?"
bun test/vram-padding-calibration.ts 0.12 5; echo "Exit: $?"
bun test/vram-padding-calibration.ts 0.12 5; echo "Exit: $?"
```

### Results Table

| % | maxReserved | Effective Padding | Run 1 | Run 2 | Run 3 | gpuLayers | Verdict |
|------|-------------|-------------------|-------|-------|-------|-----------|---------|
| 0.06 | 5 GB | 1.44 GB | | | | | |
| 0.08 | 5 GB | 1.92 GB | | | | | |
| 0.10 | 5 GB | 2.40 GB | | | | | |
| 0.12 | 5 GB | 2.88 GB | | | | | |
| 0.14 | 5 GB | 3.36 GB | | | | | |
| 0.16 | 5 GB | 3.84 GB | | | | | |
| 0.18 | 5 GB | 4.32 GB | | | | | |
| 0.20 | 5 GB | 4.80 GB | | | | | |

> **Effective Padding** = `24 * percentage` (since all values < 0.208, the cap of 5 GB never kicks in)
> **Verdict**: PASS (3/3), FAIL, or FLAKY (2/3)

### Record

**Last failing percentage:** `__________`
**First passing percentage:** `__________`
**Minimum effective padding (GB):** `__________` (= 24 * first passing percentage)

---

## Step 3: Find Minimum maxReserved

**Goal:** With `percentage=0.4` (so the percentage never caps), find the lowest maxReserved that prevents the crash.

**How this works:** `24 * 0.4 = 9.6 GB` which is larger than any maxReserved we'll test (0.5-5.0 GB). So effective padding = maxReserved exactly. This cleanly isolates the maxReserved parameter.

### Terminal 1: Keep ballast running (24064 MB)

### Terminal 2: Run each config

**Strategy:** Start low (expect failure), work up in 0.5 GB increments. Once you find the transition, refine with 0.1 GB increments.

**Coarse search (0.5 GB steps):**

```bash
bun test/vram-padding-calibration.ts 0.4 0.5 && echo "PASS" || echo "FAIL: $?"
bun test/vram-padding-calibration.ts 0.4 1.0 && echo "PASS" || echo "FAIL: $?"
bun test/vram-padding-calibration.ts 0.4 1.5 && echo "PASS" || echo "FAIL: $?"
bun test/vram-padding-calibration.ts 0.4 2.0 && echo "PASS" || echo "FAIL: $?"
bun test/vram-padding-calibration.ts 0.4 2.5 && echo "PASS" || echo "FAIL: $?"
bun test/vram-padding-calibration.ts 0.4 3.0 && echo "PASS" || echo "FAIL: $?"
bun test/vram-padding-calibration.ts 0.4 3.5 && echo "PASS" || echo "FAIL: $?"
bun test/vram-padding-calibration.ts 0.4 4.0 && echo "PASS" || echo "FAIL: $?"
bun test/vram-padding-calibration.ts 0.4 4.5 && echo "PASS" || echo "FAIL: $?"
bun test/vram-padding-calibration.ts 0.4 5.0 && echo "PASS" || echo "FAIL: $?"
```

### Coarse Results

| % | maxReserved | Effective Padding | Run 1 | gpuLayers | Verdict |
|------|-------------|-------------------|-------|-----------|---------|
| 0.40 | 0.5 GB | 0.5 GB | | | |
| 0.40 | 1.0 GB | 1.0 GB | | | |
| 0.40 | 1.5 GB | 1.5 GB | | | |
| 0.40 | 2.0 GB | 2.0 GB | | | |
| 0.40 | 2.5 GB | 2.5 GB | | | |
| 0.40 | 3.0 GB | 3.0 GB | | | |
| 0.40 | 3.5 GB | 3.5 GB | | | |
| 0.40 | 4.0 GB | 4.0 GB | | | |
| 0.40 | 4.5 GB | 4.5 GB | | | |
| 0.40 | 5.0 GB | 5.0 GB | | | |

**Coarse transition:** fails at `______` GB, passes at `______` GB.

**Fine search (0.1 GB steps between coarse transition):**

Example: if coarse shows fail at 1.5 GB, pass at 2.0 GB:
```bash
bun test/vram-padding-calibration.ts 0.4 1.6 && echo "PASS" || echo "FAIL: $?"
bun test/vram-padding-calibration.ts 0.4 1.7 && echo "PASS" || echo "FAIL: $?"
bun test/vram-padding-calibration.ts 0.4 1.8 && echo "PASS" || echo "FAIL: $?"
bun test/vram-padding-calibration.ts 0.4 1.9 && echo "PASS" || echo "FAIL: $?"
```

### Fine Results

| % | maxReserved | Effective Padding | Run 1 | Run 2 | Run 3 | gpuLayers | Verdict |
|------|-------------|-------------------|-------|-------|-------|-----------|---------|
| 0.40 | | | | | | | |
| 0.40 | | | | | | | |
| 0.40 | | | | | | | |
| 0.40 | | | | | | | |

_(Fill in the values between the coarse transition, do 3 runs on the first passing value)_

### Record

**Last failing maxReserved:** `__________` GB
**First passing maxReserved:** `__________` GB

---

## Step 4: Combined Validation

**Goal:** Use the minimum values from Steps 2 and 3 together, then validate across multiple ballast levels to ensure the combined config works across a range of VRAM pressure scenarios.

### Your combined values

**percentage:** `__________` (from Step 2)
**maxReserved:** `__________` GB (from Step 3)

### Test Matrix

For each ballast level: kill old ballast, start new one, wait for BALLAST_READY, run 3 tests.

```bash
# Test 4a: At the crash point (23.5 GB ballast)
./test/vram-ballast 24064
# (Terminal 2, 3 runs)
bun test/vram-padding-calibration.ts <percentage> <maxReservedGB>

# Test 4b: Moderate pressure (23 GB ballast, ~856 MiB free)
./test/vram-ballast 23552
# (Terminal 2, 3 runs)
bun test/vram-padding-calibration.ts <percentage> <maxReservedGB>

# Test 4c: Light pressure (22 GB ballast, ~1.9 GB free)
./test/vram-ballast 22528
# (Terminal 2, 3 runs)
bun test/vram-padding-calibration.ts <percentage> <maxReservedGB>

# Test 4d: Medium pressure (16 GB ballast, ~8 GB free)
./test/vram-ballast 16384
# (Terminal 2, 3 runs)
bun test/vram-padding-calibration.ts <percentage> <maxReservedGB>

# Test 4e: No pressure (no ballast)
# (Kill ballast first, then:)
bun test/vram-padding-calibration.ts <percentage> <maxReservedGB>
```

### Results Table

| Ballast | Free VRAM (cudaMemGetInfo) | Effective Padding | gpuLayers | Run 1 | Run 2 | Run 3 | Verdict |
|---------|---------------------------|-------------------|-----------|-------|-------|-------|---------|
| 23.5 GB (24064 MB) | | | | | | | |
| 23 GB (23552 MB) | | | | | | | |
| 22 GB (22528 MB) | | | | | | | |
| 16 GB (16384 MB) | | | | | | | |
| 0 (no ballast) | | | | | | | |

### Decision
- **All pass?** The combined values work across the full range. Proceed to Step 5.
- **Crash point (23.5 GB) fails?** The two parameters interact non-linearly. Note the failure and try bumping the failing parameter by one increment. Report the interaction to the maintainer.
- **Other levels fail unexpectedly?** Higher padding shouldn't cause failures at lower pressure. Something else is wrong — check nvidia-smi, restart ballast, retry.

---

## Step 5: Additional Models

**Goal:** Test the combined padding values with larger models. If padding matters for compute buffers, larger models need more.

### 5a: Individual Model Tests

For each model, use the 23.5 GB ballast and your combined padding values. You'll need to modify the test script or run manual commands.

**Embed model (already tested in Steps 0-4):**
```bash
# Already validated — copy results from Step 4a
```

**Rerank model (qwen3-reranker 0.6B Q8_0):**

Create a modified test or run manually:
```bash
# Quick test — modify the modelPath in a copy of the script:
# Change resolveModelFile URI to the reranker model
# Change createEmbeddingContext() to the appropriate context type
#
# Or note: the reranker model URI and load pattern is in src/llm.ts
# Check src/llm.ts for the exact model URI and context creation code
```

**Query expansion model (qmd-query-expansion 1.7B Q4_K_M):**
```bash
# Same approach — check src/llm.ts for model URI and load pattern
# This is a chat/completion model, not embedding
```

> **Note:** For rerank and query expansion, the test script needs modification to load the correct model type. If you don't want to modify the script, you can test just the embed model (which is the one that originally crashed) and note in the issue comment that larger models were not tested.

### Results Table

| Model | Size (disk) | % | maxReserved | Ballast | gpuLayers | Run 1 | Run 2 | Run 3 | Verdict |
|-------|-------------|------|-------------|---------|-----------|-------|-------|-------|---------|
| embeddinggemma 300M Q8_0 | ~300 MB | | | 23.5 GB | | | | | _(from Step 4a)_ |
| qwen3-reranker 0.6B Q8_0 | ~600 MB | | | 23.5 GB | | | | | |
| qmd-query-expansion 1.7B Q4_K_M | ~1 GB | | | 23.5 GB | | | | | |

### 5b: Multi-Model Test (Production Simulation)

**Goal:** Load embed AND rerank on the same `Llama` instance, as production does. This tests whether combined VRAM consumption triggers the crash even with padding.

> This requires a custom script. If skipping, note it in the issue comment.

| Models Loaded | % | maxReserved | Ballast | embed gpuLayers | rerank gpuLayers | Result |
|---------------|------|-------------|---------|-----------------|------------------|--------|
| embed + rerank | | | 23.5 GB | | | |
| embed + rerank | | | 23 GB | | | |
| embed + rerank | | | 0 | | | |

---

## Results Summary

Fill this in after completing all steps.

### Key Findings

| Metric | Value |
|--------|-------|
| **Minimum safe percentage** (Step 2) | |
| **Minimum safe maxReserved** (Step 3) | |
| **Combined values** (Step 4) | % = , maxReserved = GB |
| **Combined effective padding on 24 GB** | GB |
| **Default effective padding on 24 GB** | 1.0 GB |
| **Padding increase needed** | x over default |
| **gpuLayers at crash point with new padding** | |
| **gpuLayers at crash point with default** | _(from Step 0)_ |

### Transition Behavior

Describe what happens at the boundary:
- Does "auto" go from full offload → partial offload → CPU-only as padding increases?
- Is the transition sharp (1 value difference between crash and success) or gradual?
- At the minimum safe padding, how many GPU layers are used?

```
Notes:



```

### nvidia-smi vs cudaMemGetInfo

Did you observe the expected 100-300 MiB discrepancy?

| Measurement | nvidia-smi free | cudaMemGetInfo free | Delta |
|-------------|----------------|---------------------|-------|
| Before ballast | | | |
| After 23.5 GB ballast | | | |
| After 23 GB ballast | | | |

### Performance Impact

| Ballast | Default Padding gpuLayers | New Padding gpuLayers | Performance Change |
|---------|--------------------------|----------------------|-------------------|
| 0 (no pressure) | _(full offload expected)_ | | |
| 16 GB | | | |
| 23 GB | | | |
| 23.5 GB | _(SIGABRT)_ | | |

### Decision: Does vramPadding Alone Fix This?

Use this framework from the plan:

- [ ] **Min safe padding <= current default (1 GB):** Upstream fix needed, `QMD_GPU_LAYERS` escape hatch is critical
- [ ] **Min safe padding < 2 GB:** Propose upstream default change, escape hatch is nice-to-have
- [ ] **Min safe padding > 3 GB:** Padding costs too much GPU perf, implement both padding bump + escape hatch

**Your assessment:**

```
Notes:



```

---

## Draft Issue Comment

After completing all tests, fill in this template and post to [node-llama-cpp#551](https://github.com/withcatai/node-llama-cpp/issues/551).

```markdown
@giladgd Here are my test results on RTX 3090 Ti (24 GB VRAM):

**Test setup:**
- GPU: NVIDIA RTX 3090 Ti (24 GB VRAM)
- Driver: [from environment metadata above]
- CUDA toolkit: [from environment metadata above]
- node-llama-cpp: [from test script output]
- Bun: [from environment metadata above]
- Model: embeddinggemma 300M Q8_0 (~300 MB on disk, ~635 MiB in VRAM)
- VRAM pressure: CUDA ballast tool allocating 23.5 GB (leaving ~356 MiB free per cudaMemGetInfo)
- This is the exact scenario that caused SIGABRT with default padding

**Step 0 — Baseline crash confirmed:**
Default padding (6% / 1 GB cap) at 23.5 GB ballast → exit code 134 (SIGABRT)

**Step 1 — High padding works:**
percentage=0.2, maxReserved=5 GB at 23.5 GB ballast → gpuLayers=[X], exit code 0

**Step 2 — Minimum working percentage (maxReserved=5 GB):**

| percentage | effective padding | gpuLayers | result |
|-----------|------------------|-----------|--------|
| [fill in rows from your data] | | | |

Last working percentage: **[X.XX]** (effective padding: [X.XX] GB)

**Step 3 — Minimum working maxReserved (percentage=0.4):**

| maxReserved | gpuLayers | result |
|-------------|-----------|--------|
| [fill in rows from your data] | | |

First working maxReserved: **[X.XX] GB**

**Step 4 — Combined values (percentage=[X.XX], maxReserved=[X.XX] GB):**

| Ballast | Free VRAM (cudaMemGetInfo) | gpuLayers | result |
|---------|---------------------------|-----------|--------|
| 23.5 GB | ~356 MiB | | |
| 23.0 GB | ~856 MiB | | |
| 22.0 GB | ~1.9 GB | | |
| 16.0 GB | ~8 GB | | |
| 0 | ~24 GB | | |

**Step 5 — Additional models at combined values + 23.5 GB ballast:**

| Model | Size | gpuLayers | result |
|-------|------|-----------|--------|
| embeddinggemma 300M Q8_0 | ~300 MB | | |
| qwen3-reranker 0.6B Q8_0 | ~600 MB | | |
| qmd-query-expansion 1.7B Q4_K_M | ~1 GB | | |

**Observations:**
- [Did auto fall back to fewer GPU layers or go CPU-only?]
- [Was the transition between crash and success sharp or gradual?]
- [Performance impact of the higher padding?]
- [nvidia-smi vs cudaMemGetInfo discrepancy observed: ~XXX MiB]
- [Any residual TOCTOU concerns at the boundary?]

**Recommendation:**
[Your conclusion: what should the new defaults be?]
```

---

## Quick Reference: Exit Codes

| Code | Meaning | Action |
|------|---------|--------|
| 0 | Success | Record as PASS |
| 1 | JS runtime error | Read error message, fix and retry |
| 2 | Invalid CLI args | Check your percentage/maxReserved values |
| 3 | Timeout (>60s) | Possible CUDA deadlock, kill and retry |
| 134 | SIGABRT | Expected crash — record as FAIL, wait 5s, verify GPU state |
| 137 | OOM kill | Kernel killed process, too much total memory pressure |

## Quick Reference: Ballast Sizes

| Ballast MB | Ballast GB | Approx Free VRAM | Command |
|-----------|-----------|------------------|---------|
| 24064 | 23.5 GB | ~356 MiB | `./test/vram-ballast 24064` |
| 23552 | 23.0 GB | ~856 MiB | `./test/vram-ballast 23552` |
| 22528 | 22.0 GB | ~1.9 GB | `./test/vram-ballast 22528` |
| 16384 | 16.0 GB | ~8 GB | `./test/vram-ballast 16384` |
