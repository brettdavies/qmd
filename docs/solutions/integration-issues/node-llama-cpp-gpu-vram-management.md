---
title: "GPU embed SIGABRT crash under VRAM pressure and embed file size guard"
date: 2026-02-12
category: "integration-issues"
tags: [gpu, cuda, node-llama-cpp, embeddings, vram, sigabrt, file-size-limit, upstream-contribution]
severity: "high"
components: [qmd/embed, node-llama-cpp, cuda, sqlite-vec]
symptoms:
  - "SIGABRT (exit code 134) crash when free VRAM < ~400 MiB"
  - "node-llama-cpp gpuLayers 'auto' does not gracefully fall back to CPU"
  - "qmd embed crashes without actionable error message under GPU memory pressure"
  - "Large files produce thousands of low-value chunks, exhausting resources and polluting search"
root_cause: "node-llama-cpp's gpuLayers 'auto' attempts GPU allocation without adequate free-VRAM preflight, causing ggml/CUDA abort() rather than a catchable error; qmd embed has no file size guard on ingestion"
resolution: "Filed upstream bug (node-llama-cpp#551); proposed QMD_GPU_LAYERS escape hatch (tobi/qmd#155); filed embed file size limit issue (tobi/qmd#156) linked to PR #153"
related_issues:
  - "withcatai/node-llama-cpp#551"
  - "tobi/qmd#155"
  - "tobi/qmd#156"
  - "tobi/qmd#153"
  - "tobi/qmd#69"
  - "tobi/qmd#91"
  - "tobi/qmd#147"
related_docs:
  - "docs/solutions/code-quality/embed-file-size-limit-review-remediation.md"
  - "docs/plans/2026-02-11-feat-gpu-layers-control-plan.md"
  - "docs/plans/2026-02-11-test-gpu-embed-fallback-behavior-plan.md"
  - "docs/plans/2026-02-10-feat-embed-file-size-limit-plan.md"
---

# GPU Embed SIGABRT Crash and File Size Guard

## Problem

Two failure modes in `qmd embed` when integrating with node-llama-cpp:

1. **CUDA OOM crash (SIGABRT)**: Running `qmd embed` while another GPU process consumes most VRAM causes an unrecoverable native crash (exit 134). No error message, no fallback, no way to catch it in JavaScript.

2. **No file size guard**: A 50MB log file produces ~19,275 chunks at 800 tokens each. This exhausts memory, dominates embed time, and pollutes search results. The security audit (#69) flagged this as H3: "Resource exhaustion via large files."

## Root Cause Analysis

### CUDA OOM: A TOCTOU Race in GPU Layer Estimation

node-llama-cpp's `gpuLayers: "auto"` estimates VRAM requirements from GGUF metadata **before** loading the model. The estimation calls `cudaMemGetInfo` once, then decides how many layers to offload. If an external process allocates VRAM between that measurement and the actual `cudaMalloc`, the estimate is stale.

The native CUDA allocation fails with `cudaMalloc: out of memory`, which triggers `abort()` in `ggml-cuda.cu:96`. The OS delivers SIGABRT, killing the process at exit code 134.

**Why this is uncatchable in JavaScript:**

| Approach | Result |
|----------|--------|
| `try { embedBatch() } catch (e) {}` | SIGABRT kills process before catch executes |
| `process.on('uncaughtException')` | Only catches JS exceptions, not native signals |
| `process.on('SIGABRT')` | Node.js/Bun handlers cannot prevent termination from native abort() |

**Empirical data (RTX 3090 Ti, 24 GB VRAM):**

| VRAM Ballast | Free VRAM | Behavior | Exit Code |
|-------------|-----------|----------|-----------|
| None | ~24 GB | Normal, 88% GPU util, 79.4 KB/s | 0 |
| 16 GB | ~8 GB | Normal, 75.4 KB/s | 0 |
| 22 GB | ~1.9 GB | OOM killed by kernel | 137 |
| 23 GB | ~856 MiB | Partial offload, 3.3x slower | 0 |
| 23.5 GB | ~356 MiB | **CUDA SIGABRT crash** | **134** |

The embed model (embeddinggemma 300M Q8_0) needs ~635 MiB of VRAM (300 MB weights + 335 MiB context buffers). The crash window is narrow: ~400-635 MiB free, where "auto" offloads some layers but the allocation fails.

### File Size: No Guard on Ingestion

`vectorIndex()` in `src/qmd.ts` only skipped empty files (`bodyBytes === 0`). Every non-empty document was tokenized and chunked regardless of size. At 800 tokens/chunk with 15% overlap: 1 MB = ~370 chunks, 5 MB = ~1,840 chunks, 50 MB = ~19,275 chunks.

## Working Solutions

### CUDA OOM: Upstream Bug + Escape Hatch Proposal

Since the crash is uncatchable in JavaScript, the fix belongs in node-llama-cpp. We filed:

- **Upstream bug**: [withcatai/node-llama-cpp#551](https://github.com/withcatai/node-llama-cpp/issues/551) -- comprehensive report with reproduction steps, VRAM test matrix, and suggested mitigations (safety margin, catch+retry, double-check, conservative mode)
- **Interim escape hatch**: [tobi/qmd#155](https://github.com/tobi/qmd/issues/155) -- proposes `QMD_GPU_LAYERS` env var and `--no-gpu` CLI flag. Implementation plan deepened with 6 review agents. Waiting for maintainer traction before implementing.

Key design decisions in the escape hatch plan:
- `getGpuLayers()` lives in `qmd.ts` (CLI layer); resolved value passed to `llm.ts` via config object -- avoids circular dependency (`llm.ts` has zero internal imports)
- No `process.env` mutation -- value passed explicitly through options objects
- Upper bound of 512 on explicit layer count (security hardening)
- `--no-gpu` kept as syntactic sugar for `QMD_GPU_LAYERS=0` -- easier to type during a crash

### File Size Limit: Implemented and PR'd

**PR**: [tobi/qmd#153](https://github.com/tobi/qmd/pull/153) -- tracking issue [#156](https://github.com/tobi/qmd/issues/156)

The implementation adds a configurable 5 MB default limit:

```bash
qmd embed                                     # Default: skip files > 5MB
QMD_MAX_EMBED_FILE_BYTES=10485760 qmd embed   # Custom limit (10MB)
qmd embed --no-size-limit                     # Bypass entirely
```

**Config parser** (`src/qmd.ts:1491`):

```typescript
export function getMaxEmbedFileBytes(): number {
  const env = process.env.QMD_MAX_EMBED_FILE_BYTES;
  if (!env) return DEFAULT_MAX_EMBED_FILE_BYTES;
  const parsed = Number(env);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    process.stderr.write(
      `${c.yellow}Warning: Invalid QMD_MAX_EMBED_FILE_BYTES="${env}", using default ${formatBytes(DEFAULT_MAX_EMBED_FILE_BYTES)}${c.reset}\n`
    );
    return DEFAULT_MAX_EMBED_FILE_BYTES;
  }
  return Math.floor(parsed);
}
```

- `Number.isFinite()` rejects NaN, Infinity, and -Infinity in one check
- `parsed <= 0` rejects negative and zero
- Invalid values warn to stderr and fall back to default (never silently disable protection)

**Size check in chunking loop** (`src/qmd.ts:1529`):

```typescript
const maxEmbedSize = noSizeLimit ? Infinity : getMaxEmbedFileBytes();
// ...
if (bodyBytes > maxEmbedSize) {
  process.stderr.write(`${c.yellow}Skipping ${item.path} (${formatBytes(bodyBytes)} exceeds ${formatBytes(maxEmbedSize)} limit)${c.reset}\n`);
  skippedFiles++;
  continue;
}
```

Uses `Buffer.byteLength(body, 'utf8')` instead of `TextEncoder` -- avoids allocating a full `Uint8Array` copy just to read `.length`.

**Test coverage**: 20 tests across 3 files (10 env var parsing, 5 SQL breakdown query, 5 CLI integration).

## What Didn't Work

**try/catch around embedBatch()**: SIGABRT is a POSIX signal from native code. V8 never gets a chance to catch anything.

**Subprocess isolation for crash recovery**: Theoretically possible (spawn embed in a child, catch exit 134 in parent). Not implemented because the crash window is narrow enough that `QMD_GPU_LAYERS=0` is simpler.

**Per-extension file size limits**: Rejected during plan review by all 9 research agents. Creates unbounded config surface (`QMD_MAX_EMBED_FILE_BYTES_MD`, `_TXT`, etc.) for what should be a guard clause. Single limit: ~25 lines. Per-extension: ~80-120 lines.

**`parseInt()` for env var parsing**: Silently returns NaN for non-numeric input. Since `bodyBytes > NaN` is always `false`, `QMD_MAX_EMBED_FILE_BYTES=banana` would silently disable all protection. `Number()` + `Number.isFinite()` is correct.

**Positional boolean parameters**: `vectorIndex(model, force, noSizeLimit)` reads as `vectorIndex(DEFAULT_EMBED_MODEL, true, false)` -- impossible to tell which boolean is which. Refactored to destructured options object.

## Key Technical Insights

1. **CUDA abort() is a process-level crash, not an exception.** Any Node.js/Bun application linking native CUDA code can crash in a way no JavaScript error handling can intercept. The only defenses are: avoid the crash path (control GPU layers), or run native code in a subprocess.

2. **node-llama-cpp `gpuLayers: "auto"` has a TOCTOU race.** VRAM is measured once before loading; external processes can change availability before allocation. The crash window is narrow (~400-635 MiB for the embed model) but real.

3. **The embed model's VRAM footprint is ~2x its file size.** 300 MB on disk = ~635 MiB in VRAM (weights + context buffers). This multiplier matters when calculating whether the model fits.

4. **`Buffer.byteLength(str, 'utf8')` vs `TextEncoder.encode(str).length`**: TextEncoder allocates a full Uint8Array copy. Buffer.byteLength computes length with O(1) memory. At 10K documents, this eliminates ~200 MB of transient heap allocations.

5. **`Infinity` as a no-op guard value.** `maxEmbedSize = Infinity` when `--no-size-limit` is active means `bodyBytes > Infinity` is always false. No additional conditional needed in the hot loop.

6. **Env var config for safety guards must validate aggressively.** Cover: unset, valid numeric, fractional, non-numeric, empty, zero, negative, Infinity, NaN. The worst failure mode is silent bypass where the guard appears active but accepts everything.

## Prevention Strategies

### For native crashes (SIGABRT/SIGSEGV) from native modules

- Treat calls into native modules like calls to external services: validate inputs, handle failures, don't assume the call returns
- For long-running services: isolate native work in subprocesses; parent survives crashes
- For batch CLI tools: use a process supervisor or simple retry
- Always provide a "safe mode" flag that avoids the risky code path entirely

### For resource exhaustion in document pipelines

- Every resource boundary (VRAM, file size, chunk count) needs a configurable guard with a safe default
- Check size before reading content, not after (use `stat()` not `read() + measure`)
- Design limits as "skip with warning," not "error and abort" -- one oversized file shouldn't prevent indexing the rest
- Report what was skipped and why, so users can tune limits for their corpus

### For upstream contributions: "Issue First, PR Second"

Rather than sending cold PRs to open source projects:

1. File a well-researched issue with empirical data and reproduction steps
2. Reference existing project priorities (security audits, related issues)
3. Propose the solution direction, ask for input
4. Link to the PR once the issue gains traction

This gives maintainers context before asking them to review code and avoids wasted effort if the approach isn't wanted. The cost is a few days of latency; the benefit is dramatically higher merge probability.

## Testing Methodology: VRAM Ballast

To test GPU-dependent features under VRAM pressure, we built a CUDA ballast tool that allocates precise VRAM amounts:

```c
// vram_ballast.cu -- allocate N MB of VRAM and hold it
cudaMalloc(&ptr, megabytes * 1024 * 1024);
// Block until killed
```

This simulates real-world scenarios (ollama running a 70B model) without actually loading a model. Test at multiple levels: plenty of VRAM, tight VRAM, contested VRAM, no VRAM. The CPU fallback path should always be tested in CI even without a GPU.

## References

- [withcatai/node-llama-cpp#551](https://github.com/withcatai/node-llama-cpp/issues/551) -- upstream CUDA SIGABRT bug
- [tobi/qmd#155](https://github.com/tobi/qmd/issues/155) -- GPU layers escape hatch proposal
- [tobi/qmd#156](https://github.com/tobi/qmd/issues/156) -- embed file size limit proposal
- [tobi/qmd#153](https://github.com/tobi/qmd/pull/153) -- file size limit PR
- [tobi/qmd#69](https://github.com/tobi/qmd/issues/69) -- security audit (H3: resource exhaustion via large files)
- [tobi/qmd#91](https://github.com/tobi/qmd/issues/91) -- reranker context overflow from large files
- `docs/plans/2026-02-11-feat-gpu-layers-control-plan.md` -- deepened implementation plan
- `docs/plans/2026-02-11-test-gpu-embed-fallback-behavior-plan.md` -- VRAM pressure test results
- `docs/solutions/code-quality/embed-file-size-limit-review-remediation.md` -- prior review remediation
