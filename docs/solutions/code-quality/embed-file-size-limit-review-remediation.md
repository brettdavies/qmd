---
title: "File Size Limit Feature - Code Review Remediation"
date: "2026-02-10"
category: "code-quality"
tags:
  - embedding
  - file-size-limit
  - code-review
  - testing
  - refactoring
  - options-object-pattern
severity: medium
component: "qmd embed, qmd status"
symptoms:
  - "Unclear function call sites with positional boolean parameters"
  - "Fractional byte values accepted from env var (should be integer)"
  - "No test coverage for config parsing with multiple edge cases"
  - "qmd status showed 'needs embedding' count without distinguishing too-large files"
root_cause: "Feature implementation prioritized functionality over API clarity, validation rigor, test coverage, and status display fidelity"
resolution: "Options object refactor, Math.floor validation, 10 unit tests, getEmbedBreakdown query"
commits:
  - "bce9777: feat(embed): show skipped file count in status and refactor review findings"
  - "dcd5434: fix(embed): address review findings for file size limit feature"
  - "942022f: feat(embed): skip oversized files with configurable size limit"
related_files:
  - src/qmd.ts
  - src/store.ts
  - src/embed-config.test.ts
related_docs:
  - docs/plans/2026-02-10-feat-embed-file-size-limit-plan.md
---

# File Size Limit Feature - Code Review Remediation

## Problem Statement

During multi-agent code review of the `feat/embed-file-size-limit` branch, four issues were identified in the initial implementation:

1. **Positional boolean parameters**: `vectorIndex(model, force, noSizeLimit)` was unclear at call sites
2. **Missing integer validation**: `getMaxEmbedFileBytes()` could return fractional byte values from env var
3. **No test coverage**: Config parsing handled 6+ edge cases with zero tests
4. **Status display gap**: `qmd status` reported all unembedded files as "Pending" without distinguishing files skipped due to size limits

## Root Cause

### 1. Positional boolean anti-pattern
The function signature used positional parameters:
```typescript
async function vectorIndex(model: string, force: boolean, noSizeLimit: boolean)
```
Call site read as `vectorIndex(DEFAULT_EMBED_MODEL, true, false)` -- impossible to tell which boolean means what.

### 2. Missing integer constraint
The config parser returned `Number(env)` directly:
```typescript
return parsed;  // Could be 1500.7 for bytes
```
SQLite's `LENGTH()` returns integers, creating a type mismatch.

### 3. Untested edge cases
`getMaxEmbedFileBytes()` handled NaN, Infinity, negative, zero, empty string, and non-numeric input but had no tests validating any of these paths.

### 4. Aggregated status metric
`showStatus()` called `getHashesNeedingEmbedding()` which returned a single count of all documents without embeddings, regardless of whether they were actionable or excluded by size limits.

## Solution

### 1. Refactor to options object pattern

**File**: `src/qmd.ts`

```typescript
// Before
async function vectorIndex(model: string = DEFAULT_EMBED_MODEL, force: boolean = false, noSizeLimit: boolean = false)

// After
async function vectorIndex({ model = DEFAULT_EMBED_MODEL, force = false, noSizeLimit = false }: {
  model?: string; force?: boolean; noSizeLimit?: boolean
} = {})
```

Call site becomes self-documenting:
```typescript
await vectorIndex({ force: !!cli.values.force, noSizeLimit: !!cli.values["no-size-limit"] });
```

### 2. Floor fractional byte values

**File**: `src/qmd.ts`

```typescript
export function getMaxEmbedFileBytes(): number {
  const env = process.env.QMD_MAX_EMBED_FILE_BYTES;
  if (!env) return DEFAULT_MAX_EMBED_FILE_BYTES;
  const parsed = Number(env);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    process.stderr.write(
      `Warning: Invalid QMD_MAX_EMBED_FILE_BYTES="${env}", using default ${formatBytes(DEFAULT_MAX_EMBED_FILE_BYTES)}\n`
    );
    return DEFAULT_MAX_EMBED_FILE_BYTES;
  }
  return Math.floor(parsed);  // Ensure integer bytes
}
```

### 3. Add comprehensive test coverage

**File**: `src/embed-config.test.ts` (new, 10 tests)

Tests use env var save/restore pattern to avoid pollution:
```typescript
beforeEach(() => {
  originalEnv = process.env.QMD_MAX_EMBED_FILE_BYTES;
  delete process.env.QMD_MAX_EMBED_FILE_BYTES;
});

afterEach(() => {
  if (originalEnv !== undefined) {
    process.env.QMD_MAX_EMBED_FILE_BYTES = originalEnv;
  } else {
    delete process.env.QMD_MAX_EMBED_FILE_BYTES;
  }
});
```

Edge cases covered: unset, valid numeric, large values, fractional (floors to integer), non-numeric string, empty string, zero, negative, Infinity, NaN.

### 4. Split embedding status with getEmbedBreakdown

**File**: `src/store.ts`

```typescript
export function getEmbedBreakdown(db: Database, maxBytes: number): { needsEmbedding: number; tooLarge: number } {
  const result = db.prepare(`
    SELECT
      COUNT(DISTINCT CASE WHEN LENGTH(c.doc) <= ? THEN d.hash END) as needs_embedding,
      COUNT(DISTINCT CASE WHEN LENGTH(c.doc) > ? THEN d.hash END) as too_large
    FROM documents d
    JOIN content c ON d.hash = c.hash
    LEFT JOIN content_vectors v ON d.hash = v.hash AND v.seq = 0
    WHERE d.active = 1 AND v.hash IS NULL
  `).get(maxBytes, maxBytes) as { needs_embedding: number; too_large: number };
  return { needsEmbedding: result.needs_embedding, tooLarge: result.too_large };
}
```

**File**: `src/qmd.ts` - updated `showStatus()`:

```typescript
const maxEmbedSize = getMaxEmbedFileBytes();
const { needsEmbedding, tooLarge } = getEmbedBreakdown(db, maxEmbedSize);

if (needsEmbedding > 0) {
  console.log(`  Pending:  ${needsEmbedding} need embedding (run 'qmd embed')`);
}
if (tooLarge > 0) {
  console.log(`  Skipped:  ${tooLarge} exceed ${formatBytes(maxEmbedSize)} size limit`);
}
```

Design choice: query-time filtering (no schema change) because the size limit is dynamic via env var.

## Verification

All 10 tests pass:
```
$ bun test src/embed-config.test.ts
 10 pass, 0 fail, 11 expect() calls
```

Status output now clearly distinguishes actionable from size-limited:
```
Documents
  Total:    150 files indexed
  Vectors:  100 embedded
  Pending:  45 need embedding (run 'qmd embed')
  Skipped:  5 exceed 5.0 MB size limit
```

## Prevention Strategies

### Options object pattern
For any function with 2+ boolean parameters, use a destructured options object. This makes call sites self-documenting and allows omitting defaults.

### Config parsing validation
- Always `Math.floor()` when integer is required
- Validate with `Number.isFinite()` and range checks
- Fall back to sensible defaults with a warning, never silently accept bad input

### Test env var edge cases
Config parsing that reads environment variables should test: unset, valid, fractional, non-numeric, empty, zero, negative, Infinity, NaN. Use save/restore in beforeEach/afterEach.

### Status must reflect actual state
When adding skip conditions to processing, immediately add corresponding status display breakdown. Users rely on status output for decision-making.

## Code Review Checklist

- [ ] Functions with 2+ boolean params use options object
- [ ] Env var parsing validates type (integer when needed)
- [ ] Config code has tests covering all validation branches
- [ ] Status displays break down counts by skip reason
- [ ] Edge cases tested: NaN, Infinity, negative, empty, zero
