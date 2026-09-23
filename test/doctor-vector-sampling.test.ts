import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { isBun } from "../src/db.js";
import { createStore, getEmbeddingVectorSamples, type Store } from "../src/store.js";

const projectRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
let store: Store;

beforeEach(() => { store = createStore(":memory:"); });
afterEach(() => { store.close(); });

function addDocument(hash: string, path: string, active: boolean = true): void {
  store.insertContent(hash, `Body for ${hash}`, "2026-01-01");
  const id = store.insertDocument("test", path, hash, hash, "2026-01-01", "2026-01-01");
  if (!active) store.db.prepare("UPDATE documents SET active = 0 WHERE id = ?").run(id);
}

function addChunk(hash: string, seq: number, model: string = "model", fingerprint: string = "current"): void {
  store.db.prepare(`
    INSERT INTO content_vectors (hash, seq, model, embed_fingerprint, embedded_at)
    VALUES (?, ?, ?, ?, '2026-01-01')
  `).run(hash, seq, model, fingerprint);
}

describe("doctor vector sampling", () => {
  test("samples each eligible chunk once and uses the first active path", () => {
    addDocument("shared", "z.md");
    addDocument("shared", "b.md");
    addDocument("shared", "a.md", false);
    addChunk("shared", 0);
    addChunk("shared", 1);
    addDocument("other", "other.md");
    addChunk("other", 0);
    addDocument("inactive", "inactive.md", false);
    addChunk("inactive", 0);
    addDocument("stale", "stale.md");
    addChunk("stale", 0, "model", "old");
    addDocument("different-model", "different.md");
    addChunk("different-model", 0, "other-model");
    addChunk("orphan", 0);

    const samples = getEmbeddingVectorSamples(store.db, "model", "current", 20);
    expect(samples.sort((a, b) => a.hash.localeCompare(b.hash) || a.seq - b.seq)).toEqual([
      { hash: "other", seq: 0, body: "Body for other", path: "other.md" },
      { hash: "shared", seq: 0, body: "Body for shared", path: "b.md" },
      { hash: "shared", seq: 1, body: "Body for shared", path: "b.md" },
    ]);
    expect(getEmbeddingVectorSamples(store.db, "model", "current", 2)).toHaveLength(2);
    expect(getEmbeddingVectorSamples(store.db, "model", "current", 0)).toEqual([]);
    expect(getEmbeddingVectorSamples(store.db, "missing", "current")).toEqual([]);
  });

  test("samples large documents with duplicate paths within a 32 MiB SQLite budget", () => {
    // SQLite's hard heap limit is process-wide and cannot be raised again.
    const worker = join(projectRoot, "test", "_helpers", "doctor-vector-sample-worker.ts");
    const args = isBun ? [worker] : [join(projectRoot, "node_modules", "tsx", "dist", "cli.mjs"), worker];
    const result = spawnSync(process.execPath, args, { encoding: "utf8", timeout: 20_000 });
    expect(result.error).toBeUndefined();
    expect(result.stderr).toBe("");
    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toEqual({ samples: 3, distinctChunks: 3, bodiesComplete: true });
  });
});
