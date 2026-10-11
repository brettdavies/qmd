/**
 * remote-query-path.test.ts - the query side of QMD_REMOTE_URL.
 *
 * 1. The SDK `createStore` (what `qmd mcp` uses) must route through RemoteQMD
 *    instead of building a private LlamaCpp, or every MCP client loads its own models.
 * 2. Queries must be formatted for the server's embed model even when RemoteQMD's
 *    background /health warmup has not resolved yet (it supplies embedModelName).
 */

import { describe, test, expect, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { createStore as createSdkStore } from "../src/index.js";
import { createStore, insertContent, insertDocument, structuredSearch } from "../src/store.js";
import { LlamaCpp, setDefaultLLM, setDefaultLlamaCpp } from "../src/llm.js";
import { RemoteQMD } from "../src/remote-qmd.js";

const QWEN_EMBED = "hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf";

/** Minimal `qmd serve` stand-in: /health answers after `healthDelayMs`, embed calls are recorded. */
async function fakeServe(healthDelayMs: number): Promise<{ url: string; embedded: string[]; server: Server }> {
  const embedded: string[] = [];
  const server = createServer((req, res) => {
    let body = "";
    req.on("data", (c) => { body += c; });
    req.on("end", () => {
      const json = (data: unknown) => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(data));
      };
      if (req.url === "/health") {
        setTimeout(() => json({ ok: true, version: "2", backend: "local", models: { embed: QWEN_EMBED, rerank: "r", generate: "g" } }), healthDelayMs);
      } else if (req.url === "/embed-batch") {
        const { texts } = JSON.parse(body) as { texts: string[] };
        embedded.push(...texts);
        json(texts.map(() => ({ embedding: [1, 0, 0], model: QWEN_EMBED })));
      } else if (req.url === "/embed") {
        const { text } = JSON.parse(body) as { text: string };
        embedded.push(text);
        json({ embedding: [1, 0, 0], model: QWEN_EMBED });
      } else {
        res.writeHead(404);
        res.end();
      }
    });
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  return { url: `http://127.0.0.1:${(server.address() as AddressInfo).port}`, embedded, server };
}

describe("QMD_REMOTE_URL on the query path", () => {
  const cleanups: (() => Promise<void>)[] = [];
  const originalRemoteUrl = process.env.QMD_REMOTE_URL;

  afterEach(async () => {
    while (cleanups.length) await cleanups.pop()!();
    setDefaultLLM(null);
    setDefaultLlamaCpp(null);
    if (originalRemoteUrl === undefined) delete process.env.QMD_REMOTE_URL;
    else process.env.QMD_REMOTE_URL = originalRemoteUrl;
  });

  test("SDK createStore uses RemoteQMD, not a private LlamaCpp, when QMD_REMOTE_URL is set", async () => {
    const serve = await fakeServe(0);
    cleanups.push(() => new Promise<void>((r) => serve.server.close(() => r())));
    process.env.QMD_REMOTE_URL = serve.url;
    setDefaultLLM(null);
    const dir = await mkdtemp(join(tmpdir(), "qmd-remote-sdk-"));
    const store = await createSdkStore({ dbPath: join(dir, "index.sqlite") });
    cleanups.push(async () => { await store.close(); await rm(dir, { recursive: true, force: true }); });

    expect(store.internal.llm).toBeInstanceOf(RemoteQMD);
    expect(store.internal.llm).not.toBeInstanceOf(LlamaCpp);
  });

  test("vec queries get the server model's prefix even before /health has answered", async () => {
    const serve = await fakeServe(300);
    cleanups.push(() => new Promise<void>((r) => serve.server.close(() => r())));
    const dir = await mkdtemp(join(tmpdir(), "qmd-remote-prefix-"));
    const store = createStore(join(dir, "index.sqlite"));
    cleanups.push(async () => { store.close(); await rm(dir, { recursive: true, force: true }); });

    const now = new Date().toISOString();
    store.ensureVecTable(3);
    insertContent(store.db, "h1", "body", now);
    insertDocument(store.db, "docs", "a.md", "A", "h1", now, now);
    store.insertEmbedding("h1", 0, 0, new Float32Array([1, 0, 0]), QWEN_EMBED, now, 1);

    // Constructed right before the query, so the /health warmup is still in flight.
    store.llm = new RemoteQMD({ serverUrl: serve.url });
    await structuredSearch(store, [{ type: "vec", query: "which port" }], { skipRerank: true });

    expect(serve.embedded.length).toBeGreaterThan(0);
    expect(serve.embedded[0]).toBe("Instruct: Retrieve relevant documents for the given query\nQuery: which port");
  });
});
