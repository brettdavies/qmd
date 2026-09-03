/**
 * vec-layout.ts - Names and resolution of the sqlite-vec vector layout.
 *
 * Vectors live in one vec0 table partitioned by an integer collection id, one
 * row per (chunk, active collection). `vector_rows` maps each vec0 rowid back
 * to its (hash, seq, collection_id), and `vector_collection_ids` maps names to
 * ids. The integer key is what makes `qmd collection rename` a one-row update:
 * vec0 rejects UPDATE on a partition key, so a name-keyed partition would have
 * to rewrite every row of the collection.
 *
 * A layout change needs a new permanent table name: `ALTER TABLE RENAME` on a
 * vec0 table leaves its shadow tables under the old name (sqlite-vec 0.1.9),
 * and SQLite refuses to rename shadow tables by hand. `vecLayout` is the one
 * place that knows which table is live, so no other module spells the names.
 */

import type { Database } from "./db.js";

export const VEC_TABLE = "vectors_by_collection";
export const VEC_ROWS_TABLE = "vector_rows";
export const VEC_COLLECTION_IDS_TABLE = "vector_collection_ids";
export const LEGACY_VEC_TABLE = "vectors_vec";

/** vec0's shadow tables for a virtual table name. */
export type VecShadowTables = {
  chunks: string;
  rowids: string;
  vectorChunks: string;
  info: string;
};

export type VecLayout =
  | { kind: "none" }
  | { kind: "legacy"; table: typeof LEGACY_VEC_TABLE; dimensions: number | null; keyedByHashSeq: boolean; shadow: VecShadowTables }
  | { kind: "partitioned"; table: typeof VEC_TABLE; dimensions: number | null; shadow: VecShadowTables };

export type ReadableVecLayout = Exclude<VecLayout, { kind: "none" }>;

function shadowTables(table: string): VecShadowTables {
  return {
    chunks: `${table}_chunks`,
    rowids: `${table}_rowids`,
    vectorChunks: `${table}_vector_chunks00`,
    info: `${table}_info`,
  };
}

function parseDimensions(sql: string | null): number | null {
  const match = sql?.match(/float\[(\d+)\]/);
  return match?.[1] ? parseInt(match[1], 10) : null;
}

/**
 * Which vector layout the database holds. A legacy table wins while it
 * exists: the migration drops it as its last step, so that drop is the flip.
 */
export function vecLayout(db: Database): VecLayout {
  const rows = db.prepare(
    `SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name IN (?, ?)`
  ).all(LEGACY_VEC_TABLE, VEC_TABLE) as { name: string; sql: string | null }[];

  const legacy = rows.find((row) => row.name === LEGACY_VEC_TABLE);
  if (legacy) {
    return {
      kind: "legacy",
      table: LEGACY_VEC_TABLE,
      dimensions: parseDimensions(legacy.sql),
      keyedByHashSeq: (legacy.sql ?? "").includes("hash_seq"),
      shadow: shadowTables(LEGACY_VEC_TABLE),
    };
  }
  const partitioned = rows.find((row) => row.name === VEC_TABLE);
  if (partitioned) {
    return {
      kind: "partitioned",
      table: VEC_TABLE,
      dimensions: parseDimensions(partitioned.sql),
      shadow: shadowTables(VEC_TABLE),
    };
  }
  return { kind: "none" };
}

/** True when the partitioned vector index exists; vector search reads nothing else. */
export function hasVectorIndex(db: Database): boolean {
  return vecLayout(db).kind === "partitioned";
}

/**
 * A schema entry can outlive the vec0 module (reopening without sqlite-vec
 * loaded), in which case touching the table throws "no such module".
 */
export function vecTableReadable(db: Database, layout: ReadableVecLayout): boolean {
  try {
    db.prepare(`SELECT 1 FROM ${layout.table} LIMIT 0`).get();
    return true;
  } catch {
    return false;
  }
}

export function createVectorMetadataTables(db: Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS ${VEC_COLLECTION_IDS_TABLE} (
      id INTEGER PRIMARY KEY,
      name TEXT NOT NULL UNIQUE
    )
  `);
  db.exec(`
    CREATE TABLE IF NOT EXISTS ${VEC_ROWS_TABLE} (
      id INTEGER PRIMARY KEY,
      hash TEXT NOT NULL,
      seq INTEGER NOT NULL,
      collection_id INTEGER NOT NULL,
      UNIQUE(hash, seq, collection_id)
    )
  `);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_vector_rows_collection ON ${VEC_ROWS_TABLE}(collection_id)`);
}

export function createPartitionedVecTable(db: Database, dimensions: number): void {
  db.exec(
    `CREATE VIRTUAL TABLE ${VEC_TABLE} USING vec0(collection_id INTEGER PARTITION KEY, embedding float[${dimensions}] distance_metric=cosine)`
  );
}

export function resolveCollectionId(db: Database, name: string): number | undefined {
  const row = db.prepare(`SELECT id FROM ${VEC_COLLECTION_IDS_TABLE} WHERE name = ?`).get(name) as { id: number } | undefined;
  return row?.id;
}

/** Ids of the named collections that have one; names without an id are absent. */
export function resolveCollectionIds(db: Database, names: readonly string[]): Map<string, number> {
  const ids = new Map<string, number>();
  const stmt = db.prepare(`SELECT id FROM ${VEC_COLLECTION_IDS_TABLE} WHERE name = ?`);
  for (const name of names) {
    const row = stmt.get(name) as { id: number } | undefined;
    if (row) ids.set(name, row.id);
  }
  return ids;
}

export function allocateCollectionId(db: Database, name: string): number {
  db.prepare(`INSERT OR IGNORE INTO ${VEC_COLLECTION_IDS_TABLE} (name) VALUES (?)`).run(name);
  const id = resolveCollectionId(db, name);
  if (id === undefined || id === null) {
    throw new Error(`Could not allocate a vector collection id for '${name}'`);
  }
  return id;
}

export function renameCollectionId(db: Database, oldName: string, newName: string): void {
  db.prepare(`UPDATE ${VEC_COLLECTION_IDS_TABLE} SET name = ? WHERE name = ?`).run(newName, oldName);
}

export function deleteCollectionId(db: Database, name: string): void {
  db.prepare(`DELETE FROM ${VEC_COLLECTION_IDS_TABLE} WHERE name = ?`).run(name);
}

/**
 * One bound parameter for `json_each(?)`: a scoped search over many
 * partitions can return more rowids than SQLite allows as separate
 * parameters (32,766).
 */
export function rowidList(rowids: readonly number[]): string {
  return JSON.stringify(rowids);
}
