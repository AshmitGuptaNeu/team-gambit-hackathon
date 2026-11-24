"""Where the knowledge base lives. Two backends with the same methods:

  PgStore      Postgres + pgvector: the real one (local Postgres, or AWS RDS which supports pgvector).
  MemoryStore  numpy in memory, optionally saved to a folder: tests, offline runs, the AMI benchmark.

Tables (PgStore), one meeting's rows are replaced together, so re-indexing a meeting is safe:
  meetings (meeting_id PK, title, started_at, attendees[], source, card_text, card_vec)
  chunks   (chunk_id PK, meeting_id FK, ord, start_s, end_s, speakers[], text, embed_text, vec)
  items    (item_id PK, meeting_id FK, chunk_id FK, kind, text, speaker, owner, start_s, vec)

search() returns cosine similarity (vectors are unit length), best first.
"""
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np

from chunking import Chunk
from items import Item

KINDS = ("chunk", "card", "item")


@dataclass
class MeetingRecord:
    meeting_id: str
    title: str
    started_at: datetime
    attendees: list
    card_text: str
    source: dict = field(default_factory=dict)


class MemoryStore:
    def __init__(self):
        self._meetings, self._chunks, self._items = {}, {}, {}
        self._vecs = {k: {} for k in KINDS}       # kind -> id -> vector
        self._matrix = {}                         # kind -> (ids, meeting_ids, matrix), rebuilt after writes
        self.version = 0

    # ---- writes
    def put_meeting(self, record, chunks, items, card_vec, chunk_vecs, item_vecs):
        self.delete_meeting(record.meeting_id)
        self._meetings[record.meeting_id] = record
        self._vecs["card"][record.meeting_id] = np.asarray(card_vec, dtype=np.float32)
        for c, v in zip(chunks, chunk_vecs, strict=True):
            self._chunks[c.chunk_id] = c
            self._vecs["chunk"][c.chunk_id] = np.asarray(v, dtype=np.float32)
        for i, v in zip(items, item_vecs, strict=True):
            self._items[i.item_id] = i
            self._vecs["item"][i.item_id] = np.asarray(v, dtype=np.float32)
        self._changed()

    def delete_meeting(self, meeting_id):
        if meeting_id not in self._meetings:
            return False
        del self._meetings[meeting_id]
        self._vecs["card"].pop(meeting_id, None)
        for table, kind in [(self._chunks, "chunk"), (self._items, "item")]:
            for key in [k for k, row in table.items() if row.meeting_id == meeting_id]:
                del table[key]
                self._vecs[kind].pop(key, None)
        self._changed()
        return True

    def _changed(self):
        self._matrix.clear()
        self.version += 1

    # ---- reads
    def meetings(self):
        return sorted(self._meetings.values(), key=lambda m: (m.started_at, m.meeting_id))

    def chunks(self, meeting_ids=None):
        keep = None if meeting_ids is None else set(meeting_ids)
        return sorted((c for c in self._chunks.values() if keep is None or c.meeting_id in keep),
                      key=lambda c: (c.meeting_id, c.ord))

    def items(self, meeting_ids=None):
        keep = None if meeting_ids is None else set(meeting_ids)
        return sorted((i for i in self._items.values() if keep is None or i.meeting_id in keep),
                      key=lambda i: (i.meeting_id, i.start, i.item_id))

    def chunk(self, chunk_id):
        return self._chunks[chunk_id]

    def _table(self, kind):
        if kind not in self._matrix:
            vecs = self._vecs[kind]
            ids = sorted(vecs)
            owner = {"card": lambda i: i, "chunk": lambda i: self._chunks[i].meeting_id,
                     "item": lambda i: self._items[i].meeting_id}[kind]
            mat = np.stack([vecs[i] for i in ids]) if ids else np.zeros((0, 1), dtype=np.float32)
            self._matrix[kind] = (ids, np.array([owner(i) for i in ids], dtype=object), mat)
        return self._matrix[kind]

    def search(self, kind, qvec, k, meeting_ids=None):
        ids, owners, mat = self._table(kind)
        if not ids or k <= 0:
            return []
        scores = mat @ np.asarray(qvec, dtype=np.float32)
        if meeting_ids is not None:
            mask = np.isin(owners, list(meeting_ids))
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(ids[j], float(scores[j])) for j in order if np.isfinite(scores[j])]

    # ---- files (for the CLIs when there is no database)
    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        rows = {"meetings": [{**asdict(m), "started_at": m.started_at.isoformat()} for m in self.meetings()],
                "chunks": [asdict(c) for c in self.chunks()], "items": [asdict(i) for i in self.items()]}
        with open(os.path.join(folder, "kb.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f)
        np.savez(os.path.join(folder, "vectors.npz"),
                 **{f"{kind}__{key}": v for kind in KINDS for key, v in self._vecs[kind].items()})

    @classmethod
    def load(cls, folder):
        store = cls()
        if not os.path.exists(os.path.join(folder, "kb.json")):
            return store
        with open(os.path.join(folder, "kb.json"), encoding="utf-8") as f:
            rows = json.load(f)
        for m in rows["meetings"]:
            store._meetings[m["meeting_id"]] = MeetingRecord(**{**m, "started_at": datetime.fromisoformat(m["started_at"])})
        store._chunks = {c["chunk_id"]: Chunk(**c) for c in rows["chunks"]}
        store._items = {i["item_id"]: Item(**i) for i in rows["items"]}
        with np.load(os.path.join(folder, "vectors.npz")) as z:
            for name in z.files:
                kind, key = name.split("__", 1)
                store._vecs[kind][key] = z[name]
        store._changed()
        return store


SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS kb_meta (key text PRIMARY KEY, value text NOT NULL);
CREATE TABLE IF NOT EXISTS meetings (
    meeting_id text PRIMARY KEY, title text NOT NULL, started_at timestamptz NOT NULL,
    attendees text[] NOT NULL, source jsonb NOT NULL, card_text text NOT NULL, card_vec vector({dim}) NOT NULL,
    indexed_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id text PRIMARY KEY, meeting_id text NOT NULL REFERENCES meetings ON DELETE CASCADE,
    ord int NOT NULL, start_s real NOT NULL, end_s real NOT NULL, speakers text[] NOT NULL,
    text text NOT NULL, embed_text text NOT NULL, utterance_uids text[] NOT NULL, vec vector({dim}) NOT NULL);
CREATE TABLE IF NOT EXISTS items (
    item_id text PRIMARY KEY, meeting_id text NOT NULL REFERENCES meetings ON DELETE CASCADE,
    chunk_id text NOT NULL REFERENCES chunks ON DELETE CASCADE, kind text NOT NULL, text text NOT NULL,
    speaker text NOT NULL, owner text, start_s real NOT NULL, vec vector({dim}) NOT NULL);
CREATE INDEX IF NOT EXISTS chunks_meeting ON chunks (meeting_id, ord);
CREATE INDEX IF NOT EXISTS items_meeting ON items (meeting_id);
CREATE INDEX IF NOT EXISTS meetings_started ON meetings (started_at);
CREATE INDEX IF NOT EXISTS chunks_vec ON chunks USING hnsw (vec vector_cosine_ops);
CREATE INDEX IF NOT EXISTS items_vec ON items USING hnsw (vec vector_cosine_ops);
INSERT INTO kb_meta VALUES ('version', '0'), ('dim', '{dim}') ON CONFLICT DO NOTHING;
"""

DROP_SQL = "DROP TABLE IF EXISTS items, chunks, meetings, kb_meta CASCADE"

SEARCH_SQL = {
    "chunk": ("chunks", "chunk_id"),
    "item": ("items", "item_id"),
    "card": ("meetings", "meeting_id"),
}


class PgStore:
    def __init__(self, url, dim):
        import psycopg
        from pgvector.psycopg import register_vector
        self.conn = psycopg.connect(url, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        with self.conn.transaction():
            self.conn.execute(SCHEMA.format(dim=int(dim)))
        stored = self.conn.execute("SELECT value FROM kb_meta WHERE key = 'dim'").fetchone()[0]
        if int(stored) != int(dim):
            raise ValueError(f"database was built with {stored}-dim vectors, the embedder makes {dim}")
        # Exact ordering when a meeting filter is combined with the HNSW index (pgvector >= 0.8).
        self.conn.execute("SET hnsw.iterative_scan = strict_order")
        self.conn.execute("SET hnsw.ef_search = 100")

    @property
    def version(self):
        return int(self.conn.execute("SELECT value FROM kb_meta WHERE key = 'version'").fetchone()[0])

    def _bump(self):
        self.conn.execute("UPDATE kb_meta SET value = (value::int + 1)::text WHERE key = 'version'")

    def put_meeting(self, record, chunks, items, card_vec, chunk_vecs, item_vecs):
        from psycopg.types.json import Jsonb
        if len(chunks) != len(chunk_vecs) or len(items) != len(item_vecs):
            raise ValueError("one vector per chunk and per item")
        with self.conn.transaction():
            self.conn.execute("DELETE FROM meetings WHERE meeting_id = %s", (record.meeting_id,))
            self.conn.execute(
                "INSERT INTO meetings (meeting_id, title, started_at, attendees, source, card_text, card_vec)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (record.meeting_id, record.title, record.started_at, list(record.attendees), Jsonb(record.source),
                 record.card_text, np.asarray(card_vec, dtype=np.float32)))
            with self.conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    [(c.chunk_id, c.meeting_id, c.ord, c.start, c.end, list(c.speakers), c.text, c.embed_text,
                      list(c.utterance_uids), np.asarray(v, dtype=np.float32)) for c, v in zip(chunks, chunk_vecs)])
                cur.executemany(
                    "INSERT INTO items VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    [(i.item_id, i.meeting_id, i.chunk_id, i.kind, i.text, i.speaker, i.owner, i.start,
                      np.asarray(v, dtype=np.float32)) for i, v in zip(items, item_vecs)])
            self._bump()

    def delete_meeting(self, meeting_id):
        with self.conn.transaction():
            n = self.conn.execute("DELETE FROM meetings WHERE meeting_id = %s", (meeting_id,)).rowcount
            if n:
                self._bump()
        return bool(n)

    def meetings(self):
        rows = self.conn.execute("SELECT meeting_id, title, started_at, attendees, card_text, source FROM meetings"
                                 " ORDER BY started_at, meeting_id").fetchall()
        return [MeetingRecord(r[0], r[1], r[2], list(r[3]), r[4], r[5]) for r in rows]

    @staticmethod
    def _chunk(r):
        return Chunk(chunk_id=r[0], meeting_id=r[1], ord=r[2], start=float(r[3]), end=float(r[4]),
                     speakers=list(r[5]), text=r[6], embed_text=r[7], utterance_uids=list(r[8]))

    def chunks(self, meeting_ids=None):
        sql = "SELECT chunk_id, meeting_id, ord, start_s, end_s, speakers, text, embed_text, utterance_uids FROM chunks"
        if meeting_ids is None:
            rows = self.conn.execute(sql + " ORDER BY meeting_id, ord").fetchall()
        else:
            rows = self.conn.execute(sql + " WHERE meeting_id = ANY(%s) ORDER BY meeting_id, ord",
                                     (list(meeting_ids),)).fetchall()
        return [self._chunk(r) for r in rows]

    def chunk(self, chunk_id):
        r = self.conn.execute("SELECT chunk_id, meeting_id, ord, start_s, end_s, speakers, text, embed_text,"
                              " utterance_uids FROM chunks WHERE chunk_id = %s", (chunk_id,)).fetchone()
        if r is None:
            raise KeyError(chunk_id)
        return self._chunk(r)

    def items(self, meeting_ids=None):
        sql = "SELECT item_id, meeting_id, chunk_id, kind, text, speaker, owner, start_s FROM items"
        if meeting_ids is None:
            rows = self.conn.execute(sql + " ORDER BY meeting_id, start_s, item_id").fetchall()
        else:
            rows = self.conn.execute(sql + " WHERE meeting_id = ANY(%s) ORDER BY meeting_id, start_s, item_id",
                                     (list(meeting_ids),)).fetchall()
        return [Item(r[0], r[1], r[2], r[3], r[4], r[5], r[6], float(r[7])) for r in rows]

    def search(self, kind, qvec, k, meeting_ids=None):
        if k <= 0:
            return []
        table, key = SEARCH_SQL[kind]
        col = "card_vec" if kind == "card" else "vec"
        q = np.asarray(qvec, dtype=np.float32)
        where, params = "", [q]
        if meeting_ids is not None:
            where, params = " WHERE meeting_id = ANY(%s)", [q, list(meeting_ids)]
        rows = self.conn.execute(
            f"SELECT {key}, 1 - ({col} <=> %s) FROM {table}{where} ORDER BY {col} <=> %s, {key} LIMIT %s",
            [params[0], *params[1:], q, k]).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def drop_all(self):
        """Tests only: remove every table this store created."""
        self.conn.execute(DROP_SQL)

    @staticmethod
    def drop(url):
        """Tests only: remove the tables without opening a store (works whatever dimension they were built with)."""
        import psycopg
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute(DROP_SQL)


def open_store(url, dim, memory_dir=None):
    """PgStore if a database URL is given, else a MemoryStore loaded from memory_dir (if any)."""
    if url:
        return PgStore(url, dim)
    return MemoryStore.load(memory_dir) if memory_dir else MemoryStore()
