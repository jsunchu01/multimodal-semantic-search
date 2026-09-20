"""Postgres + pgvector vector store: a real relational DB, `vector` column
type, HNSW index for cosine similarity search. Requires a locally-run
Postgres with the `vector` extension enabled -- a fresh install, not
something this code can do for you (no CLI execution available here).

Verified against pgvector-python's current README (psycopg 3 usage) -- NOT
copied from the reference repo's asyncpg-based adapter, both because we're
staying synchronous for M2 (async is deferred to M7, where it actually pays
off) and because HNSW is used here instead of the reference repo's ivfflat:
HNSW doesn't need existing data in the table to build a good index, unlike
ivfflat, which is better suited to a table that's already populated.

The table is created lazily on first add() -- the embedding dimension is
inferred from whatever the configured embedder actually produces, rather
than being duplicated into vector_store config where it could silently
drift out of sync with the embedder's real output size.
"""

from __future__ import annotations

import json

from mmss.components.base import Chunk, ScoredChunk
from mmss.components.vector_store.base import VectorStore
from mmss.registry import register


@register("vector_store", "pgvector")
class PGVectorStore(VectorStore):
    def __init__(self, dsn: str = "postgresql:///mmss_dev", table_name: str = "mmss_chunks") -> None:
        self._dsn = dsn
        self._table = table_name
        self._conn = None
        self._table_ready = False

    @property
    def name(self) -> str:
        return f"pgvector:{self._table}"

    def _connect(self):
        if self._conn is None or self._conn.closed:
            import psycopg
            from pgvector.psycopg import register_vector

            self._conn = psycopg.connect(self._dsn, autocommit=True)
            self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(self._conn)
        return self._conn

    def _ensure_table(self, dimension: int) -> None:
        if self._table_ready:
            return
        conn = self._connect()
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {self._table} (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                text TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                page_start INTEGER,
                page_end INTEGER,
                source_parser TEXT,
                content_hash TEXT,
                metadata JSONB DEFAULT '{{}}'::jsonb,
                embedding vector({dimension}) NOT NULL
            )"""
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self._table}_embedding_idx "
            f"ON {self._table} USING hnsw (embedding vector_cosine_ops)"
        )
        self._table_ready = True

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        from pgvector import Vector

        self._ensure_table(len(embeddings[0]))
        conn = self._connect()
        with conn.cursor() as cur:
            for chunk, vec in zip(chunks, embeddings, strict=True):
                cur.execute(
                    f"""INSERT INTO {self._table}
                        (id, doc_id, text, chunk_type, page_start, page_end,
                         source_parser, content_hash, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            text = EXCLUDED.text,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata""",
                    (
                        chunk.id,
                        chunk.doc_id,
                        chunk.text,
                        chunk.chunk_type,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.source_parser,
                        chunk.content_hash,
                        json.dumps(chunk.metadata),
                        Vector(vec),
                    ),
                )

    def query(
        self, embedding: list[float], top_k: int, filters: dict | None = None
    ) -> list[ScoredChunk]:
        from pgvector import Vector

        conn = self._connect()
        query_vec = Vector(embedding)
        try:
            rows = conn.execute(
                f"""SELECT id, doc_id, text, chunk_type, page_start, page_end,
                           source_parser, content_hash, metadata,
                           1 - (embedding <=> %s) AS score
                    FROM {self._table}
                    ORDER BY embedding <=> %s
                    LIMIT %s""",
                (query_vec, query_vec, top_k),
            ).fetchall()
        except Exception:
            # Table doesn't exist yet -- nothing has been ingested. Broad
            # except here rather than a specific psycopg.errors class since
            # that exact exception name hasn't been independently verified.
            return []

        return [
            ScoredChunk(chunk=self._row_to_chunk(row), score=row[9], source="dense")
            for row in rows
        ]

    def get_all(self, filters: dict | None = None) -> list[Chunk]:
        # No _table_ready gate here (unlike the table-creation path) -- that
        # flag only reflects THIS instance's history and is always False on
        # a fresh process, even when the table genuinely exists from a prior
        # ingest run. Same try/except-on-missing-table pattern as query().
        conn = self._connect()
        try:
            rows = conn.execute(
                f"""SELECT id, doc_id, text, chunk_type, page_start, page_end,
                           source_parser, content_hash, metadata
                    FROM {self._table}"""
            ).fetchall()
        except Exception:
            return []
        return [self._row_to_chunk(row) for row in rows]

    @staticmethod
    def _row_to_chunk(row) -> Chunk:
        metadata = row[8]
        if isinstance(metadata, str):
            # Defensive: only parse if psycopg handed back a raw JSON
            # string rather than an already-deserialized dict.
            metadata = json.loads(metadata)
        return Chunk(
            id=row[0],
            doc_id=row[1],
            text=row[2],
            chunk_type=row[3],
            page_start=row[4],
            page_end=row[5],
            source_parser=row[6],
            content_hash=row[7],
            metadata=metadata or {},
        )

    def persist(self) -> None:
        pass  # autocommit=True -- every write is already durable, nothing to flush

    def delete(self, doc_id: str) -> None:
        conn = self._connect()
        try:
            conn.execute(f"DELETE FROM {self._table} WHERE doc_id = %s", (doc_id,))
        except Exception:
            # Same "table doesn't exist yet" tolerance as query()/get_all() --
            # nothing has been ingested, so there's nothing to delete either.
            pass
