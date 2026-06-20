"""Vector store abstraction layer.

A single `VectorStore` interface with interchangeable backends:
  pgvector | pinecone | qdrant | weaviate | milvus

Select via `settings.vector_backend`. Backend SDKs are imported lazily so you
only need the dependency for the backend you actually use.

Record shape (uniform across backends):
    { id, embedding, text, metadata }

Every backend implements:
  * ensure_collection()                 -- create index/collection if missing
  * upsert(records)                      -- batched, idempotent by id
  * query(vector, top_k, flt)            -- similarity search
  * delete(ids)
"""
from __future__ import annotations

import abc
import json
from typing import Any, Optional, Sequence

from loguru import logger

from app.config import settings
from app.models.schemas import RetrievedChunk, VectorRecord


def _batched(items: Sequence, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


class VectorStore(abc.ABC):
    """Backend-agnostic vector database interface."""

    def __init__(self, dimension: int, collection: str, namespace: str) -> None:
        self.dimension = dimension
        self.collection = collection
        self.namespace = namespace

    @abc.abstractmethod
    async def ensure_collection(self) -> None: ...

    @abc.abstractmethod
    async def upsert(self, records: list[VectorRecord]) -> int: ...

    @abc.abstractmethod
    async def query(
        self, vector: list[float], top_k: int = 5, flt: Optional[dict[str, Any]] = None
    ) -> list[RetrievedChunk]: ...

    @abc.abstractmethod
    async def delete(self, ids: list[str]) -> None: ...

    async def close(self) -> None:  # optional override
        return None


# --------------------------------------------------------------------------- #
# pgvector (default — uses the local Postgres already configured)
# --------------------------------------------------------------------------- #
class PgVectorStore(VectorStore):
    def __init__(self, dimension: int, collection: str, namespace: str) -> None:
        super().__init__(dimension, collection, namespace)
        self._pool = None

    async def _get_pool(self):
        if self._pool is None:
            import asyncpg  # lazy

            self._pool = await asyncpg.create_pool(
                user=settings.db_username_local,
                password=settings.db_password_local,
                host=settings.db_host_local,
                port=settings.db_port_local,
                database=settings.db_name_local,
                min_size=1,
                max_size=10,
            )
        return self._pool

    async def ensure_collection(self) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.collection} (
                    id          TEXT PRIMARY KEY,
                    namespace   TEXT NOT NULL DEFAULT 'default',
                    embedding   vector({self.dimension}) NOT NULL,
                    text        TEXT NOT NULL,
                    metadata    JSONB NOT NULL DEFAULT '{{}}'::jsonb
                );
                """
            )
            # HNSW index for cosine similarity (fast ANN at scale).
            # The regular `vector` type caps HNSW at 2000 dims, so for higher
            # dimensions (e.g. text-embedding-3-large at 2048) we index a
            # `halfvec` cast — HNSW supports halfvec up to 4000 dims with
            # negligible recall loss. Queries must cast to halfvec to match.
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.collection}_emb_idx
                ON {self.collection}
                USING hnsw ((embedding::halfvec({self.dimension})) halfvec_cosine_ops);
                """
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self.collection}_ns_idx "
                f"ON {self.collection} (namespace);"
            )
        logger.info("pgvector collection '{}' ready (dim={})", self.collection, self.dimension)

    @staticmethod
    def _vec_literal(vec: list[float]) -> str:
        return "[" + ",".join(repr(float(x)) for x in vec) + "]"

    async def upsert(self, records: list[VectorRecord]) -> int:
        if not records:
            return 0
        pool = await self._get_pool()
        total = 0
        async with pool.acquire() as conn:
            for batch in _batched(records, settings.upsert_batch_size):
                rows = [
                    (
                        r.id,
                        self.namespace,
                        self._vec_literal(r.embedding),
                        r.text,
                        json.dumps(r.metadata),
                    )
                    for r in batch
                ]
                await conn.executemany(
                    f"""
                    INSERT INTO {self.collection} (id, namespace, embedding, text, metadata)
                    VALUES ($1, $2, $3::vector, $4, $5::jsonb)
                    ON CONFLICT (id) DO UPDATE
                        SET embedding = EXCLUDED.embedding,
                            text      = EXCLUDED.text,
                            metadata  = EXCLUDED.metadata;
                    """,
                    rows,
                )
                total += len(rows)
        logger.info("pgvector upserted {} records", total)
        return total

    async def query(
        self, vector: list[float], top_k: int = 5, flt: Optional[dict[str, Any]] = None
    ) -> list[RetrievedChunk]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, text, metadata,
                       1 - (embedding::halfvec({self.dimension}) <=> $1::halfvec({self.dimension})) AS score
                FROM {self.collection}
                WHERE namespace = $2
                ORDER BY embedding::halfvec({self.dimension}) <=> $1::halfvec({self.dimension})
                LIMIT $3;
                """,
                self._vec_literal(vector), self.namespace, top_k,
            )
        return [
            RetrievedChunk(
                id=r["id"],
                text=r["text"],
                score=float(r["score"]),
                metadata=json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
            )
            for r in rows
        ]

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self.collection} WHERE id = ANY($1::text[]);", ids
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


# --------------------------------------------------------------------------- #
# Pinecone
# --------------------------------------------------------------------------- #
class PineconeStore(VectorStore):
    def __init__(self, dimension: int, collection: str, namespace: str) -> None:
        super().__init__(dimension, collection, namespace)
        self._index = None

    async def ensure_collection(self) -> None:
        from pinecone import Pinecone, ServerlessSpec  # lazy

        pc = Pinecone(api_key=settings.pinecone_api_key)
        existing = {i["name"] for i in pc.list_indexes()}
        if settings.pinecone_index not in existing:
            pc.create_index(
                name=settings.pinecone_index,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud, region=settings.pinecone_region
                ),
            )
        self._index = pc.Index(settings.pinecone_index)
        logger.info("Pinecone index '{}' ready", settings.pinecone_index)

    def _idx(self):
        if self._index is None:
            raise RuntimeError("Call ensure_collection() before using the store")
        return self._index

    async def upsert(self, records: list[VectorRecord]) -> int:
        import asyncio

        idx = self._idx()
        total = 0
        for batch in _batched(records, settings.upsert_batch_size):
            vectors = [
                {
                    "id": r.id,
                    "values": r.embedding,
                    "metadata": {**r.metadata, "text": r.text},
                }
                for r in batch
            ]
            await asyncio.to_thread(idx.upsert, vectors=vectors, namespace=self.namespace)
            total += len(vectors)
        logger.info("Pinecone upserted {} records", total)
        return total

    async def query(
        self, vector: list[float], top_k: int = 5, flt: Optional[dict[str, Any]] = None
    ) -> list[RetrievedChunk]:
        import asyncio

        idx = self._idx()
        res = await asyncio.to_thread(
            idx.query,
            vector=vector,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
            filter=flt,
        )
        out = []
        for m in res.get("matches", []):
            meta = dict(m.get("metadata") or {})
            text = meta.pop("text", "")
            out.append(RetrievedChunk(id=m["id"], text=text, score=float(m["score"]), metadata=meta))
        return out

    async def delete(self, ids: list[str]) -> None:
        import asyncio

        if ids:
            await asyncio.to_thread(self._idx().delete, ids=ids, namespace=self.namespace)


# --------------------------------------------------------------------------- #
# Qdrant
# --------------------------------------------------------------------------- #
class QdrantStore(VectorStore):
    def __init__(self, dimension: int, collection: str, namespace: str) -> None:
        super().__init__(dimension, collection, namespace)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from qdrant_client import AsyncQdrantClient  # lazy

            self._client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )
        return self._client

    async def ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams  # lazy

        client = self._get_client()
        existing = {c.name for c in (await client.get_collections()).collections}
        if self.collection not in existing:
            await client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )
        logger.info("Qdrant collection '{}' ready", self.collection)

    @staticmethod
    def _point_id(raw: str) -> str:
        # Qdrant requires UUID or unsigned int ids; map our hash-ids to UUIDs.
        import uuid

        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))

    async def upsert(self, records: list[VectorRecord]) -> int:
        from qdrant_client.models import PointStruct  # lazy

        client = self._get_client()
        total = 0
        for batch in _batched(records, settings.upsert_batch_size):
            points = [
                PointStruct(
                    id=self._point_id(r.id),
                    vector=r.embedding,
                    payload={**r.metadata, "text": r.text, "namespace": self.namespace, "_id": r.id},
                )
                for r in batch
            ]
            await client.upsert(collection_name=self.collection, points=points)
            total += len(points)
        logger.info("Qdrant upserted {} records", total)
        return total

    async def query(
        self, vector: list[float], top_k: int = 5, flt: Optional[dict[str, Any]] = None
    ) -> list[RetrievedChunk]:
        client = self._get_client()
        res = await client.search(
            collection_name=self.collection, query_vector=vector, limit=top_k
        )
        out = []
        for p in res:
            payload = dict(p.payload or {})
            text = payload.pop("text", "")
            rid = payload.pop("_id", str(p.id))
            out.append(RetrievedChunk(id=rid, text=text, score=float(p.score), metadata=payload))
        return out

    async def delete(self, ids: list[str]) -> None:
        client = self._get_client()
        await client.delete(
            collection_name=self.collection,
            points_selector=[self._point_id(i) for i in ids],
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


# --------------------------------------------------------------------------- #
# Weaviate
# --------------------------------------------------------------------------- #
class WeaviateStore(VectorStore):
    def __init__(self, dimension: int, collection: str, namespace: str) -> None:
        super().__init__(dimension, collection, namespace)
        self._client = None
        self._class = collection.capitalize()

    def _get_client(self):
        if self._client is None:
            import weaviate  # lazy
            from weaviate.classes.init import Auth

            auth = Auth.api_key(settings.weaviate_api_key) if settings.weaviate_api_key else None
            self._client = weaviate.connect_to_local() if "localhost" in settings.weaviate_url \
                else weaviate.connect_to_weaviate_cloud(
                    cluster_url=settings.weaviate_url, auth_credentials=auth
                )
        return self._client

    async def ensure_collection(self) -> None:
        import asyncio

        def _ensure():
            from weaviate.classes.config import Configure
            client = self._get_client()
            if not client.collections.exists(self._class):
                client.collections.create(
                    name=self._class,
                    vectorizer_config=Configure.Vectorizer.none(),
                )
        await asyncio.to_thread(_ensure)
        logger.info("Weaviate class '{}' ready", self._class)

    async def upsert(self, records: list[VectorRecord]) -> int:
        import asyncio

        def _upsert():
            client = self._get_client()
            coll = client.collections.get(self._class)
            with coll.batch.dynamic() as batch:
                for r in records:
                    batch.add_object(
                        properties={**r.metadata, "text": r.text, "_id": r.id},
                        vector=r.embedding,
                        uuid=_uuid5(r.id),
                    )
            return len(records)

        n = await asyncio.to_thread(_upsert)
        logger.info("Weaviate upserted {} records", n)
        return n

    async def query(
        self, vector: list[float], top_k: int = 5, flt: Optional[dict[str, Any]] = None
    ) -> list[RetrievedChunk]:
        import asyncio

        def _query():
            client = self._get_client()
            coll = client.collections.get(self._class)
            res = coll.query.near_vector(near_vector=vector, limit=top_k, return_metadata=["distance"])
            out = []
            for o in res.objects:
                props = dict(o.properties)
                text = props.pop("text", "")
                rid = props.pop("_id", str(o.uuid))
                dist = o.metadata.distance if o.metadata else 1.0
                out.append(RetrievedChunk(id=rid, text=text, score=1.0 - float(dist), metadata=props))
            return out

        return await asyncio.to_thread(_query)

    async def delete(self, ids: list[str]) -> None:
        import asyncio

        def _delete():
            client = self._get_client()
            coll = client.collections.get(self._class)
            for i in ids:
                coll.data.delete_by_id(_uuid5(i))
        await asyncio.to_thread(_delete)

    async def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


# --------------------------------------------------------------------------- #
# Milvus
# --------------------------------------------------------------------------- #
class MilvusStore(VectorStore):
    def __init__(self, dimension: int, collection: str, namespace: str) -> None:
        super().__init__(dimension, collection, namespace)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from pymilvus import MilvusClient  # lazy

            self._client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token or "")
        return self._client

    async def ensure_collection(self) -> None:
        import asyncio

        def _ensure():
            client = self._get_client()
            if not client.has_collection(self.collection):
                client.create_collection(
                    collection_name=self.collection,
                    dimension=self.dimension,
                    metric_type="COSINE",
                    auto_id=False,
                    primary_field_name="id",
                    id_type="string",
                    max_length=512,
                )
        await asyncio.to_thread(_ensure)
        logger.info("Milvus collection '{}' ready", self.collection)

    async def upsert(self, records: list[VectorRecord]) -> int:
        import asyncio

        def _upsert():
            client = self._get_client()
            total = 0
            for batch in _batched(records, settings.upsert_batch_size):
                data = [
                    {"id": r.id, "vector": r.embedding, "text": r.text, **_flat_meta(r.metadata)}
                    for r in batch
                ]
                client.upsert(collection_name=self.collection, data=data)
                total += len(data)
            return total

        n = await asyncio.to_thread(_upsert)
        logger.info("Milvus upserted {} records", n)
        return n

    async def query(
        self, vector: list[float], top_k: int = 5, flt: Optional[dict[str, Any]] = None
    ) -> list[RetrievedChunk]:
        import asyncio

        def _query():
            client = self._get_client()
            res = client.search(
                collection_name=self.collection,
                data=[vector],
                limit=top_k,
                output_fields=["text", "source_url", "title", "section"],
            )
            out = []
            for hit in res[0]:
                entity = hit.get("entity", {})
                text = entity.pop("text", "")
                out.append(RetrievedChunk(
                    id=str(hit.get("id")), text=text,
                    score=float(hit.get("distance", 0.0)), metadata=entity,
                ))
            return out

        return await asyncio.to_thread(_query)

    async def delete(self, ids: list[str]) -> None:
        import asyncio

        await asyncio.to_thread(
            lambda: self._get_client().delete(collection_name=self.collection, ids=ids)
        )


# --------------------------------------------------------------------------- #
# helpers + factory
# --------------------------------------------------------------------------- #
def _uuid5(raw: str) -> str:
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def _flat_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Milvus dynamic fields accept scalars; stringify anything complex."""
    out: dict[str, Any] = {}
    for k, v in meta.items():
        out[k] = v if isinstance(v, (str, int, float, bool)) or v is None else json.dumps(v)
    return out


_BACKENDS: dict[str, type[VectorStore]] = {
    "pgvector": PgVectorStore,
    "pinecone": PineconeStore,
    "qdrant": QdrantStore,
    "weaviate": WeaviateStore,
    "milvus": MilvusStore,
}


def get_vector_store(
    backend: str | None = None,
    dimension: int | None = None,
    collection: str | None = None,
    namespace: str | None = None,
) -> VectorStore:
    backend = (backend or settings.vector_backend).lower()
    if backend not in _BACKENDS:
        raise ValueError(
            f"Unknown vector backend {backend!r}. Choose one of {list(_BACKENDS)}"
        )
    cls = _BACKENDS[backend]
    store = cls(
        dimension=dimension or settings.embedding_dimension,
        collection=collection or settings.vector_collection,
        namespace=namespace or settings.vector_namespace,
    )
    logger.info("Using vector backend: {}", backend)
    return store
