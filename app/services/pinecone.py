from __future__ import annotations

import logging
from typing import Any

from pinecone import Pinecone, ServerlessSpec
from pinecone.db_data.index import Index

from app.core.utils import logger
from configs.setting import get_settings

  

class PineconeService:
    """Thin wrapper around the Pinecone client for Dependency injenction

    Responsibilities (per architecture):
      - client init
      - upsert  (dense + optional sparse vectors with metadata)
      - delete  (batch by IDs, used for tombstoned chunks)
      - query   (dense, sparse, or hybrid)
    """

    def __init__(self, settings: Settings) -> None:
        settings = get_settings()
        self.index: Index | None = None
        try:
            pc = Pinecone(api_key=settings.pinecone_api_key)

            existing = [idx.name for idx in pc.list_indexes()]
            if settings.pinecone_index_name not in existing:
                pc.create_index(
                    name=settings.pinecone_index_name,
                    dimension=settings.embedding_dimensions,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=settings.pinecone_cloud,
                        region=settings.pinecone_region,
                    ),
                )

            self.index = pc.Index(settings.pinecone_index_name)
            logger.info("[Startup] Pinecone connected", index=settings.pinecone_index_name)
        except Exception:
            logger.warning("[Startup] Pinecone unavailable — starting in degraded mode")
            self.index = None

    # ------------------------------------------------------------------
    # Write path (ingestion)
    # ------------------------------------------------------------------

    def upsert(
        self,
        vectors: list[dict[str, Any]],
        *,
        namespace: str | None = None,
        batch_size: int = 100,
    ) -> int:
        """Upsert vectors into Pinecone in batches.

        Each dict must contain:
          ``id``            : str
          ``values``        : list[float]  — dense vector
          ``metadata``      : dict         — chunk metadata stored alongside the vector
          ``sparse_values`` : dict         — optional; keys ``indices`` (list[int])
                                            and ``values`` (list[float]) for hybrid search

        Returns the total number of vectors upserted.
        """
        ns = namespace or self._namespace
        total = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self._index.upsert(vectors=batch, namespace=ns)
            total += len(batch)
            logger.debug("pinecone upsert: %d vectors (namespace=%s)", len(batch), ns)
        return total

    def delete(
        self,
        ids: list[str],
        *,
        namespace: str | None = None,
        batch_size: int = 1000,
    ) -> None:
        """Delete vectors by ID. No-op if ``ids`` is empty.

        Called by the ingestion runner after tombstoning stale chunks in Mongo.
        """
        if not ids:
            return
        ns = namespace or self._namespace
        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            self._index.delete(ids=batch, namespace=ns)
            logger.debug("pinecone delete: %d ids (namespace=%s)", len(batch), ns)

    # ------------------------------------------------------------------
    # Read path (hybrid_retriever)
    # ------------------------------------------------------------------

    def query(
        self,
        dense_vector: list[float],
        *,
        top_k: int = 10,
        sparse_vector: dict[str, Any] | None = None,
        alpha: float = 0.75,
        filter: dict[str, Any] | None = None,
        namespace: str | None = None,
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> list[dict[str, Any]]:
        """Query Pinecone for nearest neighbours.

        When ``sparse_vector`` is supplied the query is hybrid.  ``alpha``
        controls the dense/sparse blend: 1.0 = pure dense, 0.0 = pure sparse.
        The standard linear combination (scale each side, send both) is used so
        Pinecone performs the final re-rank server-side.

        Returns a list of match dicts with keys ``id``, ``score``, ``metadata``.
        """
        ns = namespace or self._namespace
        kwargs: dict[str, Any] = {
            "top_k": top_k,
            "namespace": ns,
            "include_metadata": include_metadata,
            "include_values": include_values,
        }
        if filter:
            kwargs["filter"] = filter

        if sparse_vector:
            kwargs["vector"] = [v * alpha for v in dense_vector]
            kwargs["sparse_vector"] = {
                "indices": sparse_vector["indices"],
                "values": [v * (1.0 - alpha) for v in sparse_vector["values"]],
            }
        else:
            kwargs["vector"] = dense_vector

        response = self._index.query(**kwargs)
        return [
            {"id": m.id, "score": m.score, "metadata": m.metadata or {}}
            for m in response.matches
        ]

    # ------------------------------------------------------------------
    # Ops / health
    # ------------------------------------------------------------------

    def describe_index_stats(self) -> dict[str, Any]:
        """Return index stats; used by the /ops health endpoint."""
        stats = self._index.describe_index_stats()
        return {
            "total_vector_count": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness,
            "namespaces": {
                ns: {"vector_count": data.vector_count}
                for ns, data in (stats.namespaces or {}).items()
            },
        }
