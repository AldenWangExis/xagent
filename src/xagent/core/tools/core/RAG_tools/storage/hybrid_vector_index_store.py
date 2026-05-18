"""Hybrid vector index store.

Milvus handles embeddings and dense vector retrieval, while LanceDB keeps
documents/parses/chunks and other control-plane related tables.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator, Sequence
from typing import Any, Dict, Optional

from ..core.schemas import IndexResult
from .contracts import DocumentRecord, FilterExpression, IndexPolicy, VectorIndexStore
from .lancedb_stores import LanceDBVectorIndexStore
from .milvus_embedding_store import MilvusEmbeddingIndexStore

logger = logging.getLogger(__name__)


class HybridVectorIndexStore(VectorIndexStore):
    """Delegate non-vector tables to LanceDB, embeddings to Milvus."""

    def __init__(
        self,
        *,
        lancedb_store: LanceDBVectorIndexStore | None = None,
        milvus_embedding_store: MilvusEmbeddingIndexStore | None = None,
    ) -> None:
        self._lancedb = lancedb_store or LanceDBVectorIndexStore()
        self._milvus = milvus_embedding_store or MilvusEmbeddingIndexStore()

    def __getattr__(self, name: str) -> Any:
        # Default path: preserve existing LanceDB behavior for methods that are
        # unrelated to embeddings.
        return getattr(self._lancedb, name)

    @staticmethod
    def _is_embeddings_table(table_name: str) -> bool:
        return table_name.startswith("embeddings_")

    def upsert_embeddings(self, model_tag: str, records: list[dict[str, Any]]) -> None:
        self._milvus.upsert_embeddings(model_tag, records)

    async def upsert_embeddings_async(
        self, model_tag: str, records: list[dict[str, Any]]
    ) -> None:
        await asyncio.to_thread(self._milvus.upsert_embeddings, model_tag, records)

    def create_index(self, model_tag: str, readonly: bool = False) -> IndexResult:
        return self._milvus.create_index(model_tag, readonly)

    def open_embeddings_table(self, model_tag: str) -> tuple[Any, str]:
        return self._milvus.open_embeddings_table(model_tag)

    def get_vector_dimension(self, table_name: str) -> Optional[int]:
        if self._is_embeddings_table(table_name):
            return self._milvus.get_vector_dimension(table_name)
        return self._lancedb.get_vector_dimension(table_name)

    async def get_vector_dimension_async(self, table_name: str) -> Optional[int]:
        return self.get_vector_dimension(table_name)

    def list_table_names(self) -> Sequence[str]:
        merged = set(self._lancedb.list_table_names())
        merged.update(self._milvus.list_table_names())
        return sorted(merged)

    def search_vectors(
        self,
        table_name: str,
        query_vector: list[float],
        *,
        top_k: int,
        filters: Optional[FilterExpression] = None,
        vector_column_name: str = "vector",
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> list[dict[str, Any]]:
        if self._is_embeddings_table(table_name):
            return self._milvus.search_vectors(
                table_name=table_name,
                query_vector=query_vector,
                top_k=top_k,
                filters=filters,
                vector_column_name=vector_column_name,
                user_id=user_id,
                is_admin=is_admin,
            )
        return self._lancedb.search_vectors(
            table_name=table_name,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            vector_column_name=vector_column_name,
            user_id=user_id,
            is_admin=is_admin,
        )

    async def search_vectors_async(
        self,
        table_name: str,
        query_vector: list[float],
        *,
        top_k: int,
        filters: Optional[FilterExpression] = None,
        vector_column_name: str = "vector",
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> list[dict[str, Any]]:
        if self._is_embeddings_table(table_name):
            return self._milvus.search_vectors(
                table_name=table_name,
                query_vector=query_vector,
                top_k=top_k,
                filters=filters,
                vector_column_name=vector_column_name,
                user_id=user_id,
                is_admin=is_admin,
            )
        return await self._lancedb.search_vectors_async(
            table_name=table_name,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            vector_column_name=vector_column_name,
            user_id=user_id,
            is_admin=is_admin,
        )

    def search_vectors_by_model(
        self,
        model_tag: str,
        query_vector: list[float],
        *,
        top_k: int,
        filters: Optional[FilterExpression] = None,
        vector_column_name: str = "vector",
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> list[dict[str, Any]]:
        return self._milvus.search_vectors_by_model(
            model_tag=model_tag,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            vector_column_name=vector_column_name,
            user_id=user_id,
            is_admin=is_admin,
        )

    async def search_vectors_by_model_async(
        self,
        model_tag: str,
        query_vector: list[float],
        *,
        top_k: int,
        filters: Optional[FilterExpression] = None,
        vector_column_name: str = "vector",
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> list[dict[str, Any]]:
        return self.search_vectors_by_model(
            model_tag=model_tag,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            vector_column_name=vector_column_name,
            user_id=user_id,
            is_admin=is_admin,
        )

    def count_rows(
        self,
        table_name: str,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> int:
        if self._is_embeddings_table(table_name):
            return self._milvus.count_rows(
                table_name=table_name,
                filters=filters,
                user_id=user_id,
                is_admin=is_admin,
            )
        return self._lancedb.count_rows(table_name, filters, user_id, is_admin)

    async def count_rows_async(
        self,
        table_name: str,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> int:
        return self.count_rows(table_name, filters, user_id, is_admin)

    def iter_batches(
        self,
        table_name: str,
        columns: Optional[Sequence[str]] = None,
        batch_size: int = 1000,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> Iterator[Any]:
        if self._is_embeddings_table(table_name):
            return self._milvus.iter_batches(
                table_name=table_name,
                columns=columns,
                batch_size=batch_size,
                filters=filters,
                user_id=user_id,
                is_admin=is_admin,
            )
        return self._lancedb.iter_batches(
            table_name=table_name,
            columns=columns,
            batch_size=batch_size,
            filters=filters,
            user_id=user_id,
            is_admin=is_admin,
        )

    async def iter_batches_async(
        self,
        table_name: str,
        columns: Optional[Sequence[str]] = None,
        batch_size: int = 1000,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> Any:
        if self._is_embeddings_table(table_name):
            for batch in self._milvus.iter_batches(
                table_name=table_name,
                columns=columns,
                batch_size=batch_size,
                filters=filters,
                user_id=user_id,
                is_admin=is_admin,
            ):
                yield batch
            return
        async for batch in self._lancedb.iter_batches_async(
            table_name=table_name,
            columns=columns,
            batch_size=batch_size,
            filters=filters,
            user_id=user_id,
            is_admin=is_admin,
        ):
            yield batch

    def delete_collection_data(
        self,
        collection_name: str,
        user_id: Optional[int],
        is_admin: bool,
        warnings_out: Optional[list[str]] = None,
    ) -> dict[str, int]:
        deleted = self._lancedb.delete_collection_data(
            collection_name=collection_name,
            user_id=user_id,
            is_admin=is_admin,
            warnings_out=warnings_out,
        )
        milvus_deleted = self._milvus.delete_collection_embeddings(
            collection_name=collection_name,
            user_id=user_id,
            is_admin=is_admin,
        )
        for key, value in milvus_deleted.items():
            deleted[key] = deleted.get(key, 0) + int(value)
        return deleted

    def delete_collection_embeddings(
        self,
        *,
        collection_name: str,
        user_id: Optional[int],
        is_admin: bool,
    ) -> dict[str, int]:
        return self._milvus.delete_collection_embeddings(
            collection_name=collection_name,
            user_id=user_id,
            is_admin=is_admin,
        )

    def delete_document_embeddings(
        self,
        *,
        collection_name: str,
        doc_id: str,
        user_id: Optional[int],
        is_admin: bool,
    ) -> dict[str, int]:
        return self._milvus.delete_document_embeddings(
            collection_name=collection_name,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )

    def rename_collection_data(
        self,
        collection_name: str,
        new_name: str,
    ) -> list[str]:
        warnings = self._lancedb.rename_collection_data(collection_name, new_name)
        try:
            self._milvus.rename_collection_embeddings(
                collection_name=collection_name,
                new_name=new_name,
            )
        except Exception as exc:  # noqa: BLE001
            message = (
                f"Failed to update Milvus embeddings collection label "
                f"'{collection_name}' -> '{new_name}': {exc}"
            )
            logger.warning(message)
            warnings.append(message)
        return warnings

    def aggregate_collection_stats(
        self,
        user_id: Optional[int],
        is_admin: bool,
    ) -> dict[str, dict[str, int]]:
        stats = self._lancedb.aggregate_collection_stats(
            user_id=user_id, is_admin=is_admin
        )
        embedding_counts = self._milvus.count_embeddings_by_collection(
            user_id=user_id,
            is_admin=is_admin,
        )
        for collection_name, count in embedding_counts.items():
            stats.setdefault(
                collection_name,
                {"documents": 0, "parses": 0, "chunks": 0, "embeddings": 0},
            )
            stats[collection_name]["embeddings"] = int(count)
        return stats

    def count_embeddings_by_collection(
        self,
        *,
        user_id: Optional[int],
        is_admin: bool,
    ) -> dict[str, int]:
        return self._milvus.count_embeddings_by_collection(
            user_id=user_id,
            is_admin=is_admin,
        )

    def aggregate_document_stats(
        self,
        collection_name: str,
        doc_id: str,
        user_id: Optional[int],
        is_admin: bool,
    ) -> dict[str, int]:
        stats = self._lancedb.aggregate_document_stats(
            collection_name=collection_name,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )
        stats["embeddings"] = self._milvus.count_embeddings_for_document(
            collection_name=collection_name,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )
        return stats

    def count_embeddings_for_document(
        self,
        *,
        collection_name: str,
        doc_id: str,
        user_id: Optional[int],
        is_admin: bool,
    ) -> int:
        return self._milvus.count_embeddings_for_document(
            collection_name=collection_name,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )

    # ---- Explicit VectorIndexStore contract delegation methods ----
    # Keep LanceDB as the default control-plane delegate.

    def list_document_records(
        self,
        collection_name: Optional[str],
        user_id: Optional[int],
        is_admin: bool,
        max_results: int = 10000,
    ) -> list[DocumentRecord]:
        return self._lancedb.list_document_records(
            collection_name=collection_name,
            user_id=user_id,
            is_admin=is_admin,
            max_results=max_results,
        )

    def count_documents_grouped_by_collection(
        self,
        collection_names: Sequence[str],
        user_id: Optional[int],
        is_admin: bool,
    ) -> Dict[str, int]:
        return self._lancedb.count_documents_grouped_by_collection(
            collection_names=collection_names,
            user_id=user_id,
            is_admin=is_admin,
        )

    def delete_document_data(
        self,
        collection_name: str,
        doc_id: str,
        user_id: Optional[int],
        is_admin: bool,
    ) -> Dict[str, int]:
        return self._lancedb.delete_document_data(
            collection_name=collection_name,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )

    def aggregate_document_counts(
        self,
        table_name: str,
        doc_id_column: str,
        collection_name: str,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> Dict[str, int]:
        return self._lancedb.aggregate_document_counts(
            table_name=table_name,
            doc_id_column=doc_id_column,
            collection_name=collection_name,
            user_id=user_id,
            is_admin=is_admin,
        )

    def build_filter_expression(
        self,
        filters: Optional[FilterExpression],
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> Optional[str]:
        return self._lancedb.build_filter_expression(
            filters=filters,
            user_id=user_id,
            is_admin=is_admin,
        )

    def upsert_documents(self, records: list[dict[str, Any]]) -> None:
        self._lancedb.upsert_documents(records)

    def upsert_parses(self, records: list[dict[str, Any]]) -> None:
        self._lancedb.upsert_parses(records)

    def upsert_chunks(self, records: list[dict[str, Any]]) -> None:
        self._lancedb.upsert_chunks(records)

    async def upsert_documents_async(self, records: list[dict[str, Any]]) -> None:
        await self._lancedb.upsert_documents_async(records)

    async def upsert_chunks_async(self, records: list[dict[str, Any]]) -> None:
        await self._lancedb.upsert_chunks_async(records)

    async def search_fts_async(
        self,
        table_name: str,
        query_text: str,
        *,
        top_k: int,
        filters: Optional[FilterExpression] = None,
        text_column_name: str = "text",
    ) -> list[dict[str, Any]]:
        return await self._lancedb.search_fts_async(
            table_name=table_name,
            query_text=query_text,
            top_k=top_k,
            filters=filters,
            text_column_name=text_column_name,
        )

    def should_reindex(
        self,
        table_name: str,
        total_upserted: int,
        policy: IndexPolicy,
    ) -> bool:
        return self._lancedb.should_reindex(
            table_name=table_name,
            total_upserted=total_upserted,
            policy=policy,
        )

    def trigger_reindex(self, table_name: str) -> bool:
        return self._lancedb.trigger_reindex(table_name)

    async def should_reindex_async(
        self,
        table_name: str,
        total_upserted: int,
        policy: IndexPolicy,
    ) -> bool:
        return await self._lancedb.should_reindex_async(
            table_name=table_name,
            total_upserted=total_upserted,
            policy=policy,
        )

    async def trigger_reindex_async(self, table_name: str) -> bool:
        return await self._lancedb.trigger_reindex_async(table_name)

    def migrate_embeddings_table(
        self,
        model_id: str,
        batch_size: int = 1000,
    ) -> dict[str, Any]:
        return self._lancedb.migrate_embeddings_table(
            model_id=model_id,
            batch_size=batch_size,
        )

    def get_raw_connection(self) -> Any:
        return self._lancedb.get_raw_connection()
