from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult


def test_upsert_embeddings_uses_milvus_delegate() -> None:
    from xagent.core.tools.core.RAG_tools.storage.hybrid_vector_index_store import (
        HybridVectorIndexStore,
    )

    lancedb_store = Mock()
    milvus_store = Mock()
    hybrid = HybridVectorIndexStore(
        lancedb_store=lancedb_store,
        milvus_embedding_store=milvus_store,
    )
    rows = [{"doc_id": "doc-1", "chunk_id": "chunk-1", "vector": [0.1, 0.2]}]

    hybrid.upsert_embeddings("text-embedding-v4", rows)

    milvus_store.upsert_embeddings.assert_called_once_with("text-embedding-v4", rows)
    lancedb_store.upsert_embeddings.assert_not_called()


def test_create_index_uses_milvus_delegate() -> None:
    from xagent.core.tools.core.RAG_tools.storage.hybrid_vector_index_store import (
        HybridVectorIndexStore,
    )

    lancedb_store = Mock()
    milvus_store = Mock()
    milvus_store.create_index.return_value = IndexResult(
        status="index_ready",
        advice=None,
        fts_enabled=False,
    )
    hybrid = HybridVectorIndexStore(
        lancedb_store=lancedb_store,
        milvus_embedding_store=milvus_store,
    )

    result = hybrid.create_index("text-embedding-v4", readonly=False)

    assert result.status == "index_ready"
    milvus_store.create_index.assert_called_once_with("text-embedding-v4", False)
    lancedb_store.create_index.assert_not_called()


def test_search_vectors_routes_embeddings_table_to_milvus() -> None:
    from xagent.core.tools.core.RAG_tools.storage.hybrid_vector_index_store import (
        HybridVectorIndexStore,
    )

    lancedb_store = Mock()
    milvus_store = Mock()
    milvus_store.search_vectors.return_value = [{"doc_id": "doc-1"}]
    hybrid = HybridVectorIndexStore(
        lancedb_store=lancedb_store,
        milvus_embedding_store=milvus_store,
    )

    rows = hybrid.search_vectors(
        table_name="embeddings_text_embedding_v4",
        query_vector=[0.1, 0.2],
        top_k=5,
    )

    assert rows == [{"doc_id": "doc-1"}]
    milvus_store.search_vectors.assert_called_once()
    lancedb_store.search_vectors.assert_not_called()


def test_search_vectors_routes_non_embeddings_table_to_lancedb() -> None:
    from xagent.core.tools.core.RAG_tools.storage.hybrid_vector_index_store import (
        HybridVectorIndexStore,
    )

    lancedb_store = Mock()
    lancedb_store.search_vectors.return_value = [{"doc_id": "doc-lancedb"}]
    milvus_store = Mock()
    hybrid = HybridVectorIndexStore(
        lancedb_store=lancedb_store,
        milvus_embedding_store=milvus_store,
    )

    rows = hybrid.search_vectors(
        table_name="documents",
        query_vector=[0.1, 0.2],
        top_k=5,
    )

    assert rows == [{"doc_id": "doc-lancedb"}]
    lancedb_store.search_vectors.assert_called_once()
    milvus_store.search_vectors.assert_not_called()


def test_delete_collection_data_merges_counts_from_both_delegates() -> None:
    from xagent.core.tools.core.RAG_tools.storage.hybrid_vector_index_store import (
        HybridVectorIndexStore,
    )

    lancedb_store = Mock()
    lancedb_store.delete_collection_data.return_value = {"documents": 2, "chunks": 4}
    milvus_store = Mock()
    milvus_store.delete_collection_embeddings.return_value = {
        "embeddings_text_embedding_v4": 3
    }
    hybrid = HybridVectorIndexStore(
        lancedb_store=lancedb_store,
        milvus_embedding_store=milvus_store,
    )

    deleted = hybrid.delete_collection_data("kb1", user_id=7, is_admin=False, warnings_out=[])

    assert deleted == {"documents": 2, "chunks": 4, "embeddings_text_embedding_v4": 3}
    lancedb_store.delete_collection_data.assert_called_once()
    milvus_store.delete_collection_embeddings.assert_called_once_with(
        collection_name="kb1",
        user_id=7,
        is_admin=False,
    )


def test_delete_document_embeddings_uses_milvus_delegate() -> None:
    from xagent.core.tools.core.RAG_tools.storage.hybrid_vector_index_store import (
        HybridVectorIndexStore,
    )

    lancedb_store = Mock()
    milvus_store = Mock()
    milvus_store.delete_document_embeddings.return_value = {"embeddings_model": 2}
    hybrid = HybridVectorIndexStore(
        lancedb_store=lancedb_store,
        milvus_embedding_store=milvus_store,
    )

    deleted = hybrid.delete_document_embeddings(
        collection_name="kb1",
        doc_id="doc-1",
        user_id=9,
        is_admin=False,
    )

    assert deleted == {"embeddings_model": 2}
    milvus_store.delete_document_embeddings.assert_called_once_with(
        collection_name="kb1",
        doc_id="doc-1",
        user_id=9,
        is_admin=False,
    )
    lancedb_store.delete_document_embeddings.assert_not_called()


def test_delete_collection_embeddings_uses_milvus_delegate() -> None:
    from xagent.core.tools.core.RAG_tools.storage.hybrid_vector_index_store import (
        HybridVectorIndexStore,
    )

    lancedb_store = Mock()
    milvus_store = Mock()
    milvus_store.delete_collection_embeddings.return_value = {"embeddings_model": 5}
    hybrid = HybridVectorIndexStore(
        lancedb_store=lancedb_store,
        milvus_embedding_store=milvus_store,
    )

    deleted = hybrid.delete_collection_embeddings(
        collection_name="kb1",
        user_id=3,
        is_admin=False,
    )

    assert deleted == {"embeddings_model": 5}
    milvus_store.delete_collection_embeddings.assert_called_once_with(
        collection_name="kb1",
        user_id=3,
        is_admin=False,
    )
    lancedb_store.delete_collection_embeddings.assert_not_called()


def test_rename_collection_data_updates_milvus_and_lancedb() -> None:
    from xagent.core.tools.core.RAG_tools.storage.hybrid_vector_index_store import (
        HybridVectorIndexStore,
    )

    lancedb_store = Mock()
    lancedb_store.rename_collection_data.return_value = []
    milvus_store = Mock()
    milvus_store.rename_collection_embeddings.return_value = {"embeddings_model": 3}
    hybrid = HybridVectorIndexStore(
        lancedb_store=lancedb_store,
        milvus_embedding_store=milvus_store,
    )

    warnings = hybrid.rename_collection_data("old-kb", "new-kb")

    assert warnings == []
    lancedb_store.rename_collection_data.assert_called_once_with("old-kb", "new-kb")
    milvus_store.rename_collection_embeddings.assert_called_once_with(
        collection_name="old-kb",
        new_name="new-kb",
    )


def test_count_embeddings_for_document_delegates_to_milvus() -> None:
    from xagent.core.tools.core.RAG_tools.storage.hybrid_vector_index_store import (
        HybridVectorIndexStore,
    )

    lancedb_store = Mock()
    milvus_store = Mock()
    milvus_store.count_embeddings_for_document.return_value = 7
    hybrid = HybridVectorIndexStore(
        lancedb_store=lancedb_store,
        milvus_embedding_store=milvus_store,
    )

    count = hybrid.count_embeddings_for_document(
        collection_name="kb1",
        doc_id="doc-1",
        user_id=5,
        is_admin=False,
    )

    assert count == 7
    milvus_store.count_embeddings_for_document.assert_called_once_with(
        collection_name="kb1",
        doc_id="doc-1",
        user_id=5,
        is_admin=False,
    )


@pytest.mark.asyncio
async def test_search_vectors_async_access_control_passes_through_to_milvus() -> None:
    from xagent.core.tools.core.RAG_tools.storage.hybrid_vector_index_store import (
        HybridVectorIndexStore,
    )

    lancedb_store = Mock()
    milvus_store = Mock()
    milvus_store.search_vectors.return_value = [{"doc_id": "doc-1"}]
    hybrid = HybridVectorIndexStore(
        lancedb_store=lancedb_store,
        milvus_embedding_store=milvus_store,
    )

    rows = await hybrid.search_vectors_async(
        table_name="embeddings_text_embedding_v4",
        query_vector=[0.1, 0.2],
        top_k=5,
        user_id=42,
        is_admin=True,
    )

    assert rows == [{"doc_id": "doc-1"}]
    milvus_store.search_vectors.assert_called_once_with(
        table_name="embeddings_text_embedding_v4",
        query_vector=[0.1, 0.2],
        top_k=5,
        filters=None,
        vector_column_name="vector",
        user_id=42,
        is_admin=True,
    )
    lancedb_store.search_vectors_async.assert_not_called()


@pytest.mark.asyncio
async def test_search_vectors_async_non_embeddings_passes_acl_to_lancedb() -> None:
    from xagent.core.tools.core.RAG_tools.storage.hybrid_vector_index_store import (
        HybridVectorIndexStore,
    )

    lancedb_store = Mock()
    lancedb_store.search_vectors_async = AsyncMock(return_value=[{"doc_id": "doc-l"}])
    milvus_store = Mock()
    hybrid = HybridVectorIndexStore(
        lancedb_store=lancedb_store,
        milvus_embedding_store=milvus_store,
    )

    rows = await hybrid.search_vectors_async(
        table_name="documents",
        query_vector=[0.1, 0.2],
        top_k=3,
        user_id=11,
        is_admin=False,
    )

    assert rows == [{"doc_id": "doc-l"}]
    lancedb_store.search_vectors_async.assert_awaited_once_with(
        table_name="documents",
        query_vector=[0.1, 0.2],
        top_k=3,
        filters=None,
        vector_column_name="vector",
        user_id=11,
        is_admin=False,
    )
    milvus_store.search_vectors.assert_not_called()
