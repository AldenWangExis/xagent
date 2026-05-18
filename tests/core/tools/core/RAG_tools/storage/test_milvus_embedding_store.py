from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Dict, List

import pytest

from xagent.core.tools.core.RAG_tools.storage.contracts import (
    FilterCondition,
    FilterOperator,
)


@dataclass
class _FakeMilvusVectorStore:
    """Lightweight in-memory vector store used by contract tests."""

    uri: str
    collection_name: str
    token: str | None = None
    db_name: str | None = None
    registry: Dict[str, "_FakeMilvusVectorStore"] | None = None
    rows: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    vector_dim: int | None = None
    delete_batches: List[List[str]] = field(default_factory=list)
    search_filters_history: List[Dict[str, Any] | None] = field(default_factory=list)

    def add_vectors(
        self,
        vectors: List[List[float]],
        ids: List[str] | None = None,
        metadatas: List[Dict[str, Any]] | None = None,
    ) -> List[str]:
        if ids is None:
            ids = [str(i) for i in range(len(vectors))]
        if metadatas is None:
            metadatas = [{} for _ in vectors]

        if self.vector_dim is None and vectors:
            self.vector_dim = len(vectors[0])

        for idx, vector in enumerate(vectors):
            self.rows[ids[idx]] = {"vector": vector, "metadata": metadatas[idx]}
        return ids

    def delete_vectors(self, ids: List[str]) -> bool:
        self.delete_batches.append(list(ids))
        for item_id in ids:
            self.rows.pop(item_id, None)
        return True

    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        self.search_filters_history.append(dict(filters) if filters else None)
        results: List[Dict[str, Any]] = []
        for item_id, row in self.rows.items():
            metadata = row["metadata"]
            if filters:
                matched = True
                for key, value in filters.items():
                    current = metadata.get(key)
                    if isinstance(value, (list, tuple, set)):
                        if current not in set(value):
                            matched = False
                            break
                    elif current != value:
                        matched = False
                        break
                if not matched:
                    continue
            dist = sqrt(sum((a - b) ** 2 for a, b in zip(row["vector"], query_vector)))
            results.append({"id": item_id, "score": dist, "metadata": metadata})
        results.sort(key=lambda item: item["score"])
        return results[:top_k]

    def query_rows(
        self,
        *,
        filters: Dict[str, Any] | None = None,
        output_fields: List[str] | None = None,
        limit: int = 20000,
    ) -> List[Dict[str, Any]]:
        del output_fields
        if limit <= 0:
            return []
        results: List[Dict[str, Any]] = []
        for item_id, row in self.rows.items():
            metadata = row.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            if filters:
                matched = True
                for key, value in filters.items():
                    current = metadata.get(key)
                    if isinstance(value, (list, tuple, set)):
                        if current not in set(value):
                            matched = False
                            break
                    elif current != value:
                        matched = False
                        break
                if not matched:
                    continue
            results.append({"id": item_id, "metadata": metadata})
            if len(results) >= limit:
                break
        return results

    def list_collections(self) -> List[str]:
        if self.registry is not None:
            return sorted(self.registry.keys())
        return [self.collection_name]

    def clear(self) -> None:
        self.rows.clear()


@pytest.fixture
def fake_store_factory() -> tuple[Any, Dict[str, _FakeMilvusVectorStore]]:
    stores: Dict[str, _FakeMilvusVectorStore] = {}

    def _factory(
        *,
        uri: str,
        collection_name: str,
        token: str | None = None,
        db_name: str | None = None,
        metric_type: str = "COSINE",
        connection_manager: Any = None,
    ) -> _FakeMilvusVectorStore:
        del metric_type, connection_manager
        if collection_name not in stores:
            stores[collection_name] = _FakeMilvusVectorStore(
                uri=uri,
                collection_name=collection_name,
                token=token,
                db_name=db_name,
                registry=stores,
            )
        return stores[collection_name]

    return _factory, stores


@pytest.fixture
def store(fake_store_factory: tuple[Any, Dict[str, _FakeMilvusVectorStore]]) -> Any:
    from xagent.core.tools.core.RAG_tools.storage.milvus_embedding_store import (
        MilvusEmbeddingIndexStore,
    )

    factory, _ = fake_store_factory
    return MilvusEmbeddingIndexStore(
        uri="http://fake-milvus:19530", store_factory=factory
    )


def _record(
    *,
    collection: str,
    doc_id: str,
    chunk_id: str,
    vector: List[float],
    user_id: int = 1,
    text: str = "hello",
) -> Dict[str, Any]:
    return {
        "collection": collection,
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "parse_hash": "p1",
        "model": "text-embedding-v4",
        "vector": vector,
        "text": text,
        "chunk_hash": f"{chunk_id}-hash",
        "created_at": "2026-05-08T00:00:00Z",
        "metadata": {"lang": "zh"},
        "user_id": user_id,
    }


def test_create_collection_and_dimension_constraints(
    store: Any,
    fake_store_factory: tuple[Any, Dict[str, _FakeMilvusVectorStore]],
) -> None:
    _, stores = fake_store_factory
    store.upsert_embeddings(
        "text-embedding-v4",
        [_record(collection="kb1", doc_id="d1", chunk_id="c1", vector=[0.1, 0.2])],
    )
    assert "embeddings_text_embedding_v4" in stores
    target_store = stores["embeddings_text_embedding_v4"]
    assert target_store.vector_dim == 2
    first_row = next(iter(target_store.rows.values()))
    stored_metadata = first_row["metadata"]
    assert {
        "collection",
        "doc_id",
        "chunk_id",
        "parse_hash",
        "model",
        "user_id",
        "text",
        "chunk_hash",
        "metadata",
        "created_at",
        "vector_dimension",
    }.issubset(set(stored_metadata.keys()))
    assert stored_metadata["vector_dimension"] == 2
    assert stored_metadata["metadata"] == '{"lang": "zh"}'


def test_upsert_is_idempotent_same_identity(store: Any) -> None:
    row = _record(
        collection="kb1", doc_id="d1", chunk_id="c1", vector=[0.1, 0.2], text="v1"
    )
    store.upsert_embeddings("text-embedding-v4", [row])
    row_v2 = dict(row)
    row_v2["text"] = "v2"
    store.upsert_embeddings("text-embedding-v4", [row_v2])

    results = store.search_vectors_by_model(
        "text-embedding-v4",
        [0.1, 0.2],
        top_k=10,
        user_id=1,
        is_admin=False,
    )
    assert len(results) == 1
    assert results[0]["text"] == "v2"


def test_upsert_same_doc_chunk_different_users_stays_isolated(store: Any) -> None:
    row_user_1 = _record(
        collection="kb1",
        doc_id="d1",
        chunk_id="c1",
        vector=[0.1, 0.2],
        user_id=1,
        text="tenant-1",
    )
    row_user_2 = _record(
        collection="kb1",
        doc_id="d1",
        chunk_id="c1",
        vector=[0.1, 0.2],
        user_id=2,
        text="tenant-2",
    )
    store.upsert_embeddings("text-embedding-v4", [row_user_1, row_user_2])

    admin_results = store.search_vectors_by_model(
        "text-embedding-v4",
        [0.1, 0.2],
        top_k=10,
        is_admin=True,
    )
    assert len(admin_results) == 2

    user_1_results = store.search_vectors_by_model(
        "text-embedding-v4",
        [0.1, 0.2],
        top_k=10,
        user_id=1,
        is_admin=False,
    )
    user_2_results = store.search_vectors_by_model(
        "text-embedding-v4",
        [0.1, 0.2],
        top_k=10,
        user_id=2,
        is_admin=False,
    )
    assert len(user_1_results) == 1
    assert len(user_2_results) == 1
    assert user_1_results[0]["text"] == "tenant-1"
    assert user_2_results[0]["text"] == "tenant-2"


def test_upsert_deletes_legacy_and_new_keys(
    store: Any,
    fake_store_factory: tuple[Any, Dict[str, _FakeMilvusVectorStore]],
) -> None:
    _, stores = fake_store_factory
    row = _record(collection="kb1", doc_id="d1", chunk_id="c1", vector=[0.1, 0.2])
    store.upsert_embeddings("text-embedding-v4", [row])

    target_store = stores["embeddings_text_embedding_v4"]
    first_delete_batch = target_store.delete_batches[0]
    assert set(first_delete_batch) == {
        store._stable_primary_key(row),
        store._legacy_primary_key(row),
    }


def test_upsert_batches_delete_and_insert_for_multiple_rows(
    store: Any,
    fake_store_factory: tuple[Any, Dict[str, _FakeMilvusVectorStore]],
) -> None:
    _, stores = fake_store_factory
    rows = [
        _record(collection="kb1", doc_id="d1", chunk_id="c1", vector=[0.1, 0.2]),
        _record(collection="kb1", doc_id="d1", chunk_id="c2", vector=[0.2, 0.3]),
    ]

    store.upsert_embeddings("text-embedding-v4", rows)

    target_store = stores["embeddings_text_embedding_v4"]
    assert len(target_store.delete_batches) == 1
    assert len(target_store.delete_batches[0]) == 4


def test_search_works_when_local_cache_is_empty(store: Any) -> None:
    store.upsert_embeddings(
        "text-embedding-v4",
        [_record(collection="kb1", doc_id="d1", chunk_id="c1", vector=[0.1, 0.2])],
    )
    store._records.clear()

    results = store.search_vectors_by_model(
        "text-embedding-v4",
        [0.1, 0.2],
        top_k=3,
        is_admin=True,
    )

    assert len(results) == 1
    assert results[0]["doc_id"] == "d1"
    assert isinstance(results[0]["metadata"], str)
    assert '"lang": "zh"' in results[0]["metadata"]


def test_count_rows_works_when_local_cache_is_empty(store: Any) -> None:
    store.upsert_embeddings(
        "text-embedding-v4",
        [_record(collection="kb1", doc_id="d1", chunk_id="c1", vector=[0.1, 0.2])],
    )
    store._records.clear()

    count = store.count_rows(
        "embeddings_text_embedding_v4",
        filters={"collection": "kb1"},
        user_id=1,
        is_admin=False,
    )
    assert count == 1


def test_delete_document_embeddings_works_when_local_cache_is_empty(store: Any) -> None:
    store.upsert_embeddings(
        "text-embedding-v4",
        [
            _record(collection="kb1", doc_id="d1", chunk_id="c1", vector=[0.1, 0.2]),
            _record(collection="kb1", doc_id="d2", chunk_id="c2", vector=[0.1, 0.2]),
        ],
    )
    store._records.clear()

    deleted = store.delete_document_embeddings(
        collection_name="kb1",
        doc_id="d1",
        user_id=1,
        is_admin=False,
    )
    assert deleted["embeddings_text_embedding_v4"] == 1

    remaining_count = store.count_rows(
        "embeddings_text_embedding_v4",
        filters={"collection": "kb1"},
        user_id=1,
        is_admin=False,
    )
    assert remaining_count == 1


def test_iter_batches_works_when_local_cache_is_empty(store: Any) -> None:
    store.upsert_embeddings(
        "text-embedding-v4",
        [_record(collection="kb1", doc_id="d1", chunk_id="c1", vector=[0.1, 0.2])],
    )
    store._records.clear()

    batches = list(
        store.iter_batches(
            "embeddings_text_embedding_v4",
            columns=["collection", "doc_id"],
            batch_size=100,
            filters={"collection": "kb1"},
            user_id=1,
            is_admin=False,
        )
    )
    assert len(batches) == 1
    rows = batches[0].to_pylist()
    assert rows == [{"collection": "kb1", "doc_id": "d1"}]


def test_mixed_vector_dimensions_rejected(store: Any) -> None:
    rows = [
        _record(collection="kb1", doc_id="d1", chunk_id="c1", vector=[0.1, 0.2]),
        _record(collection="kb1", doc_id="d1", chunk_id="c2", vector=[0.1, 0.2, 0.3]),
    ]
    with pytest.raises(ValueError, match="dimension"):
        store.upsert_embeddings("text-embedding-v4", rows)


def test_collection_filter_applies(store: Any) -> None:
    store.upsert_embeddings(
        "text-embedding-v4",
        [
            _record(collection="kb1", doc_id="d1", chunk_id="c1", vector=[0.1, 0.2]),
            _record(collection="kb2", doc_id="d2", chunk_id="c2", vector=[0.1, 0.2]),
        ],
    )
    results = store.search_vectors_by_model(
        "text-embedding-v4",
        [0.1, 0.2],
        top_k=10,
        filters=FilterCondition("collection", FilterOperator.EQ, "kb1"),
        is_admin=True,
    )
    assert len(results) == 1
    assert results[0]["doc_id"] == "d1"


def test_search_pushes_supported_filters_to_vector_store(
    store: Any,
    fake_store_factory: tuple[Any, Dict[str, _FakeMilvusVectorStore]],
) -> None:
    _, stores = fake_store_factory
    store.upsert_embeddings(
        "text-embedding-v4",
        [
            _record(collection="kb1", doc_id="d1", chunk_id="c1", vector=[0.1, 0.2]),
            _record(collection="kb2", doc_id="d2", chunk_id="c2", vector=[0.1, 0.2]),
        ],
    )
    _ = store.search_vectors_by_model(
        "text-embedding-v4",
        [0.1, 0.2],
        top_k=5,
        filters=FilterCondition("collection", FilterOperator.EQ, "kb1"),
        is_admin=True,
    )

    target_store = stores["embeddings_text_embedding_v4"]
    assert target_store.search_filters_history
    assert target_store.search_filters_history[-1] == {"collection": "kb1"}


def test_non_admin_user_filter_is_enforced(store: Any) -> None:
    store.upsert_embeddings(
        "text-embedding-v4",
        [
            _record(
                collection="kb1",
                doc_id="d1",
                chunk_id="c1",
                vector=[0.1, 0.2],
                user_id=1,
            ),
            _record(
                collection="kb1",
                doc_id="d2",
                chunk_id="c2",
                vector=[0.1, 0.2],
                user_id=2,
            ),
        ],
    )
    results = store.search_vectors_by_model(
        "text-embedding-v4",
        [0.1, 0.2],
        top_k=10,
        user_id=1,
        is_admin=False,
    )
    assert len(results) == 1
    assert results[0]["doc_id"] == "d1"


def test_admin_path_does_not_force_user_filter(store: Any) -> None:
    store.upsert_embeddings(
        "text-embedding-v4",
        [
            _record(
                collection="kb1",
                doc_id="d1",
                chunk_id="c1",
                vector=[0.1, 0.2],
                user_id=1,
            ),
            _record(
                collection="kb1",
                doc_id="d2",
                chunk_id="c2",
                vector=[0.1, 0.2],
                user_id=2,
            ),
        ],
    )
    results = store.search_vectors_by_model(
        "text-embedding-v4",
        [0.1, 0.2],
        top_k=10,
        is_admin=True,
    )
    assert len(results) == 2


def test_non_admin_without_user_id_is_fail_closed(store: Any) -> None:
    store.upsert_embeddings(
        "text-embedding-v4",
        [
            _record(
                collection="kb1",
                doc_id="d1",
                chunk_id="c1",
                vector=[0.1, 0.2],
                user_id=1,
            )
        ],
    )
    results = store.search_vectors_by_model(
        "text-embedding-v4",
        [0.1, 0.2],
        top_k=10,
        user_id=None,
        is_admin=False,
    )
    assert results == []


def test_unsupported_filter_operator_is_fail_closed(store: Any) -> None:
    store.upsert_embeddings(
        "text-embedding-v4",
        [_record(collection="kb1", doc_id="d1", chunk_id="c1", vector=[0.1, 0.2])],
    )
    results = store.search_vectors_by_model(
        "text-embedding-v4",
        [0.1, 0.2],
        top_k=10,
        filters=FilterCondition("doc_id", FilterOperator.GT, "d1"),
        is_admin=True,
    )
    assert results == []


def test_delete_collection_embeddings_scope(store: Any) -> None:
    store.upsert_embeddings(
        "text-embedding-v4",
        [
            _record(collection="kb1", doc_id="d1", chunk_id="c1", vector=[0.1, 0.2]),
            _record(collection="kb2", doc_id="d2", chunk_id="c2", vector=[0.1, 0.2]),
        ],
    )

    deleted = store.delete_collection_embeddings("kb1", user_id=1, is_admin=False)
    assert deleted["embeddings_text_embedding_v4"] == 1

    remaining = store.search_vectors_by_model(
        "text-embedding-v4",
        [0.1, 0.2],
        top_k=10,
        is_admin=True,
    )
    assert len(remaining) == 1
    assert remaining[0]["collection"] == "kb2"


def test_delete_document_embeddings_scope(store: Any) -> None:
    store.upsert_embeddings(
        "text-embedding-v4",
        [
            _record(collection="kb1", doc_id="d1", chunk_id="c1", vector=[0.1, 0.2]),
            _record(collection="kb1", doc_id="d2", chunk_id="c2", vector=[0.1, 0.2]),
        ],
    )
    deleted = store.delete_document_embeddings(
        collection_name="kb1",
        doc_id="d1",
        user_id=1,
        is_admin=False,
    )
    assert deleted["embeddings_text_embedding_v4"] == 1

    remaining = store.search_vectors_by_model(
        "text-embedding-v4",
        [0.1, 0.2],
        top_k=10,
        is_admin=True,
    )
    assert len(remaining) == 1
    assert remaining[0]["doc_id"] == "d2"


def test_create_index_readonly_does_not_mutate(store: Any) -> None:
    readonly_result = store.create_index("text-embedding-v4", readonly=True)
    assert readonly_result.status == "readonly"

    active_result = store.create_index("text-embedding-v4", readonly=False)
    assert active_result.status in {"index_ready", "no_index"}


def test_dense_search_result_fields_follow_contract(store: Any) -> None:
    store.upsert_embeddings(
        "text-embedding-v4",
        [_record(collection="kb1", doc_id="d1", chunk_id="c1", vector=[0.1, 0.2])],
    )
    results = store.search_vectors_by_model(
        "text-embedding-v4",
        [0.1, 0.2],
        top_k=1,
        is_admin=True,
    )
    assert len(results) == 1
    row = results[0]
    assert {"doc_id", "chunk_id", "text", "_distance", "metadata"}.issubset(row.keys())
