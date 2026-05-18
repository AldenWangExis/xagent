"""Milvus-backed embedding-only storage helper.

This module intentionally focuses on the embedding plane only. It does not
manage documents/parses/chunks control data, which remains delegated to the
LanceDB stores via ``HybridVectorIndexStore``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import Any, Dict, Optional, Sequence

import pyarrow as pa  # type: ignore[import-not-found,unused-ignore]

from xagent.providers.vector_store.milvus import MilvusVectorStore

from ..core.exceptions import ConfigurationError, DatabaseOperationError
from ..core.schemas import IndexResult
from ..LanceDB.model_tag_utils import to_model_tag
from .contracts import FilterCondition, FilterExpression, FilterOperator

StoreFactory = Callable[..., MilvusVectorStore]


class MilvusEmbeddingIndexStore:
    """Embedding-plane helper for Milvus operations."""

    def __init__(
        self,
        uri: str | None = None,
        *,
        token: str | None = None,
        db_name: str | None = None,
        store_factory: StoreFactory | None = None,
    ) -> None:
        resolved_uri_raw = uri if uri is not None else os.getenv("MILVUS_URI", "")
        resolved_uri = resolved_uri_raw.strip()
        if not resolved_uri:
            raise ConfigurationError(
                "MILVUS_URI must be configured when XAGENT_VECTOR_BACKEND=milvus."
            )

        self._uri = resolved_uri
        self._token = token or os.getenv("MILVUS_TOKEN")
        self._db_name = db_name or os.getenv("MILVUS_DB_NAME")
        self._store_factory: StoreFactory = store_factory or MilvusVectorStore

        self._stores: dict[str, MilvusVectorStore] = {}
        self._records: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self._vector_dimensions: dict[str, int] = {}
        self._known_tables: set[str] = set()
        self._probe_store: MilvusVectorStore | None = None

    @staticmethod
    def _table_name(model_tag: str) -> str:
        return f"embeddings_{to_model_tag(model_tag)}"

    @staticmethod
    def _stable_primary_key(record: dict[str, Any]) -> str:
        parts = [
            str(record.get("collection", "")),
            str(record.get("doc_id", "")),
            str(record.get("chunk_id", "")),
            str(record.get("parse_hash", "")),
            str(record.get("model", "")),
            str(record.get("user_id", "")),
        ]
        payload = "\0".join(parts).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _legacy_primary_key(record: dict[str, Any]) -> str:
        parts = [
            str(record.get("collection", "")),
            str(record.get("doc_id", "")),
            str(record.get("chunk_id", "")),
            str(record.get("parse_hash", "")),
            str(record.get("model", "")),
            str(record.get("user_id", "")),
        ]
        payload = "|".join(parts).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

    @staticmethod
    def _normalize_metadata(record: dict[str, Any]) -> Optional[str]:
        raw = record.get("metadata")
        if raw is None:
            return None
        if isinstance(raw, str):
            return raw
        if isinstance(raw, dict):
            try:
                return json.dumps(raw, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _normalize_created_at(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:  # noqa: BLE001
                return str(value)
        return value

    @staticmethod
    def _coerce_user_id(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _build_storage_metadata(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "collection": row.get("collection"),
            "doc_id": row.get("doc_id"),
            "chunk_id": row.get("chunk_id"),
            "parse_hash": row.get("parse_hash"),
            "model": row.get("model"),
            "user_id": row.get("user_id"),
            "text": row.get("text"),
            "chunk_hash": row.get("chunk_hash"),
            "metadata": row.get("metadata"),
            "created_at": row.get("created_at"),
            "vector_dimension": row.get("vector_dimension"),
        }

    def _ensure_store(self, model_tag: str) -> tuple[str, MilvusVectorStore]:
        table_name = self._table_name(model_tag)
        store = self._stores.get(table_name)
        if store is None:
            store = self._store_factory(
                uri=self._uri,
                collection_name=table_name,
                token=self._token,
                db_name=self._db_name,
            )
            self._stores[table_name] = store
        return table_name, store

    def _ensure_probe_store(self) -> MilvusVectorStore:
        if self._probe_store is None:
            self._probe_store = self._store_factory(
                uri=self._uri,
                collection_name="__xagent_probe__",
                token=self._token,
                db_name=self._db_name,
            )
        return self._probe_store

    @staticmethod
    def _extract_model_tag_from_table(table_name: str) -> str:
        if table_name.startswith("embeddings_"):
            return table_name[len("embeddings_") :]
        return table_name

    def _vector_dim_for_records(self, records: list[dict[str, Any]]) -> int:
        dims = {
            len(record["vector"])
            for record in records
            if isinstance(record.get("vector"), (list, tuple))
        }
        if not dims:
            raise ValueError("Embedding records must include non-empty vector values.")
        if len(dims) != 1:
            raise ValueError(
                f"Mixed vector dimension payload is not allowed: {sorted(dims)}"
            )
        return next(iter(dims))

    @staticmethod
    def _predicate_for_condition(
        condition: FilterCondition,
    ) -> tuple[Callable[[dict[str, Any]], bool], bool]:
        field = condition.field
        operator = condition.operator
        value = condition.value

        if operator is FilterOperator.EQ:
            return (lambda row: row.get(field) == value), True
        if operator is FilterOperator.IN:
            values = set(value if isinstance(value, (list, tuple, set)) else [])
            return (lambda row: row.get(field) in values), True
        return (lambda _row: False), False

    def _compile_filter(
        self,
        filters: Optional[FilterExpression],
    ) -> tuple[Callable[[dict[str, Any]], bool], bool]:
        if filters is None:
            return (lambda _row: True), True

        if isinstance(filters, FilterCondition):
            return self._predicate_for_condition(filters)

        if isinstance(filters, tuple):
            predicates: list[Callable[[dict[str, Any]], bool]] = []
            for item in filters:
                predicate, supported = self._compile_filter(item)
                if not supported:
                    return (lambda _row: False), False
                predicates.append(predicate)
            return (lambda row: all(pred(row) for pred in predicates)), True

        if isinstance(filters, list):
            or_predicates: list[Callable[[dict[str, Any]], bool]] = []
            for item in filters:
                predicate, supported = self._compile_filter(item)
                if not supported:
                    return (lambda _row: False), False
                or_predicates.append(predicate)
            return (lambda row: any(pred(row) for pred in or_predicates)), True

        return (lambda _row: False), False

    @staticmethod
    def _matches_dict_filters(
        row: dict[str, Any],
        filters: Optional[dict[str, Any]],
    ) -> bool:
        if not filters:
            return True
        for key, value in filters.items():
            current = row.get(key)
            if isinstance(value, (list, tuple, set)):
                if current not in set(value):
                    return False
                continue
            if current != value:
                return False
        return True

    def _collect_store_filters(
        self,
        filters: Optional[FilterExpression],
    ) -> tuple[dict[str, Any], bool]:
        if filters is None:
            return {}, True

        if isinstance(filters, FilterCondition):
            if filters.operator is FilterOperator.EQ:
                return {filters.field: filters.value}, True
            if filters.operator is FilterOperator.IN:
                values = filters.value
                if isinstance(values, (list, tuple, set)):
                    return {filters.field: list(values)}, True
            return {}, False

        if isinstance(filters, tuple):
            merged: dict[str, Any] = {}
            for item in filters:
                item_filters, supported = self._collect_store_filters(item)
                if not supported:
                    return {}, False
                for key, value in item_filters.items():
                    if key in merged and merged[key] != value:
                        return {}, False
                    merged[key] = value
            return merged, True

        if isinstance(filters, list):
            # OR expressions are not representable in provider pushdown dict.
            return {}, False

        return {}, False

    def _query_rows_for_table(
        self,
        table_name: str,
        *,
        filters: Optional[dict[str, Any]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> tuple[str, MilvusVectorStore, list[tuple[str, dict[str, Any]]]]:
        model_tag = self._extract_model_tag_from_table(table_name)
        resolved_table_name, store = self._ensure_store(model_tag)

        query_rows_fn = getattr(store, "query_rows", None)
        if callable(query_rows_fn):
            raw_rows = query_rows_fn(filters=filters, output_fields=["id", "metadata"])
        else:
            raw_rows = [
                {"id": item_id, "metadata": row}
                for item_id, row in self._records.get(resolved_table_name, {}).items()
            ]

        matched_rows: list[tuple[str, dict[str, Any]]] = []
        for item in raw_rows:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id", "")).strip()
            metadata = item.get("metadata")
            if not item_id or not isinstance(metadata, dict):
                continue
            row = dict(metadata)
            row["user_id"] = self._coerce_user_id(row.get("user_id"))
            if not self._matches_dict_filters(row, filters):
                continue
            if not self._passes_access_control(row, user_id=user_id, is_admin=is_admin):
                continue
            matched_rows.append((item_id, row))
            # Keep a best-effort cache for fallback paths without making
            # correctness depend on process-local state.
            self._records[resolved_table_name][item_id] = row
            self._known_tables.add(resolved_table_name)

        return resolved_table_name, store, matched_rows

    @staticmethod
    def _passes_access_control(
        row: dict[str, Any],
        *,
        user_id: Optional[int],
        is_admin: bool,
    ) -> bool:
        if is_admin:
            return True
        if user_id is None:
            return False
        return MilvusEmbeddingIndexStore._coerce_user_id(row.get("user_id")) == user_id

    @staticmethod
    def _to_result_row(row: dict[str, Any], distance: float) -> dict[str, Any]:
        metadata_value = row.get("metadata")
        if metadata_value is not None and not isinstance(metadata_value, str):
            if isinstance(metadata_value, dict):
                try:
                    metadata_value = json.dumps(
                        metadata_value, ensure_ascii=False, sort_keys=True
                    )
                except (TypeError, ValueError):
                    metadata_value = None
            else:
                metadata_value = None
        return {
            "collection": row.get("collection"),
            "doc_id": row.get("doc_id"),
            "chunk_id": row.get("chunk_id"),
            "text": row.get("text"),
            "parse_hash": row.get("parse_hash"),
            "model": row.get("model"),
            "created_at": row.get("created_at"),
            "user_id": MilvusEmbeddingIndexStore._coerce_user_id(row.get("user_id")),
            "metadata": metadata_value,
            "_distance": distance,
        }

    def _resolve_row_from_hit(
        self,
        *,
        table_name: str,
        hit: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        item_id = str(hit.get("id", ""))
        if not item_id:
            return None

        resolved: dict[str, Any] = {}
        hit_metadata = hit.get("metadata")
        if isinstance(hit_metadata, dict):
            resolved.update(hit_metadata)

        cached = self._records.get(table_name, {}).get(item_id)
        if isinstance(cached, dict):
            for key, value in cached.items():
                resolved.setdefault(key, value)

        if not resolved:
            return None

        resolved["user_id"] = self._coerce_user_id(resolved.get("user_id"))
        return resolved

    def upsert_embeddings(self, model_tag: str, records: list[dict[str, Any]]) -> None:
        if not records:
            return

        table_name, store = self._ensure_store(model_tag)
        vector_dim = self._vector_dim_for_records(records)

        expected_dim = self._vector_dimensions.get(table_name)
        if expected_dim is not None and expected_dim != vector_dim:
            raise ValueError(
                f"Vector dimension mismatch for {table_name}: expected {expected_dim}, got {vector_dim}"
            )
        self._vector_dimensions[table_name] = vector_dim
        self._known_tables.add(table_name)

        normalized_rows: list[dict[str, Any]] = []
        row_ids: list[tuple[str, str]] = []
        ids_to_delete: list[str] = []

        for record in records:
            row = dict(record)
            row["model"] = row.get("model") or model_tag
            row["metadata"] = self._normalize_metadata(row)
            row["created_at"] = self._normalize_created_at(row.get("created_at"))
            row["user_id"] = self._coerce_user_id(row.get("user_id"))
            row["vector_dimension"] = row.get("vector_dimension") or len(row["vector"])
            normalized_rows.append(row)

            item_id = self._stable_primary_key(row)
            legacy_id = self._legacy_primary_key(row)
            row_ids.append((item_id, legacy_id))
            ids_to_delete.append(item_id)
            if legacy_id != item_id:
                ids_to_delete.append(legacy_id)

        # Keep writes idempotent by replacing previous identity rows in batch.
        if ids_to_delete:
            # Preserve order while removing duplicates.
            deduped_ids = list(dict.fromkeys(ids_to_delete))
            store.delete_vectors(deduped_ids)

        store.add_vectors(
            vectors=[row["vector"] for row in normalized_rows],
            ids=[item_id for item_id, _ in row_ids],
            metadatas=[self._build_storage_metadata(row) for row in normalized_rows],
        )

        for row, (item_id, legacy_id) in zip(normalized_rows, row_ids, strict=False):
            self._records[table_name].pop(legacy_id, None)
            self._records[table_name][item_id] = row

    def create_index(self, model_tag: str, readonly: bool = False) -> IndexResult:
        table_name = self._table_name(model_tag)
        if readonly:
            return IndexResult(
                status="readonly",
                advice=(
                    "Readonly mode enabled for Milvus backend. "
                    "Skipping index creation operations."
                ),
                fts_enabled=False,
            )
        if self.count_rows(table_name, user_id=None, is_admin=True) > 0:
            self._known_tables.add(table_name)
            return IndexResult(status="index_ready", advice=None, fts_enabled=False)
        return IndexResult(
            status="no_index",
            advice="No embeddings found for this model in Milvus backend.",
            fts_enabled=False,
        )

    def open_embeddings_table(self, model_tag: str) -> tuple[None, str]:
        table_name = self._table_name(model_tag)
        if table_name not in self.list_table_names():
            raise DatabaseOperationError(
                f"Embeddings table not found for model_tag='{model_tag}' in Milvus backend."
            )
        return None, table_name

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
        table_name = self._table_name(model_tag)
        return self.search_vectors(
            table_name=table_name,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            vector_column_name=vector_column_name,
            user_id=user_id,
            is_admin=is_admin,
        )

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
        del vector_column_name
        if top_k <= 0:
            return []
        if not is_admin and user_id is None:
            return []

        model_tag = self._extract_model_tag_from_table(table_name)
        resolved_table_name, store = self._ensure_store(model_tag)

        predicate, supported = self._compile_filter(filters)
        if not supported:
            return []

        store_filters, pushdown_supported = self._collect_store_filters(filters)
        if not is_admin and user_id is not None:
            # Tenant scope can be pushed down for narrower candidate selection.
            store_filters = dict(store_filters)
            store_filters["user_id"] = user_id
            pushdown_supported = True

        candidate_top_k = top_k if pushdown_supported else max(top_k * 5, top_k)
        raw_hits = store.search_vectors(
            query_vector,
            top_k=candidate_top_k,
            filters=store_filters if pushdown_supported else None,
        )
        results: list[dict[str, Any]] = []
        for hit in raw_hits:
            row = self._resolve_row_from_hit(table_name=resolved_table_name, hit=hit)
            if row is None:
                continue
            if not self._passes_access_control(row, user_id=user_id, is_admin=is_admin):
                continue
            if not predicate(row):
                continue
            distance = float(hit.get("score", 0.0))
            results.append(self._to_result_row(row, distance))
            if len(results) >= top_k:
                break
        return results

    def list_table_names(self) -> Sequence[str]:
        names = set(self._known_tables)
        names.update(self._records.keys())
        names.update(self._stores.keys())
        try:
            probe_store = self._ensure_probe_store()
            list_collections = getattr(probe_store, "list_collections", None)
            if callable(list_collections):
                for collection_name in list_collections():
                    normalized = str(collection_name).strip()
                    if normalized.startswith("embeddings_"):
                        names.add(normalized)
        except Exception:  # noqa: BLE001
            # Best-effort discovery only; callers still get cached names.
            pass
        return sorted(name for name in names if name.startswith("embeddings_"))

    def get_vector_dimension(self, table_name: str) -> Optional[int]:
        return self._vector_dimensions.get(table_name)

    def iter_batches(
        self,
        table_name: str,
        columns: Optional[Sequence[str]] = None,
        batch_size: int = 1000,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> Iterator[Any]:
        _resolved_table_name, _store, matched_rows = self._query_rows_for_table(
            table_name,
            filters=filters,
            user_id=user_id,
            is_admin=is_admin,
        )
        rows = []
        for _item_id, row in matched_rows:
            if columns:
                rows.append({key: row.get(key) for key in columns})
            else:
                rows.append(dict(row))

        if batch_size <= 0:
            batch_size = 1000

        for idx in range(0, len(rows), batch_size):
            batch_rows = rows[idx : idx + batch_size]
            if not batch_rows:
                continue
            yield pa.RecordBatch.from_pylist(batch_rows)

    def count_rows(
        self,
        table_name: str,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> int:
        _resolved_table_name, _store, matched_rows = self._query_rows_for_table(
            table_name,
            filters=filters,
            user_id=user_id,
            is_admin=is_admin,
        )
        return len(matched_rows)

    def delete_collection_embeddings(
        self,
        collection_name: str,
        user_id: Optional[int],
        is_admin: bool,
    ) -> dict[str, int]:
        deleted: dict[str, int] = {}
        for table_name in self.list_table_names():
            resolved_table_name, store, matched_rows = self._query_rows_for_table(
                table_name,
                filters={"collection": collection_name},
                user_id=user_id,
                is_admin=is_admin,
            )
            item_ids = [item_id for item_id, _row in matched_rows]
            if not item_ids:
                continue
            store.delete_vectors(item_ids)
            records = self._records.get(resolved_table_name)
            if records is not None:
                for item_id in item_ids:
                    records.pop(item_id, None)
            deleted[resolved_table_name] = len(item_ids)
        return deleted

    def delete_document_embeddings(
        self,
        *,
        collection_name: str,
        doc_id: str,
        user_id: Optional[int],
        is_admin: bool,
    ) -> dict[str, int]:
        deleted: dict[str, int] = {}
        for table_name in self.list_table_names():
            resolved_table_name, store, matched_rows = self._query_rows_for_table(
                table_name,
                filters={"collection": collection_name, "doc_id": doc_id},
                user_id=user_id,
                is_admin=is_admin,
            )
            item_ids = [item_id for item_id, _row in matched_rows]
            if not item_ids:
                continue
            store.delete_vectors(item_ids)
            records = self._records.get(resolved_table_name)
            if records is not None:
                for item_id in item_ids:
                    records.pop(item_id, None)
            deleted[resolved_table_name] = len(item_ids)
        return deleted

    def rename_collection_embeddings(
        self,
        *,
        collection_name: str,
        new_name: str,
    ) -> dict[str, int]:
        if not collection_name or collection_name == new_name:
            return {}

        renamed_counts: dict[str, int] = {}
        for table_name, records in list(self._records.items()):
            matching_rows: list[dict[str, Any]] = []
            old_ids: list[str] = []
            for item_id, row in list(records.items()):
                if row.get("collection") != collection_name:
                    continue
                updated_row = dict(row)
                updated_row["collection"] = new_name
                matching_rows.append(updated_row)
                old_ids.append(item_id)
                records.pop(item_id, None)

            if not old_ids:
                continue
            self._stores[table_name].delete_vectors(old_ids)
            for row in matching_rows:
                model_tag = str(row.get("model", ""))
                self.upsert_embeddings(model_tag, [row])
            renamed_counts[table_name] = len(old_ids)

        return renamed_counts

    def count_embeddings_by_collection(
        self,
        *,
        user_id: Optional[int],
        is_admin: bool,
    ) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for table_name in self.list_table_names():
            _resolved_table_name, _store, matched_rows = self._query_rows_for_table(
                table_name,
                user_id=user_id,
                is_admin=is_admin,
            )
            for _item_id, row in matched_rows:
                collection_name = str(row.get("collection", "")).strip()
                if collection_name:
                    counts[collection_name] += 1
        return dict(counts)

    def count_embeddings_for_document(
        self,
        *,
        collection_name: str,
        doc_id: str,
        user_id: Optional[int],
        is_admin: bool,
    ) -> int:
        count = 0
        for table_name in self.list_table_names():
            _resolved_table_name, _store, matched_rows = self._query_rows_for_table(
                table_name,
                filters={"collection": collection_name, "doc_id": doc_id},
                user_id=user_id,
                is_admin=is_admin,
            )
            count += len(matched_rows)
        return count
