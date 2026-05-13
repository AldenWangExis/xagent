from __future__ import annotations

import importlib
import logging
import os
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional
from uuid import uuid4

from .base import VectorStore

logger = logging.getLogger(__name__)

__all__ = [
    "MilvusConnectionManager",
    "MilvusVectorStore",
    "get_client",
    "get_client_from_env",
]


if TYPE_CHECKING:
    # NOTE:
    # Readability alias: returned runtime object is `pymilvus.MilvusClient`.
    # We keep this as `Any` in typing to avoid strict mypy failures on untyped
    # third-party imports (`disallow_any_unimported=true` in this project).
    MilvusClient = Any
else:
    MilvusClient = Any


def _import_milvus_client_class() -> Any:
    try:
        pymilvus_module = importlib.import_module("pymilvus")
    except ImportError as e:
        raise ImportError(
            "pymilvus is not installed. Please install it with: pip install pymilvus"
        ) from e
    return getattr(pymilvus_module, "MilvusClient")


class MilvusConnectionManager:
    """Milvus connection manager."""

    def get_client(
        self,
        uri: str,
        token: Optional[str] = None,
        db_name: Optional[str] = None,
    ) -> "MilvusClient":
        if not uri or not uri.strip():
            raise ValueError("Milvus uri must be non-empty")

        milvus_client_class = _import_milvus_client_class()
        return milvus_client_class(
            uri=uri.strip(),
            token=(token or "").strip(),
            db_name=(db_name or "").strip(),
        )

    def get_client_from_env(
        self,
        uri_env_var: str = "MILVUS_URI",
        token_env_var: str = "MILVUS_TOKEN",
        db_name_env_var: str = "MILVUS_DB_NAME",
    ) -> "MilvusClient":
        uri = os.getenv(uri_env_var)
        if uri is None:
            raise KeyError(f"Environment variable {uri_env_var} is not set")
        if not uri.strip():
            raise ValueError(f"Environment variable {uri_env_var} is empty")

        token = os.getenv(token_env_var)
        db_name = os.getenv(db_name_env_var)
        return self.get_client(uri=uri, token=token, db_name=db_name)


class MilvusVectorStore(VectorStore):
    """Milvus vector store implementation."""

    support_store_texts: ClassVar[bool] = True

    def __init__(
        self,
        uri: str,
        collection_name: str = "vectors",
        token: Optional[str] = None,
        db_name: Optional[str] = None,
        metric_type: str = "COSINE",
        connection_manager: Optional[MilvusConnectionManager] = None,
    ):
        self._uri = uri
        self._collection_name = collection_name
        self._token = token
        self._db_name = db_name
        self._metric_type = metric_type
        self._conn_manager = connection_manager or MilvusConnectionManager()
        self._client = self._conn_manager.get_client(
            uri=uri,
            token=token,
            db_name=db_name,
        )
        self._vector_dim: Optional[int] = None

    # Stable id strings written by xagent are sha256 hex digests (64 chars).
    # Reserve headroom (legacy sha1 = 40 chars, future migrations) without
    # bloating storage on Milvus v2.6+.
    _ID_FIELD_MAX_LENGTH: ClassVar[int] = 128

    def _ensure_collection(self, vector_dim: int) -> None:
        if vector_dim <= 0:
            raise ValueError("vector dimension must be greater than zero")

        if self._vector_dim is None:
            self._vector_dim = vector_dim

        if not self._client.has_collection(self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                dimension=vector_dim,
                primary_field_name="id",
                id_type="string",
                max_length=self._ID_FIELD_MAX_LENGTH,
                vector_field_name="vector",
                metric_type=self._metric_type,
                auto_id=False,
                enable_dynamic_field=True,
            )

    @staticmethod
    def _matches_filters(
        metadata: Dict[str, Any],
        filters: Optional[Dict[str, Any]],
    ) -> bool:
        if not filters:
            return True

        for key, value in filters.items():
            current = metadata.get(key)
            if isinstance(value, (list, tuple, set)):
                if current not in set(value):
                    return False
                continue
            if current != value:
                return False
        return True

    @staticmethod
    def _escape_filter_key(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def _format_filter_literal(value: Any) -> Optional[str]:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return None

    def _build_filter_expression(self, filters: Optional[Dict[str, Any]]) -> Optional[str]:
        if not filters:
            return None

        clauses: list[str] = []
        for key, value in filters.items():
            escaped_key = self._escape_filter_key(str(key))
            metadata_field = f'metadata["{escaped_key}"]'
            if isinstance(value, (list, tuple, set)):
                values = list(value)
                if not values:
                    return None
                literals: list[str] = []
                for item in values:
                    literal = self._format_filter_literal(item)
                    if literal is None:
                        return None
                    literals.append(literal)
                clauses.append(f"{metadata_field} in [{', '.join(literals)}]")
                continue

            literal = self._format_filter_literal(value)
            if literal is None:
                return None
            clauses.append(f"{metadata_field} == {literal}")

        if not clauses:
            return None
        return " and ".join(f"({clause})" for clause in clauses)

    def add_vectors(
        self,
        vectors: List[List[float]],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        if not vectors:
            return []

        if ids is None:
            ids = [str(uuid4()) for _ in vectors]
        elif len(ids) != len(vectors):
            raise ValueError("ids length must match vectors length")

        if metadatas is None:
            metadatas = [{} for _ in vectors]
        elif len(metadatas) != len(vectors):
            raise ValueError("metadatas length must match vectors length")

        vector_dim = len(vectors[0])
        self._ensure_collection(vector_dim=vector_dim)

        payload = []
        for i, vector in enumerate(vectors):
            if len(vector) != vector_dim:
                raise ValueError("all vectors must have the same dimension")
            payload.append(
                {
                    "id": ids[i],
                    "vector": vector,
                    "metadata": metadatas[i],
                }
            )

        self._client.insert(collection_name=self._collection_name, data=payload)
        return ids

    def delete_vectors(self, ids: List[str]) -> bool:
        if not ids:
            return True

        try:
            if not self._client.has_collection(self._collection_name):
                return True
            self._client.delete(collection_name=self._collection_name, ids=ids)
            return True
        except Exception as e:
            logger.error("Failed to delete vectors in Milvus: %s", e)
            return False

    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []
        if not query_vector:
            return []

        # search() must not create collections; that side effect belongs to the
        # write path. Returning early avoids spurious VARCHAR/index params being
        # required on read-only callers.
        if not self._client.has_collection(self._collection_name):
            return []

        filter_expr = self._build_filter_expression(filters)
        # If filter pushdown is unavailable, fetch extra candidates and filter in Python.
        limit = top_k if filter_expr else max(top_k, top_k * 5 if filters else top_k)
        # Bounded consistency keeps latency predictable on freshly-inserted data.
        # Strong waits for QueryNode to catch up to the latest write timestamp,
        # which can hang for tens of seconds on a standalone Milvus before the
        # growing segment is loaded — see Milvus issue tracker for v2.6 stalls
        # on Strong + small inserts. Bounded is the official default for dense
        # vector search and is sufficient for xagent's KB semantics.
        search_kwargs: Dict[str, Any] = {
            "collection_name": self._collection_name,
            "data": [query_vector],
            "limit": limit,
            "output_fields": ["metadata"],
            "consistency_level": "Bounded",
        }
        if filter_expr:
            search_kwargs["filter"] = filter_expr

        try:
            raw = self._client.search(**search_kwargs)
        except Exception as exc:
            if not filter_expr:
                raise
            logger.warning("Milvus filter pushdown failed, fallback to client filtering: %s", exc)
            raw = self._client.search(
                collection_name=self._collection_name,
                data=[query_vector],
                limit=max(top_k, top_k * 5 if filters else top_k),
                output_fields=["metadata"],
                consistency_level="Bounded",
            )

        hits = raw[0] if raw else []
        results: List[Dict[str, Any]] = []
        for hit in hits:
            entity = hit.get("entity", {})
            metadata = entity.get("metadata", hit.get("metadata", {}))
            if not isinstance(metadata, dict):
                metadata = {}

            if not self._matches_filters(metadata, filters):
                continue

            item_id = hit.get("id", entity.get("id"))
            score = hit.get("distance", hit.get("score", 0.0))
            results.append(
                {
                    "id": str(item_id) if item_id is not None else "",
                    "score": float(score),
                    "metadata": metadata,
                }
            )
            if len(results) >= top_k:
                break

        return results

    def query_rows(
        self,
        *,
        filters: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
        limit: int = 20000,
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        if not self._client.has_collection(self._collection_name):
            return []

        query_fn = getattr(self._client, "query", None)
        if not callable(query_fn):
            return []

        fields = output_fields or ["id", "metadata"]
        filter_expr = self._build_filter_expression(filters)
        query_kwargs: Dict[str, Any] = {
            "collection_name": self._collection_name,
            "output_fields": fields,
            "limit": limit,
        }
        if filter_expr:
            query_kwargs["filter"] = filter_expr

        try:
            raw_rows = query_fn(**query_kwargs)
        except Exception as exc:
            if not filter_expr:
                raise
            logger.warning(
                "Milvus query filter pushdown failed, fallback to client filtering: %s",
                exc,
            )
            raw_rows = query_fn(
                collection_name=self._collection_name,
                output_fields=fields,
                limit=limit,
            )

        rows: List[Dict[str, Any]] = []
        for item in raw_rows or []:
            if not isinstance(item, dict):
                continue
            entity = item.get("entity")
            if isinstance(entity, dict):
                item_id = item.get("id", entity.get("id"))
                metadata = entity.get("metadata", item.get("metadata", {}))
            else:
                item_id = item.get("id")
                metadata = item.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            if not self._matches_filters(metadata, filters):
                continue
            rows.append(
                {
                    "id": str(item_id) if item_id is not None else "",
                    "metadata": metadata,
                }
            )
        return rows

    def list_collections(self) -> List[str]:
        list_fn = getattr(self._client, "list_collections", None)
        if callable(list_fn):
            try:
                raw = list_fn()
                if isinstance(raw, str):
                    return [raw]
                if isinstance(raw, (list, tuple, set)):
                    return [str(name) for name in raw]
                return [str(name) for name in list(raw)]
            except Exception as exc:
                logger.debug("Failed to list Milvus collections: %s", exc)
        if self._client.has_collection(self._collection_name):
            return [self._collection_name]
        return []

    def clear(self) -> None:
        if not self._client.has_collection(self._collection_name):
            return
        self._client.truncate_collection(self._collection_name)


def get_client(
    uri: str,
    token: Optional[str] = None,
    db_name: Optional[str] = None,
) -> "MilvusClient":
    manager = MilvusConnectionManager()
    return manager.get_client(uri=uri, token=token, db_name=db_name)


def get_client_from_env(
    uri_env_var: str = "MILVUS_URI",
    token_env_var: str = "MILVUS_TOKEN",
    db_name_env_var: str = "MILVUS_DB_NAME",
) -> "MilvusClient":
    manager = MilvusConnectionManager()
    return manager.get_client_from_env(
        uri_env_var=uri_env_var,
        token_env_var=token_env_var,
        db_name_env_var=db_name_env_var,
    )
