"""Real-Milvus smoke test for :mod:`MilvusEmbeddingIndexStore`.

This test is **env-gated** and excluded from default CI. It only runs when
``MILVUS_URI`` is set OR when ``XAGENT_MILVUS_LITE=1`` is set.

Two supported backends:

- ``MILVUS_URI=http[s]://host:port`` — real Milvus standalone/cluster
- ``XAGENT_MILVUS_LITE=1`` — embedded milvus-lite via local file path,
  recommended for macOS / Apple Silicon where Milvus standalone runs through
  Rosetta emulation and may stall on the segment→index→load pipeline.

Covers the minimal lifecycle required by the Milvus minimal support DoD
(`docs/milvus-minimal-support-plan.md` § Acceptance Criteria item 13):

    insert -> create_index -> dense search -> tenant filter -> delete -> empty

Run locally:

    # Option A: real Milvus standalone/cluster (Linux x86_64 / CI)
    export MILVUS_URI=http://localhost:19530
    uv run pytest tests/integration/test_milvus_embedding_store_smoke.py -v -m milvus

    # Option B: milvus-lite (macOS / Apple Silicon / fast local loop)
    XAGENT_MILVUS_LITE=1 uv run pytest tests/integration/test_milvus_embedding_store_smoke.py -v -m milvus
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest

pytestmark = pytest.mark.milvus

_MILVUS_URI = os.getenv("MILVUS_URI", "").strip()
_USE_MILVUS_LITE = os.getenv("XAGENT_MILVUS_LITE", "").strip() in {"1", "true", "yes"}

if _USE_MILVUS_LITE:
    if importlib.util.find_spec("milvus_lite") is None:
        pytest.skip(
            "XAGENT_MILVUS_LITE=1 requires milvus-lite; skipping smoke test.",
            allow_module_level=True,
        )
    # milvus-lite is an embedded Milvus engine that ships with pymilvus and
    # uses a local file as the storage backend. It bypasses Docker and is
    # immune to the ARM64-emulation stalls of real Milvus standalone on
    # Apple Silicon.
    _MILVUS_URI = str(
        Path(tempfile.gettempdir()) / f"xagent_smoke_milvus_{uuid.uuid4().hex[:6]}.db"
    )
    # pymilvus reads MILVUS_URI at import time for default ORM connection
    # config; a file path is not a valid http URI for that path so we clear it
    # to keep the import side-effect quiet.
    os.environ.pop("MILVUS_URI", None)

if not _MILVUS_URI:
    pytest.skip(
        "Neither MILVUS_URI nor XAGENT_MILVUS_LITE configured; "
        "skipping real-Milvus smoke test.",
        allow_module_level=True,
    )

try:  # pragma: no cover - import guarded by smoke skip below
    from pymilvus import MilvusClient  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip(
        "pymilvus is not installed; skipping real-Milvus smoke test.",
        allow_module_level=True,
    )

from xagent.core.tools.core.RAG_tools.storage.milvus_embedding_store import (
    MilvusEmbeddingIndexStore,
)
from xagent.providers.vector_store.milvus import MilvusVectorStore


# Use a fresh model_tag per test run so the smoke test does not collide with
# other developers running against a shared Milvus instance, and it is safe to
# delete the resulting collection on teardown.
_SMOKE_RUN_ID = uuid.uuid4().hex[:8]
_MODEL_TAG = f"smoke-test-{_SMOKE_RUN_ID}"
_COLLECTION = f"kb-smoke-{_SMOKE_RUN_ID}"
_VECTOR_DIM = 8


def _drop_collection_if_exists(client: Any, collection_name: str) -> None:
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)


@pytest.fixture(scope="module")
def milvus_client() -> Iterator[Any]:
    """Direct pymilvus client used only for teardown / health probing."""
    from pymilvus import MilvusClient

    token = os.getenv("MILVUS_TOKEN", "").strip()
    db_name = os.getenv("MILVUS_DB_NAME", "").strip()
    client = MilvusClient(
        uri=_MILVUS_URI,
        token=token or None,
        db_name=db_name or None,
    )
    try:
        yield client
    finally:
        # Best-effort cleanup of any collection the test may have created.
        try:
            _drop_collection_if_exists(
                client, f"embeddings_{_MODEL_TAG.replace('-', '_')}"
            )
        except Exception:  # pragma: no cover - best effort
            pass
        # Remove the milvus-lite database file so re-runs start clean.
        if _USE_MILVUS_LITE:
            try:
                Path(_MILVUS_URI).unlink(missing_ok=True)
            except Exception:  # pragma: no cover - best effort
                pass


@pytest.fixture
def store() -> MilvusEmbeddingIndexStore:
    """Real ``MilvusEmbeddingIndexStore`` wired against a live Milvus."""
    return MilvusEmbeddingIndexStore(
        uri=_MILVUS_URI,
        token=os.getenv("MILVUS_TOKEN") or None,
        db_name=os.getenv("MILVUS_DB_NAME") or None,
        store_factory=MilvusVectorStore,
    )


def _records(*, collection: str, user_id: int) -> List[Dict[str, Any]]:
    base_vec = [0.01 * i for i in range(_VECTOR_DIM)]
    return [
        {
            "collection": collection,
            "doc_id": f"doc-{_SMOKE_RUN_ID}-1",
            "chunk_id": "chunk-1",
            "parse_hash": "ph-1",
            "model": _MODEL_TAG,
            "vector": base_vec,
            "text": "hello world from xagent smoke test",
            "chunk_hash": "ch-1",
            "metadata": {"lang": "en"},
            "created_at": "2026-05-11T00:00:00Z",
            "user_id": user_id,
        },
        {
            "collection": collection,
            "doc_id": f"doc-{_SMOKE_RUN_ID}-2",
            "chunk_id": "chunk-2",
            "parse_hash": "ph-2",
            "model": _MODEL_TAG,
            "vector": [v + 0.05 for v in base_vec],
            "text": "second chunk for the smoke test",
            "chunk_hash": "ch-2",
            "metadata": {"lang": "en"},
            "created_at": "2026-05-11T00:00:01Z",
            "user_id": user_id,
        },
    ]


def test_insert_dense_search_delete_roundtrip(
    store: MilvusEmbeddingIndexStore,
    milvus_client: Any,
) -> None:
    """Insert -> create_index -> dense search -> delete against real Milvus.

    Pins down DoD item 13: smoke covers the minimum lifecycle and asserts the
    Milvus side state observable through the storage abstraction.
    """
    records = _records(collection=_COLLECTION, user_id=42)

    # 1. Insert
    store.upsert_embeddings(_MODEL_TAG, records)

    # 2. Index ready (no-op for Milvus index in the current minimal support)
    index_result = store.create_index(_MODEL_TAG, readonly=False)
    assert index_result.status in {"index_ready", "no_index"}

    # 3. Dense search returns the inserted rows for an admin.
    hits = store.search_vectors_by_model(
        _MODEL_TAG,
        records[0]["vector"],
        top_k=5,
        is_admin=True,
    )
    assert len(hits) >= 1
    hit_ids = {row["doc_id"] for row in hits}
    assert records[0]["doc_id"] in hit_ids

    # 4. Non-admin without user_id is fail-closed.
    forbidden = store.search_vectors_by_model(
        _MODEL_TAG,
        records[0]["vector"],
        top_k=5,
        user_id=None,
        is_admin=False,
    )
    assert forbidden == []

    # 5. Non-admin with matching user_id sees rows.
    owner_hits = store.search_vectors_by_model(
        _MODEL_TAG,
        records[0]["vector"],
        top_k=5,
        user_id=42,
        is_admin=False,
    )
    assert len(owner_hits) >= 1

    # 6. Delete by collection removes the rows.
    deleted_counts = store.delete_collection_embeddings(
        collection_name=_COLLECTION,
        user_id=42,
        is_admin=False,
    )
    table_name = f"embeddings_{_MODEL_TAG.replace('-', '_')}"
    assert deleted_counts.get(table_name, 0) >= 1

    # 7. Subsequent search is empty.
    after_delete = store.search_vectors_by_model(
        _MODEL_TAG,
        records[0]["vector"],
        top_k=5,
        is_admin=True,
    )
    after_delete_ids = {row["doc_id"] for row in after_delete}
    assert records[0]["doc_id"] not in after_delete_ids
    assert records[1]["doc_id"] not in after_delete_ids


def test_search_result_shape_matches_contract(
    store: MilvusEmbeddingIndexStore,
) -> None:
    """Ensure Milvus-side fields survive the round-trip and match the contract.

    Pins down DoD items 4 and 5: written rows can be reconstructed from a real
    Milvus response (text/metadata/created_at must come back, not depend on
    the in-process cache).
    """
    collection = f"{_COLLECTION}-shape"
    records = _records(collection=collection, user_id=99)
    store.upsert_embeddings(_MODEL_TAG, records)

    # Drop the in-process cache so that any field we read must come from Milvus
    # itself. If Bug-2 ever regresses, this assertion is the canary.
    store._records.clear()  # type: ignore[attr-defined]

    hits = store.search_vectors_by_model(
        _MODEL_TAG,
        records[0]["vector"],
        top_k=1,
        is_admin=True,
    )
    assert hits, "Expected at least one hit from real Milvus"

    row = hits[0]
    required_fields = {
        "collection",
        "doc_id",
        "chunk_id",
        "text",
        "parse_hash",
        "model",
        "created_at",
        "user_id",
        "metadata",
        "_distance",
    }
    assert required_fields.issubset(row.keys())
    assert row["text"] is not None
    assert row["created_at"] is not None
    assert row["collection"] == collection

    # Cleanup
    store.delete_collection_embeddings(
        collection_name=collection,
        user_id=99,
        is_admin=False,
    )
