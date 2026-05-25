"""Tests for the KB semantic coordinator skeleton.

The scenarios in this file intentionally exercise only the #495 skeleton:
context resolution, backend binding resolution, handle creation, and test reset
isolation. They must not depend on API, pipeline, search, or lifecycle migration.
"""

from __future__ import annotations

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
from xagent.core.tools.core.RAG_tools.storage.factory import (
    StorageFactory,
    reset_rag_storage_for_tests,
)
from xagent.core.tools.core.RAG_tools.utils.user_scope import user_scope_context


def test_kb_public_surface_imports() -> None:
    """Given the new semantic package, all #495 public symbols are importable."""
    import xagent.core.tools.core.RAG_tools.kb as kb

    expected_symbols = [
        "KBCoordinator",
        "get_kb_coordinator",
        "reset_kb_coordinator_for_tests",
        "KBContextRequest",
        "KBUserScope",
        "KBAccessMode",
        "KBCollectionContext",
        "KBStorageBackend",
        "KBBackendCapabilities",
        "KBHandleProvider",
        "LanceDBCollectionHandle",
    ]

    for symbol in expected_symbols:
        assert hasattr(kb, symbol)


def test_kb_access_mode_values_are_stable() -> None:
    """Given #495 access modes, their persisted string values stay stable."""
    from xagent.core.tools.core.RAG_tools.kb import KBAccessMode

    assert KBAccessMode.READ.value == "read"
    assert KBAccessMode.WRITE.value == "write"
    assert KBAccessMode.ADMIN.value == "admin"


def test_lancedb_backend_capabilities_are_explicit() -> None:
    """Given LanceDB fallback, capabilities explicitly describe supported areas."""
    from xagent.core.tools.core.RAG_tools.kb import KBBackendCapabilities

    capabilities = KBBackendCapabilities.lancedb()

    assert capabilities.supports_documents is True
    assert capabilities.supports_parses is True
    assert capabilities.supports_chunks is True
    assert capabilities.supports_embeddings is True
    assert capabilities.supports_search is True
    assert capabilities.supports_versions is True
    assert capabilities.supports_raw_connection is True


def test_get_kb_coordinator_returns_process_singleton() -> None:
    """Given process-global access, get_kb_coordinator returns a singleton."""
    from xagent.core.tools.core.RAG_tools.kb import (
        get_kb_coordinator,
        reset_kb_coordinator_for_tests,
    )

    reset_kb_coordinator_for_tests()

    first = get_kb_coordinator()
    second = get_kb_coordinator()

    assert first is second


def test_reset_kb_coordinator_for_tests_clears_singleton() -> None:
    """Given a coordinator singleton, reset creates a fresh instance."""
    from xagent.core.tools.core.RAG_tools.kb import (
        get_kb_coordinator,
        reset_kb_coordinator_for_tests,
    )

    reset_kb_coordinator_for_tests()
    first = get_kb_coordinator()

    reset_kb_coordinator_for_tests()
    second = get_kb_coordinator()

    assert second is not first


@pytest.mark.asyncio
async def test_get_context_uses_explicit_user_scope() -> None:
    """Given explicit user scope, get_context uses it over context defaults."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBContextRequest,
        get_kb_coordinator,
    )

    request = KBContextRequest(
        collection="docs", user_id=123, is_admin=True, hide_missing=True
    )

    context = await get_kb_coordinator().get_context(request)

    assert context.user_scope.user_id == 123
    assert context.user_scope.is_admin is True


@pytest.mark.asyncio
async def test_get_context_falls_back_to_request_user_scope() -> None:
    """Given request context scope, get_context uses it when request omits scope."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBContextRequest,
        get_kb_coordinator,
    )

    with user_scope_context(user_id=456, is_admin=False):
        context = await get_kb_coordinator().get_context(
            KBContextRequest(collection="docs", hide_missing=True)
        )

    assert context.user_scope.user_id == 456
    assert context.user_scope.is_admin is False


@pytest.mark.asyncio
async def test_get_context_strips_collection_name() -> None:
    """Given whitespace around collection, get_context normalizes it."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBContextRequest,
        get_kb_coordinator,
    )

    context = await get_kb_coordinator().get_context(
        KBContextRequest(collection="  docs  ", hide_missing=True)
    )

    assert context.collection == "docs"


@pytest.mark.asyncio
async def test_get_context_rejects_empty_collection_name() -> None:
    """Given an empty collection name, get_context raises a clear ValueError."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBContextRequest,
        get_kb_coordinator,
    )

    with pytest.raises(ValueError, match="collection"):
        await get_kb_coordinator().get_context(
            KBContextRequest(collection="   ", hide_missing=True)
        )


@pytest.mark.asyncio
async def test_get_context_hides_missing_collection_when_requested() -> None:
    """Given hide_missing, a missing collection returns a LanceDB fallback context."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBContextRequest,
        KBStorageBackend,
        get_kb_coordinator,
    )

    context = await get_kb_coordinator().get_context(
        KBContextRequest(collection="missing", hide_missing=True)
    )

    assert context.collection_info is None
    assert context.backend is KBStorageBackend.LANCEDB


@pytest.mark.asyncio
async def test_get_context_allows_missing_collection_for_create_without_writing_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given allow_create, missing collection returns context without metadata writes."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBContextRequest,
        get_kb_coordinator,
    )

    metadata_store = StorageFactory.get_factory().get_metadata_store()
    save_calls = 0

    async def fail_if_saved(collection: CollectionInfo) -> None:
        nonlocal save_calls
        save_calls += 1
        raise AssertionError(f"unexpected metadata write: {collection.name}")

    monkeypatch.setattr(metadata_store, "save_collection", fail_if_saved)

    context = await get_kb_coordinator().get_context(
        KBContextRequest(collection="new_collection", allow_create=True)
    )

    assert context.collection_info is None
    assert save_calls == 0


@pytest.mark.asyncio
async def test_get_context_raises_for_missing_collection_by_default() -> None:
    """Given default flags, missing collection is a visible error."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBContextRequest,
        get_kb_coordinator,
    )

    with pytest.raises(ValueError, match="missing"):
        await get_kb_coordinator().get_context(KBContextRequest(collection="missing"))


@pytest.mark.asyncio
async def test_get_context_defaults_legacy_collection_without_kb_storage_to_lancedb() -> (
    None
):
    """Given legacy metadata without kb_storage, context falls back to LanceDB."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBContextRequest,
        KBStorageBackend,
        get_kb_coordinator,
    )

    metadata_store = StorageFactory.get_factory().get_metadata_store()
    await metadata_store.save_collection(CollectionInfo(name="legacy"))

    context = await get_kb_coordinator().get_context(
        KBContextRequest(collection="legacy")
    )

    assert context.collection_info is not None
    assert context.collection_info.name == "legacy"
    assert context.backend is KBStorageBackend.LANCEDB


@pytest.mark.asyncio
async def test_get_context_does_not_write_default_binding_for_legacy_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given missing binding, get_context does not persist an implicit default."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBContextRequest,
        get_kb_coordinator,
    )

    metadata_store = StorageFactory.get_factory().get_metadata_store()
    await metadata_store.save_collection(CollectionInfo(name="legacy"))

    save_calls = 0

    async def fail_if_saved(collection: CollectionInfo) -> None:
        nonlocal save_calls
        save_calls += 1
        raise AssertionError(f"unexpected metadata write: {collection.name}")

    monkeypatch.setattr(metadata_store, "save_collection", fail_if_saved)

    context = await get_kb_coordinator().get_context(
        KBContextRequest(collection="legacy")
    )

    assert context.collection_info is not None
    assert save_calls == 0


@pytest.mark.asyncio
async def test_get_context_resolves_dict_kb_storage_binding() -> None:
    """Given dict kb_storage binding, get_context resolves its backend."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBContextRequest,
        KBStorageBackend,
        get_kb_coordinator,
    )

    metadata_store = StorageFactory.get_factory().get_metadata_store()
    await metadata_store.save_collection(
        CollectionInfo(
            name="bound",
            extra_metadata={"kb_storage": {"backend": "lancedb", "version": 1}},
        )
    )

    context = await get_kb_coordinator().get_context(
        KBContextRequest(collection="bound")
    )

    assert context.backend is KBStorageBackend.LANCEDB


@pytest.mark.asyncio
async def test_get_context_resolves_string_kb_storage_binding() -> None:
    """Given string kb_storage binding, get_context resolves its backend."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBContextRequest,
        KBStorageBackend,
        get_kb_coordinator,
    )

    metadata_store = StorageFactory.get_factory().get_metadata_store()
    await metadata_store.save_collection(
        CollectionInfo(name="bound", extra_metadata={"kb_storage": "lancedb"})
    )

    context = await get_kb_coordinator().get_context(
        KBContextRequest(collection="bound")
    )

    assert context.backend is KBStorageBackend.LANCEDB


@pytest.mark.asyncio
async def test_get_context_rejects_invalid_kb_storage_binding_shape() -> None:
    """Given invalid kb_storage shape, get_context fails explicitly."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBContextRequest,
        get_kb_coordinator,
    )

    metadata_store = StorageFactory.get_factory().get_metadata_store()
    await metadata_store.save_collection(
        CollectionInfo(name="broken", extra_metadata={"kb_storage": ["lancedb"]})
    )

    with pytest.raises(ValueError, match="kb_storage"):
        await get_kb_coordinator().get_context(KBContextRequest(collection="broken"))


def test_open_collection_rejects_unsupported_backend() -> None:
    """Given an unsupported backend, the handle provider raises a clear error."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBAccessMode,
        KBBackendCapabilities,
        KBCollectionContext,
        KBHandleProvider,
        KBStorageBackend,
        KBUserScope,
    )

    factory = StorageFactory.get_factory()
    context = KBCollectionContext(
        collection="future",
        user_scope=KBUserScope(user_id=1, is_admin=False),
        access_mode=KBAccessMode.READ,
        allow_create=False,
        hide_missing=False,
        metadata_store=factory.get_metadata_store(),
        vector_index_store=factory.get_vector_index_store(),
        backend=KBStorageBackend.MILVUS,
        capabilities=KBBackendCapabilities.unsupported(),
        collection_info=None,
    )

    with pytest.raises(ValueError, match="milvus"):
        KBHandleProvider().open(context)


@pytest.mark.asyncio
async def test_open_collection_returns_lancedb_handle() -> None:
    """Given LanceDB context, open_collection returns a LanceDB handle."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBContextRequest,
        LanceDBCollectionHandle,
        get_kb_coordinator,
    )

    handle = await get_kb_coordinator().open_collection(
        KBContextRequest(collection="docs", hide_missing=True)
    )

    assert isinstance(handle, LanceDBCollectionHandle)


@pytest.mark.asyncio
async def test_lancedb_collection_handle_exposes_context_and_stores() -> None:
    """Given a LanceDB handle, it exposes context and its factory-backed stores."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBContextRequest,
        get_kb_coordinator,
    )

    factory = StorageFactory.get_factory()

    handle = await get_kb_coordinator().open_collection(
        KBContextRequest(collection="docs", hide_missing=True)
    )

    assert handle.context.collection == "docs"
    assert handle.metadata_store is factory.get_metadata_store()
    assert handle.vector_index_store is factory.get_vector_index_store()
    assert handle.backend is handle.context.backend
    assert handle.capabilities is handle.context.capabilities


def test_get_context_sync_resolves_context() -> None:
    """Given sync legacy code, get_context_sync resolves equivalent context."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBContextRequest,
        get_kb_coordinator,
    )

    context = get_kb_coordinator().get_context_sync(
        KBContextRequest(collection="docs", hide_missing=True)
    )

    assert context.collection == "docs"


def test_open_collection_sync_returns_lancedb_handle() -> None:
    """Given sync legacy code, open_collection_sync returns a LanceDB handle."""
    from xagent.core.tools.core.RAG_tools.kb import (
        KBContextRequest,
        LanceDBCollectionHandle,
        get_kb_coordinator,
    )

    handle = get_kb_coordinator().open_collection_sync(
        KBContextRequest(collection="docs", hide_missing=True)
    )

    assert isinstance(handle, LanceDBCollectionHandle)


def test_reset_rag_storage_for_tests_resets_kb_coordinator_state() -> None:
    """Given storage reset, KB coordinator state is reset as part of test isolation."""
    from xagent.core.tools.core.RAG_tools.kb import get_kb_coordinator

    first = get_kb_coordinator()

    reset_rag_storage_for_tests()
    second = get_kb_coordinator()

    assert second is not first


def test_existing_rag_storage_and_management_imports_still_work() -> None:
    """Given legacy callers, old storage and management imports still work."""
    import xagent.core.tools.core.RAG_tools.management.collection_manager as manager
    import xagent.core.tools.core.RAG_tools.management.collections as collections
    import xagent.core.tools.core.RAG_tools.storage as storage

    assert hasattr(storage, "get_metadata_store")
    assert hasattr(storage, "get_vector_index_store")
    assert hasattr(collections, "list_collections")
    assert hasattr(collections, "delete_collection")
    assert hasattr(manager, "get_collection_sync")
