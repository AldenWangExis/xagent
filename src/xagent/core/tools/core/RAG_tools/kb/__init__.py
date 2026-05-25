"""KB semantic coordinator public surface."""

from .collection_handle import KBHandleProvider, LanceDBCollectionHandle
from .coordinator import (
    KBCoordinator,
    get_kb_coordinator,
    reset_kb_coordinator_for_tests,
)
from .models import (
    KBAccessMode,
    KBBackendCapabilities,
    KBCollectionContext,
    KBContextRequest,
    KBStorageBackend,
    KBUserScope,
)

__all__ = [
    "KBAccessMode",
    "KBBackendCapabilities",
    "KBCollectionContext",
    "KBContextRequest",
    "KBHandleProvider",
    "KBCoordinator",
    "KBStorageBackend",
    "KBUserScope",
    "LanceDBCollectionHandle",
    "get_kb_coordinator",
    "reset_kb_coordinator_for_tests",
]
