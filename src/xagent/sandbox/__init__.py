"""
Sandbox Support.
"""

import os

from .base import (
    CodeType,
    ExecResult,
    Sandbox,
    SandboxConfig,
    SandboxInfo,
    SandboxService,
    SandboxSnapshot,
    SandboxTemplate,
    TemplateType,
)

# Use the `latest` image as a fallback
# We should pin the version at release by env "SANDBOX_IMAGE" (`latest` may lead to caching problems)
DEFAULT_SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "xprobe/xagent-sandbox:latest")

__all__ = [
    "DEFAULT_SANDBOX_IMAGE",
    "TemplateType",
    "CodeType",
    "SandboxTemplate",
    "SandboxConfig",
    "SandboxInfo",
    "SandboxSnapshot",
    "ExecResult",
    "Sandbox",
    "SandboxService",
]

try:
    from .boxlite_sandbox import (
        BoxliteSandbox,
        BoxliteSandboxService,
        BoxliteStore,
        MemBoxliteStore,
    )

    __all__ += [
        "BoxliteSandbox",
        "BoxliteStore",
        "MemBoxliteStore",
        "BoxliteSandboxService",
    ]
except ImportError:
    pass
