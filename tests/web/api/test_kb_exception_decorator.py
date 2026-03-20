"""Unit tests for KB API exception decorator.

These tests cover the mapping behavior of `handle_kb_exceptions` in `xagent.web.api.kb`.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from xagent.web.api.kb import handle_kb_exceptions


@pytest.mark.asyncio
async def test_handle_kb_exceptions_passthrough_http_exception() -> None:
    """HTTPException should be re-raised without being wrapped."""

    @handle_kb_exceptions
    async def _fn() -> None:
        raise HTTPException(status_code=418, detail="teapot")

    with pytest.raises(HTTPException) as exc_info:
        await _fn()

    assert exc_info.value.status_code == 418
    assert exc_info.value.detail == "teapot"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exc", "expected_status", "expected_prefix"),
    [
        (ValueError("bad"), 400, "数据格式错误:"),
        (KeyError("missing"), 400, "数据格式错误:"),
        (TypeError("wrong type"), 400, "数据格式错误:"),
    ],
)
async def test_handle_kb_exceptions_maps_data_errors_to_400(
    exc: Exception, expected_status: int, expected_prefix: str
) -> None:
    """ValueError/KeyError/TypeError should map to 400 with a data error message."""

    @handle_kb_exceptions
    async def _fn() -> None:
        raise exc

    with pytest.raises(HTTPException) as exc_info:
        await _fn()

    assert exc_info.value.status_code == expected_status
    assert isinstance(exc_info.value.detail, str)
    assert exc_info.value.detail.startswith(expected_prefix)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exc", "expected_status", "expected_prefix"),
    [
        (PermissionError("nope"), 403, "File system error:"),
        (OSError("io"), 403, "File system error:"),
    ],
)
async def test_handle_kb_exceptions_maps_fs_errors_to_403(
    exc: Exception, expected_status: int, expected_prefix: str
) -> None:
    """PermissionError/OSError should map to 403 with a file system error message."""

    @handle_kb_exceptions
    async def _fn() -> None:
        raise exc

    with pytest.raises(HTTPException) as exc_info:
        await _fn()

    assert exc_info.value.status_code == expected_status
    assert isinstance(exc_info.value.detail, str)
    assert exc_info.value.detail.startswith(expected_prefix)


@pytest.mark.asyncio
async def test_handle_kb_exceptions_maps_unknown_errors_to_500() -> None:
    """Other exceptions should map to 500 with an internal error message."""

    class _Boom(RuntimeError):
        pass

    @handle_kb_exceptions
    async def _fn() -> None:
        raise _Boom("boom")

    with pytest.raises(HTTPException) as exc_info:
        await _fn()

    assert exc_info.value.status_code == 500
    assert isinstance(exc_info.value.detail, str)
    assert exc_info.value.detail.startswith("服务器内部错误:")
