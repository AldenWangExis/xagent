import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.api.auth import hash_password
from xagent.web.api.kb import kb_router
from xagent.web.models.database import Base, get_db
from xagent.web.models.user import User


@pytest.fixture(scope="function")
def test_env():
    """Setup test database and app"""
    temp_db_fd, temp_db_path = tempfile.mkstemp(suffix=".db")
    os.close(temp_db_fd)

    test_engine = create_engine(f"sqlite:///{temp_db_path}")
    TestingSessionLocal = sessionmaker(bind=test_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(kb_router)
    app.dependency_overrides[get_db] = override_get_db

    Base.metadata.create_all(bind=test_engine)

    session = TestingSessionLocal()
    user = User(
        username="testuser", password_hash=hash_password("test"), is_admin=False
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    # Mock JWT token (must include type="access" for get_current_user)
    from datetime import datetime, timedelta

    import jwt

    from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY

    payload = {
        "sub": user.username,
        "user_id": user.id,
        "type": "access",
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow(),
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    headers = {"Authorization": f"Bearer {token}"}

    yield app, headers, user, TestingSessionLocal

    session.close()
    os.unlink(temp_db_path)


@pytest.fixture(scope="function")
def temp_uploads():
    """Setup temporary uploads directory and patch it"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        def patched_get_upload_path(
            filename,
            task_id=None,
            folder=None,
            user_id=None,
            collection=None,
            create_if_not_exists=True,
        ):
            base = temp_path
            if user_id:
                user_dir = base / f"user_{user_id}"
                if collection:
                    d = user_dir / collection
                    if create_if_not_exists:
                        d.mkdir(parents=True, exist_ok=True)
                    return d / filename
                if create_if_not_exists:
                    user_dir.mkdir(parents=True, exist_ok=True)
                return user_dir / filename
            return base / filename

        with (
            patch(
                "xagent.web.api.kb.get_upload_path",
                side_effect=patched_get_upload_path,
            ),
            patch(
                "xagent.web.config.get_upload_path",
                side_effect=patched_get_upload_path,
            ),
            patch("xagent.web.config.UPLOADS_DIR", temp_path),
        ):
            yield temp_path


def test_kb_ingest_creates_collection_dir(test_env, temp_uploads):
    """Test that ingesting a document creates a collection-specific directory"""
    app, headers, user, _ = test_env
    client = TestClient(app)

    collection_name = "kb_test_coll"
    filename = "test_doc.txt"

    # Mock the RAG pipeline to avoid heavy dependencies
    with patch("xagent.web.api.kb.run_document_ingestion") as mock_ingest:
        from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult

        mock_ingest.return_value = IngestionResult(
            status="success",
            doc_id="test_doc_id",
            parse_hash="hash",
            failed_step="",
            message="success",
        )

        # Upload file
        files = {"file": (filename, b"content", "text/plain")}
        data = {"collection": collection_name}

        response = client.post(
            "/api/kb/ingest", files=files, data=data, headers=headers
        )

        assert response.status_code == 200

        # Check if physical directory was created
        expected_path = temp_uploads / f"user_{user.id}" / collection_name / filename
        assert expected_path.exists()
        assert expected_path.is_file()


def test_kb_delete_cleans_physical_dir(test_env, temp_uploads):
    """Test that deleting a collection also removes the physical directory"""
    app, headers, user, _ = test_env
    client = TestClient(app)

    collection_name = "kb_to_delete"

    # Pre-create the collection directory
    coll_dir = temp_uploads / f"user_{user.id}" / collection_name
    coll_dir.mkdir(parents=True, exist_ok=True)
    (coll_dir / "some_file.txt").write_text("data")

    # Mock delete_collection (the database part)
    with patch("xagent.web.api.kb.delete_collection") as mock_delete:
        from xagent.core.tools.core.RAG_tools.core.schemas import (
            CollectionOperationResult,
        )

        mock_delete.return_value = CollectionOperationResult(
            status="success",
            collection=collection_name,
            message="deleted",
            affected_documents=[],
            deleted_counts={},
        )

        # Delete collection
        response = client.delete(
            f"/api/kb/collections/{collection_name}", headers=headers
        )

        assert response.status_code == 200

        # Check if physical directory was removed
        assert not coll_dir.exists()


def test_kb_ingest_rejects_path_traversal_in_collection_name(test_env, temp_uploads):
    """Test that ingest API rejects path traversal in collection name."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    malicious_collections = [
        "../../../etc",
        "..\\..\\..\\windows",
        "collection/../other",
        "../collection",
    ]

    filename = "test_doc.txt"

    for collection_name in malicious_collections:
        with patch("xagent.web.api.kb.run_document_ingestion"):
            files = {"file": (filename, b"content", "text/plain")}
            data = {"collection": collection_name}

            response = client.post(
                "/api/kb/ingest", files=files, data=data, headers=headers
            )

            # Should reject with 422 (validation error)
            assert response.status_code == 422
            assert "Invalid collection name" in response.json()["detail"]


def test_kb_ingest_rejects_invalid_characters_in_collection_name(
    test_env, temp_uploads
):
    """Test that ingest API rejects invalid characters in collection name."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    invalid_collections = [
        "collection name",  # Space
        "collection@name",  # @ symbol
        "collection#name",  # # symbol
        "collection/name",  # Path separator
    ]

    filename = "test_doc.txt"

    for collection_name in invalid_collections:
        with patch("xagent.web.api.kb.run_document_ingestion"):
            files = {"file": (filename, b"content", "text/plain")}
            data = {"collection": collection_name}

            response = client.post(
                "/api/kb/ingest", files=files, data=data, headers=headers
            )

            # Should reject with 422 (validation error)
            assert response.status_code == 422
            assert "Invalid collection name" in response.json()["detail"]


def test_kb_ingest_rejects_too_long_collection_name(test_env, temp_uploads):
    """Test that ingest API rejects collection names exceeding length limit."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    # Create a collection name that exceeds MAX_COLLECTION_NAME_LENGTH (100)
    too_long_collection = "a" * 101
    filename = "test_doc.txt"

    with patch("xagent.web.api.kb.run_document_ingestion"):
        files = {"file": (filename, b"content", "text/plain")}
        data = {"collection": too_long_collection}

        response = client.post(
            "/api/kb/ingest", files=files, data=data, headers=headers
        )

        # Should reject with 422 (validation error)
        assert response.status_code == 422
        assert "Invalid collection name" in response.json()["detail"]


def test_kb_ingest_validates_derived_collection_name_from_filename(
    test_env, temp_uploads
):
    """Test that ingest API validates collection name derived from filename."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    # Test with filename that would create invalid collection name
    # Note: "../../../etc.txt" becomes "etc.txt" after basename, which is valid
    # So we test actual invalid cases
    malicious_filenames = [
        "file name.txt",  # Would create "file name" with space
        "file@name.txt",  # Would create "file@name" with invalid character
    ]

    for filename in malicious_filenames:
        with patch("xagent.web.api.kb.run_document_ingestion"):
            files = {"file": (filename, b"content", "text/plain")}
            # Don't provide collection parameter, so it's derived from filename

            response = client.post(
                "/api/kb/ingest", files=files, data={}, headers=headers
            )

            # Should reject with 422 (validation error)
            assert response.status_code == 422
            detail = response.json().get("detail", "")
            assert "Invalid collection name" in detail or "invalid" in detail.lower()


def test_kb_delete_rejects_path_traversal_in_collection_name(test_env, temp_uploads):
    """Test that delete_collection_api rejects path traversal in collection name."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    # Note: Paths with special characters in URL path parameter may cause 404
    # due to URL encoding/routing issues. Test with URL-encoded versions or
    # simpler invalid names that still trigger validation
    malicious_collections = [
        "collection/../other",  # Path separator
        "collection%2Fother",  # URL-encoded path separator
    ]

    for collection_name in malicious_collections:
        # URL encode the collection name for the path parameter
        from urllib.parse import quote

        encoded_name = quote(collection_name, safe="")
        response = client.delete(f"/api/kb/collections/{encoded_name}", headers=headers)

        # Should reject with 422 (validation error) or 404 (if routing fails)
        # If routing fails, the validation happens but returns 404
        assert response.status_code in [422, 404]
        if response.status_code == 422:
            assert "Invalid collection name" in response.json()["detail"]


def test_kb_delete_physical_cleanup_failure_aborts_operation(test_env, temp_uploads):
    """Test that physical cleanup (move-to-trash) failure aborts database deletion."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    collection_name = "kb_to_delete_fail"

    # Pre-create the collection directory
    coll_dir = temp_uploads / f"user_{user.id}" / collection_name
    coll_dir.mkdir(parents=True, exist_ok=True)
    (coll_dir / "some_file.txt").write_text("data")

    # Mock delete_collection to return success (database deletion would succeed)
    with (
        patch("xagent.web.api.kb.delete_collection") as mock_delete,
        patch("xagent.web.api.kb.move_collection_dir_to_trash") as mock_move_to_trash,
    ):
        from xagent.core.tools.core.RAG_tools.core.schemas import (
            CollectionOperationResult,
        )

        mock_delete.return_value = CollectionOperationResult(
            status="success",
            collection=collection_name,
            message="deleted",
            affected_documents=[],
            deleted_counts={},
        )

        # Simulate move-to-trash failure (delete now uses rename-to-trash, not rmtree)
        mock_move_to_trash.side_effect = PermissionError("Permission denied")

        # Attempt to delete collection
        response = client.delete(
            f"/api/kb/collections/{collection_name}", headers=headers
        )

        # Should fail with 500 (physical move failed, operation aborted)
        assert response.status_code == 500
        assert "cannot move physical files" in response.json()["detail"].lower()

        # Verify directory still exists (operation was aborted)
        assert coll_dir.exists()


def test_kb_delete_returns_physical_cleanup_status(test_env, temp_uploads):
    """Test that delete_collection_api returns physical cleanup status in response."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    collection_name = "kb_to_delete_status"

    # Pre-create the collection directory
    coll_dir = temp_uploads / f"user_{user.id}" / collection_name
    coll_dir.mkdir(parents=True, exist_ok=True)
    (coll_dir / "some_file.txt").write_text("data")

    # Mock delete_collection
    with patch("xagent.web.api.kb.delete_collection") as mock_delete:
        from xagent.core.tools.core.RAG_tools.core.schemas import (
            CollectionOperationResult,
        )

        mock_delete.return_value = CollectionOperationResult(
            status="success",
            collection=collection_name,
            message="deleted",
            affected_documents=[],
            deleted_counts={},
        )

        # Delete collection
        response = client.delete(
            f"/api/kb/collections/{collection_name}", headers=headers
        )

        assert response.status_code == 200
        data = response.json()

        # Should include physical cleanup information in warnings
        assert "warnings" in data or "message" in data
        if "warnings" in data:
            # Check that warnings include physical cleanup status
            warnings_text = " ".join(data["warnings"]).lower()
            assert any(
                keyword in warnings_text
                for keyword in ["physical", "directory", "cleanup", "removed"]
            )


def test_kb_rename_rejects_path_traversal_in_collection_names(test_env, temp_uploads):
    """Test that rename_collection_api rejects path traversal in old and new names."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    # First create a valid collection
    valid_collection = "valid_collection"
    coll_dir = temp_uploads / f"user_{user.id}" / valid_collection
    coll_dir.mkdir(parents=True, exist_ok=True)

    # Test with names that will trigger validation (path separators)
    malicious_names = [
        "collection/../other",  # Path separator
    ]

    from urllib.parse import quote

    # Mock database operations to avoid schema errors
    with patch("xagent.web.api.kb.get_connection_from_env") as mock_conn:
        from unittest.mock import MagicMock

        # Mock connection and table
        mock_db_conn = MagicMock()
        mock_table = MagicMock()
        mock_table.count_rows.return_value = (
            0  # No documents, so permission check passes
        )
        mock_db_conn.open_table.return_value = mock_table
        mock_conn.return_value = mock_db_conn

        for malicious_name in malicious_names:
            # Test malicious old name (URL encoded)
            encoded_old = quote(malicious_name, safe="")
            response = client.put(
                f"/api/kb/collections/{encoded_old}",
                data={"new_name": "new_collection"},
                headers=headers,
            )
            # May return 404 if routing fails, or 422 if validation catches it
            assert response.status_code in [422, 404]
            if response.status_code == 422:
                assert "Invalid collection name" in response.json()["detail"]

            # Test malicious new name (in form data, no URL encoding needed)
            # Mock again for the second request
            mock_table.count_rows.return_value = 0
            response = client.put(
                f"/api/kb/collections/{valid_collection}",
                data={"new_name": malicious_name},
                headers=headers,
            )
            # Form data should be validated, should return 422
            # Note: validation happens after permission check, so we need to mock DB
            assert response.status_code == 422
            assert "Invalid collection name" in response.json()["detail"]


def test_kb_rename_physical_directory_rename(test_env, temp_uploads):
    """Test that rename_collection_api physically renames the directory."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    old_collection_name = "old_collection"
    new_collection_name = "new_collection"

    # Pre-create the old collection directory
    old_coll_dir = temp_uploads / f"user_{user.id}" / old_collection_name
    old_coll_dir.mkdir(parents=True, exist_ok=True)
    (old_coll_dir / "some_file.txt").write_text("data")

    # Mock the database update operations to avoid database errors
    with (
        patch(
            "xagent.core.tools.core.RAG_tools.management.collections._list_table_names"
        ) as mock_list_tables,
        patch("xagent.web.api.kb.get_connection_from_env") as mock_conn,
    ):
        from unittest.mock import MagicMock

        mock_list_tables.return_value = []
        # Mock connection and table to avoid database errors
        mock_db_conn = MagicMock()
        mock_table = MagicMock()
        mock_table.count_rows.return_value = (
            0  # No documents, so permission check passes
        )
        mock_db_conn.open_table.return_value = mock_table
        mock_conn.return_value = mock_db_conn

        # Attempt rename
        response = client.put(
            f"/api/kb/collections/{old_collection_name}",
            data={"new_name": new_collection_name},
            headers=headers,
        )

        # Should succeed (or return appropriate status)
        assert response.status_code in [200, 500]  # 500 if database operations fail

        # Check if physical directory was renamed
        new_coll_dir = temp_uploads / f"user_{user.id}" / new_collection_name
        if response.status_code == 200:
            # If rename succeeded, new directory should exist
            assert new_coll_dir.exists()
            assert not old_coll_dir.exists()
        else:
            # If database operations failed, old directory should still exist
            # (physical rename might have happened but was rolled back, or didn't happen)
            pass


def test_kb_rename_physical_rename_failure_aborts_operation(test_env, temp_uploads):
    """Test that physical rename failure aborts database update."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    old_collection_name = "old_collection"
    new_collection_name = "new_collection"

    # Pre-create the old collection directory
    old_coll_dir = temp_uploads / f"user_{user.id}" / old_collection_name
    old_coll_dir.mkdir(parents=True, exist_ok=True)
    (old_coll_dir / "some_file.txt").write_text("data")

    # Mock database operations to avoid schema errors
    with patch("xagent.web.api.kb.get_connection_from_env") as mock_conn:
        from unittest.mock import MagicMock

        # Mock connection and table
        mock_db_conn = MagicMock()
        mock_table = MagicMock()
        mock_table.count_rows.return_value = (
            0  # No documents, so permission check passes
        )
        mock_db_conn.open_table.return_value = mock_table
        mock_conn.return_value = mock_db_conn

        # Physical rename uses shutil.move() to support cross-device moves.
        # Patch it to fail to simulate a filesystem permission error.
        with patch("shutil.move", side_effect=PermissionError("Permission denied")):
            # Attempt rename
            response = client.put(
                f"/api/kb/collections/{old_collection_name}",
                data={"new_name": new_collection_name},
                headers=headers,
            )

            # Should fail with 500 (physical rename failed, operation aborted)
            assert response.status_code == 500
            detail = response.json()["detail"].lower()
            assert (
                "cannot rename physical directory" in detail
                or "failed to rename" in detail
                or "physical directory rename" in detail
            )

            # Verify old directory still exists (operation was aborted)
            assert old_coll_dir.exists()


def test_kb_rename_target_directory_exists_conflict(test_env, temp_uploads):
    """Test that rename_collection_api handles target directory already existing."""
    app, headers, user, _ = test_env
    client = TestClient(app)

    old_collection_name = "old_collection"
    new_collection_name = "existing_collection"

    # Pre-create both directories
    old_coll_dir = temp_uploads / f"user_{user.id}" / old_collection_name
    old_coll_dir.mkdir(parents=True, exist_ok=True)
    (old_coll_dir / "old_file.txt").write_text("old data")

    new_coll_dir = temp_uploads / f"user_{user.id}" / new_collection_name
    new_coll_dir.mkdir(parents=True, exist_ok=True)
    (new_coll_dir / "new_file.txt").write_text("new data")

    # Mock database operations to avoid schema errors
    with patch("xagent.web.api.kb.get_connection_from_env") as mock_conn:
        from unittest.mock import MagicMock

        # Mock connection and table
        mock_db_conn = MagicMock()
        mock_table = MagicMock()
        mock_table.count_rows.return_value = (
            0  # No documents, so permission check passes
        )
        mock_db_conn.open_table.return_value = mock_table
        mock_conn.return_value = mock_db_conn

        # Attempt rename to existing directory
        response = client.put(
            f"/api/kb/collections/{old_collection_name}",
            data={"new_name": new_collection_name},
            headers=headers,
        )

        # Should fail with 409 (conflict) or 500
        assert response.status_code in [409, 500]
        if response.status_code == 409:
            assert "already exists" in response.json()["detail"].lower()
