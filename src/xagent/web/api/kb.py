"""Knowledge base API route handlers"""

import asyncio
import concurrent.futures
import functools
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import JSONResponse
from sqlalchemy import or_
from sqlalchemy.orm import Session

from ...core.tools.core.RAG_tools.core.schemas import (
    ChunkStrategy,
    CollectionOperationResult,
    FusionConfig,
    IngestionConfig,
    IngestionResult,
    ListCollectionsResult,
    ParseMethod,
    ParseResultResponse,
    SearchConfig,
    SearchPipelineResult,
    SearchType,
    WebCrawlConfig,
    WebIngestionResult,
)
from ...core.tools.core.RAG_tools.management.collections import (
    delete_collection,
    list_collections,
)
from ...core.tools.core.RAG_tools.parse.parse_display import (
    paginate_parse_results,
    reconstruct_parse_result_from_db,
)
from ...core.tools.core.RAG_tools.pipelines.document_ingestion import (
    run_document_ingestion,
)
from ...core.tools.core.RAG_tools.pipelines.document_search import run_document_search
from ...core.tools.core.RAG_tools.pipelines.web_ingestion import run_web_ingestion
from ...core.tools.core.RAG_tools.progress import get_progress_manager
from ...providers.vector_store.lancedb import get_connection_from_env
from ..auth_dependencies import get_current_user
from ..config import (
    MAX_FILE_SIZE,
    UPLOADS_DIR,
    get_upload_path,
    is_allowed_file,
    sanitize_path_component,
)
from ..kb_physical_sync import collection_physical_lock, move_collection_dir_to_trash
from ..models.database import get_db
from ..models.uploaded_file import UploadedFile
from ..models.user import User

T = TypeVar("T", bound=Callable[..., Any])
logger = logging.getLogger(__name__)


def handle_kb_exceptions(func: T) -> T:
    """Decorator to handle common exceptions in KB API routes."""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except (ValueError, KeyError, TypeError) as e:
            logger.error("Data format error in %s: %s", func.__name__, e)
            raise HTTPException(status_code=400, detail=f"数据格式错误: {str(e)}")
        except (PermissionError, OSError) as e:
            logger.error("File system error in %s: %s", func.__name__, e)
            raise HTTPException(status_code=403, detail=f"File system error: {str(e)}")
        except Exception as e:
            logger.exception("Unexpected error in %s: %s", func.__name__, e)
            raise HTTPException(
                status_code=500,
                detail=f"服务器内部错误: {str(e)}",
            )

    return cast(T, wrapper)


# Create router
kb_router = APIRouter(prefix="/api/kb", tags=["kb"])


def _parse_separators(separators: Optional[str]) -> Optional[List[str]]:
    """Parse optional custom separators (JSON array of strings) from form input.

    Returns None if input is missing/empty or invalid; returns a list of
    non-empty strings when valid (possibly empty list for input '[]').
    """
    if not separators or not separators.strip():
        return None
    try:
        raw = json.loads(separators)
        if isinstance(raw, list) and all(isinstance(x, str) for x in raw):
            return [s for s in raw if s]
        logger.warning("separators must be a list of strings; ignoring")
        return None
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("invalid separators JSON, using default: %s", e)
        return None


@kb_router.post(
    "/ingest",
    response_model=IngestionResult,
)
@handle_kb_exceptions
async def ingest(
    collection: str = Form(None),
    file: UploadFile = File(...),
    *,
    # Ingestion configuration parameters
    parse_method: Optional[ParseMethod] = Form(
        None,
        description="Parser used during ingestion. Options: default, pypdf, pdfplumber, unstructured, pymupdf, deepdoc",
    ),
    chunk_strategy: Optional[ChunkStrategy] = Form(
        None,
        description="Chunking strategy. Options: recursive (default), fixed_size, markdown",
    ),
    chunk_size: Optional[int] = Form(
        None,
        gt=0,
        description="Chunk size in characters (default: 1000)",
    ),
    chunk_overlap: Optional[int] = Form(
        None,
        ge=0,
        description="Chunk overlap in characters (default: 200)",
    ),
    separators: Optional[str] = Form(
        None,
        description=(
            "Custom chunk separators as JSON array of strings, e.g. "
            '["\\n\\n", "\\n", "。"]. Only used when chunk_strategy is recursive. '
            "Omit or empty to use default separators."
        ),
    ),
    embedding_model_id: str = Form(
        "text-embedding-v4",
        description="Embedding model ID (default: text-embedding-v4)",
    ),
    embedding_batch_size: Optional[int] = Form(
        None,
        gt=0,
        description="Batch size for embedding (default: 10)",
    ),
    max_retries: Optional[int] = Form(
        None,
        ge=0,
        description="Maximum retries for embedding failures (default: 3)",
    ),
    retry_delay: Optional[float] = Form(
        None,
        ge=0.0,
        description="Delay between retries in seconds (default: 1.0)",
    ),
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> IngestionResult | JSONResponse:
    """Upload and ingest a document into the knowledge base.

    Args:
        collection: Target collection name. If not provided, uses file name.
        file: The document file to upload and process.
        parse_method: Parser used during ingestion.
        chunk_strategy: Strategy for chunking the document.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.
        separators: Optional JSON array of custom chunk separators (recursive only).
        embedding_model_id: Embedding model ID from model hub.
        embedding_batch_size: Batch size for embedding operations.
        max_retries: Maximum retry attempts for failures.
        retry_delay: Delay between retry attempts in seconds.
    """
    if not file.filename or not file.filename.strip():
        raise HTTPException(status_code=422, detail="No filename provided")

    # SECURITY: Extract only basename to prevent path traversal attacks
    # e.g., "../../../etc/passwd.pdf" becomes "passwd.pdf"
    safe_filename = Path(file.filename).name

    if not is_allowed_file(safe_filename, "general"):
        raise HTTPException(
            status_code=422,
            detail=f"File type {Path(safe_filename).suffix.lower()} not supported",
        )

    if not collection or not collection.strip():
        collection = Path(safe_filename).stem
        logger.info("Using file name as collection: %s", collection)

    try:
        # SECURITY: Validate collection name at API boundary
        safe_collection = sanitize_path_component(collection, "collection")

        try:
            file_path = get_upload_path(
                safe_filename,
                user_id=int(_user.id),
                collection=safe_collection,
                collection_is_sanitized=True,
            )
        except TypeError as e:
            # Backward compatibility for tests/mocks that patch get_upload_path
            # with an older signature that doesn't accept this keyword.
            if "collection_is_sanitized" not in str(e):
                raise
            file_path = get_upload_path(
                safe_filename,
                user_id=int(_user.id),
                collection=safe_collection,
            )
    except ValueError as e:
        logger.warning("Invalid collection name rejected: %s - %s", collection, e)
        raise HTTPException(
            status_code=422, detail=f"Invalid collection name: {str(e)}"
        ) from e

    try:
        total_size = 0
        chunk_size = 1024 * 1024  # 1MB
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "File size exceeds maximum limit of "
                            f"{MAX_FILE_SIZE // (1024 * 1024)}MB"
                        ),
                    )
                buffer.write(chunk)
        logger.info(
            "File uploaded: %s -> %s (user: %s, collection: %s)",
            safe_filename,
            file_path,
            _user.id,
            collection,
        )
    except (PermissionError, OSError) as e:
        logger.error("File system error saving file %s: %s", safe_filename, e)
        raise HTTPException(
            status_code=403, detail=f"File system error: {str(e)}"
        ) from e
    except HTTPException:
        # Ensure partial file is removed on early abort (e.g., file too large)
        try:
            if file_path.exists():
                file_path.unlink()
        except OSError:
            pass
        raise

    # Register file in unified file management (file_id) for /api/files/list and file_id download/preview/delete
    storage_path_str = str(file_path)
    existing = (
        db.query(UploadedFile)
        .filter(UploadedFile.storage_path == storage_path_str)
        .first()
    )
    if existing:
        existing.file_size = int(total_size)  # type: ignore[assignment]
        existing.mime_type = getattr(file, "content_type", None) or existing.mime_type
        db.flush()
        file_record = existing
    else:
        import mimetypes

        mime_type = (
            getattr(file, "content_type", None)
            or mimetypes.guess_type(safe_filename)[0]
            or "application/octet-stream"
        )
        file_record = UploadedFile(
            user_id=int(_user.id),
            filename=safe_filename,
            storage_path=storage_path_str,
            mime_type=mime_type,
            file_size=int(total_size),
        )
        db.add(file_record)
        db.flush()
    db.commit()
    db.refresh(file_record)

    final_chunk_size = chunk_size if chunk_size is not None and chunk_size > 0 else 1000
    final_chunk_overlap = (
        chunk_overlap if chunk_overlap is not None and chunk_overlap >= 0 else 200
    )
    if final_chunk_overlap >= final_chunk_size:
        final_chunk_overlap = min(int(final_chunk_size * 0.2), final_chunk_size - 1)
        logger.warning(
            "Auto-adjusting chunk_overlap to %s to ensure it's less than chunk_size (%s)",
            final_chunk_overlap,
            final_chunk_size,
        )

    parsed_separators = _parse_separators(separators)
    final_strategy = (
        chunk_strategy if chunk_strategy is not None else ChunkStrategy.RECURSIVE
    )
    if separators and separators.strip() and final_strategy != ChunkStrategy.RECURSIVE:
        logger.warning(
            "separators are only used when chunk_strategy is recursive; "
            "current strategy is %s, ignoring separators",
            final_strategy.value,
        )

    config = IngestionConfig(
        parse_method=parse_method if parse_method is not None else ParseMethod.DEFAULT,
        chunk_strategy=final_strategy,
        chunk_size=final_chunk_size,
        chunk_overlap=final_chunk_overlap,
        separators=parsed_separators,
        embedding_model_id=embedding_model_id,
        embedding_batch_size=embedding_batch_size
        if embedding_batch_size is not None and embedding_batch_size > 0
        else 10,
        max_retries=max_retries if max_retries is not None and max_retries >= 0 else 3,
        retry_delay=retry_delay
        if retry_delay is not None and retry_delay >= 0
        else 1.0,
    )

    progress_manager = get_progress_manager()

    def _run_ingestion() -> IngestionResult:
        return run_document_ingestion(
            collection=collection,
            source_path=str(file_path),
            ingestion_config=config,
            progress_manager=progress_manager,
            user_id=int(_user.id),
            is_admin=bool(_user.is_admin),
        )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_run_ingestion)
        result: IngestionResult = future.result()

    if result.status == "error":
        return JSONResponse(status_code=500, content=result.model_dump())

    return JSONResponse(
        status_code=200,
        content={**result.model_dump(), "file_id": file_record.file_id},
    )


@kb_router.get(
    "/collections",
    response_model=ListCollectionsResult,
)
@handle_kb_exceptions
async def list_collections_api(
    _user: User = Depends(get_current_user),
) -> ListCollectionsResult:
    """List all collections with their statistics."""
    kb_collections_timeout_seconds = 15

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(list_collections, int(_user.id), bool(_user.is_admin)),
            timeout=kb_collections_timeout_seconds,
        )
        return result
    except asyncio.TimeoutError:
        logger.error(
            "Listing KB collections timed out after %s seconds",
            kb_collections_timeout_seconds,
        )
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is temporarily unavailable. Please retry.",
        )


@kb_router.post(
    "/search",
    response_model=SearchPipelineResult,
)
@handle_kb_exceptions
async def search(
    collection: str = Form(..., description="Target collection to search within"),
    query_text: str = Form(..., description="Query text to search for"),
    embedding_model_id: str = Form(
        "text-embedding-v4",
        description="Embedding model ID (default: text-embedding-v4)",
    ),
    *,
    # Search configuration parameters
    search_type: Optional[SearchType] = Form(
        None,
        description="Search strategy: dense, sparse, or hybrid (default: hybrid)",
    ),
    top_k: Optional[int] = Form(
        None,
        ge=1,
        le=100,
        description="Maximum number of results to return (default: 5)",
    ),
    filters: Optional[Dict[str, Any]] = Form(
        None,
        description="Optional filters to apply during search (LanceDB format)",
    ),
    fusion_config: Optional[Dict[str, Any]] = Form(
        None,
        description="Optional fusion configuration for hybrid search",
    ),
    rerank_model_id: Optional[str] = Form(
        None,
        description="Optional rerank model ID for result reordering",
    ),
    rerank_top_k: Optional[int] = Form(
        None,
        description="Optional override for rerank result count",
    ),
    readonly: Optional[bool] = Form(
        None,
        description="Avoid index modifications (default: False)",
    ),
    nprobes: Optional[int] = Form(
        None,
        description="Number of partitions to probe for ANN search",
    ),
    refine_factor: Optional[int] = Form(
        None,
        description="Refine factor for ANN search re-ranking",
    ),
    fallback_to_sparse: Optional[bool] = Form(
        None,
        description="Allow hybrid search to fallback to sparse (default: True)",
    ),
    _user: User = Depends(get_current_user),
) -> SearchPipelineResult:
    """Search documents in the knowledge base.

    Args:
        collection: Target collection to search within.
        query_text: Query text to search for.
        embedding_model_id: Embedding model ID (required for dense/hybrid search).
        search_type: Search strategy (dense, sparse, or hybrid).
        top_k: Maximum number of results to return.
        filters: Optional filters for search.
        fusion_config: Optional fusion configuration for hybrid search.
        rerank_model_id: Optional rerank model for result reordering.
        rerank_top_k: Override for rerank result count.
        readonly: Whether to avoid index modifications.
        nprobes: Number of partitions to probe for ANN search.
        refine_factor: Refine factor for ANN search re-ranking.
        fallback_to_sparse: Allow hybrid search to fallback to sparse.
    """
    # CRITICAL: Handle empty strings from Swagger UI - convert to None BEFORE any processing
    if filters == "":
        filters = None
    if fusion_config == "":
        fusion_config = None

    if not collection or not query_text:
        raise HTTPException(status_code=422, detail="Missing required parameters")

    if not embedding_model_id:
        raise HTTPException(
            status_code=422,
            detail="embedding_model_id is required",
        )

    # Build configuration from individual parameters
    config = SearchConfig(
        search_type=search_type or SearchType.HYBRID,
        top_k=top_k or 5,
        filters=filters,
        fusion_config=FusionConfig.model_validate(fusion_config)
        if fusion_config
        else None,
        embedding_model_id=embedding_model_id,
        rerank_model_id=rerank_model_id,
        rerank_top_k=rerank_top_k,
        readonly=readonly or False,
        nprobes=nprobes,
        refine_factor=refine_factor,
        fallback_to_sparse=fallback_to_sparse
        if fallback_to_sparse is not None
        else True,
    )

    progress_manager = get_progress_manager()
    result = run_document_search(
        collection=collection,
        query_text=query_text,
        config=config,
        progress_manager=progress_manager,
        user_id=int(_user.id),
        is_admin=bool(_user.is_admin),
    )

    return result


@kb_router.post(
    "/ingest-web",
    response_model=WebIngestionResult,
)
@handle_kb_exceptions
async def ingest_web(
    collection: str = Form(..., description="Target collection name"),
    start_url: str = Form(..., description="Starting URL for crawling"),
    # WebCrawlConfig parameters
    max_pages: Optional[int] = Form(
        100,
        description="Maximum number of pages to crawl (default: 100)",
    ),
    max_depth: Optional[int] = Form(
        3,
        description="Maximum crawl depth (default: 3)",
    ),
    url_patterns: Optional[str] = Form(
        None,
        description="Comma-separated URL match patterns (regex)",
    ),
    exclude_patterns: Optional[str] = Form(
        None,
        description="Comma-separated exclusion patterns (regex)",
    ),
    same_domain_only: Optional[bool] = Form(
        True,
        description="Only crawl same domain (default: True)",
    ),
    content_selector: Optional[str] = Form(
        None,
        description="CSS selector for main content area",
    ),
    remove_selectors: Optional[str] = Form(
        None,
        description="Comma-separated CSS selectors to remove",
    ),
    concurrent_requests: Optional[int] = Form(
        3,
        ge=1,
        le=10,
        description="Concurrent requests (default: 3, max: 10)",
    ),
    request_delay: Optional[float] = Form(
        1.0,
        ge=0,
        description="Delay between requests in seconds (default: 1.0)",
    ),
    timeout: Optional[int] = Form(
        30,
        ge=1,
        description="Request timeout in seconds (default: 30)",
    ),
    respect_robots_txt: Optional[bool] = Form(
        True,
        description="Respect robots.txt (default: True)",
    ),
    # IngestionConfig parameters
    parse_method: Optional[ParseMethod] = Form(
        None,
        description="Parser used during ingestion",
    ),
    chunk_strategy: Optional[ChunkStrategy] = Form(
        None,
        description="Chunking strategy",
    ),
    chunk_size: Optional[int] = Form(
        None,
        gt=0,
        description="Chunk size in characters (default: 1000)",
    ),
    chunk_overlap: Optional[int] = Form(
        None,
        ge=0,
        description="Chunk overlap (default: 200)",
    ),
    separators: Optional[str] = Form(
        None,
        description=(
            "Custom chunk separators as JSON array of strings; "
            "only used when chunk_strategy is recursive."
        ),
    ),
    embedding_model_id: str = Form(
        "text-embedding-v4",
        description="Embedding model ID",
    ),
    embedding_batch_size: Optional[int] = Form(
        None,
        gt=0,
        description="Batch size for embedding (default: 10)",
    ),
    max_retries: Optional[int] = Form(
        None,
        ge=0,
        description="Maximum retries for embedding failures (default: 3)",
    ),
    retry_delay: Optional[float] = Form(
        None,
        ge=0.0,
        description="Delay between retries in seconds (default: 1.0)",
    ),
    _user: User = Depends(get_current_user),
) -> WebIngestionResult | JSONResponse:
    """Ingest website content into the knowledge base.

    Args:
        collection: Target collection name
        start_url: Starting URL for crawling
        max_pages: Maximum number of pages to crawl
        max_depth: Maximum crawl depth
        url_patterns: Comma-separated URL match patterns (regex)
        exclude_patterns: Comma-separated exclusion patterns (regex)
        same_domain_only: Only crawl same domain
        content_selector: CSS selector for main content area
        remove_selectors: Comma-separated CSS selectors to remove
        concurrent_requests: Number of concurrent requests
        request_delay: Delay between requests in seconds
        timeout: Request timeout in seconds
        respect_robots_txt: Respect robots.txt rules
        parse_method: Parser for document ingestion
        chunk_strategy: Chunking strategy
        chunk_size: Chunk size in characters
        chunk_overlap: Chunk overlap in characters
        embedding_model_id: Embedding model ID
        embedding_batch_size: Batch size for embedding
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries
    """
    try:
        try:
            safe_collection = sanitize_path_component(collection, "collection")
        except ValueError as e:
            logger.warning("Invalid collection name rejected: %s - %s", collection, e)
            raise HTTPException(
                status_code=422, detail=f"Invalid collection name: {str(e)}"
            ) from e

        url_patterns_list = (
            [p.strip() for p in url_patterns.split(",")] if url_patterns else None
        )
        exclude_patterns_list = (
            [p.strip() for p in exclude_patterns.split(",")]
            if exclude_patterns
            else None
        )
        remove_selectors_list = (
            [s.strip() for s in remove_selectors.split(",")]
            if remove_selectors
            else None
        )

        crawl_config = WebCrawlConfig(
            start_url=start_url,
            max_pages=max_pages or 100,
            max_depth=max_depth or 3,
            url_patterns=url_patterns_list,
            exclude_patterns=exclude_patterns_list,
            same_domain_only=(
                same_domain_only if same_domain_only is not None else True
            ),
            content_selector=content_selector,
            remove_selectors=remove_selectors_list,
            concurrent_requests=concurrent_requests or 3,
            request_delay=request_delay or 1.0,
            timeout=timeout or 30,
            respect_robots_txt=(
                respect_robots_txt if respect_robots_txt is not None else True
            ),
        )

        final_chunk_size = (
            chunk_size if chunk_size is not None and chunk_size > 0 else 1000
        )
        final_chunk_overlap = (
            chunk_overlap if chunk_overlap is not None and chunk_overlap >= 0 else 200
        )
        if final_chunk_overlap >= final_chunk_size:
            final_chunk_overlap = min(int(final_chunk_size * 0.2), final_chunk_size - 1)
            logger.warning(
                "Auto-adjusting chunk_overlap from %s to %s to ensure it's less than chunk_size (%s)",
                chunk_overlap,
                final_chunk_overlap,
                final_chunk_size,
            )

        web_parsed_separators = _parse_separators(separators)
        web_final_strategy = (
            chunk_strategy if chunk_strategy is not None else ChunkStrategy.RECURSIVE
        )
        if (
            separators
            and separators.strip()
            and web_final_strategy != ChunkStrategy.RECURSIVE
        ):
            logger.warning(
                "separators are only used when chunk_strategy is recursive; "
                "current strategy is %s, ignoring separators",
                web_final_strategy.value,
            )

        ingestion_config = IngestionConfig(
            parse_method=(
                parse_method if parse_method is not None else ParseMethod.DEFAULT
            ),
            chunk_strategy=web_final_strategy,
            chunk_size=final_chunk_size,
            chunk_overlap=final_chunk_overlap,
            separators=web_parsed_separators,
            embedding_model_id=embedding_model_id,
            embedding_batch_size=(
                embedding_batch_size
                if embedding_batch_size is not None and embedding_batch_size > 0
                else 10
            ),
            max_retries=(
                max_retries if max_retries is not None and max_retries >= 0 else 3
            ),
            retry_delay=(
                retry_delay if retry_delay is not None and retry_delay >= 0 else 1.0
            ),
        )

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: asyncio.run(
                run_web_ingestion(
                    collection=safe_collection,
                    crawl_config=crawl_config,
                    ingestion_config=ingestion_config,
                    user_id=int(_user.id),
                    is_admin=bool(_user.is_admin),
                )
            ),
        )

        if result.status == "error":
            return JSONResponse(status_code=500, content=result.model_dump())

        return result

    except HTTPException:
        raise
    except (ValueError, KeyError, TypeError) as e:
        logger.error("Data format error in web ingestion: %s", e)
        raise HTTPException(
            status_code=400, detail=f"Data format error: {str(e)}"
        ) from e
    except Exception as e:
        logger.exception("Unexpected error in web ingestion: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Server internal error: {str(e)}",
        ) from e


@kb_router.delete(
    "/collections/{collection_name}",
)
@handle_kb_exceptions
async def delete_collection_api(
    collection_name: str,
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CollectionOperationResult:
    """Delete a collection and all its data.

    This function ensures data consistency by attempting physical file deletion
    before database deletion. If physical deletion fails, the operation is
    aborted to prevent inconsistent state.

    Args:
        collection_name: Name of the collection to delete

    Returns:
        Deletion result with status, affected documents, and cleanup information

    Raises:
        HTTPException: If physical deletion fails (prevents database deletion)
    """
    from filelock import Timeout

    try:
        try:
            safe_collection = sanitize_path_component(collection_name, "collection")
        except ValueError as e:
            raise HTTPException(
                status_code=422, detail=f"Invalid collection name: {str(e)}"
            ) from e

        # Step 1: Move physical directory to trash (under lock) BEFORE database deletion.
        # Note: trash cleanup is NOT automatic; operators should run kb_physical_sync.cleanup_trash
        # via cron/scheduler if they want periodic cleanup.
        physical_cleanup_status = "not_found"  # not_found, success, failed
        physical_cleanup_error = None
        collection_dir: Path | None = None

        try:
            collection_dir = get_upload_path(
                "", user_id=int(_user.id), collection=safe_collection
            )

            if collection_dir.exists() and collection_dir.is_dir():
                try:
                    with collection_physical_lock(collection_dir):
                        move_collection_dir_to_trash(
                            collection_dir,
                            UPLOADS_DIR,
                            int(_user.id),
                            safe_collection,
                        )
                    physical_cleanup_status = "success"
                    logger.info(
                        "Collection directory moved to trash: %s",
                        collection_dir,
                    )
                except Timeout:
                    physical_cleanup_status = "failed"
                    physical_cleanup_error = (
                        "Another operation is in progress; please try again later."
                    )
                    raise HTTPException(
                        status_code=409,
                        detail=physical_cleanup_error,
                    )
                except (PermissionError, OSError) as e:
                    physical_cleanup_status = "failed"
                    physical_cleanup_error = str(e)
                    logger.error(
                        "Failed to move collection directory to trash for %s: %s. "
                        "Aborting to prevent inconsistent state.",
                        collection_name,
                        e,
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "Failed to delete collection: cannot move physical files. "
                            f"Error: {str(e)}. "
                            "Please ensure the directory is not in use and you have proper permissions."
                        ),
                    ) from e
                except Exception as e:
                    physical_cleanup_status = "failed"
                    physical_cleanup_error = str(e)
                    logger.error(
                        "Unexpected error during physical cleanup for %s: %s",
                        collection_name,
                        e,
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "Failed to delete collection: unexpected error during physical cleanup. "
                            f"Error: {str(e)}"
                        ),
                    ) from e
            else:
                physical_cleanup_status = "not_found"
                logger.debug(
                    "Collection directory does not exist (or is not a directory): %s. "
                    "This is normal for collections without physical files.",
                    collection_dir,
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(
                "Error determining collection directory path for %s: %s. "
                "Proceeding with database deletion.",
                collection_name,
                e,
            )
            physical_cleanup_status = "error"
            physical_cleanup_error = f"Path resolution error: {str(e)}"

        result = delete_collection(collection_name, int(_user.id), bool(_user.is_admin))

        # Remove UploadedFile records for this collection path
        if collection_dir is not None:
            prefix = str(collection_dir.resolve()) + os.sep
            dir_str = str(collection_dir.resolve())
            deleted = (
                db.query(UploadedFile)
                .filter(
                    # IMPORTANT:
                    # Constrain by user_id first, then match by storage_path.
                    # Otherwise, top-level OR would delete other users' files
                    # in the same collection and/or delete all files belonging
                    # to the user regardless of collection scope.
                    UploadedFile.user_id == int(_user.id),
                    or_(
                        UploadedFile.storage_path.startswith(prefix),
                        UploadedFile.storage_path == dir_str,
                    ),
                )
                .delete(synchronize_session=False)
            )
            if deleted:
                db.commit()
                logger.info(
                    "Deleted %s UploadedFile record(s) for collection %s",
                    deleted,
                    collection_name,
                )

        # Step 3: Add physical cleanup status to warnings and message for visibility
        # This ensures users are always aware of physical cleanup status, not just in logs
        cleanup_warnings = list(result.warnings) if result.warnings else []
        cleanup_info_message = ""

        if physical_cleanup_status == "success":
            cleanup_info = (
                f"Physical directory moved to trash: {collection_dir} "
                "(trash cleanup requires external scheduler/cron)"
            )
            cleanup_warnings.append(cleanup_info)
            cleanup_info_message = f" {cleanup_info}."
        elif physical_cleanup_status == "not_found":
            cleanup_info = "Physical directory cleanup: No physical directory found (collection had no files)"
            cleanup_warnings.append(cleanup_info)
            cleanup_info_message = f" {cleanup_info}."
        elif physical_cleanup_status == "error" and physical_cleanup_error:
            # Path resolution error - database deletion proceeded, but physical cleanup status is unknown
            cleanup_info = f"Physical directory cleanup: Warning - {physical_cleanup_error}. Database deletion proceeded, but physical file cleanup status is uncertain."
            cleanup_warnings.append(cleanup_info)
            cleanup_info_message = f" {cleanup_info}"
        elif physical_cleanup_status == "failed" and physical_cleanup_error:
            # This should not happen if we aborted above, but include for completeness
            cleanup_info = (
                f"Physical directory cleanup: Failed - {physical_cleanup_error}"
            )
            cleanup_warnings.append(cleanup_info)
            cleanup_info_message = f" {cleanup_info}"

        # Step 4: Determine final status based on both database and physical cleanup results
        # If database deletion succeeded but physical cleanup had issues, mark as partial_success
        final_status = result.status
        if result.status == "success" and physical_cleanup_status in (
            "error",
            "failed",
        ):
            # Database deletion succeeded but physical cleanup had problems
            final_status = "partial_success"
            if not cleanup_info_message:
                cleanup_info_message = " Database deletion succeeded, but physical file cleanup encountered issues."

        # Step 5: Update message to include physical cleanup information
        updated_message = result.message
        if cleanup_info_message:
            updated_message = f"{result.message}{cleanup_info_message}"

        # Create updated result with cleanup information
        # Note: CollectionOperationResult is frozen, so we create a new instance
        updated_result = CollectionOperationResult(
            status=final_status,
            collection=result.collection,
            message=updated_message,
            warnings=cleanup_warnings,
            affected_documents=result.affected_documents,
            deleted_counts=result.deleted_counts,
        )

        return updated_result

    except HTTPException:
        # Re-raise HTTP exceptions (including physical deletion failures)
        raise
    except Exception as e:
        logger.exception(f"Failed to delete collection '{collection_name}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete collection: {str(e)}",
        )


@kb_router.post(
    "/collections/{collection_name}/documents/check",
)
async def check_documents_exist_api(
    collection_name: str,
    body: Dict[str, Any] = Body(
        ..., description="JSON body with 'filenames': list of filename strings"
    ),
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Check which of the given filenames already exist in the collection.

    Used by the frontend to show "file already exists, re-upload?" before ingest.
    Duplicate is determined by: same collection + document with same source_path basename.

    For duplicate check we always filter by current user's documents only (including
    for admins), so "already exists" matches what will be overwritten on re-upload.
    """
    try:
        from ...core.tools.core.RAG_tools.LanceDB.schema_manager import (
            ensure_documents_table,
        )
        from ...core.tools.core.RAG_tools.utils.lancedb_query_utils import query_to_list
        from ...core.tools.core.RAG_tools.utils.string_utils import (
            build_lancedb_filter_expression,
        )
        from ...core.tools.core.RAG_tools.utils.user_permissions import UserPermissions
        from ...providers.vector_store.lancedb import get_connection_from_env

        filenames = body.get("filenames")
        if not isinstance(filenames, list):
            raise HTTPException(
                status_code=422,
                detail="Request body must contain 'filenames' as a list of strings",
            )
        if not all(isinstance(f, str) for f in filenames):
            raise HTTPException(
                status_code=422,
                detail="All 'filenames' elements must be strings",
            )
        requested = {f.strip() for f in filenames if f and f.strip()}
        if not requested:
            return {"existing_filenames": []}

        conn = get_connection_from_env()
        ensure_documents_table(conn)
        table = conn.open_table("documents")

        base_filter = build_lancedb_filter_expression({"collection": collection_name})
        # Use own-files-only filter even for admins so duplicate check matches re-upload behavior
        user_filter = UserPermissions.get_user_filter(int(_user.id), is_admin=False)
        combined_filter = (
            f"({base_filter}) and ({user_filter})"
            if user_filter and base_filter
            else (user_filter or base_filter)
        )
        MAX_SEARCH_RESULTS = 10000
        records = query_to_list(
            table.search().where(combined_filter).limit(MAX_SEARCH_RESULTS)
        )

        existing_basenames = set()
        for record in records:
            sp = record.get("source_path")
            if sp:
                existing_basenames.add(os.path.basename(str(sp)))

        existing_filenames = sorted(requested & existing_basenames)
        return {"existing_filenames": existing_filenames}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to check documents exist: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check documents: {str(e)}",
        ) from e


@kb_router.delete(
    "/collections/{collection_name}/documents/{filename}",
)
@handle_kb_exceptions
async def delete_document_api(
    collection_name: str,
    filename: str,
    _user: User = Depends(get_current_user),
) -> dict:
    """Delete a document and all its associated data.

    Args:
        collection_name: Name of the collection
        filename: Document filename (will be used to find doc_id)

    Returns:
        Deletion result with status, list of deleted doc_ids, and filename

    Note:
        This endpoint uses filename lookup which may have a race condition if
        the same filename is uploaded multiple times concurrently. For production
        use, consider using doc_id directly or adding a filename index column.
    """
    # NOTE: Exceptions are normalized by @handle_kb_exceptions for consistent API responses.
    from ...core.tools.core.RAG_tools.LanceDB.schema_manager import (
        ensure_documents_table,
    )
    from ...core.tools.core.RAG_tools.management.collections import delete_document
    from ...core.tools.core.RAG_tools.utils.lancedb_query_utils import query_to_list
    from ...core.tools.core.RAG_tools.utils.string_utils import (
        build_lancedb_filter_expression,
    )
    from ...core.tools.core.RAG_tools.utils.user_permissions import UserPermissions
    from ...providers.vector_store.lancedb import get_connection_from_env

    # Look up doc_id(s) by filename
    conn = get_connection_from_env()
    ensure_documents_table(conn)
    table = conn.open_table("documents")

    # Filter by collection first to reduce search space
    base_filter = build_lancedb_filter_expression({"collection": collection_name})

    user_filter = UserPermissions.get_user_filter(int(_user.id), bool(_user.is_admin))

    if user_filter and base_filter:
        combined_filter = f"({base_filter}) and ({user_filter})"
    elif user_filter:
        combined_filter = user_filter
    else:
        combined_filter = base_filter

    MAX_SEARCH_RESULTS = 10000
    records = query_to_list(
        table.search().where(combined_filter).limit(MAX_SEARCH_RESULTS)
    )

    matching_docs = []
    for record in records:
        source_path = record.get("source_path", "")
        if source_path and os.path.basename(str(source_path)) == filename:
            matching_docs.append(
                {
                    "doc_id": record.get("doc_id"),
                    "source_path": source_path,
                }
            )

    if not matching_docs:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {filename}",
        )

    deleted_doc_ids = []
    deletion_errors = []

    for doc_info in matching_docs:
        doc_id = doc_info["doc_id"]
        try:
            delete_document(
                collection_name, doc_id, int(_user.id), bool(_user.is_admin)
            )
            deleted_doc_ids.append(doc_id)
            logger.info(
                "Deleted document '%s' (doc_id: %s) from collection '%s'",
                filename,
                doc_id,
                collection_name,
            )
        except Exception as e:
            error_msg = f"Failed to delete doc_id {doc_id}: {str(e)}"
            deletion_errors.append(error_msg)
            logger.error("%s", error_msg)

    if deletion_errors:
        return {
            "status": "partial_success" if deleted_doc_ids else "failed",
            "message": f"Deleted {len(deleted_doc_ids)} of {len(matching_docs)} documents",
            "collection": collection_name,
            "filename": filename,
            "deleted_doc_ids": deleted_doc_ids,
            "errors": deletion_errors,
        }

    return {
        "status": "success",
        "message": f"Successfully deleted {len(deleted_doc_ids)} document(s)",
        "collection": collection_name,
        "filename": filename,
        "deleted_doc_ids": deleted_doc_ids,
    }


@kb_router.put(
    "/collections/{collection_name}",
)
@handle_kb_exceptions
async def rename_collection_api(
    collection_name: str,
    new_name: str = Form(..., description="New collection name"),
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Rename a collection.

    Args:
        collection_name: Current collection name
        new_name: New collection name

    Returns:
        Success message
    """
    from ...core.tools.core.RAG_tools.management.collections import (
        _list_table_names,
    )
    from ...core.tools.core.RAG_tools.management.status import (
        clear_ingestion_status,
        load_ingestion_status,
        write_ingestion_status,
    )
    from ...core.tools.core.RAG_tools.utils.string_utils import (
        escape_lancedb_string,
    )

    conn = get_connection_from_env()

    if not new_name or not new_name.strip():
        raise HTTPException(
            status_code=422,
            detail="New collection name cannot be empty",
        )

    new_name = new_name.strip()

    if new_name == collection_name:
        return {"status": "success", "message": "Collection name unchanged"}

    warnings: list[str] = []

    # SECURITY: Validate both old and new collection names to prevent path traversal
    try:
        safe_old_collection = sanitize_path_component(collection_name, "collection")
        safe_new_collection = sanitize_path_component(new_name, "collection")
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail=f"Invalid collection name: {str(e)}"
        ) from e

    physical_rename_status = "not_found"
    physical_rename_error: Optional[str] = None
    old_collection_dir: Optional[Path] = None
    new_collection_dir: Optional[Path] = None

    # Step 1: Rename physical directory under lock BEFORE updating database
    try:
        from filelock import Timeout

        from ..config import get_upload_path

        old_collection_dir = get_upload_path(
            "",
            user_id=int(_user.id),
            collection=safe_old_collection,
            create_if_not_exists=False,
        )
        new_collection_dir = get_upload_path(
            "",
            user_id=int(_user.id),
            collection=safe_new_collection,
            create_if_not_exists=False,
        )

        if old_collection_dir.exists() and old_collection_dir.is_dir():
            if new_collection_dir.exists():
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Cannot rename collection: target directory already exists. "
                        f"A collection named '{new_name}' already has physical files."
                    ),
                )

            try:
                with collection_physical_lock(old_collection_dir):
                    # Sync DB first; if DB commit fails, do NOT touch the filesystem.
                    old_str = str(old_collection_dir)
                    new_str = str(new_collection_dir)
                    uploads_resolved = UPLOADS_DIR.resolve()
                    records = (
                        db.query(UploadedFile)
                        .filter(
                            UploadedFile.user_id == int(_user.id),
                            UploadedFile.storage_path.startswith(old_str + os.sep),
                        )
                        .all()
                    )
                    previous_paths: dict[int, str] = {
                        int(getattr(rec, "id")): str(getattr(rec, "storage_path"))
                        for rec in records
                    }
                    for rec in records:
                        suffix = rec.storage_path[len(old_str) :]
                        if ".." in suffix:
                            logger.warning(
                                "Skipping storage_path update (invalid suffix): %s",
                                suffix,
                            )
                            continue
                        new_path = new_str + suffix
                        try:
                            Path(new_path).resolve().relative_to(uploads_resolved)
                        except ValueError:
                            logger.warning(
                                "Skipping storage_path update (path outside UPLOADS_DIR): %s",
                                new_path,
                            )
                            continue
                        rec.storage_path = new_path  # type: ignore[assignment]
                    db.commit()  # Commit DB updates BEFORE physical move
                    if records:
                        logger.info(
                            "Updated %d uploaded_files record(s) for renamed collection %s -> %s",
                            len(records),
                            collection_name,
                            new_name,
                        )

                    # Now do the physical move. shutil.move handles cross-device moves.
                    import shutil

                    try:
                        shutil.move(str(old_collection_dir), str(new_collection_dir))
                    except Exception as move_exc:
                        # Best-effort rollback: revert DB paths if physical move fails.
                        logger.error(
                            "Physical collection move failed after DB update for %s -> %s: %s; rolling back DB paths",
                            collection_name,
                            new_name,
                            move_exc,
                        )
                        for rec in records:
                            prior = previous_paths.get(int(getattr(rec, "id")), None)
                            if prior is not None:
                                rec.storage_path = prior  # type: ignore[assignment]
                        try:
                            db.commit()
                        except Exception as rollback_exc:
                            logger.exception(
                                "Rollback DB paths failed for collection rename %s -> %s: %s",
                                collection_name,
                                new_name,
                                rollback_exc,
                            )
                        raise
                physical_rename_status = "success"
                logger.info(
                    "Physically renamed collection directory: %s -> %s",
                    old_collection_dir,
                    new_collection_dir,
                )
            except Timeout:
                physical_rename_status = "failed"
                physical_rename_error = (
                    "Another operation is in progress; please try again later."
                )
                raise HTTPException(
                    status_code=409,
                    detail=physical_rename_error,
                )
            except (PermissionError, OSError) as e:
                physical_rename_status = "failed"
                physical_rename_error = str(e)
                logger.error(
                    "Failed to physically rename collection directory for %s: %s. "
                    "Aborting rename to prevent inconsistent state.",
                    collection_name,
                    e,
                )
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Failed to rename collection: cannot rename physical directory. "
                        f"Error: {str(e)}. "
                        "Please ensure the directory is not in use and you have proper permissions."
                    ),
                ) from e
            except Exception as e:
                physical_rename_status = "failed"
                physical_rename_error = str(e)
                logger.error(
                    "Unexpected error during physical rename for %s: %s",
                    collection_name,
                    e,
                )
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Failed to rename collection: unexpected error during physical directory rename. "
                        f"Error: {str(e)}"
                    ),
                ) from e
        else:
            physical_rename_status = "not_found"
            logger.debug(
                "Collection directory does not exist (or is not a directory): %s. "
                "This is normal for collections without physical files.",
                old_collection_dir,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(
            "Error determining collection directory path for rename %s -> %s: %s. "
            "Proceeding with database rename.",
            collection_name,
            new_name,
            e,
        )
        physical_rename_status = "error"
        physical_rename_error = f"Path resolution error: {str(e)}"

    # Step 2: Update collection name in all tables
    table_names = _list_table_names(conn, warnings)

    for table_name in ["documents", "parses", "chunks"]:
        if table_name in table_names:
            try:
                table = conn.open_table(table_name)
                table.update(
                    f"collection = '{escape_lancedb_string(collection_name)}'",
                    {"collection": new_name},
                )
            except Exception as e:
                logger.warning("Failed to update '%s': %s", table_name, e)
                warnings.append(f"Failed to update '{table_name}': {e}")

    for table_name in table_names:
        if not table_name.startswith("embeddings_"):
            continue
        try:
            table = conn.open_table(table_name)
            table.update(
                f"collection = '{escape_lancedb_string(collection_name)}'",
                {"collection": new_name},
            )
        except Exception as e:
            logger.warning("Failed to update embeddings table '%s': %s", table_name, e)
            warnings.append(f"Failed to update '{table_name}': {e}")

    # Migrate ingestion status from old collection name to new
    try:
        status_entries = load_ingestion_status(collection=collection_name)
        for entry in status_entries:
            doc_id = entry.get("doc_id")
            if doc_id:
                write_ingestion_status(
                    new_name,
                    doc_id,
                    status=entry.get("status", "pending"),
                    message=entry.get("message", ""),
                    parse_hash=entry.get("parse_hash", ""),
                )
                clear_ingestion_status(collection_name, doc_id)
    except Exception as e:
        logger.warning("Failed to update ingestion status: %s", e)
        warnings.append(f"Failed to update ingestion status: {e}")

    # Step 3: Add physical rename status to warnings and message for visibility
    rename_info_message = ""
    if (
        physical_rename_status == "success"
        and old_collection_dir is not None
        and new_collection_dir is not None
    ):
        rename_info = f"Physical directory renamed: {old_collection_dir.name} -> {new_collection_dir.name}"
        warnings.append(rename_info)
        rename_info_message = f" {rename_info}."
    elif physical_rename_status == "not_found":
        rename_info = "Physical directory rename: No physical directory found (collection had no files)"
        warnings.append(rename_info)
        rename_info_message = f" {rename_info}."
    elif physical_rename_status == "error" and physical_rename_error:
        rename_info = (
            f"Physical directory rename: Warning - {physical_rename_error}. "
            "Database rename proceeded, but physical directory rename status is uncertain."
        )
        warnings.append(rename_info)
        rename_info_message = f" {rename_info}"
    elif physical_rename_status == "failed" and physical_rename_error:
        rename_info = f"Physical directory rename: Failed - {physical_rename_error}"
        warnings.append(rename_info)
        rename_info_message = f" {rename_info}"

    # Step 4: Determine final status
    final_status = "success" if not warnings else "partial_success"
    if physical_rename_status in ("error", "failed"):
        final_status = "partial_success"
        if not rename_info_message:
            rename_info_message = " Database rename succeeded, but physical directory rename encountered issues."

    # Step 5: Build final message
    base_message = f"Collection renamed from '{collection_name}' to '{new_name}'"
    if warnings and len(warnings) > (1 if physical_rename_status != "not_found" else 0):
        final_message = f"{base_message} with some warnings"
    else:
        final_message = base_message
    if rename_info_message:
        final_message = f"{final_message}{rename_info_message}"

    if warnings:
        return {
            "status": final_status,
            "message": final_message,
            "warnings": warnings,
        }

    return {
        "status": "success",
        "message": f"Collection renamed from '{collection_name}' to '{new_name}'",
    }


@kb_router.get(
    "/collections/{collection_name}/parses/{doc_id}/parse_result",
    response_model=ParseResultResponse,
)
@handle_kb_exceptions
async def get_parse_result_api(
    collection_name: str,
    doc_id: str,
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Number of elements per page"),
    parse_hash: Optional[str] = Query(
        None,
        description="Optional parse hash to filter. If None, uses the latest parse.",
    ),
    _user: User = Depends(get_current_user),
) -> ParseResultResponse:
    """Get parsed document results with pagination.

    Args:
        collection_name: Collection name
        doc_id: Document ID
        page: Page number (1-indexed, default: 1)
        page_size: Number of elements per page (default: 20)
        parse_hash: Optional parse hash to filter. If None, uses the latest parse.

    Returns:
        ParseResultResponse with paginated text segments, tables, and figures
    """
    from ...core.tools.core.RAG_tools.core.exceptions import DocumentNotFoundError
    from ...core.tools.core.RAG_tools.utils.string_utils import sanitize_for_doc_id

    safe_doc_id = sanitize_for_doc_id(doc_id)
    if safe_doc_id != doc_id:
        logger.warning("Invalid doc_id format detected: %s", doc_id)
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    if page < 1:
        raise HTTPException(status_code=422, detail="Page number must be >= 1")
    if page_size < 1 or page_size > 100:
        raise HTTPException(
            status_code=422, detail="Page size must be between 1 and 100"
        )

    try:
        elements, actual_parse_hash = reconstruct_parse_result_from_db(
            collection_name,
            doc_id,
            parse_hash,
            user_id=int(_user.id),
            is_admin=bool(_user.is_admin),
        )
    except DocumentNotFoundError as e:
        logger.warning("Parse result not found: %s", e)
        raise HTTPException(status_code=404, detail=str(e))

    paginated_elements, pagination_info = paginate_parse_results(
        elements, page, page_size
    )

    return ParseResultResponse(
        doc_id=doc_id,
        parse_hash=actual_parse_hash or "",
        elements=paginated_elements,
        pagination=pagination_info,
    )
