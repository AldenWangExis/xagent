"""Langfuse tracing integration for LangGraph adapter."""

import contextvars
import functools
import logging
import threading
import time
from importlib.metadata import version as get_version
from typing import Any, AsyncIterator, Callable, Dict, Iterator, Optional, Union, cast

from langfuse import Langfuse
from packaging.version import Version

from .langfuse_config import LangfuseConfig, load_langfuse_config

# Check langfuse version for API compatibility
# langfuse v4+ uses start_observation, v3 uses start_span
LANGFUSE_VERSION = Version(get_version("langfuse"))

# Set up logger
logger = logging.getLogger(__name__)

# Global tracer instance and lock for thread safety
_tracer: Optional["LangfuseTracer"] = None
_tracer_lock = threading.Lock()

# Context variable to track current parent span
_current_parent_span: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "current_parent_span", default=None
)


class LangfuseTracer:
    """Langfuse tracer for LangGraph execution."""

    def __init__(self, config: LangfuseConfig):
        self.config = config
        self.langfuse: Optional[Langfuse] = None

        if config.enabled and config.public_key and config.secret_key:
            self.langfuse = Langfuse(
                public_key=config.public_key,
                secret_key=config.secret_key,
                host=config.host,
                debug=config.debug,
                flush_at=config.flush_at,
                flush_interval=config.flush_interval,
            )

    def is_enabled(self) -> bool:
        """Check if tracing is enabled and properly configured."""
        return self.langfuse is not None

    # Only for test
    def flush(self) -> None:
        if self.langfuse:
            self.langfuse.flush()

    def create_span(self, name: str, parent_span: Optional[Any] = None) -> Any:
        if self.is_enabled() and self.langfuse:
            # Use v4 API (start_observation) for langfuse >= 4.0, otherwise v3 API (start_span)
            if LANGFUSE_VERSION >= Version("4.0.0"):
                # v4+ API
                if parent_span:
                    return parent_span.start_observation(name=name)
                else:
                    return self.langfuse.start_observation(name=name)
            else:
                # v3 API: call start_span (removed in v4). Cast to Any to keep mypy
                # happy across langfuse versions without per-version ignores.
                if parent_span:
                    parent_any = cast(Any, parent_span)
                    return getattr(parent_any, "start_span")(name=name)
                else:
                    langfuse_any = cast(Any, self.langfuse)
                    return getattr(langfuse_any, "start_span")(name=name)
        return None


def init_tracer(storage_root: str) -> None:
    """Initialize the global tracer in a thread-safe manner."""
    global _tracer
    with _tracer_lock:
        if _tracer is None:
            config = (
                LangfuseConfig()
                if not storage_root
                else load_langfuse_config(storage_root)
            )
            _tracer = LangfuseTracer(config)


def get_tracer() -> Optional[LangfuseTracer]:
    """Get the global tracer instance in a thread-safe manner."""
    with _tracer_lock:
        return _tracer


def reset_tracer() -> None:
    """Reset the global tracer instance in a thread-safe manner."""
    global _tracer
    with _tracer_lock:
        _tracer = None


def _serialize_message(msg: Any) -> Dict[str, Any]:
    """Helper function to serialize a single message for langfuse tracing.

    Handles different message types (AIMessage, HumanMessage, ToolMessage)
    and extracts relevant attributes for each type.
    """
    if hasattr(msg, "content"):
        msg_data = {
            "type": msg.__class__.__name__,
            "content": msg.content,
        }

        # Handle AI messages with tool_calls
        if (
            hasattr(msg, "tool_calls")
            and msg.tool_calls
            and hasattr(msg.tool_calls, "__len__")
            and len(msg.tool_calls) > 0
        ):
            msg_data["tool_calls"] = msg.tool_calls

        # Handle ToolMessage attributes
        if hasattr(msg, "tool_call_id") and getattr(msg, "tool_call_id", None):
            msg_data["tool_call_id"] = msg.tool_call_id
        if hasattr(msg, "name") and getattr(msg, "name", None):
            msg_data["tool_name"] = msg.name

        return msg_data
    elif hasattr(msg, "dict"):
        return dict(msg.dict())
    else:
        return {"type": "Unknown", "content": str(msg)}


def _serialize_messages(messages: list[Any]) -> list[Any]:
    """Helper function to serialize a list of messages for langfuse tracing."""
    if not messages:
        return []

    return [_serialize_message(msg) for msg in messages]


def _create_child_span(name: str) -> Any:
    """Helper function to create a child span with current parent context."""
    tracer = get_tracer()
    if not tracer or not tracer.is_enabled():
        return None

    parent_span = _current_parent_span.get()
    return tracer.create_span(name=name, parent_span=parent_span)


def _create_root_span(name: str) -> Any:
    """Helper function to create a root span (for TracedGraph methods)."""
    tracer = get_tracer()
    if not tracer or not tracer.is_enabled():
        return None

    return tracer.create_span(name=name)


def _extract_messages(data: Any) -> list[Any]:
    """Helper function to extract messages from state or result dict."""
    if isinstance(data, dict) and "messages" in data and data["messages"]:
        return list(data["messages"])
    return []


def _add_messages_to_metadata(
    metadata: Dict[str, Any],
    messages_before: Optional[list[Any]] = None,
    messages_after: Optional[list[Any]] = None,
) -> None:
    """Helper function to add message data to metadata dict."""
    if messages_before:
        metadata["messages_before"] = _serialize_messages(messages_before)
        metadata["messages_before_count"] = len(messages_before)

    if messages_after:
        metadata["messages_after"] = _serialize_messages(messages_after)
        metadata["messages_after_count"] = len(messages_after)


def _safe_update_span(span: Any, metadata: Dict[str, Any]) -> None:
    """Helper function to safely update and end a span with error handling."""
    if not span:
        return

    try:
        span.update(metadata=metadata)
        span.end()
    except Exception as e:
        # Log span update failure but don't break execution
        logger.warning(f"Failed to update Langfuse span: {e}")


class ExecutionTimer:
    """Context manager for tracking execution time."""

    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.execution_time: Optional[float] = None

    def __enter__(self) -> "ExecutionTimer":
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.start_time is not None:
            self.execution_time = time.time() - self.start_time

    def get_execution_time(self) -> float:
        """Get current execution time, works even during exceptions."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time


def trace_node(node_id: str, node_type: str) -> Callable[[Callable], Callable]:
    """Decorator to trace node execution with spans."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(state: Any) -> Any:
            logger.info(f"Running {func.__name__}: {node_id}")
            tracer = get_tracer()
            if not tracer or not tracer.is_enabled():
                return await func(state)

            with ExecutionTimer() as timer:
                # Extract all messages before function call
                messages_before = _extract_messages(state)

                # Create span for this node execution (as child of current parent span)
                span = _create_child_span(name=f"agent_{node_id}")

                try:
                    result = await func(state)

                    # Extract all messages after function call
                    messages_after = _extract_messages(result)

                    # Update span with execution metadata
                    if span:
                        metadata: Dict[str, Union[str, int, float, Any]] = {
                            "execution_time_seconds": timer.get_execution_time(),
                            "node_id": node_id,
                            "node_type": node_type,
                        }

                        # Add message data to metadata
                        _add_messages_to_metadata(
                            metadata, messages_before, messages_after
                        )

                        _safe_update_span(span, metadata)

                    return result
                except Exception as e:
                    # Update span with error metadata
                    if span:
                        error_metadata: Dict[str, Union[str, int, float, Any]] = {
                            "execution_time_seconds": timer.get_execution_time(),
                            "node_id": node_id,
                            "node_type": node_type,
                            "error": str(e),
                        }

                        # Add messages before function call for exception case
                        _add_messages_to_metadata(
                            error_metadata, messages_before=messages_before
                        )

                        _safe_update_span(span, error_metadata)
                    raise

        return wrapper

    return decorator


def trace_tool_call(tool_id: str) -> Callable[[Callable], Callable]:
    """Decorator to trace tool calls with spans."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(state: Any) -> Any:
            tracer = get_tracer()
            if not tracer or not tracer.is_enabled():
                return await func(state)

            with ExecutionTimer() as timer:
                # Extract all messages before function call
                messages_before = _extract_messages(state)

                # Extract tool call information for metadata
                metadata: Dict[str, Union[str, int, float, Any]] = {
                    "id": tool_id,
                }

                # Create span for this tool call (as child of current parent span)
                span = _create_child_span(name=f"tool_{tool_id}")

                if messages_before:
                    last_message = messages_before[-1]
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        # Log tool call metadata
                        metadata["tool_calls"] = last_message.tool_calls
                        metadata["tool_call_count"] = len(last_message.tool_calls)

                try:
                    result = await func(state)

                    # Extract all messages after function call
                    messages_after = _extract_messages(result)

                    # Update span with execution metadata
                    if span:
                        metadata["execution_time_seconds"] = timer.get_execution_time()

                        # Add message data to metadata
                        _add_messages_to_metadata(
                            metadata, messages_before, messages_after
                        )

                        _safe_update_span(span, metadata)

                    return result
                except Exception as e:
                    # Update span with error metadata
                    if span:
                        error_metadata: Dict[str, Union[str, int, float, Any]] = {
                            "execution_time_seconds": timer.get_execution_time(),
                            "id": tool_id,
                            "error": str(e),
                        }

                        # Add messages before function call for exception case
                        _add_messages_to_metadata(
                            error_metadata, messages_before=messages_before
                        )

                        _safe_update_span(span, error_metadata)
                    raise

        return wrapper

    return decorator


class TracedGraph:
    """Wrapper for LangGraph Pregel instances that adds parent span tracing."""

    def __init__(self, compiled_graph: Any, graph_id: str = "langgraph_execution"):
        """Initialize the traced graph wrapper.

        Args:
            compiled_graph: The compiled LangGraph Pregel instance
            graph_id: Identifier for the graph being executed
        """
        self.compiled_graph = compiled_graph
        self.graph_id = graph_id

    def invoke(
        self, input_data: Any, config: Optional[Dict] = None, **kwargs: Any
    ) -> Any:
        """Synchronous invoke with parent span tracing."""
        tracer = get_tracer()
        if not tracer or not tracer.is_enabled():
            return self.compiled_graph.invoke(input_data, config, **kwargs)

        start_time = time.time()
        span = _create_root_span(name=f"graph_execution_{self.graph_id}")

        try:
            # Set the parent span context for child operations
            token = _current_parent_span.set(span)
            try:
                result = self.compiled_graph.invoke(input_data, config, **kwargs)
            finally:
                _current_parent_span.reset(token)

            execution_time = time.time() - start_time

            if span:
                metadata = {
                    "execution_time_seconds": execution_time,
                    "graph_id": self.graph_id,
                    "method": "invoke",
                    "input_messages": self._serialize_messages(input_data),
                    "output_messages": self._serialize_messages(result),
                }
                _safe_update_span(span, metadata)

            return result
        except Exception as e:
            execution_time = time.time() - start_time

            if span:
                error_metadata = {
                    "execution_time_seconds": execution_time,
                    "graph_id": self.graph_id,
                    "method": "invoke",
                    "error": str(e),
                    "input_messages": self._serialize_messages(input_data),
                }
                _safe_update_span(span, error_metadata)
            raise

    async def ainvoke(
        self, input_data: Any, config: Optional[Dict] = None, **kwargs: Any
    ) -> Any:
        """Asynchronous invoke with parent span tracing."""
        tracer = get_tracer()
        if not tracer or not tracer.is_enabled():
            return await self.compiled_graph.ainvoke(input_data, config, **kwargs)

        start_time = time.time()
        span = _create_root_span(name=f"graph_execution_{self.graph_id}")

        try:
            # Set the parent span context for child operations
            token = _current_parent_span.set(span)
            try:
                result = await self.compiled_graph.ainvoke(input_data, config, **kwargs)
            finally:
                _current_parent_span.reset(token)

            execution_time = time.time() - start_time

            if span:
                metadata = {
                    "execution_time_seconds": execution_time,
                    "graph_id": self.graph_id,
                    "method": "ainvoke",
                    "input_messages": self._serialize_messages(input_data),
                    "output_messages": self._serialize_messages(result),
                }
                _safe_update_span(span, metadata)

            return result
        except Exception as e:
            execution_time = time.time() - start_time

            if span:
                error_metadata = {
                    "execution_time_seconds": execution_time,
                    "graph_id": self.graph_id,
                    "method": "ainvoke",
                    "error": str(e),
                    "input_messages": self._serialize_messages(input_data),
                }
                _safe_update_span(span, error_metadata)
            raise

    def stream(
        self, input_data: Any, config: Optional[Dict] = None, **kwargs: Any
    ) -> Iterator[Any]:
        """Synchronous stream with parent span tracing."""
        tracer = get_tracer()
        if not tracer or not tracer.is_enabled():
            yield from self.compiled_graph.stream(input_data, config, **kwargs)
            return

        start_time = time.time()
        span = _create_root_span(name=f"graph_stream_{self.graph_id}")
        stream_events = 0
        last_result = None

        try:
            # Set the parent span context for child operations
            token = _current_parent_span.set(span)
            try:
                for result in self.compiled_graph.stream(input_data, config, **kwargs):
                    stream_events += 1
                    last_result = result
                    yield result
            finally:
                _current_parent_span.reset(token)

            execution_time = time.time() - start_time

            if span:
                metadata = {
                    "execution_time_seconds": execution_time,
                    "graph_id": self.graph_id,
                    "method": "stream",
                    "stream_events_count": stream_events,
                    "input_messages": self._serialize_messages(input_data),
                    "output_messages": self._serialize_messages(last_result),
                }
                _safe_update_span(span, metadata)
        except Exception as e:
            execution_time = time.time() - start_time

            if span:
                error_metadata = {
                    "execution_time_seconds": execution_time,
                    "graph_id": self.graph_id,
                    "method": "stream",
                    "stream_events_count": stream_events,
                    "error": str(e),
                    "input_messages": self._serialize_messages(input_data),
                }
                _safe_update_span(span, error_metadata)
            raise

    async def astream(
        self, input_data: Any, config: Optional[Dict] = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        """Asynchronous stream with parent span tracing."""
        tracer = get_tracer()
        if not tracer or not tracer.is_enabled():
            async for result in self.compiled_graph.astream(
                input_data, config, subgraphs=True, **kwargs
            ):
                yield result
            return

        start_time = time.time()
        span = _create_root_span(name=f"graph_stream_{self.graph_id}")
        stream_events = 0
        last_result = None

        try:
            # Set the parent span context for child operations
            token = _current_parent_span.set(span)
            try:
                async for result in self.compiled_graph.astream(
                    input_data, config, **kwargs
                ):
                    stream_events += 1
                    last_result = result
                    yield result
            finally:
                _current_parent_span.reset(token)

            execution_time = time.time() - start_time

            if span:
                metadata = {
                    "execution_time_seconds": execution_time,
                    "graph_id": self.graph_id,
                    "method": "astream",
                    "stream_events_count": stream_events,
                    "input_messages": self._serialize_messages(input_data),
                    "output_messages": self._serialize_messages(last_result),
                }
                _safe_update_span(span, metadata)
        except Exception as e:
            execution_time = time.time() - start_time

            if span:
                error_metadata = {
                    "execution_time_seconds": execution_time,
                    "graph_id": self.graph_id,
                    "method": "astream",
                    "stream_events_count": stream_events,
                    "error": str(e),
                    "input_messages": self._serialize_messages(input_data),
                }
                _safe_update_span(span, error_metadata)
            raise

    def _serialize_messages(self, data: Any) -> list[Any]:
        """Helper method to serialize messages in input/output data."""
        try:
            if isinstance(data, dict) and "messages" in data and data["messages"]:
                return _serialize_messages(data["messages"])
            return []
        except Exception as e:
            logger.warning(f"Failed to serialize messages: {e}")
            return []

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the wrapped compiled graph."""
        return getattr(self.compiled_graph, name)
