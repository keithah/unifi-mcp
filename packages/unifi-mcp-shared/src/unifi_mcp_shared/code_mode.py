"""Worker-process sandbox and public tools for manifest-backed Code Mode."""

from __future__ import annotations

import ast
import asyncio
import json
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from unifi_mcp_shared.code_mode_catalog import CodeModeCatalog
from unifi_mcp_shared.response_serialization import normalize_call_tool_result

_CODE_MODE_INTERNAL_DISPATCH: ContextVar[bool] = ContextVar("code_mode_internal_dispatch", default=False)
_SOURCE_LIMIT = 64_000
_FAILURE = {"success": False, "error": "Code Mode execution failed."}
_LIMIT_FAILURE = {"success": False, "error": "Code Mode execution exceeded a configured safety limit."}
_SOURCE_FAILURE = {"success": False, "error": "Code Mode source exceeds the configured limit."}
_FORBIDDEN_FAILURE = {"success": False, "error": "Code Mode can invoke only manifest-backed domain tools."}

SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "category": {"type": ["string", "null"]},
        "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
        "detail": {"type": "string", "enum": ["brief", "detailed", "full"], "default": "brief"},
    },
    "required": ["query"],
}
SCHEMA_SCHEMA = {
    "type": "object",
    "properties": {
        "names": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "detail": {"type": "string", "enum": ["detailed", "full"], "default": "detailed"},
    },
    "required": ["names"],
}
EXECUTE_SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string", "maxLength": _SOURCE_LIMIT}},
    "required": ["code"],
}
_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"success": {"type": "boolean"}, "data": {"type": "object"}, "error": {"type": "string"}},
    "required": ["success"],
}


class _LimitExceeded(Exception):
    pass


class _ForbiddenTool(Exception):
    pass


class _BridgeOutcome(Enum):
    FAILURE = _FAILURE["error"]
    FORBIDDEN = _FORBIDDEN_FAILURE["error"]
    LIMIT = _LIMIT_FAILURE["error"]


@dataclass(frozen=True)
class CodeModeLimits:
    max_duration_seconds: float = 30.0
    max_memory_bytes: int = 100_000_000
    max_tool_calls: int = 25
    max_output_chars: int = 16_000
    max_recursion_depth: int = 100
    max_suspensions: int = 128

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> CodeModeLimits:
        values = dict(config or {})

        def read_limit(
            key: str, default: float | int, minimum: float | int, maximum: float | int, cast: Callable
        ) -> Any:
            raw = values.get(key, default)
            if isinstance(raw, bool):
                raise ValueError(f"{key} must be between {minimum} and {maximum}.")
            try:
                value = cast(raw)
            except (TypeError, ValueError):
                raise ValueError(f"{key} must be between {minimum} and {maximum}.") from None
            if not minimum <= value <= maximum:
                raise ValueError(f"{key} must be between {minimum} and {maximum}.")
            return value

        return cls(
            max_duration_seconds=read_limit("max_duration_seconds", 30.0, 0.01, 120.0, float),
            max_memory_bytes=read_limit("max_memory_bytes", 100_000_000, 1_000_000, 512_000_000, int),
            max_tool_calls=read_limit("max_tool_calls", 25, 1, 100, int),
            max_output_chars=read_limit("max_output_chars", 16_000, 1, 64_000, int),
        )


class CodeModeExecutor:
    """Execute one Code Mode program in a fresh one-worker Monty pool."""

    def __init__(self, *, server: Any, catalog: CodeModeCatalog, limits: CodeModeLimits) -> None:
        self.server = server
        self.catalog = catalog
        self.limits = limits

    async def execute(self, code: object, *, context: Context | None) -> dict[str, Any]:
        if not isinstance(code, str):
            return dict(_FAILURE)
        if len(code) > _SOURCE_LIMIT:
            return dict(_SOURCE_FAILURE)
        source_tree = _source_is_allowed(code)
        if source_tree is None:
            return dict(_FAILURE)

        calls = 0
        lock = asyncio.Lock()
        outcome: _BridgeOutcome | None = None

        async def recorded_failure() -> dict[str, Any] | None:
            async with lock:
                if outcome is None:
                    return None
                return {"success": False, "error": outcome.value}

        async def guarded_call_tool(name: object, arguments: object = None) -> Any:
            nonlocal calls, outcome
            async with lock:
                if outcome is not None:
                    return None
                if not isinstance(name, str) or (arguments is not None and not isinstance(arguments, dict)):
                    outcome = _BridgeOutcome.FORBIDDEN
                    return None
                params = {} if arguments is None else arguments
                if name not in self.catalog.names:
                    outcome = _BridgeOutcome.FORBIDDEN
                    return None
                if calls >= self.limits.max_tool_calls:
                    outcome = _BridgeOutcome.LIMIT
                    return None
                calls += 1
                token = _CODE_MODE_INTERNAL_DISPATCH.set(True)
                try:
                    raw_result = await self.server.call_tool(name, params, context=context)
                except ToolError as exc:
                    message = str(exc)
                    if _is_canonical_strict_dispatch_error(self.catalog, name, params, message):
                        return {"success": False, "error": message}
                    outcome = _BridgeOutcome.FAILURE
                    return None
                except asyncio.CancelledError:
                    raise
                except Exception:
                    outcome = _BridgeOutcome.FAILURE
                    return None
                finally:
                    _CODE_MODE_INTERNAL_DISPATCH.reset(token)
                try:
                    return normalize_call_tool_result(raw_result)
                except Exception:
                    outcome = _BridgeOutcome.FAILURE
                    return None

        try:
            from pydantic_monty import AsyncMonty, CollectStreams, MontyRuntimeError
        except Exception:
            return dict(_FAILURE)

        streams = CollectStreams(max_bytes=self.limits.max_output_chars)
        try:
            async with AsyncMonty(
                min_processes=1,
                max_processes=1,
                max_checkouts_per_worker=1,
                request_timeout=self.limits.max_duration_seconds + 5,
            ) as pool:
                async with pool.checkout(
                    limits={
                        "max_duration_secs": self.limits.max_duration_seconds,
                        "max_memory": self.limits.max_memory_bytes,
                        "max_recursion_depth": self.limits.max_recursion_depth,
                        "max_suspensions": self.limits.max_suspensions,
                    }
                ) as session:
                    result = await session.feed_run(
                        code,
                        external_lookup={"call_tool": guarded_call_tool},
                        print_callback=streams,
                    )
        except MontyRuntimeError as exc:
            failure = await recorded_failure()
            if failure is not None:
                return failure
            if _is_configured_monty_limit(exc, limits=self.limits, source_tree=source_tree):
                return dict(_LIMIT_FAILURE)
            return dict(_FAILURE)
        except Exception:
            failure = await recorded_failure()
            if failure is not None:
                return failure
            return dict(_FAILURE)
        failure = await recorded_failure()
        if failure is not None:
            return failure
        if not _json_within_limit(result, self.limits.max_output_chars):
            return dict(_LIMIT_FAILURE)
        return {"success": True, "data": {"result": result}}


def _source_is_allowed(code: str) -> ast.Module | None:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if len(node.names) != 1 or node.names[0].name != "asyncio" or node.names[0].asname is not None:
                return None
        elif isinstance(node, ast.ImportFrom):
            return None
        elif isinstance(node, ast.Raise):
            return None
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"__import__", "compile", "eval", "exec"}:
                return None
            if isinstance(node.func, ast.Attribute) and node.func.attr == "throw":
                return None
    return tree


def _json_within_limit(value: Any, maximum: int) -> bool:
    try:
        encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"))
        size = 0
        for chunk in encoder.iterencode(value):
            size += len(chunk)
            if size > maximum:
                return False
    except (TypeError, ValueError):
        return False
    return True


def _is_canonical_strict_dispatch_error(
    catalog: CodeModeCatalog, name: str, arguments: dict[str, Any], message: str
) -> bool:
    if "***REDACTED***" in message:
        return False
    tool = catalog.by_name.get(name)
    schema = tool.get("schema") if isinstance(tool, dict) else None
    input_schema = schema.get("input") if isinstance(schema, dict) else None
    properties = input_schema.get("properties") if isinstance(input_schema, dict) else None
    if not isinstance(properties, dict):
        return False
    allowed = frozenset(key for key in properties if isinstance(key, str))
    unknown = set(arguments) - allowed
    if not unknown:
        return False
    unknown_str = ", ".join(sorted(unknown))
    valid_str = ", ".join(sorted(allowed))
    hint = _strict_dispatch_mac_hint(name, unknown, allowed, arguments)
    expected = (
        f"Invalid params for '{name}': unknown arguments {{{unknown_str}}}. Valid arguments: [{valid_str}].{hint}"
    )
    return message == expected


def _strict_dispatch_mac_hint(
    tool_name: str, unknown: set[str], allowed: frozenset[str], arguments: dict[str, Any]
) -> str:
    mac_spellings = frozenset({"mac", "mac_address", "client_mac", "device_mac"})
    if not unknown & mac_spellings:
        return ""
    canonical = sorted(allowed & mac_spellings)
    if len(canonical) != 1 or canonical[0] in arguments:
        return ""
    return " '%s' takes the MAC address as '%s'." % (tool_name, canonical[0])


def _is_configured_monty_limit(error: Any, *, limits: CodeModeLimits, source_tree: ast.Module) -> bool:
    inner = error.exception()
    message = str(inner)
    traceback = error.traceback()
    if not isinstance(traceback, list):
        return False
    if isinstance(inner, TimeoutError):
        return not traceback and message.startswith("time limit exceeded:")
    if isinstance(inner, MemoryError):
        return message.startswith("memory limit exceeded:") and (not traceback or source_tree is not None)
    if isinstance(inner, RecursionError):
        return source_tree is not None and message == "maximum recursion depth exceeded"
    return (
        source_tree is not None
        and isinstance(inner, RuntimeError)
        and message == f"suspension limit {limits.max_suspensions} exceeded"
    )


def install_code_mode_dispatch_guard(server: Any, *, prefix: str) -> None:
    """Reject external hidden-tool calls while allowing the bridge task scope."""
    public_names = frozenset({f"{prefix}_code_search", f"{prefix}_code_get_schema", f"{prefix}_code_execute"})
    dispatch = server.call_tool

    async def guarded_dispatch(name: str, arguments: dict[str, Any], context: Context | None = None) -> Any:
        if name not in public_names and not _CODE_MODE_INTERNAL_DISPATCH.get():
            raise ToolError(
                "Direct domain tool calls are unavailable in code_mode; "
                "use the Code Mode discovery and execution tools."
            )
        return await dispatch(name, arguments, context=context)

    server.call_tool = guarded_dispatch


def register_code_mode_tools(
    *,
    server: Any,
    tool_decorator: Callable,
    register_tool: Callable,
    manifest_path: Path,
    prefix: str,
    server_label: str,
    limits_config: Mapping[str, Any] | None,
) -> None:
    """Register exactly the public discovery, schema, and sandbox tools."""
    catalog = CodeModeCatalog.from_manifest(manifest_path, prefix=prefix)
    executor = CodeModeExecutor(server=server, catalog=catalog, limits=CodeModeLimits.from_mapping(limits_config))
    search_name = f"{prefix}_code_search"
    schema_name = f"{prefix}_code_get_schema"
    execute_name = f"{prefix}_code_execute"
    read_annotations = ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    )
    execute_annotations = ToolAnnotations(
        readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False
    )

    @tool_decorator(
        name=search_name,
        title=f"{server_label} Code Search",
        description=f"Search the manifest-backed {server_label} domain tool catalog for Code Mode.",
        annotations=read_annotations,
    )
    async def unifi_code_search(
        query: str, category: str | None = None, limit: int = 10, detail: Literal["brief", "detailed", "full"] = "brief"
    ) -> dict[str, Any]:
        return catalog.search(query, category=category, limit=limit, detail=detail)

    @tool_decorator(
        name=schema_name,
        title=f"{server_label} Code Schema",
        description=f"Get exact manifest schemas for discovered {server_label} domain tools.",
        annotations=read_annotations,
    )
    async def unifi_code_get_schema(
        names: list[str], detail: Literal["detailed", "full"] = "detailed"
    ) -> dict[str, Any]:
        return catalog.get_schema(names, detail=detail)

    @tool_decorator(
        name=execute_name,
        title=f"{server_label} Code Execute",
        description=f"Run bounded Python Code Mode against authorized {server_label} domain tools.",
        annotations=execute_annotations,
    )
    async def unifi_code_execute(code: str, context: Context | None = None) -> dict[str, Any]:
        return await executor.execute(code, context=context)

    registrations = (
        (
            search_name,
            f"{server_label} Code Search",
            "Search the manifest-backed domain tool catalog for Code Mode.",
            SEARCH_SCHEMA,
            read_annotations,
        ),
        (
            schema_name,
            f"{server_label} Code Schema",
            "Get exact manifest schemas for discovered domain tools.",
            SCHEMA_SCHEMA,
            read_annotations,
        ),
        (
            execute_name,
            f"{server_label} Code Execute",
            "Run bounded Python Code Mode against authorized domain tools.",
            EXECUTE_SCHEMA,
            execute_annotations,
        ),
    )
    for name, title, description, input_schema, tool_annotations in registrations:
        server.register_allowed_kwargs(name, input_schema)
        register_tool(
            name=name,
            title=title,
            description=description,
            input_schema=input_schema,
            output_schema=_OUTPUT_SCHEMA,
            annotations={
                "readOnlyHint": tool_annotations.read_only_hint,
                "destructiveHint": tool_annotations.destructive_hint,
                "idempotentHint": tool_annotations.idempotent_hint,
                "openWorldHint": tool_annotations.open_world_hint,
            },
        )
