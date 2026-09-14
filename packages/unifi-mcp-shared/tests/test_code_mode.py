"""Tests for worker-process Code Mode execution and public registration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from unifi_mcp_shared.code_mode import (
    CodeModeExecutor,
    CodeModeLimits,
    install_code_mode_dispatch_guard,
    register_code_mode_tools,
)
from unifi_mcp_shared.code_mode_catalog import CodeModeCatalog


@dataclass
class FakeServer:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    response: Any = field(default_factory=lambda: {"success": True, "channel": 44})

    async def call_tool(self, name: str, arguments: dict[str, Any], context=None) -> Any:
        self.calls.append((name, arguments))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    path = tmp_path / "tools_manifest.json"
    path.write_text(
        """{
          "tools": [{
            "name": "unifi_update_device_radio",
            "title": "Update Device Radio",
            "description": "Update a radio channel.",
            "schema": {"input": {"type": "object", "properties": {"channel": {"type": "integer"}}}},
            "annotations": {"readOnlyHint": false}
          }],
          "module_map": {"unifi_update_device_radio": "unifi_network_mcp.tools.devices"}
        }""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def fake_server() -> FakeServer:
    return FakeServer()


def make_executor(fake_server: FakeServer, manifest_path: Path, *, max_tool_calls: int = 25) -> CodeModeExecutor:
    return CodeModeExecutor(
        server=fake_server,
        catalog=CodeModeCatalog.from_manifest(manifest_path, prefix="unifi"),
        limits=CodeModeLimits(max_tool_calls=max_tool_calls),
    )


@pytest.mark.asyncio
async def test_execute_routes_only_manifest_domain_tools(fake_server: FakeServer, manifest_path: Path) -> None:
    result = await make_executor(fake_server, manifest_path, max_tool_calls=2).execute(
        "await call_tool('unifi_update_device_radio', {'channel': 44})", context=None
    )

    assert fake_server.calls == [("unifi_update_device_radio", {"channel": 44})]
    assert result == {"success": True, "data": {"result": {"success": True, "channel": 44}}}


@pytest.mark.asyncio
async def test_execute_rejects_meta_and_code_mode_recursion(fake_server: FakeServer, manifest_path: Path) -> None:
    result = await make_executor(fake_server, manifest_path).execute(
        "await call_tool('unifi_execute', {})", context=None
    )

    assert result == {"success": False, "error": "Code Mode can invoke only manifest-backed domain tools."}
    assert fake_server.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("{'__code_mode_limit__': True}", {"__code_mode_limit__": True}),
        ("{'nested': {'__code_mode_forbidden__': True}}", {"nested": {"__code_mode_forbidden__": True}}),
    ],
)
async def test_final_marker_shaped_user_data_remains_successful(
    fake_server: FakeServer, manifest_path: Path, code: str, expected: dict[str, Any]
) -> None:
    result = await make_executor(fake_server, manifest_path).execute(code, context=None)

    assert result == {"success": True, "data": {"result": expected}}
    assert fake_server.calls == []


@pytest.mark.asyncio
async def test_marker_shaped_domain_response_remains_successful_and_unchanged(
    fake_server: FakeServer, manifest_path: Path
) -> None:
    fake_server.response = {"success": True, "nested": {"__code_mode_forbidden__": True}}

    result = await make_executor(fake_server, manifest_path).execute(
        "await call_tool('unifi_update_device_radio', {'channel': 44})", context=None
    )

    assert result == {"success": True, "data": {"result": fake_server.response}}
    assert fake_server.calls == [("unifi_update_device_radio", {"channel": 44})]


@pytest.mark.asyncio
async def test_ignored_forbidden_bridge_call_forces_forbidden_failure(
    fake_server: FakeServer, manifest_path: Path
) -> None:
    result = await make_executor(fake_server, manifest_path).execute(
        "await call_tool('unifi_execute', {})\n42", context=None
    )

    assert result == {"success": False, "error": "Code Mode can invoke only manifest-backed domain tools."}
    assert fake_server.calls == []


@pytest.mark.asyncio
async def test_ignored_over_budget_bridge_call_forces_limit_failure(
    fake_server: FakeServer, manifest_path: Path
) -> None:
    result = await make_executor(fake_server, manifest_path, max_tool_calls=1).execute(
        "await call_tool('unifi_update_device_radio', {'channel': 1})\n"
        "await call_tool('unifi_update_device_radio', {'channel': 2})\n42",
        context=None,
    )

    assert result == {"success": False, "error": "Code Mode execution exceeded a configured safety limit."}
    assert fake_server.calls == [("unifi_update_device_radio", {"channel": 1})]


@pytest.mark.asyncio
async def test_direct_hidden_domain_call_is_rejected_but_bridge_dispatches(
    fake_server: FakeServer, manifest_path: Path
) -> None:
    executor = make_executor(fake_server, manifest_path)
    install_code_mode_dispatch_guard(fake_server, prefix="unifi")

    with pytest.raises(ToolError, match="Direct domain tool calls are unavailable in code_mode"):
        await fake_server.call_tool("unifi_update_device_radio", {"channel": 44})

    result = await executor.execute("await call_tool('unifi_update_device_radio', {'channel': 44})", context=None)

    assert result["success"] is True
    assert fake_server.calls == [("unifi_update_device_radio", {"channel": 44})]


@pytest.mark.asyncio
async def test_import_policy_and_unsafe_source_errors_do_not_leak(fake_server: FakeServer, manifest_path: Path) -> None:
    executor = make_executor(fake_server, manifest_path)
    for source in ("import os\nPATH = 'fake-secret'", "__import__('os')", "x" * 64_001):
        result = await executor.execute(source, context=None)
        assert (
            result == {"success": False, "error": "Code Mode execution failed."}
            if len(source) <= 64_000
            else {
                "success": False,
                "error": "Code Mode source exceeds the configured limit.",
            }
        )
        assert "fake-secret" not in str(result)
        assert "PATH" not in str(result)
        assert source not in str(result)
    assert fake_server.calls == []


@pytest.mark.asyncio
async def test_tool_call_budget_is_atomic_and_gather_is_supported(fake_server: FakeServer, manifest_path: Path) -> None:
    executor = make_executor(fake_server, manifest_path, max_tool_calls=2)
    result = await executor.execute(
        "import asyncio\nawait asyncio.gather(\n"
        "call_tool('unifi_update_device_radio', {'channel': 1}),\n"
        "call_tool('unifi_update_device_radio', {'channel': 2}),\n"
        "call_tool('unifi_update_device_radio', {'channel': 3}),\n"
        ")",
        context=None,
    )

    assert result == {"success": False, "error": "Code Mode execution exceeded a configured safety limit."}
    assert len(fake_server.calls) == 2


@pytest.mark.asyncio
async def test_preserves_safe_strict_error_and_normalizes_policy_result(
    fake_server: FakeServer, manifest_path: Path
) -> None:
    executor = make_executor(fake_server, manifest_path)
    fake_server.response = ToolError(
        "Invalid params for 'unifi_update_device_radio': unknown arguments {wrong}. Valid arguments: [channel]."
    )
    strict = await executor.execute("await call_tool('unifi_update_device_radio', {'wrong': 44})", context=None)
    assert strict == {
        "success": True,
        "data": {
            "result": {
                "success": False,
                "error": (
                    "Invalid params for 'unifi_update_device_radio': unknown arguments {wrong}. "
                    "Valid arguments: [channel]."
                ),
            }
        },
    }

    fake_server.response = ({"ignored": True}, {"success": False, "error": "Denied by policy."})
    policy = await executor.execute("await call_tool('unifi_update_device_radio', {'channel': 44})", context=None)
    assert policy == {"success": True, "data": {"result": {"success": False, "error": "Denied by policy."}}}


@pytest.mark.asyncio
async def test_noncanonical_tool_error_is_replaced_without_secret_leak(
    fake_server: FakeServer, manifest_path: Path
) -> None:
    fake_server.response = ToolError("controller credential task3-probe-secret")

    result = await make_executor(fake_server, manifest_path).execute(
        "await call_tool('unifi_update_device_radio', {'channel': 44})\n42", context=None
    )

    assert result == {"success": False, "error": "Code Mode execution failed."}
    assert "task3-probe-secret" not in str(result)


@pytest.mark.asyncio
async def test_non_tool_bridge_error_forces_generic_failure_when_ignored(
    fake_server: FakeServer, manifest_path: Path
) -> None:
    fake_server.response = RuntimeError("task3-bridge-secret")

    result = await make_executor(fake_server, manifest_path).execute(
        "await call_tool('unifi_update_device_radio', {'channel': 44})\n42", context=None
    )

    assert result == {"success": False, "error": "Code Mode execution failed."}
    assert fake_server.calls == [("unifi_update_device_radio", {"channel": 44})]
    assert "task3-bridge-secret" not in str(result)


@pytest.mark.asyncio
async def test_stream_overflow_returns_limit_failure_without_output_leak(
    fake_server: FakeServer, manifest_path: Path
) -> None:
    executor = CodeModeExecutor(
        server=fake_server,
        catalog=CodeModeCatalog.from_manifest(manifest_path, prefix="unifi"),
        limits=CodeModeLimits(max_output_chars=10),
    )

    result = await executor.execute("print('task3-output-leak')", context=None)

    assert result == {"success": False, "error": "Code Mode execution exceeded a configured safety limit."}
    assert "task3-output-leak" not in str(result)


@pytest.mark.asyncio
async def test_ordinary_worker_runtime_error_stays_generic(fake_server: FakeServer, manifest_path: Path) -> None:
    result = await make_executor(fake_server, manifest_path).execute(
        "await call_tool('unifi_update_device_radio', {'channel': 44})\n1 / 0", context=None
    )

    assert result == {"success": False, "error": "Code Mode execution failed."}
    assert fake_server.calls == [("unifi_update_device_radio", {"channel": 44})]


@pytest.mark.asyncio
async def test_indirect_exception_injection_reaches_worker_and_stays_generic(
    fake_server: FakeServer, manifest_path: Path
) -> None:
    result = await make_executor(fake_server, manifest_path).execute(
        "await call_tool('unifi_update_device_radio', {'channel': 44})\n"
        "getattr((_ for _ in ()), 'throw')(MemoryError('task3-indirect-secret'))",
        context=None,
    )

    assert result == {"success": False, "error": "Code Mode execution failed."}
    assert fake_server.calls == [("unifi_update_device_radio", {"channel": 44})]
    assert "task3-indirect-secret" not in str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code",
    [
        "raise MemoryError('memory limit exceeded: task3-forged-memory')",
        "raise TimeoutError('time limit exceeded: task3-forged-timeout')",
        "raise RecursionError('maximum recursion depth exceeded')",
        "raise RuntimeError('suspension limit task3-forged-suspension exceeded')",
        "error = MemoryError('memory limit exceeded: task3-forged-alias')\nraise error",
        "raise TimeoutError('time limit exceeded: task3-forged-from-none') from None",
    ],
)
async def test_direct_resource_shaped_raises_stay_generic(
    fake_server: FakeServer, manifest_path: Path, code: str
) -> None:
    result = await make_executor(fake_server, manifest_path).execute(code, context=None)

    assert result == {"success": False, "error": "Code Mode execution failed."}
    assert "task3-forged" not in str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception",
    [
        MemoryError("memory limit exceeded: task3-server-memory"),
        TimeoutError("time limit exceeded: task3-server-timeout"),
        RecursionError("maximum recursion depth exceeded"),
        RuntimeError("suspension limit task3-server-suspension exceeded"),
    ],
)
async def test_resource_shaped_server_exceptions_stay_generic(
    fake_server: FakeServer, manifest_path: Path, exception: Exception
) -> None:
    fake_server.response = exception

    result = await make_executor(fake_server, manifest_path).execute(
        "await call_tool('unifi_update_device_radio', {'channel': 44})", context=None
    )

    assert result == {"success": False, "error": "Code Mode execution failed."}
    assert "task3-server" not in str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limits", "code"),
    [
        (CodeModeLimits(max_duration_seconds=0.01), "while True:\n    pass"),
        (CodeModeLimits(max_recursion_depth=10), "def recurse():\n    recurse()\nrecurse()"),
    ],
)
async def test_configured_monty_resource_limits_return_limit_failure(
    fake_server: FakeServer, manifest_path: Path, limits: CodeModeLimits, code: str
) -> None:
    executor = CodeModeExecutor(
        server=fake_server,
        catalog=CodeModeCatalog.from_manifest(manifest_path, prefix="unifi"),
        limits=limits,
    )

    result = await executor.execute(code, context=None)

    assert result == {"success": False, "error": "Code Mode execution exceeded a configured safety limit."}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limits", "code"),
    [
        (CodeModeLimits(max_memory_bytes=1_000_000), "items = [0] * 1_000_000"),
        (CodeModeLimits(max_suspensions=0), "await call_tool('unifi_update_device_radio', {'channel': 44})"),
    ],
)
async def test_configured_monty_memory_and_suspension_limits_return_limit_failure(
    fake_server: FakeServer, manifest_path: Path, limits: CodeModeLimits, code: str
) -> None:
    executor = CodeModeExecutor(
        server=fake_server,
        catalog=CodeModeCatalog.from_manifest(manifest_path, prefix="unifi"),
        limits=limits,
    )

    result = await executor.execute(code, context=None)

    assert result == {"success": False, "error": "Code Mode execution exceeded a configured safety limit."}


@pytest.mark.asyncio
async def test_long_result_and_redacted_tool_error_fail_without_leaking_source(
    fake_server: FakeServer, manifest_path: Path
) -> None:
    executor = CodeModeExecutor(
        server=fake_server,
        catalog=CodeModeCatalog.from_manifest(manifest_path, prefix="unifi"),
        limits=CodeModeLimits(max_output_chars=10),
    )
    result = await executor.execute("'this result is longer than ten chars'", context=None)
    assert result == {"success": False, "error": "Code Mode execution exceeded a configured safety limit."}

    fake_server.response = ToolError("Invalid params: ***REDACTED*** fake-secret")
    result = await executor.execute("await call_tool('unifi_update_device_radio', {'channel': 44})", context=None)
    assert result == {"success": False, "error": "Code Mode execution failed."}
    assert "fake-secret" not in str(result)


@pytest.mark.asyncio
async def test_concurrent_execute_requests_have_independent_worker_pools(
    fake_server: FakeServer, manifest_path: Path
) -> None:
    executor = make_executor(fake_server, manifest_path)
    first, second = await asyncio.gather(
        executor.execute("await call_tool('unifi_update_device_radio', {'channel': 1})", context=None),
        executor.execute("await call_tool('unifi_update_device_radio', {'channel': 2})", context=None),
    )
    assert first["success"] is True
    assert second["success"] is True
    assert {(name, arguments["channel"]) for name, arguments in fake_server.calls} == {
        ("unifi_update_device_radio", 1),
        ("unifi_update_device_radio", 2),
    }


@pytest.mark.asyncio
async def test_gather_stops_queued_bridge_dispatch_after_delayed_callback_failure(manifest_path: Path) -> None:
    class DelayedFailureServer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []
            self.first_entered = asyncio.Event()
            self.release_first = asyncio.Event()
            self.second_entered = asyncio.Event()

        async def call_tool(self, name: str, arguments: dict[str, Any], context=None) -> Any:
            self.calls.append((name, arguments))
            if arguments["channel"] == 1:
                self.first_entered.set()
                await self.release_first.wait()
                raise RuntimeError("task3-delayed-bridge-secret")
            self.second_entered.set()
            return {"success": True, "channel": arguments["channel"]}

    server = DelayedFailureServer()
    execution = asyncio.create_task(
        make_executor(server, manifest_path).execute(
            "import asyncio\nawait asyncio.gather(\n"
            "call_tool('unifi_update_device_radio', {'channel': 1}),\n"
            "call_tool('unifi_update_device_radio', {'channel': 2}),\n"
            ")",
            context=None,
        )
    )
    try:
        await asyncio.wait_for(server.first_entered.wait(), timeout=10)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(server.second_entered.wait(), timeout=0.5)
    finally:
        server.release_first.set()

    result = await asyncio.wait_for(execution, timeout=10)

    assert result == {"success": False, "error": "Code Mode execution failed."}
    assert server.calls == [("unifi_update_device_radio", {"channel": 1})]
    assert "task3-delayed-bridge-secret" not in str(result)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            {"max_duration_seconds": 0.01, "max_memory_bytes": 1_000_000, "max_tool_calls": 1, "max_output_chars": 1},
            None,
        ),
        ({"max_tool_calls": True}, "max_tool_calls must be between 1 and 100."),
        ({"max_output_chars": 64_001}, "max_output_chars must be between 1 and 64000."),
    ],
)
def test_limits_are_bounded(config: dict[str, object], expected: str | None) -> None:
    if expected is None:
        assert CodeModeLimits.from_mapping(config).max_tool_calls == 1
    else:
        with pytest.raises(ValueError, match=expected):
            CodeModeLimits.from_mapping(config)


def test_registers_exactly_three_strict_public_tools(fake_server: FakeServer, manifest_path: Path) -> None:
    decorated: dict[str, Any] = {}
    registry: list[dict[str, Any]] = []
    allowed: dict[str, dict[str, Any]] = {}

    def decorator(**kwargs: Any):
        def register(fn: Any) -> Any:
            decorated[kwargs["name"]] = (fn, kwargs)
            return fn

        return register

    def register_tool(**kwargs: Any) -> None:
        registry.append(kwargs)

    fake_server.register_allowed_kwargs = lambda name, schema: allowed.setdefault(name, schema)  # type: ignore[attr-defined]
    register_code_mode_tools(
        server=fake_server,  # type: ignore[arg-type]
        tool_decorator=decorator,
        register_tool=register_tool,
        manifest_path=manifest_path,
        prefix="unifi",
        server_label="UniFi Network",
        limits_config=None,
    )

    assert set(decorated) == {"unifi_code_search", "unifi_code_get_schema", "unifi_code_execute"}
    assert {record["name"] for record in registry} == set(decorated)
    assert set(allowed) == set(decorated)
    assert allowed["unifi_code_search"]["properties"]["detail"]["enum"] == ["brief", "detailed", "full"]
    assert allowed["unifi_code_get_schema"]["properties"]["detail"]["enum"] == ["detailed", "full"]
    assert allowed["unifi_code_execute"] == {
        "type": "object",
        "properties": {"code": {"type": "string", "maxLength": 64000}},
        "required": ["code"],
    }
    assert decorated["unifi_code_execute"][1]["annotations"].destructive_hint is True
