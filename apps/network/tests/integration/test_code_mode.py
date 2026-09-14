"""Integration coverage for Network's public Code Mode radio path."""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import AsyncGenerator, Callable
from types import ModuleType
from typing import cast
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from mcp.server.mcpserver.exceptions import ToolError


class RecordingDeviceManager:
    """Minimal radio manager used after the lazy devices module is imported."""

    def __init__(self) -> None:
        self._connection = type("Connection", (), {"site": "default"})()
        self.update_calls: list[tuple[str, str, dict[str, int]]] = []

    async def get_device_radio(self, mac_address: str) -> dict[str, object]:
        return {
            "mac": mac_address,
            "name": "Test AP",
            "radios": [{"radio": "wifi0", "name": "wifi0", "channel": 36}],
        }

    async def update_device_radio(self, mac_address: str, radio: str, updates: dict[str, int]) -> bool:
        self.update_calls.append((mac_address, radio, updates))
        return True


def _json_payload(result: object) -> dict[str, object]:
    """Decode the content-only result returned by the public MCP handler."""
    content = getattr(result, "content", result)
    for block in content:
        text = getattr(block, "text", None)
        if text is not None:
            return json.loads(text)
    raise AssertionError(f"No JSON content in {result!r}")


def _clear_network_import_state() -> None:
    """Remove the package and children so import-time configuration is fresh."""
    for name in tuple(sys.modules):
        if name == "unifi_network_mcp" or name.startswith("unifi_network_mcp."):
            sys.modules.pop(name)


@pytest_asyncio.fixture
async def network_code_mode_server(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setenv("UNIFI_TOOL_REGISTRATION_MODE", "code_mode")
    monkeypatch.setenv("UNIFI_HOST", "127.0.0.1")
    monkeypatch.setenv("UNIFI_USERNAME", "test")
    monkeypatch.setenv("UNIFI_PASSWORD", "test")

    original_import_state = {
        name: module
        for name, module in sys.modules.items()
        if name == "unifi_network_mcp" or name.startswith("unifi_network_mcp.")
    }
    _clear_network_import_state()

    try:
        main = importlib.import_module("unifi_network_mcp.main")
        monkeypatch.setattr(main.connection_manager, "initialize", AsyncMock(return_value=False))
        monkeypatch.setattr(main.connection_manager, "cleanup", AsyncMock())
        monkeypatch.setattr(main.event_manager, "stop_listening", AsyncMock())
        monkeypatch.setattr("unifi_mcp_shared.bootstrap.assert_credentials_configured", lambda *args, **kwargs: None)
        monkeypatch.setattr("unifi_mcp_shared.transport.run_transports", AsyncMock())
        monkeypatch.setattr(
            "unifi_mcp_shared.transport.resolve_http_config", lambda *args, **kwargs: (False, "http", "127.0.0.1", 3000)
        )
        await main.main_async()
        yield main.server
    finally:
        _clear_network_import_state()
        sys.modules.update(original_import_state)


@pytest.mark.asyncio
async def test_code_mode_fixture_restores_preexisting_import_graph_after_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fixture teardown restores the normal-mode graph it temporarily replaces."""
    original_parent = ModuleType("unifi_network_mcp")
    original_runtime = ModuleType("unifi_network_mcp.runtime")
    setattr(original_parent, "runtime", original_runtime)
    monkeypatch.setitem(sys.modules, "unifi_network_mcp", original_parent)
    monkeypatch.setitem(sys.modules, "unifi_network_mcp.runtime", original_runtime)

    fixture_factory = cast(
        Callable[[pytest.MonkeyPatch], AsyncGenerator[object, None]],
        getattr(network_code_mode_server, "__wrapped__"),
    )
    fixture_generator = fixture_factory(monkeypatch)
    await anext(fixture_generator)
    await fixture_generator.aclose()

    assert sys.modules["unifi_network_mcp"] is original_parent
    assert sys.modules["unifi_network_mcp.runtime"] is original_runtime
    assert original_parent.runtime is original_runtime


@pytest.mark.asyncio
async def test_code_mode_fixture_purges_parent_and_children_after_setup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fixture teardown must remove a parent package that caches a child module."""
    stale_parent = ModuleType("unifi_network_mcp")
    stale_runtime = ModuleType("unifi_network_mcp.runtime")
    setattr(stale_parent, "runtime", stale_runtime)
    ambient_import_state = {
        name: module
        for name, module in sys.modules.items()
        if name == "unifi_network_mcp" or name.startswith("unifi_network_mcp.")
    }
    _clear_network_import_state()
    original_import_module = importlib.import_module

    def fail_main_import(name: str, package: str | None = None) -> ModuleType:
        if name == "unifi_network_mcp.main":
            monkeypatch.setitem(sys.modules, "unifi_network_mcp", stale_parent)
            monkeypatch.setitem(sys.modules, "unifi_network_mcp.runtime", stale_runtime)
            raise RuntimeError("synthetic Code Mode setup failure")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fail_main_import)
    fixture_factory = cast(
        Callable[[pytest.MonkeyPatch], AsyncGenerator[object, None]],
        getattr(network_code_mode_server, "__wrapped__"),
    )
    fixture_generator = fixture_factory(monkeypatch)
    try:
        with pytest.raises(RuntimeError, match="synthetic Code Mode setup failure"):
            await anext(fixture_generator)
        await fixture_generator.aclose()

        assert "unifi_network_mcp" not in sys.modules
        assert "unifi_network_mcp.runtime" not in sys.modules
    finally:
        _clear_network_import_state()
        sys.modules.update(ambient_import_state)


@pytest_asyncio.fixture
async def fake_device_manager(
    network_code_mode_server: object, monkeypatch: pytest.MonkeyPatch
) -> RecordingDeviceManager:
    manager = RecordingDeviceManager()
    runtime = importlib.import_module("unifi_network_mcp.runtime")
    monkeypatch.setattr(runtime, "device_manager", manager)
    return manager


@pytest.mark.asyncio
async def test_code_mode_discovers_and_runs_radio_channel_preview_then_confirm(
    network_code_mode_server: object, fake_device_manager: RecordingDeviceManager
) -> None:
    server = network_code_mode_server
    names = {tool.name for tool in await server.list_tools()}
    assert names == {"unifi_code_search", "unifi_code_get_schema", "unifi_code_execute"}

    search = await server.call_tool("unifi_code_search", {"query": "set AP radio channel"})
    assert "unifi_update_device_radio" in json.dumps(_json_payload(search))

    preview = await server.call_tool(
        "unifi_code_execute",
        {
            "code": "await call_tool('unifi_update_device_radio', {'mac_address': 'aa:bb:cc:dd:ee:ff', 'radio': 'wifi0', 'channel': 44})"
        },
    )
    preview_payload = _json_payload(preview)
    assert preview_payload["data"]["result"].get("requires_confirmation") is True, preview_payload
    assert fake_device_manager.update_calls == []

    confirmed = await server.call_tool(
        "unifi_code_execute",
        {
            "code": "await call_tool('unifi_update_device_radio', {'mac_address': 'aa:bb:cc:dd:ee:ff', 'radio': 'wifi0', 'channel': 44, 'confirm': True})"
        },
    )
    assert _json_payload(confirmed)["success"] is True
    assert fake_device_manager.update_calls == [("aa:bb:cc:dd:ee:ff", "wifi0", {"channel": 44})]

    with pytest.raises(ToolError, match="Direct domain tool calls are unavailable in code_mode"):
        await server.call_tool("unifi_update_device_radio", {})


@pytest.mark.asyncio
async def test_code_mode_public_search_rejects_unknown_arguments_before_handler(
    network_code_mode_server: object,
) -> None:
    with pytest.raises(ToolError, match=r"Invalid params for 'unifi_code_search': unknown arguments \{unexpected\}"):
        await network_code_mode_server.call_tool("unifi_code_search", {"query": "radio", "unexpected": True})
