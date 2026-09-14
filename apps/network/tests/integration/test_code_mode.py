"""Integration coverage for Network's public Code Mode radio path."""

from __future__ import annotations

import importlib
import json
import sys
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


@pytest_asyncio.fixture
async def network_code_mode_server(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setenv("UNIFI_TOOL_REGISTRATION_MODE", "code_mode")
    monkeypatch.setenv("UNIFI_HOST", "127.0.0.1")
    monkeypatch.setenv("UNIFI_USERNAME", "test")
    monkeypatch.setenv("UNIFI_PASSWORD", "test")

    for name in tuple(sys.modules):
        if name.startswith("unifi_network_mcp."):
            sys.modules.pop(name)

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
        for name in tuple(sys.modules):
            if name.startswith("unifi_network_mcp."):
                sys.modules.pop(name)


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
