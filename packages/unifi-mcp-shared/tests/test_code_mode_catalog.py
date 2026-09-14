"""Tests for manifest-backed Code Mode discovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from unifi_mcp_shared.code_mode_catalog import CodeModeCatalog
from unifi_mcp_shared.tool_index import load_tool_manifest, rank_tools_by_search


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    path = tmp_path / "tools_manifest.json"
    schema = {
        "input": {
            "type": "object",
            "properties": {
                "mac_address": {"type": "string", "description": "Access point MAC."},
                "radio": {"type": "string"},
                "channel": {"type": "integer"},
                "confirm": {"type": "boolean"},
            },
            "required": ["mac_address", "radio", "channel"],
        },
        "output": {"type": "object", "properties": {"success": {"type": "boolean"}}},
    }
    path.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "name": "unifi_update_device_radio",
                        "title": "Update Device Radio",
                        "description": "Preview or confirm an AP radio channel update.",
                        "schema": schema,
                        "annotations": {"readOnlyHint": False, "openWorldHint": False},
                    },
                    {
                        "name": "unifi_list_clients",
                        "title": "List Clients",
                        "description": "List connected clients.",
                        "schema": {"input": {"type": "object", "properties": {}}},
                    },
                    {
                        "name": "unifi_execute",
                        "description": "Legacy meta execution",
                        "schema": {"input": {"type": "object"}},
                    },
                    {
                        "name": "unifi_code_search",
                        "description": "Code Mode search",
                        "schema": {"input": {"type": "object"}},
                    },
                    {
                        "name": "protect_list_cameras",
                        "description": "List cameras.",
                        "schema": {"input": {"type": "object"}},
                    },
                ],
                "module_map": {
                    "unifi_update_device_radio": "unifi_network_mcp.tools.devices",
                    "unifi_list_clients": "unifi_network_mcp.tools.clients",
                    "unifi_execute": "unifi_network_mcp.tools.meta",
                    "unifi_code_search": "unifi_network_mcp.tools.code_mode",
                    "protect_list_cameras": "unifi_protect_mcp.tools.cameras",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_public_manifest_helpers_preserve_bounded_and_unbounded_ranking(manifest_path: Path) -> None:
    manifest = load_tool_manifest(manifest_path)

    assert manifest is not None
    tools = [{"name": f"unifi_client_{index}", "description": "Inspect client"} for index in range(25)]
    assert len(rank_tools_by_search(tools, "client")) == 20
    assert len(rank_tools_by_search(tools, "client", limit=None)) == 25


def test_search_surfaces_radio_mutation_and_hides_meta_tools(manifest_path: Path) -> None:
    catalog = CodeModeCatalog.from_manifest(manifest_path, prefix="unifi")

    result = catalog.search("set AP radio channel", category=None, limit=10, detail="brief")

    assert [tool["name"] for tool in result["tools"]] == ["unifi_update_device_radio"]
    assert result["total_matches"] == 1
    assert result["truncated"] is False
    assert result["categories"] == ["clients", "devices"]
    assert result["tools"][0] == {
        "name": "unifi_update_device_radio",
        "title": "Update Device Radio",
        "description": "Preview or confirm an AP radio channel update.",
        "category": "devices",
    }


def test_search_detail_levels_and_category_filter_are_bounded(manifest_path: Path) -> None:
    catalog = CodeModeCatalog.from_manifest(manifest_path, prefix="unifi")

    detailed = catalog.search("radio", category="devices", limit=1, detail="detailed")
    full = catalog.search("radio", category="devices", limit=1, detail="full")

    assert detailed["tools"][0]["parameters"] == [
        {"name": "mac_address", "type": "string", "required": True, "description": "Access point MAC."},
        {"name": "radio", "type": "string", "required": True},
        {"name": "channel", "type": "integer", "required": True},
        {"name": "confirm", "type": "boolean", "required": False},
    ]
    assert "schema" not in detailed["tools"][0]
    assert full["tools"][0]["schema"]["input"]["properties"]["channel"] == {"type": "integer"}
    assert full["tools"][0]["annotations"] == {"readOnlyHint": False, "openWorldHint": False}


def test_get_schema_returns_exact_full_record_and_reports_unknown_names(manifest_path: Path) -> None:
    catalog = CodeModeCatalog.from_manifest(manifest_path, prefix="unifi")
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))["tools"][0]

    result = catalog.get_schema(["unifi_update_device_radio", "unifi_execute", "unifi_missing"], detail="full")

    assert result["unknown"] == ["unifi_execute", "unifi_missing"]
    assert result["tools"] == [
        {
            "name": expected["name"],
            "title": expected["title"],
            "description": expected["description"],
            "category": "devices",
            "parameters": [
                {"name": "mac_address", "type": "string", "required": True, "description": "Access point MAC."},
                {"name": "radio", "type": "string", "required": True},
                {"name": "channel", "type": "integer", "required": True},
                {"name": "confirm", "type": "boolean", "required": False},
            ],
            "schema": expected["schema"],
            "annotations": expected["annotations"],
        }
    ]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"query": "radio", "category": None, "limit": 1, "detail": "invalid"},
            "detail must be one of: brief, detailed, full.",
        ),
        (
            {"query": "  ", "category": None, "limit": 1, "detail": "brief"},
            "query must be a non-empty string.",
        ),
        (
            {"query": "radio", "category": None, "limit": 0, "detail": "brief"},
            "limit must be an integer between 1 and 20.",
        ),
        (
            {"query": "radio", "category": "missing", "limit": 1, "detail": "brief"},
            "category must be one of: clients, devices.",
        ),
    ],
)
def test_search_validation_errors_are_deterministic(
    manifest_path: Path, kwargs: dict[str, object], expected: str
) -> None:
    catalog = CodeModeCatalog.from_manifest(manifest_path, prefix="unifi")

    result = catalog.search(**kwargs)

    assert result == {
        "tools": [],
        "total_matches": 0,
        "truncated": False,
        "categories": ["clients", "devices"],
        "error": expected,
    }


def test_get_schema_rejects_invalid_detail_without_throwing(manifest_path: Path) -> None:
    catalog = CodeModeCatalog.from_manifest(manifest_path, prefix="unifi")

    assert catalog.get_schema(["unifi_update_device_radio"], detail="brief") == {
        "tools": [],
        "unknown": [],
        "error": "detail must be one of: detailed, full.",
    }


def test_search_rejects_unhashable_invalid_detail_without_throwing(manifest_path: Path) -> None:
    catalog = CodeModeCatalog.from_manifest(manifest_path, prefix="unifi")

    assert catalog.search("radio", category=None, limit=1, detail=[]) == {  # type: ignore[arg-type]
        "tools": [],
        "total_matches": 0,
        "truncated": False,
        "categories": ["clients", "devices"],
        "error": "detail must be one of: brief, detailed, full.",
    }


def test_get_schema_rejects_unhashable_invalid_detail_without_throwing(manifest_path: Path) -> None:
    catalog = CodeModeCatalog.from_manifest(manifest_path, prefix="unifi")

    assert catalog.get_schema(["unifi_update_device_radio"], detail=[]) == {  # type: ignore[arg-type]
        "tools": [],
        "unknown": [],
        "error": "detail must be one of: detailed, full.",
    }
