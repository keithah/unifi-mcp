"""Manifest-backed domain catalog for Code Mode discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from unifi_mcp_shared.meta_tools import is_meta_tool
from unifi_mcp_shared.tool_index import load_tool_manifest, rank_tools_by_search

_SEARCH_DETAILS = frozenset({"brief", "detailed", "full"})
_SCHEMA_DETAILS = frozenset({"detailed", "full"})


@dataclass(frozen=True)
class CodeModeCatalog:
    """A read-only catalog of manifest-backed, non-meta domain tools."""

    tools: tuple[dict[str, Any], ...]
    by_name: dict[str, dict[str, Any]]
    categories: tuple[str, ...]

    @classmethod
    def from_manifest(cls, manifest_path: Path, *, prefix: str) -> CodeModeCatalog:
        manifest = load_tool_manifest(manifest_path) or {}
        module_map = manifest.get("module_map")
        tool_records = manifest.get("tools")
        if not isinstance(module_map, dict) or not isinstance(tool_records, list):
            return cls(tools=(), by_name={}, categories=())

        code_mode_names = frozenset({f"{prefix}_code_search", f"{prefix}_code_get_schema", f"{prefix}_code_execute"})
        tools: list[dict[str, Any]] = []
        for record in tool_records:
            if not isinstance(record, dict):
                continue
            name = record.get("name")
            if (
                not isinstance(name, str)
                or not name.startswith(prefix)
                or name in code_mode_names
                or is_meta_tool(name)
            ):
                continue
            module = module_map.get(name)
            category = module.rsplit(".", 1)[-1] if isinstance(module, str) and module else ""
            if not category:
                continue
            tools.append({**record, "_category": category})

        categories = tuple(sorted({tool["_category"] for tool in tools}))
        return cls(tools=tuple(tools), by_name={tool["name"]: tool for tool in tools}, categories=categories)

    @property
    def names(self) -> frozenset[str]:
        """Return exact names permitted for Code Mode domain dispatch."""
        return frozenset(self.by_name)

    def search(
        self,
        query: str,
        *,
        category: str | None,
        limit: int,
        detail: Literal["brief", "detailed", "full"],
    ) -> dict[str, Any]:
        """Search the catalog with deterministic, bounded result shaping."""
        error = self._search_error(query=query, category=category, limit=limit, detail=detail)
        if error is not None:
            return self._search_result([], total_matches=0, truncated=False, error=error)

        matching_tools = list(self.tools)
        if category is not None:
            matching_tools = [tool for tool in matching_tools if tool["_category"].lower() == category.lower()]
        ranked = rank_tools_by_search(matching_tools, query, limit=None)
        total_matches = len(ranked)
        visible = ranked[:limit]
        return self._search_result(
            [self._shape(tool, detail=detail) for tool in visible],
            total_matches=total_matches,
            truncated=total_matches > len(visible),
        )

    def get_schema(self, names: list[str], *, detail: Literal["detailed", "full"]) -> dict[str, Any]:
        """Return exact-name schema details and explicitly identify unknown names."""
        if not isinstance(detail, str) or detail not in _SCHEMA_DETAILS:
            return {"tools": [], "unknown": [], "error": "detail must be one of: detailed, full."}
        tools: list[dict[str, Any]] = []
        unknown: list[str] = []
        for name in names:
            tool = self.by_name.get(name)
            if tool is None:
                unknown.append(name)
            else:
                tools.append(self._shape(tool, detail=detail))
        return {"tools": tools, "unknown": unknown}

    def _search_error(self, *, query: object, category: object, limit: object, detail: object) -> str | None:
        if not isinstance(detail, str) or detail not in _SEARCH_DETAILS:
            return "detail must be one of: brief, detailed, full."
        if not isinstance(query, str) or not query.strip():
            return "query must be a non-empty string."
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            return "limit must be an integer between 1 and 20."
        if category is not None:
            if not isinstance(category, str) or category.lower() not in {value.lower() for value in self.categories}:
                return f"category must be one of: {', '.join(self.categories)}."
        return None

    def _search_result(
        self, tools: list[dict[str, Any]], *, total_matches: int, truncated: bool, error: str | None = None
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "tools": tools,
            "total_matches": total_matches,
            "truncated": truncated,
            "categories": list(self.categories),
        }
        if error is not None:
            result["error"] = error
        return result

    @staticmethod
    def _shape(tool: dict[str, Any], *, detail: str) -> dict[str, Any]:
        result = {
            "name": tool["name"],
            "title": tool.get("title", ""),
            "description": tool.get("description", ""),
            "category": tool["_category"],
        }
        if detail in _SCHEMA_DETAILS:
            result["parameters"] = _parameter_summaries(tool)
        if detail == "full":
            result["schema"] = tool.get("schema", {})
            result["annotations"] = tool.get("annotations", {})
        return result


def _parameter_summaries(tool: dict[str, Any]) -> list[dict[str, Any]]:
    schema = tool.get("schema")
    input_schema = schema.get("input") if isinstance(schema, dict) else None
    properties = input_schema.get("properties") if isinstance(input_schema, dict) else None
    required = input_schema.get("required", []) if isinstance(input_schema, dict) else []
    if not isinstance(properties, dict):
        return []
    required_names = set(required) if isinstance(required, list) else set()
    parameters: list[dict[str, Any]] = []
    for name, definition in properties.items():
        if not isinstance(name, str):
            continue
        definition = definition if isinstance(definition, dict) else {}
        parameter: dict[str, Any] = {
            "name": name,
            "type": definition.get("type", "object"),
            "required": name in required_names,
        }
        if isinstance(definition.get("description"), str):
            parameter["description"] = definition["description"]
        parameters.append(parameter)
    return parameters
