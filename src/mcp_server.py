from __future__ import annotations

from datetime import datetime
from importlib import import_module
from typing import Any, Optional

from .database.relational_db import init_db
from .tools.note_tools import (
    add_note,
    delete_note,
    get_note_by_id,
    search_notes,
    semantic_search,
    update_note,
)

FastMCP = import_module("mcp.server.fastmcp").FastMCP

server = FastMCP(
    name="notes-agent",
    instructions=(
        "MCP server for Notes-Agent. All tool calls must include user_id to keep notes isolated per user."
    ),
)


def _parse_optional_datetime(raw_value: Optional[str], field_name: str) -> Optional[datetime]:
    if raw_value is None:
        return None

    value = raw_value.strip()
    if not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"'{field_name}' must be an ISO-8601 datetime string.") from exc


def _invoke_tool(tool_obj: Any, payload: dict[str, Any]) -> Any:
    try:
        return tool_obj.invoke(payload)
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError("Unexpected error while executing the note operation.") from exc


@server.tool(name="add_note", description="Create a new note.")
def mcp_add_note(user_id: str, title: str, body: str, tags: Optional[list[str]] = None) -> dict:
    return _invoke_tool(
        add_note,
        {
            "user_id": user_id,
            "title": title,
            "body": body,
            "tags": tags,
        },
    )


@server.tool(name="get_note_by_id", description="Get a specific single note by its ID.")
def mcp_get_note_by_id(user_id: str, id: str) -> dict:
    return _invoke_tool(get_note_by_id, {"user_id": user_id, "id": id})


@server.tool(
    name="search_notes",
    description="List or search notes by keyword, tags, and optional date range.",
)
def mcp_search_notes(
    user_id: str,
    query: str = "",
    tags: Optional[list[str]] = None,
    date: Optional[str] = None,
    date_end: Optional[str] = None,
    limit: int = 10,
) -> list[dict]:
    if limit <= 0:
        raise ValueError("'limit' must be a positive integer.")

    parsed_date = _parse_optional_datetime(date, "date")
    parsed_date_end = _parse_optional_datetime(date_end, "date_end")

    return _invoke_tool(
        search_notes,
        {
            "user_id": user_id,
            "query": query,
            "tags": tags,
            "date": parsed_date,
            "date_end": parsed_date_end,
            "limit": limit,
        },
    )


@server.tool(name="semantic_search", description="Search notes based on semantic meaning, concepts, or natural language questions.")
def mcp_semantic_search(user_id: str, query: str, limit: int = 5) -> list[dict]:
    if limit <= 0:
        raise ValueError("'limit' must be a positive integer.")

    return _invoke_tool(
        semantic_search,
        {
            "user_id": user_id,
            "query": query,
            "limit": limit,
        },
    )


@server.tool(name="update_note", description="Update an existing note by ID.")
def mcp_update_note(
    user_id: str,
    id: str,
    title: Optional[str] = None,
    body: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> dict:
    return _invoke_tool(
        update_note,
        {
            "user_id": user_id,
            "id": id,
            "title": title,
            "body": body,
            "tags": tags,
        },
    )


@server.tool(name="delete_note", description="Delete a note by ID.")
def mcp_delete_note(user_id: str, id: str) -> bool:
    return _invoke_tool(delete_note, {"user_id": user_id, "id": id})


def main() -> None:
    init_db()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
