import os
import sys
import unittest
from datetime import datetime
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mcp_server import (
    mcp_add_note,
    mcp_search_notes,
    mcp_delete_note,
    _invoke_tool,
    add_note,
    server,
)


class TestMCPServer(unittest.TestCase):
    def test_server_registers_expected_tools(self):
        registered = set(server._tool_manager._tools.keys())
        expected = {
            "add_note",
            "get_note_by_id",
            "search_notes",
            "semantic_search",
            "update_note",
            "delete_note",
        }
        self.assertEqual(registered, expected)

    @patch("src.mcp_server._invoke_tool")
    def test_add_note_routes_user_id_and_payload(self, mock_invoke_tool):
        mock_invoke_tool.return_value = {"id": "n1", "title": "T"}

        result = mcp_add_note(
            user_id="user-1",
            title="My title",
            body="My body",
            tags=["work", "meeting"],
        )

        mock_invoke_tool.assert_called_once_with(
            add_note,
            {
                "user_id": "user-1",
                "title": "My title",
                "body": "My body",
                "tags": ["work", "meeting"],
            },
        )
        self.assertEqual(result["id"], "n1")

    @patch("src.mcp_server._invoke_tool")
    def test_search_notes_parses_date_range(self, mock_invoke_tool):
        mock_invoke_tool.return_value = []

        mcp_search_notes(
            user_id="user-2",
            query="api",
            tags=["urgent"],
            date="2026-04-01T00:00:00",
            date_end="2026-04-30T23:59:59",
            limit=3,
        )

        tool_obj, payload = mock_invoke_tool.call_args[0]

        self.assertEqual(tool_obj.name, "search_notes")
        self.assertEqual(payload["user_id"], "user-2")
        self.assertEqual(payload["query"], "api")
        self.assertEqual(payload["tags"], ["urgent"])
        self.assertIsInstance(payload["date"], datetime)
        self.assertIsInstance(payload["date_end"], datetime)
        self.assertEqual(payload["limit"], 3)

    def test_search_notes_rejects_non_positive_limit(self):
        with self.assertRaises(ValueError) as exc:
            mcp_search_notes(user_id="user-3", limit=0)

        self.assertIn("positive integer", str(exc.exception))

    def test_unexpected_errors_are_mapped(self):
        class FakeTool:
            def invoke(self, payload):
                raise RuntimeError("db timeout")

        with self.assertRaises(RuntimeError) as exc:
            _invoke_tool(FakeTool(), {"id": "note-1"})

        self.assertIn("Unexpected error while executing the note operation", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
