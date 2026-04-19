import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from unittest.mock import patch, MagicMock

os.environ["TEST_ENV"] = "true"

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.nodes import (
    should_excute_tool,
    should_execute_tool_after_guard,
    guard_tool_call_node,
    call_llm_node,
    _retrieve_or_compact_history
)
from src.agent.state import AgentState
from langchain_core.messages import SystemMessage, ToolMessage


class TestAgentNodes(unittest.TestCase):
    
    # ---------------------------------------------------------
    # Tests for should_excute_tool
    # ---------------------------------------------------------
    def test_should_execute_tool_with_tool_calls(self):
        # Arrange
        message = AIMessage(content="", tool_calls=[{"name": "search_notes", "args": {}, "id": "1"}])
        state = {"messages": [message]}
        
        # Act
        result = should_excute_tool(state)
        
        # Assert
        self.assertEqual(result, "tools")

    def test_should_execute_tool_without_tool_calls(self):
        # Arrange
        message = AIMessage(content="Hello there")
        state = {"messages": [message]}
        
        # Act
        result = should_excute_tool(state)
        
        # Assert
        self.assertEqual(result, "end")

    # ---------------------------------------------------------
    # Tests for should_execute_tool_after_guard
    # ---------------------------------------------------------
    def test_should_execute_tool_after_guard_with_error(self):
        # Arrange
        state = {"error": "Missing ID"}
        
        # Act
        result = should_execute_tool_after_guard(state)
        
        # Assert
        self.assertEqual(result, "retry_llm")

    def test_should_execute_tool_after_guard_without_error(self):
        # Arrange
        state = {"error": ""}
        
        # Act
        result = should_execute_tool_after_guard(state)
        
        # Assert
        self.assertEqual(result, "execute_tool")

    # ---------------------------------------------------------
    # Tests for guard_tool_call_node
    # ---------------------------------------------------------
    def test_guard_node_no_tool_calls(self):
        # Arrange
        state = {
            "messages": [AIMessage(content="Hello")],
            "active_note_ids": [],
            "pending_confirmation": None,
            "error": ""
        }
        
        # Act
        result = guard_tool_call_node(state)
        
        # Assert
        self.assertEqual(result.get("error", ""), "")

    def test_guard_node_missing_id_for_destructive_tool(self):
        # Arrange
        tool_message = AIMessage(content="", tool_calls=[{"name": "delete_note", "args": {}, "id": "1"}])
        state = {
            "messages": [tool_message],
            "active_note_ids": ["test-id"],
            "pending_confirmation": None
        }
        
        # Act
        result = guard_tool_call_node(state)
        
        # Assert
        self.assertIn("requires an 'id'", result["error"])

    def test_guard_node_unrecognized_id(self):
        # Arrange
        tool_message = AIMessage(content="", tool_calls=[{"name": "delete_note", "args": {"id": "fake-id"}, "id": "1"}])
        state = {
            "messages": [tool_message],
            "active_note_ids": ["real-id"],
            "pending_confirmation": None
        }
        
        # Act
        result = guard_tool_call_node(state)
        
        # Assert
        self.assertIn("not in your context memory", result["error"])

    @patch('src.agent.nodes.llm')
    def test_guard_node_sets_pending_confirmation_for_destructive_tool(self, mock_llm):
        # Arrange
        mock_model = MagicMock()
        mock_llm.client = mock_model
        mock_model.invoke.return_value = AIMessage(content="NO")

        tool_message = AIMessage(content="", tool_calls=[{"name": "delete_note", "args": {"id": "real-id"}, "id": "1"}])
        state = {
            "messages": [tool_message],
            "active_note_ids": ["real-id"],
            "pending_confirmation": None
        }
        
        # Act
        result = guard_tool_call_node(state)
        
        # Assert
        self.assertIn("MUST ask the user", result["error"])
        self.assertIsNotNone(result["pending_confirmation"])
        self.assertEqual(result["pending_confirmation"]["tool"], "delete_note")

    @patch('src.agent.nodes.llm')
    def test_guard_node_allows_destructive_tool_if_confirmed(self, mock_llm):
        # Arrange
        mock_model = MagicMock()
        mock_llm.client = mock_model
        mock_model.invoke.return_value = AIMessage(content="YES")

        tool_message = AIMessage(content="", tool_calls=[{"name": "delete_note", "args": {"id": "real-id"}, "id": "1"}])
        state = {
            "messages": [AIMessage(content="Delete it"), tool_message],
            "active_note_ids": ["real-id"],
            "pending_confirmation": None
        }
        
        # Act
        result = guard_tool_call_node(state)
        
        # Assert
        self.assertEqual(result.get("error", ""), "")

    # ---------------------------------------------------------
    # Tests for _retrieve_or_compact_history
    # ---------------------------------------------------------
    def test_retrieve_history_no_compaction(self):
        # Arrange
        messages = [HumanMessage(content="msg1"), AIMessage(content="msg2")]
        mock_model = MagicMock()
        
        # Act
        result = _retrieve_or_compact_history(messages, ["id1"], mock_model)
        
        # Assert
        self.assertEqual(len(result), 3) # 1 System message + 2 original messages
        self.assertIsInstance(result[0], SystemMessage)
        self.assertEqual(result[1].content, "msg1")
        self.assertEqual(result[2].content, "msg2")
        mock_model.invoke.assert_not_called()

    def test_retrieve_history_with_compaction(self):
        # Arrange
        from src.agent.nodes import MAX_HISTORY_MESSAGES
        messages = [HumanMessage(content=f"msg{i}") for i in range(MAX_HISTORY_MESSAGES)]
        mock_model = MagicMock()
        compacted_message = AIMessage(content="Compacted summary")
        mock_model.invoke.return_value = compacted_message
        
        # Act
        result = _retrieve_or_compact_history(messages, ["id1"], mock_model)
        
        # Assert
        self.assertEqual(len(result), 2) # 1 System message + 1 compacted message
        self.assertIsInstance(result[0], SystemMessage)
        self.assertEqual(result[1].content, compacted_message.content)
        mock_model.invoke.assert_called_once()

    # ---------------------------------------------------------
    # Tests for call_llm_node
    # ---------------------------------------------------------
    @patch('src.agent.nodes.llm')
    def test_call_llm_node_basic(self, mock_llm):
        # Arrange
        mock_model = MagicMock()
        mock_llm.client.bind_tools.return_value = mock_model
        
        response_msg = AIMessage(content="Hello from LLM")
        mock_model.invoke.return_value = response_msg
        
        state = {
            "messages": [HumanMessage(content="Hi")],
            "active_note_ids": ["note-1"],
            "pending_confirmation": None,
            "error": ""
        }
        
        # Act
        result = call_llm_node(state)
        
        # Assert
        self.assertEqual(result["messages"], [response_msg])
        self.assertEqual(result["error"], "")
        self.assertEqual(result["active_note_ids"], ["note-1"])
        mock_model.invoke.assert_called_once()

    @patch('src.agent.nodes.llm')
    def test_call_llm_node_extracts_ids(self, mock_llm):
        # Arrange
        mock_model = MagicMock()
        mock_llm.client.bind_tools.return_value = mock_model
        mock_model.invoke.return_value = AIMessage(content="Done")

        # Simulate a tool message that returns a JSON list of notes
        tool_content = '[{"id": "note-2", "title": "New note"}]'
        tool_msg = ToolMessage(content=tool_content, tool_call_id="call1", name="search_notes")
        
        state = {
            "messages": [HumanMessage(content="Search notes"), AIMessage(content="", tool_calls=[{"name": "search_notes", "args": {}, "id": "call1"}]), tool_msg],
            "active_note_ids": ["note-1"],
            "pending_confirmation": None,
            "error": ""
        }
        
        # Act
        result = call_llm_node(state)
        
        # Assert
        self.assertIn("note-2", result["active_note_ids"])
        self.assertIn("note-1", result["active_note_ids"])

    @patch('src.agent.nodes.llm')
    def test_call_llm_node_with_error(self, mock_llm):
        # Arrange
        mock_model = MagicMock()
        mock_llm.client.bind_tools.return_value = mock_model
        mock_model.invoke.return_value = AIMessage(content="Fixed the error")
        
        state = {
            "messages": [HumanMessage(content="Hi")],
            "active_note_ids": [],
            "pending_confirmation": None,
            "error": "You must confirm first"
        }
        
        # Act
        call_llm_node(state)
        
        # Assert
        passed_messages = mock_model.invoke.call_args[0][0]
        # Check if the error message is appended
        self.assertTrue(any("Previous error: You must confirm first" in m.content for m in passed_messages))

if __name__ == '__main__':
    unittest.main()

