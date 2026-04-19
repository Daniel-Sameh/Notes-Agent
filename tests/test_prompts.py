import unittest
from unittest.mock import patch, MagicMock
from langchain_core.messages import SystemMessage

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.prompts import get_system_prompt

class TestSystemPrompt(unittest.TestCase):

    def test_get_system_prompt_without_active_ids(self):
        # Arrange & Act
        result = get_system_prompt(active_note_ids=None)
        
        # Assert
        self.assertIsInstance(result, SystemMessage)
        self.assertIn("No notes are currently in focus", result.content)
        self.assertNotIn("ACTIVE NOTE CONTEXT:\nThese Note IDs are in focus", result.content)

    def test_get_system_prompt_with_active_ids(self):
        # Arrange
        active_ids = ["test-uuid-1234", "test-uuid-5678"]
        
        # Act
        result = get_system_prompt(active_note_ids=active_ids)
        
        # Assert
        self.assertIsInstance(result, SystemMessage)
        self.assertIn("These Note IDs are in focus from recent tool results", result.content)
        self.assertIn("['test-uuid-1234', 'test-uuid-5678']", result.content)

    def test_prompt_contains_crucial_instructions(self):
        # Arrange & Act
        result = get_system_prompt()
        
        # Assert
        self.assertIn("SECTION 1 - WHEN TO USE TOOLS", result.content)
        self.assertIn("SECTION 2 - RECOGNISING IMPLICIT NOTE INTENT", result.content)
        self.assertIn("SECTION 3 - CORE NOTE-TAKING RULES", result.content)
        self.assertIn("SECTION 4 - DISAMBIGUATION & CONFIRMATION", result.content)
        self.assertIn("SECTION 5 - GRACEFUL ERROR & SCOPE HANDLING", result.content)

if __name__ == '__main__':
    unittest.main()
