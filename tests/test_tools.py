import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from unittest.mock import patch, MagicMock

os.environ["TEST_ENV"] = "true"

from src.tools.note_tools import (
    add_note,
    get_note_by_id,
    search_notes,
    semantic_search,
    update_note,
    delete_note,
    _note_to_dict
)

# Mock model class for returns
class MockNote:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not hasattr(self, 'tags'):
            self.tags = None
        if not hasattr(self, 'created_at'):
            self.created_at = None
        if not hasattr(self, 'updated_at'):
            self.updated_at = None

class TestNoteTools(unittest.TestCase):

    @patch('src.tools.note_tools.add_to_vector_db')
    @patch('src.tools.note_tools.create_note')
    def test_add_note(self, mock_create_note, mock_add_to_vector_db):
        # Arrange
        mock_created = MockNote(id="123", user_id="u1", title="Test Title", body="Test Body", tags="tag1,tag2")
        mock_create_note.return_value = mock_created
        
        # Act
        # Because it's an langchain tool, we invoke via .invoke
        result = add_note.invoke({"user_id": "u1", "title": "Test Title", "body": "Test Body", "tags": ["tag1", "tag2"]})
        
        # Assert
        mock_create_note.assert_called_once_with(user_id="u1", title="Test Title", body="Test Body", tags=["tag1", "tag2"])
        mock_add_to_vector_db.assert_called_once_with(user_id="u1", id="123", title="Test Title", body="Test Body", tags="tag1,tag2")
        self.assertEqual(result["id"], "123")
        self.assertEqual(result["title"], "Test Title")

    @patch('src.tools.note_tools.db_get_note_by_id')
    def test_get_note_by_id_success(self, mock_get_note_by_id):
        # Arrange
        mock_note = MockNote(id="123", user_id="u1", title="Test Title", body="Test Body")
        mock_get_note_by_id.return_value = mock_note
        
        # Act
        result = get_note_by_id.invoke({"user_id": "u1", "id": "123"})
        
        # Assert
        mock_get_note_by_id.assert_called_once_with(user_id="u1", id="123")
        self.assertEqual(result["id"], "123")

    @patch('src.tools.note_tools.db_get_note_by_id')
    def test_get_note_by_id_not_found(self, mock_get_note_by_id):
        # Arrange
        mock_get_note_by_id.return_value = None
        
        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_note_by_id.invoke({"user_id": "u1", "id": "missing-id"})
        self.assertIn("not found", str(context.exception))

    @patch('src.tools.note_tools.delete_from_vector_db')
    @patch('src.tools.note_tools.db_delete_note')
    def test_delete_note_success(self, mock_delete_note, mock_delete_from_vector_db):
        # Arrange
        mock_delete_note.return_value = True
        
        # Act
        result = delete_note.invoke({"user_id": "u1", "id": "123"})
        
        # Assert
        mock_delete_note.assert_called_once_with(user_id="u1", id="123")
        mock_delete_from_vector_db.assert_called_once_with(id="123")
        self.assertTrue(result)

    def test__note_to_dict(self):
        # Arrange
        import datetime
        dt = datetime.datetime.now()
        mock_note = MockNote(id="1", user_id="u1", title="T", body="B", tags="a,b", created_at=dt, updated_at=dt)
        
        # Act
        result = _note_to_dict(mock_note)
        
        # Assert
        self.assertEqual(result["id"], "1")
        self.assertEqual(result["tags"], ["a", "b"])
        self.assertEqual(result["created_at"], dt.isoformat())

    @patch('src.tools.note_tools.db_search_notes')
    def test_search_notes(self, mock_search_notes):
        # Arrange
        mock_search_notes.return_value = [MockNote(id="1", user_id="u1", title="Test", body="Body")]
        
        # Act
        result = search_notes.invoke({"user_id": "u1", "query": "Test"})
        
        # Assert
        mock_search_notes.assert_called_once_with(user_id="u1", query="Test", tags=None, date=None, date_end=None, limit=10)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "1")

    @patch('src.tools.note_tools.db_get_note_by_id')
    @patch('src.tools.note_tools.semantic_search_vector_db')
    def test_semantic_search(self, mock_semantic_search_vector_db, mock_get_note_by_id):
        # Arrange
        mock_semantic_search_vector_db.return_value = ["1"]
        mock_get_note_by_id.return_value = MockNote(id="1", user_id="u1", title="Test", body="Body")
        
        # Act
        result = semantic_search.invoke({"user_id": "u1", "query": "concept"})
        
        # Assert
        mock_semantic_search_vector_db.assert_called_once_with(user_id="u1", query="concept", limit=5)
        mock_get_note_by_id.assert_called_once_with(user_id="u1", id="1")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "Test")

    @patch('src.tools.note_tools.update_in_vector_db')
    @patch('src.tools.note_tools.db_update_note')
    def test_update_note_success(self, mock_db_update_note, mock_update_in_vector_db):
        # Arrange
        mock_updated = MockNote(id="1", user_id="u1", title="New", body="B", tags="a")
        mock_db_update_note.return_value = mock_updated
        
        # Act
        result = update_note.invoke({"user_id": "u1", "id": "1", "title": "New"})
        
        # Assert
        mock_db_update_note.assert_called_once_with(user_id="u1", id="1", title="New", body=None, tags=None)
        mock_update_in_vector_db.assert_called_once_with(user_id="u1", id="1", title="New", body="B", tags="a")
        self.assertEqual(result["title"], "New")

    @patch('src.tools.note_tools.db_update_note')
    def test_update_note_not_found(self, mock_db_update_note):
        # Arrange
        mock_db_update_note.return_value = None
        
        # Act & Assert
        with self.assertRaises(ValueError) as context:
            update_note.invoke({"user_id": "u1", "id": "missing", "title": "New"})
        self.assertIn("not found", str(context.exception))

if __name__ == '__main__':
    unittest.main()