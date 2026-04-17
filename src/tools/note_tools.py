from langchain_core.tools import tool
from datetime import datetime
from typing import List, Optional
from .schema import Note as NoteSchema
from ..database.models import Note as NoteModel
from ..database.relational_db import (
    create_note, 
    search_notes as db_search_notes, 
    get_note_by_id as db_get_note_by_id,
    update_note as db_update_note,
    delete_note as db_delete_note
)


def _note_to_dict(note: NoteModel) -> NoteSchema:
    return {
        "id": note.id,
        "title": note.title,
        "body": note.body,
        "tags": note.tags.split(",") if note.tags else [],
        "created_at": note.created_at,
        "updated_at": note.updated_at,
    }

@tool
def add_note(title: str, body: str, tags: Optional[List[str]] = None) -> NoteSchema:
    """Create a new note and return it."""
    note = create_note(title=title, body=body, tags=tags)
    return _note_to_dict(note)

@tool
def get_note(query: str, tags: Optional[List[str]] = None) -> List[NoteSchema]:
    """Get notes that match a text query and optional tags."""
    notes = db_search_notes(query=query, tags=tags)
    return [_note_to_dict(note) for note in notes]

@tool
def get_note_by_id(id: int) -> NoteSchema:
    """Get a note by its ID."""
    note = db_get_note_by_id(id=id)
    if not note:
        raise ValueError(f"Note with id '{id}' not found.")
    return _note_to_dict(note)

@tool
def search_notes(
    query: str = "",
    tags: Optional[List[str]] = None,
    date: Optional[datetime] = None,
    date_end: Optional[datetime] = None,
    limit: int = 10,
) -> List[NoteSchema]:
    """Search notes by query, tags, and optional datetime range."""
    notes = db_search_notes(query=query, tags=tags, date=date, date_end=date_end, limit=limit)
    return [_note_to_dict(note) for note in notes]

@tool
def update_note(
    id: int,
    title: Optional[str] = None,
    body: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> NoteSchema:
    """Update an existing note's title, body, or tags by its ID."""
    updated_note = db_update_note(id=str(id), title=title, body=body, tags=tags)
    if not updated_note:
        raise ValueError(f"Note with ID '{id}' not found.")
    return _note_to_dict(updated_note)

@tool
def delete_note(id: int) -> bool:
    """Delete a note by its ID."""
    success = db_delete_note(id=str(id))
    if not success:
         raise ValueError(f"Note with ID '{id}' not found.")
    return success

tools = [add_note, get_note, get_note_by_id, search_notes, update_note, delete_note]