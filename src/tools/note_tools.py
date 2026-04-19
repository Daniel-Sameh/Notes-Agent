from langchain_core.tools import tool
from datetime import datetime
from typing import List, Optional, Annotated
from langgraph.prebuilt import InjectedState
from pydantic import Field
from ..database.models import Note as NoteModel
from ..database.relational_db import (
    create_note,
    search_notes as db_search_notes,
    get_note_by_id as db_get_note_by_id,
    update_note as db_update_note,
    delete_note as db_delete_note,
)
from ..database.vector_db import (
    add_to_vector_db,
    update_in_vector_db,
    delete_from_vector_db,
    semantic_search_vector_db,
)

def _note_to_dict(note: NoteModel) -> dict:
    return {
        "id": note.id,
        "user_id": note.user_id,
        "title": note.title,
        "body": note.body,
        "tags": note.tags.split(",") if note.tags else [],
        "created_at": note.created_at.isoformat() if note.created_at else None,
        "updated_at": note.updated_at.isoformat() if note.updated_at else None,
    }

@tool
def add_note(
    user_id: Annotated[str, InjectedState("user_id")],
    title: Annotated[str, Field(description="The title of the note.")],
    body: Annotated[str, Field(description="The main content of the note.")],
    tags: Annotated[
        Optional[List[str]],
        Field(
            description="A list of tags for the note.",
            json_schema_extra={"type": "array", "items": {"type": "string"}},
        ),
    ] = None,
) -> dict:
    """Create a new note and return it."""
    note = create_note(user_id=user_id, title=title, body=body, tags=tags)
    tags_str = ",".join(tags) if tags else ""
    add_to_vector_db(user_id=user_id, id=note.id, title=note.title, body=note.body, tags=tags_str)
    return _note_to_dict(note)

@tool
def get_note_by_id(
    user_id: Annotated[str, InjectedState("user_id")],
    id: Annotated[str, Field(description="The exact database ID of the note to retrieve.")],
) -> dict:
    """Get a specific single note by its ID."""
    note = db_get_note_by_id(user_id=user_id, id=id)
    if not note:
        raise ValueError(f"Note with id '{id}' not found.")
    return _note_to_dict(note)

@tool
def search_notes(
    user_id: Annotated[str, InjectedState("user_id")],
    query: Annotated[str, Field(description="An exact phrase or keyword to search for in note titles or bodies.")] = "",
    tags: Annotated[
        Optional[List[str]],
        Field(description="A list of specific tags to filter by. MUST be an array of strings."),
    ] = None,
    date: Annotated[Optional[datetime], Field(description="Start date to filter notes created on or after this time.")] = None,
    date_end: Annotated[Optional[datetime], Field(description="End date to filter notes created on or before this time.")] = None,
    limit: Annotated[int, Field(description="Maximum number of notes to return.")] = 10,
) -> List[dict]:
    """Search notes by exact text query, tags, and date range. Use 'semantic_search' for natural language concepts."""
    notes = db_search_notes(user_id=user_id, query=query, tags=tags, date=date, date_end=date_end, limit=limit)
    return [_note_to_dict(note) for note in notes]

@tool
def semantic_search(
    user_id: Annotated[str, InjectedState("user_id")],
    query: Annotated[str, Field(description="A natural language sentence or concept to search for in your notes.")],
    limit: Annotated[int, Field(description="Maximum number of semantic results to return.")] = 5,
) -> List[dict]:
    """Search notes based on semantic meaning, concepts, or natural language questions. Best for broad searches."""
    matching_ids = semantic_search_vector_db(user_id=user_id, query=query, limit=limit)
    if not matching_ids:
        return []
    results = []
    for note_id in matching_ids:
        note = db_get_note_by_id(user_id=user_id, id=note_id)
        if note:
            results.append(_note_to_dict(note))
    return results


@tool
def update_note(
    user_id: Annotated[str, InjectedState("user_id")],
    id: Annotated[str, Field(description="The exact database ID of the note to update.")],
    title: Annotated[Optional[str], Field(description="The new title for the note.")] = None,
    body: Annotated[Optional[str], Field(description="The new content body for the note.")] = None,
    tags: Annotated[
        Optional[List[str]],
        Field(
            description="A new list of tags.",
            json_schema_extra={"type": "array", "items": {"type": "string"}},
        ),
    ] = None,
) -> dict:
    """Update an existing note's title, body, or tags. You MUST provide the correct note ID first."""
    updated_note = db_update_note(user_id=user_id, id=id, title=title, body=body, tags=tags)
    if not updated_note:
        raise ValueError(f"Note with ID '{id}' not found.")
    tags_str = ",".join(updated_note.tags.split(",")) if updated_note.tags else ""
    update_in_vector_db(user_id=user_id, id=id, title=updated_note.title, body=updated_note.body, tags=tags_str)
    return _note_to_dict(updated_note)


@tool
def delete_note(
    user_id: Annotated[str, InjectedState("user_id")],
    id: Annotated[str, Field(description="The exact database ID of the note to delete.")],
) -> bool:
    """Delete a note completely using its exact database ID."""
    success = db_delete_note(user_id=user_id, id=id)
    if not success:
        raise ValueError(f"Note with ID '{id}' not found.")
    delete_from_vector_db(id=id)
    return success


tools = [add_note, get_note_by_id, search_notes, semantic_search, update_note, delete_note]