from typing import TypedDict, List, Optional
from datetime import datetime


class Note(TypedDict):
    id: str
    title: str
    body: str
    tags: List[str]
    created_at: Optional[str]
    updated_at: Optional[str]

class AddNoteInput(TypedDict):
    title: str
    body: str
    tags: Optional[List[str]] = None

class GetNoteInput(TypedDict):
    query: str
    tags: Optional[List[str]] = None

class SearchNoteInput(TypedDict):
    query: str
    tags: Optional[List[str]] = None
    date: Optional[str] = None
    date_end: Optional[str] = None
    limit: Optional[int] = 10

class UpdateNoteInput(TypedDict):
    note_to_be_modified: str
    query: str
    tags: Optional[List[str]] = None

class DeleteNoteInput(TypedDict):
    id: int

