from typing import TypedDict, List, Optional
from datetime import datetime


class Note(TypedDict):
    id: int
    title: str
    body: str
    tags: List[str]
    created_at: datetime
    updated_at: datetime

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
    date: Optional[datetime] = None
    date_end: Optional[datetime] = None
    limit: Optional[int] = 10

class UpdateNoteInput(TypedDict):
    note_to_be_modified: str
    query: str
    tags: Optional[List[str]] = None

class DeleteNoteInput(TypedDict):
    id: int

