from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from typing import Generator, Optional, List
from .models import Base, Note
from .utils import utc_now
from ..config import settings

DATABASE_URL = settings.db_url
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Notes CRUD Operations
def create_note(title: str, body: str, tags: Optional[List[str]]) -> Note:
    with SessionLocal() as db:
        note = Note(title=title, body=body, tags=",".join(tags) if tags else "")
        db.add(note)
        db.commit()
        db.refresh(note)
        return note

def get_note_by_id(id: str) -> Optional[Note]:
    with SessionLocal() as db:
        return db.query(Note).filter(Note.id == id).first()

def search_notes(query: str = "", tags: Optional[List[str]] = None, date: Optional[datetime] = None, date_end: Optional[datetime] = None, limit: int = 10) -> List[Note]:
    with SessionLocal() as db:
        notes = set()
        if query:
            notes.update(db.query(Note).filter(Note.body.contains(query)).all())
        if tags:
            notes.update(db.query(Note).filter(Note.tags.contains(",".join(tags))).all())
        if date and date_end:
            notes.update(db.query(Note).filter(Note.created_at >= date, Note.created_at < date_end).all())
        elif date:
            notes.update(db.query(Note).filter(Note.created_at >= date).all())
        elif date_end:
            notes.update(db.query(Note).filter(Note.created_at < date_end).all())
        return list(notes)[:limit]

def update_note(id: str, title: Optional[str], body: Optional[str], tags: Optional[List[str]]) -> Optional[Note]:
    with SessionLocal() as db:
        note = db.query(Note).filter(Note.id == id).first()
        if note:
            note.title = title if title else note.title
            note.body = body if body else note.body
            note.tags = ", ".join(tags) if tags else ""
            note.updated_at = utc_now()
            db.commit()
            db.refresh(note)
        return note

def delete_note(id: str) -> bool:
    with SessionLocal() as db:
        note = db.query(Note).filter(Note.id == id).first()
        if note:
            db.delete(note)
            db.commit()
            return True
        return False
