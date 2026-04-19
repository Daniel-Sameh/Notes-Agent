from sqlalchemy import create_engine, or_
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from typing import Generator, Optional, List
from .models import Base, Note, User
from .utils import utc_now
from ..config import settings

DATABASE_URL = settings.database_url
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# FIX (ISSUE 12): expire_on_commit=False prevents DetachedInstanceError on returned
# ORM objects after the session closes. Without this, accessing any attribute on a
# returned Note outside the session block raises DetachedInstanceError.
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Notes CRUD Operations

def create_note(user_id: str, title: str, body: str, tags: Optional[List[str]]) -> Note:
    with SessionLocal() as db:
        note = Note(user_id=user_id, title=title, body=body, tags=",".join(tags) if tags else "")
        db.add(note)
        db.commit()
        db.refresh(note)
        return note


def get_note_by_id(user_id: str, id: str) -> Optional[Note]:
    with SessionLocal() as db:
        return db.query(Note).filter(Note.id == id, Note.user_id == user_id).first()


def search_notes(
    user_id: str,
    query: str = "",
    tags: Optional[List[str]] = None,
    date: Optional[datetime] = None,
    date_end: Optional[datetime] = None,
    limit: int = 10,
) -> List[Note]:
    with SessionLocal() as db:
        base_query = db.query(Note).filter(Note.user_id == user_id)
        notes = set()

        if query:
            notes.update(base_query.filter(Note.body.contains(query)).all())

        if tags:
            tag_filters = [Note.tags.contains(tag) for tag in tags]
            notes.update(base_query.filter(or_(*tag_filters)).all())

        if date and date_end:
            notes.update(base_query.filter(Note.created_at >= date, Note.created_at < date_end).all())
        elif date:
            notes.update(base_query.filter(Note.created_at >= date).all())
        elif date_end:
            notes.update(base_query.filter(Note.created_at < date_end).all())

        if not (query or tags or date or date_end):
            notes.update(
                base_query.order_by(Note.updated_at.desc()).limit(limit).all()
            )

        return list(notes)[:limit]

def update_note(
    user_id: str,
    id: str,
    title: Optional[str],
    body: Optional[str],
    tags: Optional[List[str]],
) -> Optional[Note]:
    with SessionLocal() as db:
        note = db.query(Note).filter(Note.id == id, Note.user_id == user_id).first()
        if note:
            if title is not None:
                note.title = title
            if body is not None:
                note.body = body
            if tags is not None:
                note.tags = ",".join(tags)
            note.updated_at = utc_now()
            db.commit()
            db.refresh(note)
        return note

def delete_note(user_id: str, id: str) -> bool:
    with SessionLocal() as db:
        note = db.query(Note).filter(Note.id == id, Note.user_id == user_id).first()
        if note:
            db.delete(note)
            db.commit()
            return True
        return False

# Users CRUD Operations
def create_user(username: str) -> User:
    with SessionLocal() as db:
        user = User(username=username)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

def get_user_by_username(username: str) -> Optional[User]:
    with SessionLocal() as db:
        return db.query(User).filter(User.username == username).first()
    
def get_user_by_id(id: str) -> Optional[User]:
    with SessionLocal() as db:
        return db.query(User).filter(User.id == id).first()

def update_user(id: str, username: str) -> Optional[User]:
    with SessionLocal() as db:
        user = db.query(User).filter(User.id == id).first()
        if user:
            user.username = username
            db.commit()
            db.refresh(user)
        return user

def delete_user(id: str) -> bool:
    with SessionLocal() as db:
        user = db.query(User).filter(User.id == id).first()
        if user:
            db.delete(user)
            db.commit()
            return True
        return False