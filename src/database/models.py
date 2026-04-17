import uuid
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from .utils import utc_now

Base = declarative_base()

class Note(Base):
    __tablename__ = "notes"
    
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), index=True)
    title = Column(String, index=True)
    body = Column(Text, nullable=False)
    tags = Column(String)  # comma-separated for simplicity
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    # Establishes the Many-to-One relationship back to User
    user = relationship("User", back_populates="notes")

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=utc_now)
    
    # Establishes the One-to-Many relationship to Notes
    notes = relationship("Note", back_populates="user", cascade="all, delete-orphan")