"""
Database layer — async session persistence via SQLAlchemy.

Supports both SQLite (default) and PostgreSQL.
Set DATABASE_URL env var to switch:
  - SQLite (default):  sqlite+aiosqlite:///sonata_bot.db
  - PostgreSQL:        postgresql+asyncpg://localhost/sonata_config_bot

Tables:
  sessions  — one row per chat session (id, label, config JSON, timestamps)
  messages  — one row per message (session FK, role, content, timestamp)
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, ForeignKey, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship

from config import DATABASE_URL

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    label = Column(String(200), default="New session")
    config_json = Column(Text, nullable=True)  # stored as JSON string
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    messages = relationship("Message", back_populates="session", order_by="Message.created_at")

    @property
    def config(self) -> dict | None:
        if self.config_json is None:
            return None
        return json.loads(self.config_json)

    @config.setter
    def config(self, value: dict | None) -> None:
        self.config_json = json.dumps(value) if value is not None else None


class Message(Base):
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    session = relationship("Session", back_populates="messages")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def init_db() -> None:
    """Create tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def create_session(session_id: uuid.UUID, greeting: str) -> Session:
    """Insert a new session + the opening messages."""
    async with async_session_factory() as db:
        s = Session(id=str(session_id), label="New session")
        db.add(s)
        db.add(
            Message(session_id=str(session_id), role="user", content="Hello! I'd like to set up a SONATA simulation."),
        )
        db.add(Message(session_id=str(session_id), role="assistant", content=greeting))
        await db.commit()
        return s


async def get_session(session_id: uuid.UUID) -> Session | None:
    """Fetch a session by ID, or None if not found."""
    async with async_session_factory() as db:
        return await db.get(Session, str(session_id))


async def get_messages(session_id: uuid.UUID) -> list[Message]:
    """Get all messages for a session, ordered by time."""
    async with async_session_factory() as db:
        result = await db.execute(
            select(Message).where(Message.session_id == str(session_id)).order_by(Message.created_at),
        )
        return list(result.scalars().all())


async def add_messages(session_id: uuid.UUID, user_content: str, assistant_content: str) -> None:
    """Append a user + assistant message pair to a session."""
    async with async_session_factory() as db:
        db.add(Message(session_id=str(session_id), role="user", content=user_content))
        db.add(Message(session_id=str(session_id), role="assistant", content=assistant_content))
        await db.commit()


async def update_session_label(session_id: uuid.UUID, label: str) -> None:
    """Update the session label (first user message preview)."""
    async with async_session_factory() as db:
        s = await db.get(Session, str(session_id))
        if s:
            s.label = label
            await db.commit()


async def update_session_config(session_id: uuid.UUID, config: dict) -> None:
    """Store the generated config JSON on the session."""
    async with async_session_factory() as db:
        s = await db.get(Session, str(session_id))
        if s:
            s.config = config
            await db.commit()


async def list_sessions() -> list[Session]:
    """Return all sessions, newest first."""
    async with async_session_factory() as db:
        result = await db.execute(select(Session).order_by(Session.created_at.desc()))
        return list(result.scalars().all())
