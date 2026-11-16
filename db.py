"""
Database layer — async session persistence via SQLAlchemy.

Supports both SQLite (default) and PostgreSQL.
Set DATABASE_URL env var to switch:
  - SQLite (default):  sqlite+aiosqlite:///sonata_bot.db
  - PostgreSQL:        postgresql+asyncpg://localhost/sonata_config_bot

Tables:
  users     — one row per GitHub account that has logged in
  sessions  — one row per chat session (owner FK, label, config JSON, timestamps)
  messages  — one row per message (session FK, role, content, timestamp)
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Index, String, Text, inspect, select, text
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


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    github_id = Column(BigInteger, unique=True, nullable=False)  # stable; the login name can change
    login = Column(String(100), nullable=False)
    name = Column(String(200), nullable=True)
    avatar_url = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class Session(Base):
    __tablename__ = "sessions"
    # Matches the "my sessions, newest first" query
    __table_args__ = (Index("ix_sessions_user_created", "user_id", "created_at"),)

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)  # NULL for sessions created before auth
    label = Column(String(200), default="New session")
    config_json = Column(Text, nullable=True)  # stored as JSON string
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

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
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    session = relationship("Session", back_populates="messages")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def init_db() -> None:
    """Create tables if they don't exist, and add columns introduced since."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_sessions_user_id)


def _add_sessions_user_id(sync_conn) -> None:
    """Lightweight migration: `create_all` never alters existing tables, so add the
    owner column (and its index) to a `sessions` table created before auth existed.
    Use Alembic for anything bigger than this."""
    columns = {c["name"] for c in inspect(sync_conn).get_columns("sessions")}
    if "user_id" not in columns:
        sync_conn.execute(text("ALTER TABLE sessions ADD COLUMN user_id VARCHAR(36) REFERENCES users(id)"))
    sync_conn.execute(
        text("CREATE INDEX IF NOT EXISTS ix_sessions_user_created ON sessions (user_id, created_at)"),
    )


async def get_user(user_id: str) -> User | None:
    """Fetch a user by our internal ID, or None if not found."""
    async with async_session_factory() as db:
        return await db.get(User, user_id)


async def upsert_github_user(github_id: int, login: str, name: str | None, avatar_url: str | None) -> User:
    """Create the user on first login; refresh their profile fields on later logins."""
    async with async_session_factory() as db:
        result = await db.execute(select(User).where(User.github_id == github_id))
        user = result.scalar_one_or_none()
        if user is None:
            user = User(github_id=github_id)
            db.add(user)
        user.login = login
        user.name = name
        user.avatar_url = avatar_url
        await db.commit()
        return user


async def create_session(session_id: uuid.UUID, greeting: str, user_id: str) -> Session:
    """Insert a new session owned by `user_id` + the opening messages."""
    async with async_session_factory() as db:
        s = Session(id=str(session_id), user_id=user_id, label="New session")
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


async def list_sessions(user_id: str) -> list[Session]:
    """Return the user's sessions, newest first."""
    async with async_session_factory() as db:
        result = await db.execute(
            select(Session).where(Session.user_id == user_id).order_by(Session.created_at.desc()),
        )
        return list(result.scalars().all())
