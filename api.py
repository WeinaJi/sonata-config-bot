"""
SONATA Config Chatbot — FastAPI server
======================================
Endpoints:
  GET  /                         Serve the chat UI (static/index.html)
  POST /session                  Start a new session, returns greeting + session_id
  GET  /sessions                 List all sessions with metadata
  GET  /session/{sid}/messages   Get conversation history for a session
  POST /chat                     Send a message, returns reply + optional config JSON
  POST /generate                 Force config extraction from current history
  GET  /download/{sid}           Download simulation_config.json for a session

Run with:
  GROQ_API_KEY=<key> DATABASE_URL=<url> uvicorn api:app --reload
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

# LLM
# from chatbot import chat_turn, extract_config, opening_message
# Agent
from agent import agent_chat_turn as chat_turn
from agent import agent_opening_message as opening_message
from chatbot import extract_config
from db import (
    add_messages,
    create_session,
    get_session,
    init_db,
    update_session_config,
    update_session_label,
)
from db import (
    get_messages as db_get_messages,
)
from db import (
    list_sessions as db_list_sessions,
)

if TYPE_CHECKING:
    from data_model import SimulationConfig

logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Create DB tables on startup."""
    await init_db()
    yield


app = FastAPI(title="SONATA Config Chatbot", version="0.1.0", lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class StartResponse(BaseModel):
    session_id: str
    reply: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    config: dict | None = None
    error: str | None = None


class GenerateRequest(BaseModel):
    session_id: str


class GenerateResponse(BaseModel):
    config: dict | None = None
    error: str | None = None


class SessionInfo(BaseModel):
    session_id: str
    label: str
    created_at: str
    has_config: bool


class MessageItem(BaseModel):
    role: str  # "user" or "assistant"
    content: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config_to_dict(config: SimulationConfig) -> dict:
    return json.loads(config.model_dump_json(exclude_none=True))


async def _get_session_or_404(session_id: str):
    """Fetch session from DB or raise 404."""
    try:
        sid = uuid.UUID(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"Invalid session ID: {session_id}") from exc
    session = await get_session(sid)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return session


async def _rebuild_history(session_id: uuid.UUID) -> list:
    """Rebuild the LangChain message history from DB messages."""
    messages = await db_get_messages(session_id)
    history = []
    for m in messages:
        if m.role == "user":
            history.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            history.append(AIMessage(content=m.content))
    return history


def _extract_label(user_message: str) -> str:
    """Create a short label from a user message."""
    text = user_message[:50]
    return text + ("…" if len(user_message) > 50 else "")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the chat UI."""
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="UI not found.")
    return HTMLResponse(content=index.read_text(encoding="utf-8"))


@app.post("/session", response_model=StartResponse)
async def new_session():
    """Create a new chat session and return the assistant's opening message."""
    session_id = uuid.uuid4()
    greeting = opening_message()
    await create_session(session_id, greeting)
    return StartResponse(session_id=str(session_id), reply=greeting)


@app.get("/sessions", response_model=list[SessionInfo])
async def list_sessions():
    """List all sessions with metadata (newest first)."""
    sessions = await db_list_sessions()
    return [
        SessionInfo(
            session_id=str(s.id),
            label=s.label or "New session",
            created_at=s.created_at.isoformat(timespec="seconds") if s.created_at else "",
            has_config=s.config is not None,
        )
        for s in sessions
    ]


@app.get("/session/{session_id}/messages", response_model=list[MessageItem])
async def get_messages_endpoint(session_id: str):
    """Get the full conversation history for a session."""
    await _get_session_or_404(session_id)
    messages = await db_get_messages(uuid.UUID(session_id))
    result = []
    for m in messages:
        # Skip the synthetic opening message
        if m.role == "user" and m.content == "Hello! I'd like to set up a SONATA simulation.":
            continue
        result.append(MessageItem(role=m.role, content=m.content))
    return result


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send a user message and get the assistant's reply."""
    session = await _get_session_or_404(req.session_id)
    sid = uuid.UUID(req.session_id)

    # Rebuild history from DB
    history = await _rebuild_history(sid)

    # Run the LLM
    reply, config = chat_turn(history, req.message)

    # Persist messages
    await add_messages(sid, req.message, reply)

    # Update label if still default
    if session.label == "New session":
        await update_session_label(sid, _extract_label(req.message))

    # Persist config if generated
    config_dict: dict | None = None
    if config is not None:
        config_dict = _config_to_dict(config)
        await update_session_config(sid, config_dict)

    return ChatResponse(reply=reply, config=config_dict)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Force extraction of the simulation config from the conversation so far."""
    await _get_session_or_404(req.session_id)
    sid = uuid.UUID(req.session_id)

    history = await _rebuild_history(sid)
    config, error = extract_config(history)

    if config is not None:
        config_dict = _config_to_dict(config)
        await update_session_config(sid, config_dict)
        return GenerateResponse(config=config_dict)

    return GenerateResponse(error=error)


@app.get("/download/{session_id}")
async def download(session_id: str):
    """Download the current simulation_config.json for a session."""
    session = await _get_session_or_404(session_id)
    if session.config is None:
        raise HTTPException(
            status_code=404,
            detail="No config generated yet for this session.",
        )

    content = json.dumps(session.config, indent=2)
    return Response(
        content=content,
        media_type="application/json",
        headers={
            "Content-Disposition": 'attachment; filename="simulation_config.json"',
        },
    )
