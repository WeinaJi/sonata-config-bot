"""
SONATA Config Chatbot — FastAPI server
======================================
Endpoints (all except / and /auth/* require a logged-in GitHub user):
  GET  /                         Serve the chat UI (static/index.html)
  GET  /auth/login|callback|me   GitHub OAuth login, see auth.py
  POST /auth/logout              Clear the session cookie
  POST /session                  Start a new session, returns greeting + session_id
  GET  /sessions                 List the user's sessions with metadata
  GET  /session/{sid}/messages   Get conversation history for a session
  POST /chat                     Send a message, returns reply + optional config JSON
  POST /generate                 Force config extraction from current history
  GET  /download/{sid}           Download simulation_config.json for a session

Run with:
  GOOGLE_API_KEY=<key> GITHUB_CLIENT_ID=<id> GITHUB_CLIENT_SECRET=<secret> uvicorn api:app --reload
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

# LLM
# from chat_chain import chat_turn, extract_config, opening_message
# Agent
from agent import agent_chat_turn as chat_turn
from agent import agent_extract_config as extract_config
from agent import agent_opening_message as opening_message
from auth import get_current_user
from auth import router as auth_router
from config import COOKIE_SECURE, SESSION_SECRET, SESSION_SECRET_IS_SET
from db import (
    User,  # runtime import: FastAPI reads it from the route signatures
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

# CORS — allow the frontend origin(s). Set CORS_ORIGINS env var in production
# (comma-separated), defaults to "*" for local development.
_cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Signed session cookie holding the logged-in user's ID (and the OAuth state during login).
# HttpOnly, so page JavaScript can't read it; SameSite=lax, so it survives the redirect back from GitHub.
if not SESSION_SECRET_IS_SET:
    logging.getLogger(__name__).warning("SESSION_SECRET not set — using a random one; logins reset on restart.")
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="sonata_session",
    max_age=7 * 24 * 3600,
    same_site="lax",
    https_only=COOKIE_SECURE,
)

app.include_router(auth_router)

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


async def _get_session_or_404(session_id: str, user: User):
    """Fetch the user's session from DB or raise 404.

    Someone else's session is also a 404 (not 403), so IDs of other users' sessions aren't revealed.
    """
    try:
        sid = uuid.UUID(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"Invalid session ID: {session_id}") from exc
    session = await get_session(sid)
    if session is None or session.user_id != user.id:
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
async def new_session(user: User = Depends(get_current_user)):
    """Create a new chat session and return the assistant's opening message."""
    session_id = uuid.uuid4()
    greeting = opening_message()
    await create_session(session_id, greeting, user.id)
    return StartResponse(session_id=str(session_id), reply=greeting)


@app.get("/sessions", response_model=list[SessionInfo])
async def list_sessions(user: User = Depends(get_current_user)):
    """List the user's sessions with metadata (newest first)."""
    sessions = await db_list_sessions(user.id)
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
async def get_messages_endpoint(session_id: str, user: User = Depends(get_current_user)):
    """Get the full conversation history for a session."""
    await _get_session_or_404(session_id, user)
    messages = await db_get_messages(uuid.UUID(session_id))
    result = []
    for m in messages:
        # Skip the synthetic opening message
        if m.role == "user" and m.content == "Hello! I'd like to set up a SONATA simulation.":
            continue
        result.append(MessageItem(role=m.role, content=m.content))
    return result


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user: User = Depends(get_current_user)):
    """Send a user message and get the assistant's reply."""
    session = await _get_session_or_404(req.session_id, user)
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
async def generate(req: GenerateRequest, user: User = Depends(get_current_user)):
    """Force extraction of the simulation config from the conversation so far."""
    await _get_session_or_404(req.session_id, user)
    sid = uuid.UUID(req.session_id)

    history = await _rebuild_history(sid)
    config, error = extract_config(history)

    if config is not None:
        config_dict = _config_to_dict(config)
        await update_session_config(sid, config_dict)
        return GenerateResponse(config=config_dict)

    return GenerateResponse(error=error)


@app.get("/download/{session_id}")
async def download(session_id: str, user: User = Depends(get_current_user)):
    """Download the current simulation_config.json for a session."""
    session = await _get_session_or_404(session_id, user)
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
