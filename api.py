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
  GROQ_API_KEY=<key> uvicorn api:app --reload
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from chatbot import chat_turn, extract_config, opening_message

if TYPE_CHECKING:
    from data_model import SimulationConfig

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="SONATA Config Chatbot", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory session store:
#   session_id -> {"history": [...], "config": dict|None, "created_at": str, "label": str}
_sessions: dict[str, dict] = {}


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


def _get_session(session_id: str) -> dict:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return session


def _session_label(history: list) -> str:
    """Extract a short label from the first real user message."""
    for m in history:
        if isinstance(m, HumanMessage) and m.content != "Hello! I'd like to set up a SONATA simulation.":
            text = m.content[:50]
            return text + ("…" if len(m.content) > 50 else "")
    return "New session"


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
    session_id = str(uuid.uuid4())
    greeting = opening_message()
    _sessions[session_id] = {
        "history": [
            HumanMessage(content="Hello! I'd like to set up a SONATA simulation."),
            AIMessage(content=greeting),
        ],
        "config": None,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": "New session",
    }
    return StartResponse(session_id=session_id, reply=greeting)


@app.get("/sessions", response_model=list[SessionInfo])
async def list_sessions():
    """List all sessions with metadata (newest first)."""
    result = []
    for sid, s in _sessions.items():
        result.append(
            SessionInfo(
                session_id=sid,
                label=s.get("label", "New session"),
                created_at=s.get("created_at", ""),
                has_config=s["config"] is not None,
            ),
        )
    # Newest first
    result.sort(key=lambda x: x.created_at, reverse=True)
    return result


@app.get("/session/{session_id}/messages", response_model=list[MessageItem])
async def get_messages(session_id: str):
    """Get the full conversation history for a session."""
    session = _get_session(session_id)
    messages = []
    for m in session["history"]:
        if isinstance(m, HumanMessage):
            # Skip the synthetic opening message
            if m.content == "Hello! I'd like to set up a SONATA simulation.":
                continue
            messages.append(MessageItem(role="user", content=m.content))
        elif isinstance(m, AIMessage):
            messages.append(MessageItem(role="assistant", content=m.content))
    return messages


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send a user message and get the assistant's reply."""
    session = _get_session(req.session_id)
    history = session["history"]

    reply, config = chat_turn(history, req.message)

    # Update session label after first real user message
    if session["label"] == "New session":
        session["label"] = _session_label(history)

    config_dict: dict | None = None
    if config is not None:
        config_dict = _config_to_dict(config)
        session["config"] = config_dict

    return ChatResponse(reply=reply, config=config_dict)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Force extraction of the simulation config from the conversation so far."""
    session = _get_session(req.session_id)
    config, error = extract_config(session["history"])

    if config is not None:
        config_dict = _config_to_dict(config)
        session["config"] = config_dict
        return GenerateResponse(config=config_dict)

    return GenerateResponse(error=error)


@app.get("/download/{session_id}")
async def download(session_id: str):
    """Download the current simulation_config.json for a session."""
    session = _get_session(session_id)
    if session["config"] is None:
        raise HTTPException(
            status_code=404,
            detail="No config generated yet for this session.",
        )

    content = json.dumps(session["config"], indent=2)
    return Response(
        content=content,
        media_type="application/json",
        headers={
            "Content-Disposition": 'attachment; filename="simulation_config.json"',
        },
    )
