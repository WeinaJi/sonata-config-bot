"""
Shared configuration for the SONATA Config Bot.
"""

import os
import secrets

# LLM model — change here or override with env var
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite")

# Google API key — required for LLM and embeddings
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Database URL — defaults to SQLite, set to postgresql+asyncpg://... for Postgres
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///sonata_bot.db")

# Render (and many hosts) provide postgres:// URLs — convert to the asyncpg driver
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# GitHub OAuth — create an OAuth App at https://github.com/settings/developers
GITHUB_CLIENT_ID = os.environ.get("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.environ.get("GITHUB_CLIENT_SECRET", "")

# Must match the OAuth App's "Authorization callback URL".
# Empty = derive it from the incoming request (<host>/auth/callback).
GITHUB_REDIRECT_URI = os.environ.get("GITHUB_REDIRECT_URI", "")

# Where the browser goes after login/logout. "/" when the backend serves the UI (Docker);
# http://localhost:5173 when running the Vite dev server.
FRONTEND_URL = os.environ.get("FRONTEND_URL", "/")

# Signs the session cookie. Without a fixed value, a random one is used and everyone is
# logged out on every restart — set it in production.
SESSION_SECRET_IS_SET = bool(os.environ.get("SESSION_SECRET"))
SESSION_SECRET = os.environ.get("SESSION_SECRET") or secrets.token_urlsafe(32)

# Send the session cookie over HTTPS only. Set to "true" in production.
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "false").lower() == "true"
