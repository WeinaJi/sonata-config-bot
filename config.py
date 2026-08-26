"""
Shared configuration for the SONATA Config Bot.
"""

import os

# LLM model — change here or override with env var
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite")

# Google API key — required for LLM and embeddings
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Database URL — defaults to SQLite, set to postgresql+asyncpg://... for Postgres
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///sonata_bot.db")
