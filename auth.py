"""
GitHub OAuth login (Authorization Code flow) with a signed session cookie.

  GET  /auth/login     Redirect the browser to GitHub's consent page
  GET  /auth/callback  GitHub redirects back here with ?code=...&state=...
  GET  /auth/me        The logged-in user, or 401
  POST /auth/logout    Clear the session cookie

Flow:
  1. /auth/login stores a random `state` in the session cookie and redirects to GitHub.
  2. The user approves; GitHub redirects to /auth/callback with a one-time `code`.
  3. We check `state` (CSRF protection), exchange `code` + client secret for an access
     token server-side, and fetch the GitHub profile with it.
  4. We upsert the user and store only our own `user_id` in the cookie. The GitHub token
     is not kept — we only need it once, to learn who the user is.

Other routes require a user with `Depends(get_current_user)`.
"""

from __future__ import annotations

import logging
import secrets
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel

from config import FRONTEND_URL, GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET, GITHUB_REDIRECT_URI
from db import User, get_user, upsert_github_user

logger = logging.getLogger(__name__)

GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"  # noqa: S105 — a URL, not a secret
GITHUB_USER_URL = "https://api.github.com/user"
GITHUB_SCOPE = "read:user"  # public profile only

router = APIRouter(prefix="/auth", tags=["auth"])


class UserInfo(BaseModel):
    login: str
    name: str | None = None
    avatar_url: str | None = None


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


async def get_current_user(request: Request) -> User:
    """Return the logged-in user, or raise 401. Use as `Depends(get_current_user)`."""
    user_id = request.session.get("user_id")
    user = await get_user(user_id) if user_id else None
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return user


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _redirect_uri(request: Request) -> str:
    return GITHUB_REDIRECT_URI or str(request.url_for("auth_callback"))


@router.get("/login")
async def login(request: Request):
    """Start the OAuth flow by redirecting to GitHub."""
    if not GITHUB_CLIENT_ID or not GITHUB_CLIENT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="GitHub login is not configured. Set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET.",
        )
    state = secrets.token_urlsafe(32)
    request.session["oauth_state"] = state
    params = {
        "client_id": GITHUB_CLIENT_ID,
        "redirect_uri": _redirect_uri(request),
        "scope": GITHUB_SCOPE,
        "state": state,
        "allow_signup": "true",
    }
    return RedirectResponse(f"{GITHUB_AUTHORIZE_URL}?{urlencode(params)}")


@router.get("/callback", name="auth_callback")
async def callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
):
    """GitHub redirects here after the user approves (or denies) access."""
    expected_state = request.session.pop("oauth_state", None)
    if error:
        raise HTTPException(status_code=400, detail=f"GitHub login failed: {error}")
    if not code or not state or not expected_state or not secrets.compare_digest(state, expected_state):
        raise HTTPException(status_code=400, detail="Invalid OAuth state. Please try logging in again.")

    async with httpx.AsyncClient(timeout=10) as client:
        token_response = await client.post(
            GITHUB_TOKEN_URL,
            data={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": _redirect_uri(request),
            },
            headers={"Accept": "application/json"},
        )
        access_token = token_response.json().get("access_token")
        if not access_token:
            logger.warning("GitHub token exchange failed: %s", token_response.text[:200])
            raise HTTPException(status_code=400, detail="GitHub did not return an access token.")

        user_response = await client.get(
            GITHUB_USER_URL,
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/vnd.github+json"},
        )
        if user_response.status_code != 200:
            raise HTTPException(status_code=502, detail="Could not fetch the GitHub profile.")
        profile = user_response.json()

    user = await upsert_github_user(
        github_id=profile["id"],
        login=profile["login"],
        name=profile.get("name"),
        avatar_url=profile.get("avatar_url"),
    )
    logger.info("GitHub login: %s", user.login)

    # Start a fresh session so nothing from before login carries over (session fixation)
    request.session.clear()
    request.session["user_id"] = user.id
    return RedirectResponse(FRONTEND_URL)


@router.get("/me", response_model=UserInfo)
async def me(request: Request):
    """Return the logged-in user, or 401 if not logged in."""
    user = await get_current_user(request)
    return UserInfo(login=user.login, name=user.name, avatar_url=user.avatar_url)


@router.post("/logout", status_code=204)
async def logout(request: Request):
    """Log out by clearing the session cookie."""
    request.session.clear()
    return Response(status_code=204)
