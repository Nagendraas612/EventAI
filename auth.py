"""
auth.py
Google OAuth 2.0 via Authlib + Starlette sessions.
"""

import os
import logging
from typing import Optional

from authlib.integrations.starlette_client import OAuth, OAuthError
from starlette.requests import Request
from starlette.responses import RedirectResponse, JSONResponse
from fastapi import APIRouter
from database import get_db

# Load environment variables
load_dotenv_once = __import__("dotenv").load_dotenv
load_dotenv_once()

logger = logging.getLogger(__name__)

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
REDIRECT_URI = os.getenv("https://eventai-w89h.onrender.com", "http://localhost:8000/auth/callback")

if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    raise EnvironmentError(
        "GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET must be set in .env"
    )

# ── OAuth client ───────────────────────────────────────────────────────────────
oauth = OAuth()
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
    access_token_url="https://oauth2.googleapis.com/token",
    client_kwargs={
        "scope": (
            "openid email profile "
            "https://www.googleapis.com/auth/drive.readonly"
        ),
        # 'select_account' lets you switch users; 
        # 'consent' forces Google to show the permission checkboxes again.
        "prompt": "select_account consent", 
        "access_type": "offline",
    },
)

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_current_user(request: Request) -> Optional[dict]:
    return request.session.get("user")


def require_user(request: Request) -> dict:
    user = get_current_user(request)
    if not user:
        raise __import__("fastapi").HTTPException(
            status_code=401, detail="Not authenticated"
        )
    return user


# ── Routes ─────────────────────────────────────────────────────────────────────
@router.get("/login")
async def login(request: Request):
    """Redirect the browser to Google's consent screen."""
    return await oauth.google.authorize_redirect(request, REDIRECT_URI)


@router.get("/callback")
async def callback(request: Request):
    """Handle the OAuth redirect; store user info and the Refresh Token."""
    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError as exc:
        logger.error("OAuth error: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=400)

    user_info = token.get("userinfo") or await oauth.google.userinfo(token=token)
    user_id = user_info["sub"]

    # ── Store Refresh Token in MongoDB ──────────────────────────
    refresh_token = token.get("refresh_token")
    if refresh_token:
        db = get_db()
        # Store it in our new 'user_profiles' collection
        await db["user_profiles"].update_one(
            {"user_id": user_id},
            {"$set": {"refresh_token": refresh_token}},
            upsert=True
        )
        logger.info("Refresh token saved for user: %s", user_id)
    # ──────────────────────────────────────────────────────────────

    # Store standard session info
    request.session["user"] = {
        "sub":     user_id,
        "email":   user_info["email"],
        "name":    user_info.get("name", ""),
        "picture": user_info.get("picture", ""),
    }
    
    # This is the temporary token for the current session
    request.session["drive_token"] = token.get("access_token", "")

    logger.info("User authenticated: %s", user_info["email"])
    return RedirectResponse(url="/")


@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/")


@router.get("/me")
async def me(request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"authenticated": False})
    return JSONResponse({"authenticated": True, "user": user})
