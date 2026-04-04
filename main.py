"""
main.py
EventAI – FastAPI entry point.

Endpoints:
  GET  /                         → Serve index.html
  GET  /health                   → DB + app health
  GET  /auth/login               → Start Google OAuth
  GET  /auth/callback            → OAuth callback
  GET  /auth/logout              → Clear session
  GET  /auth/me                  → Current user info
  POST /upload-reference         → Upload face photo; encoding saved to MongoDB
  GET  /my-encodings             → List saved encodings for current user
  DELETE /my-encodings           → Delete all saved encodings
  POST /search                   → Kick off deep Drive search (SSE streaming)
  GET  /download/{search_id}     → Download result ZIP
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

import auth
import database
import engine


import logging

# 1. Mute the 'pkg_resources' warning specifically
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

# 2. Force external libraries to only show WARNINGS or ERRORS
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# 3. Keep YOUR logs active (so you still see '✓ MongoDB Atlas connected')
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("main")

SECRET_KEY: str = os.getenv("SESSION_SECRET", secrets.token_hex(32))
BASE_DIR   = Path(__file__).parent

# In-memory store for completed ZIP blobs  { search_id: bytes }
_zip_store: dict[str, bytes] = {}


# ── App lifecycle ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    ok = await database.ping()
    if ok:
        logger.info("✓ MongoDB Atlas connected")
    else:
        logger.error("✗ MongoDB Atlas NOT reachable — check MONGO_URI")
    yield
    # Shutdown: close motor client
    client = database.get_client()
    client.close()
    logger.info("MongoDB client closed")


app = FastAPI(title="EventAI", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    session_cookie="eventai_session",  # Explicit name helps debugging
    max_age=3600 * 8,
    same_site="lax",                   # 'lax' is usually fine, but 'strict' would break it
    https_only=False,                  # CRITICAL: Since you aren't using https://
)

app.include_router(auth.router)


# ── Static + HTML ──────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # We use index.html for BOTH login and dashboard
    # The JavaScript inside index.html will decide which one to show
    try:
        index_html = (BASE_DIR / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(index_html)
    except Exception as e:
        logger.error(f"Error loading index.html: {e}")
        raise HTTPException(status_code=500, detail="index.html missing")


@app.get("/health")
async def health():
    return {"status": "ok", "mongo": await database.ping()}


# ── Reference face upload ──────────────────────────────────────────────────────
@app.post("/upload-reference")
async def upload_reference(
    request: Request,
    file: UploadFile = File(...),
):
    user = auth.require_user(request)
    user_id = user["sub"]

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:          # 10 MB hard limit
        raise HTTPException(413, "Reference photo too large (max 10 MB).")

    # Run CPU-heavy encoding in thread pool so the event loop isn't blocked
    loop = asyncio.get_running_loop()
    try:
        encodings = await loop.run_in_executor(
            None, engine.encode_reference_image, contents
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    saved_ids = []
    for enc in encodings:
        doc_id = await database.save_face_encoding(
            user_id, file.filename or "reference.jpg", enc
        )
        saved_ids.append(doc_id)

    return {
        "message": f"Saved {len(saved_ids)} face encoding(s).",
        "encoding_ids": saved_ids,
    }


@app.get("/my-encodings")
async def my_encodings(request: Request):
    user = auth.require_user(request)
    refs = await database.get_all_references(user["sub"])
    
    # ── THE FIX ──────────────────────────────────────────────────────────
    # Convert MongoDB ObjectIds to plain strings so FastAPI can send them
    serializable_refs = []
    for ref in refs:
        serializable_refs.append({
            "ref_id": str(ref["ref_id"]), # Convert ObjectId to string
            "filename": ref["filename"]
        })
    # ─────────────────────────────────────────────────────────────────────

    return {"count": len(serializable_refs), "references": serializable_refs}


@app.delete("/my-encodings")
async def delete_encodings(request: Request):
    user    = auth.require_user(request)
    deleted = await database.delete_face_encodings(user["sub"])
    return {"deleted": deleted}

# Add this to main.py if it's missing!
@app.delete("/delete-reference/{ref_id}")
async def delete_ref_endpoint(ref_id: str, request: Request):
    user = auth.require_user(request)
    # This calls the logic we wrote in database.py
    success = await database.delete_specific_reference(user["sub"], ref_id)
    
    if not success:
        logger.error(f"Delete failed for ref_id: {ref_id}")
        raise HTTPException(status_code=404, detail="Reference DNA not found")
        
    logger.info(f"Deleted DNA reference {ref_id} for user {user['sub']}")
    return {"success": True}


# ── Deep search (SSE streaming progress) ──────────────────────────────────────
# Find your current @app.post("/search") and replace it AND ADD the new local one:

# ── Deep search (SSE streaming progress) ──────────────────────────────────────
@app.post("/search")
async def search(
    request: Request,
    drive_link: str = Form(...),
    tolerance: float = Form(0.50),  
    model: str = Form("hog"),       
    upsample: int = Form(0)         
):
    user       = auth.require_user(request)
    user_id    = user["sub"]
    drv_token  = request.session.get("drive_token", "")

    # 1. Fetch the offline refresh token from MongoDB
    refresh_token = await database.get_refresh_token(user_id)

    if not drv_token and not refresh_token:
        raise HTTPException(401, "Google Drive token missing. Please log in again.")

    known_encodings = await database.load_face_encodings(user_id)
    if not known_encodings:
        raise HTTPException(400, "No reference face found. Upload a reference photo first.")

    async def event_stream() -> AsyncGenerator[str, None]:
        search_id = str(uuid.uuid4())
        progress_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def progress_cb(current, total, filename, matched_count=0):
            loop.call_soon_threadsafe(
                progress_queue.put_nowait,
                {
                    "current": current, 
                    "total": total, 
                    "filename": filename, 
                    "matched": matched_count 
                },
            )

        # 2. Pass the refresh_token straight into the engine!
        future = loop.run_in_executor(
            None,
            lambda: engine.run_deep_search(
                drive_link, drv_token, refresh_token, known_encodings, tolerance, model, upsample, progress_cb
            ),
        )

        while not future.done():
            try:
                prog = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                yield f"data: {json.dumps({'type':'progress', **prog})}\n\n"
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"

        while not progress_queue.empty():
            prog = progress_queue.get_nowait()
            yield f"data: {json.dumps({'type':'progress', **prog})}\n\n"

        try:
            zip_bytes = await future
            _zip_store[search_id] = zip_bytes
            import zipfile, io
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                matched = len(zf.namelist())
            payload = {"type": "done", "search_id": search_id, "matched": matched}
            yield f"data: {json.dumps(payload)}\n\n"
        except Exception as exc:
            logger.error("Search failed: %s", exc)
            yield f"data: {json.dumps({'type':'error','message':str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- NEW ENDPOINT FOR OPTION 2: LOCAL FILES ---
@app.post("/search-local")
async def search_local(
    request: Request,
    files: list[UploadFile] = File(...),
    tolerance: float = Form(0.50),
    model: str = Form("hog"),
    upsample: int = Form(0)
):
    user = auth.require_user(request)
    user_id = user["sub"]
    
    known_encodings = await database.load_face_encodings(user_id)
    if not known_encodings:
        raise HTTPException(400, "No reference face found. Upload a reference photo first.")

    # Read files into memory to pass to engine
    file_data_list = []
    for f in files:
        content = await f.read()
        file_data_list.append((f.filename, content))
        
    async def event_stream() -> AsyncGenerator[str, None]:
        search_id = str(uuid.uuid4())
        progress_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def progress_cb(current, total, filename, matched_count=0):
            loop.call_soon_threadsafe(
                progress_queue.put_nowait,
                {
                    "current": current, 
                    "total": total, 
                    "filename": filename, 
                    "matched": matched_count # <-- THIS WAS MISSING!
                },
            )

        future = loop.run_in_executor(
            None,
            lambda: engine.run_local_search(
                file_data_list, known_encodings, tolerance, model, upsample, progress_cb
            ),
        )

        while not future.done():
            try:
                prog = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                yield f"data: {json.dumps({'type':'progress', **prog})}\n\n"
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"

        while not progress_queue.empty():
            prog = progress_queue.get_nowait()
            yield f"data: {json.dumps({'type':'progress', **prog})}\n\n"

        try:
            zip_bytes = await future
            _zip_store[search_id] = zip_bytes
            import zipfile, io
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                matched = len(zf.namelist())
            payload = {"type": "done", "search_id": search_id, "matched": matched}
            yield f"data: {json.dumps(payload)}\n\n"
        except Exception as exc:
            logger.error("Search local failed: %s", exc)
            yield f"data: {json.dumps({'type':'error','message':str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── ZIP download ───────────────────────────────────────────────────────────────
@app.get("/download/{search_id}")
async def download_zip(search_id: str, request: Request):
    auth.require_user(request)
    zip_bytes = _zip_store.get(search_id)
    if not zip_bytes:
        raise HTTPException(404, "ZIP not found. It may have expired.")

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="eventai_matches_{search_id[:8]}.zip"'
        },
    )


# ── Dev server ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
