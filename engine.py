"""
engine.py
Deep face-search engine:
  • Streams photos from Google Drive OR Local Uploads.
  • Auto-Refreshes expired Google tokens seamlessly.
  • Uses dynamic tolerance from the UI slider to prevent false positives.
  • Averages Face DNA for massive accuracy boosts.
"""

from __future__ import annotations

import os
import io
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator

import numpy as np
import requests
import face_recognition
from PIL import Image
from PIL import Image
import io

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DRIVE_API_BASE   = "https://www.googleapis.com/drive/v3"
DRIVE_LIST_URL   = f"{DRIVE_API_BASE}/files"
DRIVE_DL_URL     = f"{DRIVE_API_BASE}/files/{{file_id}}?alt=media"
PAGE_SIZE        = 100
MAX_WORKERS      = 4  # <--- Increased for faster parallel processing (was 1)
SUPPORTED_EXTS   = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# ── Token Auto-Refresher ───────────────────────────────────────────────────────
def _refresh_token_if_needed(resp: requests.Response, token_state: dict) -> bool:
    """If 401 Expired, automatically swap the refresh token for a new access token."""
    if resp.status_code == 401 and token_state.get("refresh_token"):
        logger.info("⚠️ Access token expired! Auto-refreshing in the background...")
        url = "https://oauth2.googleapis.com/token"
        payload = {
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
            "refresh_token": token_state["refresh_token"],
            "grant_type": "refresh_token"
        }
        token_resp = requests.post(url, data=payload)
        
        if token_resp.status_code == 200:
            token_state["access_token"] = token_resp.json()["access_token"]
            logger.info("✅ Successfully generated a fresh Access Token!")
            return True
        else:
            logger.error(f"Failed to refresh Google Token: {token_resp.text}")
            
    return False

# ── Google Drive helpers ───────────────────────────────────────────────────────
def _folder_id_from_link(drive_link: str) -> str:
    import re
    patterns = [
        r"/folders/([a-zA-Z0-9_-]{10,})",
        r"[?&]id=([a-zA-Z0-9_-]{10,})",
    ]
    for pat in patterns:
        m = re.search(pat, drive_link)
        if m:
            return m.group(1)
    raise ValueError(f"Cannot extract folder ID from link: {drive_link!r}")

def _list_drive_files(folder_id: str, token_state: dict) -> Generator[dict, None, None]:
    params = {
        "q": (f"'{folder_id}' in parents and mimeType contains 'image/' and trashed = false"),
        "fields": "nextPageToken, files(id, name)",
        "pageSize": PAGE_SIZE,
    }
    while True:
        headers = {"Authorization": f"Bearer {token_state['access_token']}"}
        resp = requests.get(DRIVE_LIST_URL, headers=headers, params=params, timeout=15)
        
        # If Token Died -> Refresh -> Retry
        if _refresh_token_if_needed(resp, token_state):
            headers = {"Authorization": f"Bearer {token_state['access_token']}"}
            resp = requests.get(DRIVE_LIST_URL, headers=headers, params=params, timeout=15)
            
        resp.raise_for_status()
        data = resp.json()

        for f in data.get("files", []):
            ext = "." + f["name"].rsplit(".", 1)[-1].lower() if "." in f["name"] else ""
            if ext in SUPPORTED_EXTS:
                yield f

        page_token = data.get("nextPageToken")
        if not page_token:
            break
        params["pageToken"] = page_token

def _download_image_bytes(file_id: str, token_state: dict) -> bytes | None:
    url = DRIVE_DL_URL.format(file_id=file_id)
    headers = {"Authorization": f"Bearer {token_state['access_token']}"}
    try:
        resp = requests.get(url, headers=headers, timeout=20, stream=True)
        
        # If Token Died -> Refresh -> Retry
        if _refresh_token_if_needed(resp, token_state):
            headers = {"Authorization": f"Bearer {token_state['access_token']}"}
            resp = requests.get(url, headers=headers, timeout=20, stream=True)
            
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        logger.warning("Failed to download file_id=%s: %s", file_id, exc)
        return None

# ── Face-encoding helpers ──────────────────────────────────────────────────────
def encode_reference_image(image_bytes: bytes, num_jitters: int = 1, model: str = "large") -> list[np.ndarray]:
    """Encode reference image with configurable accuracy/speed tradeoff.
    
    Args:
        image_bytes: Raw image data
        num_jitters: Face encoding iterations (1=fast, 10=accurate but slow)
        model: Face encoding model ("small"=fast, "large"=accurate)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Resize if too large for faster processing (~1-2 seconds vs 5-10 seconds)
    max_size = 800  # Sufficient for face detection, saves time
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    arr = np.array(img)
    encodings = face_recognition.face_encodings(
        arr,
        known_face_locations=face_recognition.face_locations(arr, model="hog"),
        num_jitters=num_jitters,
        model=model,
    )
    if not encodings:
        raise ValueError("No face detected in the reference photo. Please upload a clear selfie.")
    return [enc.tolist() for enc in encodings]

def prepare_encodings(known_encodings: list) -> list[np.ndarray]:
    np_encodings = [np.array(enc) if not isinstance(enc, np.ndarray) else enc for enc in known_encodings]
    if len(np_encodings) > 1:
        avg_encoding = np.mean(np_encodings, axis=0)
        return [avg_encoding]
    return np_encodings

# ── Core Image Processor ───────────────────────────────────────────────────────
def _process_image_bytes(filename: str, img_bytes: bytes, known_encodings: list[np.ndarray], tolerance: float, model_type: str, upsample: int) -> tuple[str, bytes | None]:
    try:
        # 1. Load image and immediately resize to save RAM & SPEED
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Reduced to 700px for 3-5x faster processing (still detects faces well)
        max_size = 700 
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # 2. Convert to numpy for face_recognition
        img_arr = np.array(img)
        
        # 3. Detect faces (using hog + upsample=0 by default for speed)
        locations = face_recognition.face_locations(img_arr, number_of_times_to_upsample=upsample, model=model_type)
        
        if not locations:
            # Memory cleanup
            del img_arr
            return filename, None

        # 4. Encode and Compare - IMPROVED FOR GROUP PHOTOS
        candidate_encodings = face_recognition.face_encodings(img_arr, known_face_locations=locations, model="large")
        
        found_match = False
        for candidate in candidate_encodings:
            # Use distance-based matching for better accuracy in group photos
            distances = face_recognition.face_distance(known_encodings, candidate)
            # Match if ANY known encoding is within tolerance
            if np.any(distances < tolerance):
                found_match = True
                break
        
        # 5. Handle result and clear memory
        result_bytes = None
        if found_match:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            result_bytes = buf.getvalue()

        # CRITICAL: Manually clear the large array from RAM
        del img_arr
        img.close()
        
        return filename, result_bytes

    except Exception as exc:
        logger.warning("Error processing %s: %s", filename, exc)
    return filename, None


def _process_drive_file(file_meta, token_state, known_encodings, tolerance, model_type, upsample):
    img_bytes = _download_image_bytes(file_meta["id"], token_state)
    if not img_bytes: return file_meta["name"], None
    return _process_image_bytes(file_meta["name"], img_bytes, known_encodings, tolerance, model_type, upsample)


# ── Public APIs ─────────────────────────────────────────────────────────────────
def run_deep_search(
    drive_link: str,
    access_token: str,
    refresh_token: str, # <--- 8th Parameter is here!
    known_encodings: list,
    tolerance: float,
    model_type: str,
    upsample: int,
    progress_callback=None,
) -> bytes:
    """Option 1: Google Drive Search"""
    
    token_state = {
        "access_token": access_token,
        "refresh_token": refresh_token
    }

    folder_id = _folder_id_from_link(drive_link)
    master_dna = prepare_encodings(known_encodings)

    all_files = list(_list_drive_files(folder_id, token_state))
    total = len(all_files)
    if total == 0: raise ValueError("No supported images found in Drive folder.")

    if progress_callback: progress_callback(0, total, "Starting scan engine...", 0)

    matched: list[tuple[str, bytes]] = []
    processed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {
            pool.submit(_process_drive_file, f, token_state, master_dna, tolerance, model_type, upsample): f
            for f in all_files
        }
        for future in as_completed(future_map):
            processed += 1
            try:
                filename, img_bytes = future.result()
                if img_bytes: matched.append((filename, img_bytes))
                # Live UI Update parameter included!
                if progress_callback: progress_callback(processed, total, filename, len(matched))
            except Exception as exc:
                logger.error("Unexpected worker error: %s", exc)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in matched: zf.writestr(name, data)
    zip_buf.seek(0)
    return zip_buf.read()


def run_local_search(
    files_data: list[tuple[str, bytes]],
    known_encodings: list,
    tolerance: float,
    model_type: str,
    upsample: int,
    progress_callback=None,
) -> bytes:
    """Option 2: Local Files Search"""
    master_dna = prepare_encodings(known_encodings)
    total = len(files_data)
    if progress_callback: progress_callback(0, total, "Starting local scan...", 0)

    matched = []
    processed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {
            pool.submit(_process_image_bytes, fname, fdata, master_dna, tolerance, model_type, upsample): fname
            for fname, fdata in files_data
        }
        for future in as_completed(future_map):
            processed += 1
            try:
                filename, out_bytes = future.result()
                if out_bytes: matched.append((filename, out_bytes))
                # Live UI Update parameter included!
                if progress_callback: progress_callback(processed, total, filename, len(matched))
            except Exception as exc:
                logger.error("Unexpected worker error: %s", exc)
    
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in matched: zf.writestr(name, data)
    zip_buf.seek(0)
    return zip_buf.read()
