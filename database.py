"""
database.py
Async MongoDB connection via Motor.
Uses a 'One Section Per User' model for scalability and token persistence.
Includes health-check ping for FastAPI startup.
"""

import os
import logging
from typing import Optional

import motor.motor_asyncio
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MONGO_URI: str = os.getenv("MONGO_URI", "")
if not MONGO_URI:
    raise EnvironmentError("MONGO_URI is not set in the .env file.")

# ── Motor client ──────────────────────────────────────────────────────────────
_client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
_db = None


def get_client() -> motor.motor_asyncio.AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = motor.motor_asyncio.AsyncIOMotorClient(
            MONGO_URI,
            serverSelectionTimeoutMS=8_000,
            maxPoolSize=20,
        )
    return _client


def get_db():
    global _db
    if _db is None:
        _db = get_client()["eventai"]
    return _db


async def ping() -> bool:
    """Health-check: returns True when Atlas responds."""
    try:
        await get_client().admin.command("ping")
        return True
    except Exception as exc:
        logger.error("MongoDB ping failed: %s", exc)
        return False


# ── Face-encoding & User Management ──────────────────────────────────────────
# We now use one collection for everything related to a user
USER_COLLECTION = "user_profiles"


async def save_face_encoding(user_id: str, filename: str, encoding) -> str:
    """
    Pushes a new encoding into the user's specific 'references' array.
    Uses 'upsert' to create the user section if it doesn't exist.

    SAFE VERSION:
    Stores encoding as a plain list of floats instead of pickle binary.
    """
    db = get_db()

    # Generate a unique ID for this specific photo reference
    ref_id = ObjectId()

    new_reference = {
        "ref_id": ref_id,
        "filename": filename,
        "encoding": [float(x) for x in encoding]
    }

    # Update the single document for this user_id
    await db[USER_COLLECTION].update_one(
        {"user_id": user_id},
        {"$push": {"references": new_reference}},
        upsert=True
    )

    logger.info("Pushed new face reference to user profile: %s", user_id)
    return str(ref_id)


async def load_face_encodings(user_id: str) -> list:
    """
    Retrieves the user's single document and extracts all encodings from the array.

    SAFE VERSION:
    Loads plain numeric arrays directly from MongoDB.
    """
    db = get_db()
    user_profile = await db[USER_COLLECTION].find_one({"user_id": user_id})

    if not user_profile or "references" not in user_profile:
        return []

    encodings = [
        ref["encoding"]
        for ref in user_profile["references"]
        if "encoding" in ref
    ]

    logger.info("Loaded %d encoding(s) from user section: %s", len(encodings), user_id)
    return encodings


async def get_all_references(user_id: str) -> list:
    db = get_db()
    user_profile = await db[USER_COLLECTION].find_one(
        {"user_id": user_id},
        {"references.encoding": 0}
    )

    if not user_profile or "references" not in user_profile:
        return []

    # Clean the data here so the rest of the app doesn't have to worry about ObjectIds
    cleaned_refs = []
    for ref in user_profile["references"]:
        cleaned_refs.append({
            "ref_id": str(ref["ref_id"]),
            "filename": ref["filename"]
        })
    return cleaned_refs


async def delete_specific_reference(user_id: str, ref_id: str) -> bool:
    """
    Deletes a specific Face DNA reference from the user's document using the ref_id.
    """
    try:
        db = get_db()
        object_id = ObjectId(ref_id)

        result = await db[USER_COLLECTION].update_one(
            {"user_id": user_id},
            {"$pull": {"references": {"ref_id": object_id}}}
        )

        return result.modified_count > 0
    except Exception as e:
        logger.error(f"Error deleting reference: {e}")
        return False


async def delete_face_encodings(user_id: str) -> int:
    """
    Deletes all saved face references for a given user.
    """
    db = get_db()
    result = await db[USER_COLLECTION].update_one(
        {"user_id": user_id},
        {"$set": {"references": []}}
    )
    return result.modified_count


# ── Token Persistence (The 401 Fix) ──────────────────────────────────────────

async def save_refresh_token(user_id: str, refresh_token: str):
    """Stores the refresh token so we can stay logged in during long scans."""
    db = get_db()
    await db[USER_COLLECTION].update_one(
        {"user_id": user_id},
        {"$set": {"refresh_token": refresh_token}},
        upsert=True
    )


async def get_refresh_token(user_id: str) -> Optional[str]:
    db = get_db()
    user = await db[USER_COLLECTION].find_one({"user_id": user_id})
    return user.get("refresh_token") if user else None
