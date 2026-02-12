"""
Signed tokens for colleague self-service actions (remove / update interests).

Each token is an HMAC-signed, base64url-encoded payload containing:
    owner_id | colleague_email | action | expiry_unix

Tokens are single-purpose and time-limited (default 30 days).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class TokenAction(str, Enum):
    REMOVE = "remove"
    UPDATE = "update"


@dataclass
class TokenPayload:
    owner_id: str
    colleague_email: str
    action: TokenAction
    expiry: float  # Unix timestamp


# Default token lifetime: 30 days
DEFAULT_TOKEN_TTL_SECONDS = 30 * 24 * 3600


def _get_secret() -> bytes:
    """Derive signing key from environment."""
    raw = os.getenv("COLLEAGUE_TOKEN_SECRET") or os.getenv("SECRET_KEY", "researchpulse-default-token-key")
    return hashlib.sha256(raw.encode()).digest()


def _sign(message: bytes) -> bytes:
    return hmac.new(_get_secret(), message, hashlib.sha256).digest()


def generate_token(
    owner_id: str,
    colleague_email: str,
    action: TokenAction,
    ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS,
) -> str:
    """Create a signed, URL-safe token string."""
    expiry = time.time() + ttl_seconds
    payload = f"{owner_id}|{colleague_email}|{action.value}|{expiry:.0f}"
    payload_bytes = payload.encode("utf-8")
    sig = _sign(payload_bytes)
    combined = payload_bytes + b"|" + base64.urlsafe_b64encode(sig)
    token = base64.urlsafe_b64encode(combined).decode("ascii")
    logger.debug("Generated %s token for %s (expires in %ds)", action.value, colleague_email, ttl_seconds)
    return token


def verify_token(token: str) -> Optional[TokenPayload]:
    """Verify and decode a token.

    Returns TokenPayload on success, None on failure (expired, tampered, malformed).
    """
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii"))
    except Exception:
        logger.warning("Token base64 decode failed")
        return None

    parts = raw.split(b"|")
    if len(parts) != 5:
        logger.warning("Token has wrong number of parts: %d", len(parts))
        return None

    payload_bytes = b"|".join(parts[:4])
    sig_b64 = parts[4]
    try:
        sig = base64.urlsafe_b64decode(sig_b64)
    except Exception:
        logger.warning("Token signature decode failed")
        return None

    expected_sig = _sign(payload_bytes)
    if not hmac.compare_digest(sig, expected_sig):
        logger.warning("Token signature mismatch")
        return None

    try:
        owner_id = parts[0].decode()
        colleague_email = parts[1].decode()
        action_str = parts[2].decode()
        expiry = float(parts[3].decode())
    except Exception:
        logger.warning("Token field decode failed")
        return None

    if time.time() > expiry:
        logger.info("Token expired for %s (action=%s)", colleague_email, action_str)
        return None

    try:
        action = TokenAction(action_str)
    except ValueError:
        logger.warning("Unknown token action: %s", action_str)
        return None

    return TokenPayload(
        owner_id=owner_id,
        colleague_email=colleague_email,
        action=action,
        expiry=expiry,
    )


def build_action_url(base_url: str, token: str) -> str:
    """Build a full action URL from a base URL and token."""
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}token={token}"


def generate_remove_url(
    base_url: str,
    owner_id: str,
    colleague_email: str,
    ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS,
) -> str:
    """Generate a signed 'remove me' URL."""
    token = generate_token(owner_id, colleague_email, TokenAction.REMOVE, ttl_seconds)
    return build_action_url(f"{base_url}/colleague/remove", token)


def generate_update_url(
    base_url: str,
    owner_id: str,
    colleague_email: str,
    ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS,
) -> str:
    """Generate a signed 'update interests' URL."""
    token = generate_token(owner_id, colleague_email, TokenAction.UPDATE, ttl_seconds)
    return build_action_url(f"{base_url}/colleague/update", token)
