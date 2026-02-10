"""
Join Code Encryption - AES encryption for colleague join codes.

Uses Fernet (AES-128-CBC) from the cryptography package for symmetric
encryption so the join code can be decrypted for display-back to the user.

The encryption key is derived from JOIN_CODE_ENCRYPTION_KEY env var.
Falls back to a deterministic key from SECRET_KEY if not set.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _get_fernet_key() -> bytes:
    """
    Derive a 32-byte Fernet key from environment.
    
    Uses JOIN_CODE_ENCRYPTION_KEY if set, otherwise derives from SECRET_KEY.
    """
    key_source = os.getenv("JOIN_CODE_ENCRYPTION_KEY") or os.getenv("SECRET_KEY", "researchpulse-default-key")
    # Fernet requires a url-safe base64-encoded 32-byte key
    raw = hashlib.sha256(key_source.encode()).digest()
    return base64.urlsafe_b64encode(raw)


def encrypt_join_code(plaintext: str) -> str:
    """Encrypt a join code for storage. Returns base64 string."""
    try:
        from cryptography.fernet import Fernet
        f = Fernet(_get_fernet_key())
        return f.encrypt(plaintext.encode()).decode()
    except ImportError:
        # Fallback: simple base64 encoding (not secure, but functional)
        logger.warning("cryptography package not installed - using base64 fallback")
        return base64.urlsafe_b64encode(plaintext.encode()).decode()


def decrypt_join_code(encrypted: str) -> Optional[str]:
    """Decrypt a join code for display. Returns plaintext or None on failure."""
    if not encrypted:
        return None
    try:
        from cryptography.fernet import Fernet
        f = Fernet(_get_fernet_key())
        return f.decrypt(encrypted.encode()).decode()
    except ImportError:
        # Fallback: simple base64 decoding
        try:
            return base64.urlsafe_b64decode(encrypted.encode()).decode()
        except Exception:
            return None
    except Exception as e:
        logger.warning("Failed to decrypt join code: %s", e)
        return None


def verify_join_code(plaintext: str, encrypted: str) -> bool:
    """Verify a plaintext code against an encrypted value."""
    decrypted = decrypt_join_code(encrypted)
    if decrypted is not None:
        return decrypted == plaintext
    return False
