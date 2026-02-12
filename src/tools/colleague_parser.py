"""
Colleague onboarding parser for ResearchPulse.

Robust parsing of structured colleague sign-up emails.
Handles many common formats for the fields: Code, Name, Research interests.

Parsing rules (per spec):
- Code: / Invite code: / CODE= (case-insensitive, whitespace-tolerant)
- Name: optional, extracted from field or email headers
- Interests: comma-separated, newline-separated, or prose
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedSignup:
    """Result of parsing a colleague sign-up email."""
    code: Optional[str] = None
    name: Optional[str] = None
    interests: List[str] = field(default_factory=list)
    raw_interests: str = ""
    parse_success: bool = False
    parse_errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

_CODE_PATTERNS = [
    # "Code: ABC123",  "code :ABC123",  "invite code: ABC123"
    re.compile(r'(?:invite\s*)?code\s*[:=]\s*([A-Za-z0-9\-_]{3,64})', re.IGNORECASE),
    # "CODE=ABC123"
    re.compile(r'CODE\s*=\s*([A-Za-z0-9\-_]{3,64})', re.IGNORECASE),
    # "#ABC123"
    re.compile(r'#([A-Za-z0-9\-_]{4,32})'),
    # "my code is ABC123"
    re.compile(r'(?:my|the)\s+code\s+(?:is\s+)?([A-Za-z0-9\-_]{4,32})', re.IGNORECASE),
    # "use code ABC123"
    re.compile(r'use\s+code\s*[:=]?\s*([A-Za-z0-9\-_]{4,32})', re.IGNORECASE),
]

_FALSE_POSITIVE_CODES = {"the", "is", "code", "join", "my", "use", "invite", "none", "null"}


def extract_code(text: str) -> Optional[str]:
    """Extract an invite code from text (subject + body combined)."""
    for pat in _CODE_PATTERNS:
        m = pat.search(text)
        if m:
            candidate = m.group(1).strip()
            if candidate.lower() not in _FALSE_POSITIVE_CODES:
                return candidate
    return None


# ---------------------------------------------------------------------------
# Name extraction
# ---------------------------------------------------------------------------

_NAME_PATTERNS = [
    re.compile(r'(?:^|\n)\s*name\s*[:=]\s*(.+)', re.IGNORECASE),
    re.compile(r"(?:my name is|i'm|i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})", re.IGNORECASE),
]


def extract_name(text: str, from_name: str = "") -> Optional[str]:
    """Extract colleague name from text, falling back to email header name."""
    for pat in _NAME_PATTERNS:
        m = pat.search(text)
        if m:
            name = m.group(1).strip().strip(".,;:!?")
            if 1 < len(name) <= 80:
                return name
    if from_name and from_name.strip():
        return from_name.strip()
    return None


# ---------------------------------------------------------------------------
# Interest extraction
# ---------------------------------------------------------------------------

_INTEREST_PATTERNS = [
    re.compile(
        r'(?:^|\n)\s*(?:research\s+)?interests?\s*[:=]\s*(.+?)(?:\n\s*\n|\n\s*(?:code|name|action)\s*[:=]|$)',
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r'(?:^|\n)\s*(?:topics?|areas?)\s*[:=]\s*(.+?)(?:\n\s*\n|\n\s*(?:code|name|action)\s*[:=]|$)',
        re.IGNORECASE | re.DOTALL,
    ),
]


def extract_interests(text: str) -> tuple[str, list[str]]:
    """Extract research interests from text.

    Returns (raw_text, parsed_list).
    """
    for pat in _INTEREST_PATTERNS:
        m = pat.search(text)
        if m:
            raw = m.group(1).strip()
            parsed = _split_interests(raw)
            if parsed:
                return raw, parsed

    return "", []


def _split_interests(raw: str) -> list[str]:
    """Split a raw interest string into individual topics."""
    # Try comma-separated first
    if "," in raw:
        items = [i.strip().strip("-•*") for i in raw.split(",")]
    elif "\n" in raw:
        items = [i.strip().strip("-•*") for i in raw.split("\n")]
    else:
        items = [raw.strip()]

    # Filter blanks and too-short items
    return [i for i in items if i and len(i) >= 2]


# ---------------------------------------------------------------------------
# Full parse
# ---------------------------------------------------------------------------

def parse_signup_email(
    subject: str,
    body: str,
    from_name: str = "",
) -> ParsedSignup:
    """Parse a colleague sign-up email.

    Combines subject and body for code extraction,
    uses body for name and interests extraction.

    Returns a ParsedSignup with parse_success=True only if a code
    was found AND at least one interest was extracted.
    """
    combined = f"{subject}\n{body}"
    result = ParsedSignup()

    # 1. Code
    result.code = extract_code(combined)
    if not result.code:
        result.parse_errors.append("No invite code found")

    # 2. Name
    result.name = extract_name(body, from_name)

    # 3. Interests
    result.raw_interests, result.interests = extract_interests(body)
    if not result.interests:
        # Try the subject too (unlikely, but spec says tolerate extra text)
        result.raw_interests, result.interests = extract_interests(combined)

    if not result.interests:
        result.parse_errors.append("No research interests found")

    result.parse_success = bool(result.code) and bool(result.interests)
    return result
