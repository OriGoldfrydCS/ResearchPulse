"""
Reply Intent Parser for ResearchPulse.

Parses email replies to calendar invitations to extract user intent:
- Reschedule requests with new date/time
- Confirmation/acceptance
- Cancellation requests
- Other feedback

Uses a combination of rule-based parsing and LLM-based interpretation.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum


class ReplyIntent(str, Enum):
    """Possible intents extracted from email replies."""
    RESCHEDULE = "reschedule"
    ACCEPT = "accept"
    DECLINE = "decline"
    CANCEL = "cancel"
    QUESTION = "question"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass
class ParsedReply:
    """Result of parsing an email reply."""
    intent: ReplyIntent
    extracted_datetime: Optional[datetime]
    extracted_datetime_text: Optional[str]  # Original text that was parsed
    confidence_score: float  # 0.0 to 1.0
    reason: str  # Explanation of the parsing result
    original_text: str


# Common date/time patterns for rule-based parsing
DATE_PATTERNS = [
    # "tomorrow at 2pm"
    r"tomorrow\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
    # "next Monday at 3pm"
    r"next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
    # "February 5 at 10am"
    r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
    # "2/5 at 10am" or "2-5 at 10am"
    r"(\d{1,2})[/\-](\d{1,2})(?:[/\-](\d{2,4}))?\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
    # "on the 5th at 2pm"
    r"on\s+the\s+(\d{1,2})(?:st|nd|rd|th)?\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
    # "at 2pm tomorrow"
    r"at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s+tomorrow",
    # "2pm on Monday"
    r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s+on\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
]


def strip_email_quotes(body: str) -> str:
    """Strip quoted previous messages and signatures from email reply body.

    Keeps only the user's new text at the top of the reply.
    Handles common quote markers: lines starting with '>', 'On ... wrote:',
    '---', '___', and common signature delimiters ('--').
    """
    lines = body.splitlines()
    clean_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Stop at quoted-reply headers ("On Feb 10, 2026, ... wrote:")
        if re.match(r"^On .+ wrote:\s*$", stripped, re.IGNORECASE):
            break

        # Stop at Gmail-style forwarded content
        if stripped.startswith("---------- Forwarded message"):
            break

        # Stop at signature delimiter
        if stripped in ("--", "---", "___", "— "):
            break

        # Stop at "From:" line (Outlook-style quoted headers)
        if re.match(r"^From:\s+", stripped, re.IGNORECASE):
            break

        # Skip lines that are purely quote-prefixed
        if stripped.startswith(">"):
            continue

        clean_lines.append(line)

    return "\n".join(clean_lines).strip()


def extract_reschedule_datetime_text(email_body_text: str) -> Optional[str]:
    """Extract the raw datetime text from a reschedule reply email body.

    Strips quoted content and signatures first, then looks for reschedule
    phrases and datetime expressions.

    Returns the matched datetime substring or None.
    """
    cleaned = strip_email_quotes(email_body_text)
    if not cleaned:
        return None

    # Look for reschedule-like phrases followed by a datetime expression
    # e.g. "reschedule to February 11, 2026 at 10:00 AM"
    reschedule_pattern = re.search(
        r"(?:reschedule|move|change|push|postpone)\s+(?:it\s+)?(?:to\s+)?"
        r"((?:january|february|march|april|may|june|july|august|september|october|november|december|"
        r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|tomorrow|next\s+\w+)"
        r"[^.\n]*)",
        cleaned, re.IGNORECASE,
    )
    if reschedule_pattern:
        return reschedule_pattern.group(1).strip()

    # Fallback: look for any date-like expression
    date_pattern = re.search(
        r"((?:january|february|march|april|may|june|july|august|september|october|november|december|"
        r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
        r"\s+\d{1,2}[^.\n]*(?:am|pm))",
        cleaned, re.IGNORECASE,
    )
    if date_pattern:
        return date_pattern.group(1).strip()

    return None

# Keywords indicating reschedule intent
RESCHEDULE_KEYWORDS = [
    "reschedule", "move", "change", "push", "postpone", "delay",
    "different time", "another time", "new time", "doesn't work",
    "can't make it", "won't work", "not available", "busy then",
    "conflict", "please change", "how about", "what about",
    "can we do", "instead", "rather", "prefer",
]

# Keywords indicating acceptance
ACCEPT_KEYWORDS = [
    "accept", "confirm", "looks good", "works for me", "perfect",
    "sounds good", "great", "thanks", "see you", "i'll be there",
]

# Keywords indicating decline/cancel
DECLINE_KEYWORDS = [
    "decline", "cancel", "remove", "delete", "don't need",
    "no longer", "forget it", "never mind", "skip",
]


def parse_reply_rules(email_text: str) -> ParsedReply:
    """
    Parse email reply using rule-based approach.
    
    Args:
        email_text: The email body text to parse
        
    Returns:
        ParsedReply with extracted intent and datetime
    """
    text_lower = email_text.lower().strip()
    
    # Check for empty or very short text
    if len(text_lower) < 5:
        return ParsedReply(
            intent=ReplyIntent.UNKNOWN,
            extracted_datetime=None,
            extracted_datetime_text=None,
            confidence_score=0.0,
            reason="Email text too short to parse",
            original_text=email_text,
        )
    
    # Check for decline/cancel keywords first
    for keyword in DECLINE_KEYWORDS:
        if keyword in text_lower:
            return ParsedReply(
                intent=ReplyIntent.CANCEL,
                extracted_datetime=None,
                extracted_datetime_text=None,
                confidence_score=0.7,
                reason=f"Found cancel keyword: '{keyword}'",
                original_text=email_text,
            )
    
    # Check for reschedule keywords
    has_reschedule_keyword = any(kw in text_lower for kw in RESCHEDULE_KEYWORDS)
    
    # Try to extract date/time
    extracted_dt, dt_text = _extract_datetime_from_text(text_lower)
    
    if extracted_dt:
        confidence = 0.85 if has_reschedule_keyword else 0.6
        return ParsedReply(
            intent=ReplyIntent.RESCHEDULE,
            extracted_datetime=extracted_dt,
            extracted_datetime_text=dt_text,
            confidence_score=confidence,
            reason=f"Extracted datetime '{dt_text}' from email",
            original_text=email_text,
        )
    
    # Check for accept keywords
    for keyword in ACCEPT_KEYWORDS:
        if keyword in text_lower:
            return ParsedReply(
                intent=ReplyIntent.ACCEPT,
                extracted_datetime=None,
                extracted_datetime_text=None,
                confidence_score=0.7,
                reason=f"Found accept keyword: '{keyword}'",
                original_text=email_text,
            )
    
    # If reschedule keywords but no datetime
    if has_reschedule_keyword:
        return ParsedReply(
            intent=ReplyIntent.RESCHEDULE,
            extracted_datetime=None,
            extracted_datetime_text=None,
            confidence_score=0.4,
            reason="Reschedule intent detected but no specific datetime found",
            original_text=email_text,
        )
    
    # Question detection
    if "?" in text_lower:
        return ParsedReply(
            intent=ReplyIntent.QUESTION,
            extracted_datetime=None,
            extracted_datetime_text=None,
            confidence_score=0.5,
            reason="Email appears to be a question",
            original_text=email_text,
        )
    
    return ParsedReply(
        intent=ReplyIntent.UNKNOWN,
        extracted_datetime=None,
        extracted_datetime_text=None,
        confidence_score=0.2,
        reason="Could not determine intent from email text",
        original_text=email_text,
    )


def _extract_datetime_from_text(text: str) -> Tuple[Optional[datetime], Optional[str]]:
    """
    Extract datetime from natural language text.
    
    Returns (datetime, original_text_matched) or (None, None) if not found.
    """
    now = datetime.now()
    
    # "tomorrow at Xpm"
    match = re.search(r"tomorrow\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text, re.IGNORECASE)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        period = match.group(3)
        hour = _convert_to_24h(hour, period)
        
        tomorrow = now + timedelta(days=1)
        dt = tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return dt, match.group(0)
    
    # "next [weekday] at Xpm"
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    match = re.search(
        r"next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
        text, re.IGNORECASE
    )
    if match:
        target_day = weekdays.index(match.group(1).lower())
        hour = int(match.group(2))
        minute = int(match.group(3)) if match.group(3) else 0
        period = match.group(4)
        hour = _convert_to_24h(hour, period)
        
        days_ahead = target_day - now.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        dt = (now + timedelta(days=days_ahead)).replace(hour=hour, minute=minute, second=0, microsecond=0)
        return dt, match.group(0)
    
    # "[Month] [day][,] [year] at X:XX am/pm"  (e.g. "February 11, 2026 at 10:00 AM")
    months = ["january", "february", "march", "april", "may", "june",
              "july", "august", "september", "october", "november", "december"]
    # Also accept 3-letter abbreviations
    month_abbrevs = ["jan", "feb", "mar", "apr", "may", "jun",
                     "jul", "aug", "sep", "oct", "nov", "dec"]

    # Try with optional comma and optional year first (most specific)
    match = re.search(
        r"(january|february|march|april|may|june|july|august|september|october|november|december|"
        r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
        r"\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
        text, re.IGNORECASE
    )
    if match:
        month_str = match.group(1).lower()
        if month_str in months:
            month = months.index(month_str) + 1
        else:
            month = month_abbrevs.index(month_str) + 1
        day = int(match.group(2))
        year = int(match.group(3)) if match.group(3) else now.year
        hour = int(match.group(4))
        minute = int(match.group(5)) if match.group(5) else 0
        period = match.group(6)
        hour = _convert_to_24h(hour, period)
        
        # If the date has passed this year and no explicit year, assume next year
        try:
            dt = datetime(year, month, day, hour, minute)
            if dt < now and not match.group(3):
                dt = datetime(year + 1, month, day, hour, minute)
            return dt, match.group(0)
        except ValueError:
            pass
    
    # "M/D at Xpm" or "M-D at Xpm"
    match = re.search(
        r"(\d{1,2})[/\-](\d{1,2})(?:[/\-](\d{2,4}))?\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
        text, re.IGNORECASE
    )
    if match:
        month = int(match.group(1))
        day = int(match.group(2))
        year = int(match.group(3)) if match.group(3) else now.year
        if year < 100:
            year += 2000
        hour = int(match.group(4))
        minute = int(match.group(5)) if match.group(5) else 0
        period = match.group(6)
        hour = _convert_to_24h(hour, period)
        
        try:
            dt = datetime(year, month, day, hour, minute)
            return dt, match.group(0)
        except ValueError:
            pass
    
    # Simple time extraction "at Xpm" (assume today or tomorrow)
    match = re.search(r"at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)", text, re.IGNORECASE)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        period = match.group(3)
        hour = _convert_to_24h(hour, period)
        
        dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if dt <= now:
            dt += timedelta(days=1)  # Assume tomorrow if time has passed
        return dt, match.group(0)
    
    return None, None


def _convert_to_24h(hour: int, period: Optional[str]) -> int:
    """Convert 12-hour time to 24-hour format."""
    if period is None:
        return hour
    period = period.lower()
    if period == "pm" and hour != 12:
        return hour + 12
    if period == "am" and hour == 12:
        return 0
    return hour


async def parse_reply_llm(email_text: str, llm_client: Any = None) -> ParsedReply:
    """
    Parse email reply using LLM for more complex understanding.
    
    Uses the LLM to extract intent and datetime when rule-based parsing
    has low confidence.
    
    Args:
        email_text: The email body text to parse
        llm_client: Optional LLM client (defaults to OpenAI)
        
    Returns:
        ParsedReply with extracted intent and datetime
    """
    # First try rule-based parsing
    rule_result = parse_reply_rules(email_text)
    
    # If high confidence, return rule-based result
    if rule_result.confidence_score >= 0.7:
        return rule_result
    
    # Try LLM parsing for low-confidence results
    try:
        import openai
        
        client = llm_client or openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "")
        )
        
        if not os.getenv("OPENAI_API_KEY"):
            # Fallback to rule-based if no API key
            return rule_result
        
        prompt = f"""Analyze this email reply to a calendar invitation and extract the user's intent.

Email text:
---
{email_text}
---

Determine:
1. Intent: One of [reschedule, accept, decline, cancel, question, other]
2. If reschedule: Extract the proposed new date and time
3. Confidence: 0.0 to 1.0

Response format (JSON):
{{
    "intent": "reschedule|accept|decline|cancel|question|other",
    "proposed_datetime": "YYYY-MM-DD HH:MM" or null,
    "datetime_text": "original text mentioning the time" or null,
    "confidence": 0.8,
    "reason": "brief explanation"
}}

Today's date is: {datetime.now().strftime('%Y-%m-%d')}"""

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=200,
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        # Parse the LLM response
        intent = ReplyIntent(result.get("intent", "unknown"))
        confidence = float(result.get("confidence", 0.5))
        
        extracted_dt = None
        if result.get("proposed_datetime"):
            try:
                extracted_dt = datetime.strptime(result["proposed_datetime"], "%Y-%m-%d %H:%M")
            except ValueError:
                pass
        
        return ParsedReply(
            intent=intent,
            extracted_datetime=extracted_dt,
            extracted_datetime_text=result.get("datetime_text"),
            confidence_score=confidence,
            reason=result.get("reason", "LLM analysis"),
            original_text=email_text,
        )
        
    except Exception as e:
        # Log error and return rule-based result
        print(f"[ReplyParser] LLM parsing failed: {e}")
        return rule_result


def parse_reply(email_text: str, use_llm: bool = False) -> ParsedReply:
    """
    Synchronous wrapper for reply parsing.
    
    Args:
        email_text: The email body text to parse
        use_llm: Whether to use LLM for complex parsing
        
    Returns:
        ParsedReply with extracted intent and datetime
    """
    if use_llm:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(parse_reply_llm(email_text))
    else:
        return parse_reply_rules(email_text)


# =============================================================================
# Testing/Debug Functions
# =============================================================================

def test_parser():
    """Test the reply parser with common examples."""
    test_cases = [
        "This time doesn't work. Can we do tomorrow at 2pm instead?",
        "Please reschedule to February 10 at 3pm",
        "I'll be there!",
        "Cancel this event please",
        "How about next Monday at 10am?",
        "Can't make it, move to 2/15 at 4pm",
        "Thanks, looks good!",
        "What papers are included?",
        "No longer needed",
    ]
    
    print("=" * 60)
    print("Reply Parser Test Results")
    print("=" * 60)
    
    for text in test_cases:
        result = parse_reply_rules(text)
        print(f"\nInput: {text}")
        print(f"  Intent: {result.intent.value}")
        print(f"  DateTime: {result.extracted_datetime}")
        print(f"  DT Text: {result.extracted_datetime_text}")
        print(f"  Confidence: {result.confidence_score:.2f}")
        print(f"  Reason: {result.reason}")


if __name__ == "__main__":
    test_parser()
