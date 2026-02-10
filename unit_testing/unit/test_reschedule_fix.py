"""
Regression tests for the reschedule pipeline fix.

Validates:
1. ParsedReply has extracted_datetime_text (not raw_datetime_text)
2. Datetime extraction handles "Month Day, Year at H:MM AM/PM" format
3. strip_email_quotes correctly strips quoted messages
4. extract_reschedule_datetime_text returns the datetime substring
5. extract_reschedule_datetime (inbound_processor) returns correct values
6. Abbreviated month names work (Feb, Mar, etc.)
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.agent.reply_parser import (
    ParsedReply,
    ReplyIntent,
    parse_reply,
    parse_reply_rules,
    strip_email_quotes,
    extract_reschedule_datetime_text,
    _extract_datetime_from_text,
)


# ============================================================================
# 1. ParsedReply field existence
# ============================================================================

class TestParsedReplyFieldName:
    """Verify ParsedReply uses extracted_datetime_text, not raw_datetime_text."""

    def test_has_extracted_datetime_text_field(self):
        reply = ParsedReply(
            intent=ReplyIntent.RESCHEDULE,
            extracted_datetime=datetime(2026, 2, 11, 10, 0),
            extracted_datetime_text="February 11, 2026 at 10:00 AM",
            confidence_score=0.85,
            reason="test",
            original_text="reschedule to February 11, 2026 at 10:00 AM",
        )
        assert reply.extracted_datetime_text == "February 11, 2026 at 10:00 AM"

    def test_no_raw_datetime_text_field(self):
        reply = ParsedReply(
            intent=ReplyIntent.RESCHEDULE,
            extracted_datetime=None,
            extracted_datetime_text=None,
            confidence_score=0.5,
            reason="test",
            original_text="test",
        )
        assert not hasattr(reply, "raw_datetime_text")


# ============================================================================
# 2. Datetime extraction with year format
# ============================================================================

class TestDatetimeExtractionWithYear:
    """Test patterns like 'February 11, 2026 at 10:00 AM'."""

    def test_month_day_year_at_time(self):
        """The exact repro scenario from the bug report."""
        result = parse_reply_rules("reschedule to February 11, 2026 at 10:00 AM")
        assert result.intent == ReplyIntent.RESCHEDULE
        assert result.extracted_datetime is not None
        assert result.extracted_datetime.year == 2026
        assert result.extracted_datetime.month == 2
        assert result.extracted_datetime.day == 11
        assert result.extracted_datetime.hour == 10
        assert result.extracted_datetime.minute == 0

    def test_month_day_no_year(self):
        """February 15 at 3pm (no year)."""
        result = parse_reply_rules("move to February 15 at 3pm")
        assert result.intent == ReplyIntent.RESCHEDULE
        assert result.extracted_datetime is not None
        assert result.extracted_datetime.month == 2
        assert result.extracted_datetime.day == 15
        assert result.extracted_datetime.hour == 15

    def test_abbreviated_month(self):
        """Feb 11 10am (abbreviated month, no 'at')."""
        result = parse_reply_rules("reschedule it to Feb 11 10am")
        assert result.intent == ReplyIntent.RESCHEDULE
        assert result.extracted_datetime is not None
        assert result.extracted_datetime.month == 2
        assert result.extracted_datetime.day == 11
        assert result.extracted_datetime.hour == 10

    def test_month_day_comma_year_pm(self):
        """March 5, 2026 at 2:30 PM."""
        result = parse_reply_rules("reschedule to March 5, 2026 at 2:30 PM")
        assert result.intent == ReplyIntent.RESCHEDULE
        assert result.extracted_datetime is not None
        assert result.extracted_datetime.month == 3
        assert result.extracted_datetime.day == 5
        assert result.extracted_datetime.hour == 14
        assert result.extracted_datetime.minute == 30

    def test_numeric_date_with_year(self):
        """11/02/2026 at 10:00 — numeric M/D/YYYY."""
        result = parse_reply_rules("move to 11/02/2026 at 10:00")
        assert result.intent == ReplyIntent.RESCHEDULE
        assert result.extracted_datetime is not None


# ============================================================================
# 3. strip_email_quotes
# ============================================================================

class TestStripEmailQuotes:
    """Test that quoted content and signatures are removed."""

    def test_strips_on_wrote_block(self):
        body = (
            "Reschedule to February 11, 2026 at 10:00 AM\n"
            "\n"
            "On Mon, Feb 10, 2026, ResearchPulse wrote:\n"
            "> Your reading reminder is scheduled for ...\n"
        )
        cleaned = strip_email_quotes(body)
        assert "Reschedule to February 11" in cleaned
        assert "ResearchPulse wrote" not in cleaned
        assert ">" not in cleaned

    def test_strips_signature(self):
        body = (
            "Please move to tomorrow at 3pm\n"
            "--\n"
            "Best regards,\n"
            "User\n"
        )
        cleaned = strip_email_quotes(body)
        assert "move to tomorrow" in cleaned
        assert "Best regards" not in cleaned

    def test_no_quotes(self):
        body = "reschedule to Feb 11 10am"
        cleaned = strip_email_quotes(body)
        assert cleaned == body

    def test_empty_body(self):
        assert strip_email_quotes("") == ""

    def test_strips_outlook_from(self):
        body = (
            "Move to March 1 at 2pm\n"
            "\n"
            "From: ResearchPulse <noreply@researchpulse.com>\n"
            "Sent: Monday, February 10, 2026 5:00 PM\n"
        )
        cleaned = strip_email_quotes(body)
        assert "Move to March 1" in cleaned
        assert "From:" not in cleaned


# ============================================================================
# 4. extract_reschedule_datetime_text
# ============================================================================

class TestExtractRescheduleDatetimeText:
    """Test the helper that pulls the datetime substring out of the reply."""

    def test_basic_reschedule_phrase(self):
        text = extract_reschedule_datetime_text(
            "reschedule to February 11, 2026 at 10:00 AM"
        )
        assert text is not None
        assert "February 11" in text

    def test_move_phrase(self):
        text = extract_reschedule_datetime_text("move it to tomorrow at 3pm")
        assert text is not None
        assert "tomorrow" in text.lower()

    def test_no_datetime(self):
        text = extract_reschedule_datetime_text("I need to reschedule please")
        # Should return None since there's no datetime expression
        assert text is None

    def test_body_with_quotes(self):
        body = (
            "Reschedule to March 1 at 2pm\n"
            "\n"
            "On Mon, Feb 10, 2026, ResearchPulse wrote:\n"
            "> reminder for papers\n"
        )
        text = extract_reschedule_datetime_text(body)
        assert text is not None
        assert "March 1" in text


# ============================================================================
# 5. extract_reschedule_datetime integration (inbound_processor)
# ============================================================================

class TestExtractRescheduleDatetime:
    """Test the inbound_processor.extract_reschedule_datetime function."""

    def test_basic_reschedule_returns_datetime(self):
        from src.tools.inbound_processor import extract_reschedule_datetime

        dt, text = extract_reschedule_datetime(
            "reschedule to February 11, 2026 at 10:00 AM"
        )
        assert dt is not None
        assert dt.year == 2026
        assert dt.month == 2
        assert dt.day == 11
        assert dt.hour == 10
        assert text is not None

    def test_no_datetime_returns_none(self):
        from src.tools.inbound_processor import extract_reschedule_datetime

        dt, text = extract_reschedule_datetime("thanks, that works!")
        assert dt is None

    def test_with_quoted_body(self):
        from src.tools.inbound_processor import extract_reschedule_datetime

        body = (
            "reschedule to Feb 15 at 3pm\n"
            "\n"
            "On Feb 10, 2026 ResearchPulse wrote:\n"
            "> You have a reading reminder\n"
        )
        dt, text = extract_reschedule_datetime(body)
        assert dt is not None
        assert dt.month == 2
        assert dt.day == 15
        assert dt.hour == 15


# ============================================================================
# 6. End-to-end ParsedReply round-trip
# ============================================================================

class TestParsedReplyRoundTrip:
    """Parse a sample email body and verify all fields are consistent."""

    def test_full_round_trip(self):
        body = "Reschedule to February 11, 2026 at 10:00 AM"
        result = parse_reply(body, use_llm=False)

        assert isinstance(result, ParsedReply)
        assert result.intent == ReplyIntent.RESCHEDULE
        assert result.extracted_datetime is not None
        assert result.extracted_datetime_text is not None
        assert result.confidence_score > 0.5
        assert result.original_text == body

    def test_field_accessed_by_correct_name(self):
        """Regression: ensure accessing extracted_datetime_text works, raw_datetime_text doesn't."""
        result = parse_reply("move to tomorrow at 5pm", use_llm=False)
        # This must NOT raise AttributeError
        _ = result.extracted_datetime_text
        # This must raise AttributeError
        with pytest.raises(AttributeError):
            _ = result.raw_datetime_text
