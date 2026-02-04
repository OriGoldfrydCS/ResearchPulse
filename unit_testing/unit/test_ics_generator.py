"""
Unit tests for the ICS (iCalendar) generator.

Tests RFC 5545 compliant calendar file generation for:
- Reading reminder events
- Reschedule events
- Cancel events
- ICS content validation
"""

import pytest
from datetime import datetime, timedelta

from src.tools.ics_generator import (
    generate_uid,
    generate_ics_content,
    generate_reading_reminder_ics,
    generate_reschedule_ics,
    generate_cancel_ics,
    validate_ics_content,
    escape_ics_text,
    fold_ics_line,
    format_datetime_utc,
    ICSEvent,
)
from unit_testing.conftest import get_test_email, get_colleague_test_email


class TestGenerateUID:
    """Test UID generation."""
    
    def test_generate_uid_format(self):
        """Test that UID has correct format."""
        uid = generate_uid()
        
        assert "@researchpulse.app" in uid
        assert len(uid) > 20  # UUID + domain
    
    def test_generate_uid_uniqueness(self):
        """Test that generated UIDs are unique."""
        uids = [generate_uid() for _ in range(100)]
        
        assert len(set(uids)) == 100  # All unique


class TestICSEvent:
    """Test ICSEvent dataclass."""
    
    def test_create_ics_event(self):
        """Test creating an ICSEvent instance."""
        event = ICSEvent(
            uid="test-uid@researchpulse.app",
            title="Test Event",
            description="Test description",
            start_time=datetime(2026, 2, 10, 14, 0),
            duration_minutes=30,
            organizer_email="test@researchpulse.app",
        )
        
        assert event.uid == "test-uid@researchpulse.app"
        assert event.title == "Test Event"
        assert event.duration_minutes == 30
        assert event.organizer_name == "ResearchPulse"  # Default
        assert event.sequence == 0
        assert event.method == "REQUEST"


class TestEscapeICSText:
    """Test ICS text escaping."""
    
    def test_escape_backslash(self):
        """Test backslash escaping."""
        assert escape_ics_text("path\\to\\file") == "path\\\\to\\\\file"
    
    def test_escape_semicolon(self):
        """Test semicolon escaping."""
        assert escape_ics_text("a; b; c") == "a\\; b\\; c"
    
    def test_escape_comma(self):
        """Test comma escaping."""
        assert escape_ics_text("a, b, c") == "a\\, b\\, c"
    
    def test_escape_newlines(self):
        """Test newline escaping."""
        assert escape_ics_text("line1\nline2") == "line1\\nline2"
        assert escape_ics_text("line1\r\nline2") == "line1\\nline2"
    
    def test_escape_empty_string(self):
        """Test empty string handling."""
        assert escape_ics_text("") == ""
    
    def test_escape_combined(self):
        """Test multiple escape sequences."""
        text = "Event, with; special\nchars"
        escaped = escape_ics_text(text)
        
        assert "\\," in escaped
        assert "\\;" in escaped
        assert "\\n" in escaped


class TestFoldICSLine:
    """Test ICS line folding."""
    
    def test_short_line_unchanged(self):
        """Test that short lines are not modified."""
        line = "Short line"
        assert fold_ics_line(line) == line
    
    def test_long_line_folded(self):
        """Test that long lines are folded."""
        line = "A" * 100  # Longer than 75 characters
        folded = fold_ics_line(line)
        
        assert "\r\n " in folded  # CRLF + space continuation
    
    def test_folded_line_continuation(self):
        """Test that continuation lines start with space."""
        line = "X" * 200
        folded = fold_ics_line(line)
        
        parts = folded.split("\r\n")
        for part in parts[1:]:  # All but first should start with space
            assert part.startswith(" ")


class TestFormatDateTimeUTC:
    """Test datetime formatting for ICS."""
    
    def test_format_datetime(self):
        """Test datetime formatting."""
        dt = datetime(2026, 2, 10, 14, 30, 0)
        formatted = format_datetime_utc(dt)
        
        assert formatted == "20260210T143000Z"
    
    def test_format_midnight(self):
        """Test midnight formatting."""
        dt = datetime(2026, 1, 1, 0, 0, 0)
        formatted = format_datetime_utc(dt)
        
        assert formatted == "20260101T000000Z"


class TestGenerateICSContent:
    """Test ICS content generation."""
    
    @pytest.fixture
    def sample_event(self):
        """Create a sample event for testing."""
        return ICSEvent(
            uid="test-123@researchpulse.app",
            title="Read: Attention is All You Need",
            description="Paper reading session",
            start_time=datetime(2026, 2, 10, 14, 0),
            duration_minutes=60,
            organizer_email="agent@researchpulse.app",
            organizer_name="ResearchPulse",
            attendee_email=get_test_email(),
            attendee_name="Test User",
            reminder_minutes=15,
            sequence=0,
            method="REQUEST",
        )
    
    def test_ics_has_required_headers(self, sample_event):
        """Test ICS content has required headers."""
        ics = generate_ics_content(sample_event)
        
        assert "BEGIN:VCALENDAR" in ics
        assert "END:VCALENDAR" in ics
        assert "VERSION:2.0" in ics
        assert "PRODID:-//ResearchPulse//Calendar Reminder//EN" in ics
        assert "METHOD:REQUEST" in ics
    
    def test_ics_has_event(self, sample_event):
        """Test ICS content has event block."""
        ics = generate_ics_content(sample_event)
        
        assert "BEGIN:VEVENT" in ics
        assert "END:VEVENT" in ics
    
    def test_ics_has_uid(self, sample_event):
        """Test ICS content has unique identifier."""
        ics = generate_ics_content(sample_event)
        
        assert "UID:test-123@researchpulse.app" in ics
    
    def test_ics_has_times(self, sample_event):
        """Test ICS content has start and end times."""
        ics = generate_ics_content(sample_event)
        
        assert "DTSTART:" in ics
        assert "DTEND:" in ics
    
    def test_ics_has_organizer(self, sample_event):
        """Test ICS content has organizer."""
        ics = generate_ics_content(sample_event)
        
        assert "ORGANIZER" in ics
        assert "agent@researchpulse.app" in ics
    
    def test_ics_has_attendee(self, sample_event):
        """Test ICS content has attendee."""
        ics = generate_ics_content(sample_event)
        
        assert "ATTENDEE" in ics
        assert get_test_email() in ics
    
    def test_ics_has_alarm(self, sample_event):
        """Test ICS content has reminder alarm."""
        ics = generate_ics_content(sample_event)
        
        assert "BEGIN:VALARM" in ics
        assert "END:VALARM" in ics
        assert "TRIGGER:-PT15M" in ics
    
    def test_ics_uses_crlf(self, sample_event):
        """Test ICS content uses CRLF line endings."""
        ics = generate_ics_content(sample_event)
        
        assert "\r\n" in ics
    
    def test_cancel_method(self):
        """Test ICS with CANCEL method."""
        event = ICSEvent(
            uid="cancel-123@researchpulse.app",
            title="Cancelled Event",
            description="This event is cancelled",
            start_time=datetime(2026, 2, 10, 14, 0),
            duration_minutes=30,
            organizer_email="agent@researchpulse.app",
            method="CANCEL",
        )
        ics = generate_ics_content(event)
        
        assert "METHOD:CANCEL" in ics
        assert "STATUS:CANCELLED" in ics


class TestGenerateReadingReminderICS:
    """Test reading reminder ICS generation."""
    
    @pytest.fixture
    def sample_papers(self):
        """Create sample papers for testing."""
        return [
            {
                "title": "Attention is All You Need",
                "url": "https://arxiv.org/abs/1706.03762",
                "importance": "high",
            },
            {
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "url": "https://arxiv.org/abs/1810.04805",
                "importance": "medium",
            },
        ]
    
    def test_generate_reminder_ics(self, sample_papers):
        """Test generating reading reminder ICS."""
        ics = generate_reading_reminder_ics(
            uid=generate_uid(),
            papers=sample_papers,
            start_time=datetime(2026, 2, 10, 14, 0),
            duration_minutes=60,
            organizer_email="agent@researchpulse.app",
            attendee_email=get_test_email(),
        )
        
        assert "BEGIN:VCALENDAR" in ics
        assert "BEGIN:VEVENT" in ics
        assert "📖 Read 2 research papers" in ics
    
    def test_single_paper_title(self, sample_papers):
        """Test title for single paper."""
        ics = generate_reading_reminder_ics(
            uid=generate_uid(),
            papers=[sample_papers[0]],
            start_time=datetime(2026, 2, 10, 14, 0),
            duration_minutes=30,
            organizer_email="agent@researchpulse.app",
            attendee_email=get_test_email(),
        )
        
        assert "Attention is All You Need" in ics
    
    def test_papers_in_description(self, sample_papers):
        """Test that papers appear in description."""
        ics = generate_reading_reminder_ics(
            uid=generate_uid(),
            papers=sample_papers,
            start_time=datetime(2026, 2, 10, 14, 0),
            duration_minutes=60,
            organizer_email="agent@researchpulse.app",
            attendee_email=get_test_email(),
        )
        
        # Paper titles should be in description (escaped)
        assert "Attention is All You Need" in ics.replace("\\n", "\n")
    
    def test_agent_note_included(self, sample_papers):
        """Test that agent note is included."""
        ics = generate_reading_reminder_ics(
            uid=generate_uid(),
            papers=sample_papers,
            start_time=datetime(2026, 2, 10, 14, 0),
            duration_minutes=60,
            organizer_email="agent@researchpulse.app",
            attendee_email=get_test_email(),
            agent_note="I recommend reading these in order",
        )
        
        assert "I recommend reading these in order" in ics.replace("\\n", "\n")
    
    def test_valid_ics_structure(self, sample_papers):
        """Test that generated ICS passes validation."""
        ics = generate_reading_reminder_ics(
            uid=generate_uid(),
            papers=sample_papers,
            start_time=datetime(2026, 2, 10, 14, 0),
            duration_minutes=60,
            organizer_email="agent@researchpulse.app",
            attendee_email=get_test_email(),
        )
        
        validation = validate_ics_content(ics)
        assert validation["valid"], f"ICS validation failed: {validation['errors']}"


class TestGenerateRescheduleICS:
    """Test reschedule ICS generation."""
    
    def test_generate_reschedule_ics(self):
        """Test generating reschedule ICS."""
        papers = [{"title": "Test Paper", "url": "https://example.com"}]
        
        ics = generate_reschedule_ics(
            uid="original-uid@researchpulse.app",  # Same UID
            papers=papers,
            new_start_time=datetime(2026, 2, 11, 15, 0),
            duration_minutes=30,
            organizer_email="agent@researchpulse.app",
            attendee_email=get_test_email(),
            sequence=1,  # Incremented
            reschedule_reason="User requested new time",
        )
        
        assert "UID:original-uid@researchpulse.app" in ics
        assert "SEQUENCE:1" in ics
        assert "Rescheduled" in ics.replace("\\n", "\n")
    
    def test_reschedule_uses_same_uid(self):
        """Test that reschedule preserves UID."""
        uid = "preserve-this-uid@researchpulse.app"
        papers = [{"title": "Test Paper"}]
        
        ics = generate_reschedule_ics(
            uid=uid,
            papers=papers,
            new_start_time=datetime(2026, 2, 11, 15, 0),
            duration_minutes=30,
            organizer_email="agent@researchpulse.app",
            attendee_email=get_test_email(),
            sequence=2,
        )
        
        assert f"UID:{uid}" in ics


class TestGenerateCancelICS:
    """Test cancel ICS generation."""
    
    def test_generate_cancel_ics(self):
        """Test generating cancel ICS."""
        ics = generate_cancel_ics(
            uid="cancel-me@researchpulse.app",
            title="Read: Paper Title",
            start_time=datetime(2026, 2, 10, 14, 0),
            duration_minutes=30,
            organizer_email="agent@researchpulse.app",
            attendee_email=get_test_email(),
            sequence=1,
            cancel_reason="User no longer needs this reminder",
        )
        
        assert "METHOD:CANCEL" in ics
        assert "STATUS:CANCELLED" in ics
        assert "[CANCELLED]" in ics
    
    def test_cancel_has_no_alarm(self):
        """Test that cancel ICS has no alarm."""
        ics = generate_cancel_ics(
            uid="cancel-me@researchpulse.app",
            title="Read: Paper Title",
            start_time=datetime(2026, 2, 10, 14, 0),
            duration_minutes=30,
            organizer_email="agent@researchpulse.app",
            attendee_email=get_test_email(),
        )
        
        # Should not have VALARM block
        assert "BEGIN:VALARM" not in ics


class TestValidateICSContent:
    """Test ICS validation function."""
    
    def test_valid_ics(self):
        """Test validation of valid ICS content."""
        valid_ics = """BEGIN:VCALENDAR\r
VERSION:2.0\r
BEGIN:VEVENT\r
UID:test@researchpulse.app\r
DTSTART:20260210T140000Z\r
DTEND:20260210T150000Z\r
END:VEVENT\r
END:VCALENDAR\r
"""
        result = validate_ics_content(valid_ics)
        
        assert result["valid"]
        assert len(result["errors"]) == 0
    
    def test_missing_vcalendar(self):
        """Test validation catches missing VCALENDAR."""
        invalid_ics = "BEGIN:VEVENT\r\nEND:VEVENT\r\n"
        result = validate_ics_content(invalid_ics)
        
        assert not result["valid"]
        assert any("VCALENDAR" in err for err in result["errors"])
    
    def test_missing_vevent(self):
        """Test validation catches missing VEVENT."""
        invalid_ics = "BEGIN:VCALENDAR\r\nVERSION:2.0\r\nEND:VCALENDAR\r\n"
        result = validate_ics_content(invalid_ics)
        
        assert not result["valid"]
        assert any("VEVENT" in err for err in result["errors"])
    
    def test_missing_uid(self):
        """Test validation catches missing UID."""
        invalid_ics = """BEGIN:VCALENDAR\r
VERSION:2.0\r
BEGIN:VEVENT\r
DTSTART:20260210T140000Z\r
DTEND:20260210T150000Z\r
END:VEVENT\r
END:VCALENDAR\r
"""
        result = validate_ics_content(invalid_ics)
        
        assert not result["valid"]
        assert any("UID" in err for err in result["errors"])
    
    def test_missing_crlf_warning(self):
        """Test validation warns about non-CRLF line endings."""
        ics_with_lf = "BEGIN:VCALENDAR\nVERSION:2.0\nUID:test\nDTSTART:20260210T140000Z\nDTEND:20260210T150000Z\nBEGIN:VEVENT\nEND:VEVENT\nEND:VCALENDAR\n"
        result = validate_ics_content(ics_with_lf)
        
        assert any("CRLF" in err for err in result["errors"])


# Negative tests
class TestICSNegativeCases:
    """Test error handling and edge cases."""
    
    def test_empty_papers_list(self):
        """Test handling empty papers list."""
        ics = generate_reading_reminder_ics(
            uid=generate_uid(),
            papers=[],
            start_time=datetime(2026, 2, 10, 14, 0),
            duration_minutes=30,
            organizer_email="agent@researchpulse.app",
            attendee_email="user@example.com",
        )
        
        # Should still generate valid ICS
        validation = validate_ics_content(ics)
        assert validation["valid"]
    
    def test_special_characters_in_title(self):
        """Test handling special characters in paper title."""
        papers = [{"title": "Title with, semicolon; and\nnewline", "url": ""}]
        
        ics = generate_reading_reminder_ics(
            uid=generate_uid(),
            papers=papers,
            start_time=datetime(2026, 2, 10, 14, 0),
            duration_minutes=30,
            organizer_email="agent@researchpulse.app",
            attendee_email="user@example.com",
        )
        
        # Should properly escape special characters
        validation = validate_ics_content(ics)
        assert validation["valid"]
    
    def test_unicode_in_description(self):
        """Test handling Unicode characters."""
        papers = [{"title": "论文标题 📚", "url": ""}]  # Chinese + emoji
        
        ics = generate_reading_reminder_ics(
            uid=generate_uid(),
            papers=papers,
            start_time=datetime(2026, 2, 10, 14, 0),
            duration_minutes=30,
            organizer_email="agent@researchpulse.app",
            attendee_email="user@example.com",
        )
        
        # Should handle Unicode
        assert isinstance(ics, str)
    
    def test_very_long_paper_list(self):
        """Test handling many papers."""
        papers = [
            {"title": f"Paper {i}", "url": f"https://example.com/{i}"}
            for i in range(50)
        ]
        
        ics = generate_reading_reminder_ics(
            uid=generate_uid(),
            papers=papers,
            start_time=datetime(2026, 2, 10, 14, 0),
            duration_minutes=300,
            organizer_email="agent@researchpulse.app",
            attendee_email="user@example.com",
        )
        
        validation = validate_ics_content(ics)
        assert validation["valid"]
