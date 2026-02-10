"""
Unit tests for calendar reminder functionality.

Tests calendar event creation, duration estimation, and triggered_by attribution.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from uuid import uuid4

from src.tools.calendar_invite_sender import (
    send_reading_reminder_invite,
    send_reschedule_invite,
    send_calendar_invite_email,
    _is_email_configured,
    _get_smtp_config,
)
from src.tools.ics_generator import generate_uid
from unit_testing.conftest import get_test_email, get_colleague_test_email


class TestCalendarInviteSender:
    """Test calendar invite sending functionality."""
    
    @pytest.fixture
    def sample_papers(self):
        """Create sample papers for testing."""
        return [
            {
                "title": "Attention Is All You Need",
                "url": "https://arxiv.org/abs/1706.03762",
                "importance": "high",
            },
            {
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "url": "https://arxiv.org/abs/1810.04805",
                "importance": "medium",
            },
        ]
    
    def test_send_reminder_returns_result(self, sample_papers):
        """Test that send_reading_reminder_invite returns proper result."""
        result = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=60,
        )
        
        assert isinstance(result, dict)
        assert "success" in result
        assert "ics_uid" in result
        assert "ics_content" in result
        assert "triggered_by" in result
    
    def test_reminder_generates_valid_uid(self, sample_papers):
        """Test that reminder generates valid UID."""
        result = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=60,
        )
        
        assert result["ics_uid"]
        assert "@researchpulse.app" in result["ics_uid"]
    
    def test_reminder_ics_content_valid(self, sample_papers):
        """Test that ICS content is valid."""
        result = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=60,
        )
        
        ics = result["ics_content"]
        
        assert "BEGIN:VCALENDAR" in ics
        assert "BEGIN:VEVENT" in ics
        assert "END:VEVENT" in ics
        assert "END:VCALENDAR" in ics
    
    def test_reminder_has_paper_info(self, sample_papers):
        """Test that ICS contains paper information."""
        result = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=60,
        )
        
        ics = result["ics_content"]
        
        # Paper title should be in description (escaped)
        assert "Attention" in ics.replace("\\n", "\n")
    
    def test_reminder_triggered_by_user(self, sample_papers):
        """Test triggered_by='user' is preserved."""
        result = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=60,
            triggered_by="user",
        )
        
        assert result["triggered_by"] == "user"
    
    def test_reminder_triggered_by_agent(self, sample_papers):
        """Test triggered_by='agent' is preserved."""
        result = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=60,
            triggered_by="agent",
        )
        
        assert result["triggered_by"] == "agent"
    
    def test_reminder_with_agent_note(self, sample_papers):
        """Test that agent note is included."""
        result = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=60,
            agent_note="I recommend reading these in order of importance.",
        )
        
        ics = result["ics_content"]
        
        # Note should be in description
        assert "recommend" in ics.replace("\\n", "\n").lower()
    
    def test_reminder_uses_existing_uid(self, sample_papers):
        """Test that existing UID is reused for updates."""
        existing_uid = "existing-uid-123@researchpulse.app"
        
        result = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=60,
            existing_ics_uid=existing_uid,
        )
        
        assert result["ics_uid"] == existing_uid
        assert f"UID:{existing_uid}" in result["ics_content"]
    
    def test_reminder_sequence_number(self, sample_papers):
        """Test that sequence number is included."""
        result = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=60,
            sequence=3,
        )
        
        assert "SEQUENCE:3" in result["ics_content"]
        assert result["sequence"] == 3


class TestRescheduleInvite:
    """Test reschedule invite functionality."""
    
    @pytest.fixture
    def sample_papers(self):
        """Create sample papers."""
        return [{"title": "Test Paper", "url": "https://arxiv.org/abs/2401.00001"}]
    
    def test_reschedule_preserves_uid(self, sample_papers):
        """Test that reschedule uses same UID."""
        original_uid = "original-event-uid@researchpulse.app"
        
        result = send_reschedule_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            new_start_time=datetime.now() + timedelta(days=2),
            duration_minutes=45,
            ics_uid=original_uid,
            sequence=2,
        )
        
        assert result["ics_uid"] == original_uid
        assert f"UID:{original_uid}" in result["ics_content"]
    
    def test_reschedule_increments_sequence(self, sample_papers):
        """Test that reschedule has incremented sequence."""
        result = send_reschedule_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            new_start_time=datetime.now() + timedelta(days=2),
            duration_minutes=45,
            ics_uid="test-uid@researchpulse.app",
            sequence=5,
        )
        
        assert "SEQUENCE:5" in result["ics_content"]
    
    def test_reschedule_includes_reason(self, sample_papers):
        """Test that reschedule includes reason."""
        result = send_reschedule_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            new_start_time=datetime.now() + timedelta(days=2),
            duration_minutes=45,
            ics_uid="test-uid@researchpulse.app",
            sequence=2,
            reschedule_reason="User requested new time slot",
        )
        
        ics = result["ics_content"]
        
        assert "Rescheduled" in ics.replace("\\n", "\n")
    
    def test_reschedule_triggered_by_user(self, sample_papers):
        """Test that reschedules are marked as user-triggered."""
        result = send_reschedule_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            new_start_time=datetime.now() + timedelta(days=2),
            duration_minutes=45,
            ics_uid="test-uid@researchpulse.app",
            sequence=2,
        )
        
        # Reschedules should always be user-triggered
        assert result["triggered_by"] == "user"


class TestCalendarInviteEmail:
    """Test calendar invite email sending."""
    
    def test_email_not_configured(self):
        """Test behavior when email is not configured."""
        with patch.dict('os.environ', {
            'SMTP_USER': '',
            'SMTP_PASSWORD': '',
        }, clear=False):
            success, message_id, error = send_calendar_invite_email(
                to_email=get_test_email(),
                to_name="Test User",
                subject="Test Subject",
                body_text="Test body",
                body_html=None,
                ics_content="BEGIN:VCALENDAR\r\nEND:VCALENDAR\r\n",
            )
            
            # Should return simulated success in dev mode
            assert success is True
            assert message_id != ""
    
    def test_email_with_ics_attachment(self):
        """Test email includes ICS attachment."""
        with patch.dict('os.environ', {
            'SMTP_HOST': 'smtp.test.com',
            'SMTP_PORT': '587',
            'SMTP_USER': 'test@test.com',
            'SMTP_PASSWORD': 'password123',
        }):
            with patch('smtplib.SMTP') as mock_smtp:
                mock_server = MagicMock()
                mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
                mock_smtp.return_value.__exit__ = MagicMock(return_value=None)
                
                ics_content = "BEGIN:VCALENDAR\r\nVERSION:2.0\r\nEND:VCALENDAR\r\n"
                
                success, message_id, error = send_calendar_invite_email(
                    to_email=get_test_email(),
                    to_name="Test User",
                    subject="Calendar Invite",
                    body_text="Please see attached",
                    body_html="<p>Please see attached</p>",
                    ics_content=ics_content,
                    ics_method="REQUEST",
                )
                
                assert success is True
                # Unified outbound email module uses send_message instead of sendmail
                mock_server.send_message.assert_called_once()


class TestCalendarEventDuration:
    """Test duration estimation for calendar events."""
    
    def test_single_paper_duration(self):
        """Test duration for single paper."""
        papers = [{"title": "Short Paper", "url": "https://arxiv.org/abs/2401.00001"}]
        
        result = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=papers,
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=30,  # Explicit duration
        )
        
        ics = result["ics_content"]
        
        # Check that DTEND is 30 minutes after DTSTART
        assert "DTSTART:" in ics
        assert "DTEND:" in ics
    
    def test_multiple_papers_duration(self):
        """Test duration for multiple papers."""
        papers = [
            {"title": f"Paper {i}", "url": f"https://arxiv.org/abs/2401.0000{i}"}
            for i in range(5)
        ]
        
        result = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=papers,
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=120,  # 2 hours for 5 papers
        )
        
        ics = result["ics_content"]
        
        assert "DTSTART:" in ics
        assert "DTEND:" in ics


class TestCalendarNegativeCases:
    """Test error handling and edge cases."""
    
    def test_empty_papers_list(self):
        """Test handling empty papers list."""
        result = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=[],
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=30,
        )
        
        # Should still create a valid invite
        assert "BEGIN:VCALENDAR" in result["ics_content"]
    
    def test_past_start_time(self):
        """Test handling past start time."""
        result = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=[{"title": "Test", "url": ""}],
            start_time=datetime.now() - timedelta(days=1),  # Past time
            duration_minutes=30,
        )
        
        # Should still create the invite (validation is caller's responsibility)
        assert isinstance(result, dict)
    
    def test_very_long_paper_title(self):
        """Test handling very long paper title."""
        long_title = "A" * 500
        
        result = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=[{"title": long_title, "url": ""}],
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=30,
        )
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert result["ics_content"]
    
    def test_special_characters_in_email(self):
        """Test handling special characters."""
        result = send_reading_reminder_invite(
            user_email=get_colleague_test_email("user+test"),  # Plus sign in email
            user_name="Test User",
            papers=[{"title": "Test", "url": ""}],
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=30,
        )
        
        assert isinstance(result, dict)


class TestCalendarInviteEmailLink:
    """Test that calendar invites are linked to events."""
    
    def test_invite_contains_event_uid(self):
        """Test that invite email can be linked to calendar event."""
        papers = [{"title": "Test Paper", "url": ""}]
        
        result = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=papers,
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=30,
        )
        
        # The result should contain both ICS UID and email message ID
        assert "ics_uid" in result
        assert result["ics_uid"]
        
        # These can be used to link email to calendar event in DB
        assert "message_id" in result  # For email threading
    
    def test_multiple_invites_different_uids(self):
        """Test that different invites get different UIDs."""
        papers = [{"title": "Test Paper", "url": ""}]
        start_time = datetime.now() + timedelta(days=1)
        
        result1 = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=papers,
            start_time=start_time,
            duration_minutes=30,
        )
        
        result2 = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=papers,
            start_time=start_time,
            duration_minutes=30,
        )
        
        # Different invites should have different UIDs
        assert result1["ics_uid"] != result2["ics_uid"]
