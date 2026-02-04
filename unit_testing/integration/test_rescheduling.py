"""
Integration tests for rescheduling via email reply.

Tests the complete reschedule flow:
1. Create a calendar invite
2. Simulate/send a reply email with reschedule request
3. Verify agent processes the reply
4. Verify calendar event is updated
5. Verify new invite is sent
6. Verify audit trail records the reschedule
"""

import pytest
import os
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from src.agent.reply_parser import parse_reply, parse_reply_rules, ReplyIntent, ParsedReply
from src.tools.calendar_invite_sender import send_reading_reminder_invite, send_reschedule_invite
from src.tools.email_poller import fetch_recent_replies, poll_and_process_replies
from unit_testing.conftest import get_test_email, get_colleague_test_email


# Check if we should run live tests
def is_live_tests_enabled():
    """Check if live integration tests are enabled."""
    return os.getenv("RUN_LIVE_TESTS", "").lower() in ("1", "true", "yes")


class TestReplyParserIntegration:
    """Integration tests for reply parsing."""
    
    def test_parse_reschedule_tomorrow(self):
        """Test parsing 'tomorrow' reschedule request."""
        body = "This time doesn't work. Can we move it to tomorrow at 3pm?"
        
        result = parse_reply(body, use_llm=False)
        
        assert result.intent == ReplyIntent.RESCHEDULE
        assert "tomorrow" in result.extracted_datetime_text.lower() or result.extracted_datetime
    
    def test_parse_reschedule_specific_date(self):
        """Test parsing specific date reschedule request."""
        body = "Please reschedule to February 15th at 10am."
        
        result = parse_reply(body, use_llm=False)
        
        assert result.intent == ReplyIntent.RESCHEDULE
        assert result.extracted_datetime or "february" in result.extracted_datetime_text.lower()
    
    def test_parse_reschedule_next_week(self):
        """Test parsing 'next week' reschedule request."""
        body = "I'm busy this week. Let's do next Monday at 2pm instead."
        
        result = parse_reply(body, use_llm=False)
        
        assert result.intent == ReplyIntent.RESCHEDULE
        assert "monday" in result.extracted_datetime_text.lower() or result.extracted_datetime
    
    def test_parse_accept(self):
        """Test parsing acceptance reply."""
        body = "Yes, this works perfectly for me. See you then!"
        
        result = parse_reply(body, use_llm=False)
        
        assert result.intent == ReplyIntent.ACCEPT
    
    def test_parse_decline(self):
        """Test parsing decline reply."""
        body = "I won't be able to make it. Please remove me from this event."
        
        result = parse_reply(body, use_llm=False)
        
        assert result.intent in [ReplyIntent.DECLINE, ReplyIntent.CANCEL]
    
    def test_parse_cancel(self):
        """Test parsing cancellation request."""
        body = "Please cancel this reminder. I've already read the paper."
        
        result = parse_reply(body, use_llm=False)
        
        assert result.intent == ReplyIntent.CANCEL
    
    def test_parse_question(self):
        """Test parsing question reply."""
        body = "What papers are included in this reading session?"
        
        result = parse_reply(body, use_llm=False)
        
        assert result.intent == ReplyIntent.QUESTION
    
    def test_parse_ambiguous(self):
        """Test parsing ambiguous reply."""
        # Using text that doesn't match any keywords
        body = "I'll get back to you later."
        
        result = parse_reply(body, use_llm=False)
        
        # Should return OTHER or UNKNOWN for ambiguous replies
        assert result.intent in [ReplyIntent.OTHER, ReplyIntent.UNKNOWN]
    
    def test_parse_confidence_score(self):
        """Test that confidence score is returned."""
        body = "Please reschedule to tomorrow at 2pm."
        
        result = parse_reply(body, use_llm=False)
        
        assert hasattr(result, 'confidence_score')
        assert 0.0 <= result.confidence_score <= 1.0


class TestRescheduleWorkflow:
    """Test the complete reschedule workflow."""
    
    @pytest.fixture
    def sample_papers(self):
        """Create sample papers."""
        return [
            {"title": "Test Paper 1", "url": "https://arxiv.org/abs/2401.00001"},
            {"title": "Test Paper 2", "url": "https://arxiv.org/abs/2401.00002"},
        ]
    
    @pytest.fixture
    def original_invite(self, sample_papers):
        """Create an original calendar invite."""
        return send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=60,
            triggered_by="agent",
        )
    
    def test_reschedule_preserves_papers(self, sample_papers, original_invite):
        """Test that reschedule preserves paper information."""
        new_start = datetime.now() + timedelta(days=2, hours=3)
        
        result = send_reschedule_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            new_start_time=new_start,
            duration_minutes=60,
            ics_uid=original_invite["ics_uid"],
            sequence=1,
            reschedule_reason="User requested new time",
        )
        
        # Should use same UID
        assert result["ics_uid"] == original_invite["ics_uid"]
        
        # Papers should still be in description
        ics = result["ics_content"]
        assert "Test Paper 1" in ics.replace("\\n", "\n")
    
    def test_reschedule_increments_sequence(self, sample_papers, original_invite):
        """Test that reschedule increments sequence number."""
        new_start = datetime.now() + timedelta(days=2)
        
        # First reschedule
        result1 = send_reschedule_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            new_start_time=new_start,
            duration_minutes=60,
            ics_uid=original_invite["ics_uid"],
            sequence=1,
        )
        
        assert "SEQUENCE:1" in result1["ics_content"]
        
        # Second reschedule
        result2 = send_reschedule_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            new_start_time=new_start + timedelta(hours=1),
            duration_minutes=60,
            ics_uid=original_invite["ics_uid"],
            sequence=2,
        )
        
        assert "SEQUENCE:2" in result2["ics_content"]
    
    def test_reschedule_triggered_by_user(self, sample_papers, original_invite):
        """Test that reschedule is marked as user-triggered."""
        new_start = datetime.now() + timedelta(days=2)
        
        result = send_reschedule_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            new_start_time=new_start,
            duration_minutes=60,
            ics_uid=original_invite["ics_uid"],
            sequence=1,
        )
        
        assert result["triggered_by"] == "user"


class TestPollAndProcessReplies:
    """Test the poll and process replies workflow."""
    
    @pytest.fixture
    def mock_store(self):
        """Create a mock store."""
        store = MagicMock()
        store.get_calendar_invite_by_message_id.return_value = None
        store.get_inbound_reply_by_message_id.return_value = None
        return store
    
    @pytest.mark.asyncio
    async def test_poll_no_credentials(self, mock_store):
        """Test polling with no credentials."""
        with patch.dict('os.environ', {'SMTP_USER': '', 'SMTP_PASSWORD': ''}):
            result = await poll_and_process_replies(mock_store, "user-123")
            
            assert result["emails_found"] == 0
            assert result["replies_matched"] == 0
    
    @pytest.mark.asyncio
    async def test_poll_no_matching_invite(self, mock_store):
        """Test polling when reply doesn't match any invite."""
        user_email = get_colleague_test_email("user")
        mock_replies = [{
            "message_id": "reply-123",
            "in_reply_to": "unknown-invite@researchpulse.app",
            "from_email": user_email,
            "subject": "Re: ResearchPulse Reading Reminder",
            "body_text": "Please reschedule to tomorrow",
            "received_at": "2026-02-03",
        }]
        
        with patch('src.tools.email_poller.fetch_recent_replies', return_value=mock_replies):
            result = await poll_and_process_replies(mock_store, "user-123")
            
            assert result["emails_found"] == 1
            assert result["replies_matched"] == 0
    
    @pytest.mark.asyncio
    async def test_poll_matching_invite_reschedule(self, mock_store):
        """Test polling with matching invite and reschedule intent."""
        user_email = get_colleague_test_email("user")
        mock_replies = [{
            "message_id": "reply-123",
            "in_reply_to": "invite-456@researchpulse.app",
            "from_email": user_email,
            "subject": "Re: ResearchPulse Reading Reminder",
            "body_text": "Please reschedule to tomorrow at 2pm.",
            "received_at": "2026-02-03",
        }]
        
        mock_invite = {
            "id": uuid.uuid4(),
            "calendar_event_id": uuid.uuid4(),
            "message_id": "invite-456@researchpulse.app",
        }
        
        mock_event = {
            "id": uuid.uuid4(),
            "start_time": datetime.now() + timedelta(days=1),
            "paper_ids": [],
        }
        
        mock_store.get_calendar_invite_by_message_id.return_value = mock_invite
        mock_store.get_inbound_reply_by_message_id.return_value = None
        mock_store.create_inbound_email_reply.return_value = {"id": uuid.uuid4()}
        mock_store.get_calendar_event.return_value = mock_event
        mock_store.reschedule_calendar_event.return_value = {
            **mock_event,
            "start_time": datetime.now() + timedelta(days=2),
        }
        mock_store.get_user.return_value = {"email": get_test_email()}
        
        with patch('src.tools.email_poller.fetch_recent_replies', return_value=mock_replies):
            with patch('src.tools.calendar_invite_sender.send_reschedule_invite'):
                result = await poll_and_process_replies(mock_store, str(uuid.uuid4()))
                
                assert result["emails_found"] == 1
                assert result["replies_matched"] == 1


class TestRescheduleNegativeCases:
    """Test error handling in reschedule workflow."""
    
    def test_parse_invalid_date_format(self):
        """Test parsing invalid date format."""
        body = "Please reschedule to blahblah o'clock."
        
        result = parse_reply(body, use_llm=False)
        
        # Should still detect intent, even if date extraction fails
        # Intent might be RESCHEDULE or UNKNOWN depending on confidence
        assert isinstance(result, ParsedReply)
    
    def test_parse_empty_body(self):
        """Test parsing empty body."""
        result = parse_reply("", use_llm=False)
        
        assert result.intent in [ReplyIntent.UNKNOWN, ReplyIntent.OTHER]
    
    def test_parse_very_long_body(self):
        """Test parsing very long body."""
        # Long email with lots of quoted text
        body = """
        Please reschedule to tomorrow.
        
        """ + ">" * 100 + " Quoted text\n" * 100
        
        result = parse_reply(body, use_llm=False)
        
        # Should still work
        assert isinstance(result, ParsedReply)
    
    def test_reschedule_to_past(self):
        """Test reschedule to past time."""
        sample_papers = [{"title": "Test", "url": ""}]
        
        # This should still create the invite (validation is caller's responsibility)
        result = send_reschedule_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            new_start_time=datetime.now() - timedelta(days=1),  # Past time
            duration_minutes=30,
            ics_uid="test-uid@researchpulse.app",
            sequence=1,
        )
        
        assert isinstance(result, dict)
        assert result["ics_content"]


class TestRescheduleAuditTrail:
    """Test that reschedules are properly tracked."""
    
    def test_reschedule_includes_reason(self):
        """Test that reschedule includes reason in ICS."""
        sample_papers = [{"title": "Test Paper", "url": ""}]
        
        result = send_reschedule_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            new_start_time=datetime.now() + timedelta(days=2),
            duration_minutes=45,
            ics_uid="test-uid@researchpulse.app",
            sequence=2,
            reschedule_reason="User requested via email: 'Please move to Thursday'",
        )
        
        ics = result["ics_content"]
        
        # Reason should appear in description
        assert "Rescheduled" in ics.replace("\\n", "\n")
    
    def test_reschedule_message_id_for_linking(self):
        """Test that reschedule has message_id for email linking."""
        sample_papers = [{"title": "Test Paper", "url": ""}]
        
        result = send_reschedule_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            new_start_time=datetime.now() + timedelta(days=2),
            duration_minutes=45,
            ics_uid="test-uid@researchpulse.app",
            sequence=2,
        )
        
        # Should have message_id for email linking
        assert "message_id" in result
        assert result["message_id"]


@pytest.mark.skipif(
    not is_live_tests_enabled(),
    reason="Live tests disabled. Set RUN_LIVE_TESTS=1 to enable."
)
class TestLiveReschedule:
    """Live integration tests for rescheduling."""
    
    @pytest.fixture
    def test_prefix(self):
        """Generate unique test prefix."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"[RP TEST] {timestamp}_{unique_id}"
    
    def test_send_reschedule_email_to_self(self, test_prefix):
        """Test sending reschedule reply to self."""
        from src.tools.decide_delivery import _send_email_smtp
        
        email_account = os.getenv("SMTP_USER", "")
        if not email_account:
            pytest.skip("No email account configured")
        
        subject = f"Re: {test_prefix} ResearchPulse Reading Reminder"
        body = "This time doesn't work. Please reschedule to Friday at 11am."
        
        success = _send_email_smtp(
            to_email=email_account,
            subject=subject,
            body=body,
        )
        
        assert success is True
        
        # The reply should be parseable
        result = parse_reply(body, use_llm=False)
        assert result.intent == ReplyIntent.RESCHEDULE
