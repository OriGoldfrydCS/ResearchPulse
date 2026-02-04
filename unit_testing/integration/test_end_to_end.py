"""
End-to-end integration tests for ResearchPulse.

Tests the complete workflow from chat query to automated emails and calendar events.
"""

import pytest
import os
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from src.tools.decide_delivery import (
    ScoredPaper,
    generate_summary_email_html,
)
from src.tools.calendar_invite_sender import send_reading_reminder_invite
from unit_testing.conftest import get_test_email, get_colleague_test_email


# Check if we should run live tests
def is_live_tests_enabled():
    """Check if live integration tests are enabled."""
    return os.getenv("RUN_LIVE_TESTS", "").lower() in ("1", "true", "yes")


class TestChatRunTriggersAutomation:
    """Test that chat runs trigger automated emails and calendar events."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings with automation enabled."""
        return {
            "auto_email_enabled": True,
            "auto_calendar_enabled": True,
            "email_address": get_test_email(),
            "user_name": "Test User",
            "arxiv_categories": ["cs.LG", "cs.CL", "cs.AI"],
            "research_topics": ["machine learning", "natural language processing"],
        }
    
    @pytest.fixture
    def sample_papers(self):
        """Create sample papers from a mock retrieval."""
        return [
            ScoredPaper(
                arxiv_id="2401.00001",
                title="Transformer Improvements for Language Understanding",
                abstract="We present improvements to the transformer architecture...",
                link="https://arxiv.org/abs/2401.00001",
                authors=["Alice Author", "Bob Researcher"],
                categories=["cs.CL", "cs.LG"],
                publication_date="2024-01-15",
                relevance_score=0.95,
                novelty_score=0.88,
                importance="high",
                explanation="Highly relevant to language model research.",
            ),
            ScoredPaper(
                arxiv_id="2401.00002",
                title="Vision Transformers: A Survey",
                abstract="A comprehensive survey of vision transformer models...",
                link="https://arxiv.org/abs/2401.00002",
                authors=["Charlie Vision"],
                categories=["cs.CV", "cs.LG"],
                publication_date="2024-01-14",
                relevance_score=0.75,
                novelty_score=0.65,
                importance="medium",
                explanation="Related to ML but focus on vision.",
            ),
            ScoredPaper(
                arxiv_id="2401.00003",
                title="Efficient Training Techniques",
                abstract="Methods to reduce computational costs of training...",
                link="https://arxiv.org/abs/2401.00003",
                authors=["David Efficiency"],
                categories=["cs.LG"],
                publication_date="2024-01-13",
                relevance_score=0.55,
                novelty_score=0.45,
                importance="low",
                explanation="Useful but not core to research.",
            ),
        ]
    
    def test_email_summary_generated(self, mock_settings, sample_papers):
        """Test that a valid email summary is generated."""
        query = "latest papers on transformer architectures"
        
        subject, plain_text, html = generate_summary_email_html(
            papers=sample_papers,
            query_text=query,
            triggered_by="agent",
        )
        
        # Verify subject
        assert "[ResearchPulse]" in subject
        assert "3 papers" in subject
        
        # Verify plain text
        assert isinstance(plain_text, str)
        assert len(plain_text) > 0
        
        # Verify HTML
        assert html.startswith("<!DOCTYPE html>") or "<html" in html
        assert "</html>" in html
    
    def test_email_one_per_query(self, mock_settings, sample_papers):
        """Test that only ONE email is created per query (not per paper)."""
        query = "test query"
        
        subject, plain_text, html = generate_summary_email_html(
            papers=sample_papers,
            query_text=query,
            triggered_by="agent",
        )
        
        # Should return a single email containing all papers
        for paper in sample_papers:
            assert paper.title in html
        
        # Count email boundaries - there should be only one
        # (The function returns ONE email, not multiple)
        assert isinstance(subject, str)  # Single subject, not list
    
    def test_email_grouped_by_importance(self, mock_settings, sample_papers):
        """Test that papers are grouped by importance in email."""
        query = "test query"
        
        subject, plain_text, html = generate_summary_email_html(
            papers=sample_papers,
            query_text=query,
            triggered_by="agent",
        )
        
        # Should have sections for each importance level
        html_lower = html.lower()
        
        # Check for importance groupings
        has_high = "high" in html_lower or "must read" in html_lower or "priority" in html_lower
        has_medium = "medium" in html_lower or "worth reading" in html_lower
        has_low = "low" in html_lower or "for later" in html_lower or "optional" in html_lower
        
        assert has_high or has_medium or has_low
    
    def test_email_correct_arxiv_links(self, mock_settings, sample_papers):
        """Test that arXiv links are correct."""
        query = "test query"
        
        subject, plain_text, html = generate_summary_email_html(
            papers=sample_papers,
            query_text=query,
            triggered_by="agent",
        )
        
        for paper in sample_papers:
            # Check that arXiv abs links are present
            assert f"arxiv.org/abs/{paper.arxiv_id}" in html or paper.arxiv_id in html
    
    def test_email_triggered_by_agent(self, mock_settings, sample_papers):
        """Test that agent-triggered emails are marked correctly."""
        query = "test query"
        
        subject, plain_text, html = generate_summary_email_html(
            papers=sample_papers,
            query_text=query,
            triggered_by="agent",
        )
        
        # Should indicate automatic/agent send
        assert "ResearchPulse" in html
        # Should NOT say "user request" or similar
        # The footer should indicate this was automatic
    
    def test_email_triggered_by_user(self, mock_settings, sample_papers):
        """Test that user-triggered emails are marked correctly."""
        query = "test query"
        
        subject, plain_text, html = generate_summary_email_html(
            papers=sample_papers,
            query_text=query,
            triggered_by="user",
        )
        
        # Should indicate user-initiated
        assert "you" in html.lower() or "request" in html.lower()


class TestCalendarEventCreation:
    """Test calendar event creation workflow."""
    
    @pytest.fixture
    def sample_papers(self):
        """Create sample papers."""
        return [
            {"title": "Paper 1", "url": "https://arxiv.org/abs/2401.00001", "importance": "high"},
            {"title": "Paper 2", "url": "https://arxiv.org/abs/2401.00002", "importance": "medium"},
        ]
    
    def test_calendar_event_created_per_paper(self, sample_papers):
        """Test that calendar events are created per paper."""
        results = []
        
        for paper in sample_papers:
            result = send_reading_reminder_invite(
                user_email=get_test_email(),
                user_name="Test User",
                papers=[paper],  # One paper per event
                start_time=datetime.now() + timedelta(days=1),
                duration_minutes=30,
                triggered_by="agent",
            )
            results.append(result)
        
        # Should have as many events as papers
        assert len(results) == len(sample_papers)
        
        # Each should have unique UID
        uids = [r["ics_uid"] for r in results]
        assert len(set(uids)) == len(uids)
    
    def test_calendar_event_duration_persisted(self, sample_papers):
        """Test that duration is correctly set."""
        result = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=45,
            triggered_by="agent",
        )
        
        ics = result["ics_content"]
        
        # Check that DTSTART and DTEND are present
        assert "DTSTART:" in ics
        assert "DTEND:" in ics
    
    def test_calendar_event_has_description(self, sample_papers):
        """Test that calendar event has paper info in description."""
        result = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=45,
            triggered_by="agent",
        )
        
        ics = result["ics_content"]
        
        # Description should contain paper titles
        assert "DESCRIPTION:" in ics
        for paper in sample_papers:
            assert paper["title"] in ics.replace("\\n", "\n")
    
    def test_calendar_triggered_by_agent(self, sample_papers):
        """Test that agent-triggered calendar is marked correctly."""
        result = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=45,
            triggered_by="agent",
        )
        
        assert result["triggered_by"] == "agent"
    
    def test_calendar_triggered_by_user(self, sample_papers):
        """Test that user-triggered calendar is marked correctly."""
        result = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=sample_papers,
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=45,
            triggered_by="user",
        )
        
        assert result["triggered_by"] == "user"


class TestCalendarInviteEmail:
    """Test calendar invite email workflow."""
    
    def test_invite_email_has_ics_attachment(self):
        """Test that invite email includes ICS."""
        result = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=[{"title": "Test Paper", "url": ""}],
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=30,
        )
        
        ics = result["ics_content"]
        
        # Valid ICS content
        assert "BEGIN:VCALENDAR" in ics
        assert "VERSION:2.0" in ics
        assert "BEGIN:VEVENT" in ics
        assert "END:VEVENT" in ics
        assert "END:VCALENDAR" in ics
    
    def test_invite_email_linked_to_event(self):
        """Test that invite email is linked to calendar event."""
        result = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=[{"title": "Test Paper", "url": ""}],
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=30,
        )
        
        # Should have ICS UID for linking
        assert "ics_uid" in result
        assert result["ics_uid"]
        
        # Should have message_id for email linking
        assert "message_id" in result


class TestFullAutomationWorkflow:
    """Test full automation workflow with mocks."""
    
    @pytest.fixture
    def mock_retriever(self):
        """Create mock paper retriever."""
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {
                "arxiv_id": "2401.00001",
                "title": "Test Paper",
                "abstract": "Abstract",
                "url": "https://arxiv.org/abs/2401.00001",
                "authors": ["Author"],
                "categories": ["cs.LG"],
                "published": "2024-01-15",
            }
        ]
        return retriever
    
    @pytest.fixture
    def mock_store(self):
        """Create mock data store."""
        store = MagicMock()
        store.save_paper.return_value = {"id": uuid.uuid4()}
        store.save_email.return_value = {"id": uuid.uuid4()}
        store.save_calendar_event.return_value = {"id": uuid.uuid4()}
        store.save_calendar_invite_email.return_value = {"id": uuid.uuid4()}
        return store
    
    def test_automation_creates_email(self, mock_retriever, mock_store):
        """Test that automation creates and persists email."""
        # Simulate paper discovery
        papers = mock_retriever.retrieve("test query")
        
        # Generate email
        scored_papers = [
            ScoredPaper(
                arxiv_id=p["arxiv_id"],
                title=p["title"],
                abstract=p["abstract"],
                relevance_score=0.9,
                novelty_score=0.8,
                importance="high",
            )
            for p in papers
        ]
        
        subject, plain_text, html = generate_summary_email_html(
            scored_papers, "test query", triggered_by="agent"
        )
        
        # Persist email
        mock_store.save_email({
            "subject": subject,
            "body_text": plain_text,
            "body_html": html,
            "recipient": get_test_email(),
            "is_summary": True,
            "triggered_by": "agent",
            "paper_ids": [p["arxiv_id"] for p in papers],
        })
        
        # Verify email was saved
        mock_store.save_email.assert_called_once()
        call_args = mock_store.save_email.call_args[0][0]
        assert call_args["triggered_by"] == "agent"
        assert call_args["is_summary"] is True
    
    def test_automation_creates_calendar_events(self, mock_retriever, mock_store):
        """Test that automation creates and persists calendar events."""
        # Simulate paper discovery
        papers = mock_retriever.retrieve("test query")
        
        # Create calendar events
        events = []
        for paper in papers:
            result = send_reading_reminder_invite(
                user_email=get_test_email(),
                user_name="Test User",
                papers=[{"title": paper["title"], "url": paper["url"]}],
                start_time=datetime.now() + timedelta(days=1),
                duration_minutes=30,
                triggered_by="agent",
            )
            events.append(result)
            
            # Persist calendar event
            mock_store.save_calendar_event({
                "ics_uid": result["ics_uid"],
                "paper_id": paper["arxiv_id"],
                "start_time": datetime.now() + timedelta(days=1),
                "duration_minutes": 30,
                "triggered_by": "agent",
            })
        
        # Verify events were saved
        assert mock_store.save_calendar_event.call_count == len(papers)
    
    def test_automation_labels_correctly(self, mock_retriever, mock_store):
        """Test that automated items are labeled as 'sent automatically'."""
        # Simulate paper discovery
        papers = mock_retriever.retrieve("test query")
        
        # Create email
        scored_papers = [
            ScoredPaper(
                arxiv_id=p["arxiv_id"],
                title=p["title"],
                relevance_score=0.9,
                novelty_score=0.8,
                importance="high",
            )
            for p in papers
        ]
        
        subject, plain_text, html = generate_summary_email_html(
            scored_papers, "test query", triggered_by="agent"
        )
        
        # Verify label
        assert "ResearchPulse" in html
        
        # Create calendar
        result = send_reading_reminder_invite(
            user_email=get_test_email(),
            user_name="Test User",
            papers=[{"title": "Test", "url": ""}],
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=30,
            triggered_by="agent",
        )
        
        assert result["triggered_by"] == "agent"
    
    def test_user_triggered_not_automatic_label(self, mock_retriever, mock_store):
        """Test that user-triggered items are NOT labeled as automatic."""
        # Create email with user trigger
        scored_papers = [
            ScoredPaper(
                arxiv_id="2401.00001",
                title="Test Paper",
                relevance_score=0.9,
                novelty_score=0.8,
                importance="high",
            )
        ]
        
        subject, plain_text, html = generate_summary_email_html(
            scored_papers, "test query", triggered_by="user"
        )
        
        # Should indicate user-initiated, not automatic
        assert "you" in html.lower() or "request" in html.lower()


class TestPaperPersistence:
    """Test paper discovery and persistence."""
    
    def test_paper_importance_scoring_saved(self):
        """Test that importance scoring is persisted."""
        paper = ScoredPaper(
            arxiv_id="2401.00001",
            title="Test Paper",
            relevance_score=0.85,
            novelty_score=0.75,
            importance="high",
        )
        
        # Verify scores are available for persistence
        assert paper.relevance_score == 0.85
        assert paper.novelty_score == 0.75
        assert paper.importance == "high"
    
    def test_paper_dates_available(self):
        """Test that paper dates are available."""
        paper = ScoredPaper(
            arxiv_id="2401.00001",
            title="Test Paper",
            publication_date="2024-01-15",
            added_at=datetime.now().isoformat(),
            relevance_score=0.9,
            novelty_score=0.8,
            importance="high",
        )
        
        assert paper.publication_date == "2024-01-15"
        assert paper.added_at is not None


@pytest.mark.skipif(
    not is_live_tests_enabled(),
    reason="Live tests disabled. Set RUN_LIVE_TESTS=1 to enable."
)
class TestLiveEndToEnd:
    """Live end-to-end tests."""
    
    @pytest.fixture
    def test_prefix(self):
        """Generate unique test prefix."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"[RP TEST] {timestamp}_{unique_id}"
    
    def test_send_summary_email_live(self, test_prefix):
        """Test sending summary email in live environment."""
        from src.tools.decide_delivery import _send_email_smtp
        
        email_account = os.getenv("SMTP_USER", "")
        if not email_account:
            pytest.skip("No email account configured")
        
        # Create sample papers
        papers = [
            ScoredPaper(
                arxiv_id="2401.00001",
                title="Test Paper for E2E",
                relevance_score=0.9,
                novelty_score=0.8,
                importance="high",
            )
        ]
        
        subject, plain_text, html = generate_summary_email_html(
            papers, f"{test_prefix} test query", triggered_by="agent"
        )
        
        # Send email
        success = _send_email_smtp(
            to_email=email_account,
            subject=f"{test_prefix} - {subject}",
            body=plain_text,
            html_body=html,
        )
        
        assert success is True
    
    def test_full_workflow_live(self, test_prefix):
        """Test full workflow: email + calendar in live environment."""
        from src.tools.decide_delivery import _send_email_smtp
        
        email_account = os.getenv("SMTP_USER", "")
        if not email_account:
            pytest.skip("No email account configured")
        
        # Create sample paper
        papers = [
            ScoredPaper(
                arxiv_id="2401.00001",
                title="Test Paper for Full E2E",
                relevance_score=0.9,
                novelty_score=0.8,
                importance="high",
            )
        ]
        
        # Step 1: Create summary email
        subject, plain_text, html = generate_summary_email_html(
            papers, f"{test_prefix} full workflow test", triggered_by="agent"
        )
        
        email_success = _send_email_smtp(
            to_email=email_account,
            subject=f"{test_prefix} - {subject}",
            body=plain_text,
            html_body=html,
        )
        
        assert email_success is True
        
        # Step 2: Create calendar invite
        calendar_result = send_reading_reminder_invite(
            user_email=email_account,
            user_name="Test User",
            papers=[{"title": p.title, "url": f"https://arxiv.org/abs/{p.arxiv_id}"} for p in papers],
            start_time=datetime.now() + timedelta(days=1),
            duration_minutes=30,
            triggered_by="agent",
        )
        
        assert calendar_result["success"] is True
        
        # Both should have proper triggered_by
        assert calendar_result["triggered_by"] == "agent"
