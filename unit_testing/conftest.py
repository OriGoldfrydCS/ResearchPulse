"""
Pytest fixtures for ResearchPulse tests.

Provides:
- Isolated test database
- Mock email/calendar providers
- Fake papers and inbox messages
- Settings fixtures
"""

import os
import sys
import uuid
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass, field

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# =============================================================================
# Test Configuration
# =============================================================================

def is_live_test_enabled() -> bool:
    """Check if live integration tests are enabled."""
    return os.getenv("RUN_LIVE_TESTS", "0").lower() in ("1", "true", "yes")


def get_test_email() -> str:
    """Get the test email address from environment.
    
    Uses SMTP_USER for live tests, falls back to a placeholder for unit tests.
    For live tests, ResearchPulse sends emails to itself and plays different roles.
    """
    return os.getenv("SMTP_USER", "researchpulse.test@gmail.com")


def get_colleague_test_email(name: str = "colleague") -> str:
    """Get a colleague email for testing.
    
    In live tests, we use the same ResearchPulse email (sending to self)
    with a +alias to distinguish roles.
    """
    base_email = get_test_email()
    if "@" in base_email:
        local, domain = base_email.split("@", 1)
        return f"{local}+{name}@{domain}"
    return base_email


# =============================================================================
# Provider Abstractions
# =============================================================================

@dataclass
class EmailMessage:
    """Represents an email message."""
    message_id: str
    from_email: str
    to_email: str
    subject: str
    body_text: str
    body_html: Optional[str] = None
    in_reply_to: Optional[str] = None
    references: Optional[str] = None
    date: datetime = field(default_factory=datetime.utcnow)
    attachments: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CalendarEvent:
    """Represents a calendar event."""
    event_id: str
    title: str
    description: str
    start_time: datetime
    duration_minutes: int = 30
    attendees: List[str] = field(default_factory=list)
    ics_uid: Optional[str] = None
    ics_text: Optional[str] = None


class MockEmailProvider:
    """Mock email provider for unit tests."""
    
    def __init__(self):
        self.sent_emails: List[EmailMessage] = []
        self.inbox: List[EmailMessage] = []
        self._send_success = True
        self._send_error = None
    
    def send_email(
        self,
        to_email: str,
        subject: str,
        body_text: str,
        body_html: Optional[str] = None,
        attachments: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Mock sending an email."""
        if not self._send_success:
            return {"success": False, "error": self._send_error or "Send failed"}
        
        msg = EmailMessage(
            message_id=f"<{uuid.uuid4()}@test.researchpulse.local>",
            from_email="test@researchpulse.local",
            to_email=to_email,
            subject=subject,
            body_text=body_text,
            body_html=body_html,
            attachments=attachments or [],
        )
        self.sent_emails.append(msg)
        return {"success": True, "message_id": msg.message_id}
    
    def list_inbox(self, since_hours: int = 24) -> List[EmailMessage]:
        """Mock listing inbox messages."""
        cutoff = datetime.utcnow() - timedelta(hours=since_hours)
        return [m for m in self.inbox if m.date >= cutoff]
    
    def read_message(self, message_id: str) -> Optional[EmailMessage]:
        """Mock reading a specific message."""
        for msg in self.inbox:
            if msg.message_id == message_id:
                return msg
        return None
    
    def add_to_inbox(self, msg: EmailMessage):
        """Add a message to the mock inbox."""
        self.inbox.append(msg)
    
    def set_send_failure(self, error: str = "Send failed"):
        """Configure sends to fail."""
        self._send_success = False
        self._send_error = error
    
    def set_send_success(self):
        """Configure sends to succeed."""
        self._send_success = True
        self._send_error = None
    
    def clear(self):
        """Clear all emails."""
        self.sent_emails.clear()
        self.inbox.clear()


class MockCalendarProvider:
    """Mock calendar provider for unit tests."""
    
    def __init__(self):
        self.events: List[CalendarEvent] = []
        self._create_success = True
        self._create_error = None
    
    def create_event(
        self,
        title: str,
        description: str,
        start_time: datetime,
        duration_minutes: int = 30,
        attendees: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Mock creating a calendar event."""
        if not self._create_success:
            return {"success": False, "error": self._create_error or "Create failed"}
        
        event = CalendarEvent(
            event_id=str(uuid.uuid4()),
            title=title,
            description=description,
            start_time=start_time,
            duration_minutes=duration_minutes,
            attendees=attendees or [],
            ics_uid=f"{uuid.uuid4()}@researchpulse.local",
        )
        self.events.append(event)
        return {"success": True, "event_id": event.event_id, "ics_uid": event.ics_uid}
    
    def update_event(
        self,
        event_id: str,
        start_time: Optional[datetime] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mock updating a calendar event."""
        for event in self.events:
            if event.event_id == event_id:
                if start_time:
                    event.start_time = start_time
                if title:
                    event.title = title
                if description:
                    event.description = description
                return {"success": True, "event_id": event_id}
        return {"success": False, "error": "Event not found"}
    
    def delete_event(self, event_id: str) -> Dict[str, Any]:
        """Mock deleting a calendar event."""
        for i, event in enumerate(self.events):
            if event.event_id == event_id:
                self.events.pop(i)
                return {"success": True}
        return {"success": False, "error": "Event not found"}
    
    def list_events(
        self,
        start_after: Optional[datetime] = None,
        start_before: Optional[datetime] = None,
    ) -> List[CalendarEvent]:
        """Mock listing calendar events."""
        events = self.events
        if start_after:
            events = [e for e in events if e.start_time >= start_after]
        if start_before:
            events = [e for e in events if e.start_time <= start_before]
        return events
    
    def set_create_failure(self, error: str = "Create failed"):
        """Configure creates to fail."""
        self._create_success = False
        self._create_error = error
    
    def set_create_success(self):
        """Configure creates to succeed."""
        self._create_success = True
        self._create_error = None
    
    def clear(self):
        """Clear all events."""
        self.events.clear()


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_email_provider():
    """Provide a mock email provider."""
    provider = MockEmailProvider()
    yield provider
    provider.clear()


@pytest.fixture
def mock_calendar_provider():
    """Provide a mock calendar provider."""
    provider = MockCalendarProvider()
    yield provider
    provider.clear()


@pytest.fixture
def fake_papers():
    """Provide a list of fake papers for testing."""
    return [
        {
            "arxiv_id": "2401.00001",
            "title": "Attention-Based Neural Networks for Time Series Forecasting",
            "authors": ["Alice Smith", "Bob Johnson"],
            "abstract": "We propose a novel attention mechanism for time series prediction...",
            "categories": ["cs.LG", "cs.AI"],
            "url": "https://arxiv.org/abs/2401.00001",
            "pdf_url": "https://arxiv.org/pdf/2401.00001.pdf",
            "published_at": datetime(2024, 1, 15),
            "relevance_score": 0.92,
            "novelty_score": 0.85,
            "importance": "high",
        },
        {
            "arxiv_id": "2401.00002",
            "title": "Efficient Training of Large Language Models",
            "authors": ["Carol Williams", "David Brown"],
            "abstract": "We present techniques for efficient LLM training...",
            "categories": ["cs.CL", "cs.LG"],
            "url": "https://arxiv.org/abs/2401.00002",
            "pdf_url": "https://arxiv.org/pdf/2401.00002.pdf",
            "published_at": datetime(2024, 1, 16),
            "relevance_score": 0.78,
            "novelty_score": 0.72,
            "importance": "medium",
        },
        {
            "arxiv_id": "2401.00003",
            "title": "A Survey of Reinforcement Learning Methods",
            "authors": ["Eve Davis"],
            "abstract": "This survey covers recent advances in RL...",
            "categories": ["cs.AI"],
            "url": "https://arxiv.org/abs/2401.00003",
            "pdf_url": "https://arxiv.org/pdf/2401.00003.pdf",
            "published_at": datetime(2024, 1, 17),
            "relevance_score": 0.55,
            "novelty_score": 0.40,
            "importance": "low",
        },
    ]


@pytest.fixture
def fake_inbox_messages():
    """Provide fake inbox messages for testing.
    
    Uses real email addresses from SMTP_USER so live tests work correctly.
    ResearchPulse plays different roles (owner, user responding, colleague signing up).
    """
    base_time = datetime.utcnow()
    test_email = get_test_email()
    user_email = get_colleague_test_email("user")
    colleague_email = get_colleague_test_email("colleague")
    
    return [
        EmailMessage(
            message_id=f"<reschedule-001@{test_email.split('@')[-1] if '@' in test_email else 'researchpulse.app'}>",
            from_email=user_email,
            to_email=test_email,
            subject="Re: [ResearchPulse] Reading Reminder: Attention-Based Neural Networks",
            body_text="This time doesn't work. Can we move it to tomorrow at 2pm?",
            in_reply_to="<invite-001@researchpulse.local>",
            date=base_time - timedelta(hours=1),
        ),
        EmailMessage(
            message_id=f"<accept-001@{test_email.split('@')[-1] if '@' in test_email else 'researchpulse.app'}>",
            from_email=user_email,
            to_email=test_email,
            subject="Re: [ResearchPulse] Reading Reminder: LLM Training",
            body_text="Thanks, looks good!",
            in_reply_to="<invite-002@researchpulse.local>",
            date=base_time - timedelta(hours=2),
        ),
        EmailMessage(
            message_id=f"<colleague-signup-001@{test_email.split('@')[-1] if '@' in test_email else 'researchpulse.app'}>",
            from_email=colleague_email,
            to_email=test_email,
            subject="Request to receive ResearchPulse updates",
            body_text="Hi, I'm a researcher in machine learning and NLP. I would like to subscribe to receive paper recommendations. My interests include transformers, LLMs, and neural architectures.",
            date=base_time - timedelta(hours=3),
        ),
        EmailMessage(
            message_id=f"<cancel-001@{test_email.split('@')[-1] if '@' in test_email else 'researchpulse.app'}>",
            from_email=user_email,
            to_email=test_email,
            subject="Re: [ResearchPulse] Reading Reminder",
            body_text="Please cancel this event, I no longer need it.",
            in_reply_to="<invite-003@researchpulse.local>",
            date=base_time - timedelta(hours=4),
        ),
    ]


@pytest.fixture
def test_settings():
    """Provide test settings/configuration.
    
    Uses real email from SMTP_USER for live testing.
    """
    return {
        "email": get_test_email(),
        "name": "Test Researcher",
        "smtp_configured": True,
        "imap_configured": True,
        "calendar_enabled": True,
        "auto_email_enabled": True,
        "auto_calendar_enabled": True,
        "research_topics": ["machine learning", "natural language processing"],
        "arxiv_categories_include": ["cs.AI", "cs.LG", "cs.CL"],
        "arxiv_categories_exclude": [],
        "time_budget_per_week_minutes": 120,
        "preferences": {
            "email_digest": False,
            "include_abstract": True,
            "include_explanation": True,
        }
    }


@pytest.fixture
def fake_colleagues():
    """Provide a list of fake colleagues for testing.
    
    Uses email+alias addresses based on SMTP_USER so live tests can
    send real emails that arrive in the same inbox.
    """
    return [
        {
            "id": str(uuid.uuid4()),
            "name": "Alice Colleague",
            "email": get_colleague_test_email("alice"),
            "research_interests": "Machine learning and computer vision",
            "keywords": ["machine learning", "computer vision", "deep learning"],
            "categories": ["cs.CV", "cs.LG"],
            "auto_send_emails": True,
            "enabled": True,
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Bob Researcher",
            "email": get_colleague_test_email("bob"),
            "research_interests": "NLP and transformers",
            "keywords": ["NLP", "transformers", "language models"],
            "categories": ["cs.CL"],
            "auto_send_emails": True,
            "enabled": True,
        },
    ]


@pytest.fixture
def mock_db_session():
    """Provide a mock database session for unit tests."""
    session = MagicMock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    return session


@pytest.fixture
def mock_store():
    """Provide a mock data store for unit tests."""
    store = MagicMock()
    
    # Configure common return values
    store.get_user.return_value = {
        "id": str(uuid.uuid4()),
        "name": "Test User",
        "email": get_test_email(),
    }
    
    store.create_email.return_value = {"id": str(uuid.uuid4()), "status": "sent"}
    store.create_calendar_event.return_value = {"id": str(uuid.uuid4()), "status": "created"}
    store.create_calendar_invite_email.return_value = {"id": str(uuid.uuid4())}
    store.create_colleague.return_value = {"id": str(uuid.uuid4())}
    
    return store


# =============================================================================
# Live Test Fixtures (only used when RUN_LIVE_TESTS=1)
# =============================================================================

@pytest.fixture
def live_email_config():
    """Get live email configuration from environment."""
    if not is_live_test_enabled():
        pytest.skip("Live tests disabled. Set RUN_LIVE_TESTS=1 to enable.")
    
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    
    if not smtp_user or not smtp_password:
        pytest.skip("SMTP_USER and SMTP_PASSWORD required for live email tests")
    
    return {
        "smtp_host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
        "smtp_port": int(os.getenv("SMTP_PORT", "587")),
        "imap_host": os.getenv("IMAP_HOST", "imap.gmail.com"),
        "imap_port": int(os.getenv("IMAP_PORT", "993")),
        "user": smtp_user,
        "password": smtp_password,
    }


@pytest.fixture
def live_test_subject_prefix():
    """Generate unique test subject prefix for live tests."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"[RP TEST] {timestamp}"


# =============================================================================
# Utility Functions
# =============================================================================

def generate_test_ics(
    title: str,
    start_time: datetime,
    duration_minutes: int = 30,
    uid: Optional[str] = None,
) -> str:
    """Generate a test ICS file content."""
    uid = uid or f"{uuid.uuid4()}@researchpulse.test"
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    dtstart = start_time.strftime("%Y%m%dT%H%M%S")
    dtend = end_time.strftime("%Y%m%dT%H%M%S")
    dtstamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    
    return f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//ResearchPulse//Test//EN
METHOD:REQUEST
BEGIN:VEVENT
UID:{uid}
DTSTAMP:{dtstamp}
DTSTART:{dtstart}
DTEND:{dtend}
SUMMARY:{title}
DESCRIPTION:Test event
STATUS:CONFIRMED
SEQUENCE:0
END:VEVENT
END:VCALENDAR"""


def wait_for_condition(
    condition_fn,
    timeout_seconds: int = 60,
    poll_interval: float = 2.0,
    message: str = "Condition not met",
):
    """Wait for a condition to be true, with timeout."""
    import time
    start = time.time()
    while time.time() - start < timeout_seconds:
        if condition_fn():
            return True
        time.sleep(poll_interval)
    raise TimeoutError(f"{message} (timeout: {timeout_seconds}s)")
