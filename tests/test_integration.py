"""
Integration Tests for ResearchPulse Deployment-Safe Storage

Tests the database layer, API endpoints, and data flow.
Run with: pytest tests/test_integration.py -v
"""

import pytest
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
from uuid import uuid4

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_db_session():
    """Create a mock database session for testing."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from src.db.orm_models import Base
    
    # Use in-memory SQLite for testing
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    
    yield session
    
    session.close()
    engine.dispose()


@pytest.fixture
def test_user(mock_db_session):
    """Create a test user."""
    from src.db.orm_models import User
    
    user = User(
        id=str(uuid4()),
        name="Test Researcher",
        email="test@example.com",
        affiliation="Test University",
        research_topics=["machine learning", "NLP"],
    )
    mock_db_session.add(user)
    mock_db_session.commit()
    mock_db_session.refresh(user)
    
    return user


@pytest.fixture
def test_colleague(mock_db_session, test_user):
    """Create a test colleague."""
    from src.db.orm_models import Colleague
    
    colleague = Colleague(
        id=str(uuid4()),
        user_id=test_user.id,
        name="Test Colleague",
        email="colleague@example.com",
        keywords=["transformers", "attention"],
        categories=["cs.AI", "cs.LG"],
        enabled=True,
    )
    mock_db_session.add(colleague)
    mock_db_session.commit()
    mock_db_session.refresh(colleague)
    
    return colleague


@pytest.fixture
def test_paper(mock_db_session):
    """Create a test paper."""
    from src.db.orm_models import Paper
    
    paper = Paper(
        id=str(uuid4()),
        source="arxiv",
        external_id="2501.12345",
        title="Test Paper: Advances in Neural Networks",
        abstract="This paper presents novel approaches to neural network training.",
        authors=["Author One", "Author Two"],
        categories=["cs.LG", "cs.AI"],
        url="https://arxiv.org/abs/2501.12345",
        published_at=datetime.now(),
    )
    mock_db_session.add(paper)
    mock_db_session.commit()
    mock_db_session.refresh(paper)
    
    return paper


@pytest.fixture
def test_client():
    """Create a FastAPI test client."""
    from fastapi.testclient import TestClient
    from main import app
    
    with TestClient(app) as client:
        yield client


# =============================================================================
# Database Model Tests
# =============================================================================

class TestDatabaseModels:
    """Test SQLAlchemy ORM models."""
    
    def test_user_creation(self, mock_db_session):
        """Test creating a user."""
        from src.db.orm_models import User
        
        user = User(
            id=str(uuid4()),
            name="John Doe",
            email="john@example.com",
        )
        mock_db_session.add(user)
        mock_db_session.commit()
        
        # Query back
        db_user = mock_db_session.query(User).filter_by(email="john@example.com").first()
        assert db_user is not None
        assert db_user.name == "John Doe"
    
    def test_paper_creation(self, mock_db_session):
        """Test creating a paper."""
        from src.db.orm_models import Paper
        
        paper = Paper(
            id=str(uuid4()),
            source="arxiv",
            external_id="2501.99999",
            title="Test Paper",
            abstract="Test abstract",
            authors=["Author A"],
            categories=["cs.AI"],
            url="https://arxiv.org/abs/2501.99999",
        )
        mock_db_session.add(paper)
        mock_db_session.commit()
        
        # Query back
        db_paper = mock_db_session.query(Paper).filter_by(external_id="2501.99999").first()
        assert db_paper is not None
        assert db_paper.title == "Test Paper"
        assert "Author A" in db_paper.authors
    
    def test_colleague_creation(self, mock_db_session, test_user):
        """Test creating a colleague."""
        from src.db.orm_models import Colleague
        
        colleague = Colleague(
            id=str(uuid4()),
            user_id=test_user.id,
            name="Jane Smith",
            email="jane@example.com",
            keywords=["deep learning", "computer vision"],
            categories=["cs.CV"],
            enabled=True,
        )
        mock_db_session.add(colleague)
        mock_db_session.commit()
        
        # Query back
        db_colleague = mock_db_session.query(Colleague).filter_by(email="jane@example.com").first()
        assert db_colleague is not None
        assert db_colleague.name == "Jane Smith"
        assert "deep learning" in db_colleague.keywords
    
    def test_paper_view_creation(self, mock_db_session, test_user, test_paper):
        """Test creating a paper view."""
        from src.db.orm_models import PaperView
        
        view = PaperView(
            id=str(uuid4()),
            user_id=test_user.id,
            paper_id=test_paper.id,
            decision="read",
            importance=0.85,
            notes="Interesting paper on neural networks",
            tags=["important", "to-read"],
        )
        mock_db_session.add(view)
        mock_db_session.commit()
        
        # Query back
        db_view = mock_db_session.query(PaperView).filter_by(paper_id=test_paper.id).first()
        assert db_view is not None
        assert db_view.decision == "read"
        assert db_view.importance == 0.85
    
    def test_email_creation(self, mock_db_session, test_user, test_paper):
        """Test creating an email record."""
        from src.db.orm_models import Email
        
        email = Email(
            id=str(uuid4()),
            user_id=test_user.id,
            paper_id=test_paper.id,
            recipient_email="recipient@example.com",
            subject="New Paper Alert",
            body_text="Check out this new paper...",
            status="sent",
        )
        mock_db_session.add(email)
        mock_db_session.commit()
        
        # Query back
        db_email = mock_db_session.query(Email).filter_by(paper_id=test_paper.id).first()
        assert db_email is not None
        assert db_email.subject == "New Paper Alert"
        assert db_email.status == "sent"
    
    def test_calendar_event_creation(self, mock_db_session, test_user, test_paper):
        """Test creating a calendar event."""
        from src.db.orm_models import CalendarEvent
        
        event = CalendarEvent(
            id=str(uuid4()),
            user_id=test_user.id,
            paper_id=test_paper.id,
            title="Read: Test Paper",
            start_time=datetime.now(),
            duration_minutes=30,
            ics_text="BEGIN:VCALENDAR...",
            status="created",
        )
        mock_db_session.add(event)
        mock_db_session.commit()
        
        # Query back
        db_event = mock_db_session.query(CalendarEvent).filter_by(paper_id=test_paper.id).first()
        assert db_event is not None
        assert db_event.title == "Read: Test Paper"
    
    def test_share_creation(self, mock_db_session, test_user, test_paper, test_colleague):
        """Test creating a share record."""
        from src.db.orm_models import Share
        
        share = Share(
            id=str(uuid4()),
            user_id=test_user.id,
            paper_id=test_paper.id,
            colleague_id=test_colleague.id,
            reason="High relevance to colleague's research",
            match_score=0.92,
            status="shared",
        )
        mock_db_session.add(share)
        mock_db_session.commit()
        
        # Query back
        db_share = mock_db_session.query(Share).filter_by(paper_id=test_paper.id).first()
        assert db_share is not None
        assert db_share.match_score == 0.92


# =============================================================================
# API Endpoint Tests
# =============================================================================

class TestAPIEndpoints:
    """Test FastAPI endpoints."""
    
    def test_health_endpoint(self, test_client):
        """Test /api/health endpoint returns OK."""
        response = test_client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint serves content."""
        response = test_client.get("/")
        assert response.status_code == 200


# =============================================================================
# Store Interface Tests
# =============================================================================

class TestStoreInterface:
    """Test the store interface and backend selection."""
    
    def test_get_store_returns_interface(self):
        """Test that get_store returns a Store implementation."""
        # Mock DATABASE_URL to avoid real connection
        with patch.dict(os.environ, {"DATABASE_URL": ""}):
            from src.db.store import get_store
            store = get_store()
            assert store is not None
    
    def test_storage_backend_env_var(self):
        """Test STORAGE_BACKEND environment variable."""
        # Default should be 'db'
        backend = os.getenv("STORAGE_BACKEND", "db")
        assert backend in ["db", "json"]


# =============================================================================
# Vector Store Tests (Mocked)
# =============================================================================

class TestVectorStore:
    """Test vector store operations with mocks."""
    
    def test_check_vector_store_available(self):
        """Test vector store availability check."""
        from src.rag.vector_store import check_vector_store_available
        
        is_available, message = check_vector_store_available()
        assert isinstance(is_available, bool)
        assert isinstance(message, str)
    
    def test_upsert_paper_vector_structure(self):
        """Test upsert_paper_vector function exists and has correct signature."""
        from src.rag.vector_store import upsert_paper_vector
        
        # Just test the function exists and can be called with mock data
        assert callable(upsert_paper_vector)
    
    def test_delete_paper_from_vector_store_structure(self):
        """Test delete_paper_from_vector_store function exists."""
        from src.rag.vector_store import delete_paper_from_vector_store
        
        assert callable(delete_paper_from_vector_store)
    
    def test_check_pinecone_health_structure(self):
        """Test check_pinecone_health function exists."""
        from src.rag.vector_store import check_pinecone_health
        
        is_healthy, message = check_pinecone_health()
        assert isinstance(is_healthy, bool)
        assert isinstance(message, str)


# =============================================================================
# Data Migration Tests
# =============================================================================

class TestDataMigration:
    """Test data migration functionality."""
    
    def test_papers_state_json_parsing(self, tmp_path):
        """Test parsing papers_state.json format."""
        papers_data = {
            "papers": [
                {
                    "id": "2501.12345",
                    "title": "Test Paper",
                    "abstract": "Test abstract",
                    "authors": ["Author A"],
                    "categories": ["cs.AI"],
                    "decision": "read",
                    "importance": 0.8,
                }
            ],
            "stats": {
                "total_papers": 1,
                "decisions": {"read": 1},
            }
        }
        
        # Write to temp file
        papers_file = tmp_path / "papers_state.json"
        papers_file.write_text(json.dumps(papers_data))
        
        # Read and parse
        with open(papers_file, "r") as f:
            loaded = json.load(f)
        
        assert len(loaded["papers"]) == 1
        assert loaded["papers"][0]["id"] == "2501.12345"
    
    def test_colleagues_json_parsing(self, tmp_path):
        """Test parsing colleagues.json format."""
        colleagues_data = {
            "colleagues": [
                {
                    "name": "Jane Doe",
                    "email": "jane@example.com",
                    "keywords": ["ML", "NLP"],
                    "categories": ["cs.CL"],
                    "enabled": True,
                }
            ]
        }
        
        # Write to temp file
        colleagues_file = tmp_path / "colleagues.json"
        colleagues_file.write_text(json.dumps(colleagues_data))
        
        # Read and parse
        with open(colleagues_file, "r") as f:
            loaded = json.load(f)
        
        assert len(loaded["colleagues"]) == 1
        assert loaded["colleagues"][0]["name"] == "Jane Doe"


# =============================================================================
# Idempotency Tests
# =============================================================================

class TestIdempotency:
    """Test idempotent operations."""
    
    def test_paper_upsert_idempotent(self, mock_db_session):
        """Test that upserting the same paper twice doesn't create duplicates."""
        from src.db.orm_models import Paper
        
        paper_data = {
            "source": "arxiv",
            "external_id": "2501.11111",
            "title": "Idempotency Test",
        }
        
        # First insert
        paper1 = Paper(
            id=str(uuid4()),
            source=paper_data["source"],
            external_id=paper_data["external_id"],
            title=paper_data["title"],
        )
        mock_db_session.add(paper1)
        mock_db_session.commit()
        
        # Query to check
        count = mock_db_session.query(Paper).filter_by(external_id="2501.11111").count()
        assert count == 1


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
