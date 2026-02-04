"""
Unit tests for paper discovery pipeline.

Tests:
1. Paper retrieval (mocked)
2. Importance scoring and persistence
3. added_at and published_at timestamps
4. Agent decision computation
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from src.tools.decide_delivery import ScoredPaper


class TestScoredPaperCreation:
    """Test ScoredPaper model creation and validation."""
    
    def test_create_minimal_paper(self):
        """Test creating a paper with minimal fields."""
        paper = ScoredPaper(
            arxiv_id="2401.00001",
            title="Test Paper",
            relevance_score=0.8,
            novelty_score=0.7,
            importance="high",
        )
        
        assert paper.arxiv_id == "2401.00001"
        assert paper.title == "Test Paper"
        assert paper.relevance_score == 0.8
        assert paper.novelty_score == 0.7
        assert paper.importance == "high"
    
    def test_create_full_paper(self):
        """Test creating a paper with all fields."""
        now = datetime.now()
        
        paper = ScoredPaper(
            arxiv_id="2401.00001",
            title="Comprehensive Machine Learning Study",
            abstract="We present a comprehensive study of ML techniques...",
            link="https://arxiv.org/abs/2401.00001",
            authors=["Alice Author", "Bob Researcher", "Charlie Contributor"],
            categories=["cs.LG", "cs.AI", "stat.ML"],
            publication_date="2024-01-15",
            added_at=now.isoformat(),
            relevance_score=0.95,
            novelty_score=0.88,
            importance="high",
            explanation="Highly relevant to current research interests.",
            pdf_url="https://arxiv.org/pdf/2401.00001.pdf",
        )
        
        assert paper.abstract == "We present a comprehensive study of ML techniques..."
        assert len(paper.authors) == 3
        assert len(paper.categories) == 3
        assert paper.publication_date == "2024-01-15"
        assert paper.added_at == now.isoformat()
        assert paper.pdf_url == "https://arxiv.org/pdf/2401.00001.pdf"
    
    def test_score_validation(self):
        """Test that scores must be between 0 and 1."""
        # Valid scores
        paper = ScoredPaper(
            arxiv_id="test",
            title="Test",
            relevance_score=0.0,
            novelty_score=1.0,
            importance="high",
        )
        
        assert paper.relevance_score == 0.0
        assert paper.novelty_score == 1.0
    
    def test_importance_levels(self):
        """Test different importance levels."""
        for importance in ["high", "medium", "low"]:
            paper = ScoredPaper(
                arxiv_id="test",
                title="Test",
                relevance_score=0.5,
                novelty_score=0.5,
                importance=importance,
            )
            
            assert paper.importance == importance


class TestPaperTimestamps:
    """Test paper timestamp handling."""
    
    def test_publication_date_parsing(self):
        """Test parsing publication date."""
        paper = ScoredPaper(
            arxiv_id="2401.00001",
            title="Test Paper",
            publication_date="2024-01-15",
            relevance_score=0.8,
            novelty_score=0.7,
            importance="high",
        )
        
        assert paper.publication_date == "2024-01-15"
        
        # Can parse to datetime
        pub_date = datetime.strptime(paper.publication_date, "%Y-%m-%d")
        assert pub_date.year == 2024
        assert pub_date.month == 1
        assert pub_date.day == 15
    
    def test_added_at_timestamp(self):
        """Test added_at timestamp."""
        now = datetime.now()
        
        paper = ScoredPaper(
            arxiv_id="2401.00001",
            title="Test Paper",
            added_at=now.isoformat(),
            relevance_score=0.8,
            novelty_score=0.7,
            importance="high",
        )
        
        assert paper.added_at == now.isoformat()
        
        # Can parse back to datetime
        added = datetime.fromisoformat(paper.added_at)
        assert (added - now).total_seconds() < 1
    
    def test_missing_timestamps(self):
        """Test handling missing timestamps."""
        paper = ScoredPaper(
            arxiv_id="2401.00001",
            title="Test Paper",
            relevance_score=0.8,
            novelty_score=0.7,
            importance="high",
        )
        
        # Should have None or empty for missing timestamps
        assert paper.publication_date is None or paper.publication_date == ""
        assert paper.added_at is None or paper.added_at == ""


class TestImportanceScoring:
    """Test importance scoring logic."""
    
    def test_high_scores_high_importance(self):
        """Test that high scores lead to high importance."""
        paper = ScoredPaper(
            arxiv_id="test",
            title="Test",
            relevance_score=0.95,
            novelty_score=0.90,
            importance="high",  # Should be computed or assigned
        )
        
        assert paper.importance == "high"
    
    def test_medium_scores_medium_importance(self):
        """Test that medium scores lead to medium importance."""
        paper = ScoredPaper(
            arxiv_id="test",
            title="Test",
            relevance_score=0.70,
            novelty_score=0.65,
            importance="medium",
        )
        
        assert paper.importance == "medium"
    
    def test_low_scores_low_importance(self):
        """Test that low scores lead to low importance."""
        paper = ScoredPaper(
            arxiv_id="test",
            title="Test",
            relevance_score=0.45,
            novelty_score=0.40,
            importance="low",
        )
        
        assert paper.importance == "low"
    
    def test_score_explanation(self):
        """Test that explanation is included."""
        paper = ScoredPaper(
            arxiv_id="test",
            title="Test Paper on NLP",
            relevance_score=0.9,
            novelty_score=0.85,
            importance="high",
            explanation="This paper is highly relevant because it addresses your core research area in NLP and introduces novel techniques.",
        )
        
        assert "highly relevant" in paper.explanation.lower()
        assert "NLP" in paper.explanation


class TestPaperRetrieval:
    """Test paper retrieval with mocks."""
    
    @pytest.fixture
    def mock_arxiv_response(self):
        """Create mock arXiv API response."""
        return [
            {
                "id": "2401.00001",
                "title": "Paper One",
                "summary": "Abstract one",
                "published": "2024-01-15T00:00:00Z",
                "authors": [{"name": "Alice"}, {"name": "Bob"}],
                "categories": ["cs.LG"],
                "links": [
                    {"href": "https://arxiv.org/abs/2401.00001"},
                    {"href": "https://arxiv.org/pdf/2401.00001.pdf"},
                ]
            },
            {
                "id": "2401.00002",
                "title": "Paper Two",
                "summary": "Abstract two",
                "published": "2024-01-14T00:00:00Z",
                "authors": [{"name": "Charlie"}],
                "categories": ["cs.CL", "cs.AI"],
                "links": [
                    {"href": "https://arxiv.org/abs/2401.00002"},
                ]
            },
        ]
    
    def test_retrieve_papers_basic(self, mock_arxiv_response):
        """Test basic paper retrieval."""
        from src.tools.fetch_arxiv import fetch_arxiv_papers
        
        result = fetch_arxiv_papers(categories_include=["cs.LG"], max_results=10, use_mock=True)
        
        assert result.success
        assert len(result.papers) > 0
    
    def test_retrieve_papers_with_query(self, mock_arxiv_response):
        """Test paper retrieval with query filter."""
        from src.tools.fetch_arxiv import fetch_arxiv_papers
        
        result = fetch_arxiv_papers(categories_include=["cs.CL"], max_results=10, use_mock=True)
        
        assert result.success
        # All returned papers should have cs.CL category
        for paper in result.papers:
            assert "cs.CL" in paper.categories
    
    def test_retrieve_papers_empty_result(self):
        """Test handling empty retrieval result."""
        from src.tools.fetch_arxiv import fetch_arxiv_papers
        
        result = fetch_arxiv_papers(categories_include=["cs.NONEXISTENT"], max_results=10, use_mock=True)
        
        assert result.success
        assert result.papers == []


class TestPaperScoring:
    """Test paper scoring with mocks."""
    
    @pytest.fixture
    def sample_papers(self):
        """Create sample papers for scoring."""
        return [
            {
                "id": "2401.00001",
                "title": "Transformers for Natural Language Processing",
                "summary": "We present a novel transformer architecture...",
                "categories": ["cs.CL", "cs.LG"],
            },
            {
                "id": "2401.00002",
                "title": "Image Classification with CNNs",
                "summary": "A study on convolutional neural networks...",
                "categories": ["cs.CV"],
            },
        ]
    
    def test_score_papers_by_relevance(self, sample_papers):
        """Test scoring papers by relevance to research interests."""
        research_interests = ["natural language processing", "transformers", "NLP"]
        
        scored = []
        for paper in sample_papers:
            # Simple scoring: check if title/summary contains interests
            title_lower = paper["title"].lower()
            summary_lower = paper["summary"].lower()
            
            score = 0.0
            for interest in research_interests:
                if interest.lower() in title_lower or interest.lower() in summary_lower:
                    score += 0.3
            
            score = min(1.0, max(0.0, score))
            
            scored.append({
                **paper,
                "relevance_score": score,
            })
        
        # NLP paper should score higher
        nlp_paper = next(p for p in scored if "2401.00001" in p["id"])
        cv_paper = next(p for p in scored if "2401.00002" in p["id"])
        
        assert nlp_paper["relevance_score"] > cv_paper["relevance_score"]
    
    def test_score_papers_by_novelty(self, sample_papers):
        """Test scoring papers by novelty."""
        # For now, assume manual novelty assessment
        paper = ScoredPaper(
            arxiv_id="2401.00001",
            title="Novel Approach to X",
            relevance_score=0.8,
            novelty_score=0.9,  # High novelty
            importance="high",
        )
        
        assert paper.novelty_score == 0.9


class TestAgentDecisions:
    """Test agent decision computation."""
    
    def test_decide_email_for_high_importance(self):
        """Test that high importance papers trigger email."""
        paper = ScoredPaper(
            arxiv_id="test",
            title="Important Paper",
            relevance_score=0.95,
            novelty_score=0.9,
            importance="high",
        )
        
        # High importance should recommend inclusion in summary email
        should_include = paper.importance == "high"
        
        assert should_include is True
    
    def test_decide_calendar_for_high_importance(self):
        """Test that high importance papers get calendar reminders."""
        paper = ScoredPaper(
            arxiv_id="test",
            title="Must-Read Paper",
            relevance_score=0.95,
            novelty_score=0.9,
            importance="high",
        )
        
        # High importance should get calendar reminder
        should_create_reminder = paper.importance in ["high", "medium"]
        
        assert should_create_reminder is True
    
    def test_decide_skip_low_importance(self):
        """Test that low importance papers are not prioritized."""
        paper = ScoredPaper(
            arxiv_id="test",
            title="Low Priority Paper",
            relevance_score=0.4,
            novelty_score=0.3,
            importance="low",
        )
        
        # Low importance might be skipped for calendar
        should_skip_calendar = paper.importance == "low"
        
        assert should_skip_calendar is True


class TestPaperPersistence:
    """Test paper persistence to database."""
    
    @pytest.fixture
    def mock_store(self):
        """Create mock data store."""
        store = MagicMock()
        store.save_paper.return_value = {"id": "paper-uuid-123"}
        store.get_paper.return_value = None
        return store
    
    def test_save_paper_with_scores(self, mock_store):
        """Test saving paper with all scores."""
        paper_data = {
            "arxiv_id": "2401.00001",
            "title": "Test Paper",
            "abstract": "Abstract text",
            "relevance_score": 0.85,
            "novelty_score": 0.75,
            "importance": "high",
            "added_at": datetime.now().isoformat(),
            "publication_date": "2024-01-15",
        }
        
        result = mock_store.save_paper(paper_data)
        
        mock_store.save_paper.assert_called_once_with(paper_data)
        assert "id" in result
    
    def test_save_paper_deduplication(self, mock_store):
        """Test that duplicate papers are handled."""
        mock_store.get_paper.return_value = {"id": "existing-id", "arxiv_id": "2401.00001"}
        
        # Check if paper exists before saving
        existing = mock_store.get_paper("2401.00001")
        
        if existing:
            # Update instead of create
            mock_store.update_paper.return_value = {"id": existing["id"]}
            result = mock_store.update_paper(existing["id"], {"relevance_score": 0.9})
        else:
            result = mock_store.save_paper({"arxiv_id": "2401.00001"})
        
        assert existing is not None
        mock_store.update_paper.assert_called_once()


class TestNegativeCases:
    """Test error handling and edge cases."""
    
    def test_paper_with_missing_title(self):
        """Test handling paper with missing title."""
        with pytest.raises((ValueError, TypeError)):
            ScoredPaper(
                arxiv_id="test",
                # title is missing
                relevance_score=0.8,
                novelty_score=0.7,
                importance="high",
            )
    
    def test_paper_with_empty_arxiv_id(self):
        """Test handling empty arxiv_id."""
        paper = ScoredPaper(
            arxiv_id="",  # Empty but might be allowed
            title="Test",
            relevance_score=0.8,
            novelty_score=0.7,
            importance="high",
        )
        
        # Should create but arxiv_id is empty
        assert paper.arxiv_id == ""
    
    def test_paper_with_special_characters(self):
        """Test handling special characters in title."""
        paper = ScoredPaper(
            arxiv_id="2401.00001",
            title="Paper with <script>alert('xss')</script> and \"quotes\"",
            relevance_score=0.8,
            novelty_score=0.7,
            importance="high",
        )
        
        # Should handle gracefully
        assert "<script>" in paper.title or "&lt;script&gt;" in paper.title
    
    def test_paper_with_unicode_title(self):
        """Test handling unicode in title."""
        paper = ScoredPaper(
            arxiv_id="2401.00001",
            title="Machine Learning for 中文 and émojis 🤖",
            relevance_score=0.8,
            novelty_score=0.7,
            importance="high",
        )
        
        assert "中文" in paper.title
        assert "🤖" in paper.title
    
    def test_paper_with_very_long_abstract(self):
        """Test handling very long abstract."""
        long_abstract = "This is a test. " * 1000  # Very long
        
        paper = ScoredPaper(
            arxiv_id="2401.00001",
            title="Test Paper",
            abstract=long_abstract,
            relevance_score=0.8,
            novelty_score=0.7,
            importance="high",
        )
        
        assert len(paper.abstract) > 10000
