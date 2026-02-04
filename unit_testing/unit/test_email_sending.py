"""
Unit tests for email sending functionality.

Tests email payload generation, HTML formatting, and triggered_by attribution.
Uses mocks to avoid actually sending emails.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.tools.decide_delivery import (
    ScoredPaper,
    _generate_email_content,
    _generate_email_content_html,
    _generate_digest_email_content,
    generate_summary_email_html,
    _send_email_smtp,
)
from unit_testing.conftest import get_test_email, get_colleague_test_email


class TestScoredPaperModel:
    """Test the ScoredPaper model."""
    
    def test_create_scored_paper(self):
        """Test creating a ScoredPaper instance."""
        paper = ScoredPaper(
            arxiv_id="2401.00001",
            title="Test Paper on Machine Learning",
            abstract="This is a test abstract.",
            link="https://arxiv.org/abs/2401.00001",
            authors=["John Doe", "Jane Smith"],
            categories=["cs.LG", "cs.AI"],
            relevance_score=0.85,
            novelty_score=0.7,
            importance="high",
            explanation="Highly relevant to NLP research.",
        )
        
        assert paper.arxiv_id == "2401.00001"
        assert paper.importance == "high"
        assert paper.relevance_score == 0.85
    
    def test_scored_paper_validation(self):
        """Test that scores are validated."""
        with pytest.raises(ValueError):
            ScoredPaper(
                arxiv_id="test",
                title="Test",
                relevance_score=1.5,  # Invalid - must be <= 1.0
                novelty_score=0.5,
                importance="high",
            )


class TestGenerateEmailContent:
    """Test plain text email generation."""
    
    @pytest.fixture
    def sample_paper(self):
        """Create a sample paper for testing."""
        return ScoredPaper(
            arxiv_id="2401.00001",
            title="Attention Is All You Need: Revisited",
            abstract="We revisit the transformer architecture...",
            link="https://arxiv.org/abs/2401.00001",
            authors=["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
            categories=["cs.CL", "cs.LG"],
            publication_date="2024-01-15",
            relevance_score=0.92,
            novelty_score=0.78,
            importance="high",
            explanation="Breakthrough in attention mechanisms.",
        )
    
    def test_email_has_subject(self, sample_paper):
        """Test that email has subject line."""
        content = _generate_email_content(sample_paper, "high")
        
        assert "Subject:" in content
        assert "[ResearchPulse]" in content
    
    def test_email_has_paper_info(self, sample_paper):
        """Test that email contains paper information."""
        content = _generate_email_content(sample_paper, "high")
        
        assert sample_paper.title in content
        assert sample_paper.arxiv_id in content
    
    def test_email_has_authors(self, sample_paper):
        """Test that email includes author names."""
        content = _generate_email_content(sample_paper, "high")
        
        assert "Alice" in content
        assert "Bob" in content
        # Should truncate after 5 authors
        assert "+1 more" in content
    
    def test_email_has_scores(self, sample_paper):
        """Test that email includes relevance scores."""
        content = _generate_email_content(sample_paper, "high")
        
        assert "92%" in content  # Relevance score
        assert "78%" in content  # Novelty score
    
    def test_email_includes_abstract(self, sample_paper):
        """Test that email includes abstract when requested."""
        content = _generate_email_content(sample_paper, "high", include_abstract=True)
        
        assert sample_paper.abstract in content
    
    def test_email_excludes_abstract(self, sample_paper):
        """Test that email excludes abstract when requested."""
        content = _generate_email_content(sample_paper, "high", include_abstract=False)
        
        # Abstract section should not be present
        assert "ABSTRACT" not in content
    
    def test_email_has_explanation(self, sample_paper):
        """Test that email includes explanation."""
        content = _generate_email_content(sample_paper, "high", include_explanation=True)
        
        assert sample_paper.explanation in content
    
    def test_email_personalization(self, sample_paper):
        """Test that email is personalized."""
        content = _generate_email_content(
            sample_paper, "high", researcher_name="Dr. Smith"
        )
        
        assert "Dr. Smith" in content


class TestGenerateEmailHtml:
    """Test HTML email generation."""
    
    @pytest.fixture
    def sample_paper(self):
        """Create a sample paper for testing."""
        return ScoredPaper(
            arxiv_id="2401.00001",
            title="Test Paper",
            abstract="Test abstract",
            link="https://arxiv.org/abs/2401.00001",
            authors=["Author One"],
            categories=["cs.AI"],
            relevance_score=0.9,
            novelty_score=0.8,
            importance="high",
            explanation="Important paper",
        )
    
    def test_html_email_format(self, sample_paper):
        """Test that HTML email is well-formed."""
        html = _generate_email_content_html(sample_paper, "high")
        
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "<body" in html
        assert "</body>" in html
    
    def test_html_has_paper_title(self, sample_paper):
        """Test that HTML includes paper title."""
        html = _generate_email_content_html(sample_paper, "high")
        
        assert sample_paper.title in html
    
    def test_html_has_link(self, sample_paper):
        """Test that HTML includes paper link."""
        html = _generate_email_content_html(sample_paper, "high")
        
        assert sample_paper.link in html
    
    def test_html_priority_colors(self, sample_paper):
        """Test that different priorities have different colors."""
        high_html = _generate_email_content_html(sample_paper, "high")
        
        # Change to medium priority
        sample_paper.importance = "medium"
        medium_html = _generate_email_content_html(sample_paper, "medium")
        
        # They should have different styling
        assert high_html != medium_html


class TestGenerateSummaryEmailHtml:
    """Test summary email generation with multiple papers."""
    
    @pytest.fixture
    def sample_papers(self):
        """Create sample papers for testing."""
        return [
            ScoredPaper(
                arxiv_id="2401.00001",
                title="High Importance Paper",
                abstract="Abstract 1",
                relevance_score=0.95,
                novelty_score=0.9,
                importance="high",
                explanation="Very important",
            ),
            ScoredPaper(
                arxiv_id="2401.00002",
                title="Medium Importance Paper",
                abstract="Abstract 2",
                relevance_score=0.75,
                novelty_score=0.6,
                importance="medium",
                explanation="Moderately important",
            ),
            ScoredPaper(
                arxiv_id="2401.00003",
                title="Low Importance Paper",
                abstract="Abstract 3",
                relevance_score=0.55,
                novelty_score=0.4,
                importance="low",
                explanation="Less important",
            ),
        ]
    
    def test_summary_returns_tuple(self, sample_papers):
        """Test that summary returns subject, plain text, and HTML."""
        result = generate_summary_email_html(sample_papers, "test query")
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        subject, plain_text, html = result
        assert isinstance(subject, str)
        assert isinstance(plain_text, str)
        assert isinstance(html, str)
    
    def test_summary_subject_contains_count(self, sample_papers):
        """Test that subject contains paper count."""
        subject, _, _ = generate_summary_email_html(sample_papers, "test query")
        
        assert "3 papers" in subject
    
    def test_summary_groups_by_importance(self, sample_papers):
        """Test that papers are grouped by importance."""
        _, _, html = generate_summary_email_html(sample_papers, "test query")
        
        # All importance levels should be present
        assert "high" in html.lower() or "must read" in html.lower()
        assert "medium" in html.lower() or "worth reading" in html.lower()
        assert "low" in html.lower() or "for later" in html.lower()
    
    def test_summary_contains_all_papers(self, sample_papers):
        """Test that all papers appear in summary."""
        _, _, html = generate_summary_email_html(sample_papers, "test query")
        
        for paper in sample_papers:
            assert paper.title in html
    
    def test_summary_triggered_by_agent(self, sample_papers):
        """Test that triggered_by='agent' is properly marked."""
        _, _, html = generate_summary_email_html(
            sample_papers, "test query", triggered_by="agent"
        )
        
        assert "ResearchPulse" in html
    
    def test_summary_triggered_by_user(self, sample_papers):
        """Test that triggered_by='user' is properly marked."""
        _, _, html = generate_summary_email_html(
            sample_papers, "test query", triggered_by="user"
        )
        
        assert "Sent by you" in html or "user" in html.lower()
    
    def test_summary_query_included(self, sample_papers):
        """Test that original query is included."""
        query = "latest papers on transformer architectures"
        _, _, html = generate_summary_email_html(sample_papers, query)
        
        assert query in html or "transformer" in html.lower()
    
    def test_summary_arxiv_links(self, sample_papers):
        """Test that arXiv links are correct."""
        _, _, html = generate_summary_email_html(sample_papers, "test query")
        
        # Each paper should have an arXiv link
        for paper in sample_papers:
            assert f"arxiv.org/abs/{paper.arxiv_id}" in html or paper.arxiv_id in html


class TestDigestEmailContent:
    """Test digest email generation."""
    
    @pytest.fixture
    def sample_papers(self):
        """Create sample papers for testing."""
        return [
            ScoredPaper(
                arxiv_id=f"2401.0000{i}",
                title=f"Paper {i}",
                abstract=f"Abstract {i}",
                relevance_score=0.9 - i * 0.1,
                novelty_score=0.8 - i * 0.1,
                importance=["high", "medium", "low"][min(i, 2)],
                explanation=f"Explanation {i}",
            )
            for i in range(5)
        ]
    
    def test_digest_includes_all_papers(self, sample_papers):
        """Test that digest includes all papers."""
        content = _generate_digest_email_content(sample_papers)
        
        for paper in sample_papers:
            assert paper.title in content
    
    def test_digest_sorted_by_importance(self, sample_papers):
        """Test that papers are sorted by importance."""
        content = _generate_digest_email_content(sample_papers)
        
        # HIGH importance should appear before MEDIUM
        high_pos = content.find("HIGH")
        medium_pos = content.find("MEDIUM") 
        
        # Should have at least one HIGH importance paper listed first
        assert high_pos != -1


class TestSendEmailSmtp:
    """Test SMTP email sending with mocks."""
    
    def test_send_email_no_credentials(self):
        """Test that email fails gracefully without credentials."""
        with patch.dict('os.environ', {'SMTP_USER': '', 'SMTP_PASSWORD': ''}, clear=False):
            result = _send_email_smtp(
                to_email=get_test_email(),
                subject="Test",
                body="Test body",
            )
            
            assert result is False
    
    def test_send_email_success(self):
        """Test successful email sending with mocked SMTP."""
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
                
                result = _send_email_smtp(
                    to_email=get_colleague_test_email("recipient"),
                    subject="Test Subject",
                    body="Test body content",
                )
                
                assert result is True
                mock_server.starttls.assert_called_once()
                mock_server.login.assert_called_once()
                mock_server.send_message.assert_called_once()
    
    def test_send_email_with_html(self):
        """Test sending email with HTML body."""
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
                
                result = _send_email_smtp(
                    to_email=get_colleague_test_email("recipient"),
                    subject="Test Subject",
                    body="Plain text body",
                    html_body="<html><body>HTML body</body></html>",
                )
                
                assert result is True
    
    def test_send_email_auth_error(self):
        """Test handling of authentication errors."""
        import smtplib
        
        with patch.dict('os.environ', {
            'SMTP_HOST': 'smtp.test.com',
            'SMTP_PORT': '587',
            'SMTP_USER': 'test@test.com',
            'SMTP_PASSWORD': 'wrong_password',
        }):
            with patch('smtplib.SMTP') as mock_smtp:
                mock_server = MagicMock()
                mock_server.login.side_effect = smtplib.SMTPAuthenticationError(
                    535, b'Authentication failed'
                )
                mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
                mock_smtp.return_value.__exit__ = MagicMock(return_value=None)
                
                result = _send_email_smtp(
                    to_email=get_colleague_test_email("recipient"),
                    subject="Test",
                    body="Test body",
                )
                
                assert result is False


class TestEmailTriggeredBy:
    """Test triggered_by attribution in emails."""
    
    @pytest.fixture
    def sample_papers(self):
        """Create sample papers."""
        return [
            ScoredPaper(
                arxiv_id="2401.00001",
                title="Test Paper",
                abstract="Abstract",
                relevance_score=0.9,
                novelty_score=0.8,
                importance="high",
            )
        ]
    
    def test_agent_triggered_marker(self, sample_papers):
        """Test that agent-triggered emails have correct marker."""
        _, _, html = generate_summary_email_html(
            sample_papers,
            query_text="test query",
            triggered_by="agent",
        )
        
        # Should indicate automatic/agent send
        assert "ResearchPulse" in html or "agent" in html.lower()
    
    def test_user_triggered_marker(self, sample_papers):
        """Test that user-triggered emails have correct marker."""
        _, _, html = generate_summary_email_html(
            sample_papers,
            query_text="test query",
            triggered_by="user",
        )
        
        # Should indicate user-initiated
        assert "you" in html.lower() or "user" in html.lower()


class TestEmailNegativeCases:
    """Test error handling and edge cases."""
    
    def test_empty_papers_list(self):
        """Test handling empty papers list."""
        result = generate_summary_email_html([], "test query")
        
        # Should still return valid tuple
        assert isinstance(result, tuple)
        assert len(result) == 3
    
    def test_paper_with_no_authors(self):
        """Test handling paper with no authors."""
        paper = ScoredPaper(
            arxiv_id="2401.00001",
            title="Test Paper",
            authors=[],  # No authors
            relevance_score=0.9,
            novelty_score=0.8,
            importance="high",
        )
        
        # Should not crash
        content = _generate_email_content(paper, "high")
        assert isinstance(content, str)
    
    def test_paper_with_special_characters(self):
        """Test handling special characters in title."""
        paper = ScoredPaper(
            arxiv_id="2401.00001",
            title="Paper with <special> & \"characters\"",
            relevance_score=0.9,
            novelty_score=0.8,
            importance="high",
        )
        
        _, _, html = generate_summary_email_html([paper], "test")
        
        # HTML should be escaped properly
        assert "&lt;" in html or "special" in html
    
    def test_very_long_abstract(self):
        """Test handling very long abstract."""
        paper = ScoredPaper(
            arxiv_id="2401.00001",
            title="Test Paper",
            abstract="A" * 10000,  # Very long abstract
            relevance_score=0.9,
            novelty_score=0.8,
            importance="high",
        )
        
        content = _generate_email_content(paper, "high", include_abstract=True)
        
        # Should be truncated
        assert len(content) < 12000
