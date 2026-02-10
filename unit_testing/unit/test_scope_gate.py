"""
Unit tests for the scope_gate module.

Tests that classify_user_request correctly routes:
    - IN_SCOPE messages to the normal agent flow
    - OUT_OF_SCOPE_ARXIV_ONLY messages with the right template
    - OUT_OF_SCOPE_GENERAL messages with the right template
    - Edge cases (missing topic, summarize without ID, non-arXiv venues, explain)
"""

import os
import sys
import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.agent.scope_gate import (
    classify_user_request,
    ScopeClass,
    ScopeResult,
    RESPONSE_OUT_OF_SCOPE_GENERAL,
    RESPONSE_OUT_OF_SCOPE_ARXIV_ONLY,
    RESPONSE_MISSING_TOPIC,
    RESPONSE_NEED_ARXIV_LINK,
    RESPONSE_NON_ARXIV_VENUE,
)


# =============================================================================
# IN_SCOPE — should proceed to the normal agent handler
# =============================================================================

class TestInScope:
    """Messages that MUST be classified as IN_SCOPE."""

    @pytest.mark.parametrize(
        "message",
        [
            "Find recent arXiv papers about diffusion models for medical imaging",
            "Search arXiv for RAG evaluation benchmarks",
            "Show the top 5 papers this week on graph neural networks",
            "Summarize this arXiv paper: 2301.00001",
            "Summarize arXiv:2301.00001",
            "Summarize https://arxiv.org/abs/2301.00001",
            "Track authors and alert me on transformers",
            "What's new on arXiv in cs.CL today?",
            "latest papers on reinforcement learning",
            "find papers about natural language processing",
            "recent preprints on computer vision",
            "papers on deep learning for drug discovery",
            "show me papers in stat.ML",
            "get papers on attention mechanisms",
            "recommend papers about large language models",
            "new papers on GANs",
            "search for papers about BERT fine-tuning",
            "papers about transformer architecture",
        ],
    )
    def test_in_scope_arxiv_queries(self, message: str):
        result = classify_user_request(message)
        assert result.scope == ScopeClass.IN_SCOPE, (
            f"Expected IN_SCOPE for: {message!r}, got {result.scope.value} "
            f"(reason={result.reason})"
        )

    @pytest.mark.parametrize(
        "message",
        [
            "share this paper with my colleague",
            "show my colleagues",
            "update my email settings",
            "change delivery policy",
            "check my inbox",
            "show my reading list",
            "show run history",
            "health check",
            "update my profile",
            "set a reminder for this paper",
            "add a calendar event",
            "show my categories",
        ],
    )
    def test_in_scope_operational(self, message: str):
        result = classify_user_request(message)
        assert result.scope == ScopeClass.IN_SCOPE, (
            f"Expected IN_SCOPE for operational: {message!r}, got {result.scope.value}"
        )

    def test_in_scope_with_research_hints(self):
        result = classify_user_request("survey on optimization")
        assert result.scope == ScopeClass.IN_SCOPE


# =============================================================================
# OUT_OF_SCOPE_GENERAL — completely off-topic
# =============================================================================

class TestOutOfScopeGeneral:
    """Messages that MUST be classified as OUT_OF_SCOPE_GENERAL."""

    @pytest.mark.parametrize(
        "message",
        [
            "Tell me a joke",
            "What is the weather today?",
            "Write me a resume",
            "Help me write code for a web app",
            "Who is the president of the United States?",
            "Write my homework essay about Shakespeare",
            "Tell me a fun fact",
            "Help me debug my Python script",
            "What time is it?",
            "Play a game with me",
            "Translate this to French",
            "Give me a recipe for chocolate cake",
            "Bitcoin price prediction",
            "Best restaurants in New York",
        ],
    )
    def test_out_of_scope_general(self, message: str):
        result = classify_user_request(message)
        assert result.scope == ScopeClass.OUT_OF_SCOPE_GENERAL, (
            f"Expected OUT_OF_SCOPE_GENERAL for: {message!r}, "
            f"got {result.scope.value} (reason={result.reason})"
        )
        assert result.response == RESPONSE_OUT_OF_SCOPE_GENERAL


# =============================================================================
# OUT_OF_SCOPE_ARXIV_ONLY — research-adjacent but not arXiv
# =============================================================================

class TestOutOfScopeArxivOnly:
    """Messages that are research-related but reference non-arXiv sources."""

    @pytest.mark.parametrize(
        "message",
        [
            "Search Google Scholar for papers on NLP",
            "Find papers on PubMed about cancer",
            "Search IEEE for robotics papers",
            "Get papers from ACM Digital Library",
            "Find Nature articles about CRISPR",
            "Search Scopus for citations",
            "Find papers on Semantic Scholar",
        ],
    )
    def test_non_arxiv_venues(self, message: str):
        result = classify_user_request(message)
        assert result.scope == ScopeClass.OUT_OF_SCOPE_ARXIV_ONLY, (
            f"Expected OUT_OF_SCOPE_ARXIV_ONLY for: {message!r}, "
            f"got {result.scope.value} (reason={result.reason})"
        )

    @pytest.mark.parametrize(
        "message",
        [
            "Explain backpropagation",
            "What is a convolutional neural network?",
            "Describe how attention works",
            "Explain gradient descent to me",
        ],
    )
    def test_explain_without_papers(self, message: str):
        result = classify_user_request(message)
        assert result.scope == ScopeClass.OUT_OF_SCOPE_ARXIV_ONLY, (
            f"Expected OUT_OF_SCOPE_ARXIV_ONLY for: {message!r}, "
            f"got {result.scope.value} (reason={result.reason})"
        )
        assert result.suggested_rewrite is not None

    def test_explain_with_papers_stays_in_scope(self):
        """If the user says 'explain' but also mentions papers, keep in scope."""
        result = classify_user_request(
            "Explain recent papers on diffusion models"
        )
        assert result.scope == ScopeClass.IN_SCOPE


# =============================================================================
# Edge cases
# =============================================================================

class TestEdgeCases:
    """Specific edge cases from the spec."""

    def test_missing_topic_follow_up(self):
        """'Find papers' without a topic should return the missing-topic prompt."""
        result = classify_user_request("find papers")
        assert result.scope == ScopeClass.IN_SCOPE
        assert result.response == RESPONSE_MISSING_TOPIC

    def test_latest_papers_no_topic(self):
        result = classify_user_request("latest papers")
        assert result.scope == ScopeClass.IN_SCOPE
        assert result.response == RESPONSE_MISSING_TOPIC

    def test_summarize_without_arxiv_id(self):
        """'Summarize this text' without arXiv link should ask for one."""
        result = classify_user_request("Summarize this paragraph about cats")
        assert result.scope == ScopeClass.IN_SCOPE
        assert result.response == RESPONSE_NEED_ARXIV_LINK

    def test_summarize_with_arxiv_id_in_scope(self):
        result = classify_user_request("Summarize arXiv:2301.00001")
        assert result.scope == ScopeClass.IN_SCOPE
        # No early response — agent should handle it
        assert result.response is None

    def test_non_arxiv_venue_offers_redirect(self):
        result = classify_user_request("Search IEEE for robotics papers")
        assert result.scope == ScopeClass.OUT_OF_SCOPE_ARXIV_ONLY
        assert result.response == RESPONSE_NON_ARXIV_VENUE

    def test_non_arxiv_venue_with_arxiv_stays_in_scope(self):
        """If the user mentions both IEEE and arXiv, keep in scope."""
        result = classify_user_request(
            "Search IEEE and arXiv for robotics papers"
        )
        assert result.scope == ScopeClass.IN_SCOPE

    def test_very_short_message(self):
        result = classify_user_request("hi")
        assert result.scope == ScopeClass.IN_SCOPE
        assert result.response == RESPONSE_MISSING_TOPIC

    def test_empty_message(self):
        result = classify_user_request("  ")
        assert result.scope == ScopeClass.IN_SCOPE
        assert result.response == RESPONSE_MISSING_TOPIC

    def test_general_with_paper_mention_stays_in_scope(self):
        """If a general request also mentions papers, keep in scope."""
        result = classify_user_request(
            "Help me write a summary of this paper from arXiv"
        )
        assert result.scope == ScopeClass.IN_SCOPE

    def test_no_matching_signal_falls_to_general(self):
        result = classify_user_request("I like pizza and movies")
        assert result.scope == ScopeClass.OUT_OF_SCOPE_GENERAL


# =============================================================================
# Response template correctness
# =============================================================================

class TestResponseTemplates:
    """Ensure response templates are non-empty and well-formed."""

    def test_out_of_scope_general_template(self):
        assert "arXiv" in RESPONSE_OUT_OF_SCOPE_GENERAL
        assert "autonomous agent" in RESPONSE_OUT_OF_SCOPE_GENERAL
        assert len(RESPONSE_OUT_OF_SCOPE_GENERAL) > 50

    def test_out_of_scope_arxiv_only_template(self):
        assert "arXiv" in RESPONSE_OUT_OF_SCOPE_ARXIV_ONLY
        assert "keywords" in RESPONSE_OUT_OF_SCOPE_ARXIV_ONLY

    def test_missing_topic_template(self):
        assert "topic" in RESPONSE_MISSING_TOPIC

    def test_need_arxiv_link_template(self):
        assert "arXiv" in RESPONSE_NEED_ARXIV_LINK

    def test_non_arxiv_venue_template(self):
        assert "arXiv" in RESPONSE_NON_ARXIV_VENUE
