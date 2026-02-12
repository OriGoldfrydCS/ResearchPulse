"""
Unit tests for recipient_resolution module.

Tests interest overlap, owner-only matches, colleague-only matches,
both-match rule, de-duplication, and disabled colleague filtering.
"""

import pytest
from src.tools.recipient_resolution import (
    MatchRule,
    RecipientEntry,
    resolve_recipients,
    _interests_overlap,
)


# =============================================================================
# Interest Overlap
# =============================================================================

class TestInterestsOverlap:
    """Test the _interests_overlap helper."""

    def test_exact_topic_match(self):
        matched, terms = _interests_overlap(
            paper_topics=["machine learning", "data science"],
            paper_categories=["cs.LG"],
            interest_topics=["machine learning"],
            interest_categories=[],
        )
        assert matched is True
        assert "machine learning" in terms

    def test_category_match(self):
        matched, terms = _interests_overlap(
            paper_topics=[],
            paper_categories=["cs.AI", "cs.CL"],
            interest_topics=[],
            interest_categories=["cs.AI"],
        )
        assert matched is True
        assert "cs.ai" in terms

    def test_substring_match(self):
        matched, terms = _interests_overlap(
            paper_topics=["reinforcement learning algorithms"],
            paper_categories=[],
            interest_topics=["reinforcement learning"],
            interest_categories=[],
        )
        assert matched is True

    def test_no_match(self):
        matched, terms = _interests_overlap(
            paper_topics=["quantum physics"],
            paper_categories=["quant-ph"],
            interest_topics=["biology"],
            interest_categories=["q-bio"],
        )
        assert matched is False
        assert terms == []


# =============================================================================
# Resolve Recipients
# =============================================================================

def _make_paper(title="Test Paper on Machine Learning", cats=None, abstract=""):
    return {
        "arxiv_id": "2401.00001",
        "title": title,
        "abstract": abstract,
        "categories": cats or ["cs.LG"],
    }


def _make_colleague(email, name="Col", topics=None, categories=None, enabled=True, auto_send=True):
    return {
        "id": f"col-{email}",
        "email": email,
        "name": name,
        "topics": topics or [],
        "categories": categories or [],
        "enabled": enabled,
        "auto_send_emails": auto_send,
    }


class TestResolveRecipients:
    """Test the resolve_recipients main function."""

    def test_owner_only_match(self):
        """Owner matches, no colleagues → single owner entry."""
        paper = _make_paper(cats=["cs.LG"])
        result = resolve_recipients(
            paper,
            owner_email="owner@test.com",
            owner_name="Owner",
            owner_topics=["machine learning"],
            owner_categories=["cs.LG"],
            colleagues=[],
        )
        assert len(result) == 1
        assert result[0].is_owner is True
        assert result[0].match_rule == MatchRule.OWNER_ONLY

    def test_colleague_only_match(self):
        """Colleague matches, owner doesn't → colleague-only entry."""
        paper = _make_paper(title="Quantum Entanglement Study", cats=["quant-ph"])
        colleague = _make_colleague(
            "col@test.com", topics=["quantum"], categories=["quant-ph"]
        )
        result = resolve_recipients(
            paper,
            owner_email="owner@test.com",
            owner_name="Owner",
            owner_topics=["biology"],
            owner_categories=["q-bio"],
            colleagues=[colleague],
        )
        assert len(result) == 1
        assert result[0].is_owner is False
        assert result[0].email == "col@test.com"
        assert result[0].match_rule == MatchRule.COLLEAGUE_ONLY

    def test_both_match(self):
        """Owner and colleague both match → two entries."""
        paper = _make_paper(cats=["cs.LG"])
        colleague = _make_colleague(
            "col@test.com", topics=["machine learning"], categories=["cs.LG"]
        )
        result = resolve_recipients(
            paper,
            owner_email="owner@test.com",
            owner_name="Owner",
            owner_topics=["machine learning"],
            owner_categories=["cs.LG"],
            colleagues=[colleague],
        )
        assert len(result) == 2
        emails = {r.email for r in result}
        assert "owner@test.com" in emails
        assert "col@test.com" in emails

    def test_deduplication_same_email(self):
        """If owner and colleague have same email, de-duplicate to one entry."""
        paper = _make_paper(cats=["cs.LG"])
        colleague = _make_colleague(
            "owner@test.com", topics=["machine learning"], categories=["cs.LG"]
        )
        result = resolve_recipients(
            paper,
            owner_email="owner@test.com",
            owner_name="Owner",
            owner_topics=["machine learning"],
            owner_categories=["cs.LG"],
            colleagues=[colleague],
        )
        assert len(result) == 1
        assert result[0].match_rule == MatchRule.BOTH

    def test_disabled_colleague_excluded(self):
        """Disabled colleagues should be skipped."""
        paper = _make_paper(cats=["cs.LG"])
        colleague = _make_colleague(
            "col@test.com", categories=["cs.LG"], enabled=False
        )
        result = resolve_recipients(
            paper,
            owner_email="owner@test.com",
            owner_name="Owner",
            owner_topics=["machine learning"],
            owner_categories=["cs.LG"],
            colleagues=[colleague],
        )
        assert len(result) == 1  # owner only
        assert result[0].is_owner is True

    def test_auto_send_disabled_excluded(self):
        """Colleagues with auto_send_emails=False should be skipped."""
        paper = _make_paper(cats=["cs.LG"])
        colleague = _make_colleague(
            "col@test.com", categories=["cs.LG"], auto_send=False
        )
        result = resolve_recipients(
            paper,
            owner_email="owner@test.com",
            owner_name="Owner",
            owner_topics=["machine learning"],
            owner_categories=["cs.LG"],
            colleagues=[colleague],
        )
        assert len(result) == 1
        assert result[0].is_owner is True

    def test_no_match_returns_empty(self):
        """When neither owner nor colleagues match, return empty list."""
        paper = _make_paper(title="Galactic Formation", cats=["astro-ph"])
        result = resolve_recipients(
            paper,
            owner_email="owner@test.com",
            owner_name="Owner",
            owner_topics=["biology"],
            owner_categories=["q-bio"],
            colleagues=[],
        )
        assert result == []

    def test_multiple_colleagues(self):
        """Multiple colleagues with different matches."""
        paper = _make_paper(title="Deep Learning for NLP", cats=["cs.LG", "cs.CL"])
        colleagues = [
            _make_colleague("alice@test.com", categories=["cs.LG"]),
            _make_colleague("bob@test.com", categories=["cs.CL"]),
            _make_colleague("carol@test.com", categories=["math.CO"]),  # no match
        ]
        result = resolve_recipients(
            paper,
            owner_email="owner@test.com",
            owner_name="Owner",
            owner_topics=[],
            owner_categories=[],
            colleagues=colleagues,
        )
        emails = {r.email for r in result}
        assert "alice@test.com" in emails
        assert "bob@test.com" in emails
        assert "carol@test.com" not in emails

    def test_colleague_id_included(self):
        """Each colleague entry should carry its colleague_id."""
        paper = _make_paper(cats=["cs.LG"])
        colleague = _make_colleague("col@test.com", categories=["cs.LG"])
        result = resolve_recipients(
            paper,
            owner_email="owner@test.com",
            owner_name="Owner",
            owner_topics=[],
            owner_categories=[],
            colleagues=[colleague],
        )
        col_entry = [r for r in result if not r.is_owner][0]
        assert col_entry.colleague_id == "col-col@test.com"
