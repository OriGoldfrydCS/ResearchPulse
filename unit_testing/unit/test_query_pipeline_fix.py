"""
Regression tests for the PCA query pipeline fix.

Covers:
  A. Temporal phrase stripping from extracted topics
  B. Topics containing the word "recent" are preserved
  C. Multi-word arXiv query construction
  D. Word-number time parsing
  E. Existing prompt backward compatibility
  F. Output quality guardrail (low-relevance warning)
"""

import os
import sys
import pytest

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agent.prompt_controller import (
    PromptParser,
    PromptController,
    OutputEnforcer,
    ParsedPrompt,
    _parse_word_number,
)
from src.tools.fetch_arxiv import _build_topic_clause


# ============================================================
# A. PCA time phrase removal
# ============================================================

class TestTemporalPhraseStripping:
    """Bug 1: temporal clauses must not leak into extracted topic."""

    def test_pca_published_within_last_two_weeks(self):
        prompt = "provide recent research paper on PCA published within the last two weeks"
        parser = PromptParser()
        parsed = parser.parse(prompt)
        assert "published" not in parsed.interests_only.lower()
        assert "weeks" not in parsed.interests_only.lower()
        # The core topic must survive
        assert "PCA" in parsed.interests_only

    def test_pca_time_days_is_14(self):
        prompt = "provide recent research paper on PCA published within the last two weeks"
        parser = PromptParser()
        parsed = parser.parse(prompt)
        assert parsed.time_days == 14

    def test_from_the_last_3_days(self):
        prompt = "papers on TSNE from the last 3 days"
        parser = PromptParser()
        parsed = parser.parse(prompt)
        assert "from" not in parsed.interests_only.lower()
        assert "days" not in parsed.interests_only.lower()
        assert "TSNE" in parsed.interests_only

    def test_during_the_past_month(self):
        prompt = "research on transformers during the past month"
        parser = PromptParser()
        parsed = parser.parse(prompt)
        assert "during" not in parsed.interests_only.lower()
        assert "month" not in parsed.interests_only.lower()


# ============================================================
# B. Topic containing the word "recent" is preserved
# ============================================================

class TestRecentInTopic:
    """The word 'recent' inside a genuine topic must NOT be removed."""

    def test_recent_advances_preserved(self):
        prompt = "papers on Recent Advances in Graph Neural Networks"
        parser = PromptParser()
        parsed = parser.parse(prompt)
        assert "Recent Advances" in parsed.interests_only or "recent advances" in parsed.interests_only.lower()

    def test_recent_trends_preserved(self):
        prompt = "papers on Recent Trends in Drug Discovery"
        parser = PromptParser()
        parsed = parser.parse(prompt)
        assert "Recent Trends" in parsed.interests_only or "recent trends" in parsed.interests_only.lower()


# ============================================================
# C. Multi-word arXiv query construction
# ============================================================

class TestArxivQueryConstruction:
    """Bug 2: query builder must produce correct clauses."""

    def test_single_word_topic(self):
        clause = _build_topic_clause("PCA")
        assert clause == "(ti:PCA OR abs:PCA)"

    def test_multi_word_phrase_and_token(self):
        clause = _build_topic_clause("Principal Component Analysis")
        # Must contain phrase match
        assert 'ti:"Principal Component Analysis"' in clause
        assert 'abs:"Principal Component Analysis"' in clause
        # Must contain token AND match
        assert "ti:Principal AND ti:Component AND ti:Analysis" in clause

    def test_deep_reinforcement_learning(self):
        clause = _build_topic_clause("deep reinforcement learning")
        assert 'ti:"deep reinforcement learning"' in clause
        assert 'abs:"deep reinforcement learning"' in clause
        assert "ti:deep AND ti:reinforcement AND ti:learning" in clause


# ============================================================
# D. Word-number time parsing
# ============================================================

class TestWordNumberParsing:
    """Bug 3: word-based numbers must parse correctly."""

    def test_single_words(self):
        assert _parse_word_number("two") == 2
        assert _parse_word_number("ten") == 10
        assert _parse_word_number("fifteen") == 15
        assert _parse_word_number("nineteen") == 19

    def test_compound_words(self):
        assert _parse_word_number("twenty two") == 22
        assert _parse_word_number("forty five") == 45
        assert _parse_word_number("ninety nine") == 99

    def test_digit_strings(self):
        assert _parse_word_number("22") == 22
        assert _parse_word_number("7") == 7

    def test_last_twenty_two_weeks(self):
        prompt = "papers on PCA from the last twenty two weeks"
        parser = PromptParser()
        parsed = parser.parse(prompt)
        assert parsed.time_days == 22 * 7

    def test_last_forty_five_days(self):
        prompt = "papers on deep learning from the last forty five days"
        parser = PromptParser()
        parsed = parser.parse(prompt)
        assert parsed.time_days == 45


# ============================================================
# E. Existing prompts still work (backward compatibility)
# ============================================================

class TestBackwardCompatibility:
    """Existing digit-based prompts must not regress."""

    def test_find_5_papers_on_multi_armed_bandits(self):
        prompt = "find 5 papers on Multi Armed Bandits"
        parser = PromptParser()
        parsed = parser.parse(prompt)
        assert parsed.requested_count == 5
        assert "Multi Armed Bandits" in parsed.interests_only

    def test_recent_papers_tsne_last_3_days(self):
        prompt = "recent papers on TSNE from the last 3 days"
        parser = PromptParser()
        parsed = parser.parse(prompt)
        assert parsed.time_days == 3
        assert "TSNE" in parsed.interests_only

    def test_last_2_weeks_digit(self):
        prompt = "papers on GAN published within the last 2 weeks"
        parser = PromptParser()
        parsed = parser.parse(prompt)
        assert parsed.time_days == 14

    def test_last_month(self):
        prompt = "papers on NLP from the last month"
        parser = PromptParser()
        parsed = parser.parse(prompt)
        assert parsed.time_days == 30


# ============================================================
# F. Output quality guardrail
# ============================================================

class TestOutputQualityGuardrail:
    """Bug 4: low-relevance guardrail message."""

    def test_all_low_relevance_adds_message(self):
        papers = [
            {"title": "A", "relevance_score": 0.05},
            {"title": "B", "relevance_score": 0.08},
            {"title": "C", "relevance_score": 0.10},
        ]
        parsed = ParsedPrompt(requested_count=3)
        enforcer = OutputEnforcer()
        result = enforcer.enforce(papers, parsed)
        assert result.message is not None
        assert "none were strongly relevant" in result.message

    def test_mixed_relevance_no_message(self):
        papers = [
            {"title": "A", "relevance_score": 0.85},
            {"title": "B", "relevance_score": 0.05},
        ]
        parsed = ParsedPrompt(requested_count=2)
        enforcer = OutputEnforcer()
        result = enforcer.enforce(papers, parsed)
        # At least one paper is above 0.10 so no low-relevance warning
        assert result.message is None or "none were strongly relevant" not in result.message

    def test_papers_not_removed_by_guardrail(self):
        papers = [
            {"title": "A", "relevance_score": 0.02},
            {"title": "B", "relevance_score": 0.01},
        ]
        parsed = ParsedPrompt(requested_count=2)
        enforcer = OutputEnforcer()
        result = enforcer.enforce(papers, parsed)
        assert len(result.papers) == 2
