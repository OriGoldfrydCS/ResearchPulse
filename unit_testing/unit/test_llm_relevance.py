"""
Unit tests for LLM Relevance Filter.

Tests cover:
- LLMRelevanceResult model structure
- LLMRelevanceFilter caching
- LLM call mocking and response parsing
- evaluate_paper_relevance_with_llm integration function
- Feature flag gating
- Fallback behavior on errors
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta


# =========================================================================
# Data Models
# =========================================================================

class TestLLMRelevanceResult:
    """Test LLMRelevanceResult Pydantic model."""

    def test_result_model_creation(self):
        from src.tools.llm_relevance import LLMRelevanceResult

        result = LLMRelevanceResult(
            arxiv_id="2501.00123",
            is_relevant=True,
            is_excluded=False,
            relevance_score=0.85,
            excluded_topic_match="",
            reasoning="Directly about multi-armed bandits.",
            model_used="gpt-4o-mini",
            tokens_used=150,
        )
        assert result.arxiv_id == "2501.00123"
        assert result.is_relevant is True
        assert result.is_excluded is False
        assert result.relevance_score == 0.85
        assert result.reasoning == "Directly about multi-armed bandits."

    def test_result_model_defaults(self):
        from src.tools.llm_relevance import LLMRelevanceResult

        result = LLMRelevanceResult(arxiv_id="test")
        assert result.is_relevant is True
        assert result.is_excluded is False
        assert result.relevance_score == 0.5
        assert result.excluded_topic_match == ""
        assert result.is_cached is False

    def test_excluded_result(self):
        from src.tools.llm_relevance import LLMRelevanceResult

        result = LLMRelevanceResult(
            arxiv_id="2501.99999",
            is_relevant=False,
            is_excluded=True,
            relevance_score=0.1,
            excluded_topic_match="Attention",
            reasoning="Paper is primarily about attention mechanisms.",
        )
        assert result.is_excluded is True
        assert result.excluded_topic_match == "Attention"
        assert result.relevance_score == 0.1


# =========================================================================
# LLM Relevance Filter
# =========================================================================

class TestLLMRelevanceFilter:
    """Test LLMRelevanceFilter class."""

    def test_filter_creation(self):
        from src.tools.llm_relevance import LLMRelevanceFilter

        f = LLMRelevanceFilter()
        assert f.model == "gpt-4o-mini"
        assert f.temperature == 0.1
        assert f.max_tokens == 300
        assert f.cache_ttl_hours == 48

    def test_filter_custom_params(self):
        from src.tools.llm_relevance import LLMRelevanceFilter

        f = LLMRelevanceFilter(
            model="gpt-4o",
            temperature=0.3,
            max_tokens=500,
            cache_ttl_hours=24,
        )
        assert f.model == "gpt-4o"
        assert f.temperature == 0.3

    def test_cache_key_deterministic(self):
        from src.tools.llm_relevance import LLMRelevanceFilter

        f = LLMRelevanceFilter()
        key1 = f._cache_key("123", ["bandits", "pca"], ["attention"])
        key2 = f._cache_key("123", ["pca", "bandits"], ["attention"])  # different order
        # Should be same cache key regardless of order (sorted internally)
        assert key1 == key2

    def test_cache_key_differs_for_different_papers(self):
        from src.tools.llm_relevance import LLMRelevanceFilter

        f = LLMRelevanceFilter()
        key1 = f._cache_key("123", ["bandits"], ["attention"])
        key2 = f._cache_key("456", ["bandits"], ["attention"])
        assert key1 != key2

    def test_cache_set_and_check(self):
        from src.tools.llm_relevance import LLMRelevanceFilter, LLMRelevanceResult

        f = LLMRelevanceFilter(cache_ttl_hours=24)
        result = LLMRelevanceResult(
            arxiv_id="test",
            is_relevant=True,
            is_excluded=False,
            relevance_score=0.8,
        )
        f._set_cache("test_key", result)
        cached = f._check_cache("test_key")
        assert cached is not None
        assert cached.is_cached is True
        assert cached.relevance_score == 0.8

    def test_cache_miss(self):
        from src.tools.llm_relevance import LLMRelevanceFilter

        f = LLMRelevanceFilter()
        assert f._check_cache("nonexistent") is None

    def test_cache_expiry(self):
        from src.tools.llm_relevance import LLMRelevanceFilter, LLMRelevanceResult

        f = LLMRelevanceFilter(cache_ttl_hours=1)
        result = LLMRelevanceResult(arxiv_id="test")
        # Manually set an expired cache entry
        f._cache["test_key"] = (result, datetime.utcnow() - timedelta(hours=2))
        assert f._check_cache("test_key") is None

    def test_evaluate_calls_llm(self):
        """Test that evaluate calls _call_llm and returns result."""
        from src.tools.llm_relevance import LLMRelevanceFilter

        f = LLMRelevanceFilter()
        mock_response = {
            "is_relevant": True,
            "is_excluded": False,
            "relevance_score": 0.9,
            "excluded_topic_match": "",
            "reasoning": "Paper is about multi-armed bandits, a core interest.",
        }
        f._call_llm = Mock(return_value=(mock_response, 200))

        result = f.evaluate(
            paper={"arxiv_id": "2501.111", "title": "UCB for Bandits", "abstract": "We study UCB..."},
            research_topics=["Multi Armed Bandits"],
            avoid_topics=["Attention"],
        )

        assert result is not None
        assert result.is_relevant is True
        assert result.is_excluded is False
        assert result.relevance_score == 0.9
        assert result.tokens_used == 200

    def test_evaluate_excluded_paper(self):
        """Test paper that LLM judges as excluded."""
        from src.tools.llm_relevance import LLMRelevanceFilter

        f = LLMRelevanceFilter()
        mock_response = {
            "is_relevant": False,
            "is_excluded": True,
            "relevance_score": 0.1,
            "excluded_topic_match": "Attention",
            "reasoning": "Paper is about attention mechanisms, an excluded topic.",
        }
        f._call_llm = Mock(return_value=(mock_response, 180))

        result = f.evaluate(
            paper={"arxiv_id": "2501.222", "title": "AttentionRetriever", "abstract": "Attention layers..."},
            research_topics=["Multi Armed Bandits"],
            avoid_topics=["Attention", "Transformers"],
        )

        assert result is not None
        assert result.is_excluded is True
        assert result.excluded_topic_match == "Attention"

    def test_evaluate_uses_cache(self):
        """Test that cached results are returned without calling LLM."""
        from src.tools.llm_relevance import LLMRelevanceFilter

        f = LLMRelevanceFilter()
        mock_response = {
            "is_relevant": True,
            "is_excluded": False,
            "relevance_score": 0.7,
            "excluded_topic_match": "",
            "reasoning": "Relevant paper.",
        }
        f._call_llm = Mock(return_value=(mock_response, 100))

        # First call — should hit LLM
        result1 = f.evaluate(
            paper={"arxiv_id": "2501.333", "title": "Bandits Paper", "abstract": "..."},
            research_topics=["Bandits"],
            avoid_topics=["GAN"],
        )
        assert f._call_llm.call_count == 1

        # Second call — should hit cache
        result2 = f.evaluate(
            paper={"arxiv_id": "2501.333", "title": "Bandits Paper", "abstract": "..."},
            research_topics=["Bandits"],
            avoid_topics=["GAN"],
        )
        assert f._call_llm.call_count == 1  # NOT called again
        assert result2.is_cached is True

    def test_evaluate_returns_none_on_llm_failure(self):
        """Test that evaluate returns None when LLM call fails."""
        from src.tools.llm_relevance import LLMRelevanceFilter

        f = LLMRelevanceFilter()
        f._call_llm = Mock(return_value=({}, 0))

        result = f.evaluate(
            paper={"arxiv_id": "2501.444", "title": "Test", "abstract": "..."},
            research_topics=["Bandits"],
            avoid_topics=["GAN"],
        )
        assert result is None


# =========================================================================
# Integration Function
# =========================================================================

class TestEvaluatePaperRelevanceWithLLM:
    """Test the evaluate_paper_relevance_with_llm integration function."""

    @patch("src.tools.llm_relevance.is_feature_enabled", return_value=False)
    def test_returns_empty_when_disabled(self, mock_flag):
        from src.tools.llm_relevance import evaluate_paper_relevance_with_llm

        result = evaluate_paper_relevance_with_llm(
            paper={"arxiv_id": "test", "title": "Test", "abstract": "..."},
            research_topics=["Bandits"],
            avoid_topics=["GAN"],
        )
        assert result == {}

    @patch("src.tools.llm_relevance.is_feature_enabled", return_value=True)
    @patch("src.tools.llm_relevance.get_llm_relevance_filter")
    def test_returns_result_when_enabled(self, mock_get_filter, mock_flag):
        from src.tools.llm_relevance import evaluate_paper_relevance_with_llm, LLMRelevanceResult

        mock_filter = Mock()
        mock_filter.evaluate.return_value = LLMRelevanceResult(
            arxiv_id="test",
            is_relevant=True,
            is_excluded=False,
            relevance_score=0.85,
            reasoning="Directly relevant.",
            model_used="gpt-4o-mini",
            tokens_used=200,
        )
        mock_get_filter.return_value = mock_filter

        result = evaluate_paper_relevance_with_llm(
            paper={"arxiv_id": "test", "title": "Bandits paper", "abstract": "UCB..."},
            research_topics=["Bandits"],
            avoid_topics=["Attention"],
        )

        assert result["is_relevant"] is True
        assert result["is_excluded"] is False
        assert result["relevance_score"] == 0.85

    @patch("src.tools.llm_relevance.is_feature_enabled", return_value=True)
    @patch("src.tools.llm_relevance.get_llm_relevance_filter")
    def test_fallback_on_error(self, mock_get_filter, mock_flag):
        from src.tools.llm_relevance import evaluate_paper_relevance_with_llm

        mock_filter = Mock()
        mock_filter.evaluate.return_value = None  # LLM failed
        mock_get_filter.return_value = mock_filter

        result = evaluate_paper_relevance_with_llm(
            paper={"arxiv_id": "test", "title": "Test", "abstract": "..."},
            research_topics=["Bandits"],
            avoid_topics=["GAN"],
            fallback_on_error=True,
        )

        # Fallback allows paper through
        assert result["is_relevant"] is True
        assert result["is_excluded"] is False
        assert result["model_used"] == "fallback"

    @patch("src.tools.llm_relevance.is_feature_enabled", return_value=True)
    @patch("src.tools.llm_relevance.get_llm_relevance_filter")
    def test_no_fallback_returns_empty(self, mock_get_filter, mock_flag):
        from src.tools.llm_relevance import evaluate_paper_relevance_with_llm

        mock_filter = Mock()
        mock_filter.evaluate.return_value = None
        mock_get_filter.return_value = mock_filter

        result = evaluate_paper_relevance_with_llm(
            paper={"arxiv_id": "test", "title": "Test", "abstract": "..."},
            research_topics=["Bandits"],
            avoid_topics=["GAN"],
            fallback_on_error=False,
        )
        assert result == {}

    @patch("src.tools.llm_relevance.is_feature_enabled", return_value=True)
    @patch("src.tools.llm_relevance.get_llm_relevance_filter")
    def test_exception_with_fallback(self, mock_get_filter, mock_flag):
        from src.tools.llm_relevance import evaluate_paper_relevance_with_llm

        mock_get_filter.side_effect = RuntimeError("Connection refused")

        result = evaluate_paper_relevance_with_llm(
            paper={"arxiv_id": "test", "title": "Test", "abstract": "..."},
            research_topics=["Bandits"],
            avoid_topics=["GAN"],
            fallback_on_error=True,
        )

        assert result["is_relevant"] is True
        assert "fallback" in result["model_used"].lower()


# =========================================================================
# Feature Flag Config
# =========================================================================

class TestLLMRelevanceConfig:
    """Test LLMRelevanceConfig in feature_flags."""

    def test_config_defaults(self):
        from src.config.feature_flags import LLMRelevanceConfig

        config = LLMRelevanceConfig()
        assert config.enabled is False
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.1
        assert config.max_tokens == 300
        assert config.cache_ttl_hours == 48
        assert config.fallback_on_error is True

    def test_config_from_env(self):
        from src.config.feature_flags import LLMRelevanceConfig

        with patch.dict("os.environ", {
            "LLM_RELEVANCE_ENABLED": "true",
            "LLM_RELEVANCE_MODEL": "gpt-4o",
            "LLM_RELEVANCE_TEMPERATURE": "0.2",
            "LLM_RELEVANCE_MAX_TOKENS": "500",
            "LLM_RELEVANCE_CACHE_TTL_HOURS": "24",
        }):
            config = LLMRelevanceConfig.from_env()
            assert config.enabled is True
            assert config.model == "gpt-4o"
            assert config.temperature == 0.2
            assert config.max_tokens == 500
            assert config.cache_ttl_hours == 24

    def test_feature_flags_includes_llm_relevance(self):
        from src.config.feature_flags import FeatureFlags

        flags = FeatureFlags.load()
        assert hasattr(flags, "llm_relevance")
        d = flags.to_dict()
        assert "llm_relevance" in d
        assert "enabled" in d["llm_relevance"]


# =========================================================================
# Prompt Templates
# =========================================================================

class TestPromptTemplates:
    """Test that prompt templates are well-formed."""

    def test_system_prompt_exists(self):
        from src.tools.llm_relevance import RELEVANCE_SYSTEM_PROMPT
        assert "is_relevant" in RELEVANCE_SYSTEM_PROMPT
        assert "is_excluded" in RELEVANCE_SYSTEM_PROMPT
        assert "relevance_score" in RELEVANCE_SYSTEM_PROMPT

    def test_user_prompt_has_placeholders(self):
        from src.tools.llm_relevance import RELEVANCE_USER_PROMPT
        assert "{interests}" in RELEVANCE_USER_PROMPT
        assert "{exclude_topics}" in RELEVANCE_USER_PROMPT
        assert "{title}" in RELEVANCE_USER_PROMPT
        assert "{abstract}" in RELEVANCE_USER_PROMPT

    def test_user_prompt_formats_correctly(self):
        from src.tools.llm_relevance import RELEVANCE_USER_PROMPT
        formatted = RELEVANCE_USER_PROMPT.format(
            interests="Multi Armed Bandits, PCA",
            exclude_topics="Attention, Transformers",
            title="UCB Exploration",
            abstract="We study...",
        )
        assert "Multi Armed Bandits" in formatted
        assert "Attention, Transformers" in formatted
        assert "UCB Exploration" in formatted
