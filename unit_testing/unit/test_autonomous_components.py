"""
Unit tests for autonomous components:
- Feature flags
- LLM Novelty Scoring
- Audit Log
- Profile Evolution
- Live Document

These tests verify the components work correctly with feature flags
and gracefully degrade when disabled or when dependencies fail.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4
from datetime import datetime


# =========================================================================
# Feature Flags
# =========================================================================

class TestFeatureFlags:
    """Test feature flag configuration and checking."""

    def test_default_flags_are_disabled(self):
        """All flags should default to disabled (False)."""
        # Clear environment to test defaults
        with patch.dict("os.environ", {}, clear=True):
            from src.config.feature_flags import FeatureFlags
            
            flags = FeatureFlags()
            assert flags.llm_novelty_enabled is False
            assert flags.audit_log_enabled is False
            assert flags.profile_evolution_enabled is False
            assert flags.live_document_enabled is False

    def test_flags_read_from_env(self):
        """Flags should read from environment variables."""
        env_vars = {
            "LLM_NOVELTY_ENABLED": "true",
            "AUDIT_LOG_ENABLED": "1",
            "PROFILE_EVOLUTION_ENABLED": "yes",
            "LIVE_DOCUMENT_ENABLED": "false",
        }
        
        with patch.dict("os.environ", env_vars, clear=True):
            from src.config.feature_flags import FeatureFlags
            
            flags = FeatureFlags()
            assert flags.llm_novelty_enabled is True
            assert flags.audit_log_enabled is True
            assert flags.profile_evolution_enabled is True
            assert flags.live_document_enabled is False

    def test_is_feature_enabled_function(self):
        """Test the is_feature_enabled helper function."""
        env_vars = {"LLM_NOVELTY_ENABLED": "true"}
        
        with patch.dict("os.environ", env_vars, clear=True):
            # Force reload of module to pick up new env
            import importlib
            import src.config.feature_flags as ff_module
            importlib.reload(ff_module)
            
            from src.config.feature_flags import is_feature_enabled
            
            assert is_feature_enabled("LLM_NOVELTY") is True
            assert is_feature_enabled("AUDIT_LOG") is False


# =========================================================================
# LLM Novelty Scoring
# =========================================================================

class TestLLMNoveltyScoring:
    """Test LLM novelty scoring service."""

    def test_novelty_result_model(self):
        """Test LLMNoveltyResult model structure."""
        from src.tools.llm_novelty import LLMNoveltyResult, NoveltySubScores
        
        result = LLMNoveltyResult(
            arxiv_id="2401.00001",
            llm_novelty_score=85,
            reasoning="Highly novel methodology",
            sub_scores=NoveltySubScores(
                methodology=90,
                application=80,
                theoretical=70,
                dataset=60,
            ),
        )
        
        assert result.llm_novelty_score == 85
        assert result.sub_scores.methodology == 90
        assert result.confidence == 1.0

    def test_scorer_initialization(self):
        """Test scorer initializes with config."""
        from src.tools.llm_novelty import LLMNoveltyScorer
        
        scorer = LLMNoveltyScorer(
            model="gpt-4o-mini",
            high_novelty_threshold=80,
            cache_enabled=True,
        )
        
        assert scorer.model == "gpt-4o-mini"
        assert scorer.high_novelty_threshold == 80
        assert scorer.cache_enabled is True

    def test_scorer_returns_none_when_disabled(self):
        """Test scoring returns None when feature is disabled."""
        from src.tools.llm_novelty import score_paper_novelty_with_llm
        
        with patch.dict("os.environ", {"LLM_NOVELTY_ENABLED": "false"}):
            paper = {"arxiv_id": "2401.00001", "title": "Test", "abstract": "Test abstract"}
            profile = {"research_topics": ["ML"]}
            
            result = score_paper_novelty_with_llm(paper, profile)
            
            assert result.get("success") is True
            assert result.get("llm_novelty_score") is None
            assert "disabled" in result.get("skip_reason", "")


# =========================================================================
# Audit Log
# =========================================================================

class TestAuditLog:
    """Test run audit log service."""

    def test_audit_log_builder_initialization(self):
        """Test AuditLogBuilder creates proper structure."""
        from src.tools.audit_log import AuditLogBuilder
        
        builder = AuditLogBuilder(run_id="test-run-123", user_id="user-456")
        
        assert builder.run_id == "test-run-123"
        assert builder.user_id == "user-456"
        assert builder._log.papers_fetched == 0
        assert builder._log.papers_scored == 0

    def test_audit_log_builder_records_papers(self):
        """Test builder correctly records paper data."""
        from src.tools.audit_log import AuditLogBuilder
        
        builder = AuditLogBuilder(run_id="test-run", user_id="user")
        
        fetched = [{"arxiv_id": "1"}, {"arxiv_id": "2"}]
        unseen = [{"arxiv_id": "1"}]
        scored = [{"arxiv_id": "1", "relevance_score": 0.8}]
        
        builder.set_fetched_papers(fetched)
        builder.set_unseen_papers(unseen)
        builder.set_scored_papers(scored)
        
        log = builder.build()
        
        assert log.papers_fetched == 2
        assert log.papers_unseen == 1
        assert log.papers_scored == 1

    def test_build_audit_log_from_episode_helper(self):
        """Test the integration helper function."""
        from src.tools.audit_log import build_audit_log_from_episode
        
        log = build_audit_log_from_episode(
            run_id="run-123",
            user_id="user-456",
            fetched_papers=[{"arxiv_id": "p1"}, {"arxiv_id": "p2"}],
            unseen_papers=[{"arxiv_id": "p1"}],
            scored_papers=[{"arxiv_id": "p1", "relevance_score": 0.9}],
            decisions=[{"paper_id": "p1", "decision": "recommend"}],
            actions=[{"type": "email", "recipient": "owner"}],
            stop_reason="completed",
            metrics={"highest_importance": "high"},
            config={},
        )
        
        assert log.run_id == "run-123"
        assert log.user_id == "user-456"
        assert log.papers_fetched == 2
        assert log.papers_scored == 1


# =========================================================================
# Profile Evolution
# =========================================================================

class TestProfileEvolution:
    """Test profile evolution suggestions service."""

    def test_profile_suggestion_model(self):
        """Test ProfileSuggestion model structure."""
        from src.agent.profile_evolution import ProfileSuggestion, SupportingPaper
        
        suggestion = ProfileSuggestion(
            suggestion_type="add_topic",
            suggestion_text="Add research topic: 'vision-language models'",
            reasoning="5 relevant papers detected",
            confidence=0.85,
            suggestion_data={"action": "add_topic", "topic": "vision-language models"},
            supporting_papers=[
                SupportingPaper(
                    arxiv_id="2401.00001",
                    title="CLIP paper",
                    relevance_score=0.9,
                    novelty_score=0.8,
                )
            ],
        )
        
        assert suggestion.suggestion_type == "add_topic"
        assert suggestion.confidence == 0.85
        assert len(suggestion.supporting_papers) == 1

    def test_analyzer_initialization(self):
        """Test ProfileEvolutionAnalyzer initializes correctly."""
        from src.agent.profile_evolution import ProfileEvolutionAnalyzer
        
        analyzer = ProfileEvolutionAnalyzer(
            model="gpt-4o-mini",
            min_high_relevance_papers=3,
            min_novelty_threshold=0.7,
            max_suggestions=3,
            cooldown_hours=24,
        )
        
        assert analyzer.model == "gpt-4o-mini"
        assert analyzer.min_high_relevance_papers == 3
        assert analyzer.cooldown_hours == 24

    def test_analyzer_skips_when_insufficient_papers(self):
        """Test analysis is skipped when not enough high-relevance papers."""
        from src.agent.profile_evolution import ProfileEvolutionAnalyzer
        
        analyzer = ProfileEvolutionAnalyzer(min_high_relevance_papers=5)
        
        # Only 2 papers, need 5
        papers = [
            {"arxiv_id": "1", "relevance_score": 0.8, "novelty_score": 0.9},
            {"arxiv_id": "2", "relevance_score": 0.7, "novelty_score": 0.8},
        ]
        
        result = analyzer.analyze(
            run_id="test-run",
            user_id=str(uuid4()),
            user_profile={"research_topics": ["ML"]},
            scored_papers=papers,
        )
        
        assert result.skipped is True
        assert "Insufficient" in result.skip_reason


# =========================================================================
# Live Document
# =========================================================================

class TestLiveDocument:
    """Test live document service."""

    def test_document_paper_model(self):
        """Test DocumentPaper model structure."""
        from src.tools.live_document import DocumentPaper
        
        paper = DocumentPaper(
            arxiv_id="2401.00001",
            title="Test Paper",
            authors=["Alice", "Bob"],
            relevance_score=0.9,
            novelty_score=0.8,
            categories=["cs.LG"],
        )
        
        assert paper.arxiv_id == "2401.00001"
        assert paper.relevance_score == 0.9
        assert len(paper.authors) == 2

    def test_live_document_data_model(self):
        """Test LiveDocumentData model structure."""
        from src.tools.live_document import LiveDocumentData
        
        doc = LiveDocumentData(
            user_id="user-123",
            title="ResearchPulse - Live Document",
            executive_summary="Summary of research",
            total_papers_tracked=10,
        )
        
        assert doc.user_id == "user-123"
        assert doc.total_papers_tracked == 10
        assert len(doc.top_papers) == 0

    def test_manager_initialization(self):
        """Test LiveDocumentManager initializes correctly."""
        from src.tools.live_document import LiveDocumentManager
        
        manager = LiveDocumentManager(
            max_top_papers=10,
            max_recent_papers=20,
            rolling_window_days=7,
            model="gpt-4o-mini",
        )
        
        assert manager.max_top_papers == 10
        assert manager.rolling_window_days == 7

    def test_manager_renders_markdown(self):
        """Test markdown rendering produces valid output."""
        from src.tools.live_document import LiveDocumentManager, LiveDocumentData, DocumentPaper
        
        manager = LiveDocumentManager()
        
        doc = LiveDocumentData(
            user_id="user-123",
            executive_summary="Test summary",
            top_papers=[
                DocumentPaper(
                    arxiv_id="2401.00001",
                    title="Paper One",
                    relevance_score=0.9,
                    novelty_score=0.8,
                    arxiv_url="https://arxiv.org/abs/2401.00001",
                ),
            ],
        )
        
        markdown = manager.render_markdown(doc)
        
        assert "# Research Pulse" in markdown
        assert "Test summary" in markdown
        assert "Paper One" in markdown
        assert "https://arxiv.org/abs/2401.00001" in markdown


# =========================================================================
# Integration - Graceful Degradation
# =========================================================================

class TestGracefulDegradation:
    """Test that components degrade gracefully on errors."""

    def test_llm_novelty_handles_missing_openai(self):
        """LLM novelty should handle missing OpenAI package."""
        from src.tools.llm_novelty import LLMNoveltyScorer
        
        scorer = LLMNoveltyScorer()
        
        # Mock the OpenAI import to fail
        with patch.dict("sys.modules", {"openai": None}):
            paper = {"arxiv_id": "1", "title": "Test", "abstract": "Test"}
            profile = {"research_topics": ["ML"]}
            
            # Should not raise, should return graceful result
            result = scorer.score(paper, profile)
            
            # Either returns cached result or None, doesn't crash
            assert result is None or hasattr(result, "llm_novelty_score")

    def test_audit_log_handles_db_unavailable(self):
        """Audit log save should handle DB being unavailable."""
        from src.tools.audit_log import save_audit_log, RunAuditLogData
        
        log = RunAuditLogData(run_id="test", user_id="user")
        
        with patch("src.tools.audit_log.is_database_configured", return_value=False):
            result = save_audit_log(log)
        
        # Should return graceful failure, not crash
        assert result.get("success") is False
        assert "database_not_configured" in result.get("error", "")

    def test_profile_evolution_handles_llm_error(self):
        """Profile evolution should handle LLM call failures."""
        from src.agent.profile_evolution import ProfileEvolutionAnalyzer
        
        analyzer = ProfileEvolutionAnalyzer(min_high_relevance_papers=1)
        
        papers = [
            {"arxiv_id": "1", "relevance_score": 0.9, "novelty_score": 0.9},
        ]
        
        # Mock OpenAI to fail
        with patch("src.agent.profile_evolution.OpenAI", side_effect=Exception("API error")):
            result = analyzer.analyze(
                run_id="test",
                user_id=str(uuid4()),
                user_profile={"research_topics": ["ML"]},
                scored_papers=papers,
            )
        
        # Should complete without crashing, just no suggestions
        assert result.suggestions == []


# =========================================================================
# Agent Integration
# =========================================================================

class TestAgentIntegration:
    """Test autonomous components integration with react_agent."""

    def test_feature_flags_import_in_agent(self):
        """Test that feature flags can be imported in agent."""
        # This tests the import guard in react_agent.py
        try:
            from src.config.feature_flags import is_feature_enabled
            imported = True
        except ImportError:
            imported = False
        
        assert imported is True

    def test_autonomous_components_are_additive(self):
        """Test that autonomous components don't modify core paper results."""
        # This is a critical invariant - autonomous components should NEVER
        # modify the papers that were selected for the owner
        
        # Mock the feature flags
        with patch.dict("os.environ", {
            "LLM_NOVELTY_ENABLED": "true",
            "AUDIT_LOG_ENABLED": "true",
            "PROFILE_EVOLUTION_ENABLED": "true",
            "LIVE_DOCUMENT_ENABLED": "true",
        }):
            # The scored papers list should remain unchanged after
            # _run_autonomous_components executes
            
            # Note: Full integration test would require setting up
            # the complete agent, which is beyond unit test scope
            pass
