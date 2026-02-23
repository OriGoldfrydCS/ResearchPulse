"""
Tests for the 7-requirement fix batch.

Req 1: Colleague interest filtering — no blanket high-importance sharing
Req 2: Re-run fetches next unseen papers
Req 3: Published dates forwarded correctly
Req 4: Live Doc uses only scored papers + TXT export
Req 5: Human-friendly arXiv category names
Req 6: Dedupe profile suggestions
Req 7: Strict paper-by-ID retrieval via prompt parser
"""

import os
import sys
import uuid
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


# =========================================================================
# Requirement 1 — Colleague interest filtering
# =========================================================================

class TestColleagueFiltering:
    """Verify that high-importance papers are NOT shared with colleagues
    unless there is a genuine topic or category overlap."""

    def test_no_blanket_high_importance_sharing(self):
        """High importance alone must NOT trigger sharing."""
        from src.tools.decide_delivery import process_colleague_surplus

        paper = {
            "arxiv_id": "1234.56789",
            "title": "Quantum Gravity Simulations",
            "abstract": "A study of quantum gravity in loop models.",
            "relevance_score": 0.95,
            "novelty_score": 0.90,
            "importance": "high",
            "categories": ["gr-qc"],
        }
        colleague = {
            "id": str(uuid.uuid4()),
            "name": "NLP Expert",
            "email": "nlp@example.com",
            "topics": ["natural language processing"],
            "categories": ["cs.CL"],
        }

        result = process_colleague_surplus(
            all_scored_papers=[paper],
            owner_paper_ids=[],
            colleagues=[colleague],
            delivery_policy={"colleague_sharing_settings": {"enabled": True}},
        )
        # The colleague should get zero shares — no topic/category match
        actions = result.get("colleague_actions", [])
        shared = [a for a in actions if a.get("action_type", "").startswith("share")]
        assert len(shared) == 0, "Colleague received papers despite no interest overlap"

    def test_topic_match_shares_paper(self):
        """Paper IS shared when there is a genuine topic overlap."""
        from src.tools.decide_delivery import process_colleague_surplus

        paper = {
            "arxiv_id": "1234.56789",
            "title": "Transformer-Based NLP Models",
            "abstract": "We present novel transformer architectures for natural language processing.",
            "relevance_score": 0.90,
            "novelty_score": 0.80,
            "importance": "high",
            "categories": ["cs.CL"],
        }
        colleague = {
            "id": str(uuid.uuid4()),
            "name": "NLP Expert",
            "email": "nlp@example.com",
            "topics": ["natural language processing", "transformers"],
            "categories": ["cs.CL"],
        }

        result = process_colleague_surplus(
            all_scored_papers=[paper],
            owner_paper_ids=[],
            colleagues=[colleague],
            delivery_policy={"colleague_sharing_settings": {"enabled": True}},
        )
        actions = result.get("colleague_actions", [])
        shared = [a for a in actions if a.get("action_type", "").startswith("share")]
        assert len(shared) >= 1, "Colleague should receive papers with matching interests"


# =========================================================================
# Requirement 2 — Re-run fetches next unseen papers
# =========================================================================

class TestRerunFetchesNewPapers:
    """Ensure re-runs use delivered (not merely seen) IDs."""

    def test_get_delivered_paper_ids_function_exists(self):
        """get_delivered_paper_ids should be importable."""
        from src.db.data_service import get_delivered_paper_ids
        assert callable(get_delivered_paper_ids)

    def test_check_seen_uses_delivered_ids(self):
        """check_seen_papers should filter by delivered IDs."""
        from src.tools.check_seen import check_seen_papers_json
        # With an empty delivered set every paper is unseen
        with patch("src.tools.check_seen.get_delivered_paper_ids", return_value=set()):
            result = check_seen_papers_json(
                papers=[
                    {"arxiv_id": "0001.00001", "title": "P1"},
                    {"arxiv_id": "0001.00002", "title": "P2"},
                ]
            )
            unseen = result.get("unseen_papers", [])
            assert len(unseen) == 2

    def test_check_seen_filters_delivered_papers(self):
        """Already-delivered papers must be excluded."""
        from src.tools.check_seen import check_seen_papers_json
        with patch("src.tools.check_seen.get_delivered_paper_ids", return_value={"0001.00001"}):
            result = check_seen_papers_json(
                papers=[
                    {"arxiv_id": "0001.00001", "title": "P1"},
                    {"arxiv_id": "0001.00002", "title": "P2"},
                ]
            )
            unseen = result.get("unseen_papers", [])
            assert len(unseen) == 1
            assert unseen[0]["arxiv_id"] == "0001.00002"

    def test_fetch_arxiv_start_index_parameter(self):
        """fetch_arxiv_papers should accept start_index."""
        from src.tools.fetch_arxiv import fetch_arxiv_papers
        result = fetch_arxiv_papers(
            categories_include=["cs.AI"],
            max_results=2,
            start_index=0,
            use_mock=True,
        )
        assert result.success
        count_0 = len(result.papers)
        assert count_0 > 0

    def test_fetch_single_paper_exists(self):
        """fetch_single_paper function should be importable and callable."""
        from src.tools.fetch_arxiv import fetch_single_paper
        assert callable(fetch_single_paper)


# =========================================================================
# Requirement 3 — Published dates
# =========================================================================

class TestPublishedDates:
    """Verify the published field flows through the persist pipeline."""

    def test_mock_papers_have_published_field(self):
        """Mock papers should include 'published' key."""
        from src.tools.fetch_arxiv import MOCK_PAPERS
        for p in MOCK_PAPERS:
            assert "published" in p, f"Mock paper {p.get('arxiv_id')} missing 'published'"

    def test_persist_state_forwards_published(self):
        """persist_state meta_key forwarding should include 'published' and 'updated'."""
        import src.tools.persist_state as ps
        import inspect
        source = inspect.getsource(ps.persist_paper_decision)
        assert '"published"' in source
        assert '"updated"' in source

    def test_upsert_paper_parses_published(self):
        """upsert_paper should try 'published', 'published_at', 'publication_date', 'updated'."""
        import src.db.data_service as ds
        import inspect
        source = inspect.getsource(ds.upsert_paper)
        assert "published" in source
        assert "updated" in source


# =========================================================================
# Requirement 4 — Live Doc correctness + TXT export
# =========================================================================

class TestLiveDocument:
    """Live document should use only scored papers and support TXT export."""

    def test_render_text_method_exists(self):
        """LiveDocumentManager must have render_text method."""
        from src.tools.live_document import LiveDocumentManager
        mgr = LiveDocumentManager()
        assert hasattr(mgr, "render_text")

    def test_render_text_output(self):
        """render_text should produce non-empty plain text."""
        from src.tools.live_document import LiveDocumentManager, LiveDocumentData
        mgr = LiveDocumentManager()
        doc = LiveDocumentData(
            user_id=str(uuid.uuid4()),
            title="Test Briefing",
            last_updated=datetime.utcnow().isoformat(),
            executive_summary="Summary of recent papers.",
            top_papers=[],
            trending_topics=[],
            category_breakdown={},
            recent_papers=[],
            total_papers_tracked=0,
            runs_included=[],
        )
        text = mgr.render_text(doc)
        assert "TEST BRIEFING" in text
        assert "EXECUTIVE SUMMARY" in text
        assert "Summary of recent papers." in text

    def test_live_doc_uses_scored_papers_not_all(self):
        """react_agent should pass _scored_papers, not analysis_papers, to live doc."""
        import inspect
        from src.agent.react_agent import ResearchReActAgent
        source = inspect.getsource(ResearchReActAgent._run_autonomous_components)
        # The live document block should reference self._scored_papers
        assert "self._scored_papers" in source
        # It should NOT pass analysis_papers for the live document
        assert "scored_papers=analysis_papers" not in source or \
               source.index("scored_papers=self._scored_papers") < source.index("scored_papers=analysis_papers") \
               if "scored_papers=analysis_papers" in source else True


# =========================================================================
# Requirement 5 — Human-friendly arXiv category names
# =========================================================================

class TestCategoryDisplayNames:
    """Category codes should be enriched with human-readable names."""

    def test_get_category_display_name(self):
        """get_category_display_name should return 'code (Name)' format."""
        from src.tools.arxiv_categories import get_category_display_name
        result = get_category_display_name("cs.AI")
        assert "cs.AI" in result
        assert "(" in result  # Should include the name in parentheses
        assert "Artificial Intelligence" in result

    def test_unknown_category_returns_raw_code(self):
        """Unknown category should fall back to the raw code."""
        from src.tools.arxiv_categories import get_category_display_name
        result = get_category_display_name("xx.UNKNOWN")
        assert result == "xx.UNKNOWN"

    def test_paper_to_dict_includes_categories_display(self):
        """paper_to_dict should include categories_display field."""
        from src.db.orm_models import paper_to_dict, Paper
        paper = Paper(
            id=uuid.uuid4(),
            source="arxiv",
            external_id="2401.00001",
            title="Test",
            categories=["cs.AI", "cs.LG"],
        )
        d = paper_to_dict(paper)
        assert "categories_display" in d
        assert len(d["categories_display"]) == 2


# =========================================================================
# Requirement 6 — Dedupe profile suggestions
# =========================================================================

class TestDedupeProfileSuggestions:
    """Duplicate pending suggestions must be skipped."""

    def test_save_profile_suggestions_dedup(self):
        """Saving the same suggestion twice should skip the duplicate."""
        from src.agent.profile_evolution import (
            save_profile_suggestions,
            ProfileEvolutionAnalysis,
            ProfileSuggestion,
        )

        suggestion = ProfileSuggestion(
            suggestion_type="add_topic",
            suggestion_text="Add research topic: 'graph neural networks'",
            reasoning="Appeared in recent papers",
            confidence=0.7,
            suggestion_data={},
            supporting_papers=[],
        )
        analysis = ProfileEvolutionAnalysis(
            run_id="test-run-1",
            user_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            suggestions=[suggestion, suggestion],  # intentional duplicate
            skip_reason=None,
        )

        # We mock the DB to simulate dedup logic.
        # save_profile_suggestions does a local import:
        #   from db.database import is_database_configured, get_db_session
        # so we patch at the db.database module level.
        mock_session = MagicMock()
        mock_query = MagicMock()
        # First call: no existing → insert; Second call: existing → skip
        mock_query.filter_by.return_value.first.side_effect = [None, MagicMock()]
        mock_session.query.return_value = mock_query

        with patch("db.database.is_database_configured", return_value=True), \
             patch("db.database.get_db_session") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = save_profile_suggestions(analysis)

        assert result["success"] is True
        assert result["saved"] == 1
        assert result["skipped_duplicates"] == 1


# =========================================================================
# Requirement 7 — Strict paper-by-ID retrieval
# =========================================================================

class TestFetchByIdTemplate:
    """The prompt parser should detect arXiv IDs and route correctly."""

    def test_parser_detects_arxiv_id(self):
        """Prompt with arXiv ID should parse to FETCH_BY_ID template."""
        from src.agent.prompt_controller import PromptParser, PromptTemplate
        parser = PromptParser()
        result = parser.parse("Fetch paper 2301.12345")
        assert result.arxiv_id == "2301.12345"
        assert result.template == PromptTemplate.FETCH_BY_ID

    def test_parser_detects_arxiv_id_versionned(self):
        """Versioned ID like 2301.12345v2 should also parse."""
        from src.agent.prompt_controller import PromptParser, PromptTemplate
        parser = PromptParser()
        result = parser.parse("Get me 2301.12345v2")
        assert result.arxiv_id == "2301.12345v2"
        assert result.template == PromptTemplate.FETCH_BY_ID

    def test_parser_no_arxiv_id_for_generic_query(self):
        """Generic queries should NOT set arxiv_id."""
        from src.agent.prompt_controller import PromptParser, PromptTemplate
        parser = PromptParser()
        result = parser.parse("Find recent papers on transformers")
        assert result.arxiv_id is None
        assert result.template != PromptTemplate.FETCH_BY_ID

    def test_fetch_single_paper_mock(self):
        """fetch_single_paper should return exactly one paper from mock data."""
        from src.tools.fetch_arxiv import fetch_single_paper
        result = fetch_single_paper("2501.01001")
        assert result.success
        assert len(result.papers) == 1
        assert result.papers[0].arxiv_id == "2501.01001"

    def test_fetch_single_paper_not_found(self):
        """fetch_single_paper with unknown ID should return empty."""
        from src.tools.fetch_arxiv import fetch_single_paper
        result = fetch_single_paper("9999.99999")
        # Should either fail or return empty papers
        if result.success:
            assert len(result.papers) == 0
        else:
            assert result.error is not None
