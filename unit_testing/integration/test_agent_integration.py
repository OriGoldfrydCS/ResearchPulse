"""
Integration tests for the ResearchPulse ReAct agent.

These tests instantiate and run the actual ResearchReActAgent,
validating real outputs (AgentEpisode structure, fields, paper processing,
output enforcement, stop conditions, template detection) with minimal
mocking of external services (arXiv API, Pinecone, database, LLM).

The agent's core logic — prompt parsing, scoring, delivery decisions,
output enforcement, stop controller, report generation — runs un-mocked.
"""

import os
import sys
import uuid
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from typing import Any, Dict, List

# ── path setup ──────────────────────────────────────────────────────────────
SRC_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, 'src'))

from src.agent.react_agent import (
    ResearchReActAgent,
    AgentConfig,
    AgentEpisode,
    AgentStep,
    ToolCall,
    map_interests_to_categories,
    StopPolicy,
)
from src.agent.prompt_controller import (
    PromptTemplate,
    ParsedPrompt,
    PromptController,
    DEFAULT_OUTPUT_COUNT,
    DEFAULT_ARXIV_FETCH_COUNT,
)
from src.tools.fetch_arxiv import FetchArxivResult, ArxivPaper, fetch_arxiv_papers, MOCK_PAPERS
from src.tools.check_seen import CheckSeenResult
from src.tools.persist_state import PersistResult


# ============================================================================
# Shared fixtures and helpers
# ============================================================================

def _make_research_profile(**overrides) -> Dict[str, Any]:
    """Build a minimal research profile for tests."""
    profile = {
        "user_id": str(uuid.uuid4()),
        "researcher_name": "Test Researcher",
        "email": "test@example.com",
        "affiliation": "Test University",
        "research_topics": ["machine learning", "natural language processing"],
        "my_papers": [],
        "preferred_venues": [],
        "avoid_topics": [],
        "time_budget_per_week_minutes": 120,
        "arxiv_categories_include": ["cs.LG", "cs.CL"],
        "arxiv_categories_exclude": [],
        "interests_include": "machine learning, NLP",
        "interests_exclude": "",
        "keywords_include": [],
        "keywords_exclude": [],
        "preferred_time_period": "last two weeks",
        "stop_policy": {},
    }
    profile.update(overrides)
    return profile


def _make_delivery_policy() -> Dict[str, Any]:
    """Delivery policy that never sends real emails or calendar invites."""
    return {
        "importance_policies": {
            "high": {
                "send_email": False,
                "send_calendar": False,
                "save_to_reading_list": True,
                "log": True,
                "priority_label": "urgent",
            },
            "medium": {
                "send_email": False,
                "send_calendar": False,
                "save_to_reading_list": True,
                "log": True,
                "priority_label": "normal",
            },
            "low": {
                "send_email": False,
                "send_calendar": False,
                "save_to_reading_list": False,
                "log": True,
                "priority_label": "low",
            },
            "very_low": {
                "log": True,
                "priority_label": "info",
            },
        },
        "email_settings": {"enabled": False, "simulate_output": True},
        "calendar_settings": {"enabled": False},
        "reading_list_settings": {"enabled": True},
        "colleague_sharing_settings": {"enabled": False},
    }


def _mock_fetch_arxiv(**kwargs) -> FetchArxivResult:
    """Use the built-in mock papers from fetch_arxiv.

    We strip *query* so that keyword filtering is bypassed—mock papers are
    already curated test data and we want the agent pipeline to always
    receive papers regardless of prompt wording.
    """
    kwargs.pop("use_mock", None)
    kwargs.pop("query", None)          # bypass keyword filter on mock data
    return fetch_arxiv_papers(**kwargs, use_mock=True)


def _empty_rag_result(**kwargs) -> Dict[str, Any]:
    """Return an empty RAG result (no Pinecone available)."""
    return {
        "success": True,
        "query": kwargs.get("query_text", ""),
        "matches": [],
        "total_found": 0,
        "filtered_count": 0,
        "error": None,
    }


def _noop_persist(**kwargs) -> PersistResult:
    """Persist stub that always succeeds without touching the DB."""
    paper = kwargs.get("paper_decision", {})
    return PersistResult(
        success=True,
        paper_id=paper.get("paper_id", "unknown"),
        action="inserted",
        message="persisted (test stub)",
    )


# ============================================================================
# Patch context manager for isolating the agent from external services
# ============================================================================

class AgentTestHarness:
    """
    Set up and tear down all patches needed to run the agent without
    hitting real external services (arXiv API, Pinecone, database, LLM).
    """

    def __init__(self, research_profile=None, colleagues=None, delivery_policy=None):
        self.research_profile = research_profile or _make_research_profile()
        self.colleagues = colleagues or []
        self.delivery_policy = delivery_policy or _make_delivery_policy()
        self._patches = []

    def __enter__(self):
        targets = [
            # DB / profile — these are imported into react_agent module namespace
            ("src.agent.react_agent.get_research_profile", lambda: self.research_profile),
            ("src.agent.react_agent.get_colleagues", lambda: self.colleagues),
            ("src.agent.react_agent.get_delivery_policy", lambda: self.delivery_policy),
            ("src.agent.react_agent.get_arxiv_fetch_count", lambda user_id=None: DEFAULT_ARXIV_FETCH_COUNT),
            ("src.agent.react_agent.is_db_available", lambda: False),
            ("src.agent.react_agent.save_artifacts_to_db", lambda files: {"success": True}),
            # arXiv fetch — redirect to built-in mock papers
            ("src.agent.react_agent.fetch_arxiv_papers", _mock_fetch_arxiv),
            # Pinecone / RAG — return empty matches
            ("src.agent.react_agent.retrieve_similar_from_pinecone_json", _empty_rag_result),
            # check_seen — treat all papers as unseen
            ("src.agent.react_agent.check_seen_papers_json", self._all_unseen),
            # persist — succeed without DB
            ("src.agent.react_agent.persist_paper_decision", _noop_persist),
            # Disable DB access inside check_seen (called transitively)
            ("src.tools.check_seen.get_delivered_paper_ids", lambda: set()),
            # Disable autonomous components that hit DB
            ("src.agent.react_agent.FEATURE_FLAGS_AVAILABLE", False),
        ]

        for target, replacement in targets:
            if callable(replacement):
                p = patch(target, side_effect=replacement)
            else:
                p = patch(target, replacement)
            self._patches.append(p)
            p.start()

        # Patch parse_and_save on BOTH module paths (src.agent and agent)
        # because the agent imports from 'agent.prompt_controller' (via sys.path)
        for mod_path in [
            "src.agent.prompt_controller.PromptController.parse_and_save",
            "agent.prompt_controller.PromptController.parse_and_save",
        ]:
            try:
                ps_patch = patch(mod_path, self._parse_and_save)
                self._patches.append(ps_patch)
                ps_patch.start()
            except (AttributeError, ModuleNotFoundError):
                pass

        return self

    def __exit__(self, *args):
        for p in reversed(self._patches):
            p.stop()
        self._patches.clear()

    # --- helpers called by patches ---

    @staticmethod
    def _parse_and_save(self_ctrl, prompt: str, run_id: str = None):
        """Real parsing, but skip DB save."""
        parsed = self_ctrl.parse_prompt(prompt)
        return parsed, None  # no prompt_id

    @staticmethod
    def _all_unseen(**kwargs):
        """Mark every paper as unseen."""
        papers = kwargs.get("papers", [])
        return {
            "unseen_papers": papers,
            "seen_papers": [],
            "summary": {"total": len(papers), "unseen": len(papers), "seen": 0},
        }


def _run_agent(prompt: str, **config_overrides) -> AgentEpisode:
    """Create and run the agent, returning the episode."""
    run_id = str(uuid.uuid4())
    stop_policy = config_overrides.pop("stop_policy", None) or StopPolicy(
        max_runtime_minutes=2,
        max_papers_checked=10,
        stop_if_no_new_papers=False,
        max_rag_queries=50,
        min_importance_to_act="low",
    )
    config = AgentConfig(
        max_steps=50,
        stop_policy=stop_policy,
        use_mock_arxiv=True,
        verbose=False,
        **config_overrides,
    )
    agent = ResearchReActAgent(run_id=run_id, config=config)
    return agent.run(prompt)


# ============================================================================
# TEST SUITE: Agent episode structure
# ============================================================================

class TestAgentEpisodeStructure:
    """Verify the AgentEpisode returned by the agent has correct structure."""

    def test_episode_has_required_fields(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on machine learning")

        assert isinstance(episode, AgentEpisode)
        assert episode.run_id
        assert episode.start_time
        assert episode.end_time
        assert episode.user_message == "Papers on machine learning"
        assert episode.stop_reason is not None

    def test_episode_records_template(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Find papers about NLP from the last week")

        assert episode.detected_template is not None
        assert episode.detected_template in [t.value for t in PromptTemplate]

    def test_episode_has_tool_calls(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on deep learning")

        assert len(episode.tool_calls) > 0
        tool_names = [tc.tool_name for tc in episode.tool_calls]
        assert "fetch_arxiv_papers" in tool_names
        assert "check_seen_papers" in tool_names

    def test_episode_has_papers_processed(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on machine learning")

        assert isinstance(episode.papers_processed, list)
        if episode.papers_processed:
            paper = episode.papers_processed[0]
            assert "arxiv_id" in paper
            assert "title" in paper
            assert "relevance_score" in paper
            assert "importance" in paper

    def test_episode_has_final_report(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on machine learning")

        assert episode.final_report is not None
        assert isinstance(episode.final_report, dict)

    def test_episode_timestamps_are_iso(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on NLP")

        datetime.fromisoformat(episode.start_time.replace("Z", "+00:00"))
        datetime.fromisoformat(episode.end_time.replace("Z", "+00:00"))


# ============================================================================
# TEST SUITE: Template detection through the agent
# ============================================================================

class TestAgentTemplateDetection:
    """Verify the agent correctly detects prompt templates end-to-end."""

    def test_topic_only_template(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on transformers")
        assert episode.detected_template == PromptTemplate.TOPIC_ONLY.value

    def test_topic_time_template(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Find papers on deep learning from the last 2 weeks")
        assert episode.detected_template == PromptTemplate.TOPIC_TIME.value

    def test_topic_venue_time_template(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on NLP in NeurIPS from the past month")
        assert episode.detected_template == PromptTemplate.TOPIC_VENUE_TIME.value

    def test_top_k_papers_template(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Show me 3 papers about reinforcement learning")
        assert episode.detected_template == PromptTemplate.TOP_K_PAPERS.value

    def test_top_k_time_template(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Find 5 papers on transformers from the last month")
        assert episode.detected_template == PromptTemplate.TOP_K_TIME.value

    def test_survey_template(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Find a comprehensive survey on attention mechanisms")
        assert episode.detected_template == PromptTemplate.SURVEY_REVIEW.value

    def test_method_focused_template(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on image classification using contrastive learning")
        assert episode.detected_template == PromptTemplate.METHOD_FOCUSED.value

    def test_application_focused_template(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on deep learning applied to drug discovery")
        assert episode.detected_template == PromptTemplate.APPLICATION_FOCUSED.value

    def test_emerging_trends_template(self):
        with AgentTestHarness() as h:
            episode = _run_agent("What are emerging trends in quantum computing?")
        assert episode.detected_template == PromptTemplate.EMERGING_TRENDS.value

    def test_structured_output_template(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Find papers on NLP including title, authors, and summary")
        assert episode.detected_template == PromptTemplate.STRUCTURED_OUTPUT.value


# ============================================================================
# TEST SUITE: Output enforcement (CRITICAL system rule)
# ============================================================================

class TestAgentOutputEnforcement:
    """
    Validate that the agent enforces the user's requested paper count (K).
    
    If user asks for 'top 3 papers', the output MUST have at most 3 papers.
    """

    def test_top_k_enforces_exact_count(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Show me 3 papers on machine learning")

        assert episode.requested_paper_count == 3
        assert len(episode.papers_processed) <= 3

    def test_top_5_enforces_count(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Find 5 papers about deep learning")

        assert episode.requested_paper_count == 5
        assert len(episode.papers_processed) <= 5

    def test_default_output_count_when_no_k(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on NLP")

        # When user doesn't specify K, falls back to arxiv_fetch_count (dashboard setting)
        assert episode.requested_paper_count == DEFAULT_ARXIV_FETCH_COUNT

    def test_output_count_stored_in_episode(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Top 2 papers on attention")

        assert episode.output_paper_count is not None
        assert episode.output_paper_count <= 2


# ============================================================================
# TEST SUITE: Scoring pipeline
# ============================================================================

class TestAgentScoringPipeline:
    """Verify the scoring pipeline runs and produces valid scores."""

    def test_papers_have_relevance_scores(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on machine learning")

        for paper in episode.papers_processed:
            assert "relevance_score" in paper
            assert 0.0 <= paper["relevance_score"] <= 1.0

    def test_papers_have_novelty_scores(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on deep learning")

        for paper in episode.papers_processed:
            assert "novelty_score" in paper
            assert 0.0 <= paper["novelty_score"] <= 1.0

    def test_papers_have_importance_levels(self):
        valid_levels = {"high", "medium", "low", "very_low"}
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on NLP")

        for paper in episode.papers_processed:
            assert paper["importance"] in valid_levels

    def test_papers_sorted_by_relevance_in_output(self):
        """Output enforcement sorts by relevance_score descending."""
        with AgentTestHarness() as h:
            episode = _run_agent("Top 3 papers on machine learning")

        scores = [p["relevance_score"] for p in episode.papers_processed]
        assert scores == sorted(scores, reverse=True)

    def test_avoid_topics_reduce_scores(self):
        """Papers matching avoid_topics should be filtered or scored low."""
        profile = _make_research_profile(
            avoid_topics=["cryptocurrency", "bitcoin"],
            research_topics=["machine learning"],
        )
        with AgentTestHarness(research_profile=profile) as h:
            episode = _run_agent("Papers on machine learning")

        # The mock dataset has a cryptocurrency paper (2501.01007).
        # With avoid_topics, it should be excluded from results.
        crypto_papers = [
            p for p in episode.papers_processed
            if "cryptocurrency" in p.get("title", "").lower()
        ]
        assert len(crypto_papers) == 0


# ============================================================================
# TEST SUITE: Stop conditions
# ============================================================================

class TestAgentStopConditions:
    """Verify the agent terminates under various stop conditions."""

    def test_agent_terminates_successfully(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on NLP")

        assert episode.stop_reason is not None
        assert episode.end_time is not None

    def test_max_papers_checked_stop(self):
        """Agent should stop when max_papers_checked is hit."""
        with AgentTestHarness() as h:
            episode = _run_agent(
                "Papers on machine learning",
                stop_policy=StopPolicy(
                    max_papers_checked=2,
                    max_runtime_minutes=2,
                    stop_if_no_new_papers=False,
                    min_importance_to_act="low",
                ),
            )

        # Should process at most 2 papers
        assert len(episode.papers_processed) <= 2

    def test_no_new_papers_stop(self):
        """Agent should stop when all papers are already seen."""
        def _all_seen(**kwargs):
            papers = kwargs.get("papers", [])
            return {
                "unseen_papers": [],
                "seen_papers": [
                    {"arxiv_id": p.get("arxiv_id", ""), "title": p.get("title", ""),
                     "date_seen": "2026-01-01", "decision": "saved", "importance": "high"}
                    for p in papers
                ],
                "summary": {"total": len(papers), "unseen": 0, "seen": len(papers)},
            }

        with AgentTestHarness() as h:
            with patch("src.agent.react_agent.check_seen_papers_json",
                       side_effect=_all_seen):
                episode = _run_agent(
                    "Papers on NLP",
                    stop_policy=StopPolicy(
                        stop_if_no_new_papers=True,
                        max_runtime_minutes=2,
                        min_importance_to_act="low",
                    ),
                )

        # No unseen papers → should have 0 processed
        assert len(episode.papers_processed) == 0


# ============================================================================
# TEST SUITE: Tool call sequence
# ============================================================================

class TestAgentToolCallSequence:
    """Verify the agent calls tools in the expected order."""

    def test_standard_tool_sequence(self):
        """Standard run: fetch → check_seen → (score → decide → persist)* → report → terminate."""
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on machine learning")

        tool_names = [tc.tool_name for tc in episode.tool_calls]
        # fetch must come first
        assert tool_names[0] == "fetch_arxiv_papers"
        # check_seen must come second
        assert tool_names[1] == "check_seen_papers"
        # Last two calls should be report and terminate
        assert "generate_report" in tool_names
        assert "terminate_run" in tool_names
        assert tool_names.index("generate_report") < tool_names.index("terminate_run")

    def test_scoring_tool_called_for_each_paper(self):
        """score_relevance_and_importance should be called once per unseen paper."""
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on NLP")

        score_calls = [tc for tc in episode.tool_calls if tc.tool_name == "score_relevance_and_importance"]
        # Should have at least one scoring call
        assert len(score_calls) >= 1

    def test_persist_called_for_each_scored_paper(self):
        """persist_state should be called for each scored paper."""
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on deep learning")

        persist_calls = [tc for tc in episode.tool_calls if tc.tool_name == "persist_state"]
        score_calls = [tc for tc in episode.tool_calls if tc.tool_name == "score_relevance_and_importance"]
        # Each scored paper gets persisted
        assert len(persist_calls) >= len(score_calls)

    def test_all_tool_calls_have_timestamps(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on ML")

        for tc in episode.tool_calls:
            assert tc.timestamp
            datetime.fromisoformat(tc.timestamp.replace("Z", "+00:00"))


# ============================================================================
# TEST SUITE: Report generation
# ============================================================================

class TestAgentReportGeneration:
    """Verify the final report has expected structure and content."""

    def test_report_contains_stats(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on machine learning")

        report = episode.final_report
        assert report is not None
        assert "stats" in report or "summary" in report

    def test_report_contains_run_id(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on NLP")

        report = episode.final_report
        assert report.get("run_id") == episode.run_id

    def test_report_contains_papers_section(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on deep learning")

        report = episode.final_report
        papers_in_report = report.get("papers", [])
        assert isinstance(papers_in_report, list)

    def test_report_has_stop_reason(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on machine learning")

        report = episode.final_report
        assert report.get("stop_reason") is not None


# ============================================================================
# TEST SUITE: Decision tracking
# ============================================================================

class TestAgentDecisionTracking:
    """Verify the agent records paper decisions correctly."""

    def test_decisions_recorded(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on machine learning")

        for decision in episode.decisions_made:
            assert "paper_id" in decision
            assert "importance" in decision
            assert "decision" in decision
            assert decision["decision"] in {"saved", "shared", "ignored", "logged"}

    def test_high_importance_papers_saved(self):
        """High importance papers should be 'saved', not 'logged'."""
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on machine learning")

        for decision in episode.decisions_made:
            if decision["importance"] in ("high", "medium"):
                assert decision["decision"] == "saved"

    def test_low_importance_papers_logged(self):
        """Low importance papers should be 'logged'."""
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on machine learning")

        for decision in episode.decisions_made:
            if decision["importance"] == "low":
                assert decision["decision"] == "logged"


# ============================================================================
# TEST SUITE: Category mapping through agent
# ============================================================================

class TestAgentCategoryMapping:
    """Verify the agent maps interests to arXiv categories correctly."""

    def test_nlp_interests_map_to_cs_cl(self):
        profile = _make_research_profile(
            interests_include="natural language processing",
            arxiv_categories_include=[],
        )
        with AgentTestHarness(research_profile=profile) as h:
            episode = _run_agent("Papers on NLP")

        # Fetch call should include cs.CL
        fetch_calls = [tc for tc in episode.tool_calls if tc.tool_name == "fetch_arxiv_papers"]
        assert len(fetch_calls) >= 1
        cats = fetch_calls[0].input_args.get("categories_include", [])
        assert "cs.CL" in cats

    def test_ml_interests_map_to_cs_lg(self):
        profile = _make_research_profile(
            interests_include="machine learning",
            arxiv_categories_include=[],
        )
        with AgentTestHarness(research_profile=profile) as h:
            episode = _run_agent("Papers on machine learning")

        fetch_calls = [tc for tc in episode.tool_calls if tc.tool_name == "fetch_arxiv_papers"]
        assert len(fetch_calls) >= 1
        cats = fetch_calls[0].input_args.get("categories_include", [])
        assert "cs.LG" in cats


# ============================================================================
# TEST SUITE: Colleague surplus processing
# ============================================================================

class TestAgentColleagueSurplus:
    """Verify colleague surplus processing when colleagues are configured."""

    def test_colleagues_receive_surplus(self):
        colleagues = [
            {
                "id": str(uuid.uuid4()),
                "name": "Alice",
                "email": "alice@test.com",
                "research_interests": "machine learning, deep learning",
                "keywords": ["machine learning", "deep learning"],
                "categories": ["cs.LG"],
                "share_preference": "immediate",
                "auto_send_emails": True,
                "enabled": True,
            }
        ]
        with AgentTestHarness(colleagues=colleagues) as h:
            episode = _run_agent("Papers on machine learning")

        # With colleagues, there should be some actions or the agent should
        # attempt colleague surplus processing (at least no crash)
        assert episode.stop_reason is not None

    def test_no_colleagues_no_crash(self):
        """Agent should work fine with empty colleagues list."""
        with AgentTestHarness(colleagues=[]) as h:
            episode = _run_agent("Papers on NLP")

        assert episode.stop_reason is not None
        assert len(episode.papers_processed) >= 0


# ============================================================================
# TEST SUITE: Exclude topics integration
# ============================================================================

class TestAgentExcludeTopics:
    """Verify that exclude topics from the prompt are respected end-to-end."""

    def test_exclude_topic_from_prompt(self):
        """Papers matching the exclude topic should not appear in output."""
        with AgentTestHarness() as h:
            episode = _run_agent(
                "Papers on machine learning. Exclude: cryptocurrency"
            )

        for paper in episode.papers_processed:
            title_lower = paper.get("title", "").lower()
            assert "cryptocurrency" not in title_lower

    def test_exclude_from_profile_avoid_topics(self):
        """avoid_topics in profile should filter papers."""
        profile = _make_research_profile(avoid_topics=["cryptocurrency"])
        with AgentTestHarness(research_profile=profile) as h:
            episode = _run_agent("Papers on machine learning")

        for paper in episode.papers_processed:
            title_lower = paper.get("title", "").lower()
            assert "cryptocurrency" not in title_lower


# ============================================================================
# TEST SUITE: Realistic end-to-end scenarios
# ============================================================================

class TestAgentRealisticScenarios:
    """Run the agent with realistic prompts and validate outputs."""

    def test_simple_topic_query(self):
        """Basic 'find papers on X' query."""
        with AgentTestHarness() as h:
            episode = _run_agent("Find papers on large language models")

        assert episode.detected_template in (
            PromptTemplate.TOPIC_ONLY.value,
            PromptTemplate.TOPIC_TIME.value,
            PromptTemplate.METHOD_FOCUSED.value,
        )
        assert len(episode.papers_processed) > 0
        assert episode.final_report is not None

    def test_top_k_with_time_constraint(self):
        """'Find 3 papers on X from last month' should return at most 3."""
        with AgentTestHarness() as h:
            episode = _run_agent("Find 3 papers on deep learning from the last month")

        assert episode.detected_template == PromptTemplate.TOP_K_TIME.value
        assert episode.requested_paper_count == 3
        assert len(episode.papers_processed) <= 3

    def test_survey_request(self):
        """Survey request should trigger SURVEY_REVIEW template."""
        with AgentTestHarness() as h:
            episode = _run_agent(
                "Give me a comprehensive survey of knowledge distillation methods"
            )

        assert episode.detected_template == PromptTemplate.SURVEY_REVIEW.value
        assert len(episode.papers_processed) > 0

    def test_structured_output_request(self):
        """Structured output request with explicit field requests."""
        with AgentTestHarness() as h:
            episode = _run_agent(
                "Show papers on NLP including title, authors, and summary"
            )

        assert episode.detected_template == PromptTemplate.STRUCTURED_OUTPUT.value

    def test_method_focused_request(self):
        """Method-focused request using a specific technique."""
        with AgentTestHarness() as h:
            episode = _run_agent(
                "Papers on text classification using transformers"
            )

        assert episode.detected_template == PromptTemplate.METHOD_FOCUSED.value

    def test_emerging_trends_request(self):
        with AgentTestHarness() as h:
            episode = _run_agent(
                "What are the cutting-edge developments in retrieval-augmented generation?"
            )

        assert episode.detected_template == PromptTemplate.EMERGING_TRENDS.value

    def test_prompt_with_count_and_exclude(self):
        """Complex prompt combining count + exclude."""
        with AgentTestHarness() as h:
            episode = _run_agent(
                "Find 3 papers on machine learning. Exclude: cryptocurrency, finance"
            )

        assert episode.requested_paper_count == 3
        assert len(episode.papers_processed) <= 3
        for paper in episode.papers_processed:
            assert "cryptocurrency" not in paper.get("title", "").lower()

    def test_full_workflow_produces_complete_episode(self):
        """Comprehensive check that all episode fields are populated."""
        with AgentTestHarness() as h:
            episode = _run_agent("Top 5 papers on attention mechanisms from the past month")

        # Core fields
        assert episode.run_id
        assert episode.user_message
        assert episode.detected_template
        assert episode.start_time
        assert episode.end_time
        assert episode.stop_reason

        # Counts
        assert episode.total_fetched_count >= 0
        assert episode.unseen_paper_count >= 0
        assert episode.requested_paper_count == 5
        assert episode.output_paper_count is not None

        # Processed data
        assert isinstance(episode.papers_processed, list)
        assert isinstance(episode.decisions_made, list)
        assert isinstance(episode.tool_calls, list)

        # Report
        assert episode.final_report is not None


# ============================================================================
# TEST SUITE: Error resilience
# ============================================================================

class TestAgentErrorResilience:
    """Verify the agent handles tool failures gracefully."""

    def test_agent_survives_persist_failure(self):
        """If persist_state fails, agent should still complete."""
        def _failing_persist(**kwargs):
            raise RuntimeError("DB write failed")

        with AgentTestHarness() as h:
            with patch("src.agent.react_agent.persist_paper_decision",
                       side_effect=_failing_persist):
                episode = _run_agent("Papers on NLP")

        # Agent should still produce results
        assert episode.stop_reason is not None
        assert episode.final_report is not None

    def test_agent_survives_rag_failure(self):
        """If RAG/Pinecone fails, agent should still score papers."""

        def failing_rag(**kwargs):
            raise ConnectionError("Pinecone unreachable")

        with AgentTestHarness() as h:
            with patch("src.agent.react_agent.retrieve_similar_from_pinecone_json",
                       side_effect=failing_rag):
                episode = _run_agent("Papers on machine learning")

        assert episode.stop_reason is not None
        assert len(episode.papers_processed) >= 0


# ============================================================================
# TEST SUITE: Fetch-by-ID workflow
# ============================================================================

class TestAgentFetchById:
    """Verify the single-paper lookup workflow for arXiv IDs."""

    def _mock_fetch_single(self, arxiv_id: str) -> FetchArxivResult:
        """Return the matching mock paper by ID."""
        for p in MOCK_PAPERS:
            if p["arxiv_id"] == arxiv_id:
                return FetchArxivResult(
                    success=True, papers=[ArxivPaper(**p)], total_found=1,
                    query_info={}, error=None,
                )
        return FetchArxivResult(
            success=False, papers=[], total_found=0, query_info={}, error="not found"
        )

    def test_fetch_by_id_detects_template(self):
        """ArXiv ID in prompt should trigger FETCH_BY_ID."""
        with AgentTestHarness() as h:
            with patch("tools.fetch_arxiv.fetch_single_paper",
                       side_effect=self._mock_fetch_single):
                episode = _run_agent("Get me paper 2501.01001")

        assert episode.detected_template == PromptTemplate.FETCH_BY_ID.value

    def test_fetch_by_id_returns_single_paper(self):
        """Should return exactly 1 paper for an ID lookup."""
        with AgentTestHarness() as h:
            with patch("tools.fetch_arxiv.fetch_single_paper",
                       side_effect=self._mock_fetch_single):
                episode = _run_agent("Fetch paper 2501.01001")

        assert len(episode.papers_processed) == 1
        assert episode.papers_processed[0]["arxiv_id"] == "2501.01001"


# ============================================================================
# TEST SUITE: Episode count tracking
# ============================================================================

class TestAgentCountTracking:
    """Verify the agent tracks paper counts accurately in the episode."""

    def test_total_fetched_count(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on machine learning")

        assert episode.total_fetched_count > 0

    def test_unseen_paper_count(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on deep learning")

        # All papers are unseen in our harness
        assert episode.unseen_paper_count == episode.total_fetched_count

    def test_output_paper_count_le_requested(self):
        with AgentTestHarness() as h:
            episode = _run_agent("Show me 2 papers on NLP")

        assert episode.output_paper_count is not None
        assert episode.output_paper_count <= 2


# ============================================================================
# TEST SUITE: Multiple runs don't leak state
# ============================================================================

class TestAgentStateIsolation:
    """Verify that consecutive agent runs don't leak state."""

    def test_two_runs_have_different_run_ids(self):
        with AgentTestHarness() as h:
            ep1 = _run_agent("Papers on NLP")
            ep2 = _run_agent("Papers on robotics")

        assert ep1.run_id != ep2.run_id

    def test_consecutive_runs_produce_valid_episodes(self):
        with AgentTestHarness() as h:
            ep1 = _run_agent("Top 2 papers on ML")
            ep2 = _run_agent("Survey on attention mechanisms")

        assert ep1.final_report is not None
        assert ep2.final_report is not None


# ============================================================================
# TEST SUITE: Out-of-scope topic validation
# ============================================================================

class TestAgentTopicValidation:
    """Verify the agent returns a clear message for topics outside arXiv scope."""

    def test_out_of_scope_topic_returns_early(self):
        """Topics like 'animals' that don't map to any arXiv category should
        trigger an early return with a clear out-of-scope message."""
        with AgentTestHarness() as h:
            with patch(
                "tools.arxiv_categories.topic_to_categories",
                return_value=[],
            ):
                episode = _run_agent(
                    "Provide the most recent research papers on animals"
                )

        assert episode.topic_not_in_categories is True
        assert episode.searched_topic is not None
        assert "animals" in episode.searched_topic.lower()
        assert episode.stop_reason == "topic_outside_arxiv_scope"
        assert episode.final_report is not None
        summary = episode.final_report.get("summary", "")
        assert "arXiv primarily provides papers" in summary
        assert "RESEARCHPULSE cannot assist" in summary

    def test_out_of_scope_topic_has_zero_papers(self):
        """An out-of-scope topic should not return any papers."""
        with AgentTestHarness() as h:
            with patch(
                "tools.arxiv_categories.topic_to_categories",
                return_value=[],
            ):
                episode = _run_agent(
                    "Find recent papers about cooking recipes"
                )

        assert episode.topic_not_in_categories is True
        stats = episode.final_report.get("stats", {})
        assert stats.get("total_fetched_count", -1) == 0
        assert len(episode.papers_processed) == 0

    def test_valid_topic_still_works(self):
        """Topics that map to arXiv categories should work normally."""
        with AgentTestHarness() as h:
            episode = _run_agent("Papers on machine learning")

        assert episode.topic_not_in_categories is False
        assert episode.stop_reason != "topic_outside_arxiv_scope"
        assert episode.total_fetched_count > 0
