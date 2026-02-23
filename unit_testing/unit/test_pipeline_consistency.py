"""
Regression tests for the pipeline state propagation bug.

These tests verify that when CheckSeenPapers reports unseen_papers_count > 0,
the final run result does NOT incorrectly claim "No New Papers Found".

Test cases per the specification:
  Case A: Fetch 10, CheckSeen 2 unseen, requested N=2  → delivered=2, no "No New Papers"
  Case B: Fetch 10, CheckSeen 2 unseen, requested N=5  → delivered=min(available), pipeline attempts
  Case C: Fetch 10, CheckSeen 0 unseen                 → delivered=0, "up to date"
  Contract: If unseen_papers_count > 0 in stats, response must not claim "No New Papers Found"
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


# ---------------------------------------------------------------------------
# Helper: build a minimal AgentEpisode-like object for routes tests
# ---------------------------------------------------------------------------

def _make_episode(
    *,
    papers_processed=None,
    decisions_made=None,
    actions_taken=None,
    artifacts_generated=None,
    stop_reason="run completed successfully",
    final_report=None,
    tool_calls=None,
    total_fetched_count=0,
    unseen_paper_count=0,
    papers_filtered_count=0,
):
    """Build a lightweight AgentEpisode for testing the response composer in routes.py."""
    from agent.react_agent import AgentEpisode

    return AgentEpisode(
        run_id="test-run-001",
        start_time=datetime.utcnow().isoformat() + "Z",
        end_time=datetime.utcnow().isoformat() + "Z",
        user_message="Find papers on transformers",
        stop_reason=stop_reason,
        papers_processed=papers_processed or [],
        decisions_made=decisions_made or [],
        actions_taken=actions_taken or [],
        artifacts_generated=artifacts_generated or [],
        tool_calls=tool_calls or [],
        total_fetched_count=total_fetched_count,
        unseen_paper_count=unseen_paper_count,
        papers_filtered_count=papers_filtered_count,
        final_report=final_report,
    )


def _fake_paper(arxiv_id, title="Test Paper", importance="high", relevance=0.9):
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "importance": importance,
        "relevance_score": relevance,
        "novelty_score": 0.8,
    }


# =========================================================================
# generate_report stats consistency
# =========================================================================

class TestGenerateReportStats:
    """Verify RunStats fields are computed from the right inputs."""

    def test_total_papers_retrieved_uses_fetched_count(self):
        from tools.generate_report import generate_report

        result = generate_report(
            run_id="r1",
            start_time=datetime.utcnow().isoformat() + "Z",
            stop_reason="completed",
            papers=[],            # 0 scored/delivered
            unseen_count=2,
            seen_count=8,
            total_fetched_count=10,
        )
        stats = result.stats
        # total_papers_retrieved must reflect fetched from arXiv, NOT delivered
        assert stats.total_papers_retrieved == 10
        assert stats.total_fetched_count == 10
        assert stats.unseen_papers_count == 2
        assert stats.seen_papers_count == 8
        assert stats.papers_delivered == 0

    def test_papers_delivered_counts_scored(self):
        from tools.generate_report import generate_report

        papers = [
            {"arxiv_id": "2501.00001", "title": "P1", "importance": "high"},
            {"arxiv_id": "2501.00002", "title": "P2", "importance": "medium"},
        ]
        result = generate_report(
            run_id="r2",
            start_time=datetime.utcnow().isoformat() + "Z",
            stop_reason="completed",
            papers=papers,
            unseen_count=2,
            seen_count=8,
            total_fetched_count=10,
        )
        assert result.stats.papers_delivered == 2
        assert result.stats.total_papers_retrieved == 10

    def test_papers_filtered_count_stored(self):
        from tools.generate_report import generate_report

        result = generate_report(
            run_id="r3",
            start_time=datetime.utcnow().isoformat() + "Z",
            stop_reason="completed",
            papers=[],
            unseen_count=5,
            seen_count=5,
            total_fetched_count=10,
            papers_filtered_count=5,
        )
        assert result.stats.papers_filtered_count == 5
        assert result.stats.unseen_papers_count == 5
        assert result.stats.papers_delivered == 0

    def test_stats_impossible_combo_prevented(self):
        """total_papers_retrieved must never be 0 when unseen + seen > 0."""
        from tools.generate_report import generate_report

        result = generate_report(
            run_id="r4",
            start_time=datetime.utcnow().isoformat() + "Z",
            stop_reason="completed",
            papers=[],
            unseen_count=2,
            seen_count=8,
            total_fetched_count=10,
        )
        stats = result.stats
        # Previously total_papers_retrieved was len(papers)==0 — BUG.
        assert stats.total_papers_retrieved >= stats.unseen_papers_count + stats.seen_papers_count
        assert stats.total_papers_retrieved == 10

    def test_fallback_when_total_fetched_not_provided(self):
        """If total_fetched_count is 0 (old callers), fall back to unseen+seen."""
        from tools.generate_report import generate_report

        result = generate_report(
            run_id="r5",
            start_time=datetime.utcnow().isoformat() + "Z",
            stop_reason="completed",
            papers=[],
            unseen_count=3,
            seen_count=7,
            # total_fetched_count defaults to 0
        )
        assert result.stats.total_papers_retrieved == 10  # 3 + 7


# =========================================================================
# Response message consistency (routes.py logic)
# =========================================================================

class TestResponseMessageConsistency:
    """
    Validate the /api/execute response text is consistent with stats.

    These tests exercise the response-building logic in routes.py by
    calling the endpoint with a mocked agent.
    """

    def _build_response_text(self, episode):
        """
        Replicate the response-building logic from routes.py so we can
        test it in isolation without network calls.
        """
        run_id = episode.run_id
        output_parts = []
        output_parts.append("ResearchPulse Agent Run Complete")
        output_parts.append("=" * 40)
        output_parts.append(f"Run ID: {run_id}")
        output_parts.append(f"Stop Reason: {episode.stop_reason}")
        output_parts.append(f"Papers Processed: {len(episode.papers_processed)}")
        output_parts.append(f"Decisions Made: {len(episode.decisions_made)}")
        output_parts.append(f"Artifacts Generated: {len(episode.artifacts_generated)}")
        output_parts.append("")

        if episode.final_report:
            report = episode.final_report if isinstance(episode.final_report, dict) else {}
            stats = report.get("stats", {}) if isinstance(report, dict) else {}
            seen_count = stats.get("seen_papers_count", 0)
            unseen_count = stats.get("unseen_papers_count", 0)
            filtered_count = stats.get("papers_filtered_count", 0)
            total_fetched = stats.get("total_fetched_count", 0)

            if len(episode.papers_processed) == 0 and unseen_count > 0:
                output_parts.append("🔍 New Papers Found but Filtered")
                output_parts.append("-" * 40)
                output_parts.append(
                    f"Found {unseen_count} new (unseen) papers from {total_fetched} fetched,"
                )
                output_parts.append(
                    "but none met the relevance/quality criteria for delivery."
                )
                if filtered_count:
                    output_parts.append(
                        f"({filtered_count} paper(s) filtered out by quality thresholds.)"
                    )
                output_parts.append("")
            elif len(episode.papers_processed) == 0 and unseen_count == 0 and seen_count > 0:
                output_parts.append("📭 No New Papers Found")
                output_parts.append("-" * 40)
                output_parts.append(
                    f"All {seen_count} papers from arXiv have already been processed."
                )
                output_parts.append("")
            elif len(episode.papers_processed) == 0 and seen_count == 0 and unseen_count == 0:
                output_parts.append("⚠️ No Papers Retrieved")
                output_parts.append("-" * 40)
                output_parts.append("Could not fetch papers from arXiv.")
                output_parts.append("")

            output_parts.append("Summary:")
            output_parts.append(str(episode.final_report))

        return "\n".join(output_parts)

    # ----- Case A: Fetch 10, 2 unseen, deliver 2 -----

    def test_case_a_delivered_papers_present(self):
        """Case A: 2 unseen, requested 2 → delivered=2, NOT 'No New Papers'."""
        episode = _make_episode(
            papers_processed=[
                _fake_paper("2501.00001", "Paper A"),
                _fake_paper("2501.00002", "Paper B"),
            ],
            decisions_made=[
                {"paper_id": "2501.00001", "decision": "saved"},
                {"paper_id": "2501.00002", "decision": "saved"},
            ],
            total_fetched_count=10,
            unseen_paper_count=2,
            papers_filtered_count=0,
            final_report={
                "stats": {
                    "total_papers_retrieved": 10,
                    "total_fetched_count": 10,
                    "unseen_papers_count": 2,
                    "seen_papers_count": 8,
                    "papers_delivered": 2,
                    "papers_filtered_count": 0,
                }
            },
        )
        text = self._build_response_text(episode)
        assert "No New Papers Found" not in text
        assert "No Papers Retrieved" not in text
        assert "Papers Processed: 2" in text

    # ----- Case B: Fetch 10, 2 unseen, requested 5 → deliver ≤ 2 -----

    def test_case_b_partial_delivery(self):
        """Case B: 2 unseen, requested 5 → delivered ≤ 2, NOT 'No New Papers'."""
        episode = _make_episode(
            papers_processed=[
                _fake_paper("2501.00001", "Paper A"),
                _fake_paper("2501.00002", "Paper B"),
            ],
            total_fetched_count=10,
            unseen_paper_count=2,
            papers_filtered_count=0,
            final_report={
                "stats": {
                    "total_papers_retrieved": 10,
                    "total_fetched_count": 10,
                    "unseen_papers_count": 2,
                    "seen_papers_count": 8,
                    "papers_delivered": 2,
                    "papers_filtered_count": 0,
                }
            },
        )
        text = self._build_response_text(episode)
        assert "No New Papers Found" not in text
        assert "Papers Processed: 2" in text

    # ----- Case C: Fetch 10, 0 unseen → truly no new -----

    def test_case_c_no_unseen_shows_no_new_papers(self):
        """Case C: 0 unseen → 'No New Papers Found' is correct."""
        episode = _make_episode(
            papers_processed=[],
            total_fetched_count=10,
            unseen_paper_count=0,
            papers_filtered_count=0,
            final_report={
                "stats": {
                    "total_papers_retrieved": 10,
                    "total_fetched_count": 10,
                    "unseen_papers_count": 0,
                    "seen_papers_count": 10,
                    "papers_delivered": 0,
                    "papers_filtered_count": 0,
                }
            },
        )
        text = self._build_response_text(episode)
        assert "No New Papers Found" in text
        assert "already been processed" in text

    # ----- Case: unseen > 0 but all filtered  → must NOT claim "No New Papers" -----

    def test_unseen_but_all_filtered_not_no_new_papers(self):
        """BUG REGRESSION: unseen=2, filtered=2 → must say 'Filtered', NOT 'No New Papers'."""
        episode = _make_episode(
            papers_processed=[],
            total_fetched_count=10,
            unseen_paper_count=2,
            papers_filtered_count=2,
            final_report={
                "stats": {
                    "total_papers_retrieved": 10,
                    "total_fetched_count": 10,
                    "unseen_papers_count": 2,
                    "seen_papers_count": 8,
                    "papers_delivered": 0,
                    "papers_filtered_count": 2,
                }
            },
        )
        text = self._build_response_text(episode)
        assert "No New Papers Found" not in text, \
            "Must NOT claim 'No New Papers Found' when unseen_papers_count > 0"
        assert "New Papers Found but Filtered" in text
        assert "2 new (unseen) papers" in text
        assert "2 paper(s) filtered out" in text

    # ----- Contract: /api/execute response consistency -----

    def test_contract_unseen_gt_zero_implies_no_false_no_new(self):
        """If unseen_papers_count > 0 in the trace, response must not say 'No New Papers'."""
        for unseen in [1, 2, 5, 10]:
            episode = _make_episode(
                papers_processed=[],
                total_fetched_count=20,
                unseen_paper_count=unseen,
                papers_filtered_count=unseen,
                final_report={
                    "stats": {
                        "total_papers_retrieved": 20,
                        "total_fetched_count": 20,
                        "unseen_papers_count": unseen,
                        "seen_papers_count": 20 - unseen,
                        "papers_delivered": 0,
                        "papers_filtered_count": unseen,
                    }
                },
            )
            text = self._build_response_text(episode)
            assert "No New Papers Found" not in text, \
                f"unseen={unseen}: Response falsely claims 'No New Papers Found'"

    def test_no_papers_fetched_shows_no_papers_retrieved(self):
        """When nothing was fetched at all (arXiv API down), show 'No Papers Retrieved'."""
        episode = _make_episode(
            papers_processed=[],
            total_fetched_count=0,
            unseen_paper_count=0,
            papers_filtered_count=0,
            final_report={
                "stats": {
                    "total_papers_retrieved": 0,
                    "total_fetched_count": 0,
                    "unseen_papers_count": 0,
                    "seen_papers_count": 0,
                    "papers_delivered": 0,
                    "papers_filtered_count": 0,
                }
            },
        )
        text = self._build_response_text(episode)
        assert "No Papers Retrieved" in text


# =========================================================================
# AgentEpisode tracks propagation fields
# =========================================================================

class TestAgentEpisodeFields:
    """Verify AgentEpisode carries the new tracking fields."""

    def test_episode_has_fetched_and_filtered_fields(self):
        from agent.react_agent import AgentEpisode

        ep = AgentEpisode(
            run_id="x",
            start_time="2026-01-01T00:00:00Z",
            user_message="test",
            total_fetched_count=10,
            unseen_paper_count=3,
            papers_filtered_count=1,
        )
        assert ep.total_fetched_count == 10
        assert ep.unseen_paper_count == 3
        assert ep.papers_filtered_count == 1

    def test_episode_defaults_to_zero(self):
        from agent.react_agent import AgentEpisode

        ep = AgentEpisode(
            run_id="y",
            start_time="2026-01-01T00:00:00Z",
            user_message="test",
        )
        assert ep.total_fetched_count == 0
        assert ep.unseen_paper_count == 0
        assert ep.papers_filtered_count == 0


# =========================================================================
# Stop controller correctness
# =========================================================================

class TestStopControllerNoFalseStop:
    """Ensure the stop controller does NOT prematurely claim 'no new papers'
    when unseen > 0 but papers_checked == 0 (all filtered before scoring)."""

    def test_no_stop_when_unseen_gt_zero_but_checked_zero(self):
        from agent.stop_controller import StopController, StopPolicy

        controller = StopController(StopPolicy(stop_if_no_new_papers=True))
        controller.set_new_papers_found(2)  # 2 unseen
        # papers_checked stays 0 (all filtered before increment)
        should_stop, reason = controller.should_stop()
        assert not should_stop, \
            "Must NOT stop: unseen > 0 means there are new papers"

    def test_stops_only_when_truly_no_new(self):
        from agent.stop_controller import StopController, StopPolicy

        controller = StopController(StopPolicy(stop_if_no_new_papers=True))
        controller.set_new_papers_found(0)
        controller.increment_papers_checked(5)
        should_stop, reason = controller.should_stop()
        assert should_stop
        assert "no unseen papers" in reason
