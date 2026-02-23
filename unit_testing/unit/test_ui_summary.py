"""
Tests for the chat summary UI rendering logic.

These tests verify the deterministic data transformations that power
the chat summary in static/index.html:
  - extractDataFromSteps: structured extraction from pipeline steps
  - Paper deduplication by arxiv_id
  - Collapsed-by-default behaviour for execution steps

The tests replicate the JavaScript logic in Python to validate correctness
without requiring a browser runtime.
"""

import json
import re
import pytest


# ---------------------------------------------------------------------------
# Python mirrors of the JS helper functions
# ---------------------------------------------------------------------------

def extract_data_from_steps(steps):
    """Python mirror of extractDataFromSteps() in index.html."""
    if not steps:
        return None

    data = {
        "fetchedCount": 0,
        "seenCount": 0,
        "unseenCount": 0,
        "deliveredCount": 0,
        "papers": [],
        "seenPapers": [],
        "unseenPapers": [],
        "deliveredPapers": [],
        "actions": [],
        "stopReason": "",
    }
    paper_map = {}
    seen_ids = set()
    unseen_ids = set()
    delivered_ids = set()

    for step in steps:
        mod = step.get("module", "")
        result = (step.get("response") or {}).get("result") or {}

        if mod == "FetchArxivPapers":
            if isinstance(result.get("papers"), list):
                data["fetchedCount"] = len(result["papers"])
                for p in result["papers"]:
                    pid = p.get("arxiv_id") or p.get("external_id") or ""
                    if pid and pid not in paper_map:
                        paper_map[pid] = {
                            "title": p.get("title", ""),
                            "arxiv_id": pid,
                            "published": p.get("published") or p.get("date", ""),
                            "importance": None,
                            "relevance": None,
                            "explanation": "",
                            "status": "fetched",
                        }
            if isinstance(result.get("total_found"), (int, float)):
                data["fetchedCount"] = max(data["fetchedCount"], int(result["total_found"]))

        if mod == "CheckSeenPapers":
            summary = result.get("summary") or {}
            if isinstance(summary.get("seen"), (int, float)):
                data["seenCount"] = int(summary["seen"])
            if isinstance(summary.get("unseen"), (int, float)):
                data["unseenCount"] = int(summary["unseen"])
            if isinstance(summary.get("total"), (int, float)) and data["fetchedCount"] == 0:
                data["fetchedCount"] = int(summary["total"])
            for p in (result.get("seen_papers") or []):
                pid = p.get("arxiv_id") or p.get("external_id") or (p if isinstance(p, str) else "")
                if pid:
                    seen_ids.add(pid)
            for p in (result.get("unseen_papers") or []):
                pid = p.get("arxiv_id") or p.get("external_id") or (p if isinstance(p, str) else "")
                if pid:
                    unseen_ids.add(pid)

        if mod == "ScoreRelevanceAndImportance":
            inp = step.get("prompt") or {}
            paper_obj = inp.get("paper") or {}
            pid = paper_obj.get("arxiv_id") or paper_obj.get("external_id") or inp.get("arxiv_id", "")
            if pid and pid in paper_map:
                entry = paper_map[pid]
                if isinstance(result.get("importance"), (int, float)):
                    entry["importance"] = result["importance"]
                if isinstance(result.get("relevance_score"), (int, float)):
                    entry["relevance"] = result["relevance_score"]
                if result.get("explanation"):
                    entry["explanation"] = result["explanation"]

        if mod == "DecideDeliveryAction":
            action = result.get("action") or result.get("decision") or ""
            inp = step.get("prompt") or {}
            paper_obj = inp.get("paper") or {}
            pid = paper_obj.get("arxiv_id") or paper_obj.get("external_id") or inp.get("arxiv_id", "")
            if action:
                data["actions"].append(action)
            if action in ("deliver", "email", "send"):
                data["deliveredCount"] += 1
                if pid:
                    delivered_ids.add(pid)

        if mod == "TerminateRun":
            data["stopReason"] = result.get("reason") or result.get("stop_reason") or ""

    # Assign status and sort into groups
    for pid, paper in paper_map.items():
        if pid in delivered_ids:
            paper["status"] = "delivered"
            data["deliveredPapers"].append(paper)
        elif pid in unseen_ids:
            paper["status"] = "new"
            data["unseenPapers"].append(paper)
        elif pid in seen_ids:
            paper["status"] = "seen"
            data["seenPapers"].append(paper)
        else:
            paper["status"] = "fetched"
            data["seenPapers"].append(paper)

    data["papers"] = list(paper_map.values())
    return data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_steps():
    """A realistic set of pipeline steps with two papers."""
    return [
        {
            "module": "FetchArxivPapers",
            "prompt": {"categories": ["cs.AI"]},
            "response": {
                "result": {
                    "papers": [
                        {"arxiv_id": "2401.00001", "title": "Paper A", "published": "2024-01-15"},
                        {"arxiv_id": "2401.00002", "title": "Paper B", "published": "2024-01-16"},
                        {"arxiv_id": "2401.00001", "title": "Paper A", "published": "2024-01-15"},  # duplicate
                    ],
                    "total_found": 3,
                },
                "success": True,
            },
        },
        {
            "module": "CheckSeenPapers",
            "prompt": {},
            "response": {
                "result": {
                    "unseen_papers": [{"arxiv_id": "2401.00001"}, {"arxiv_id": "2401.00002"}],
                    "seen_papers": [],
                    "summary": {"total": 3, "unseen": 2, "seen": 1},
                },
                "success": True,
            },
        },
        {
            "module": "ScoreRelevanceAndImportance",
            "prompt": {"paper": {"arxiv_id": "2401.00001"}},
            "response": {
                "result": {"importance": 8, "relevance_score": 9, "explanation": "Highly relevant"},
                "success": True,
            },
        },
        {
            "module": "ScoreRelevanceAndImportance",
            "prompt": {"paper": {"arxiv_id": "2401.00002"}},
            "response": {
                "result": {"importance": 5, "relevance_score": 3, "explanation": "Marginal"},
                "success": True,
            },
        },
        {
            "module": "DecideDeliveryAction",
            "prompt": {"paper": {"arxiv_id": "2401.00001"}},
            "response": {"result": {"action": "deliver"}, "success": True},
        },
        {
            "module": "DecideDeliveryAction",
            "prompt": {"paper": {"arxiv_id": "2401.00002"}},
            "response": {"result": {"action": "skip"}, "success": True},
        },
        {
            "module": "TerminateRun",
            "prompt": {},
            "response": {"result": {"reason": "all_processed"}, "success": True},
        },
    ]


# ---------------------------------------------------------------------------
# Tests: extractDataFromSteps
# ---------------------------------------------------------------------------

class TestExtractDataFromSteps:
    """Verify the structured step extraction logic."""

    def test_returns_none_for_empty(self):
        assert extract_data_from_steps([]) is None
        assert extract_data_from_steps(None) is None

    def test_fetched_count(self, sample_steps):
        data = extract_data_from_steps(sample_steps)
        assert data["fetchedCount"] == 3  # total_found takes precedence

    def test_seen_unseen_counts(self, sample_steps):
        data = extract_data_from_steps(sample_steps)
        assert data["seenCount"] == 1
        assert data["unseenCount"] == 2

    def test_delivered_count(self, sample_steps):
        data = extract_data_from_steps(sample_steps)
        assert data["deliveredCount"] == 1  # only 2401.00001 is "deliver"

    def test_papers_deduplicated(self, sample_steps):
        """Duplicate arxiv_ids in FetchArxivPapers must be collapsed."""
        data = extract_data_from_steps(sample_steps)
        ids = [p["arxiv_id"] for p in data["papers"]]
        assert ids == ["2401.00001", "2401.00002"]  # no duplicates

    def test_scores_attached_to_papers(self, sample_steps):
        data = extract_data_from_steps(sample_steps)
        by_id = {p["arxiv_id"]: p for p in data["papers"]}
        assert by_id["2401.00001"]["importance"] == 8
        assert by_id["2401.00001"]["relevance"] == 9
        assert by_id["2401.00002"]["importance"] == 5
        assert by_id["2401.00002"]["relevance"] == 3

    def test_actions_collected(self, sample_steps):
        data = extract_data_from_steps(sample_steps)
        assert data["actions"] == ["deliver", "skip"]

    def test_stop_reason(self, sample_steps):
        data = extract_data_from_steps(sample_steps)
        assert data["stopReason"] == "all_processed"

    def test_published_date_preserved(self, sample_steps):
        data = extract_data_from_steps(sample_steps)
        by_id = {p["arxiv_id"]: p for p in data["papers"]}
        assert by_id["2401.00001"]["published"] == "2024-01-15"

    def test_no_score_step_leaves_none(self):
        """Papers without a ScoreRelevanceAndImportance step keep None scores."""
        steps = [
            {
                "module": "FetchArxivPapers",
                "prompt": {},
                "response": {
                    "result": {
                        "papers": [{"arxiv_id": "2401.99999", "title": "Orphan Paper"}]
                    },
                    "success": True,
                },
            },
        ]
        data = extract_data_from_steps(steps)
        assert data["papers"][0]["importance"] is None
        assert data["papers"][0]["relevance"] is None

    def test_fallback_external_id(self):
        """If arxiv_id is missing, external_id is used."""
        steps = [
            {
                "module": "FetchArxivPapers",
                "prompt": {},
                "response": {
                    "result": {
                        "papers": [{"external_id": "ext-001", "title": "External Paper"}]
                    },
                    "success": True,
                },
            },
        ]
        data = extract_data_from_steps(steps)
        assert data["papers"][0]["arxiv_id"] == "ext-001"


# ---------------------------------------------------------------------------
# Tests: collapsed state
# ---------------------------------------------------------------------------

class TestStepsCollapsedDefault:
    """Verify that the HTML template renders all steps collapsed."""

    def test_no_open_attribute_on_first_step(self):
        """Regression: index.html must NOT inject 'open' on the first step."""
        with open("static/index.html", "r", encoding="utf-8") as f:
            html = f.read()

        # Find the renderStepsTrace function and check for idx === 0 open
        match = re.search(r'function renderStepsTrace\b.*?\n\s*\}', html, re.DOTALL)
        assert match, "renderStepsTrace function not found in index.html"
        fn_body = match.group(0)
        # Must NOT contain an open attribute conditional on idx
        assert "idx === 0" not in fn_body, (
            "renderStepsTrace still contains idx === 0 conditional — "
            "first step should be collapsed like all others"
        )

    def test_details_element_has_no_open(self):
        """All <details class='step-detail'> should be without 'open'."""
        with open("static/index.html", "r", encoding="utf-8") as f:
            html = f.read()
        # Look for any <details class="step-detail" open>
        assert '<details class="step-detail" open' not in html, (
            "Found a <details> with hardcoded 'open' in step details"
        )


# ---------------------------------------------------------------------------
# Tests: paper card rendering logic
# ---------------------------------------------------------------------------

class TestPaperCardLogic:
    """Verify paper card data preparation edge cases."""

    def test_relevance_tag_classes(self):
        """Score thresholds: >=7 high, >=4 mid, <4 low."""
        def tag_class(score):
            if score >= 7:
                return "high"
            elif score >= 4:
                return "mid"
            else:
                return "low"

        assert tag_class(9) == "high"
        assert tag_class(7) == "high"
        assert tag_class(6) == "mid"
        assert tag_class(4) == "mid"
        assert tag_class(3) == "low"
        assert tag_class(0) == "low"

    def test_all_papers_returned(self, sample_steps):
        """All unique papers are returned (no limit) for grouped rendering."""
        fetch_step = sample_steps[0]
        papers = fetch_step["response"]["result"]["papers"]
        for i in range(3, 8):
            papers.append({"arxiv_id": f"2401.0000{i}", "title": f"Paper {chr(65+i)}", "published": "2024-01-20"})
        fetch_step["response"]["result"]["total_found"] = len(papers)

        data = extract_data_from_steps(sample_steps)
        assert len(data["papers"]) == 7  # 2 original + 5 new unique (no limit)

    def test_dedup_across_multiple_fetch_steps(self):
        """If there are two FetchArxivPapers steps, papers are still deduped."""
        steps = [
            {
                "module": "FetchArxivPapers",
                "prompt": {},
                "response": {
                    "result": {"papers": [
                        {"arxiv_id": "2401.00001", "title": "Paper A"},
                    ]},
                    "success": True,
                },
            },
            {
                "module": "FetchArxivPapers",
                "prompt": {},
                "response": {
                    "result": {"papers": [
                        {"arxiv_id": "2401.00001", "title": "Paper A (dup)"},
                        {"arxiv_id": "2401.00002", "title": "Paper B"},
                    ]},
                    "success": True,
                },
            },
        ]
        data = extract_data_from_steps(steps)
        ids = [p["arxiv_id"] for p in data["papers"]]
        assert ids == ["2401.00001", "2401.00002"]
        # Title should be from first occurrence
        assert data["papers"][0]["title"] == "Paper A"


# ---------------------------------------------------------------------------
# Tests: paper grouping by status
# ---------------------------------------------------------------------------

class TestPaperGrouping:
    """Verify papers are sorted into delivered / new / seen groups."""

    def test_delivered_paper_grouped(self, sample_steps):
        """Paper with a 'deliver' action goes into deliveredPapers."""
        data = extract_data_from_steps(sample_steps)
        delivered_ids = [p["arxiv_id"] for p in data["deliveredPapers"]]
        assert "2401.00001" in delivered_ids

    def test_unseen_paper_grouped(self, sample_steps):
        """Unseen paper that is NOT delivered goes into unseenPapers."""
        data = extract_data_from_steps(sample_steps)
        unseen_ids = [p["arxiv_id"] for p in data["unseenPapers"]]
        # 2401.00002 is unseen and action=skip, so it's new/unseen
        assert "2401.00002" in unseen_ids

    def test_seen_paper_grouped(self):
        """Paper in seen_papers list goes into seenPapers."""
        steps = [
            {
                "module": "FetchArxivPapers",
                "prompt": {},
                "response": {
                    "result": {
                        "papers": [
                            {"arxiv_id": "2401.00001", "title": "Seen Paper"},
                            {"arxiv_id": "2401.00002", "title": "Unseen Paper"},
                        ]
                    },
                    "success": True,
                },
            },
            {
                "module": "CheckSeenPapers",
                "prompt": {},
                "response": {
                    "result": {
                        "seen_papers": [{"arxiv_id": "2401.00001"}],
                        "unseen_papers": [{"arxiv_id": "2401.00002"}],
                        "summary": {"total": 2, "seen": 1, "unseen": 1},
                    },
                    "success": True,
                },
            },
        ]
        data = extract_data_from_steps(steps)
        seen_ids = [p["arxiv_id"] for p in data["seenPapers"]]
        unseen_ids = [p["arxiv_id"] for p in data["unseenPapers"]]
        assert "2401.00001" in seen_ids
        assert "2401.00002" in unseen_ids

    def test_delivered_trumps_unseen(self, sample_steps):
        """A paper that is unseen AND delivered should be in deliveredPapers only."""
        data = extract_data_from_steps(sample_steps)
        # 2401.00001 is in unseen_papers AND delivered
        delivered_ids = [p["arxiv_id"] for p in data["deliveredPapers"]]
        unseen_ids = [p["arxiv_id"] for p in data["unseenPapers"]]
        assert "2401.00001" in delivered_ids
        assert "2401.00001" not in unseen_ids

    def test_status_field_set(self, sample_steps):
        """Each paper has a status field matching its group."""
        data = extract_data_from_steps(sample_steps)
        by_id = {p["arxiv_id"]: p for p in data["papers"]}
        assert by_id["2401.00001"]["status"] == "delivered"
        assert by_id["2401.00002"]["status"] == "new"

    def test_groups_are_exhaustive(self, sample_steps):
        """Every paper appears in exactly one group."""
        data = extract_data_from_steps(sample_steps)
        grouped_ids = (
            [p["arxiv_id"] for p in data["deliveredPapers"]]
            + [p["arxiv_id"] for p in data["unseenPapers"]]
            + [p["arxiv_id"] for p in data["seenPapers"]]
        )
        all_ids = [p["arxiv_id"] for p in data["papers"]]
        assert sorted(grouped_ids) == sorted(all_ids)
