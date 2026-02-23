"""
Tool: generate_report - Generate structured JSON report and markdown summary.

This tool creates a comprehensive run report at the end of an episodic agent run,
including all retrieved papers, decisions, actions, and stop reason.

**Report Contents:**
1. Run metadata (run_id, start_time, end_time, duration)
2. Retrieved papers list with scoring
3. Unseen papers count
4. RAG query count
5. Decisions and actions taken
6. Stop reason
7. Artifact file paths

**Output Formats:**
- Structured JSON: Full data for API/programmatic use
- Markdown summary: Human-readable for GUI display
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# Add parent to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Input/Output Models
# =============================================================================

class PaperSummary(BaseModel):
    """Summary of a single paper in the report."""
    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(default_factory=list, description="Paper authors")
    categories: List[str] = Field(default_factory=list, description="arXiv categories")
    link: Optional[str] = Field(None, description="Paper URL")
    relevance_score: Optional[float] = Field(None, description="Relevance score 0-1")
    novelty_score: Optional[float] = Field(None, description="Novelty score 0-1")
    importance: Optional[Literal["high", "medium", "low"]] = Field(None, description="Importance level")
    decision: Optional[Literal["saved", "shared", "ignored", "logged"]] = Field(None, description="Decision made")
    is_unseen: bool = Field(True, description="Whether paper was unseen before this run")


class ActionSummary(BaseModel):
    """Summary of an action taken during the run."""
    action_type: str = Field(..., description="Type of action (email, calendar, share, etc.)")
    target: str = Field(..., description="Target (researcher, colleague name, etc.)")
    paper_id: str = Field(..., description="Related paper ID")
    paper_title: str = Field(..., description="Related paper title")
    details: Dict[str, Any] = Field(default_factory=dict, description="Action details")


class DecisionSummary(BaseModel):
    """Summary of decisions made for papers."""
    paper_id: str = Field(..., description="arXiv paper ID")
    paper_title: str = Field(..., description="Paper title")
    importance: Literal["high", "medium", "low"] = Field(..., description="Assessed importance")
    decision: Literal["saved", "shared", "ignored", "logged"] = Field(..., description="Decision made")
    actions: List[str] = Field(default_factory=list, description="Actions taken")


class ArtifactInfo(BaseModel):
    """Information about a generated artifact file."""
    file_type: Literal["email", "calendar", "reading_list", "share", "report", "other"] = Field(
        ..., description="Type of artifact"
    )
    file_path: str = Field(..., description="Path to the artifact file")
    description: str = Field("", description="Description of the artifact")
    paper_id: Optional[str] = Field(None, description="Related paper ID if applicable")


class RunStats(BaseModel):
    """Statistics for the run."""
    total_papers_retrieved: int = Field(0, description="Total papers fetched from arXiv")
    total_fetched_count: int = Field(0, description="Alias: total papers fetched from arXiv API")
    unseen_papers_count: int = Field(0, description="Papers not seen before")
    seen_papers_count: int = Field(0, description="Papers already seen")
    papers_delivered: int = Field(0, description="Papers that passed all filters and were delivered")
    papers_filtered_count: int = Field(0, description="Unseen papers filtered out by quality/relevance")
    rag_query_count: int = Field(0, description="Number of RAG queries made")
    papers_scored: int = Field(0, description="Papers that were scored")
    decisions_made: int = Field(0, description="Number of decisions made")
    actions_taken: int = Field(0, description="Total actions taken")
    artifacts_generated: int = Field(0, description="Artifact files generated")
    highest_importance_found: Optional[str] = Field(None, description="Highest importance found")


class GenerateReportInput(BaseModel):
    """Input for generate_report tool."""
    run_id: str = Field(..., description="Unique run identifier")
    start_time: str = Field(..., description="Run start time (ISO format)")
    end_time: Optional[str] = Field(None, description="Run end time (ISO format)")
    stop_reason: str = Field(..., description="Reason the run stopped")
    papers: List[Dict[str, Any]] = Field(default_factory=list, description="List of papers processed")
    decisions: List[Dict[str, Any]] = Field(default_factory=list, description="Decisions made")
    actions: List[Dict[str, Any]] = Field(default_factory=list, description="Actions taken")
    artifacts: List[Dict[str, Any]] = Field(default_factory=list, description="Artifact files generated")
    rag_query_count: int = Field(0, description="Number of RAG queries made")
    unseen_count: int = Field(0, description="Number of unseen papers")
    seen_count: int = Field(0, description="Number of seen papers")
    highest_importance: Optional[str] = Field(None, description="Highest importance found")
    additional_notes: str = Field("", description="Additional notes for the report")


class GenerateReportResult(BaseModel):
    """Result of generate_report tool."""
    success: bool = Field(..., description="Whether report generation succeeded")
    run_id: str = Field(..., description="Run identifier")
    report_json: Dict[str, Any] = Field(default_factory=dict, description="Full structured JSON report")
    markdown_summary: str = Field("", description="Human-readable markdown summary")
    stats: RunStats = Field(default_factory=RunStats, description="Run statistics")
    error: Optional[str] = Field(None, description="Error message if failed")


# =============================================================================
# Helper Functions
# =============================================================================

def _get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


def _parse_timestamp(ts: str) -> datetime:
    """Parse ISO timestamp string to datetime."""
    # Handle various formats
    ts = ts.rstrip("Z")
    for fmt in ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            continue
    return datetime.utcnow()


def _calculate_duration_minutes(start: str, end: str) -> float:
    """Calculate duration in minutes between two timestamps."""
    start_dt = _parse_timestamp(start)
    end_dt = _parse_timestamp(end)
    delta = end_dt - start_dt
    return round(delta.total_seconds() / 60.0, 2)


def _importance_emoji(importance: Optional[str]) -> str:
    """Get emoji for importance level."""
    return {
        "high": "🔴",
        "medium": "🟡",
        "low": "🟢",
    }.get(importance or "", "⚪")


def _decision_emoji(decision: Optional[str]) -> str:
    """Get emoji for decision type."""
    return {
        "saved": "💾",
        "shared": "📤",
        "ignored": "⏭️",
        "logged": "📝",
    }.get(decision or "", "❓")


def _build_paper_summaries(papers: List[Dict[str, Any]]) -> List[PaperSummary]:
    """Build paper summaries from raw paper data."""
    summaries = []
    for paper in papers:
        summary = PaperSummary(
            arxiv_id=paper.get("arxiv_id", paper.get("paper_id", "")),
            title=paper.get("title", "Unknown"),
            authors=paper.get("authors", []),
            categories=paper.get("categories", []),
            link=paper.get("link"),
            relevance_score=paper.get("relevance_score"),
            novelty_score=paper.get("novelty_score"),
            importance=paper.get("importance"),
            decision=paper.get("decision"),
            is_unseen=paper.get("is_unseen", True),
        )
        summaries.append(summary)
    return summaries


def _build_decision_summaries(decisions: List[Dict[str, Any]]) -> List[DecisionSummary]:
    """Build decision summaries from raw decision data."""
    summaries = []
    for dec in decisions:
        summary = DecisionSummary(
            paper_id=dec.get("paper_id", ""),
            paper_title=dec.get("paper_title", dec.get("title", "Unknown")),
            importance=dec.get("importance", "low"),
            decision=dec.get("decision", "logged"),
            actions=dec.get("actions", []),
        )
        summaries.append(summary)
    return summaries


def _build_action_summaries(actions: List[Dict[str, Any]]) -> List[ActionSummary]:
    """Build action summaries from raw action data."""
    summaries = []
    for act in actions:
        summary = ActionSummary(
            action_type=act.get("action_type", "unknown"),
            target=act.get("target", act.get("colleague_name", "researcher")),
            paper_id=act.get("paper_id", ""),
            paper_title=act.get("paper_title", "Unknown"),
            details=act.get("details", {}),
        )
        summaries.append(summary)
    return summaries


def _build_artifact_infos(artifacts: List[Dict[str, Any]]) -> List[ArtifactInfo]:
    """Build artifact infos from raw artifact data."""
    infos = []
    for art in artifacts:
        info = ArtifactInfo(
            file_type=art.get("file_type", "other"),
            file_path=art.get("file_path", ""),
            description=art.get("description", ""),
            paper_id=art.get("paper_id"),
        )
        infos.append(info)
    return infos


def _generate_markdown_summary(
    run_id: str,
    start_time: str,
    end_time: str,
    duration_minutes: float,
    stop_reason: str,
    stats: RunStats,
    papers: List[PaperSummary],
    decisions: List[DecisionSummary],
    actions: List[ActionSummary],
    artifacts: List[ArtifactInfo],
    additional_notes: str = "",
) -> str:
    """Generate a human-readable markdown summary of the run."""
    lines = []
    
    # Header
    lines.extend([
        "# ResearchPulse Run Report",
        "",
        f"**Run ID:** `{run_id}`",
        f"**Started:** {start_time}",
        f"**Ended:** {end_time}",
        f"**Duration:** {duration_minutes:.1f} minutes",
        "",
        "---",
        "",
    ])
    
    # Stop Reason
    lines.extend([
        "## ⏹️ Stop Reason",
        "",
        f"> {stop_reason}",
        "",
    ])
    
    # Stats Summary
    lines.extend([
        "## 📊 Summary Statistics",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Papers Retrieved | {stats.total_papers_retrieved} |",
        f"| Unseen Papers | {stats.unseen_papers_count} |",
        f"| Previously Seen | {stats.seen_papers_count} |",
        f"| RAG Queries | {stats.rag_query_count} |",
        f"| Papers Scored | {stats.papers_scored} |",
        f"| Decisions Made | {stats.decisions_made} |",
        f"| Actions Taken | {stats.actions_taken} |",
        f"| Artifacts Generated | {stats.artifacts_generated} |",
        "",
    ])
    
    if stats.highest_importance_found:
        lines.append(f"**Highest Importance Found:** {_importance_emoji(stats.highest_importance_found)} {stats.highest_importance_found.upper()}")
        lines.append("")
    
    # Papers List
    if papers:
        lines.extend([
            "## 📄 Retrieved Papers",
            "",
        ])
        
        # Group by importance
        high_papers = [p for p in papers if p.importance == "high"]
        medium_papers = [p for p in papers if p.importance == "medium"]
        low_papers = [p for p in papers if p.importance == "low"]
        other_papers = [p for p in papers if p.importance not in ["high", "medium", "low"]]
        
        for importance, paper_list in [
            ("🔴 High Importance", high_papers),
            ("🟡 Medium Importance", medium_papers),
            ("🟢 Low Importance", low_papers),
            ("⚪ Not Scored", other_papers),
        ]:
            if paper_list:
                lines.append(f"### {importance}")
                lines.append("")
                for p in paper_list:
                    status = "🆕" if p.is_unseen else "👀"
                    decision_str = f" {_decision_emoji(p.decision)} {p.decision}" if p.decision else ""
                    scores = ""
                    if p.relevance_score is not None:
                        scores = f" (R:{p.relevance_score:.0%}"
                        if p.novelty_score is not None:
                            scores += f", N:{p.novelty_score:.0%}"
                        scores += ")"
                    
                    title_short = p.title[:60] + "..." if len(p.title) > 60 else p.title
                    link_md = f"[{title_short}]({p.link})" if p.link else title_short
                    lines.append(f"- {status} **{p.arxiv_id}**: {link_md}{scores}{decision_str}")
                lines.append("")
    
    # Decisions
    if decisions:
        lines.extend([
            "## 🎯 Decisions Made",
            "",
            "| Paper | Importance | Decision | Actions |",
            "|-------|------------|----------|---------|",
        ])
        for dec in decisions:
            actions_str = ", ".join(dec.actions) if dec.actions else "-"
            title_short = dec.paper_title[:30] + "..." if len(dec.paper_title) > 30 else dec.paper_title
            lines.append(
                f"| {dec.paper_id}: {title_short} | "
                f"{_importance_emoji(dec.importance)} {dec.importance} | "
                f"{_decision_emoji(dec.decision)} {dec.decision} | "
                f"{actions_str} |"
            )
        lines.append("")
    
    # Actions
    if actions:
        lines.extend([
            "## 🚀 Actions Taken",
            "",
        ])
        
        # Group by action type
        action_groups: Dict[str, List[ActionSummary]] = {}
        for act in actions:
            action_groups.setdefault(act.action_type, []).append(act)
        
        for action_type, action_list in action_groups.items():
            emoji = {"email": "📧", "calendar": "📅", "reading_list": "📚", 
                    "share": "📤", "log": "📝"}.get(action_type, "⚡")
            lines.append(f"### {emoji} {action_type.replace('_', ' ').title()}")
            lines.append("")
            for act in action_list:
                lines.append(f"- **{act.paper_id}**: {act.paper_title[:40]}... → {act.target}")
            lines.append("")
    
    # Artifacts
    if artifacts:
        lines.extend([
            "## 📁 Generated Artifacts",
            "",
        ])
        
        # Group by type
        artifact_groups: Dict[str, List[ArtifactInfo]] = {}
        for art in artifacts:
            artifact_groups.setdefault(art.file_type, []).append(art)
        
        for art_type, art_list in artifact_groups.items():
            lines.append(f"### {art_type.title()} Files ({len(art_list)})")
            lines.append("")
            for art in art_list:
                lines.append(f"- `{art.file_path}`")
                if art.description:
                    lines.append(f"  - {art.description}")
            lines.append("")
    
    # Additional Notes
    if additional_notes:
        lines.extend([
            "## 📝 Notes",
            "",
            additional_notes,
            "",
        ])
    
    # Footer
    lines.extend([
        "---",
        "",
        f"*Report generated at {_get_timestamp()}*",
        "*ResearchPulse - Research Awareness and Sharing Agent*",
    ])
    
    return "\n".join(lines)


# =============================================================================
# Main Tool Implementation
# =============================================================================

def generate_report(
    run_id: str,
    start_time: str,
    stop_reason: str,
    papers: List[Dict[str, Any]] = None,
    decisions: List[Dict[str, Any]] = None,
    actions: List[Dict[str, Any]] = None,
    artifacts: List[Dict[str, Any]] = None,
    rag_query_count: int = 0,
    unseen_count: int = 0,
    seen_count: int = 0,
    highest_importance: Optional[str] = None,
    end_time: Optional[str] = None,
    additional_notes: str = "",
    total_fetched_count: int = 0,
    papers_filtered_count: int = 0,
) -> GenerateReportResult:
    """
    Generate a comprehensive run report with structured JSON and markdown summary.
    
    This tool should be called at the end of an episodic agent run to create
    a final report documenting all papers, decisions, actions, and outcomes.
    
    Args:
        run_id: Unique identifier for the run
        
        start_time: ISO format timestamp when run started
        
        stop_reason: Human-readable reason why the run stopped
            (e.g., "max_papers_checked reached", "agent emitted TERMINATE")
            
        papers: List of papers processed. Each paper dict can include:
            - arxiv_id/paper_id, title, authors, categories, link
            - relevance_score, novelty_score, importance
            - decision, is_unseen
            
        decisions: List of decisions made. Each dict should have:
            - paper_id, paper_title/title, importance, decision, actions
            
        actions: List of actions taken. Each dict should have:
            - action_type, target/colleague_name, paper_id, paper_title, details
            
        artifacts: List of artifact files generated. Each dict should have:
            - file_type (email/calendar/reading_list/share/report/other)
            - file_path, description, paper_id (optional)
            
        rag_query_count: Number of RAG/Pinecone queries made
        
        unseen_count: Number of papers that were unseen before this run
        
        seen_count: Number of papers that were already seen
        
        highest_importance: Highest importance level found ("high"/"medium"/"low")
        
        end_time: ISO format timestamp when run ended (default: now)
        
        additional_notes: Additional notes to include in the report
        
    Returns:
        GenerateReportResult with:
        - success: Whether generation succeeded
        - run_id: The run identifier
        - report_json: Full structured JSON report
        - markdown_summary: Human-readable markdown summary
        - stats: RunStats with counts
        - error: Error message if failed
        
    Example:
        >>> result = generate_report(
        ...     run_id="abc-123",
        ...     start_time="2026-01-08T10:00:00Z",
        ...     stop_reason="max_papers_checked reached",
        ...     papers=[{"arxiv_id": "2501.00001", "title": "...", "importance": "high"}],
        ...     decisions=[{"paper_id": "2501.00001", "decision": "saved", "importance": "high"}],
        ...     rag_query_count=15,
        ...     unseen_count=8,
        ... )
        >>> print(result.markdown_summary)
    """
    papers = papers or []
    decisions = decisions or []
    actions = actions or []
    artifacts = artifacts or []
    end_time = end_time or _get_timestamp()
    
    try:
        # Calculate duration
        duration_minutes = _calculate_duration_minutes(start_time, end_time)
        
        # Build structured summaries
        paper_summaries = _build_paper_summaries(papers)
        decision_summaries = _build_decision_summaries(decisions)
        action_summaries = _build_action_summaries(actions)
        artifact_infos = _build_artifact_infos(artifacts)
        
        # Calculate stats
        # total_papers_retrieved = papers fetched from arXiv API (not just scored/delivered)
        effective_fetched = total_fetched_count if total_fetched_count > 0 else (unseen_count + seen_count)
        stats = RunStats(
            total_papers_retrieved=effective_fetched,
            total_fetched_count=effective_fetched,
            unseen_papers_count=unseen_count,
            seen_papers_count=seen_count,
            papers_delivered=len(papers),
            papers_filtered_count=papers_filtered_count,
            rag_query_count=rag_query_count,
            papers_scored=sum(1 for p in paper_summaries if p.importance is not None),
            decisions_made=len(decisions),
            actions_taken=len(actions),
            artifacts_generated=len(artifacts),
            highest_importance_found=highest_importance,
        )
        
        # Build full JSON report
        report_json = {
            "run_id": run_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration_minutes": duration_minutes,
            "stop_reason": stop_reason,
            "stats": stats.model_dump(),
            "papers": [p.model_dump() for p in paper_summaries],
            "decisions": [d.model_dump() for d in decision_summaries],
            "actions": [a.model_dump() for a in action_summaries],
            "artifacts": [a.model_dump() for a in artifact_infos],
            "additional_notes": additional_notes,
            "generated_at": _get_timestamp(),
        }
        
        # Generate markdown summary
        markdown_summary = _generate_markdown_summary(
            run_id=run_id,
            start_time=start_time,
            end_time=end_time,
            duration_minutes=duration_minutes,
            stop_reason=stop_reason,
            stats=stats,
            papers=paper_summaries,
            decisions=decision_summaries,
            actions=action_summaries,
            artifacts=artifact_infos,
            additional_notes=additional_notes,
        )
        
        return GenerateReportResult(
            success=True,
            run_id=run_id,
            report_json=report_json,
            markdown_summary=markdown_summary,
            stats=stats,
            error=None,
        )
        
    except Exception as e:
        return GenerateReportResult(
            success=False,
            run_id=run_id,
            report_json={},
            markdown_summary="",
            stats=RunStats(),
            error=str(e),
        )


def generate_report_from_input(input_data: Dict[str, Any]) -> GenerateReportResult:
    """
    Generate report from a single input dictionary.
    
    Convenience wrapper for generate_report that accepts all parameters
    in a single dictionary (useful for LangChain tool invocation).
    """
    return generate_report(
        run_id=input_data.get("run_id", "unknown"),
        start_time=input_data.get("start_time", _get_timestamp()),
        stop_reason=input_data.get("stop_reason", "unknown"),
        papers=input_data.get("papers", []),
        decisions=input_data.get("decisions", []),
        actions=input_data.get("actions", []),
        artifacts=input_data.get("artifacts", []),
        rag_query_count=input_data.get("rag_query_count", 0),
        unseen_count=input_data.get("unseen_count", 0),
        seen_count=input_data.get("seen_count", 0),
        highest_importance=input_data.get("highest_importance"),
        end_time=input_data.get("end_time"),
        additional_notes=input_data.get("additional_notes", ""),
    )


def generate_report_json(
    run_id: str,
    start_time: str,
    stop_reason: str,
    **kwargs,
) -> dict:
    """JSON-serializable version of generate_report."""
    result = generate_report(
        run_id=run_id,
        start_time=start_time,
        stop_reason=stop_reason,
        **kwargs,
    )
    return result.model_dump()


# =============================================================================
# LangChain Tool Definition (for ReAct agent)
# =============================================================================

GENERATE_REPORT_DESCRIPTION = """
Generate a comprehensive run report with structured JSON and markdown summary.

Use this tool at the end of an episodic agent run to document:
- All retrieved papers with their scores and decisions
- Summary statistics (unseen count, RAG queries, etc.)
- Actions taken for researcher and colleagues
- Generated artifact file paths
- Stop reason explaining why the run ended

Input:
- run_id (required): Unique run identifier
- start_time (required): ISO timestamp when run started
- stop_reason (required): Why the run stopped
- papers: List of papers processed (with scores/decisions)
- decisions: List of decisions made per paper
- actions: List of actions taken
- artifacts: List of artifact files generated
- rag_query_count: Number of RAG queries made
- unseen_count: Number of unseen papers
- seen_count: Number of previously seen papers
- highest_importance: Highest importance found

Output:
- success: Whether generation succeeded
- report_json: Full structured JSON report
- markdown_summary: Human-readable markdown for GUI
- stats: Summary statistics
- error: Error message if failed

Call this tool after terminate_run to produce the final run report.
"""

GENERATE_REPORT_SCHEMA = {
    "name": "generate_report",
    "description": GENERATE_REPORT_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "Unique run identifier"
            },
            "start_time": {
                "type": "string",
                "description": "ISO format timestamp when run started"
            },
            "stop_reason": {
                "type": "string",
                "description": "Human-readable reason why the run stopped"
            },
            "papers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "arxiv_id": {"type": "string"},
                        "title": {"type": "string"},
                        "authors": {"type": "array", "items": {"type": "string"}},
                        "categories": {"type": "array", "items": {"type": "string"}},
                        "link": {"type": "string"},
                        "relevance_score": {"type": "number"},
                        "novelty_score": {"type": "number"},
                        "importance": {"type": "string", "enum": ["high", "medium", "low"]},
                        "decision": {"type": "string", "enum": ["saved", "shared", "ignored", "logged"]},
                        "is_unseen": {"type": "boolean"}
                    }
                },
                "description": "List of papers processed"
            },
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "paper_id": {"type": "string"},
                        "paper_title": {"type": "string"},
                        "importance": {"type": "string"},
                        "decision": {"type": "string"},
                        "actions": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "description": "List of decisions made"
            },
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action_type": {"type": "string"},
                        "target": {"type": "string"},
                        "paper_id": {"type": "string"},
                        "paper_title": {"type": "string"},
                        "details": {"type": "object"}
                    }
                },
                "description": "List of actions taken"
            },
            "artifacts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file_type": {"type": "string"},
                        "file_path": {"type": "string"},
                        "description": {"type": "string"},
                        "paper_id": {"type": "string"}
                    }
                },
                "description": "List of artifact files generated"
            },
            "rag_query_count": {
                "type": "integer",
                "description": "Number of RAG queries made"
            },
            "unseen_count": {
                "type": "integer",
                "description": "Number of unseen papers"
            },
            "seen_count": {
                "type": "integer",
                "description": "Number of previously seen papers"
            },
            "highest_importance": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Highest importance level found"
            },
            "end_time": {
                "type": "string",
                "description": "ISO timestamp when run ended (default: now)"
            },
            "additional_notes": {
                "type": "string",
                "description": "Additional notes to include"
            }
        },
        "required": ["run_id", "start_time", "stop_reason"]
    }
}


# =============================================================================
# Self-Check
# =============================================================================

def self_check() -> bool:
    """
    Run self-check tests for generate_report tool.
    
    Returns:
        True if all checks pass, False otherwise.
    """
    print("=" * 60)
    print("generate_report Self-Check")
    print("=" * 60)
    
    all_passed = True
    
    def check(name: str, condition: bool) -> bool:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        return condition

    # Test 1: Basic report generation
    print("\n1. Basic Report Generation:")
    try:
        result = generate_report(
            run_id="test-run-001",
            start_time="2026-01-08T10:00:00Z",
            stop_reason="max_papers_checked reached",
        )
        all_passed &= check("returns GenerateReportResult", isinstance(result, GenerateReportResult))
        all_passed &= check("success is True", result.success)
        all_passed &= check("run_id matches", result.run_id == "test-run-001")
        all_passed &= check("report_json not empty", len(result.report_json) > 0)
        all_passed &= check("markdown_summary not empty", len(result.markdown_summary) > 0)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 2: Report with papers
    print("\n2. Report with Papers:")
    try:
        papers = [
            {
                "arxiv_id": "2501.00001",
                "title": "LLM Scaling Laws",
                "authors": ["Alice Smith", "Bob Jones"],
                "categories": ["cs.CL", "cs.LG"],
                "link": "https://arxiv.org/abs/2501.00001",
                "relevance_score": 0.85,
                "novelty_score": 0.72,
                "importance": "high",
                "decision": "saved",
                "is_unseen": True,
            },
            {
                "arxiv_id": "2501.00002",
                "title": "Transformer Optimization",
                "importance": "medium",
                "decision": "logged",
                "is_unseen": True,
            },
            {
                "arxiv_id": "2501.00003",
                "title": "Old Paper",
                "importance": "low",
                "is_unseen": False,
            },
        ]
        result = generate_report(
            run_id="test-run-002",
            start_time="2026-01-08T10:00:00Z",
            stop_reason="completed successfully",
            papers=papers,
            unseen_count=2,
            seen_count=1,
            highest_importance="high",
        )
        all_passed &= check("success is True", result.success)
        all_passed &= check("stats.total_papers_retrieved is 3", result.stats.total_papers_retrieved == 3)
        all_passed &= check("stats.unseen_papers_count is 2", result.stats.unseen_papers_count == 2)
        all_passed &= check("stats.highest_importance_found is high", result.stats.highest_importance_found == "high")
        all_passed &= check("papers in report_json", len(result.report_json.get("papers", [])) == 3)
        all_passed &= check("markdown has high importance section", "High Importance" in result.markdown_summary)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 3: Report with decisions
    print("\n3. Report with Decisions:")
    try:
        decisions = [
            {
                "paper_id": "2501.00001",
                "paper_title": "LLM Scaling Laws",
                "importance": "high",
                "decision": "saved",
                "actions": ["email", "calendar", "reading_list"],
            },
            {
                "paper_id": "2501.00002",
                "paper_title": "Transformer Optimization",
                "importance": "medium",
                "decision": "logged",
                "actions": ["reading_list"],
            },
        ]
        result = generate_report(
            run_id="test-run-003",
            start_time="2026-01-08T10:00:00Z",
            stop_reason="completed successfully",
            decisions=decisions,
        )
        all_passed &= check("stats.decisions_made is 2", result.stats.decisions_made == 2)
        all_passed &= check("decisions in report_json", len(result.report_json.get("decisions", [])) == 2)
        all_passed &= check("markdown has Decisions section", "Decisions Made" in result.markdown_summary)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 4: Report with actions
    print("\n4. Report with Actions:")
    try:
        actions = [
            {
                "action_type": "email",
                "target": "researcher",
                "paper_id": "2501.00001",
                "paper_title": "LLM Paper",
            },
            {
                "action_type": "share",
                "target": "Dr. Wei Chen",
                "paper_id": "2501.00001",
                "paper_title": "LLM Paper",
            },
        ]
        result = generate_report(
            run_id="test-run-004",
            start_time="2026-01-08T10:00:00Z",
            stop_reason="completed successfully",
            actions=actions,
        )
        all_passed &= check("stats.actions_taken is 2", result.stats.actions_taken == 2)
        all_passed &= check("actions in report_json", len(result.report_json.get("actions", [])) == 2)
        all_passed &= check("markdown has Actions section", "Actions Taken" in result.markdown_summary)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 5: Report with artifacts
    print("\n5. Report with Artifacts:")
    try:
        artifacts = [
            {
                "file_type": "email",
                "file_path": "artifacts/emails/email_001.txt",
                "description": "Email notification for paper",
                "paper_id": "2501.00001",
            },
            {
                "file_type": "calendar",
                "file_path": "artifacts/calendar/event_001.ics",
                "description": "Reading calendar event",
            },
            {
                "file_type": "reading_list",
                "file_path": "artifacts/reading_list.md",
                "description": "Reading list entry",
            },
        ]
        result = generate_report(
            run_id="test-run-005",
            start_time="2026-01-08T10:00:00Z",
            stop_reason="completed successfully",
            artifacts=artifacts,
        )
        all_passed &= check("stats.artifacts_generated is 3", result.stats.artifacts_generated == 3)
        all_passed &= check("artifacts in report_json", len(result.report_json.get("artifacts", [])) == 3)
        all_passed &= check("markdown has Artifacts section", "Generated Artifacts" in result.markdown_summary)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 6: RAG query count
    print("\n6. RAG Query Count:")
    try:
        result = generate_report(
            run_id="test-run-006",
            start_time="2026-01-08T10:00:00Z",
            stop_reason="max_rag_queries reached",
            rag_query_count=50,
        )
        all_passed &= check("stats.rag_query_count is 50", result.stats.rag_query_count == 50)
        all_passed &= check("report_json has rag_query_count", result.report_json.get("stats", {}).get("rag_query_count") == 50)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 7: Duration calculation
    print("\n7. Duration Calculation:")
    try:
        result = generate_report(
            run_id="test-run-007",
            start_time="2026-01-08T10:00:00Z",
            end_time="2026-01-08T10:05:30Z",
            stop_reason="completed",
        )
        duration = result.report_json.get("duration_minutes", 0)
        all_passed &= check("duration is ~5.5 minutes", 5.4 < duration < 5.6)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 8: Stop reason in report
    print("\n8. Stop Reason in Report:")
    try:
        result = generate_report(
            run_id="test-run-008",
            start_time="2026-01-08T10:00:00Z",
            stop_reason="no paper exceeds min_importance_to_act",
        )
        all_passed &= check("stop_reason in report_json", result.report_json.get("stop_reason") == "no paper exceeds min_importance_to_act")
        all_passed &= check("stop_reason in markdown", "no paper exceeds min_importance_to_act" in result.markdown_summary)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 9: Additional notes
    print("\n9. Additional Notes:")
    try:
        result = generate_report(
            run_id="test-run-009",
            start_time="2026-01-08T10:00:00Z",
            stop_reason="completed",
            additional_notes="This was a successful demo run.",
        )
        all_passed &= check("notes in report_json", result.report_json.get("additional_notes") == "This was a successful demo run.")
        all_passed &= check("notes in markdown", "successful demo run" in result.markdown_summary)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 10: JSON output function
    print("\n10. JSON Output Function:")
    try:
        result = generate_report_json(
            run_id="test-run-010",
            start_time="2026-01-08T10:00:00Z",
            stop_reason="test",
        )
        all_passed &= check("returns dict", isinstance(result, dict))
        all_passed &= check("has success", "success" in result)
        all_passed &= check("has report_json", "report_json" in result)
        all_passed &= check("has markdown_summary", "markdown_summary" in result)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 11: Input from dict
    print("\n11. Generate from Input Dict:")
    try:
        input_data = {
            "run_id": "test-run-011",
            "start_time": "2026-01-08T10:00:00Z",
            "stop_reason": "completed",
            "papers": [{"arxiv_id": "2501.00001", "title": "Test"}],
            "rag_query_count": 10,
        }
        result = generate_report_from_input(input_data)
        all_passed &= check("success is True", result.success)
        all_passed &= check("has papers", len(result.report_json.get("papers", [])) == 1)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 12: Tool schema
    print("\n12. Tool Schema:")
    all_passed &= check("schema has name", GENERATE_REPORT_SCHEMA["name"] == "generate_report")
    all_passed &= check("schema has description", len(GENERATE_REPORT_SCHEMA["description"]) > 100)
    all_passed &= check("run_id required", "run_id" in GENERATE_REPORT_SCHEMA["parameters"]["required"])
    all_passed &= check("start_time required", "start_time" in GENERATE_REPORT_SCHEMA["parameters"]["required"])
    all_passed &= check("stop_reason required", "stop_reason" in GENERATE_REPORT_SCHEMA["parameters"]["required"])

    # Test 13: Markdown formatting
    print("\n13. Markdown Formatting:")
    try:
        result = generate_report(
            run_id="test-run-013",
            start_time="2026-01-08T10:00:00Z",
            stop_reason="test",
            papers=[{"arxiv_id": "2501.00001", "title": "Test Paper", "importance": "high", "is_unseen": True}],
        )
        md = result.markdown_summary
        all_passed &= check("has header", "# ResearchPulse Run Report" in md)
        all_passed &= check("has Run ID", "Run ID" in md)
        all_passed &= check("has table", "|" in md)
        all_passed &= check("has emoji", "📊" in md or "🔴" in md)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 14: Error handling
    print("\n14. Error Handling:")
    try:
        # This should not crash even with minimal input
        result = generate_report(
            run_id="",
            start_time="invalid-time",
            stop_reason="",
        )
        all_passed &= check("handles edge cases", result is not None)
    except Exception as e:
        all_passed &= check(f"should handle gracefully: {e}", False)

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("All checks PASSED!")
    else:
        print("Some checks FAILED!")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = self_check()
    sys.exit(0 if success else 1)
