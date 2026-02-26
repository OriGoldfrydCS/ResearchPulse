"""
Tool: persist_state - Persist paper decisions to papers_state.json.

This tool writes paper decisions to the papers_state.json database, ensuring:
1. Idempotency - same paper won't be duplicated within a run
2. Proper tracking of embedded_in_pinecone status
3. Atomic writes with stats recalculation

**Persistence Logic:**

1. **Single Paper**: Persist one paper decision with all metadata
2. **Batch Persist**: Persist multiple papers efficiently (single write)
3. **Idempotency**: Uses paper_id as unique key - updates if exists, inserts if new
4. **Pinecone Tracking**: Sets embedded_in_pinecone=True when paper is vectorized

**Decision Types:**
- saved: Paper saved for reading
- shared: Paper shared with colleagues
- ignored: Paper explicitly ignored
- logged: Paper logged without action
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel, Field

# Add parent to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Use data_service for DB-first access with fallback to local JSON
from db.data_service import (
    get_paper_by_id,
    get_papers_state,
    get_seen_paper_ids,
    upsert_paper,
    upsert_papers,
)
from db.json_store import save_json, DEFAULT_DATA_DIR


# =============================================================================
# Input/Output Models
# =============================================================================

class PaperDecision(BaseModel):
    """A paper decision to persist."""
    paper_id: str = Field(..., description="arXiv paper ID (e.g., '2501.00123')")
    title: str = Field(..., description="Paper title")
    decision: Literal["saved", "shared", "ignored", "logged"] = Field(
        "logged", description="Decision made for this paper"
    )
    importance: Literal["high", "medium", "low", "very_low"] = Field(
        "low", description="Assessed importance level"
    )
    notes: str = Field("", description="Notes about the paper/decision")
    embedded_in_pinecone: bool = Field(
        False, description="Whether paper has been embedded in vector store"
    )
    # Optional metadata
    abstract: Optional[str] = Field(None, description="Paper abstract (not persisted)")
    link: Optional[str] = Field(None, description="Paper URL (not persisted)")
    authors: Optional[List[str]] = Field(None, description="Authors (not persisted)")
    categories: Optional[List[str]] = Field(None, description="arXiv categories (not persisted)")
    relevance_score: Optional[float] = Field(None, description="Relevance score (for notes)")
    novelty_score: Optional[float] = Field(None, description="Novelty score (for notes)")


class PersistResult(BaseModel):
    """Result of a persist operation."""
    success: bool = Field(..., description="Whether the operation succeeded")
    paper_id: str = Field(..., description="Paper ID that was persisted")
    action: Literal["inserted", "updated", "skipped"] = Field(
        ..., description="Action taken"
    )
    message: str = Field("", description="Human-readable message")


class BatchPersistResult(BaseModel):
    """Result of a batch persist operation."""
    success: bool = Field(..., description="Whether the operation succeeded")
    total_papers: int = Field(..., description="Total papers in request")
    inserted: int = Field(0, description="Papers newly inserted")
    updated: int = Field(0, description="Papers updated")
    skipped: int = Field(0, description="Papers skipped (already identical)")
    errors: List[str] = Field(default_factory=list, description="Error messages if any")
    paper_results: List[PersistResult] = Field(
        default_factory=list, description="Per-paper results"
    )
    new_stats: Dict[str, int] = Field(
        default_factory=dict, description="Updated stats after persist"
    )


class RunPersistTracker:
    """
    Tracks papers persisted within a single run for idempotency.
    
    This ensures that if the same paper is processed multiple times
    within a single agent run, it won't create duplicate entries.
    """
    
    def __init__(self):
        self._persisted_ids: Set[str] = set()
        self._run_id: Optional[str] = None
    
    def start_run(self, run_id: str) -> None:
        """Start tracking for a new run."""
        self._run_id = run_id
        self._persisted_ids.clear()
    
    def mark_persisted(self, paper_id: str) -> None:
        """Mark a paper as persisted in this run."""
        self._persisted_ids.add(paper_id)
    
    def is_persisted_in_run(self, paper_id: str) -> bool:
        """Check if paper was already persisted in this run."""
        return paper_id in self._persisted_ids
    
    def get_persisted_count(self) -> int:
        """Get count of papers persisted in this run."""
        return len(self._persisted_ids)
    
    def reset(self) -> None:
        """Reset tracker for a new run."""
        self._persisted_ids.clear()
        self._run_id = None


# Global run tracker instance
_run_tracker = RunPersistTracker()


def get_run_tracker() -> RunPersistTracker:
    """Get the global run tracker instance."""
    return _run_tracker


def reset_run_tracker() -> None:
    """Reset the global run tracker (for testing or new runs)."""
    _run_tracker.reset()


# =============================================================================
# Helper Functions
# =============================================================================

def _get_timestamp() -> str:
    """Get current timestamp in ISO format with Z suffix."""
    return datetime.utcnow().isoformat() + "Z"


def _build_notes(
    decision: PaperDecision,
    include_scores: bool = True,
) -> str:
    """
    Build notes string from decision metadata.
    
    Combines user-provided notes with optional scoring information.
    """
    parts = []
    
    if decision.notes:
        parts.append(decision.notes)
    
    if include_scores:
        score_parts = []
        if decision.relevance_score is not None:
            score_parts.append(f"relevance={decision.relevance_score:.2f}")
        if decision.novelty_score is not None:
            score_parts.append(f"novelty={decision.novelty_score:.2f}")
        if score_parts:
            parts.append(f"Scores: {', '.join(score_parts)}")
    
    return " | ".join(parts) if parts else ""


def _paper_record_changed(
    existing: Dict[str, Any],
    new_decision: PaperDecision,
) -> bool:
    """
    Check if a paper record would change with new decision.
    
    Used for idempotency - skip updates if nothing changed.
    """
    if existing.get("decision") != new_decision.decision:
        return True
    if existing.get("importance") != new_decision.importance:
        return True
    if existing.get("embedded_in_pinecone") != new_decision.embedded_in_pinecone:
        return True
    # Notes can change, but we don't consider it a significant change
    # for idempotency purposes
    return False


# =============================================================================
# Main Tool Implementation
# =============================================================================

def persist_paper_decision(
    paper_decision: Dict[str, Any],
    data_dir: Path | None = None,
    skip_if_persisted_in_run: bool = True,
    include_scores_in_notes: bool = True,
) -> PersistResult:
    """
    Persist a single paper decision to papers_state.json.
    
    This tool writes a paper decision to the database with:
    - Idempotency: Won't duplicate if paper already exists (updates instead)
    - Run tracking: Optional skip if already persisted in current run
    - Automatic timestamp and stats recalculation
    
    Args:
        paper_decision: Dictionary with paper decision data:
            - paper_id (required): arXiv paper ID
            - title (required): Paper title
            - decision: "saved" | "shared" | "ignored" | "logged" (default: "logged")
            - importance: "high" | "medium" | "low" (default: "low")
            - notes: Optional notes about the decision
            - embedded_in_pinecone: Whether embedded in vector store (default: False)
            - relevance_score: Optional relevance score (for notes)
            - novelty_score: Optional novelty score (for notes)
            
        data_dir: Optional data directory path. Uses default if not provided.
        
        skip_if_persisted_in_run: If True, skip papers already persisted in
            the current run (for idempotency within a run).
            
        include_scores_in_notes: If True, include relevance/novelty scores
            in the notes field.
            
    Returns:
        PersistResult with success status, action taken, and message.
        
    Example:
        >>> result = persist_paper_decision({
        ...     "paper_id": "2501.00123",
        ...     "title": "LLM Scaling Laws",
        ...     "decision": "saved",
        ...     "importance": "high",
        ...     "embedded_in_pinecone": True,
        ...     "notes": "Highly relevant to current research"
        ... })
        >>> print(result.action)  # "inserted" or "updated"
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    
    # Parse and validate input
    decision = PaperDecision(**paper_decision)
    
    # Check run-level idempotency
    if skip_if_persisted_in_run and _run_tracker.is_persisted_in_run(decision.paper_id):
        return PersistResult(
            success=True,
            paper_id=decision.paper_id,
            action="skipped",
            message=f"Paper {decision.paper_id} already persisted in this run"
        )
    
    # Check if paper already exists
    existing = get_paper_by_id(decision.paper_id)
    
    # Determine action
    if existing is None:
        action = "inserted"
    elif _paper_record_changed(existing, decision):
        action = "updated"
    else:
        # Paper exists and hasn't changed meaningfully
        _run_tracker.mark_persisted(decision.paper_id)
        return PersistResult(
            success=True,
            paper_id=decision.paper_id,
            action="skipped",
            message=f"Paper {decision.paper_id} unchanged, skipped"
        )
    
    # Build the paper record
    paper_record = {
        "paper_id": decision.paper_id,
        "title": decision.title,
        "date_seen": existing.get("date_seen") if existing else _get_timestamp(),
        "decision": decision.decision,
        "importance": decision.importance,
        "notes": _build_notes(decision, include_scores_in_notes),
        "embedded_in_pinecone": decision.embedded_in_pinecone,
    }
    # Forward optional paper metadata from the input dict
    for meta_key in ["abstract", "authors", "categories", "link", "published", "updated", "agent_email_decision", "agent_calendar_decision", "relevance_score", "novelty_score"]:
        val = paper_decision.get(meta_key)
        if val is not None:
            paper_record[meta_key] = val
    
    # Persist using upsert (handles atomic write and stats)
    try:
        upsert_paper(paper_record)
        _run_tracker.mark_persisted(decision.paper_id)
        
        return PersistResult(
            success=True,
            paper_id=decision.paper_id,
            action=action,
            message=f"Paper {decision.paper_id} {action} successfully"
        )
    except Exception as e:
        return PersistResult(
            success=False,
            paper_id=decision.paper_id,
            action="skipped",
            message=f"Failed to persist {decision.paper_id}: {str(e)}"
        )


def persist_paper_decisions_batch(
    paper_decisions: List[Dict[str, Any]],
    data_dir: Path | None = None,
    skip_if_persisted_in_run: bool = True,
    include_scores_in_notes: bool = True,
) -> BatchPersistResult:
    """
    Persist multiple paper decisions in a single batch write.
    
    More efficient than calling persist_paper_decision multiple times
    as it only writes to disk once.
    
    Args:
        paper_decisions: List of paper decision dictionaries.
            Each dict should have paper_id, title, decision, importance, etc.
            
        data_dir: Optional data directory path.
        
        skip_if_persisted_in_run: Skip papers already persisted in current run.
        
        include_scores_in_notes: Include scores in notes field.
        
    Returns:
        BatchPersistResult with counts and per-paper results.
        
    Example:
        >>> result = persist_paper_decisions_batch([
        ...     {"paper_id": "2501.00001", "title": "Paper 1", "decision": "saved", "importance": "high"},
        ...     {"paper_id": "2501.00002", "title": "Paper 2", "decision": "logged", "importance": "low"},
        ... ])
        >>> print(f"Inserted: {result.inserted}, Updated: {result.updated}")
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    
    if not paper_decisions:
        return BatchPersistResult(
            success=True,
            total_papers=0,
            inserted=0,
            updated=0,
            skipped=0,
            new_stats={}
        )
    
    # Load current state once
    current_state = get_papers_state(data_dir)
    existing_papers = {p.get("paper_id"): p for p in current_state.get("papers", [])}
    
    # Track results
    paper_results: List[PersistResult] = []
    records_to_upsert: List[Dict[str, Any]] = []
    inserted = 0
    updated = 0
    skipped = 0
    errors: List[str] = []
    
    for paper_dict in paper_decisions:
        try:
            decision = PaperDecision(**paper_dict)
            
            # Run-level idempotency check
            if skip_if_persisted_in_run and _run_tracker.is_persisted_in_run(decision.paper_id):
                paper_results.append(PersistResult(
                    success=True,
                    paper_id=decision.paper_id,
                    action="skipped",
                    message=f"Already persisted in this run"
                ))
                skipped += 1
                continue
            
            existing = existing_papers.get(decision.paper_id)
            
            # Determine action
            if existing is None:
                action = "inserted"
                inserted += 1
            elif _paper_record_changed(existing, decision):
                action = "updated"
                updated += 1
            else:
                action = "skipped"
                skipped += 1
                _run_tracker.mark_persisted(decision.paper_id)
                paper_results.append(PersistResult(
                    success=True,
                    paper_id=decision.paper_id,
                    action="skipped",
                    message="Unchanged"
                ))
                continue
            
            # Build record
            paper_record = {
                "paper_id": decision.paper_id,
                "title": decision.title,
                "date_seen": existing.get("date_seen") if existing else _get_timestamp(),
                "decision": decision.decision,
                "importance": decision.importance,
                "notes": _build_notes(decision, include_scores_in_notes),
                "embedded_in_pinecone": decision.embedded_in_pinecone,
            }
            # Forward optional paper metadata from the input dict
            for meta_key in ["abstract", "authors", "categories", "link", "published", "updated", "agent_email_decision", "agent_calendar_decision", "relevance_score", "novelty_score"]:
                val = paper_dict.get(meta_key)
                if val is not None:
                    paper_record[meta_key] = val
            
            records_to_upsert.append(paper_record)
            _run_tracker.mark_persisted(decision.paper_id)
            
            paper_results.append(PersistResult(
                success=True,
                paper_id=decision.paper_id,
                action=action,
                message=f"{action.capitalize()} successfully"
            ))
            
        except Exception as e:
            errors.append(f"Paper {paper_dict.get('paper_id', 'unknown')}: {str(e)}")
            paper_results.append(PersistResult(
                success=False,
                paper_id=paper_dict.get("paper_id", "unknown"),
                action="skipped",
                message=str(e)
            ))
    
    # Batch upsert if we have records
    new_stats = {}
    if records_to_upsert:
        try:
            updated_state = upsert_papers(records_to_upsert)
            new_stats = updated_state.get("stats", {})
        except Exception as e:
            errors.append(f"Batch write failed: {str(e)}")
            return BatchPersistResult(
                success=False,
                total_papers=len(paper_decisions),
                inserted=0,
                updated=0,
                skipped=len(paper_decisions),
                errors=errors,
                paper_results=paper_results,
                new_stats={}
            )
    else:
        # Get current stats if no updates
        new_stats = current_state.get("stats", {})
    
    return BatchPersistResult(
        success=len(errors) == 0,
        total_papers=len(paper_decisions),
        inserted=inserted,
        updated=updated,
        skipped=skipped,
        errors=errors,
        paper_results=paper_results,
        new_stats=new_stats
    )


def mark_paper_embedded(
    paper_id: str,
    data_dir: Path | None = None,
) -> PersistResult:
    """
    Mark a paper as embedded in Pinecone vector store.
    
    This is a convenience function for updating just the embedded_in_pinecone
    flag without changing other fields.
    
    Args:
        paper_id: arXiv paper ID
        data_dir: Optional data directory path
        
    Returns:
        PersistResult with success status
        
    Example:
        >>> result = mark_paper_embedded("2501.00123")
        >>> print(result.success)  # True
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    
    # Get existing paper
    existing = get_paper_by_id(paper_id)
    
    if existing is None:
        return PersistResult(
            success=False,
            paper_id=paper_id,
            action="skipped",
            message=f"Paper {paper_id} not found in database"
        )
    
    if existing.get("embedded_in_pinecone", False):
        return PersistResult(
            success=True,
            paper_id=paper_id,
            action="skipped",
            message=f"Paper {paper_id} already marked as embedded"
        )
    
    # Update only embedded_in_pinecone
    paper_record = {**existing, "embedded_in_pinecone": True}
    
    try:
        upsert_paper(paper_record)
        return PersistResult(
            success=True,
            paper_id=paper_id,
            action="updated",
            message=f"Paper {paper_id} marked as embedded in Pinecone"
        )
    except Exception as e:
        return PersistResult(
            success=False,
            paper_id=paper_id,
            action="skipped",
            message=f"Failed to update {paper_id}: {str(e)}"
        )


def mark_papers_embedded_batch(
    paper_ids: List[str],
    data_dir: Path | None = None,
) -> BatchPersistResult:
    """
    Mark multiple papers as embedded in Pinecone.
    
    Args:
        paper_ids: List of arXiv paper IDs
        data_dir: Optional data directory path
        
    Returns:
        BatchPersistResult with counts
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    
    if not paper_ids:
        return BatchPersistResult(
            success=True,
            total_papers=0,
            inserted=0,
            updated=0,
            skipped=0,
            new_stats={}
        )
    
    # Load current state
    current_state = get_papers_state(data_dir)
    existing_papers = {p.get("paper_id"): p for p in current_state.get("papers", [])}
    
    paper_results: List[PersistResult] = []
    records_to_update: List[Dict[str, Any]] = []
    updated = 0
    skipped = 0
    errors: List[str] = []
    
    for paper_id in paper_ids:
        existing = existing_papers.get(paper_id)
        
        if existing is None:
            paper_results.append(PersistResult(
                success=False,
                paper_id=paper_id,
                action="skipped",
                message="Not found"
            ))
            errors.append(f"Paper {paper_id} not found")
            skipped += 1
            continue
        
        if existing.get("embedded_in_pinecone", False):
            paper_results.append(PersistResult(
                success=True,
                paper_id=paper_id,
                action="skipped",
                message="Already embedded"
            ))
            skipped += 1
            continue
        
        # Update record
        record = {**existing, "embedded_in_pinecone": True}
        records_to_update.append(record)
        updated += 1
        
        paper_results.append(PersistResult(
            success=True,
            paper_id=paper_id,
            action="updated",
            message="Marked as embedded"
        ))
    
    # Batch update
    new_stats = {}
    if records_to_update:
        try:
            updated_state = upsert_papers(records_to_update)
            new_stats = updated_state.get("stats", {})
        except Exception as e:
            errors.append(f"Batch update failed: {str(e)}")
    else:
        new_stats = current_state.get("stats", {})
    
    return BatchPersistResult(
        success=len([e for e in errors if "Batch" in e]) == 0,
        total_papers=len(paper_ids),
        inserted=0,
        updated=updated,
        skipped=skipped,
        errors=errors,
        paper_results=paper_results,
        new_stats=new_stats
    )


# =============================================================================
# JSON Output Functions (for ReAct agent)
# =============================================================================

def persist_paper_decision_json(
    paper_decision: Dict[str, Any],
    data_dir: Path | None = None,
) -> dict:
    """JSON-serializable version of persist_paper_decision."""
    result = persist_paper_decision(paper_decision, data_dir)
    return result.model_dump()


def persist_paper_decisions_batch_json(
    paper_decisions: List[Dict[str, Any]],
    data_dir: Path | None = None,
) -> dict:
    """JSON-serializable version of persist_paper_decisions_batch."""
    result = persist_paper_decisions_batch(paper_decisions, data_dir)
    return result.model_dump()


# =============================================================================
# LangChain Tool Definition (for ReAct agent)
# =============================================================================

PERSIST_STATE_DESCRIPTION = """
Persist paper decisions to papers_state.json database.

Use this tool to save paper processing decisions after scoring and delivery actions.

Input:
- paper_decision: Dictionary with:
  - paper_id (required): arXiv paper ID (e.g., "2501.00123")
  - title (required): Paper title
  - decision: "saved" | "shared" | "ignored" | "logged" (default: "logged")
  - importance: "high" | "medium" | "low" (default: "low")
  - notes: Optional notes about the decision
  - embedded_in_pinecone: True if paper was embedded in vector store
  - relevance_score: Optional (will be added to notes)
  - novelty_score: Optional (will be added to notes)

Output:
- success: Whether persistence succeeded
- paper_id: The paper ID
- action: "inserted" | "updated" | "skipped"
- message: Human-readable status message

Features:
- Idempotency: Same paper won't be duplicated within a run
- Atomic writes: Uses temp file + rename for safe persistence
- Auto-stats: Automatically recalculates decision stats

Use persist_state after decide_delivery_action to record what action was taken.
"""

PERSIST_STATE_SCHEMA = {
    "name": "persist_state",
    "description": PERSIST_STATE_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "paper_decision": {
                "type": "object",
                "description": "Paper decision to persist",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "arXiv paper ID (e.g., '2501.00123')"
                    },
                    "title": {
                        "type": "string",
                        "description": "Paper title"
                    },
                    "decision": {
                        "type": "string",
                        "enum": ["saved", "shared", "ignored", "logged"],
                        "description": "Decision made for this paper"
                    },
                    "importance": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Importance level"
                    },
                    "notes": {
                        "type": "string",
                        "description": "Notes about the decision"
                    },
                    "embedded_in_pinecone": {
                        "type": "boolean",
                        "description": "Whether embedded in vector store"
                    },
                    "relevance_score": {
                        "type": "number",
                        "description": "Relevance score 0-1 (optional)"
                    },
                    "novelty_score": {
                        "type": "number",
                        "description": "Novelty score 0-1 (optional)"
                    }
                },
                "required": ["paper_id", "title"]
            }
        },
        "required": ["paper_decision"]
    }
}

PERSIST_STATE_BATCH_SCHEMA = {
    "name": "persist_state_batch",
    "description": "Persist multiple paper decisions in one operation. More efficient for batch processing.",
    "parameters": {
        "type": "object",
        "properties": {
            "paper_decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "paper_id": {"type": "string"},
                        "title": {"type": "string"},
                        "decision": {"type": "string", "enum": ["saved", "shared", "ignored", "logged"]},
                        "importance": {"type": "string", "enum": ["high", "medium", "low"]},
                        "notes": {"type": "string"},
                        "embedded_in_pinecone": {"type": "boolean"},
                        "relevance_score": {"type": "number"},
                        "novelty_score": {"type": "number"}
                    },
                    "required": ["paper_id", "title"]
                },
                "description": "List of paper decisions"
            }
        },
        "required": ["paper_decisions"]
    }
}


# =============================================================================
# Self-Check
# =============================================================================

def self_check() -> bool:
    """
    Run self-check tests for persist_state tool.
    
    Returns:
        True if all checks pass, False otherwise.
    """
    import tempfile
    import shutil
    import json
    
    print("=" * 60)
    print("persist_state Self-Check")
    print("=" * 60)
    
    all_passed = True
    
    def check(name: str, condition: bool) -> bool:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        return condition
    
    # Create temp directory for tests
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Initialize test database
        initial_state = {
            "papers": [
                {
                    "paper_id": "existing.001",
                    "title": "Existing Paper",
                    "date_seen": "2026-01-01T00:00:00Z",
                    "decision": "logged",
                    "importance": "low",
                    "notes": "Initial entry",
                    "embedded_in_pinecone": False
                }
            ],
            "last_updated": "2026-01-01T00:00:00Z",
            "total_papers_seen": 1,
            "stats": {"saved": 0, "shared": 0, "ignored": 0, "logged": 1}
        }
        
        (temp_dir / "papers_state.json").write_text(
            json.dumps(initial_state, indent=2),
            encoding="utf-8"
        )
        
        # Reset run tracker for clean tests
        reset_run_tracker()
        
        # Test 1: Insert new paper
        print("\n1. Insert New Paper:")
        try:
            result = persist_paper_decision({
                "paper_id": "new.001",
                "title": "New Paper",
                "decision": "saved",
                "importance": "high",
                "notes": "Test insert"
            }, data_dir=temp_dir)
            
            all_passed &= check("returns PersistResult", isinstance(result, PersistResult))
            all_passed &= check("success is True", result.success)
            all_passed &= check("action is inserted", result.action == "inserted")
            
            # Verify in database
            state = get_papers_state(temp_dir)
            all_passed &= check("total_papers_seen is 2", state["total_papers_seen"] == 2)
            all_passed &= check("stats.saved is 1", state["stats"]["saved"] == 1)
        except Exception as e:
            all_passed &= check(f"insert failed: {e}", False)
        
        # Test 2: Update existing paper
        print("\n2. Update Existing Paper:")
        try:
            # Reset tracker to allow update
            reset_run_tracker()
            
            result = persist_paper_decision({
                "paper_id": "existing.001",
                "title": "Existing Paper",
                "decision": "saved",  # Changed from logged
                "importance": "medium",  # Changed from low
            }, data_dir=temp_dir)
            
            all_passed &= check("success is True", result.success)
            all_passed &= check("action is updated", result.action == "updated")
            
            # Verify update
            paper = get_paper_by_id("existing.001", temp_dir)
            all_passed &= check("decision updated", paper["decision"] == "saved")
            all_passed &= check("importance updated", paper["importance"] == "medium")
            all_passed &= check("date_seen preserved", paper["date_seen"] == "2026-01-01T00:00:00Z")
        except Exception as e:
            all_passed &= check(f"update failed: {e}", False)
        
        # Test 3: Run-level idempotency
        print("\n3. Run-Level Idempotency:")
        try:
            # Don't reset tracker - should skip
            result = persist_paper_decision({
                "paper_id": "existing.001",
                "title": "Existing Paper",
                "decision": "shared",  # Try to change
                "importance": "high",
            }, data_dir=temp_dir)
            
            all_passed &= check("success is True", result.success)
            all_passed &= check("action is skipped", result.action == "skipped")
            
            # Verify no change
            paper = get_paper_by_id("existing.001", temp_dir)
            all_passed &= check("decision unchanged", paper["decision"] == "saved")
        except Exception as e:
            all_passed &= check(f"idempotency failed: {e}", False)
        
        # Test 4: embedded_in_pinecone flag
        print("\n4. Embedded in Pinecone Flag:")
        try:
            reset_run_tracker()
            
            result = persist_paper_decision({
                "paper_id": "embed.001",
                "title": "Embedded Paper",
                "decision": "saved",
                "importance": "high",
                "embedded_in_pinecone": True,
            }, data_dir=temp_dir)
            
            all_passed &= check("success is True", result.success)
            
            paper = get_paper_by_id("embed.001", temp_dir)
            all_passed &= check("embedded_in_pinecone is True", paper["embedded_in_pinecone"] is True)
        except Exception as e:
            all_passed &= check(f"embed flag failed: {e}", False)
        
        # Test 5: mark_paper_embedded function
        print("\n5. Mark Paper Embedded:")
        try:
            reset_run_tracker()
            
            # First add a paper without embedding
            persist_paper_decision({
                "paper_id": "mark.001",
                "title": "Paper to Mark",
                "decision": "saved",
                "importance": "high",
                "embedded_in_pinecone": False,
            }, data_dir=temp_dir)
            
            # Now mark it as embedded
            result = mark_paper_embedded("mark.001", data_dir=temp_dir)
            
            all_passed &= check("success is True", result.success)
            all_passed &= check("action is updated", result.action == "updated")
            
            paper = get_paper_by_id("mark.001", temp_dir)
            all_passed &= check("embedded_in_pinecone now True", paper["embedded_in_pinecone"] is True)
            
            # Try marking again - should skip
            result2 = mark_paper_embedded("mark.001", data_dir=temp_dir)
            all_passed &= check("second mark skipped", result2.action == "skipped")
        except Exception as e:
            all_passed &= check(f"mark embedded failed: {e}", False)
        
        # Test 6: Batch persist
        print("\n6. Batch Persist:")
        try:
            reset_run_tracker()
            
            result = persist_paper_decisions_batch([
                {"paper_id": "batch.001", "title": "Batch 1", "decision": "saved", "importance": "high"},
                {"paper_id": "batch.002", "title": "Batch 2", "decision": "shared", "importance": "medium"},
                {"paper_id": "batch.003", "title": "Batch 3", "decision": "logged", "importance": "low"},
            ], data_dir=temp_dir)
            
            all_passed &= check("success is True", result.success)
            all_passed &= check("total_papers is 3", result.total_papers == 3)
            all_passed &= check("inserted is 3", result.inserted == 3)
            all_passed &= check("has new_stats", len(result.new_stats) > 0)
        except Exception as e:
            all_passed &= check(f"batch persist failed: {e}", False)
        
        # Test 7: Batch mark embedded
        print("\n7. Batch Mark Embedded:")
        try:
            result = mark_papers_embedded_batch(
                ["batch.001", "batch.002"],
                data_dir=temp_dir
            )
            
            all_passed &= check("success is True", result.success)
            all_passed &= check("updated is 2", result.updated == 2)
            
            p1 = get_paper_by_id("batch.001", temp_dir)
            p2 = get_paper_by_id("batch.002", temp_dir)
            all_passed &= check("batch.001 embedded", p1["embedded_in_pinecone"] is True)
            all_passed &= check("batch.002 embedded", p2["embedded_in_pinecone"] is True)
        except Exception as e:
            all_passed &= check(f"batch mark failed: {e}", False)
        
        # Test 8: Scores in notes
        print("\n8. Scores in Notes:")
        try:
            reset_run_tracker()
            
            result = persist_paper_decision({
                "paper_id": "scores.001",
                "title": "Paper with Scores",
                "decision": "saved",
                "importance": "high",
                "notes": "Manual note",
                "relevance_score": 0.85,
                "novelty_score": 0.72,
            }, data_dir=temp_dir)
            
            paper = get_paper_by_id("scores.001", temp_dir)
            all_passed &= check("notes has manual note", "Manual note" in paper["notes"])
            all_passed &= check("notes has relevance", "relevance=0.85" in paper["notes"])
            all_passed &= check("notes has novelty", "novelty=0.72" in paper["notes"])
        except Exception as e:
            all_passed &= check(f"scores in notes failed: {e}", False)
        
        # Test 9: JSON output functions
        print("\n9. JSON Output Functions:")
        try:
            reset_run_tracker()
            
            result = persist_paper_decision_json({
                "paper_id": "json.001",
                "title": "JSON Test",
                "decision": "logged",
                "importance": "low",
            }, data_dir=temp_dir)
            
            all_passed &= check("returns dict", isinstance(result, dict))
            all_passed &= check("has success", "success" in result)
            all_passed &= check("has action", "action" in result)
            
            batch_result = persist_paper_decisions_batch_json([
                {"paper_id": "json.002", "title": "JSON Batch", "decision": "saved", "importance": "high"}
            ], data_dir=temp_dir)
            
            all_passed &= check("batch returns dict", isinstance(batch_result, dict))
            all_passed &= check("batch has total_papers", "total_papers" in batch_result)
        except Exception as e:
            all_passed &= check(f"JSON output failed: {e}", False)
        
        # Test 10: Tool schema
        print("\n10. Tool Schema:")
        all_passed &= check("schema has name", PERSIST_STATE_SCHEMA["name"] == "persist_state")
        all_passed &= check("schema has description", len(PERSIST_STATE_SCHEMA["description"]) > 50)
        all_passed &= check("paper_decision required", 
            "paper_decision" in PERSIST_STATE_SCHEMA["parameters"]["required"])
        all_passed &= check("batch schema exists", PERSIST_STATE_BATCH_SCHEMA["name"] == "persist_state_batch")
        
        # Test 11: Run tracker
        print("\n11. Run Tracker:")
        reset_run_tracker()
        tracker = get_run_tracker()
        all_passed &= check("tracker exists", tracker is not None)
        all_passed &= check("initial count is 0", tracker.get_persisted_count() == 0)
        
        tracker.start_run("test-run-001")
        tracker.mark_persisted("test.001")
        all_passed &= check("marked paper tracked", tracker.is_persisted_in_run("test.001"))
        all_passed &= check("unmarked not tracked", not tracker.is_persisted_in_run("test.002"))
        all_passed &= check("count is 1", tracker.get_persisted_count() == 1)
        
        tracker.reset()
        all_passed &= check("reset clears tracking", tracker.get_persisted_count() == 0)
        
        # Test 12: Skip unchanged papers
        print("\n12. Skip Unchanged Papers:")
        try:
            reset_run_tracker()
            
            # First insert
            persist_paper_decision({
                "paper_id": "skip.001",
                "title": "Skip Test",
                "decision": "logged",
                "importance": "low",
            }, data_dir=temp_dir)
            
            reset_run_tracker()  # Reset to allow second call
            
            # Same values - should skip
            result = persist_paper_decision({
                "paper_id": "skip.001",
                "title": "Skip Test",
                "decision": "logged",
                "importance": "low",
            }, data_dir=temp_dir)
            
            all_passed &= check("unchanged skipped", result.action == "skipped")
        except Exception as e:
            all_passed &= check(f"skip unchanged failed: {e}", False)
            
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        reset_run_tracker()
    
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
