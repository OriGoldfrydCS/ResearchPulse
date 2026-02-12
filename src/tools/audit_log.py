"""
Tool: audit_log - Build and persist structured run audit logs.

This module provides functionality to build and save structured audit logs
at the end of each agent run. The audit log captures:

1. Run metadata and timing
2. All papers retrieved, scored, shared, and discarded
3. Colleague sharing details
4. LLM usage statistics
5. User profile snapshot (for audit purposes)

The audit log is:
- Stored in the database (run_audit_logs table)
- Queryable via API for UI display and analytics
- Non-blocking: failures do not affect the run completion

Feature flag: AUDIT_LOG_ENABLED (defaults to False)
"""

from __future__ import annotations

import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Add parent to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class PaperAuditEntry(BaseModel):
    """Audit entry for a single paper."""
    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    relevance_score: Optional[float] = Field(None, description="Relevance score 0-1")
    novelty_score: Optional[float] = Field(None, description="Heuristic novelty score 0-1")
    llm_novelty_score: Optional[float] = Field(None, description="LLM novelty score 0-100")
    importance: Optional[str] = Field(None, description="Importance level: high/medium/low")
    decision: Optional[str] = Field(None, description="Decision: saved/shared/logged/discarded")
    shared_with: List[str] = Field(default_factory=list, description="Colleague IDs paper was shared with")
    discard_reason: Optional[str] = Field(None, description="Reason for discarding if applicable")


class ColleagueShareAuditEntry(BaseModel):
    """Audit entry for colleague shares."""
    colleague_id: str = Field(..., description="Colleague UUID")
    colleague_name: str = Field(..., description="Colleague name")
    colleague_email: str = Field(..., description="Colleague email")
    papers_shared: List[str] = Field(default_factory=list, description="List of arxiv_ids shared")
    paper_count: int = Field(0, description="Number of papers shared")


class LLMUsageAuditEntry(BaseModel):
    """Audit entry for LLM usage."""
    novelty_calls: int = Field(0, description="Number of LLM novelty scoring calls")
    profile_evolution_calls: int = Field(0, description="Number of profile evolution LLM calls")
    total_tokens: int = Field(0, description="Total tokens used")
    estimated_cost_usd: float = Field(0.0, description="Estimated cost in USD")


class RunAuditLogData(BaseModel):
    """Complete audit log data for a run."""
    # Run metadata
    run_id: str = Field(..., description="Unique run identifier")
    user_id: str = Field(..., description="User UUID")
    timestamp: str = Field(..., description="ISO8601 timestamp")
    execution_time_ms: Optional[int] = Field(None, description="Total execution time in ms")
    stop_reason: Optional[str] = Field(None, description="Why the run stopped")
    
    # User profile snapshot
    user_profile_snapshot: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Snapshot of user profile at run time"
    )
    
    # Paper statistics
    papers_retrieved_count: int = Field(0, description="Total papers retrieved from arXiv")
    papers_scored_count: int = Field(0, description="Papers that were scored")
    papers_shared_count: int = Field(0, description="Papers shared (owner + colleagues)")
    papers_discarded_count: int = Field(0, description="Papers discarded")
    
    # Detailed paper data
    papers: List[PaperAuditEntry] = Field(default_factory=list, description="All papers with scores/decisions")
    papers_shared: List[Dict[str, Any]] = Field(default_factory=list, description="Papers that were shared")
    papers_discarded: List[Dict[str, Any]] = Field(default_factory=list, description="Papers that were discarded")
    
    # Colleague sharing
    colleague_shares: Dict[str, ColleagueShareAuditEntry] = Field(
        default_factory=dict, 
        description="Shares per colleague"
    )
    
    # LLM usage
    llm_usage: LLMUsageAuditEntry = Field(
        default_factory=LLMUsageAuditEntry,
        description="LLM usage statistics"
    )
    
    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# Audit Log Builder
# =============================================================================

class AuditLogBuilder:
    """
    Builds structured audit logs from agent run data.
    
    Usage:
        builder = AuditLogBuilder(run_id, user_id)
        builder.set_user_profile(profile)
        builder.add_fetched_papers(papers)
        for paper in scored_papers:
            builder.add_scored_paper(paper, decision, shared_with)
        builder.set_stop_reason(reason)
        audit_log = builder.build()
    """
    
    def __init__(self, run_id: str, user_id: str):
        self.run_id = run_id
        self.user_id = user_id
        self.start_time = datetime.utcnow()
        
        # Data accumulators
        self._user_profile: Dict[str, Any] = {}
        self._fetched_papers: List[Dict] = []
        self._scored_papers: List[PaperAuditEntry] = []
        self._shared_papers: List[Dict] = []
        self._discarded_papers: List[Dict] = []
        self._colleague_shares: Dict[str, ColleagueShareAuditEntry] = {}
        self._llm_usage = LLMUsageAuditEntry()
        self._stop_reason: Optional[str] = None
        self._end_time: Optional[datetime] = None
    
    def set_user_profile(self, profile: Dict[str, Any]) -> "AuditLogBuilder":
        """Set the user profile snapshot."""
        # Only capture relevant fields for audit
        self._user_profile = {
            "research_topics": profile.get("research_topics", []),
            "arxiv_categories_include": profile.get("arxiv_categories_include", []),
            "arxiv_categories_exclude": profile.get("arxiv_categories_exclude", []),
            "avoid_topics": profile.get("avoid_topics", []),
            "interests_include": profile.get("interests_include", ""),
        }
        return self
    
    def add_fetched_papers(self, papers: List[Dict]) -> "AuditLogBuilder":
        """Record papers fetched from arXiv."""
        self._fetched_papers = papers
        return self
    
    def add_scored_paper(
        self, 
        paper: Dict,
        decision: str = "logged",
        shared_with: Optional[List[str]] = None,
        discard_reason: Optional[str] = None
    ) -> "AuditLogBuilder":
        """Add a scored paper to the audit log."""
        entry = PaperAuditEntry(
            arxiv_id=paper.get("arxiv_id", ""),
            title=paper.get("title", ""),
            relevance_score=paper.get("relevance_score"),
            novelty_score=paper.get("novelty_score"),
            llm_novelty_score=paper.get("llm_novelty_score"),
            importance=paper.get("importance"),
            decision=decision,
            shared_with=shared_with or [],
            discard_reason=discard_reason,
        )
        self._scored_papers.append(entry)
        
        if decision in ("saved", "shared"):
            self._shared_papers.append({
                "arxiv_id": entry.arxiv_id,
                "title": entry.title,
                "shared_with": entry.shared_with,
            })
        elif decision == "discarded" or discard_reason:
            self._discarded_papers.append({
                "arxiv_id": entry.arxiv_id,
                "title": entry.title,
                "reason": discard_reason or "below_threshold",
            })
        
        return self
    
    def add_colleague_share(
        self, 
        colleague_id: str, 
        colleague_name: str,
        colleague_email: str,
        paper_id: str
    ) -> "AuditLogBuilder":
        """Record a paper shared with a colleague."""
        if colleague_id not in self._colleague_shares:
            self._colleague_shares[colleague_id] = ColleagueShareAuditEntry(
                colleague_id=colleague_id,
                colleague_name=colleague_name,
                colleague_email=colleague_email,
                papers_shared=[],
                paper_count=0,
            )
        
        entry = self._colleague_shares[colleague_id]
        if paper_id not in entry.papers_shared:
            entry.papers_shared.append(paper_id)
            entry.paper_count = len(entry.papers_shared)
        
        return self
    
    def add_llm_usage(
        self,
        novelty_calls: int = 0,
        profile_evolution_calls: int = 0,
        tokens: int = 0,
        cost_usd: float = 0.0
    ) -> "AuditLogBuilder":
        """Add LLM usage statistics."""
        self._llm_usage.novelty_calls += novelty_calls
        self._llm_usage.profile_evolution_calls += profile_evolution_calls
        self._llm_usage.total_tokens += tokens
        self._llm_usage.estimated_cost_usd += cost_usd
        return self
    
    def set_stop_reason(self, reason: str) -> "AuditLogBuilder":
        """Set the stop reason."""
        self._stop_reason = reason
        return self
    
    def finalize(self) -> "AuditLogBuilder":
        """Mark the build as complete and record end time."""
        self._end_time = datetime.utcnow()
        return self
    
    def build(self) -> RunAuditLogData:
        """Build the final audit log data."""
        if not self._end_time:
            self._end_time = datetime.utcnow()
        
        # Calculate execution time
        execution_time_ms = int((self._end_time - self.start_time).total_seconds() * 1000)
        
        # Calculate statistics
        papers_shared_count = len(self._shared_papers)
        for entry in self._colleague_shares.values():
            papers_shared_count += entry.paper_count
        
        return RunAuditLogData(
            run_id=self.run_id,
            user_id=self.user_id,
            timestamp=self.start_time.isoformat() + "Z",
            execution_time_ms=execution_time_ms,
            stop_reason=self._stop_reason,
            user_profile_snapshot=self._user_profile,
            papers_retrieved_count=len(self._fetched_papers),
            papers_scored_count=len(self._scored_papers),
            papers_shared_count=papers_shared_count,
            papers_discarded_count=len(self._discarded_papers),
            papers=[p.model_dump() for p in self._scored_papers],
            papers_shared=self._shared_papers,
            papers_discarded=self._discarded_papers,
            colleague_shares={k: v.model_dump() for k, v in self._colleague_shares.items()},
            llm_usage=self._llm_usage,
        )


# =============================================================================
# Database Operations
# =============================================================================

def save_audit_log(
    audit_log: RunAuditLogData,
    user_id_uuid: Optional[uuid.UUID] = None
) -> Dict[str, Any]:
    """
    Save audit log to database.
    
    Args:
        audit_log: The audit log data to save
        user_id_uuid: Optional UUID (if not provided, parsed from audit_log.user_id)
    
    Returns:
        Dict with success status and saved log ID
    """
    try:
        from db.database import is_database_configured, get_db_session
        from db.orm_models import RunAuditLog
        
        if not is_database_configured():
            logger.warning("Database not configured, skipping audit log save")
            return {"success": False, "error": "database_not_configured"}
        
        # Parse user_id if not provided
        if user_id_uuid is None:
            user_id_uuid = uuid.UUID(audit_log.user_id)
        
        with get_db_session() as db:
            # Create the audit log record
            record = RunAuditLog(
                run_id=audit_log.run_id,
                user_id=user_id_uuid,
                created_at=datetime.utcnow(),
                papers_retrieved_count=audit_log.papers_retrieved_count,
                papers_scored_count=audit_log.papers_scored_count,
                papers_shared_count=audit_log.papers_shared_count,
                papers_discarded_count=audit_log.papers_discarded_count,
                papers_retrieved=[{"arxiv_id": p.arxiv_id, "title": p.title} 
                                  for p in (audit_log.papers or [])],
                papers_shared=audit_log.papers_shared,
                papers_discarded=audit_log.papers_discarded,
                colleague_shares={k: v.model_dump() if hasattr(v, 'model_dump') else v 
                                  for k, v in (audit_log.colleague_shares or {}).items()},
                execution_time_ms=audit_log.execution_time_ms,
                llm_calls_count=(audit_log.llm_usage.novelty_calls + 
                                 audit_log.llm_usage.profile_evolution_calls),
                llm_tokens_used=audit_log.llm_usage.total_tokens,
                stop_reason=audit_log.stop_reason,
                user_profile_snapshot=audit_log.user_profile_snapshot,
                full_log=audit_log.model_dump(),
            )
            
            db.add(record)
            db.commit()
            db.refresh(record)
            
            logger.info(f"Saved audit log for run {audit_log.run_id}: {record.id}")
            
            return {
                "success": True,
                "audit_log_id": str(record.id),
                "run_id": audit_log.run_id,
            }
            
    except Exception as e:
        logger.error(f"Failed to save audit log for run {audit_log.run_id}: {e}")
        return {"success": False, "error": str(e)}


def get_audit_logs(
    user_id: uuid.UUID,
    limit: int = 50,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Get audit logs for a user.
    
    Args:
        user_id: User UUID
        limit: Maximum number of logs to return
        offset: Pagination offset
    
    Returns:
        List of audit log dictionaries
    """
    try:
        from db.database import is_database_configured, get_db_session
        from db.orm_models import RunAuditLog, run_audit_log_to_dict
        
        if not is_database_configured():
            return []
        
        with get_db_session() as db:
            logs = db.query(RunAuditLog).filter_by(
                user_id=user_id
            ).order_by(
                RunAuditLog.created_at.desc()
            ).limit(limit).offset(offset).all()
            
            return [run_audit_log_to_dict(log) for log in logs]
            
    except Exception as e:
        logger.error(f"Failed to get audit logs: {e}")
        return []


def get_audit_log_by_run_id(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific audit log by run ID.
    
    Args:
        run_id: The run identifier
    
    Returns:
        Audit log dictionary or None if not found
    """
    try:
        from db.database import is_database_configured, get_db_session
        from db.orm_models import RunAuditLog, run_audit_log_to_dict
        
        if not is_database_configured():
            return None
        
        with get_db_session() as db:
            log = db.query(RunAuditLog).filter_by(run_id=run_id).first()
            if log:
                return run_audit_log_to_dict(log)
            return None
            
    except Exception as e:
        logger.error(f"Failed to get audit log for run {run_id}: {e}")
        return None


# =============================================================================
# Integration Helper
# =============================================================================

def build_audit_log_from_episode(
    run_id: str,
    user_id: str,
    research_profile: Dict[str, Any],
    fetched_papers: List[Dict],
    scored_papers: List[Dict],
    decisions: List[Dict],
    actions: List[Dict],
    colleagues: List[Dict],
    stop_reason: str,
    start_time: Optional[datetime] = None,
) -> RunAuditLogData:
    """
    Build an audit log from episode data.
    
    This is a convenience function for integrating with the existing
    agent workflow. It constructs an audit log from the data accumulated
    during a run.
    
    Args:
        run_id: Unique run identifier
        user_id: User UUID string
        research_profile: User's research profile
        fetched_papers: Papers retrieved from arXiv
        scored_papers: Papers that were scored
        decisions: Decision records
        actions: Action records (including colleague shares)
        colleagues: List of colleagues
        stop_reason: Why the run stopped
        start_time: Optional start time (uses now if not provided)
    
    Returns:
        RunAuditLogData ready for saving
    """
    builder = AuditLogBuilder(run_id, user_id)
    
    if start_time:
        builder.start_time = start_time
    
    # Set profile and papers
    builder.set_user_profile(research_profile)
    builder.add_fetched_papers(fetched_papers)
    
    # Build colleague lookup
    colleague_lookup = {c.get("id"): c for c in colleagues}
    
    # Track which papers went to which colleagues
    paper_to_colleagues: Dict[str, List[str]] = {}
    
    # Process actions to find colleague shares
    for action in actions:
        if action.get("action_type") == "share":
            colleague_id = action.get("colleague_id")
            paper_id = action.get("paper_id")
            if colleague_id and paper_id:
                if paper_id not in paper_to_colleagues:
                    paper_to_colleagues[paper_id] = []
                paper_to_colleagues[paper_id].append(colleague_id)
                
                # Record colleague share
                colleague = colleague_lookup.get(colleague_id, {})
                builder.add_colleague_share(
                    colleague_id=colleague_id,
                    colleague_name=colleague.get("name", "Unknown"),
                    colleague_email=colleague.get("email", ""),
                    paper_id=paper_id,
                )
    
    # Process scored papers
    decision_lookup = {d.get("paper_id"): d for d in decisions}
    
    for paper in scored_papers:
        arxiv_id = paper.get("arxiv_id", "")
        decision_record = decision_lookup.get(arxiv_id, {})
        decision = decision_record.get("decision", "logged")
        shared_with = paper_to_colleagues.get(arxiv_id, [])
        
        # Determine if discarded
        discard_reason = None
        importance = paper.get("importance", "low")
        relevance = paper.get("relevance_score", 0)
        
        if importance == "low" and relevance < 0.3:
            discard_reason = "below_relevance_threshold"
        
        builder.add_scored_paper(
            paper=paper,
            decision=decision,
            shared_with=shared_with,
            discard_reason=discard_reason,
        )
    
    # Add discarded papers from fetched but not scored
    scored_ids = {p.get("arxiv_id") for p in scored_papers}
    for paper in fetched_papers:
        arxiv_id = paper.get("arxiv_id", "")
        if arxiv_id not in scored_ids:
            builder.add_scored_paper(
                paper={
                    "arxiv_id": arxiv_id,
                    "title": paper.get("title", ""),
                },
                decision="discarded",
                discard_reason="already_seen",
            )
    
    builder.set_stop_reason(stop_reason)
    builder.finalize()
    
    return builder.build()
