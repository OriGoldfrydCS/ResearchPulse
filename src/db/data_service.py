"""
Data Service - Unified data access layer.

This module provides a single interface for all data operations:
- DB (Supabase/PostgreSQL) is the primary source (production)
- Local JSON files are fallback only (development/offline)

All data from data/*.json files are migrated to and served from the database.
"""

from __future__ import annotations

import json
import logging
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import uuid

from .database import is_database_configured, get_db_session
from .orm_models import (
    User, Paper, PaperView, Colleague, Run, DeliveryPolicy,
    ArxivCategoryDB, PromptRequest, SavedPrompt, PromptTemplate, user_to_dict, colleague_to_dict, paper_view_to_dict,
    policy_to_dict, arxiv_category_to_dict, prompt_request_to_dict
)
from .json_store import (
    DEFAULT_DATA_DIR, load_json, save_json,
    JSONStoreError
)

logger = logging.getLogger(__name__)


def _get_calendar_invite_sender():
    """Load calendar_invite_sender module using importlib to avoid relative import issues."""
    import sys
    # Add tools directory to path first
    tools_path = str(Path(__file__).parent.parent / "tools")
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    
    module_path = Path(__file__).parent.parent / "tools" / "calendar_invite_sender.py"
    spec = importlib.util.spec_from_file_location("calendar_invite_sender", module_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise ImportError(f"Could not load calendar_invite_sender from {module_path}")

# =============================================================================
# Configuration
# =============================================================================

# Default user ID for single-user mode (created on first run)
DEFAULT_USER_EMAIL = "default@researchpulse.local"


def _get_data_dir() -> Path:
    """Get the data directory path."""
    return DEFAULT_DATA_DIR


# =============================================================================
# Database Availability
# =============================================================================

def is_db_available() -> bool:
    """Check if database is configured and available."""
    try:
        return is_database_configured()
    except Exception:
        return False


# =============================================================================
# User Management
# =============================================================================

def get_or_create_default_user() -> Optional[Dict[str, Any]]:
    """
    Get or create the default user for single-user mode.
    
    Returns:
        User dictionary or None if DB not available.
    """
    if not is_db_available():
        return None
    
    try:
        with get_db_session() as db:
            # First check if any user exists (single-user mode)
            user = db.query(User).first()
            
            if not user:
                # Create default user from local profile if exists
                profile = _load_local_research_profile()
                email = profile.get("researcher_email", DEFAULT_USER_EMAIL)
                
                # Double-check the email doesn't exist
                existing = db.query(User).filter_by(email=email).first()
                if existing:
                    return user_to_dict(existing)
                
                user = User(
                    name=profile.get("researcher_name", "Researcher"),
                    email=email,
                    affiliation=profile.get("affiliation"),
                    research_topics=profile.get("research_topics", []),
                    my_papers=profile.get("my_papers", []),
                    preferred_venues=profile.get("preferred_venues", []),
                    avoid_topics=profile.get("avoid_topics", []),
                    time_budget_per_week_minutes=profile.get("time_budget_per_week_minutes", 120),
                    arxiv_categories_include=profile.get("arxiv_categories_include", []),
                    arxiv_categories_exclude=profile.get("arxiv_categories_exclude", []),
                    stop_policy=profile.get("stop_policy", {}),
                )
                db.add(user)
                db.commit()
                db.refresh(user)
                logger.info(f"Created default user: {user.email}")
            
            return user_to_dict(user)
    except Exception as e:
        logger.error(f"Error getting/creating default user: {e}")
        return None


def _get_default_user_id() -> Optional[str]:
    """Get the default user's UUID."""
    user = get_or_create_default_user()
    return user.get("id") if user else None


# =============================================================================
# Research Profile
# =============================================================================

def _load_local_research_profile() -> Dict[str, Any]:
    """Load research profile from local JSON file."""
    try:
        return load_json(_get_data_dir() / "research_profile.json")
    except Exception:
        return {}


def get_research_profile() -> Dict[str, Any]:
    """
    Get the research profile.
    
    Priority: Database > Local JSON file
    
    Returns:
        Research profile dictionary.
    """
    if is_db_available():
        user = get_or_create_default_user()
        if user:
            # Convert User to research profile format
            # If avoid_topics list is empty but interests_exclude has free text,
            # parse it into avoid_topics so keyword-level exclusion works.
            avoid_topics = user.get("avoid_topics", [])
            interests_exclude_text = user.get("interests_exclude", "")
            if not avoid_topics and interests_exclude_text:
                avoid_topics = [
                    t.strip() for t in interests_exclude_text.split(",") if t.strip()
                ]

            return {
                "user_id": user.get("id"),  # Include user_id for autonomous components
                "researcher_name": user.get("name"),
                "email": user.get("email"),  # Include email for notifications
                "affiliation": user.get("affiliation"),
                "research_topics": user.get("research_topics", []),
                "my_papers": user.get("my_papers", []),
                "preferred_venues": user.get("preferred_venues", []),
                "avoid_topics": avoid_topics,
                "time_budget_per_week_minutes": user.get("time_budget_per_week_minutes", 120),
                "arxiv_categories_include": user.get("arxiv_categories_include", []),
                "arxiv_categories_exclude": user.get("arxiv_categories_exclude", []),
                "interests_include": user.get("interests_include", ""),
                "interests_exclude": user.get("interests_exclude", ""),
                "keywords_include": user.get("keywords_include", []),
                "keywords_exclude": user.get("keywords_exclude", []),
                "preferred_time_period": user.get("preferred_time_period", "last two weeks"),
                "stop_policy": user.get("stop_policy", {}),
                # Additional fields from local file
                **{k: v for k, v in _load_local_research_profile().items() 
                   if k not in ["researcher_name", "email", "affiliation", "research_topics", 
                                "my_papers", "preferred_venues", "avoid_topics",
                                "time_budget_per_week_minutes", "arxiv_categories_include",
                                "arxiv_categories_exclude", "interests_include", "interests_exclude",
                                "keywords_include", "keywords_exclude", "preferred_time_period", "stop_policy"]}
            }
    
    # Fallback to local file
    return _load_local_research_profile()


def update_research_profile(updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update the research profile.
    
    Args:
        updates: Dictionary of fields to update.
        
    Returns:
        Updated research profile.
    """
    if is_db_available():
        try:
            with get_db_session() as db:
                user = db.query(User).filter_by(email=DEFAULT_USER_EMAIL).first()
                if user:
                    for key, value in updates.items():
                        if hasattr(user, key):
                            setattr(user, key, value)
                        elif key == "researcher_name":
                            user.name = value
                    db.commit()
                    logger.info("Updated research profile in database")
        except Exception as e:
            logger.error(f"Error updating research profile in DB: {e}")
    
    # Also update local file for development
    try:
        profile = _load_local_research_profile()
        profile.update(updates)
        save_json(_get_data_dir() / "research_profile.json", profile)
    except Exception as e:
        logger.warning(f"Could not update local profile file: {e}")
    
    return get_research_profile()


def get_research_topics() -> List[str]:
    """Get the list of research topics."""
    profile = get_research_profile()
    return profile.get("research_topics", [])


def get_arxiv_categories() -> tuple[List[str], List[str]]:
    """Get arXiv category include/exclude lists."""
    profile = get_research_profile()
    return (
        profile.get("arxiv_categories_include", []),
        profile.get("arxiv_categories_exclude", [])
    )


def get_stop_policy() -> Dict[str, Any]:
    """Get the stop policy from the research profile."""
    profile = get_research_profile()
    return profile.get("stop_policy", {})


# =============================================================================
# Colleagues
# =============================================================================

def _load_local_colleagues() -> List[Dict[str, Any]]:
    """Load colleagues from local JSON file."""
    try:
        data = load_json(_get_data_dir() / "colleagues.json")
        return data.get("colleagues", [])
    except Exception:
        return []


def get_colleagues() -> List[Dict[str, Any]]:
    """
    Get all colleagues.
    
    Priority: Database > Local JSON file
    
    Returns:
        List of colleague dictionaries.
    """
    if is_db_available():
        try:
            user_id = _get_default_user_id()
            if user_id:
                with get_db_session() as db:
                    colleagues = db.query(Colleague).filter_by(
                        user_id=uuid.UUID(user_id)
                    ).all()
                    if colleagues:
                        return [colleague_to_dict(c) for c in colleagues]
        except Exception as e:
            logger.error(f"Error loading colleagues from DB: {e}")
    
    # Fallback to local file
    return _load_local_colleagues()


def get_colleague_by_id(colleague_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific colleague by ID."""
    colleagues = get_colleagues()
    for colleague in colleagues:
        if colleague.get("id") == colleague_id:
            return colleague
    return None


def get_colleagues_for_topics(topics: List[str]) -> List[Dict[str, Any]]:
    """Find colleagues interested in any of the given topics."""
    colleagues = get_colleagues()
    topics_lower = {t.lower() for t in topics}
    
    matching = []
    for colleague in colleagues:
        colleague_topics = {t.lower() for t in colleague.get("topics", [])}
        if colleague_topics & topics_lower:
            matching.append(colleague)
    
    return matching


def save_colleagues(colleagues_list: List[Dict[str, Any]]) -> None:
    """
    Save all colleagues.
    
    Args:
        colleagues_list: List of colleague dictionaries.
    """
    if is_db_available():
        try:
            user_id = _get_default_user_id()
            if user_id:
                with get_db_session() as db:
                    # Upsert colleagues
                    for col_data in colleagues_list:
                        existing = db.query(Colleague).filter_by(
                            user_id=uuid.UUID(user_id),
                            email=col_data.get("email")
                        ).first()
                        
                        if existing:
                            existing.name = col_data.get("name", existing.name)
                            existing.affiliation = col_data.get("affiliation")
                            existing.topics = col_data.get("topics", [])
                            existing.keywords = col_data.get("keywords", [])
                            existing.categories = col_data.get("arxiv_categories_interest", [])
                            existing.sharing_preference = col_data.get("sharing_preference", "weekly")
                            existing.notes = col_data.get("notes")
                        else:
                            new_colleague = Colleague(
                                user_id=uuid.UUID(user_id),
                                name=col_data.get("name"),
                                email=col_data.get("email"),
                                affiliation=col_data.get("affiliation"),
                                topics=col_data.get("topics", []),
                                keywords=col_data.get("keywords", []),
                                categories=col_data.get("arxiv_categories_interest", []),
                                sharing_preference=col_data.get("sharing_preference", "weekly"),
                                notes=col_data.get("notes"),
                            )
                            db.add(new_colleague)
                    db.commit()
                    logger.info(f"Saved {len(colleagues_list)} colleagues to database")
        except Exception as e:
            logger.error(f"Error saving colleagues to DB: {e}")


# =============================================================================
# Delivery Policy
# =============================================================================

def _get_default_delivery_policy() -> Dict[str, Any]:
    """
    Return a sensible default delivery policy when none is configured.
    
    This ensures emails, calendar invites, and reading list entries are
    generated for high/medium importance papers even without explicit config.
    """
    return {
        "importance_policies": {
            "high": {
                "notify_researcher": True,
                "send_email": True,
                "create_calendar_entry": True,
                "add_to_reading_list": True,
                "allow_colleague_sharing": True,
                "priority_label": "urgent"
            },
            "medium": {
                "notify_researcher": True,
                "send_email": True,
                "create_calendar_entry": False,
                "add_to_reading_list": True,
                "allow_colleague_sharing": True,
                "priority_label": "normal"
            },
            "low": {
                "notify_researcher": False,
                "send_email": False,
                "create_calendar_entry": False,
                "add_to_reading_list": False,
                "allow_colleague_sharing": False,
                "priority_label": "low"
            }
        },
        "email_settings": {
            "enabled": True,
            "simulate_output": True,
            "include_abstract": True,
            "include_relevance_explanation": True,
            "digest_mode": False,
        },
        "calendar_settings": {
            "enabled": True,
            "simulate_output": True,
            "event_duration_minutes": 30,
            "default_reminder_minutes": 60,
            "schedule_within_days": 7,
        },
        "reading_list_settings": {
            "enabled": True,
            "include_link": True,
            "include_importance": True,
        },
        "colleague_sharing_settings": {
            "enabled": True,
            "simulate_output": True,
            "respect_sharing_preferences": True,
        }
    }


def _load_local_delivery_policy() -> Dict[str, Any]:
    """Load delivery policy from local JSON file."""
    try:
        return load_json(_get_data_dir() / "delivery_policy.json")
    except Exception:
        return {}


def get_delivery_policy() -> Dict[str, Any]:
    """
    Get the delivery policy configuration.
    
    Priority: Database > Local JSON file > Default Policy
    
    Returns:
        Delivery policy dictionary.
    """
    if is_db_available():
        try:
            user_id = _get_default_user_id()
            if user_id:
                with get_db_session() as db:
                    policy = db.query(DeliveryPolicy).filter_by(
                        user_id=uuid.UUID(user_id)
                    ).first()
                    if policy and policy.policy_json:
                        return policy.policy_json
        except Exception as e:
            logger.error(f"Error loading delivery policy from DB: {e}")
    
    # Fallback to local file
    local_policy = _load_local_delivery_policy()
    if local_policy:
        return local_policy
    
    # Return sensible defaults if no policy is configured
    logger.info("No delivery policy configured, using default policy")
    return _get_default_delivery_policy()


def get_policy_for_importance(importance: str) -> Dict[str, Any]:
    """Get the delivery policy for a specific importance level."""
    policy = get_delivery_policy()
    importance_policies = policy.get("importance_policies", {})
    return importance_policies.get(importance, importance_policies.get("log_only", {}))


def save_delivery_policy(policy_data: Dict[str, Any]) -> None:
    """
    Save the delivery policy.
    
    Args:
        policy_data: Complete policy dictionary.
    """
    if is_db_available():
        try:
            user_id = _get_default_user_id()
            if user_id:
                with get_db_session() as db:
                    existing = db.query(DeliveryPolicy).filter_by(
                        user_id=uuid.UUID(user_id)
                    ).first()
                    
                    if existing:
                        existing.policy_json = policy_data
                    else:
                        new_policy = DeliveryPolicy(
                            user_id=uuid.UUID(user_id),
                            policy_json=policy_data,
                        )
                        db.add(new_policy)
                    db.commit()
                    logger.info("Saved delivery policy to database")
        except Exception as e:
            logger.error(f"Error saving delivery policy to DB: {e}")


# =============================================================================
# Prompt Request Storage (System Controller Compliance)
# =============================================================================

def save_prompt_request(
    raw_prompt: str,
    parsed_data: Dict[str, Any],
    run_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Save a parsed prompt request to the database.
    
    This stores all user research prompts for:
    - Audit trail of all research requests
    - Template usage analytics
    - Output enforcement compliance tracking
    
    Args:
        raw_prompt: The original user prompt text
        parsed_data: Dictionary from PromptController.parse_prompt() containing:
            - template: Template enum name (e.g., "TOP_K_PAPERS")
            - topic: Extracted topic
            - venue: Extracted venue (optional)
            - time_period: Raw time period string (optional)
            - time_days: Converted days (optional)
            - requested_count: User-specified K (optional)
            - output_count: Final output count after defaults
            - retrieval_count: Internal retrieval limit
            - method_or_approach: Method focus (optional)
            - application_domain: Application domain (optional)
            - is_survey_request: Boolean
            - is_trends_request: Boolean
            - is_structured_output: Boolean
        run_id: Agent run ID if part of a workflow (optional)
        
    Returns:
        Saved prompt request as dictionary, or None if failed
    """
    if not is_db_available():
        logger.warning("Database not available, cannot save prompt request")
        return None
    
    try:
        user_id = _get_default_user_id()
        if not user_id:
            logger.error("No user ID available for prompt request")
            return None
        
        with get_db_session() as db:
            prompt_req = PromptRequest(
                user_id=uuid.UUID(user_id),
                run_id=run_id,
                raw_prompt=raw_prompt,
                template=parsed_data.get("template", "UNRECOGNIZED"),
                topic=parsed_data.get("topic"),
                venue=parsed_data.get("venue"),
                time_period=parsed_data.get("time_period"),
                time_days=parsed_data.get("time_days"),
                requested_count=parsed_data.get("requested_count"),
                output_count=parsed_data.get("output_count"),
                retrieval_count=parsed_data.get("retrieval_count"),
                method_or_approach=parsed_data.get("method_or_approach"),
                application_domain=parsed_data.get("application_domain"),
                is_survey_request=parsed_data.get("is_survey_request", False),
                is_trends_request=parsed_data.get("is_trends_request", False),
                is_structured_output=parsed_data.get("is_structured_output", False),
                compliance_status="pending",
            )
            db.add(prompt_req)
            db.commit()
            db.refresh(prompt_req)
            
            logger.info(f"Saved prompt request {prompt_req.id} (template: {prompt_req.template})")
            return prompt_request_to_dict(prompt_req)
            
    except Exception as e:
        logger.error(f"Error saving prompt request: {e}")
        return None


def update_prompt_compliance(
    prompt_id: str,
    papers_retrieved: int,
    papers_returned: int,
    output_enforced: bool = False,
    output_insufficient: bool = False,
    compliance_message: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Update the compliance status of a prompt request after output enforcement.
    
    This records whether the output enforcement succeeded and provides
    an audit trail for compliance verification.
    
    Args:
        prompt_id: UUID of the prompt request
        papers_retrieved: Number of papers retrieved internally
        papers_returned: Number of papers returned to user
        output_enforced: Whether output was truncated
        output_insufficient: Whether fewer papers than requested
        compliance_message: Any compliance notes
        
    Returns:
        Updated prompt request as dictionary, or None if failed
    """
    if not is_db_available():
        return None
    
    try:
        with get_db_session() as db:
            prompt_req = db.query(PromptRequest).filter_by(
                id=uuid.UUID(prompt_id)
            ).first()
            
            if not prompt_req:
                logger.error(f"Prompt request {prompt_id} not found")
                return None
            
            prompt_req.papers_retrieved = papers_retrieved
            prompt_req.papers_returned = papers_returned
            prompt_req.output_enforced = output_enforced
            prompt_req.output_insufficient = output_insufficient
            prompt_req.compliance_message = compliance_message
            
            # Determine compliance status
            requested = prompt_req.requested_count or prompt_req.output_count
            if requested and papers_returned > requested:
                prompt_req.compliance_status = "violation"
                prompt_req.compliance_message = (
                    f"CRITICAL: Returned {papers_returned} papers but user requested {requested}"
                )
            elif output_insufficient:
                prompt_req.compliance_status = "compliant_partial"
            else:
                prompt_req.compliance_status = "compliant"
            
            db.commit()
            db.refresh(prompt_req)
            
            logger.info(f"Updated prompt compliance: {prompt_req.compliance_status}")
            return prompt_request_to_dict(prompt_req)
            
    except Exception as e:
        logger.error(f"Error updating prompt compliance: {e}")
        return None


def get_prompt_requests(
    limit: int = 50,
    template_filter: Optional[str] = None,
    compliance_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get prompt requests with optional filtering.
    
    Args:
        limit: Maximum number of requests to return
        template_filter: Filter by template type (e.g., "TOP_K_PAPERS")
        compliance_filter: Filter by compliance status
        
    Returns:
        List of prompt request dictionaries
    """
    if not is_db_available():
        return []
    
    try:
        user_id = _get_default_user_id()
        if not user_id:
            return []
        
        with get_db_session() as db:
            query = db.query(PromptRequest).filter_by(
                user_id=uuid.UUID(user_id)
            )
            
            if template_filter:
                query = query.filter(PromptRequest.template == template_filter)
            
            if compliance_filter:
                query = query.filter(PromptRequest.compliance_status == compliance_filter)
            
            requests = query.order_by(
                PromptRequest.created_at.desc()
            ).limit(limit).all()
            
            return [prompt_request_to_dict(r) for r in requests]
            
    except Exception as e:
        logger.error(f"Error getting prompt requests: {e}")
        return []


def get_prompt_request_by_run(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the prompt request associated with a specific run.
    
    Args:
        run_id: The agent run ID
        
    Returns:
        Prompt request dictionary or None
    """
    if not is_db_available():
        return None
    
    try:
        with get_db_session() as db:
            prompt_req = db.query(PromptRequest).filter_by(
                run_id=run_id
            ).first()
            
            if prompt_req:
                return prompt_request_to_dict(prompt_req)
            return None
            
    except Exception as e:
        logger.error(f"Error getting prompt request by run: {e}")
        return None


def get_prompt_compliance_summary() -> Dict[str, Any]:
    """
    Get a summary of prompt compliance statistics.
    
    Returns:
        Dictionary with compliance statistics
    """
    if not is_db_available():
        return {"error": "Database not available"}
    
    try:
        user_id = _get_default_user_id()
        if not user_id:
            return {"error": "No user found"}
        
        with get_db_session() as db:
            from sqlalchemy import func
            
            requests = db.query(PromptRequest).filter_by(
                user_id=uuid.UUID(user_id)
            ).all()
            
            if not requests:
                return {
                    "total_requests": 0,
                    "compliance_breakdown": {},
                    "template_usage": {},
                }
            
            # Count by compliance status
            compliance_counts = {}
            template_counts = {}
            
            for req in requests:
                status = req.compliance_status or "pending"
                compliance_counts[status] = compliance_counts.get(status, 0) + 1
                
                template = req.template or "UNRECOGNIZED"
                template_counts[template] = template_counts.get(template, 0) + 1
            
            return {
                "total_requests": len(requests),
                "compliance_breakdown": compliance_counts,
                "template_usage": template_counts,
                "violation_rate": compliance_counts.get("violation", 0) / len(requests) * 100,
            }
            
    except Exception as e:
        logger.error(f"Error getting compliance summary: {e}")
        return {"error": str(e)}


# =============================================================================
# Papers State
# =============================================================================

def _load_local_papers_state() -> Dict[str, Any]:
    """Load papers state from local JSON file."""
    try:
        return load_json(_get_data_dir() / "papers_state.json")
    except Exception:
        return {"papers": [], "stats": {}}


def get_papers_state() -> Dict[str, Any]:
    """
    Get the papers state (all paper views for the user).
    
    Priority: Database > Local JSON file
    
    Returns:
        Papers state dictionary with papers list and stats.
    """
    if is_db_available():
        try:
            user_id = _get_default_user_id()
            if user_id:
                with get_db_session() as db:
                    views = db.query(PaperView).filter_by(
                        user_id=uuid.UUID(user_id)
                    ).order_by(PaperView.last_seen_at.desc()).all()
                    
                    if views:
                        papers = []
                        stats = {"saved": 0, "shared": 0, "ignored": 0, "logged": 0}
                        
                        for view in views:
                            paper_record = {
                                "paper_id": view.paper.external_id if view.paper else None,
                                "title": view.paper.title if view.paper else None,
                                "date_seen": view.first_seen_at.isoformat() + "Z" if view.first_seen_at else None,
                                "decision": view.decision,
                                "importance": view.importance,
                                "notes": view.notes,
                                "embedded_in_pinecone": view.embedded_in_pinecone,
                                "relevance_score": view.relevance_score,
                                "novelty_score": view.novelty_score,
                            }
                            papers.append(paper_record)
                            
                            if view.decision in stats:
                                stats[view.decision] += 1
                        
                        return {
                            "papers": papers,
                            "last_updated": datetime.utcnow().isoformat() + "Z",
                            "total_papers_seen": len(papers),
                            "stats": stats,
                        }
        except Exception as e:
            logger.error(f"Error loading papers state from DB: {e}")
    
    # Fallback to local file
    return _load_local_papers_state()


def get_seen_paper_ids() -> Set[str]:
    """Get all paper IDs that have been seen."""
    state = get_papers_state()
    papers = state.get("papers", [])
    return {p.get("paper_id") for p in papers if p.get("paper_id")}


def get_paper_by_id(paper_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific paper record by ID."""
    state = get_papers_state()
    papers = state.get("papers", [])
    for paper in papers:
        if paper.get("paper_id") == paper_id:
            return paper
    return None


def upsert_paper(paper_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Insert or update a paper record.
    
    If using DB, creates both Paper and PaperView records.
    
    Args:
        paper_record: Paper data including paper_id, title, decision, importance.
        
    Returns:
        Updated papers state.
    """
    if is_db_available():
        try:
            user_id = _get_default_user_id()
            if user_id:
                with get_db_session() as db:
                    paper_id = paper_record.get("paper_id")
                    
                    # Find or create Paper
                    paper = db.query(Paper).filter_by(
                        source="arxiv",
                        external_id=paper_id
                    ).first()
                    
                    # Parse published date if available
                    published_at = None
                    raw_pub = paper_record.get("published") or paper_record.get("published_at") or paper_record.get("publication_date")
                    if raw_pub:
                        try:
                            from datetime import datetime as _dt
                            if isinstance(raw_pub, str):
                                raw_pub = raw_pub.replace("Z", "+00:00")
                                published_at = _dt.fromisoformat(raw_pub).replace(tzinfo=None)
                            else:
                                published_at = raw_pub
                        except Exception:
                            pass

                    if not paper:
                        paper = Paper(
                            source="arxiv",
                            external_id=paper_id,
                            title=paper_record.get("title", ""),
                            abstract=paper_record.get("abstract"),
                            authors=paper_record.get("authors", []),
                            categories=paper_record.get("categories", []),
                            url=paper_record.get("arxiv_url") or paper_record.get("link"),
                            pdf_url=paper_record.get("pdf_url"),
                            published_at=published_at,
                        )
                        db.add(paper)
                        db.flush()
                    else:
                        # Update existing paper with any new metadata
                        if published_at and not paper.published_at:
                            paper.published_at = published_at
                        if paper_record.get("abstract") and not paper.abstract:
                            paper.abstract = paper_record.get("abstract")
                        if paper_record.get("authors") and not paper.authors:
                            paper.authors = paper_record.get("authors")
                        if paper_record.get("categories") and not paper.categories:
                            paper.categories = paper_record.get("categories")
                        url = paper_record.get("arxiv_url") or paper_record.get("link")
                        if url and not paper.url:
                            paper.url = url
                    
                    # Find or create PaperView
                    view = db.query(PaperView).filter_by(
                        user_id=uuid.UUID(user_id),
                        paper_id=paper.id
                    ).first()
                    
                    if view:
                        view.decision = paper_record.get("decision", view.decision)
                        view.importance = paper_record.get("importance", view.importance)
                        view.notes = paper_record.get("notes", view.notes)
                        view.relevance_score = paper_record.get("relevance_score", view.relevance_score)
                        view.novelty_score = paper_record.get("novelty_score", view.novelty_score)
                        view.embedded_in_pinecone = paper_record.get("embedded_in_pinecone", view.embedded_in_pinecone)
                        # Update agent delivery decisions if provided
                        if paper_record.get("agent_email_decision") is not None:
                            view.agent_email_decision = paper_record["agent_email_decision"]
                        if paper_record.get("agent_calendar_decision") is not None:
                            view.agent_calendar_decision = paper_record["agent_calendar_decision"]
                        view.seen_count += 1
                    else:
                        view = PaperView(
                            user_id=uuid.UUID(user_id),
                            paper_id=paper.id,
                            decision=paper_record.get("decision", "logged"),
                            importance=paper_record.get("importance", "low"),
                            notes=paper_record.get("notes"),
                            relevance_score=paper_record.get("relevance_score"),
                            novelty_score=paper_record.get("novelty_score"),
                            embedded_in_pinecone=paper_record.get("embedded_in_pinecone", False),
                            agent_email_decision=paper_record.get("agent_email_decision"),
                            agent_calendar_decision=paper_record.get("agent_calendar_decision"),
                        )
                        db.add(view)
                    
                    db.commit()
                    logger.debug(f"Upserted paper {paper_id} to database")
                    return get_papers_state()  # Return from DB, don't fall back to local
        except Exception as e:
            logger.error(f"Error upserting paper to DB: {e}")
            # Fall through to local file only on DB error
    
    # Only use local file for development or when DB fails
    try:
        from .json_store import upsert_paper as json_upsert_paper
        return json_upsert_paper(paper_record)
    except Exception as e:
        logger.warning(f"Could not update local papers file: {e}")
        return get_papers_state()


def upsert_papers(paper_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Insert or update multiple paper records in batch.
    
    Args:
        paper_records: List of paper record dictionaries.
        
    Returns:
        Updated papers state.
    """
    for record in paper_records:
        upsert_paper(record)
    
    return get_papers_state()


# =============================================================================
# arXiv Categories (from arxiv_categories.py)
# =============================================================================

def get_arxiv_categories_db() -> List[Dict[str, Any]]:
    """
    Get all arXiv categories from database.
    
    Returns:
        List of category dictionaries.
    """
    if is_db_available():
        try:
            with get_db_session() as db:
                categories = db.query(ArxivCategoryDB).all()
                if categories:
                    return [arxiv_category_to_dict(c) for c in categories]
        except Exception as e:
            logger.error(f"Error loading arXiv categories from DB: {e}")
    
    # Fallback to local file
    try:
        data = load_json(_get_data_dir() / "arxiv_categories.json")
        return data.get("categories", [])
    except Exception:
        return []


def get_category_name(category_code: str) -> Optional[str]:
    """Get the human-readable name for a category code."""
    categories = get_arxiv_categories_db()
    for cat in categories:
        code = cat.get("category_code") or cat.get("code")
        if code == category_code:
            return cat.get("category_name") or cat.get("name")
    return None


# =============================================================================
# Migration Functions
# =============================================================================

def migrate_all_to_db() -> Dict[str, int]:
    """
    Migrate all local JSON data to database.
    
    Returns:
        Dictionary with counts of migrated records.
    """
    if not is_db_available():
        logger.error("Database not available, cannot migrate")
        return {"error": "Database not available"}
    
    results = {}
    
    # 1. Migrate research profile (creates/updates user)
    logger.info("Migrating research profile...")
    user = get_or_create_default_user()
    if user:
        results["user"] = 1
        
        # Update with full profile data
        local_profile = _load_local_research_profile()
        if local_profile:
            update_research_profile(local_profile)
    
    # 2. Migrate colleagues
    logger.info("Migrating colleagues...")
    local_colleagues = _load_local_colleagues()
    if local_colleagues:
        save_colleagues(local_colleagues)
        results["colleagues"] = len(local_colleagues)
    
    # 3. Migrate delivery policy
    logger.info("Migrating delivery policy...")
    local_policy = _load_local_delivery_policy()
    if local_policy:
        save_delivery_policy(local_policy)
        results["delivery_policy"] = 1
    
    # 4. Migrate papers
    logger.info("Migrating papers...")
    local_papers = _load_local_papers_state()
    papers_list = local_papers.get("papers", [])
    if papers_list:
        for paper in papers_list:
            upsert_paper(paper)
        results["papers"] = len(papers_list)
    
    # 5. Migrate arXiv categories (handled by arxiv_categories.py)
    logger.info("Migrating arXiv categories...")
    from ..tools.arxiv_categories import sync_to_db
    cat_count = sync_to_db()
    results["arxiv_categories"] = cat_count
    
    logger.info(f"Migration complete: {results}")
    return results


def delete_local_data_files() -> List[str]:
    """
    Delete local JSON data files after successful migration.
    
    WARNING: Only call this after confirming data is in the database!
    
    Returns:
        List of deleted file paths.
    """
    deleted = []
    data_dir = _get_data_dir()
    
    files_to_delete = [
        "research_profile.json",
        "colleagues.json",
        "delivery_policy.json",
        "papers_state.json",
        "arxiv_categories.json",
    ]
    
    for filename in files_to_delete:
        filepath = data_dir / filename
        if filepath.exists():
            try:
                filepath.unlink()
                deleted.append(str(filepath))
                logger.info(f"Deleted local file: {filepath}")
            except Exception as e:
                logger.error(f"Failed to delete {filepath}: {e}")
    
    return deleted


# =============================================================================
# Artifact Storage Functions
# =============================================================================

def save_artifact_to_db(
    file_type: str,
    file_path: str,
    content: str,
    paper_id: Optional[str] = None,
    colleague_id: Optional[str] = None,
    description: Optional[str] = None,
    triggered_by: str = "agent",
    colleague_email: Optional[str] = None,
    colleague_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Save an artifact (calendar event, reading list entry, email, share) to the database.
    
    Args:
        file_type: Type of artifact - 'calendar', 'reading_list', 'email', 'share'
        file_path: Original file path (used for context)
        content: Content of the artifact
        paper_id: arXiv paper ID (optional)
        colleague_id: Colleague ID for shares (optional)
        description: Description of the artifact (optional)
        triggered_by: Who triggered this action - 'agent' or 'user' (default: 'agent')
        colleague_email: Colleague email for sending share notifications (optional)
        colleague_name: Colleague name for email personalization (optional)
        
    Returns:
        Result dictionary with success status
    """
    if not is_db_available():
        logger.warning(f"[ARTIFACT] Database not available, cannot save {file_type} artifact")
        return {"success": False, "error": "Database not available"}
    
    user_id = _get_default_user_id()
    if not user_id:
        logger.warning(f"[ARTIFACT] No user found, cannot save {file_type} artifact")
        return {"success": False, "error": "No user found"}
    
    # Get the user's email from profile for email artifacts
    user = get_or_create_default_user()
    user_email = user.get("email", "") if user else ""
    user_name = user.get("name", "Researcher") if user else "Researcher"
    
    logger.info(f"[ARTIFACT] Saving {file_type} artifact (triggered_by={triggered_by}, paper_id={paper_id})")
    
    try:
        from .postgres_store import PostgresStore
        store = PostgresStore()
        
        # Get paper DB ID if we have an arXiv ID
        db_paper_id = None
        paper_title = None
        paper_url = None
        paper_importance = "medium"
        if paper_id:
            with get_db_session() as db:
                paper = db.query(Paper).filter_by(
                    source="arxiv",
                    external_id=paper_id
                ).first()
                if paper:
                    db_paper_id = paper.id
                    paper_title = paper.title
                    paper_url = paper.url or f"https://arxiv.org/abs/{paper_id}"
                    # Get importance from paper view
                    view = db.query(PaperView).filter_by(
                        user_id=uuid.UUID(user_id),
                        paper_id=paper.id
                    ).first()
                    if view and view.importance:
                        paper_importance = view.importance
        
        if file_type == "calendar":
            # Parse start time from ICS content or use current time
            from datetime import datetime, timedelta
            import re
            
            # Try to extract DTSTART from ICS
            start_time = datetime.utcnow() + timedelta(days=1)  # Default to tomorrow
            dtstart_match = re.search(r'DTSTART[^:]*:(\d{8}T\d{6})', content)
            if dtstart_match:
                try:
                    start_time = datetime.strptime(dtstart_match.group(1), "%Y%m%dT%H%M%S")
                except ValueError:
                    pass
            
            # Extract duration from ICS DTSTART/DTEND instead of hardcoding 30
            duration_minutes = 30  # fallback default
            dtend_match = re.search(r'DTEND[^:]*:(\d{8}T\d{6})', content)
            if dtstart_match and dtend_match:
                try:
                    dt_start_parsed = datetime.strptime(dtstart_match.group(1), "%Y%m%dT%H%M%S")
                    dt_end_parsed = datetime.strptime(dtend_match.group(1), "%Y%m%dT%H%M%S")
                    computed_dur = int((dt_end_parsed - dt_start_parsed).total_seconds() / 60)
                    if 5 <= computed_dur <= 480:
                        duration_minutes = computed_dur
                except ValueError:
                    pass
            
            # Extract title from ICS
            title = description or "Reading scheduled"
            summary_match = re.search(r'SUMMARY:(.+?)(?:\r?\n|$)', content)
            if summary_match:
                title = summary_match.group(1).strip()
            
            logger.info(f"[ARTIFACT] Creating calendar event: '{title}' at {start_time}, duration={duration_minutes}m (triggered_by={triggered_by})")
            
            # Store the DB UUID in paper_ids (not arXiv external ID)
            db_paper_id_str = str(db_paper_id) if db_paper_id else None
            result = store.create_calendar_event(
                user_id=uuid.UUID(user_id),
                paper_id=db_paper_id,
                title=title,
                start_time=start_time,
                duration_minutes=duration_minutes,
                ics_text=content,
                triggered_by=triggered_by,
                paper_ids=[db_paper_id_str] if db_paper_id_str else None,
            )
            
            event_id = result.get("id")
            logger.info(f"[ARTIFACT] Created calendar event {event_id} (triggered_by={triggered_by})")
            
            # Send calendar invite email if user has an email configured
            if user_email and user_email != "default@researchpulse.local":
                try:
                    calendar_invite_sender = _get_calendar_invite_sender()
                    send_reading_reminder_invite = calendar_invite_sender.send_reading_reminder_invite
                    
                    papers_for_invite = []
                    if paper_title:
                        papers_for_invite.append({
                            "title": paper_title,
                            "url": paper_url,
                            "importance": paper_importance,
                        })
                    
                    agent_note = "Scheduled automatically by ResearchPulse based on paper importance." if triggered_by == "agent" else None
                    
                    invite_result = send_reading_reminder_invite(
                        user_email=user_email,
                        user_name=user_name,
                        papers=papers_for_invite,
                        start_time=start_time,
                        duration_minutes=duration_minutes,
                        reminder_minutes=15,
                        triggered_by=triggered_by,
                        agent_note=agent_note,
                    )
                    
                    if invite_result.get("success"):
                        logger.info(f"[ARTIFACT] Sent calendar invite email to {user_email}")
                        
                        # Backfill ics_uid to CalendarEvent so reschedule can find it
                        invite_ics_uid = invite_result.get("ics_uid", "")
                        if event_id and invite_ics_uid:
                            try:
                                store.update_calendar_event(
                                    uuid.UUID(event_id),
                                    {"ics_uid": invite_ics_uid, "sequence_number": 0},
                                )
                                logger.info(f"[ARTIFACT] Saved ics_uid to calendar event {event_id}")
                            except Exception as uid_err:
                                logger.warning(f"[ARTIFACT] Could not backfill ics_uid: {uid_err}")
                        
                        # Store the invite email record
                        if event_id:
                            store.create_calendar_invite_email(
                                calendar_event_id=uuid.UUID(event_id),
                                user_id=uuid.UUID(user_id),
                                message_id=invite_result.get("message_id", ""),
                                recipient_email=user_email,
                                subject=invite_result.get("subject", title),
                                ics_uid=invite_ics_uid,
                                triggered_by=triggered_by,
                            )
                    else:
                        logger.warning(f"[ARTIFACT] Failed to send calendar invite: {invite_result.get('error')}")
                        
                except Exception as invite_err:
                    logger.warning(f"[ARTIFACT] Could not send calendar invite email: {invite_err}")
            
            return {"success": True, "type": "calendar", "id": str(event_id) if event_id else ""}
        
        elif file_type == "reading_list":
            # Reading list entries are saved as part of the paper view with notes
            # We'll store this in a special table or as paper notes
            if db_paper_id:
                with get_db_session() as db:
                    view = db.query(PaperView).filter_by(
                        user_id=uuid.UUID(user_id),
                        paper_id=db_paper_id
                    ).first()
                    if view:
                        existing_notes = view.notes or ""
                        if "Reading List Entry:" not in existing_notes:
                            view.notes = existing_notes + f"\n\nReading List Entry:\n{content}"
                            db.commit()
            return {"success": True, "type": "reading_list", "paper_id": paper_id}
        
        elif file_type == "email":
            # Extract subject from content
            lines = content.split("\n")
            subject = "Paper Recommendation"
            body = content
            
            for line in lines:
                if line.startswith("Subject:"):
                    subject = line.replace("Subject:", "").strip()
            
            # Use the user's email from profile, NOT from content parsing
            recipient = user_email
            if not recipient or recipient == "default@researchpulse.local":
                logger.warning(f"[ARTIFACT] No valid user email configured, cannot save email artifact")
                return {"success": False, "error": "No user email configured in settings"}
            
            logger.info(f"[ARTIFACT] Creating email record: '{subject}' to {recipient} (triggered_by={triggered_by})")
            
            result = store.create_email(
                user_id=uuid.UUID(user_id),
                paper_id=db_paper_id,
                recipient_email=recipient,
                subject=subject,
                body_text=body,
                triggered_by=triggered_by,
                paper_ids=[paper_id] if paper_id else None,
            )
            
            email_id = result.get("id")
            logger.info(f"[ARTIFACT] Created email record {email_id} (triggered_by={triggered_by})")
            
            # Actually send the email via SMTP
            if email_id:
                try:
                    # Try relative import first, fall back to absolute import for different execution contexts
                    try:
                        from ..tools.decide_delivery import _send_email_smtp, _generate_email_content_html, ScoredPaper
                    except (ImportError, ValueError):
                        # Fallback: add tools path and import directly
                        import sys
                        from pathlib import Path
                        tools_path = str(Path(__file__).parent.parent / "tools")
                        if tools_path not in sys.path:
                            sys.path.insert(0, tools_path)
                        from decide_delivery import _send_email_smtp, _generate_email_content_html, ScoredPaper
                    
                    # Clean up the body for sending (remove header lines if present)
                    send_body = body
                    body_lines = body.split("\n")
                    clean_start = 0
                    for i, line in enumerate(body_lines):
                        if line.strip() == "" and i > 0:
                            # Skip header section (Subject:, From:, etc.)
                            if any(body_lines[j].startswith(("Subject:", "From:", "Date:", "To:")) for j in range(i)):
                                clean_start = i + 1
                                break
                    if clean_start > 0:
                        send_body = "\n".join(body_lines[clean_start:])
                    
                    # Generate HTML email from paper data
                    html_body = None
                    logger.info(f"[ARTIFACT] db_paper_id={db_paper_id}, paper_id={paper_id}")
                    
                    # Try to generate HTML - either from DB or by parsing content
                    try:
                        import re
                        
                        # Parse paper info from content
                        priority = "medium"
                        content_upper = content.upper()
                        if "HIGH PRIORITY" in content_upper or "URGENT" in content_upper:
                            priority = "high"
                        elif "LOW PRIORITY" in content_upper:
                            priority = "low"
                        
                        # Parse scores
                        relevance_score = 0.5
                        novelty_score = 0.5
                        rel_match = re.search(r'Relevance Score:\s*(\d+)%', content)
                        if rel_match:
                            relevance_score = int(rel_match.group(1)) / 100.0
                        nov_match = re.search(r'Novelty Score:\s*(\d+)%', content)
                        if nov_match:
                            novelty_score = int(nov_match.group(1)) / 100.0
                        
                        # Parse title
                        title = "Research Paper"
                        title_match = re.search(r'Title:\s*(.+?)(?:\n|$)', content)
                        if title_match:
                            title = title_match.group(1).strip()
                        
                        # Parse arXiv ID
                        arxiv_id = paper_id or ""
                        arxiv_match = re.search(r'arXiv ID:\s*(\S+)', content)
                        if arxiv_match:
                            arxiv_id = arxiv_match.group(1).strip()
                        
                        # Parse authors
                        authors = []
                        authors_match = re.search(r'Authors:\s*(.+?)(?:\n|$)', content)
                        if authors_match:
                            authors = [a.strip() for a in authors_match.group(1).split(",")]
                        
                        # Parse categories
                        categories = []
                        cat_match = re.search(r'Categories:\s*(.+?)(?:\n|$)', content)
                        if cat_match:
                            categories = [c.strip() for c in cat_match.group(1).split(",")]
                        
                        # Parse abstract
                        abstract = ""
                        abs_match = re.search(r'ABSTRACT\s*-+\s*(.+?)(?:\n={10,}|$)', content, re.DOTALL)
                        if abs_match:
                            abstract = abs_match.group(1).strip()
                        
                        # Parse explanation
                        explanation = ""
                        exp_match = re.search(r'Assessment:\s*(.+?)(?:\n-|$)', content, re.DOTALL)
                        if exp_match:
                            explanation = exp_match.group(1).strip()
                        
                        # Try to get more info from DB if available
                        if db_paper_id:
                            with get_db_session() as db:
                                paper_row = db.query(Paper).filter_by(id=db_paper_id).first()
                                if paper_row:
                                    arxiv_id = paper_row.arxiv_id or arxiv_id
                                    title = paper_row.title or title
                                    authors = paper_row.authors.split(", ") if paper_row.authors else authors
                                    abstract = paper_row.abstract or abstract
                                    categories = paper_row.categories.split(", ") if paper_row.categories else categories
                        
                        # Create ScoredPaper for HTML generation
                        scored_paper = ScoredPaper(
                            arxiv_id=arxiv_id,
                            title=title,
                            authors=authors,
                            abstract=abstract,
                            publication_date="",
                            link=f"http://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                            categories=categories,
                            relevance_score=relevance_score,
                            novelty_score=novelty_score,
                            explanation=explanation,
                            importance=priority,
                        )
                        
                        # Get researcher name
                        researcher_name = "Researcher"
                        with get_db_session() as db:
                            user_obj = db.query(User).filter_by(id=uuid.UUID(user_id)).first()
                            if user_obj and user_obj.name:
                                researcher_name = user_obj.name
                        
                        html_body = _generate_email_content_html(
                            paper=scored_paper,
                            priority=priority,
                            include_abstract=True,
                            include_explanation=True,
                            researcher_name=researcher_name,
                        )
                        logger.info(f"[ARTIFACT] Generated HTML email ({len(html_body)} chars)")
                    except Exception as html_err:
                        logger.warning(f"[ARTIFACT] Could not generate HTML email: {html_err}")
                    
                    logger.info(f"[ARTIFACT] Sending email via SMTP to {recipient}...")
                    email_sent = _send_email_smtp(
                        to_email=recipient,
                        subject=subject,
                        body=send_body,
                        html_body=html_body,
                    )
                    
                    # Update email status based on send result
                    if email_sent:
                        store.update_email_status(
                            email_id=uuid.UUID(email_id),
                            status="sent",
                        )
                        logger.info(f"[ARTIFACT] Email sent successfully to {recipient}")
                    else:
                        store.update_email_status(
                            email_id=uuid.UUID(email_id),
                            status="failed",
                            error="SMTP send failed - check SMTP configuration",
                        )
                        logger.warning(f"[ARTIFACT] Email SMTP send failed for {recipient}")
                        
                except Exception as send_err:
                    logger.error(f"[ARTIFACT] Error sending email via SMTP: {send_err}")
                    try:
                        store.update_email_status(
                            email_id=uuid.UUID(email_id),
                            status="failed",
                            error=str(send_err),
                        )
                    except Exception:
                        pass
            
            return {"success": True, "type": "email", "id": str(email_id) if email_id else ""}
        
        elif file_type == "share":
            # Get colleague DB ID
            db_colleague_id = None
            resolved_colleague_email = colleague_email
            resolved_colleague_name = colleague_name
            if colleague_id:
                with get_db_session() as db:
                    from .orm_models import Colleague as ColleagueORM
                    try:
                        colleague_uuid = uuid.UUID(colleague_id)
                        colleague_row = db.query(ColleagueORM).filter_by(id=colleague_uuid).first()
                    except (ValueError, AttributeError):
                        colleague_row = None
                    if colleague_row:
                        db_colleague_id = colleague_row.id
                        # Resolve email/name from DB if not provided
                        if not resolved_colleague_email:
                            resolved_colleague_email = colleague_row.email
                        if not resolved_colleague_name:
                            resolved_colleague_name = colleague_row.name
            
            share_id = None
            if db_paper_id and db_colleague_id:
                # Compute a basic match score from the content (topic overlap indicator)
                _match_score = 0.75  # Default for matched shares
                import re as _re
                _score_match = _re.search(r'match_score[=:]\s*([\d.]+)', content)
                if _score_match:
                    try:
                        _match_score = float(_score_match.group(1))
                    except (ValueError, TypeError):
                        pass
                result = store.create_share(
                    user_id=uuid.UUID(user_id),
                    paper_id=db_paper_id,
                    colleague_id=db_colleague_id,
                    reason=description or "Matched research interests",
                    match_score=_match_score,
                )
                share_id = result.get("id", "")
                logger.info(f"[ARTIFACT] Created share record {share_id} for colleague {resolved_colleague_name}")
            else:
                logger.warning(f"[ARTIFACT] Could not create share record: paper_id={db_paper_id}, colleague_id={db_colleague_id}")
            
            # Send share email to colleague regardless of DB record success
            if resolved_colleague_email:
                try:
                    try:
                        from ..tools.decide_delivery import _send_email_smtp
                        from ..tools.outbound_email import EmailType
                    except (ImportError, ValueError):
                        import sys as _sys
                        from pathlib import Path as _Path
                        tools_path = str(_Path(__file__).parent.parent / "tools")
                        if tools_path not in _sys.path:
                            _sys.path.insert(0, tools_path)
                        from decide_delivery import _send_email_smtp
                        from outbound_email import EmailType
                    
                    # Extract subject from share content
                    share_subject = "ResearchPulse: Paper recommendation"
                    for line in content.split("\n"):
                        if line.startswith("Subject:"):
                            share_subject = line.replace("Subject:", "").strip()
                            break
                    
                    # Generate HTML email using the colleague paper email template
                    share_html_body = None
                    try:
                        try:
                            from ..tools.email_templates import render_colleague_paper_email
                        except (ImportError, ValueError):
                            from email_templates import render_colleague_paper_email
                        
                        # Parse paper info from content for template
                        import re as _re2
                        _title = "Research Paper"
                        _arxiv_id = paper_id or ""
                        _authors = []
                        _categories = []
                        _abstract = ""
                        _relevance_reason = description or "Matched research interests"
                        _link = ""
                        _pub_date = ""
                        
                        for _line in content.split("\n"):
                            if _line.startswith("Title:"):
                                _title = _line.replace("Title:", "").strip()
                            elif _line.startswith("arXiv ID:"):
                                _arxiv_id = _line.replace("arXiv ID:", "").strip()
                            elif _line.startswith("Authors:"):
                                _authors = [a.strip() for a in _line.replace("Authors:", "").split(",")]
                            elif _line.startswith("Link:"):
                                _link = _line.replace("Link:", "").strip()
                            elif _line.startswith("Categories:"):
                                _categories = [c.strip() for c in _line.replace("Categories:", "").split(",")]
                            elif _line.startswith("Why this might be relevant:"):
                                _relevance_reason = _line.replace("Why this might be relevant:", "").strip()
                        
                        # Try to get richer info from DB
                        if db_paper_id:
                            with get_db_session() as _db:
                                _paper_row = _db.query(Paper).filter_by(id=db_paper_id).first()
                                if _paper_row:
                                    _title = _paper_row.title or _title
                                    _arxiv_id = _paper_row.external_id or _paper_row.arxiv_id if hasattr(_paper_row, 'arxiv_id') else _arxiv_id
                                    _authors = _paper_row.authors.split(", ") if isinstance(_paper_row.authors, str) else (_paper_row.authors or _authors)
                                    _abstract = _paper_row.abstract or _abstract
                                    _categories = _paper_row.categories.split(", ") if isinstance(_paper_row.categories, str) else (_paper_row.categories or _categories)
                                    _link = _paper_row.url or _link
                                    _pub_date = _paper_row.published_at.strftime("%Y-%m-%d") if hasattr(_paper_row, 'published_at') and _paper_row.published_at else _pub_date
                        
                        if not _link and _arxiv_id:
                            _link = f"https://arxiv.org/abs/{_arxiv_id}"
                        
                        # Get owner name
                        _owner_name = "a researcher"
                        with get_db_session() as _db:
                            _user_obj = _db.query(User).filter_by(id=uuid.UUID(user_id)).first()
                            if _user_obj and _user_obj.name:
                                _owner_name = _user_obj.name
                        
                        # Get colleague interests
                        _matched_interests = []
                        if db_colleague_id:
                            with get_db_session() as _db:
                                from .orm_models import Colleague as _ColleagueORM
                                _coll = _db.query(_ColleagueORM).filter_by(id=db_colleague_id).first()
                                if _coll and _coll.topics:
                                    _matched_interests = _coll.topics[:5]
                                elif _coll and _coll.keywords:
                                    _matched_interests = _coll.keywords[:5]
                        
                        _paper_data = {
                            "title": _title,
                            "arxiv_id": _arxiv_id,
                            "authors": _authors,
                            "categories": _categories,
                            "publication_date": _pub_date,
                            "abstract": _abstract,
                            "link": _link,
                        }
                        
                        _subj, _plain, share_html_body = render_colleague_paper_email(
                            paper=_paper_data,
                            colleague_name=resolved_colleague_name or "Colleague",
                            relevance_reason=_relevance_reason,
                            matched_interests=_matched_interests,
                            owner_name=_owner_name,
                        )
                        # Use the template's subject line
                        share_subject = _subj
                        logger.info(f"[ARTIFACT] Generated HTML share email ({len(share_html_body)} chars)")
                    except Exception as _html_err:
                        logger.warning(f"[ARTIFACT] Could not generate HTML share email: {_html_err}")
                    
                    logger.info(f"[ARTIFACT] Sending share email to colleague {resolved_colleague_name} <{resolved_colleague_email}>")
                    email_sent = _send_email_smtp(
                        to_email=resolved_colleague_email,
                        subject=share_subject,
                        body=content,
                        html_body=share_html_body,
                        email_type=EmailType.UPDATE,
                    )
                    
                    # Update share status based on send result
                    if share_id and db_paper_id and db_colleague_id:
                        if email_sent:
                            store.update_share_status(
                                share_id=uuid.UUID(str(share_id)),
                                status="sent",
                            )
                            logger.info(f"[ARTIFACT] Share email sent successfully to {resolved_colleague_email}")
                        else:
                            store.update_share_status(
                                share_id=uuid.UUID(str(share_id)),
                                status="failed",
                                error="SMTP send failed",
                            )
                            logger.warning(f"[ARTIFACT] Share email SMTP send failed for {resolved_colleague_email}")
                except Exception as send_err:
                    logger.error(f"[ARTIFACT] Error sending share email to {resolved_colleague_email}: {send_err}")
                    if share_id and db_paper_id and db_colleague_id:
                        try:
                            store.update_share_status(
                                share_id=uuid.UUID(str(share_id)),
                                status="failed",
                                error=str(send_err),
                            )
                        except Exception:
                            pass
            
            return {"success": True, "type": "share", "id": str(share_id) if share_id else ""}
        
        else:
            return {"success": False, "error": f"Unknown artifact type: {file_type}"}
            
    except Exception as e:
        logger.error(f"[ARTIFACT] Error saving artifact to DB: {e}")
        return {"success": False, "error": str(e)}


def save_artifacts_to_db(artifacts: List[Dict[str, Any]], triggered_by: str = "agent") -> Dict[str, Any]:
    """
    Save multiple artifacts to the database.
    
    Args:
        artifacts: List of artifact dictionaries with file_type, file_path, content, etc.
        triggered_by: Who triggered these actions - 'agent' or 'user' (default: 'agent')
        
    Returns:
        Dictionary with success counts and errors
    """
    results = {
        "success": True,
        "saved": 0,
        "failed": 0,
        "errors": [],
        "details": [],
    }
    
    logger.info(f"[ARTIFACTS] Saving {len(artifacts)} artifacts to DB (triggered_by={triggered_by})")
    
    for artifact in artifacts:
        result = save_artifact_to_db(
            file_type=artifact.get("file_type", ""),
            file_path=artifact.get("file_path", ""),
            content=artifact.get("content", ""),
            paper_id=artifact.get("paper_id"),
            colleague_id=artifact.get("colleague_id"),
            description=artifact.get("description"),
            triggered_by=triggered_by,
            colleague_email=artifact.get("colleague_email"),
            colleague_name=artifact.get("colleague_name"),
        )
        
        if result.get("success"):
            results["saved"] += 1
            results["details"].append(result)
            logger.info(f"[ARTIFACTS] Saved {artifact.get('file_type')} artifact successfully")
        else:
            results["failed"] += 1
            results["errors"].append(result.get("error", "Unknown error"))
            logger.warning(f"[ARTIFACTS] Failed to save {artifact.get('file_type')} artifact: {result.get('error')}")
    
    if results["failed"] > 0:
        results["success"] = False
    
    logger.info(f"[ARTIFACTS] Completed: {results['saved']} saved, {results['failed']} failed")
    
    return results


# =============================================================================
# Saved Prompts - User-saved prompt templates for Quick Prompt Builder
# =============================================================================

def save_prompt_template(
    name: str,
    prompt_text: str,
    template_type: Optional[str] = None,
    areas: Optional[List[str]] = None,
    topics: Optional[List[str]] = None,
    time_period: Optional[str] = None,
    paper_count: Optional[int] = None,
) -> Optional[str]:
    """
    Save a prompt template for quick reuse.
    
    Args:
        name: User-given name for the prompt
        prompt_text: The generated prompt text
        template_type: Type of template (topic_time, top_k, etc.)
        areas: Selected research areas
        topics: Focus topics
        time_period: Time period selection
        paper_count: Number of papers requested
        
    Returns:
        ID of the saved prompt, or None if failed
    """
    if not is_db_available():
        logger.warning("Database not available for saving prompt template")
        return None
    
    try:
        user_id = _get_default_user_id()
        if not user_id:
            logger.error("No default user found")
            return None
        
        with get_db_session() as db:
            saved_prompt = SavedPrompt(
                user_id=uuid.UUID(user_id),
                name=name,
                prompt_text=prompt_text,
                template_type=template_type,
                areas=areas or [],
                topics=topics or [],
                time_period=time_period,
                paper_count=paper_count,
            )
            db.add(saved_prompt)
            db.commit()
            db.refresh(saved_prompt)
            
            logger.info(f"Saved prompt template: {name} (ID: {saved_prompt.id})")
            return str(saved_prompt.id)
            
    except Exception as e:
        logger.error(f"Error saving prompt template: {e}")
        return None


def get_saved_prompts(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get all saved prompt templates for the current user.
    
    Args:
        limit: Maximum number of prompts to return
        
    Returns:
        List of saved prompt dictionaries
    """
    if not is_db_available():
        return []
    
    try:
        user_id = _get_default_user_id()
        if not user_id:
            return []
        
        with get_db_session() as db:
            prompts = db.query(SavedPrompt).filter_by(
                user_id=uuid.UUID(user_id)
            ).order_by(
                SavedPrompt.created_at.desc()
            ).limit(limit).all()
            
            return [p.to_dict() for p in prompts]
            
    except Exception as e:
        logger.error(f"Error getting saved prompts: {e}")
        return []


def get_saved_prompt_by_id(prompt_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific saved prompt by ID.
    
    Args:
        prompt_id: The prompt UUID
        
    Returns:
        Saved prompt dictionary or None
    """
    if not is_db_available():
        return None
    
    try:
        with get_db_session() as db:
            prompt = db.query(SavedPrompt).filter_by(
                id=uuid.UUID(prompt_id)
            ).first()
            
            if prompt:
                return prompt.to_dict()
            return None
            
    except Exception as e:
        logger.error(f"Error getting saved prompt: {e}")
        return None


def delete_saved_prompt(prompt_id: str) -> bool:
    """
    Delete a saved prompt template.
    
    Args:
        prompt_id: The prompt UUID to delete
        
    Returns:
        True if deleted, False otherwise
    """
    if not is_db_available():
        return False
    
    try:
        with get_db_session() as db:
            prompt = db.query(SavedPrompt).filter_by(
                id=uuid.UUID(prompt_id)
            ).first()
            
            if prompt:
                db.delete(prompt)
                db.commit()
                logger.info(f"Deleted saved prompt: {prompt_id}")
                return True
            return False
            
    except Exception as e:
        logger.error(f"Error deleting saved prompt: {e}")
        return False


def update_saved_prompt(
    prompt_id: str,
    name: Optional[str] = None,
    prompt_text: Optional[str] = None,
    template_type: Optional[str] = None,
    areas: Optional[List[str]] = None,
    topics: Optional[List[str]] = None,
    time_period: Optional[str] = None,
    paper_count: Optional[int] = None,
) -> bool:
    """
    Update a saved prompt template.
    
    Args:
        prompt_id: The prompt UUID to update
        Other args: Fields to update (None = don't change)
        
    Returns:
        True if updated, False otherwise
    """
    if not is_db_available():
        return False
    
    try:
        with get_db_session() as db:
            prompt = db.query(SavedPrompt).filter_by(
                id=uuid.UUID(prompt_id)
            ).first()
            
            if not prompt:
                return False
            
            if name is not None:
                prompt.name = name
            if prompt_text is not None:
                prompt.prompt_text = prompt_text
            if template_type is not None:
                prompt.template_type = template_type
            if areas is not None:
                prompt.areas = areas
            if topics is not None:
                prompt.topics = topics
            if time_period is not None:
                prompt.time_period = time_period
            if paper_count is not None:
                prompt.paper_count = paper_count
            
            db.commit()
            logger.info(f"Updated saved prompt: {prompt_id}")
            return True
            
    except Exception as e:
        logger.error(f"Error updating saved prompt: {e}")
        return False


# =============================================================================
# Default Templates for Seeding
# =============================================================================

DEFAULT_TEMPLATES = [
    {
        "name": "Recent AI Papers",
        "prompt_text": "Find recent papers on artificial intelligence and machine learning from the last week. Focus on transformer architectures and large language models.",
        "template_type": "topic_time",
        "areas": ["cs.AI", "cs.LG", "cs.CL"],
        "topics": ["artificial intelligence", "machine learning", "transformers"],
        "time_period": "last_week",
        "paper_count": 10
    },
    {
        "name": "Top 5 NLP Papers",
        "prompt_text": "Find the top 5 most relevant papers on natural language processing from the past month.",
        "template_type": "top_k",
        "areas": ["cs.CL"],
        "topics": ["natural language processing", "NLP"],
        "time_period": "last_month",
        "paper_count": 5
    },
    {
        "name": "Computer Vision Survey",
        "prompt_text": "Find survey papers and comprehensive reviews on computer vision and image recognition published this year.",
        "template_type": "survey",
        "areas": ["cs.CV"],
        "topics": ["computer vision", "image recognition", "deep learning"],
        "time_period": "this_year",
        "paper_count": 10
    },
    {
        "name": "Reinforcement Learning Trends",
        "prompt_text": "What are the trending topics and methodologies in reinforcement learning research from the past month?",
        "template_type": "trends",
        "areas": ["cs.LG", "cs.AI"],
        "topics": ["reinforcement learning", "RL", "deep RL"],
        "time_period": "last_month",
        "paper_count": 10
    }
]


def seed_default_templates() -> int:
    """
    Seed default prompt templates if the saved_prompts table is empty.
    
    Call this on server startup to ensure templates are available.
    
    Returns:
        Number of templates seeded (0 if templates already exist)
    """
    if not is_db_available():
        logger.warning("Database not available for seeding templates")
        return 0
    
    try:
        # Check if any templates already exist
        existing = get_saved_prompts(limit=1)
        if existing:
            logger.info("Templates already exist, skipping seed")
            return 0
        
        seeded = 0
        for template in DEFAULT_TEMPLATES:
            prompt_id = save_prompt_template(
                name=template["name"],
                prompt_text=template["prompt_text"],
                template_type=template.get("template_type"),
                areas=template.get("areas", []),
                topics=template.get("topics", []),
                time_period=template.get("time_period"),
                paper_count=template.get("paper_count"),
            )
            if prompt_id:
                seeded += 1
                logger.info(f"Seeded template: {template['name']}")
        
        logger.info(f"Seeded {seeded} default templates")
        return seeded
        
    except Exception as e:
        logger.error(f"Error seeding templates: {e}")
        return 0


# =============================================================================
# Prompt Templates - Quick reusable prompt templates (PromptTemplate model)
# =============================================================================

BUILTIN_PROMPT_TEMPLATES = [
    {"name": "Template 1: Topic + Venue + Time", "text": "Provide recent research papers on <TOPIC> published in <VENUE> within the last <TIME_PERIOD>."},
    {"name": "Template 2: Topic + Time", "text": "Provide recent research papers on <TOPIC> published within the last <TIME_PERIOD>."},
    {"name": "Template 3: Topic Only", "text": "Provide the most recent research papers on <TOPIC>."},
    {"name": "Template 4: Top-K Papers", "text": "Provide the top <K> most relevant or influential research papers on <TOPIC>."},
    {"name": "Template 5: Top-K + Time", "text": "Provide the top <K> research papers on <TOPIC> from the last <TIME_PERIOD>."},
    {"name": "Template 6: Survey / Review", "text": "Provide recent survey or review papers on <TOPIC>."},
    {"name": "Template 7: Method-Focused", "text": "Provide recent papers on <TOPIC> that focus on <METHOD_OR_APPROACH>."},
    {"name": "Template 8: Application-Focused", "text": "Provide recent papers on <TOPIC> applied to <APPLICATION_DOMAIN>."},
    {"name": "Template 9: Emerging Trends", "text": "Identify emerging research trends based on recent papers on <TOPIC>."},
    {"name": "Template 10: Structured Output", "text": "Provide recent papers on <TOPIC> including title, authors, venue, year, and a one-sentence summary."},
]


def get_prompt_templates() -> List[Dict[str, Any]]:
    """
    Get all prompt templates, ordered by: builtin first, then custom by name.
    
    Returns:
        List of template dictionaries with id, name, text, is_builtin fields
    """
    if not is_db_available():
        return []
    try:
        with get_db_session() as db:
            templates = db.query(PromptTemplate).order_by(
                PromptTemplate.is_builtin.desc(),
                PromptTemplate.name
            ).all()
            return [t.to_dict() for t in templates]
    except Exception as e:
        logger.error(f"Error getting prompt templates: {e}")
        return []


def create_prompt_template(name: str, text: str) -> Optional[str]:
    """
    Create a custom prompt template (non-builtin).
    
    Args:
        name: Display name for the template
        text: The template text content
        
    Returns:
        Template ID string if successful, None otherwise
    """
    if not is_db_available():
        return None
    try:
        with get_db_session() as db:
            template = PromptTemplate(name=name, text=text, is_builtin=False)
            db.add(template)
            db.commit()
            db.refresh(template)
            logger.info(f"Created prompt template: {name}")
            return str(template.id)
    except Exception as e:
        logger.error(f"Error creating prompt template: {e}")
        return None


def delete_prompt_template(template_id: str) -> bool:
    """
    Delete a prompt template. Only custom (non-builtin) templates can be deleted.
    
    Args:
        template_id: UUID of the template to delete
        
    Returns:
        True if deleted, False if builtin or not found
    """
    if not is_db_available():
        return False
    try:
        with get_db_session() as db:
            template = db.query(PromptTemplate).filter_by(id=uuid.UUID(template_id)).first()
            if template and not template.is_builtin:
                db.delete(template)
                db.commit()
                logger.info(f"Deleted prompt template: {template_id}")
                return True
            elif template and template.is_builtin:
                logger.warning(f"Cannot delete builtin template: {template_id}")
            return False
    except Exception as e:
        logger.error(f"Error deleting prompt template: {e}")
        return False


def seed_builtin_prompt_templates() -> int:
    """
    Seed builtin prompt templates if they don't exist.
    
    Call this on server startup to ensure the 10 default templates are available.
    Templates are matched by name to avoid duplicates.
    
    Returns:
        Number of templates seeded
    """
    if not is_db_available():
        logger.warning("Database not available for seeding builtin templates")
        return 0
    
    seeded = 0
    try:
        with get_db_session() as db:
            for template_data in BUILTIN_PROMPT_TEMPLATES:
                existing = db.query(PromptTemplate).filter_by(name=template_data["name"]).first()
                if not existing:
                    template = PromptTemplate(
                        name=template_data["name"],
                        text=template_data["text"],
                        is_builtin=True
                    )
                    db.add(template)
                    seeded += 1
                    logger.debug(f"Seeded builtin template: {template_data['name']}")
            db.commit()
        
        if seeded > 0:
            logger.info(f"Seeded {seeded} builtin prompt templates")
        return seeded
    except Exception as e:
        logger.error(f"Error seeding builtin templates: {e}")
        return 0


# =============================================================================
# Self-Check
# =============================================================================

def self_check() -> bool:
    """Run self-check tests for the data service."""
    print("=" * 60)
    print("Data Service Self-Check")
    print("=" * 60)
    
    all_passed = True
    
    def check(name: str, condition: bool) -> bool:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        return condition
    
    # Check DB availability
    print("\n1. Database Status:")
    db_available = is_db_available()
    check(f"Database configured: {db_available}", True)
    
    # Test research profile
    print("\n2. Research Profile:")
    try:
        profile = get_research_profile()
        all_passed &= check("loads successfully", profile is not None)
        all_passed &= check("has researcher_name", "researcher_name" in profile)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)
    
    # Test colleagues
    print("\n3. Colleagues:")
    try:
        colleagues = get_colleagues()
        all_passed &= check("loads successfully", colleagues is not None)
        all_passed &= check("is a list", isinstance(colleagues, list))
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)
    
    # Test delivery policy
    print("\n4. Delivery Policy:")
    try:
        policy = get_delivery_policy()
        all_passed &= check("loads successfully", policy is not None)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)
    
    # Test papers state
    print("\n5. Papers State:")
    try:
        state = get_papers_state()
        all_passed &= check("loads successfully", state is not None)
        all_passed &= check("has papers list", "papers" in state)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)
    
    # Summary
    print("\n" + "=" * 60)
    source = "DATABASE" if db_available else "LOCAL FILES"
    print(f"Data source: {source}")
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
