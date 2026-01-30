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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import uuid

from .database import is_database_configured, get_db_session
from .orm_models import (
    User, Paper, PaperView, Colleague, Run, DeliveryPolicy,
    ArxivCategoryDB, user_to_dict, colleague_to_dict, paper_view_to_dict,
    policy_to_dict, arxiv_category_to_dict
)
from .json_store import (
    DEFAULT_DATA_DIR, load_json, save_json,
    JSONStoreError
)

logger = logging.getLogger(__name__)

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
            return {
                "researcher_name": user.get("name"),
                "affiliation": user.get("affiliation"),
                "research_topics": user.get("research_topics", []),
                "my_papers": user.get("my_papers", []),
                "preferred_venues": user.get("preferred_venues", []),
                "avoid_topics": user.get("avoid_topics", []),
                "time_budget_per_week_minutes": user.get("time_budget_per_week_minutes", 120),
                "arxiv_categories_include": user.get("arxiv_categories_include", []),
                "arxiv_categories_exclude": user.get("arxiv_categories_exclude", []),
                "stop_policy": user.get("stop_policy", {}),
                # Additional fields from local file
                **{k: v for k, v in _load_local_research_profile().items() 
                   if k not in ["researcher_name", "affiliation", "research_topics", 
                                "my_papers", "preferred_venues", "avoid_topics",
                                "time_budget_per_week_minutes", "arxiv_categories_include",
                                "arxiv_categories_exclude", "stop_policy"]}
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

def _load_local_delivery_policy() -> Dict[str, Any]:
    """Load delivery policy from local JSON file."""
    try:
        return load_json(_get_data_dir() / "delivery_policy.json")
    except Exception:
        return {}


def get_delivery_policy() -> Dict[str, Any]:
    """
    Get the delivery policy configuration.
    
    Priority: Database > Local JSON file
    
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
    return _load_local_delivery_policy()


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
                    
                    if not paper:
                        paper = Paper(
                            source="arxiv",
                            external_id=paper_id,
                            title=paper_record.get("title", ""),
                            abstract=paper_record.get("abstract"),
                            authors=paper_record.get("authors", []),
                            categories=paper_record.get("categories", []),
                            url=paper_record.get("arxiv_url"),
                            pdf_url=paper_record.get("pdf_url"),
                        )
                        db.add(paper)
                        db.flush()
                    
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
                        )
                        db.add(view)
                    
                    db.commit()
                    logger.debug(f"Upserted paper {paper_id} to database")
        except Exception as e:
            logger.error(f"Error upserting paper to DB: {e}")
    
    # Also update local file for development
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
