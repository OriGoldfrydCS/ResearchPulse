"""
Dashboard API Routes for ResearchPulse.

Provides REST API endpoints for the dashboard functionality:
- Papers: List, search, view, delete, mark unseen
- Emails: View history
- Calendar: View events
- Shares: View sharing history
- Colleagues: CRUD operations
- Runs: View history, trigger new runs
- Policies: View/update delivery policies
- Health: System health checks
"""

from __future__ import annotations

import os
import json
import csv
import io
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List, Dict, Any
from uuid import UUID

if TYPE_CHECKING:
    from ..tools.live_document import LiveDocumentData

# Load .env file if present (must be done before any os.getenv calls)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from fastapi import APIRouter, HTTPException, Query, Response, BackgroundTasks, Body
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError

from ..db.store import get_default_store
from ..db.database import check_connection, is_database_configured

logger = logging.getLogger(__name__)


# =============================================================================
# Router Setup
# =============================================================================

router = APIRouter(prefix="/api", tags=["dashboard"])


# =============================================================================
# LLM Inference Helper for Colleague Research Interests
# =============================================================================

async def infer_research_keywords_categories(research_interests: str) -> Dict[str, Any]:
    """
    Use LLM to infer keywords and arXiv categories from research interests text.
    
    Returns:
        dict with 'keywords' (list of strings) and 'categories' (list of arXiv category codes)
    """
    if not research_interests or not research_interests.strip():
        return {"keywords": [], "categories": []}
    
    try:
        import openai
        
        # Use project's LLM configuration
        api_key = os.getenv("LLM_API_KEY", "")
        api_base = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
        model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
        
        if not api_key:
            # Return empty if no API key configured
            return {"keywords": [], "categories": []}
        
        client = openai.AsyncOpenAI(api_key=api_key, base_url=api_base)
        
        prompt = f"""Analyze the following research interests and extract:
1. Keywords: A list of 5-10 specific research keywords/terms that best describe these interests
2. Categories: A list of arXiv category codes that match these interests

Research Interests:
---
{research_interests}
---

Common arXiv category codes include:
- cs.AI (Artificial Intelligence), cs.LG (Machine Learning), cs.CL (Computation and Language)
- cs.CV (Computer Vision), cs.NE (Neural and Evolutionary Computing), cs.RO (Robotics)
- stat.ML (Statistics - Machine Learning), math.OC (Optimization and Control)
- q-bio.QM (Quantitative Methods), q-bio.NC (Neurons and Cognition)
- eess.SP (Signal Processing), eess.SY (Systems and Control)
- physics.* (Various physics categories), math.* (Various math categories)

Response format (JSON only - no other text):
{{
    "keywords": ["keyword1", "keyword2", ...],
    "categories": ["cs.AI", "cs.LG", ...]
}}

Be specific and relevant. Focus on academic/research terminology. Return ONLY valid JSON."""

        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,  # Reasoning models need more tokens for thinking + output
        )
        
        content = response.choices[0].message.content or ""
        
        if not content.strip():
            return {"keywords": [], "categories": []}
        
        # Try to extract JSON from the response
        import re
        # Look for JSON object in the response (handle nested arrays)
        json_match = re.search(r'\{[\s\S]*"keywords"[\s\S]*"categories"[\s\S]*\}', content)
        if json_match:
            content = json_match.group(0)
        
        result = json.loads(content)
        
        return {
            "keywords": result.get("keywords", []),
            "categories": result.get("categories", [])
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Return empty on error - don't block colleague creation
        return {"keywords": [], "categories": []}


async def generate_interest_headline(research_interests: str) -> str:
    """
    Generate a concise interest headline from research interests text.
    
    This creates a short, readable summary like "Computer Vision, Medical Imaging, and RAG"
    for display in the colleague list UI.
    
    Args:
        research_interests: Raw research interests text from user
        
    Returns:
        Short headline string (max ~60 chars) or empty string if generation fails
    """
    if not research_interests or not research_interests.strip():
        return ""
    
    try:
        import openai
        
        api_key = os.getenv("LLM_API_KEY", "")
        api_base = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
        model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
        
        if not api_key:
            # Fallback: extract first few words from the interests
            words = research_interests.split()[:8]
            return " ".join(words)[:60] + ("..." if len(research_interests) > 60 else "")
        
        client = openai.AsyncOpenAI(api_key=api_key, base_url=api_base)
        
        prompt = f"""Create a very short headline (max 60 characters) summarizing these research interests.
Use format like: "Topic1, Topic2, and Topic3" or "Topic1 & Topic2"

Research interests:
---
{research_interests}
---

Rules:
- Maximum 60 characters
- 2-4 key topics only
- Use common abbreviations for brevity (ML, NLP, CV, etc.)
- No periods or quotes in output
- Be concise and professional

Output ONLY the headline text, nothing else:"""

        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )
        
        headline = (response.choices[0].message.content or "").strip()
        
        # Clean up: remove quotes, periods, etc.
        headline = headline.strip('"\'.')
        
        # Ensure max length
        if len(headline) > 80:
            headline = headline[:77] + "..."
            
        return headline
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Fallback: extract first few words
        words = research_interests.split()[:6]
        fallback = " ".join(words)
        return fallback[:60] + ("..." if len(fallback) > 60 else "")


# =============================================================================
# ArXiv Categories Endpoint  
# =============================================================================

@router.get("/arxiv-categories")
async def get_arxiv_categories(
    refresh: bool = Query(False, description="Force refresh from arXiv"),
    group: Optional[str] = Query(None, description="Filter by group (e.g., 'Computer Science')"),
    search: Optional[str] = Query(None, description="Search categories by name")
):
    """
    Get all available arXiv categories for selection.
    
    Categories are dynamically fetched from arXiv and cached locally.
    Use refresh=true to force a refresh from the official arXiv taxonomy.
    """
    try:
        from ..tools.arxiv_categories import (
            get_all_categories_json,
            get_categories_by_group,
            search_categories,
            refresh_taxonomy,
            get_category_groups
        )
        
        # Force refresh if requested
        if refresh:
            await refresh_taxonomy(force=True)
        
        # Search mode
        if search:
            results = search_categories(search, limit=50)
            return {
                "categories": [
                    {"code": cat.code, "name": cat.name, "group": cat.group}
                    for cat in results
                ],
                "search_query": search
            }
        
        # Filter by group
        if group:
            results = get_categories_by_group(group)
            return {
                "categories": [
                    {"code": cat.code, "name": cat.name, "group": cat.group}
                    for cat in results
                ],
                "group": group
            }
        
        # Return all categories
        result = get_all_categories_json()
        return {
            "categories": result["categories"],
            "total": result["total"],
            "groups": result["groups"]
        }
        
    except Exception as e:
        # Fallback to basic categories if dynamic fetch fails
        return {
            "categories": [
                {"code": "cs.AI", "name": "Artificial Intelligence"},
                {"code": "cs.CL", "name": "Computation and Language"},
                {"code": "cs.CV", "name": "Computer Vision and Pattern Recognition"},
                {"code": "cs.LG", "name": "Machine Learning"},
                {"code": "cs.IR", "name": "Information Retrieval"},
                {"code": "stat.ML", "name": "Machine Learning (Statistics)"},
            ],
            "error": str(e),
            "fallback": True
        }


@router.get("/arxiv-categories/topic/{topic}")
async def get_categories_for_topic(topic: str):
    """
    Get arXiv categories relevant to a research topic.
    
    This uses intelligent mapping to suggest categories based on the topic.
    If categories are not found locally, it will fetch from arXiv.
    """
    try:
        from ..tools.arxiv_categories import topic_to_categories_json
        return topic_to_categories_json(topic)
    except Exception as e:
        return {"topic": topic, "categories": [], "error": str(e)}


@router.post("/arxiv-categories/refresh")
async def refresh_arxiv_categories():
    """Force refresh of arXiv categories from the official taxonomy."""
    try:
        from ..tools.arxiv_categories import refresh_taxonomy
        count = await refresh_taxonomy(force=True)
        return {"success": True, "categories_count": count}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# Data Migration Endpoints
# =============================================================================

@router.get("/data-source")
async def get_data_source():
    """Check current data source (database or local files)."""
    from ..db import is_db_available
    db_available = is_db_available()
    return {
        "database_available": db_available,
        "primary_source": "database" if db_available else "local_files",
        "local_files_exist": _check_local_files_exist(),
    }


def _check_local_files_exist() -> dict:
    """Check which local data files exist."""
    from pathlib import Path
    data_dir = Path(__file__).parent.parent.parent / "data"
    files = [
        "research_profile.json",
        "colleagues.json", 
        "delivery_policy.json",
        "papers_state.json",
        "arxiv_categories.json",
    ]
    return {f: (data_dir / f).exists() for f in files}


@router.post("/migrate-to-db")
async def migrate_data_to_db():
    """
    Migrate all local JSON data files to database.
    
    This copies data from:
    - data/research_profile.json → users table
    - data/colleagues.json → colleagues table
    - data/delivery_policy.json → delivery_policies table
    - data/papers_state.json → papers + paper_views tables
    - data/arxiv_categories.json → arxiv_categories table
    
    The local files are NOT deleted - use DELETE /api/local-data for that.
    """
    from ..db import migrate_all_to_db, is_db_available
    
    if not is_db_available():
        return {
            "success": False,
            "error": "Database not available. Set DATABASE_URL environment variable.",
        }
    
    try:
        results = migrate_all_to_db()
        return {
            "success": True,
            "migrated": results,
            "message": "Data migrated successfully. Local files still exist - use DELETE /api/local-data to remove them.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.delete("/local-data")
async def delete_local_data():
    """
    Delete local JSON data files after confirming data is in database.
    
    WARNING: This permanently deletes local data files!
    Only use after successful migration to database.
    """
    from ..db import delete_local_data_files, is_db_available
    
    if not is_db_available():
        return {
            "success": False,
            "error": "Database not available. Cannot delete local files without DB backup.",
        }
    
    try:
        deleted = delete_local_data_files()
        return {
            "success": True,
            "deleted_files": deleted,
            "message": f"Deleted {len(deleted)} local data files. Data is now served from database only.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# Request/Response Models
# =============================================================================

class ColleagueCreate(BaseModel):
    """
    Model for creating a colleague.
    
    State model:
    - name, email, interests: User-provided, editable
    - categories, keywords: DERIVED from interests (not directly editable)
    - enabled: Whether colleague is active
    - auto_send_emails: Whether to auto-send research emails
    - added_by: 'manual' (owner) or 'email' (self-signup)
    """
    name: str
    email: str
    affiliation: Optional[str] = None
    # Interests text - categories and keywords are derived from this
    research_interests: Optional[str] = None
    interests: Optional[str] = None  # Alias for research_interests
    # These are deprecated in UI - derived automatically from interests
    keywords: List[str] = Field(default_factory=list, deprecated=True)
    categories: List[str] = Field(default_factory=list, deprecated=True)
    topics: List[str] = Field(default_factory=list)
    sharing_preference: str = "weekly"
    enabled: bool = True
    auto_send_emails: bool = True
    notes: Optional[str] = None


class ColleagueUpdate(BaseModel):
    """
    Model for updating a colleague.
    
    When interests are updated, categories and keywords are re-derived.
    Categories cannot be directly edited.
    """
    name: Optional[str] = None
    # Email is read-only - cannot be changed after creation
    affiliation: Optional[str] = None
    research_interests: Optional[str] = None
    interests: Optional[str] = None  # Alias for research_interests
    # These are deprecated - derived from interests
    keywords: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    sharing_preference: Optional[str] = None
    enabled: Optional[bool] = None
    auto_send_emails: Optional[bool] = None
    notes: Optional[str] = None


class PaperViewUpdate(BaseModel):
    notes: Optional[str] = None
    tags: Optional[List[str]] = None
    importance: Optional[str] = None
    decision: Optional[str] = None


class PolicyUpdate(BaseModel):
    policy_json: Dict[str, Any]
    colleague_share_enabled: Optional[bool] = None
    colleague_share_min_score: Optional[float] = None
    digest_mode: Optional[bool] = None


class RunTrigger(BaseModel):
    prompt: Optional[str] = None
    categories: Optional[List[str]] = None


class ProfileUpdate(BaseModel):
    """Researcher profile update model."""
    researcher_name: Optional[str] = None
    email: Optional[str] = None
    affiliation: Optional[str] = None
    research_topics: Optional[List[str]] = None
    arxiv_categories_include: Optional[List[str]] = None
    arxiv_categories_exclude: Optional[List[str]] = None
    interests_include: Optional[str] = None  # Free text research interests
    interests_exclude: Optional[str] = None  # Free text topics to exclude
    preferred_time_period: Optional[str] = None  # Preferred search time period
    keywords_include: Optional[List[str]] = None  # Deprecated
    keywords_exclude: Optional[List[str]] = None  # Deprecated
    colleague_ids_to_always_share: Optional[List[str]] = None


class HealthResponse(BaseModel):
    status: str
    database: Dict[str, Any]
    pinecone: Dict[str, Any]
    email: Dict[str, Any]
    timestamp: str


# =============================================================================
# Helper: derive research_topics list from interests_include free text
# =============================================================================

def _parse_interests_to_topics(interests_text: str) -> List[str]:
    """Parse the free-text interests_include into a structured research_topics list.
    
    Splits on commas and newlines, strips whitespace, and removes empty entries.
    This ensures research_topics always mirrors what the user typed in
    'Research Interests'.
    """
    if not interests_text or not interests_text.strip():
        return []
    import re
    # Split on commas or newlines
    parts = re.split(r'[,\n]+', interests_text)
    topics = []
    for part in parts:
        t = part.strip()
        if t and t not in topics:
            topics.append(t)
    return topics


# =============================================================================
# User Endpoint
# =============================================================================

@router.get("/user")
async def get_current_user():
    """Get or create the current user (single-user mode)."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        return user
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/user")
async def update_user(data: Dict[str, Any]):
    """Update the current user profile."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        updated = store.update_user(user_id, data)
        return updated
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Research Profile Endpoints (stored in database)
# =============================================================================

@router.get("/profile")
async def get_research_profile():
    """Get the researcher's profile with research areas, topics, and preferences."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        
        # Map database fields to profile response format
        return {
            "researcher_name": user.get("name", ""),
            "email": user.get("email", ""),
            "affiliation": user.get("affiliation", ""),
            "research_topics": user.get("research_topics", []),
            "arxiv_categories_include": user.get("arxiv_categories_include", []),
            "arxiv_categories_exclude": user.get("arxiv_categories_exclude", []),
            "interests_include": user.get("interests_include", ""),
            "interests_exclude": user.get("interests_exclude", ""),
            "preferred_time_period": user.get("preferred_time_period", "last two weeks"),
            "keywords_include": user.get("keywords_include", []),
            "keywords_exclude": user.get("keywords_exclude", []),
            "colleague_ids_to_always_share": []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/profile")
async def update_research_profile(data: ProfileUpdate, background_tasks: BackgroundTasks):
    """Update the researcher's profile (stored in database)."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        # Map profile fields to user fields
        update_dict = {}
        if data.researcher_name is not None:
            update_dict["name"] = data.researcher_name
        if data.email is not None:
            update_dict["email"] = data.email
        if data.affiliation is not None:
            update_dict["affiliation"] = data.affiliation
        if data.research_topics is not None:
            update_dict["research_topics"] = data.research_topics
        if data.arxiv_categories_include is not None:
            update_dict["arxiv_categories_include"] = data.arxiv_categories_include
        if data.arxiv_categories_exclude is not None:
            update_dict["arxiv_categories_exclude"] = data.arxiv_categories_exclude
        if data.interests_include is not None:
            update_dict["interests_include"] = data.interests_include
            # Auto-derive research_topics from interests_include (single source of truth)
            # Only override if research_topics was NOT explicitly provided in this request
            if data.research_topics is None:
                update_dict["research_topics"] = _parse_interests_to_topics(data.interests_include)
        if data.interests_exclude is not None:
            update_dict["interests_exclude"] = data.interests_exclude
        if data.preferred_time_period is not None:
            update_dict["preferred_time_period"] = data.preferred_time_period
        if data.keywords_include is not None:
            update_dict["keywords_include"] = data.keywords_include
        if data.keywords_exclude is not None:
            update_dict["keywords_exclude"] = data.keywords_exclude
        
        # Detect whether interest-related fields changed (triggers paper re-scoring)
        _INTEREST_KEYS = {
            "research_topics", "arxiv_categories_include", "arxiv_categories_exclude",
            "interests_include", "interests_exclude", "avoid_topics",
        }
        interests_changed = bool(set(update_dict) & _INTEREST_KEYS)

        # Update user in database
        updated_user = store.update_user(user_id, update_dict)

        # Re-score existing papers in the background when interests change
        if interests_changed:
            from ..tools.rescore_papers import rescore_papers_for_user
            background_tasks.add_task(rescore_papers_for_user, user_id)

        # Return in profile format
        return {
            "researcher_name": updated_user.get("name", ""),
            "email": updated_user.get("email", ""),
            "affiliation": updated_user.get("affiliation", ""),
            "research_topics": updated_user.get("research_topics", []),
            "arxiv_categories_include": updated_user.get("arxiv_categories_include", []),
            "arxiv_categories_exclude": updated_user.get("arxiv_categories_exclude", []),
            "interests_include": updated_user.get("interests_include", ""),
            "interests_exclude": updated_user.get("interests_exclude", ""),
            "preferred_time_period": updated_user.get("preferred_time_period", "last two weeks"),
            "keywords_include": updated_user.get("keywords_include", []),
            "keywords_exclude": updated_user.get("keywords_exclude", []),
            "colleague_ids_to_always_share": []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Paper Endpoints
# =============================================================================

@router.get("/papers")
async def list_papers(
    seen: Optional[bool] = Query(None, description="Filter by seen status"),
    decision: Optional[str] = Query(None, description="Filter by decision (logged, emailed, scheduled, etc.)"),
    importance: Optional[str] = Query(None, description="Filter by importance (low, medium, high)"),
    category: Optional[str] = Query(None, description="Filter by arXiv category"),
    q: Optional[str] = Query(None, description="Search query for title/abstract"),
    limit: int = Query(50, le=200, description="Maximum results"),
    offset: int = Query(0, description="Offset for pagination"),
    sort_by: str = Query("added_at", description="Sort field (added_at, published_at, importance, title)"),
    sort_dir: str = Query("desc", description="Sort direction (asc, desc)"),
    is_starred: Optional[bool] = Query(None, description="Filter by starred status"),
    is_read: Optional[bool] = Query(None, description="Filter by read status"),
    relevance_state: Optional[str] = Query(None, description="Filter by relevance state (relevant, not_relevant)"),
    include_not_relevant: bool = Query(False, description="Include papers marked as not relevant"),
):
    """List papers with filters and sorting."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        papers = store.list_papers(
            user_id=user_id,
            seen=seen,
            decision=decision,
            importance=importance,
            category=category,
            query=q,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_dir=sort_dir,
            is_starred=is_starred,
            is_read=is_read,
            relevance_state=relevance_state,
            include_not_relevant=include_not_relevant,
        )
        return {"papers": papers, "count": len(papers), "offset": offset, "limit": limit, "sort_by": sort_by, "sort_dir": sort_dir}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/papers/{paper_id}")
async def get_paper(paper_id: UUID):
    """Get a single paper with view details."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        paper_uuid = paper_id
        
        view = store.get_paper_view(user_id, paper_uuid)
        if not view:
            paper = store.get_paper(paper_uuid)
            if paper:
                return {"paper": paper, "view": None}
            raise HTTPException(status_code=404, detail="Paper not found")
        return view
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid paper ID")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/papers/{paper_id}")
async def delete_paper(
    paper_id: UUID,
    delete_vectors: bool = Query(True, description="Also delete Pinecone vectors"),
):
    """Delete a paper view and optionally Pinecone vectors."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        paper_uuid = paper_id
        
        # Delete the paper view
        deleted = store.delete_paper_view(user_id, paper_uuid)
        logger.info(f"Paper {paper_uuid} deleted for user {user_id}: {deleted} (live document will exclude on next fetch)")
        
        # Delete Pinecone vectors if requested
        if delete_vectors:
            try:
                from ..rag.vector_store import delete_paper_from_vector_store
                vector_id = f"paper:{paper_id}"
                delete_paper_from_vector_store(vector_id)
            except Exception:
                # Pinecone delete failed, but view is deleted
                pass
        
        return {"deleted": deleted, "paper_id": paper_id}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid paper ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/papers/{paper_id}/mark-unseen")
async def mark_paper_unseen(paper_id: UUID):
    """Mark a paper as unseen (delete view record)."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        paper_uuid = paper_id
        
        deleted = store.delete_paper_view(user_id, paper_uuid)
        return {"success": deleted, "paper_id": paper_id}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid paper ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/papers/{paper_id}")
async def update_paper_view(paper_id: UUID, data: PaperViewUpdate):
    """Update paper view (notes, tags, importance, decision)."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        paper_uuid = paper_id
        
        update_data = data.model_dump(exclude_none=True)
        updated = store.update_paper_view(user_id, paper_uuid, update_data)
        return updated
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Paper Summarisation Endpoints
# =============================================================================

@router.post("/papers/{paper_id}/summarize")
async def summarize_paper_endpoint(paper_id: UUID, force: bool = Query(False, description="Regenerate even if cached")):
    """
    Generate an AI summary for a paper by downloading its PDF and using OpenAI.
    
    - Returns cached summary if one exists (unless force=True).
    - Summary is persisted in the paper_views table.
    """
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])

        # Get the paper view with paper data
        view = store.get_paper_view(user_id, paper_id)
        if not view:
            raise HTTPException(status_code=404, detail="Paper not found")

        paper_view_id = UUID(view["id"])
        paper_data = view.get("paper", {})
        external_id = paper_data.get("external_id", "")
        pdf_url = paper_data.get("pdf_url") or (f"https://arxiv.org/pdf/{external_id}.pdf" if external_id else "")
        title = paper_data.get("title", "")

        if not pdf_url:
            raise HTTPException(status_code=400, detail="No PDF URL available for this paper")

        from ..tools.summarize_paper import summarize_paper
        result = summarize_paper(
            paper_view_id=paper_view_id,
            pdf_url=pdf_url,
            title=title,
            force=force,
        )

        if result["success"]:
            return {
                "success": True,
                "summary": result["summary"],
                "cached": result.get("cached", False),
                "generated_at": result.get("generated_at"),
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Summarisation failed"))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/papers/{paper_id}/summary")
async def get_paper_summary(paper_id: UUID):
    """
    Get the stored summary for a paper (if any).
    
    Returns the cached summary without generating a new one.
    """
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])

        view = store.get_paper_view(user_id, paper_id)
        if not view:
            raise HTTPException(status_code=404, detail="Paper not found")

        summary = view.get("summary")
        generated_at = view.get("summary_generated_at")

        return {
            "success": True,
            "has_summary": summary is not None,
            "summary": summary,
            "generated_at": generated_at,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Paper Toggle Endpoints (Single Item)
# =============================================================================

@router.post("/papers/{paper_id}/toggle-star")
async def toggle_star(paper_id: UUID):
    """Toggle starred status for a paper."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        paper_uuid = paper_id
        
        updated = store.toggle_star(user_id, paper_uuid)
        return updated
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/papers/{paper_id}/toggle-read")
async def toggle_read(paper_id: UUID):
    """Toggle read status for a paper."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        paper_uuid = paper_id
        
        updated = store.toggle_read(user_id, paper_uuid)
        return updated
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/papers/{paper_id}/mark-read")
async def mark_read(paper_id: UUID):
    """Mark a paper as read (idempotent)."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        paper_uuid = paper_id
        
        updated = store.mark_read(user_id, paper_uuid)
        return updated
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/papers/{paper_id}/toggle-relevance")
async def toggle_relevance(paper_id: UUID, note: Optional[str] = None):
    """Toggle relevance state for a paper (relevant <-> not_relevant)."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        paper_uuid = paper_id
        
        updated = store.toggle_relevance(user_id, paper_uuid, note=note)
        return updated
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SetRelevanceRequest(BaseModel):
    relevance_state: str  # 'relevant' or 'not_relevant'
    note: Optional[str] = None


@router.post("/papers/{paper_id}/set-relevance")
async def set_relevance(paper_id: UUID, data: SetRelevanceRequest):
    """Set explicit relevance state for a paper."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        paper_uuid = paper_id
        
        updated = store.set_relevance(user_id, paper_uuid, data.relevance_state, data.note)
        return updated
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/papers/{paper_id}/feedback-history")
async def get_paper_feedback_history(paper_id: UUID, limit: int = Query(50, le=200)):
    """Get feedback history for a specific paper."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        paper_uuid = paper_id
        
        history = store.get_feedback_history(user_id, paper_id=paper_uuid, limit=limit)
        return {"history": history, "count": len(history)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/papers/export/csv")
async def export_papers_csv(
    decision: Optional[str] = Query(None),
    importance: Optional[str] = Query(None),
):
    """Export papers to CSV."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        papers = store.list_papers(
            user_id=user_id,
            decision=decision,
            importance=importance,
            limit=1000,
            offset=0,
        )
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Title", "Authors", "Categories", "URL", "Decision", "Importance", "First Seen", "Tags"])
        
        for view in papers:
            paper = view.get("paper", {})
            writer.writerow([
                paper.get("title", ""),
                ", ".join(paper.get("authors", [])),
                ", ".join(paper.get("categories", [])),
                paper.get("url", ""),
                view.get("decision", ""),
                view.get("importance", ""),
                view.get("first_seen_at", ""),
                ", ".join(view.get("tags", [])),
            ])
        
        output.seek(0)
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=papers.csv"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Email Endpoints
# =============================================================================

@router.get("/emails")
async def list_emails(
    status: Optional[str] = Query(None, description="Filter by status (queued, sent, failed)"),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
):
    """List email history."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        emails = store.list_emails(user_id, status=status, limit=limit, offset=offset)
        return {"emails": emails, "count": len(emails)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Calendar Endpoints
# =============================================================================

@router.get("/calendar")
async def list_calendar_events(
    limit: int = Query(50, le=200),
    offset: int = Query(0),
):
    """List calendar events."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        events = store.list_calendar_events(user_id, limit=limit, offset=offset)
        return {"events": events, "count": len(events)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calendar/{event_id}")
async def get_calendar_event(event_id: UUID):
    """Get a single calendar event with details."""
    try:
        store = get_default_store()
        event = store.get_calendar_event(event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Calendar event not found")
        
        # Include invite emails and reply history
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        invite_emails = store.list_calendar_invite_emails(user_id, calendar_event_id=event_id)
        reply_history = store.get_reply_history_for_event(event_id)
        
        return {
            "event": event,
            "invite_emails": invite_emails,
            "reply_history": reply_history,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RescheduleRequest(BaseModel):
    new_start_time: str  # ISO format datetime
    new_duration_minutes: Optional[int] = None
    reschedule_note: Optional[str] = None


@router.post("/calendar/{event_id}/reschedule")
async def reschedule_calendar_event_endpoint(event_id: UUID, data: RescheduleRequest):
    """
    Reschedule a calendar event to a new time.
    
    This endpoint:
    1. Updates the calendar event with new start time
    2. Increments the ICS sequence number
    3. Generates and sends a new calendar invite email with updated time
    4. Returns the updated event
    """
    try:
        from datetime import timedelta
        from ..tools.ics_generator import generate_reschedule_ics
        from ..tools.calendar_invite_sender import send_calendar_invite_email
        
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        user_email = user.get("email", "")
        user_name = user.get("name", "Researcher")
        
        # Get the existing event
        existing_event = store.get_calendar_event(event_id)
        if not existing_event:
            raise HTTPException(status_code=404, detail="Calendar event not found")
        
        # Parse the new start time
        try:
            new_start_time = datetime.fromisoformat(data.new_start_time.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid datetime format. Use ISO format (YYYY-MM-DDTHH:MM:SS)")
        
        # Determine new duration
        new_duration = data.new_duration_minutes or existing_event.get("duration_minutes", 30)
        
        # Build reschedule note
        reschedule_note = data.reschedule_note or "Rescheduled by user"
        if not reschedule_note.startswith("Rescheduled"):
            reschedule_note = f"Rescheduled: {reschedule_note}"
        
        # Update the calendar event
        updated_event = store.reschedule_calendar_event(
            event_id=event_id,
            new_start_time=new_start_time,
            reschedule_note=reschedule_note,
            new_duration_minutes=new_duration,
        )
        
        # Send updated calendar invite if user has email
        invite_email_record = None
        email_sent = False
        
        if user_email and user_email != "user@researchpulse.local":
            try:
                # Generate new reminder token for the reschedule
                import uuid as uuid_mod
                reminder_token = str(uuid_mod.uuid4())[:32]
                
                # Get ICS UID from existing event or generate new one
                ics_uid = existing_event.get("ics_uid")
                if not ics_uid:
                    from ..tools.ics_generator import generate_uid
                    ics_uid = generate_uid()
                
                sequence = (existing_event.get("sequence_number") or 0) + 1
                
                # Get paper details for ICS
                papers_data = []
                paper_ids = existing_event.get("paper_ids") or []
                for paper_id in paper_ids[:10]:  # Limit for performance
                    try:
                        paper_uuid = UUID(paper_id)
                        view = store.get_paper_view(user_id, paper_uuid)
                        if view and view.get("paper"):
                            paper = view["paper"]
                            external_id = paper.get("external_id") or ""
                            papers_data.append({
                                "title": paper.get("title") or "Untitled",
                                "url": f"https://arxiv.org/abs/{external_id}" if external_id else "",
                                "importance": view.get("importance") or "medium",
                            })
                    except Exception:
                        continue
                
                # Generate reschedule ICS
                ics_content = generate_reschedule_ics(
                    uid=ics_uid,
                    papers=papers_data,
                    new_start_time=new_start_time,
                    duration_minutes=new_duration,
                    organizer_email=os.getenv("SMTP_FROM_EMAIL", "researchpulse@example.com"),
                    attendee_email=user_email,
                    reminder_minutes=15,
                    sequence=sequence,
                    reschedule_note=reschedule_note,
                )
                
                # Generate email
                email_subject = f"📅 Updated: {existing_event.get('title', 'Reading Reminder')}"
                email_body_text = f"""Your reading reminder has been rescheduled.

📅 NEW TIME: {new_start_time.strftime('%A, %B %d, %Y at %I:%M %p')}
⏱️ Duration: {new_duration} minutes

Reason: {reschedule_note}

---
This calendar invite was updated by ResearchPulse.

[RP_REMINDER_ID: {reminder_token}]
"""
                
                # Send the email (returns 3 values: success, message_id, error)
                email_sent, actual_message_id, send_error = send_calendar_invite_email(
                    to_email=user_email,
                    to_name=user_name,
                    subject=email_subject,
                    body_text=email_body_text,
                    body_html=None,
                    ics_content=ics_content,
                    ics_method="REQUEST",
                )
                
                # Use actual message_id from SMTP if available
                message_id = actual_message_id or f"<{ics_uid}-{sequence}@researchpulse.local>"
                
                # Create email record
                email_record = store.create_email(
                    user_id=user_id,
                    paper_id=None,
                    recipient_email=user_email,
                    subject=email_subject,
                    body_text=email_body_text,
                    body_preview=f"Rescheduled to {new_start_time.strftime('%B %d at %I:%M %p')}",
                    triggered_by='user',
                    paper_ids=paper_ids,
                )
                
                email_id = UUID(email_record["id"])
                if email_sent:
                    store.update_email_status(email_id, status="sent")
                else:
                    store.update_email_status(email_id, status="failed", error=send_error)
                
                # Create CalendarInviteEmail record
                invite_email_record = store.create_calendar_invite_email(
                    calendar_event_id=event_id,
                    user_id=user_id,
                    message_id=message_id,
                    recipient_email=user_email,
                    subject=email_subject,
                    ics_uid=ics_uid,
                    email_id=email_id,
                    reminder_token=reminder_token,  # For reliable reply matching
                    ics_sequence=sequence,
                    ics_method="REQUEST",
                    triggered_by="user",
                )
                
            except Exception as e:
                print(f"Error sending reschedule invite: {e}")
        
        return {
            "success": True,
            "event": updated_event,
            "invite_email": invite_email_record,
            "email_sent": email_sent,
            "message": f"Event rescheduled to {new_start_time.strftime('%B %d, %Y at %I:%M %p')}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calendar/{event_id}/history")
async def get_calendar_event_history(event_id: UUID):
    """
    Get the rescheduling history for a calendar event.
    
    Returns all replies, invite emails, and changes made to the event.
    """
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        event = store.get_calendar_event(event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Calendar event not found")
        
        # Get all invite emails for this event
        invite_emails = store.list_calendar_invite_emails(user_id, calendar_event_id=event_id)
        
        # Get all replies related to this event
        reply_history = store.get_reply_history_for_event(event_id)
        
        # Build timeline
        timeline = []
        
        # Add event creation
        timeline.append({
            "type": "event_created",
            "timestamp": event.get("created_at"),
            "details": {
                "title": event.get("title"),
                "start_time": event.get("start_time"),
                "triggered_by": event.get("triggered_by"),
            }
        })
        
        # Add invite emails
        for invite in invite_emails:
            timeline.append({
                "type": "invite_sent",
                "timestamp": invite.get("sent_at") or invite.get("created_at"),
                "details": {
                    "subject": invite.get("subject"),
                    "recipient": invite.get("recipient_email"),
                    "ics_sequence": invite.get("ics_sequence"),
                    "ics_method": invite.get("ics_method"),
                }
            })
        
        # Add replies
        for reply in reply_history:
            timeline.append({
                "type": "reply_received",
                "timestamp": reply.get("received_at"),
                "details": {
                    "from_email": reply.get("from_email"),
                    "intent": reply.get("intent"),
                    "action_taken": reply.get("action_taken"),
                    "processed": reply.get("processed"),
                }
            })
        
        # Add reschedule note if present
        if event.get("reschedule_note"):
            timeline.append({
                "type": "rescheduled",
                "timestamp": event.get("updated_at"),
                "details": {
                    "note": event.get("reschedule_note"),
                    "new_start_time": event.get("start_time"),
                    "sequence_number": event.get("sequence_number"),
                }
            })
        
        # Sort timeline by timestamp
        timeline.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
        
        return {
            "event": event,
            "timeline": timeline,
            "invite_count": len(invite_emails),
            "reply_count": len(reply_history),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class IngestReplyRequest(BaseModel):
    message_id: str
    from_email: str
    subject: Optional[str] = None
    body_text: Optional[str] = None
    body_html: Optional[str] = None
    in_reply_to: Optional[str] = None
    references: Optional[str] = None


@router.post("/calendar/ingest-reply")
async def ingest_email_reply(data: IngestReplyRequest):
    """
    Ingest an incoming email reply for a calendar invite.
    
    This endpoint is called when an email reply is received (via webhook or polling).
    It stores the reply and marks it for processing.
    """
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        # Find the original calendar invite by in_reply_to or references
        original_invite = None
        
        if data.in_reply_to:
            # Try to find by message_id
            original_invite = store.get_calendar_invite_by_message_id(data.in_reply_to)
        
        if not original_invite and data.references:
            # Try to find by references (may contain multiple message IDs)
            for ref in data.references.split():
                ref = ref.strip('<>')
                original_invite = store.get_calendar_invite_by_message_id(f"<{ref}>")
                if original_invite:
                    break
        
        if not original_invite:
            return {
                "success": False,
                "error": "Could not find original calendar invite for this reply",
                "message_id": data.message_id,
            }
        
        # Create inbound reply record
        reply_record = store.create_inbound_email_reply(
            user_id=user_id,
            original_invite_id=UUID(original_invite["id"]),
            message_id=data.message_id,
            from_email=data.from_email,
            subject=data.subject,
            body_text=data.body_text,
            body_html=data.body_html,
            in_reply_to=data.in_reply_to,
            references=data.references,
        )
        
        return {
            "success": True,
            "reply": reply_record,
            "original_invite_id": original_invite["id"],
            "calendar_event_id": original_invite.get("calendar_event_id"),
            "message": "Reply ingested successfully. Call /api/calendar/process-replies to process."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calendar/process-replies")
async def process_email_replies():
    """
    Process all unprocessed email replies.
    
    This endpoint:
    1. Gets all unprocessed replies
    2. Parses each reply to extract intent (reschedule, accept, decline, etc.)
    3. Takes appropriate action (e.g., reschedule the event)
    4. Updates reply records with processing results
    """
    try:
        from ..agent.reply_parser import parse_reply, ReplyIntent
        from datetime import timedelta
        
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        user_email = user.get("email", "")
        user_name = user.get("name", "Researcher")
        
        # Get all unprocessed replies
        unprocessed = store.list_unprocessed_replies(user_id)
        
        if not unprocessed:
            return {
                "success": True,
                "processed_count": 0,
                "message": "No unprocessed replies found"
            }
        
        results = []
        
        for reply in unprocessed:
            reply_id = UUID(reply["id"])
            body_text = reply.get("body_text") or ""
            
            try:
                # Parse the reply to extract intent
                # Try LLM first, fall back to rules
                try:
                    parsed = parse_reply(body_text, use_llm=True)
                except Exception:
                    parsed = parse_reply(body_text, use_llm=False)
                
                intent = parsed.intent.value
                extracted_datetime = parsed.extracted_datetime
                extracted_datetime_text = parsed.extracted_datetime_text
                confidence_score = parsed.confidence
                
                action_taken = None
                new_event_id = None
                processing_error = None
                
                # Take action based on intent
                if parsed.intent == ReplyIntent.RESCHEDULE and extracted_datetime:
                    # Get the original invite and event
                    original_invite_id = reply.get("original_invite_id")
                    if original_invite_id:
                        invite = None
                        # Find the invite
                        invites = store.list_calendar_invite_emails(user_id)
                        for inv in invites:
                            if str(inv.get("id")) == str(original_invite_id):
                                invite = inv
                                break
                        
                        if invite:
                            event_id = UUID(invite["calendar_event_id"])
                            
                            # Reschedule the event
                            try:
                                reschedule_note = f"Rescheduled after user email reply: '{parsed.reason}'" if parsed.reason else "Rescheduled after user email reply"
                                
                                updated_event = store.reschedule_calendar_event(
                                    event_id=event_id,
                                    new_start_time=extracted_datetime,
                                    reschedule_note=reschedule_note,
                                )
                                
                                action_taken = f"rescheduled_to_{extracted_datetime.isoformat()}"
                                new_event_id = event_id
                                
                                # Send new calendar invite
                                if user_email and user_email != "user@researchpulse.local":
                                    from ..tools.ics_generator import generate_reschedule_ics
                                    from ..tools.calendar_invite_sender import send_calendar_invite_email
                                    
                                    event = store.get_calendar_event(event_id)
                                    ics_uid = event.get("ics_uid") or ""
                                    sequence = event.get("sequence_number") or 1
                                    duration = event.get("duration_minutes") or 30
                                    
                                    # Get papers
                                    papers_data = []
                                    paper_ids = event.get("paper_ids") or []
                                    for paper_id in paper_ids[:5]:
                                        try:
                                            paper_uuid = UUID(paper_id)
                                            view = store.get_paper_view(user_id, paper_uuid)
                                            if view and view.get("paper"):
                                                paper = view["paper"]
                                                external_id = paper.get("external_id") or ""
                                                papers_data.append({
                                                    "title": paper.get("title") or "Untitled",
                                                    "url": f"https://arxiv.org/abs/{external_id}" if external_id else "",
                                                    "importance": view.get("importance") or "medium",
                                                })
                                        except Exception:
                                            continue
                                    
                                    ics_content = generate_reschedule_ics(
                                        uid=ics_uid,
                                        papers=papers_data,
                                        new_start_time=extracted_datetime,
                                        duration_minutes=duration,
                                        organizer_email=os.getenv("SMTP_FROM_EMAIL", "researchpulse@example.com"),
                                        attendee_email=user_email,
                                        reminder_minutes=15,
                                        sequence=sequence,
                                        reschedule_note=reschedule_note,
                                    )
                                    
                                    email_subject = f"📅 Rescheduled: {event.get('title', 'Reading Reminder')}"
                                    email_body_text = f"""Your reading reminder has been rescheduled based on your reply.

📅 NEW TIME: {extracted_datetime.strftime('%A, %B %d, %Y at %I:%M %p')}

Your original message: "{body_text[:200]}..."

---
ResearchPulse automatically processed your reply and updated the calendar event.
"""
                                    
                                    send_calendar_invite_email(
                                        to_email=user_email,
                                        to_name=user_name,
                                        subject=email_subject,
                                        body_text=email_body_text,
                                        body_html=None,
                                        ics_content=ics_content,
                                        ics_method="REQUEST",
                                    )
                                
                            except Exception as e:
                                processing_error = str(e)
                                action_taken = "reschedule_failed"
                        else:
                            processing_error = "Original invite not found"
                            action_taken = "no_action_invite_not_found"
                    else:
                        processing_error = "No original invite ID in reply"
                        action_taken = "no_action"
                
                elif parsed.intent == ReplyIntent.ACCEPT:
                    action_taken = "noted_acceptance"
                elif parsed.intent == ReplyIntent.DECLINE:
                    action_taken = "noted_decline"
                elif parsed.intent == ReplyIntent.CANCEL:
                    action_taken = "noted_cancel_request"
                else:
                    action_taken = "no_action_required"
                
                # Update reply with processing results
                store.update_inbound_reply_processing(
                    reply_id=reply_id,
                    intent=intent,
                    extracted_datetime=extracted_datetime,
                    extracted_datetime_text=extracted_datetime_text,
                    confidence_score=confidence_score,
                    action_taken=action_taken,
                    new_event_id=new_event_id,
                    processing_result={"parsed": parsed.__dict__ if hasattr(parsed, '__dict__') else str(parsed)},
                    processing_error=processing_error,
                )
                
                results.append({
                    "reply_id": str(reply_id),
                    "intent": intent,
                    "action_taken": action_taken,
                    "success": processing_error is None,
                    "error": processing_error,
                })
                
            except Exception as e:
                # Mark as processed with error
                store.update_inbound_reply_processing(
                    reply_id=reply_id,
                    intent="error",
                    action_taken="processing_failed",
                    processing_error=str(e),
                )
                results.append({
                    "reply_id": str(reply_id),
                    "intent": "error",
                    "action_taken": "processing_failed",
                    "success": False,
                    "error": str(e),
                })
        
        successful = sum(1 for r in results if r["success"])
        
        return {
            "success": True,
            "processed_count": len(results),
            "successful_count": successful,
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calendar/poll-inbox")
async def poll_inbox_for_replies():
    """
    Poll the email inbox for replies to calendar invitations.
    
    This endpoint:
    1. Connects to the IMAP server (Gmail)
    2. Fetches recent emails that appear to be replies to calendar invites
    3. Matches replies to original calendar invite emails
    4. Ingests new replies into the database
    5. Processes each reply to extract intent and take action
    
    Note: Requires IMAP credentials (uses SMTP_USER and SMTP_PASSWORD from .env)
    """
    try:
        from ..tools.email_poller import fetch_recent_replies
        
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        user_email = user.get("email", "")
        
        # Fetch recent replies from inbox
        replies = fetch_recent_replies(since_hours=48)
        
        ingested_count = 0
        already_processed_count = 0
        no_match_count = 0
        errors = []
        
        for reply in replies:
            try:
                # Check if this reply was already processed
                existing = store.get_inbound_reply_by_message_id(reply["message_id"])
                if existing:
                    already_processed_count += 1
                    continue
                
                # Find the original calendar invite by message_id
                in_reply_to = reply.get("in_reply_to")
                if not in_reply_to:
                    no_match_count += 1
                    continue
                
                invite = store.get_calendar_invite_by_message_id(in_reply_to)
                if not invite:
                    # Try to find by partial match or thread_id
                    no_match_count += 1
                    continue
                
                # Ingest the reply
                store.create_inbound_email_reply(
                    user_id=user_id,
                    original_invite_id=UUID(invite["id"]),
                    message_id=reply["message_id"],
                    in_reply_to=in_reply_to,
                    from_email=reply["from_email"],
                    subject=reply.get("subject"),
                    body_text=reply.get("body_text"),
                )
                
                ingested_count += 1
                
            except Exception as e:
                errors.append(f"Error ingesting reply {reply.get('message_id', 'unknown')}: {str(e)}")
        
        # Now process any unprocessed replies
        process_result = None
        if ingested_count > 0:
            # Call the process-replies logic inline
            from ..agent.reply_parser import parse_reply, ReplyIntent
            
            unprocessed = store.list_unprocessed_replies(user_id)
            processed_results = []
            
            for reply_data in unprocessed:
                reply_id = UUID(reply_data["id"])
                body_text = reply_data.get("body_text") or ""
                
                try:
                    # Parse the reply
                    parsed = parse_reply(body_text, use_llm=False)  # Use rules for speed
                    
                    intent = parsed.intent.value
                    extracted_datetime = parsed.extracted_datetime
                    action_taken = None
                    new_event_id = None
                    
                    # Take action based on intent
                    if parsed.intent == ReplyIntent.RESCHEDULE and extracted_datetime:
                        original_invite_id = reply_data.get("original_invite_id")
                        if original_invite_id:
                            invites = store.list_calendar_invite_emails(user_id)
                            invite = None
                            for inv in invites:
                                if str(inv.get("id")) == str(original_invite_id):
                                    invite = inv
                                    break
                            
                            if invite:
                                event_id = UUID(invite["calendar_event_id"])
                                reschedule_note = "Rescheduled after user email reply"
                                
                                store.reschedule_calendar_event(
                                    event_id=event_id,
                                    new_start_time=extracted_datetime,
                                    reschedule_note=reschedule_note,
                                )
                                
                                action_taken = f"rescheduled_to_{extracted_datetime.isoformat()}"
                                new_event_id = event_id
                    
                    store.update_inbound_reply_processing(
                        reply_id=reply_id,
                        intent=intent,
                        extracted_datetime=extracted_datetime,
                        extracted_datetime_text=parsed.raw_datetime_text if hasattr(parsed, 'raw_datetime_text') else None,
                        confidence_score=parsed.confidence,
                        action_taken=action_taken or "no_action",
                        new_event_id=new_event_id,
                    )
                    
                    processed_results.append({
                        "reply_id": str(reply_id),
                        "intent": intent,
                        "action_taken": action_taken,
                        "success": True,
                    })
                    
                except Exception as e:
                    store.update_inbound_reply_processing(
                        reply_id=reply_id,
                        intent="error",
                        action_taken="processing_failed",
                        processing_error=str(e),
                    )
                    processed_results.append({
                        "reply_id": str(reply_id),
                        "intent": "error",
                        "success": False,
                        "error": str(e),
                    })
            
            process_result = {
                "processed_count": len(processed_results),
                "successful_count": sum(1 for r in processed_results if r["success"]),
                "results": processed_results,
            }
        
        return {
            "success": True,
            "emails_scanned": len(replies),
            "ingested_count": ingested_count,
            "already_processed_count": already_processed_count,
            "no_match_count": no_match_count,
            "errors": errors,
            "processing": process_result,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/colleagues/poll-signups")
async def poll_colleague_signups():
    """
    Poll the email inbox for colleague signup requests.
    
    This endpoint scans recent emails for messages where someone is requesting
    to receive research paper updates from ResearchPulse. When found, it:
    1. Extracts sender info (name, email)
    2. Uses LLM to extract research interests from email body
    3. Infers keywords and arXiv categories from interests
    4. Creates a new colleague with added_by='email'
    
    Keywords to detect signup intent: subscribe, sign me up, add me, 
    send me papers, interested in receiving, want to receive, etc.
    """
    try:
        from ..tools.email_poller import process_colleague_signups
        
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = str(user["id"])
        
        result = await process_colleague_signups(store, user_id)
        
        return {
            "success": True,
            **result
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Share Endpoints
# =============================================================================

@router.get("/shares")
async def list_shares(
    colleague_id: Optional[str] = Query(None, description="Filter by colleague"),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
):
    """List paper shares."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        colleague_uuid = UUID(colleague_id) if colleague_id else None
        shares = store.list_shares(user_id, colleague_id=colleague_uuid, limit=limit, offset=offset)
        return {"shares": shares, "count": len(shares)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/shares/{share_id}")
async def delete_share(share_id: UUID):
    """Delete a single share record."""
    try:
        store = get_default_store()
        deleted = store.delete_share(share_id)
        return {"deleted": deleted, "share_id": str(share_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk/shares/delete")
async def bulk_delete_shares(share_ids: List[str] = Body(embed=False)):
    """Delete multiple share records at once."""
    try:
        store = get_default_store()
        deleted_count = 0
        for sid in share_ids:
            try:
                share_uuid = UUID(sid)
                deleted = store.delete_share(share_uuid)
                if deleted:
                    deleted_count += 1
            except (ValueError, Exception):
                continue
        return {"deleted": deleted_count, "total": len(share_ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Colleague Endpoints
# =============================================================================

@router.get("/colleagues")
async def list_colleagues(
    enabled_only: bool = Query(False, description="Only show enabled colleagues"),
    sort_by: str = Query("name", description="Sort field: name, email, created_at"),
    sort_dir: str = Query("asc", description="Sort direction: asc or desc"),
):
    """List all colleagues with optional sorting."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        colleagues = store.list_colleagues(user_id, enabled_only=enabled_only)
        
        # Sort colleagues
        reverse = sort_dir.lower() == "desc"
        if sort_by == "name":
            colleagues.sort(key=lambda c: (c.get("name") or "").lower(), reverse=reverse)
        elif sort_by == "email":
            colleagues.sort(key=lambda c: (c.get("email") or "").lower(), reverse=reverse)
        elif sort_by == "created_at":
            colleagues.sort(key=lambda c: c.get("created_at") or "", reverse=reverse)
        
        return {"colleagues": colleagues, "count": len(colleagues)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/colleagues/{colleague_id}")
async def get_colleague(colleague_id: str):
    """Get a single colleague."""
    try:
        colleague_uuid = UUID(colleague_id)
        store = get_default_store()
        colleague = store.get_colleague(colleague_uuid)
        if not colleague:
            raise HTTPException(status_code=404, detail="Colleague not found")
        return colleague
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid colleague ID")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/colleagues")
async def create_colleague(data: ColleagueCreate):
    """
    Create a new colleague.
    
    Categories are derived from interests (not directly editable).
    If interests are provided, we:
    1. Use LLM to infer keywords and categories
    2. Generate an interest_headline for UI display
    """
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        colleague_data = data.model_dump()
        
        # Get the interests text (use research_interests or interests field)
        interests_text = colleague_data.get("research_interests") or colleague_data.get("interests") or ""
        
        # Categories are DERIVED from interests, never directly set by user
        # Clear any user-provided categories - they will be inferred
        colleague_data["categories"] = []
        colleague_data["keywords"] = []
        
        if interests_text and interests_text.strip():
            # Use LLM to infer keywords and categories from interests
            inferred = await infer_research_keywords_categories(interests_text)
            colleague_data["keywords"] = inferred.get("keywords", [])
            colleague_data["categories"] = inferred.get("categories", [])
            
            # Generate interest headline for UI display
            headline = await generate_interest_headline(interests_text)
            colleague_data["interest_headline"] = headline
            
            # Also store the interests text in dedicated field
            colleague_data["interests"] = interests_text
        
        # Ensure proper defaults
        colleague_data.setdefault("keywords", [])
        colleague_data.setdefault("categories", [])
        colleague_data.setdefault("topics", [])
        colleague_data.setdefault("sharing_preference", "weekly")
        colleague_data.setdefault("enabled", True)
        colleague_data.setdefault("added_by", "manual")  # Owner-added = 'manual'
        colleague_data.setdefault("auto_send_emails", True)
        
        colleague = store.create_colleague(user_id, colleague_data)
        return colleague
    except IntegrityError as e:
        raise HTTPException(status_code=400, detail="Colleague with this email already exists")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to save colleague: {str(e)}")


@router.put("/colleagues/{colleague_id}")
async def update_colleague(colleague_id: str, data: ColleagueUpdate):
    """
    Update a colleague.
    
    If interests are changed, categories are re-derived automatically.
    Categories cannot be directly edited - they are always derived from interests.
    """
    try:
        colleague_uuid = UUID(colleague_id)
        store = get_default_store()
        
        update_data = data.model_dump(exclude_none=True)
        
        # Check if interests are being updated
        interests_text = update_data.get("research_interests") or update_data.get("interests")
        
        if interests_text is not None:
            # If interests changed, re-derive categories and headline
            if interests_text and interests_text.strip():
                # Use LLM to infer new keywords and categories
                inferred = await infer_research_keywords_categories(interests_text)
                update_data["keywords"] = inferred.get("keywords", [])
                update_data["categories"] = inferred.get("categories", [])
                
                # Regenerate interest headline
                headline = await generate_interest_headline(interests_text)
                update_data["interest_headline"] = headline
                
                # Store in dedicated interests field too
                update_data["interests"] = interests_text
            else:
                # Interests cleared - clear derived fields too
                update_data["keywords"] = []
                update_data["categories"] = []
                update_data["interest_headline"] = None
                update_data["interests"] = None
        
        # Never allow direct category updates - remove if present
        # Categories are always derived from interests
        if "categories" in update_data and interests_text is None:
            # Only remove categories if we're not already processing interests
            # This preserves categories derived from interests
            pass
        
        colleague = store.update_colleague(colleague_uuid, update_data)
        return colleague
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/colleagues/{colleague_id}")
async def delete_colleague(colleague_id: str):
    """Delete a colleague."""
    try:
        colleague_uuid = UUID(colleague_id)
        store = get_default_store()
        deleted = store.delete_colleague(colleague_uuid)
        return {"deleted": deleted, "colleague_id": colleague_id}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid colleague ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk/colleagues/delete")
async def bulk_delete_colleagues(colleague_ids: List[str] = Body(embed=False)):
    """Delete multiple colleagues at once."""
    try:
        store = get_default_store()
        
        deleted_count = 0
        failed_ids = []
        for colleague_id in colleague_ids:
            try:
                colleague_uuid = UUID(colleague_id)
                deleted = store.delete_colleague(colleague_uuid)
                if deleted:
                    deleted_count += 1
                else:
                    failed_ids.append(colleague_id)
            except Exception as e:
                failed_ids.append(colleague_id)
        
        return {"deleted_count": deleted_count, "failed_ids": failed_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/colleagues/{colleague_id}/email-stats")
async def get_colleague_email_stats(colleague_id: str):
    """
    Get email activity statistics for a colleague.
    
    Returns counts for different time ranges:
    - last_24h: Emails sent in the last 24 hours
    - last_7d: Emails sent in the last 7 days
    - last_30d: Emails sent in the last 30 days
    - total: Total emails ever sent
    
    All data is stored in DB (colleague_email_log table).
    """
    try:
        from datetime import timedelta
        from sqlalchemy import func
        from ..db.orm_models import ColleagueEmailLog
        from ..db.database import get_db_session
        
        colleague_uuid = UUID(colleague_id)
        now = datetime.utcnow()
        
        with get_db_session() as session:
            # Base query
            base_query = session.query(func.count(ColleagueEmailLog.id)).filter(
                ColleagueEmailLog.colleague_id == colleague_uuid
            )
            
            # Count for each time range
            total = base_query.scalar() or 0
            last_24h = base_query.filter(
                ColleagueEmailLog.sent_at >= now - timedelta(hours=24)
            ).scalar() or 0
            last_7d = base_query.filter(
                ColleagueEmailLog.sent_at >= now - timedelta(days=7)
            ).scalar() or 0
            last_30d = base_query.filter(
                ColleagueEmailLog.sent_at >= now - timedelta(days=30)
            ).scalar() or 0
        
        return {
            "colleague_id": colleague_id,
            "stats": {
                "last_24h": last_24h,
                "last_7d": last_7d,
                "last_30d": last_30d,
                "total": total
            }
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid colleague ID")
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Return zeros if table doesn't exist yet or other error
        return {
            "colleague_id": colleague_id,
            "stats": {
                "last_24h": 0,
                "last_7d": 0,
                "last_30d": 0,
                "total": 0
            },
            "error": str(e)
        }


@router.get("/colleagues/{colleague_id}/email-history")
async def get_colleague_email_history(
    colleague_id: str,
    range: str = Query("all", description="Time range: 24h, 7d, 30d, or all"),
    limit: int = Query(20, le=100),
    offset: int = Query(0)
):
    """
    Get email history for a colleague.
    
    Returns a list of sent emails with subject, date, and snippet.
    """
    try:
        from datetime import timedelta
        from ..db.orm_models import ColleagueEmailLog
        from ..db.database import get_db_session
        
        colleague_uuid = UUID(colleague_id)
        now = datetime.utcnow()
        
        with get_db_session() as session:
            query = session.query(ColleagueEmailLog).filter(
                ColleagueEmailLog.colleague_id == colleague_uuid
            )
            
            # Apply time range filter
            if range == "24h":
                query = query.filter(ColleagueEmailLog.sent_at >= now - timedelta(hours=24))
            elif range == "7d":
                query = query.filter(ColleagueEmailLog.sent_at >= now - timedelta(days=7))
            elif range == "30d":
                query = query.filter(ColleagueEmailLog.sent_at >= now - timedelta(days=30))
            
            # Order by most recent first
            query = query.order_by(ColleagueEmailLog.sent_at.desc())
            
            # Apply pagination
            total = query.count()
            emails = query.offset(offset).limit(limit).all()
            
            return {
                "colleague_id": colleague_id,
                "range": range,
                "total": total,
                "emails": [
                    {
                        "id": str(e.id),
                        "subject": e.subject,
                        "snippet": e.snippet,
                        "email_type": e.email_type,
                        "sent_at": e.sent_at.isoformat() + "Z" if e.sent_at else None,
                        "paper_arxiv_id": e.paper_arxiv_id
                    }
                    for e in emails
                ]
            }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid colleague ID")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "colleague_id": colleague_id,
            "range": range,
            "total": 0,
            "emails": [],
            "error": str(e)
        }


# =============================================================================
# Run Endpoints
# =============================================================================

@router.get("/runs")
async def list_runs(
    limit: int = Query(50, le=200),
    offset: int = Query(0),
):
    """List run history."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        runs = store.list_runs(user_id, limit=limit, offset=offset)
        return {"runs": runs, "count": len(runs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    """Get a single run."""
    try:
        store = get_default_store()
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return run
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run")
async def trigger_run(data: RunTrigger, background_tasks: BackgroundTasks):
    """Trigger a new agent run."""
    try:
        from ..agent.react_agent import run_agent_episode
        import uuid as uuid_module
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        run_id = str(uuid_module.uuid4())
        prompt = data.prompt or "Process today's papers and make delivery decisions."
        
        # Create run record
        run = store.create_run(user_id, run_id, prompt)
        
        # Run agent in background thread (run_agent_episode is synchronous)
        def execute_run_sync():
            try:
                episode = run_agent_episode(
                    run_id=run_id,
                    user_message=prompt,
                )
                
                # Extract results from AgentEpisode
                papers_count = len(episode.papers_processed) if episode.papers_processed else 0
                decisions_count = len(episode.decisions_made) if episode.decisions_made else 0
                
                # Save papers to database for dashboard display
                for paper in episode.papers_processed:
                    try:
                        # First, upsert the paper itself
                        external_id = paper.get("arxiv_id") or paper.get("paper_id") or paper.get("id", "")
                        if external_id:
                            # Parse published date from ArxivPaper's 'published' field
                            _pub_raw = paper.get("published") or paper.get("published_at") or paper.get("updated")
                            _pub_at = None
                            if _pub_raw:
                                try:
                                    from datetime import datetime as _dt
                                    if isinstance(_pub_raw, str):
                                        if _pub_raw.endswith("Z"):
                                            _pub_raw = _pub_raw[:-1]
                                        _pub_at = _dt.fromisoformat(_pub_raw).replace(tzinfo=None)
                                    else:
                                        _pub_at = _pub_raw
                                except Exception:
                                    pass
                            paper_record = store.upsert_paper({
                                "source": "arxiv",
                                "external_id": external_id,
                                "title": paper.get("title", "Untitled"),
                                "abstract": paper.get("abstract", ""),
                                "authors": paper.get("authors", []),
                                "categories": paper.get("categories", []),
                                "url": paper.get("url"),
                                "pdf_url": paper.get("pdf_url"),
                                "published_at": _pub_at,
                            })
                            
                            # Then upsert the paper view for this user
                            paper_uuid = UUID(paper_record["id"])
                            store.upsert_paper_view(user_id, paper_uuid, {
                                "decision": paper.get("decision", "logged"),
                                "importance": paper.get("importance", "low"),
                                "relevance_score": paper.get("relevance_score") or paper.get("score"),
                                "novelty_score": paper.get("novelty_score"),
                                "heuristic_score": paper.get("heuristic_score"),
                            })
                    except Exception:
                        pass  # Continue on individual paper save errors
                
                # Save emails to database for dashboard display
                for action in episode.actions_taken:
                    if action.get("type") == "email":
                        try:
                            store.create_email(
                                user_id=user_id,
                                run_id=run_id,
                                paper_id=action.get("paper_id", ""),
                                recipient=action.get("recipient", ""),
                                subject=action.get("subject", ""),
                                body=action.get("body", ""),
                                status="sent",
                            )
                        except Exception:
                            pass
                    elif action.get("type") == "calendar":
                        try:
                            store.create_calendar_event(
                                user_id=user_id,
                                run_id=run_id,
                                paper_id=action.get("paper_id", ""),
                                title=action.get("title", ""),
                                start_time=action.get("start_time", ""),
                                end_time=action.get("end_time", ""),
                            )
                        except Exception:
                            pass
                
                store.update_run(run_id, {
                    "status": "done",
                    "report": episode.final_report or episode.model_dump(),
                    "papers_processed": papers_count,
                    "decisions_made": decisions_count,
                    "stop_reason": episode.stop_reason,
                })
                
                # Send digest email if digest_mode is enabled
                try:
                    from pathlib import Path
                    import json as json_module
                    from ..tools.decide_delivery import send_digest_email, ScoredPaper
                    
                    policy_path = Path("data/delivery_policy.json")
                    if policy_path.exists():
                        with open(policy_path) as f:
                            delivery_policy = json_module.load(f)
                        email_settings = delivery_policy.get("email_settings", {})
                        
                        if email_settings.get("digest_mode", False) and episode.papers_processed:
                            profile_path = Path("data/research_profile.json")
                            if profile_path.exists():
                                with open(profile_path) as f:
                                    profile = json_module.load(f)
                                
                                # Convert papers to ScoredPaper objects
                                scored_papers = []
                                for p in episode.papers_processed:
                                    if isinstance(p, dict):
                                        try:
                                            scored_papers.append(ScoredPaper(
                                                arxiv_id=p.get("arxiv_id", p.get("paper_id", p.get("id", ""))),
                                                title=p.get("title", "Untitled"),
                                                abstract=p.get("abstract", ""),
                                                authors=p.get("authors", []),
                                                categories=p.get("categories", []),
                                                relevance_score=p.get("relevance_score", p.get("score", 0.5)),
                                                novelty_score=p.get("novelty_score", 0.5),
                                                importance=p.get("importance", "medium"),
                                                explanation=p.get("explanation", ""),
                                                link=p.get("link") or p.get("url"),
                                            ))
                                        except Exception:
                                            pass  # Skip papers that can't be converted
                                
                                if scored_papers:
                                    send_digest_email(
                                        papers=scored_papers,
                                        researcher_email=profile.get("email", ""),
                                        researcher_name=profile.get("researcher_name", "Researcher"),
                                        email_settings=email_settings,
                                        query_text=prompt,
                                        triggered_by="agent",
                                    )
                except Exception as digest_err:
                    print(f"Warning: Could not send digest email: {digest_err}")
            except Exception as e:
                store.update_run(run_id, {
                    "status": "error",
                    "error_message": str(e),
                })
        
        # Run in thread pool to avoid blocking
        executor = ThreadPoolExecutor(max_workers=1)
        background_tasks.add_task(lambda: executor.submit(execute_run_sync).result())
        
        return {"run_id": run_id, "status": "started", "message": "Run triggered in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Policy Endpoints
# =============================================================================

@router.get("/policies")
async def get_policies():
    """Get delivery policies for the current user."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        policy = store.get_delivery_policy(user_id)
        if not policy:
            # Return default policy
            return {
                "policy_json": {},
                "colleague_share_enabled": True,
                "colleague_share_min_score": 0.5,
                "digest_mode": False,
            }
        return policy
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/policies")
async def update_policies(data: PolicyUpdate):
    """Update delivery policies."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        policy_data = data.model_dump(exclude_none=True)
        policy = store.upsert_delivery_policy(user_id, policy_data)
        return policy
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Health Endpoint
# =============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check for all services.
    
    Checks:
    - Database connection (Postgres/Supabase)
    - Pinecone connection (if configured)
    - Email provider (if configured)
    """
    health = {
        "status": "healthy",
        "database": {"status": "unknown", "message": ""},
        "pinecone": {"status": "unknown", "message": ""},
        "email": {"status": "unknown", "message": ""},
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    overall_healthy = True
    
    # Check database
    if is_database_configured():
        try:
            db_healthy, db_message = check_connection()
            health["database"] = {
                "status": "healthy" if db_healthy else "unhealthy",
                "message": db_message,
            }
            if not db_healthy:
                overall_healthy = False
        except Exception as e:
            health["database"] = {"status": "unhealthy", "message": str(e)}
            overall_healthy = False
    else:
        health["database"] = {"status": "local_storage", "message": "Using local JSON files (no database configured)"}
        # Local storage is OK for development
        # In production, this would be an error
        if os.getenv("ENV") == "production":
            overall_healthy = False
    
    # Check Pinecone
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if pinecone_key:
        try:
            from ..rag.vector_store import check_pinecone_health
            pc_healthy, pc_message = check_pinecone_health()
            health["pinecone"] = {
                "status": "healthy" if pc_healthy else "unhealthy",
                "message": pc_message,
            }
        except ImportError:
            health["pinecone"] = {"status": "not_configured", "message": "Vector store module not available"}
        except Exception as e:
            health["pinecone"] = {"status": "unhealthy", "message": str(e)}
    else:
        health["pinecone"] = {"status": "optional", "message": "Optional: Pinecone not configured (RAG features disabled)"}
    
    # Check email provider
    smtp_host = os.getenv("SMTP_HOST")
    resend_key = os.getenv("RESEND_API_KEY")
    sendgrid_key = os.getenv("SENDGRID_API_KEY")
    
    if smtp_host or resend_key or sendgrid_key:
        health["email"] = {"status": "configured", "message": "Email provider configured"}
    else:
        health["email"] = {"status": "optional", "message": "Optional: No email provider (emails simulated)"}
    
    health["status"] = "healthy" if overall_healthy else "unhealthy"
    
    return health


# =============================================================================
# Debug / Test Routes
# =============================================================================

@router.post("/debug/test-agent-actions")
async def test_agent_actions():
    """
    Debug route to test the agent's email/calendar creation pipeline.
    
    Creates a test email and calendar event as if the agent triggered them,
    to verify the end-to-end flow is working.
    """
    import logging
    from datetime import timedelta
    from uuid import uuid4
    
    logger = logging.getLogger(__name__)
    logger.info("[DEBUG] Testing agent action pipeline...")
    
    try:
        from ..db.data_service import save_artifact_to_db, is_db_available, get_or_create_default_user
        
        if not is_db_available():
            return {
                "success": False,
                "error": "Database not available",
                "message": "Cannot test without database connection"
            }
        
        user = get_or_create_default_user()
        if not user:
            return {
                "success": False,
                "error": "No user found",
                "message": "Cannot test without a user profile"
            }
        
        user_email = user.get("email", "")
        
        results = {
            "user_email": user_email,
            "email_result": None,
            "calendar_result": None,
        }
        
        # Try to get a real paper from the database for realistic testing
        sample_paper_id = None
        sample_paper_title = "Sample Research Paper"
        try:
            from ..db.postgres_store import PostgresStore
            store = PostgresStore()
            user_id = UUID(user.get("id"))
            papers = store.list_papers(user_id, limit=1)
            if papers:
                sample_paper = papers[0]
                # Get nested paper object for external_id (arXiv ID) and title
                nested_paper = sample_paper.get("paper", {})
                sample_paper_id = nested_paper.get("external_id")  # arXiv ID
                sample_paper_title = nested_paper.get("title") or sample_paper.get("title") or "Sample Paper"
                logger.info(f"[DEBUG] Using sample paper: {sample_paper_id} - {sample_paper_title[:50]}...")
        except Exception as e:
            logger.warning(f"[DEBUG] Could not fetch sample paper: {e}")
        
        # Test creating an email artifact
        test_email_content = f"""Subject: ResearchPulse: New paper - {sample_paper_title[:60]}

Dear Researcher,

ResearchPulse has found a paper that matches your interests:

Title: {sample_paper_title}
{f"arXiv ID: {sample_paper_id}" if sample_paper_id else ""}
Link: https://arxiv.org/abs/{sample_paper_id or "test"}

This email was sent automatically by ResearchPulse.

Generated at: {datetime.utcnow().isoformat()}"""
        
        email_result = save_artifact_to_db(
            file_type="email",
            file_path="/debug/test-email.txt",
            content=test_email_content,
            paper_id=sample_paper_id,
            description="Test email from debug route",
            triggered_by="agent",
        )
        results["email_result"] = email_result
        logger.info(f"[DEBUG] Email result: {email_result}")
        
        # Test creating a calendar event artifact
        test_start = datetime.utcnow() + timedelta(days=1)
        event_title = f"📖 Read: {sample_paper_title[:40]}..." if sample_paper_id else "ResearchPulse Test Reading"
        test_ics = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//ResearchPulse//Debug Test//EN
BEGIN:VEVENT
UID:test-{uuid4()}@researchpulse.local
DTSTART:{test_start.strftime('%Y%m%dT%H%M%S')}Z
DTEND:{(test_start + timedelta(minutes=30)).strftime('%Y%m%dT%H%M%S')}Z
SUMMARY:{event_title}
DESCRIPTION:Reading reminder from ResearchPulse.{f" Paper: {sample_paper_title}" if sample_paper_id else ""} Link: https://arxiv.org/abs/{sample_paper_id or "test"}
END:VEVENT
END:VCALENDAR"""
        
        calendar_result = save_artifact_to_db(
            file_type="calendar",
            file_path="/debug/test-event.ics",
            content=test_ics,
            paper_id=sample_paper_id,
            description="Test calendar from debug route",
            triggered_by="agent",
        )
        results["calendar_result"] = calendar_result
        logger.info(f"[DEBUG] Calendar result: {calendar_result}")
        
        success = (
            results["email_result"].get("success", False) and 
            results["calendar_result"].get("success", False)
        )
        
        return {
            "success": success,
            "message": "Test completed. Check Emails and Calendar tabs for items with 'Auto' badge.",
            "results": results,
        }
        
    except Exception as e:
        logger.error(f"[DEBUG] Test failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Test failed with exception"
        }


# =============================================================================
# Bulk Actions
# =============================================================================

@router.post("/bulk/papers/delete")
async def bulk_delete_papers(paper_ids: List[str] = Body(embed=False)):
    """Delete multiple papers at once."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        results = []
        for paper_id in paper_ids:
            try:
                paper_uuid = UUID(paper_id)
                deleted = store.delete_paper_view(user_id, paper_uuid)
                results.append({"paper_id": paper_id, "deleted": deleted})
            except Exception as e:
                results.append({"paper_id": paper_id, "deleted": False, "error": str(e)})
        
        return {"results": results, "total": len(paper_ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk/papers/mark-unseen")
async def bulk_mark_unseen(paper_ids: List[str] = Body(embed=False)):
    """Mark multiple papers as unseen."""
    return await bulk_delete_papers(paper_ids)


@router.post("/bulk/papers/toggle-read")
async def bulk_toggle_read(paper_ids: List[str] = Body(embed=False)):
    """Toggle read status for multiple papers."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        paper_uuids = [UUID(pid) for pid in paper_ids]
        updated = store.bulk_toggle_read(user_id, paper_uuids)
        return {"updated": updated, "count": len(updated)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk/papers/mark-read")
async def bulk_mark_read(paper_ids: List[str] = Body(embed=False)):
    """Mark multiple papers as read."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        updated_count = 0
        failed_ids = []
        for paper_id in paper_ids:
            try:
                paper_uuid = UUID(paper_id)
                store.mark_read(user_id, paper_uuid)
                updated_count += 1
            except Exception as e:
                failed_ids.append(paper_id)
        
        return {"updated_count": updated_count, "failed_ids": failed_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk/papers/mark-unread")
async def bulk_mark_unread(paper_ids: List[str] = Body(embed=False)):
    """Mark multiple papers as unread."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        updated_count = 0
        failed_ids = []
        for paper_id in paper_ids:
            try:
                paper_uuid = UUID(paper_id)
                store.update_paper_view(user_id, paper_uuid, {"is_read": False, "read_at": None})
                updated_count += 1
            except Exception as e:
                failed_ids.append(paper_id)
        
        return {"updated_count": updated_count, "failed_ids": failed_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk/papers/toggle-star")
async def bulk_toggle_star(paper_ids: List[str] = Body(embed=False)):
    """Toggle starred status for multiple papers."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        paper_uuids = [UUID(pid) for pid in paper_ids]
        updated = store.bulk_toggle_star(user_id, paper_uuids)
        return {"updated": updated, "count": len(updated)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BulkRelevanceRequest(BaseModel):
    paper_ids: List[str]
    relevance_state: str  # 'relevant' or 'not_relevant'
    note: Optional[str] = None


@router.post("/bulk/papers/set-relevance")
async def bulk_set_relevance(data: BulkRelevanceRequest):
    """Set relevance state for multiple papers."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        paper_uuids = [UUID(pid) for pid in data.paper_ids]
        updated = store.bulk_set_relevance(user_id, paper_uuids, data.relevance_state, data.note)
        return {"updated": updated, "count": len(updated)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk/papers/mark-not-relevant")
async def bulk_mark_not_relevant(paper_ids: List[str] = Body(embed=False)):
    """Mark multiple papers as not relevant."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        paper_uuids = [UUID(pid) for pid in paper_ids]
        updated = store.bulk_set_relevance(user_id, paper_uuids, "not_relevant")
        return {"updated": updated, "count": len(updated)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk/papers/mark-not-relevant-and-remove")
async def bulk_mark_not_relevant_and_remove(paper_ids: List[str] = Body(embed=False)):
    """Mark multiple papers as not relevant and soft-delete them."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        updated_count = 0
        failed_ids = []
        for paper_id in paper_ids:
            try:
                paper_uuid = UUID(paper_id)
                store.update_paper_view(user_id, paper_uuid, {"is_relevant": False, "is_deleted": True})
                updated_count += 1
            except Exception as e:
                failed_ids.append(paper_id)
        
        return {"updated_count": updated_count, "failed_ids": failed_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk/papers/star")
async def bulk_star_papers(paper_ids: List[str] = Body(embed=False)):
    """Star/pin multiple papers."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        updated_count = 0
        failed_ids = []
        for paper_id in paper_ids:
            try:
                paper_uuid = UUID(paper_id)
                store.update_paper_view(user_id, paper_uuid, {"is_starred": True})
                updated_count += 1
            except Exception as e:
                failed_ids.append(paper_id)
        
        return {"updated_count": updated_count, "failed_ids": failed_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk/papers/unstar")
async def bulk_unstar_papers(paper_ids: List[str] = Body(embed=False)):
    """Unstar/unpin multiple papers."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        updated_count = 0
        failed_ids = []
        for paper_id in paper_ids:
            try:
                paper_uuid = UUID(paper_id)
                store.update_paper_view(user_id, paper_uuid, {"is_starred": False})
                updated_count += 1
            except Exception as e:
                failed_ids.append(paper_id)
        
        return {"updated_count": updated_count, "failed_ids": failed_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk/papers/download-pdf")
async def bulk_download_pdf(paper_ids: List[str] = Body(embed=False)):
    """Get PDF download URLs for multiple papers."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        pdf_urls = []
        failed_ids = []
        for paper_id in paper_ids:
            try:
                paper_uuid = UUID(paper_id)
                views = store.list_papers(user_id, limit=10000)
                view = next((v for v in views if str(v.get("paper_id")) == paper_id), None)
                if view and view.get("paper"):
                    paper = view["paper"]
                    external_id = paper.get("external_id", "")
                    if external_id:
                        pdf_url = f"https://arxiv.org/pdf/{external_id}.pdf"
                        pdf_urls.append({"paper_id": paper_id, "title": paper.get("title", ""), "pdf_url": pdf_url})
                    else:
                        failed_ids.append(paper_id)
                else:
                    failed_ids.append(paper_id)
            except Exception as e:
                failed_ids.append(paper_id)
        
        return {"pdf_urls": pdf_urls, "failed_ids": failed_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BulkReminderRequest(BaseModel):
    paper_ids: List[str]
    reminder_date: Optional[str] = None  # ISO format date, defaults to tomorrow
    reminder_time: Optional[str] = None  # HH:MM format, defaults to 10:00


class CreateReminderRequest(BaseModel):
    paper_ids: List[str]
    reminder_date: Optional[str] = None  # ISO format date, defaults to tomorrow
    reminder_time: Optional[str] = None  # HH:MM format, defaults to 10:00
    triggered_by: str = "user"  # 'user' or 'agent'
    agent_note: Optional[str] = None  # Optional note from agent


@router.post("/bulk/papers/create-reminder")
async def bulk_create_reminder(paper_ids: List[str] = Body(...)):
    """
    Create a calendar reminder for reading selected papers.
    
    This endpoint:
    1. Fetches paper details from the database
    2. ResearchPulse agent estimates required reading time based on:
       - Number of papers
       - Paper complexity (importance level)
    3. Creates a calendar event with paper titles and links
    4. Sends a calendar invitation email with .ics attachment
    5. Persists the reminder and invite email with triggered_by='user'
    6. Returns the created calendar event and email records
    """
    try:
        from datetime import timedelta
        from ..tools.ics_generator import generate_uid, generate_reading_reminder_ics
        from ..tools.calendar_invite_sender import send_calendar_invite_email
        import uuid as uuid_module
        
        # No date/time specified, will use defaults (tomorrow at 10:00 AM)
        reminder_date = None
        reminder_time = None
        triggered_by = "user"
        agent_note = None
        
        if not paper_ids:
            raise HTTPException(status_code=400, detail="No paper IDs provided")
        
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        user_email = user.get("email", "")
        user_name = user.get("name", "Researcher")
        
        # Fetch paper details
        papers_data = []
        paper_id_list = []
        for paper_id in paper_ids:
            try:
                paper_uuid = UUID(paper_id)
                view = store.get_paper_view(user_id, paper_uuid)
                if view and view.get("paper"):
                    paper = view["paper"]
                    external_id = paper.get("external_id") or ""
                    abstract = paper.get("abstract") or ""
                    papers_data.append({
                        "id": paper_id,
                        "title": paper.get("title") or "Untitled",
                        "url": f"https://arxiv.org/abs/{external_id}" if external_id else (paper.get("url") or ""),
                        "importance": view.get("importance") or "medium",
                        "abstract_length": len(abstract),
                        "authors": paper.get("authors") or [],
                    })
                    paper_id_list.append(paper_id)
            except Exception as e:
                print(f"Error processing paper {paper_id}: {e}")
                continue
        
        if not papers_data:
            raise HTTPException(status_code=400, detail="No valid papers found for the given IDs")
        
        # Agent estimates reading time based on papers
        estimated_minutes = _estimate_reading_time(papers_data)
        
        # Determine start time (default: tomorrow at 10:00 AM)
        if reminder_date:
            try:
                start_date = datetime.fromisoformat(reminder_date.replace('Z', '+00:00'))
            except:
                start_date = datetime.utcnow() + timedelta(days=1)
        else:
            start_date = datetime.utcnow() + timedelta(days=1)
        
        if reminder_time:
            try:
                hour, minute = map(int, reminder_time.split(':'))
                start_date = start_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
            except:
                start_date = start_date.replace(hour=10, minute=0, second=0, microsecond=0)
        else:
            start_date = start_date.replace(hour=10, minute=0, second=0, microsecond=0)
        
        # Generate event title and description
        if len(papers_data) == 1:
            title = f"📖 Read: {papers_data[0]['title'][:50]}..."
        else:
            title = f"📖 Read {len(papers_data)} research papers (ResearchPulse)"
        
        description = _generate_reminder_description(papers_data, estimated_minutes, agent_note)
        
        # Generate unique ICS UID for this calendar event
        ics_uid = generate_uid()
        
        # Generate ICS content for calendar invite
        ics_content = generate_reading_reminder_ics(
            uid=ics_uid,
            papers=papers_data,
            start_time=start_date,
            duration_minutes=estimated_minutes,
            organizer_email=os.getenv("SMTP_FROM_EMAIL", "researchpulse@example.com"),
            attendee_email=user_email,
            reminder_minutes=15,
            sequence=0,
            agent_note=agent_note,
        )
        
        # Create the calendar event with triggered_by and ICS UID
        event_record = store.create_calendar_event(
            user_id=user_id,
            paper_id=None,  # Bulk reminder, no single paper
            title=title,
            start_time=start_date,
            duration_minutes=estimated_minutes,
            ics_text=ics_content,
            description=description,
            reminder_minutes=15,  # 15 minute notification before
            triggered_by=triggered_by,
            paper_ids=paper_id_list,
        )
        
        event_id = UUID(event_record["id"])
        
        # Update the event with ICS UID (store method may need enhancement)
        try:
            store.update_calendar_event(event_id, {"ics_uid": ics_uid, "sequence_number": 0})
        except Exception:
            pass  # Continue if update fails
        
        # Send calendar invitation email if user has email configured
        invite_email_record = None
        email_sent = False
        
        if user_email and user_email != "user@researchpulse.local":
            try:
                # Generate unique reminder token for reliable reply matching
                import uuid as uuid_mod
                reminder_token = str(uuid_mod.uuid4())[:32]  # 32-char token for brevity
                
                # Generate email body with embedded token
                email_subject = f"📅 Reading Reminder: {title}"
                email_body_text = _generate_invite_email_body(
                    papers_data, start_date, estimated_minutes, agent_note, reminder_token
                )
                email_body_html = _generate_invite_email_html(
                    papers_data, start_date, estimated_minutes, agent_note, reminder_token
                )
                
                # Send the calendar invite email (returns 3 values: success, message_id, error)
                email_sent, actual_message_id, send_error = send_calendar_invite_email(
                    to_email=user_email,
                    to_name=user_name,
                    subject=email_subject,
                    body_text=email_body_text,
                    body_html=email_body_html,
                    ics_content=ics_content,
                    ics_method="REQUEST",
                )
                
                # Use actual message_id from SMTP if available, else generate one
                message_id = actual_message_id or f"<{ics_uid}@researchpulse.local>"
                
                # Create Email record
                email_record = store.create_email(
                    user_id=user_id,
                    paper_id=None,
                    recipient_email=user_email,
                    subject=email_subject,
                    body_text=email_body_text,
                    body_preview=f"Reading reminder for {len(papers_data)} paper(s) scheduled for {start_date.strftime('%B %d at %I:%M %p')}",
                    triggered_by=triggered_by,
                    paper_ids=paper_id_list,
                )
                
                email_id = UUID(email_record["id"])
                
                # Update email status
                if email_sent:
                    store.update_email_status(email_id, status="sent")
                else:
                    store.update_email_status(email_id, status="failed", error=send_error)
                
                # Create CalendarInviteEmail record to link email to calendar event
                invite_email_record = store.create_calendar_invite_email(
                    calendar_event_id=event_id,
                    user_id=user_id,
                    message_id=message_id,
                    recipient_email=user_email,
                    subject=email_subject,
                    ics_uid=ics_uid,
                    email_id=email_id,
                    reminder_token=reminder_token,  # For reliable reply matching
                    ics_sequence=0,
                    ics_method="REQUEST",
                    triggered_by=triggered_by,
                )
                
            except Exception as e:
                print(f"Error sending calendar invite email: {e}")
        
        return {
            "success": True,
            "event": event_record,
            "invite_email": invite_email_record,
            "email_sent": email_sent,
            "papers_count": len(papers_data),
            "estimated_reading_time_minutes": estimated_minutes,
            "scheduled_for": start_date.isoformat() + "Z",
            "message": f"Reading reminder created for {start_date.strftime('%B %d, %Y at %I:%M %p')}" + 
                       (f" and calendar invite sent to {user_email}" if email_sent else "")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _generate_invite_email_body(
    papers: List[Dict[str, Any]], 
    start_time: datetime, 
    duration_minutes: int, 
    agent_note: Optional[str] = None,
    reminder_token: Optional[str] = None,
) -> str:
    """Generate plain text email body for calendar invite.
    
    Args:
        papers: List of paper dicts with title, url, importance
        start_time: Event start time
        duration_minutes: Event duration
        agent_note: Optional note from agent
        reminder_token: Unique token for reply matching (embedded in footer)
    """
    lines = []
    lines.append("ResearchPulse Reading Reminder")
    lines.append("=" * 40)
    lines.append("")
    lines.append(f"📅 Scheduled: {start_time.strftime('%A, %B %d, %Y at %I:%M %p')}")
    lines.append(f"⏱️ Duration: {duration_minutes} minutes")
    lines.append("")
    
    if agent_note:
        lines.append("💡 Agent Note:")
        lines.append(agent_note)
        lines.append("")
    
    lines.append("Papers to Read:")
    lines.append("-" * 40)
    
    for i, paper in enumerate(papers, 1):
        importance = paper.get("importance", "medium").upper()
        lines.append(f"{i}. [{importance}] {paper['title']}")
        lines.append(f"   Link: {paper['url']}")
        lines.append("")
    
    lines.append("-" * 40)
    lines.append("")
    lines.append("To reschedule this reading session, simply reply to this email with")
    lines.append("your preferred date and time (e.g., 'Please move to tomorrow at 2pm').")
    lines.append("")
    lines.append("This reminder was created by ResearchPulse.")
    
    # Embed reminder token for reliable reply matching (kept at bottom, unobtrusive)
    if reminder_token:
        lines.append("")
        lines.append(f"[RP_REMINDER_ID: {reminder_token}]")
    
    return "\n".join(lines)


def _generate_invite_email_html(
    papers: List[Dict[str, Any]], 
    start_time: datetime, 
    duration_minutes: int, 
    agent_note: Optional[str] = None,
    reminder_token: Optional[str] = None,
) -> str:
    """Generate HTML email body for calendar invite.
    
    Args:
        papers: List of paper dicts with title, url, importance
        start_time: Event start time
        duration_minutes: Event duration
        agent_note: Optional note from agent
        reminder_token: Unique token for reply matching (embedded in hidden span)
    """
    html = []
    html.append("""<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px 8px 0 0; }
        .content { padding: 20px; background: #f9f9f9; }
        .paper { background: white; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #667eea; }
        .importance-high { border-left-color: #e53e3e; }
        .importance-critical { border-left-color: #c53030; }
        .importance-medium { border-left-color: #dd6b20; }
        .importance-low { border-left-color: #38a169; }
        .footer { padding: 15px; background: #eee; border-radius: 0 0 8px 8px; font-size: 12px; color: #666; }
        .agent-note { background: #e6f3ff; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #3182ce; }
        a { color: #667eea; }
        .rp-token { display: none; font-size: 1px; color: transparent; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📖 ResearchPulse Reading Reminder</h1>
    </div>
    <div class="content">
""")
    
    html.append(f"""
        <p><strong>📅 Scheduled:</strong> {start_time.strftime('%A, %B %d, %Y at %I:%M %p')}</p>
        <p><strong>⏱️ Duration:</strong> {duration_minutes} minutes</p>
""")
    
    if agent_note:
        html.append(f"""
        <div class="agent-note">
            <strong>💡 Agent Note:</strong><br>
            {agent_note}
        </div>
""")
    
    html.append("""
        <h2>Papers to Read</h2>
""")
    
    for i, paper in enumerate(papers, 1):
        importance = paper.get("importance", "medium")
        importance_class = f"importance-{importance}"
        html.append(f"""
        <div class="paper {importance_class}">
            <strong>{i}. {paper['title']}</strong>
            <br>
            <small>Importance: {importance.upper()}</small>
            <br>
            <a href="{paper['url']}">📄 Read Paper</a>
        </div>
""")
    
    # Build footer with optional reminder token
    footer_content = """
    </div>
    <div class="footer">
        <p>To reschedule this reading session, simply reply to this email with your preferred date and time 
        (e.g., "Please move to tomorrow at 2pm").</p>
        <p>This reminder was created by <strong>ResearchPulse</strong>.</p>"""
    
    if reminder_token:
        footer_content += f"""
        <span class="rp-token">[RP_REMINDER_ID: {reminder_token}]</span>"""
    
    footer_content += """
    </div>
</body>
</html>
"""
    html.append(footer_content)
    
    return "".join(html)


def _estimate_reading_time(papers: List[Dict[str, Any]]) -> int:
    """
    ResearchPulse agent estimates reading time based on the **actual full paper**
    (PDF page count) when available, falling back to heuristics otherwise.

    Factors considered:
    - Full-paper page count (fetched from arXiv PDF)
    - Paper importance (critical/high papers need deep reading)
    - Novelty score (novel work takes longer to digest)
    - Category difficulty (math-heavy fields take longer)

    Returns estimated time in minutes.
    """
    from src.tools.decide_delivery import (
        _fetch_pdf_page_count,
        _estimate_from_pages,
        _estimate_from_abstract,
        ScoredPaper,
    )

    total_minutes = 0

    for paper in papers:
        pdf_url = paper.get("pdf_url") or paper.get("url") or ""
        importance = paper.get("importance", "medium")
        if importance == "critical":
            importance = "high"  # ScoredPaper only allows high/medium/low

        scored = ScoredPaper(
            arxiv_id=paper.get("external_id", paper.get("arxiv_id", "")),
            title=paper.get("title", ""),
            abstract=paper.get("abstract", ""),
            link=paper.get("url"),
            pdf_url=paper.get("pdf_url"),
            categories=paper.get("categories", []),
            relevance_score=paper.get("relevance_score", 0.5),
            novelty_score=paper.get("novelty_score", 0.5),
            importance=importance,
        )

        page_count = _fetch_pdf_page_count(pdf_url)
        if page_count and page_count > 0:
            total_minutes += _estimate_from_pages(page_count, scored)
        else:
            total_minutes += _estimate_from_abstract(scored)

    # Round to nearest 5 minutes
    total_minutes = ((total_minutes + 2) // 5) * 5

    # Minimum 10 minutes, maximum 4 hours
    return max(10, min(total_minutes, 240))


def _generate_reminder_description(papers: List[Dict[str, Any]], estimated_minutes: int, agent_note: Optional[str] = None) -> str:
    """
    Generate the calendar event description with paper details.
    """
    lines = []
    lines.append(f"📚 ResearchPulse Reading Session")
    lines.append(f"Estimated time: {estimated_minutes} minutes")
    lines.append("")
    
    if agent_note:
        lines.append("💡 Agent Note:")
        lines.append(agent_note)
        lines.append("")
    
    lines.append("Papers to read:")
    lines.append("")
    
    for i, paper in enumerate(papers, 1):
        importance = paper.get("importance", "medium").upper()
        lines.append(f"{i}. [{importance}] {paper['title']}")
        lines.append(f"   Link: {paper['url']}")
        lines.append("")
    
    lines.append("---")
    lines.append("This reminder was created by ResearchPulse at your request.")
    lines.append("Tip: Start with high-priority papers for maximum productivity.")
    
    return "\n".join(lines)


@router.post("/bulk/papers/email-summary")
async def bulk_email_summary(paper_ids: List[str] = Body(embed=False)):
    """
    Generate and send an email summary for selected papers.
    
    This endpoint:
    1. Fetches paper details from the database
    2. Uses ResearchPulse agent to generate a SHORT, intelligent summary
    3. Sends the summary email to the user's email address
    4. Persists the email in the database with triggered_by='user'
    5. Returns the created email record
    """
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        user_email = user.get("email", "")
        user_name = user.get("name", "Researcher")
        
        if not user_email or user_email == "user@researchpulse.local":
            raise HTTPException(
                status_code=400, 
                detail="No email address configured. Please update your profile in Settings."
            )
        
        # Fetch paper details
        papers_data = []
        paper_id_list = []
        for paper_id in paper_ids:
            try:
                paper_uuid = UUID(paper_id)
                view = store.get_paper_view(user_id, paper_uuid)
                if view and view.get("paper"):
                    paper = view["paper"]
                    external_id = paper.get("external_id") or ""
                    
                    # Get abstract - fetch from arXiv if not in DB
                    abstract = paper.get("abstract") or ""
                    if not abstract and external_id:
                        abstract = _fetch_abstract_from_arxiv(external_id) or ""
                    
                    papers_data.append({
                        "id": paper_id,
                        "title": paper.get("title") or "Untitled",
                        "authors": paper.get("authors") or [],
                        "abstract": abstract,
                        "url": f"https://arxiv.org/abs/{external_id}" if external_id else (paper.get("url") or ""),
                        "pdf_url": f"https://arxiv.org/pdf/{external_id}" if external_id else (paper.get("pdf_url") or ""),
                        "importance": view.get("importance") or "medium",
                        "categories": paper.get("categories") or [],
                    })
                    paper_id_list.append(paper_id)
            except Exception:
                continue
        
        if not papers_data:
            raise HTTPException(status_code=400, detail="No valid papers found for the given IDs")
        
        # Generate agent-style concise summary
        # The agent decides the summary style based on number and content of papers
        summary_body = _generate_agent_email_summary(papers_data, user_name)
        
        # Create email subject
        if len(papers_data) == 1:
            subject = f"📚 ResearchPulse: Summary of \"{papers_data[0]['title'][:50]}...\""
        else:
            subject = f"📚 ResearchPulse: Summary of {len(papers_data)} Research Papers"
        
        # Create the email record with triggered_by='user'
        email_record = store.create_email(
            user_id=user_id,
            paper_id=None,  # Bulk email, no single paper
            recipient_email=user_email,
            subject=subject,
            body_text=summary_body,
            body_preview=summary_body[:200] + "..." if len(summary_body) > 200 else summary_body,
            triggered_by='user',
            paper_ids=paper_id_list,
        )
        
        email_id = UUID(email_record["id"])
        
        # Actually send the email via SMTP
        try:
            # Try relative import first, fall back to absolute import for different execution contexts
            try:
                from ..tools.decide_delivery import _send_email_smtp
            except (ImportError, ValueError):
                # Fallback: add tools path and import directly
                import sys
                from pathlib import Path
                tools_path = str(Path(__file__).parent.parent / "tools")
                if tools_path not in sys.path:
                    sys.path.insert(0, tools_path)
                from decide_delivery import _send_email_smtp  # type: ignore[import-not-found]
            email_sent = _send_email_smtp(
                to_email=user_email,
                subject=subject,
                body=summary_body,
            )
            
            if email_sent:
                email_record = store.update_email_status(email_id, status="sent")
                return {
                    "success": True,
                    "email": email_record,
                    "papers_count": len(papers_data),
                    "message": f"Email summary sent to {user_email}"
                }
            else:
                email_record = store.update_email_status(email_id, status="failed")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to send email. Check SMTP configuration (SMTP_HOST, SMTP_USER, SMTP_PASSWORD)."
                )
        except ImportError:
            email_record = store.update_email_status(email_id, status="failed")
            raise HTTPException(
                status_code=500,
                detail="Email sending module not available."
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _fetch_abstract_from_arxiv(arxiv_id: str) -> str:
    """
    Fetch abstract from arXiv API if not stored in DB.
    
    Args:
        arxiv_id: The arXiv paper ID (e.g., "2601.23262v1")
        
    Returns:
        The paper abstract text, or empty string if not found.
    """
    import urllib.request
    import xml.etree.ElementTree as ET
    
    try:
        # Handle version suffix (e.g., "2601.23262v1" -> "2601.23262")
        base_id = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id
        url = f"http://export.arxiv.org/api/query?id_list={base_id}"
        print(f"[DEBUG] Fetching abstract from arXiv for ID: {base_id}")
        response = urllib.request.urlopen(url, timeout=10)
        xml_data = response.read().decode('utf-8')
        root = ET.fromstring(xml_data)
        
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entry = root.find('atom:entry', ns)
        if entry is not None:
            summary = entry.find('atom:summary', ns)
            if summary is not None and summary.text:
                abstract = summary.text.strip()
                print(f"[DEBUG] Found abstract ({len(abstract)} chars)")
                return abstract
        print(f"[DEBUG] No abstract found in arXiv response for {base_id}")
    except Exception as e:
        print(f"[DEBUG] Error fetching abstract from arXiv: {e}")
    return ""


def _generate_paper_summary(abstract: str, title: str) -> str:
    """
    Generate a concise 2-3 sentence summary of a paper from its abstract.
    
    Uses LLM if available, otherwise extracts key sentences heuristically.
    """
    if not abstract:
        return "No abstract available for this paper."
    
    # Try LLM-based summary if available
    try:
        import os
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        if api_key:
            import openai
            
            # Check if Azure or OpenAI
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if azure_endpoint:
                client = openai.AzureOpenAI(
                    api_key=api_key,
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                    azure_endpoint=azure_endpoint
                )
                model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
            else:
                client = openai.OpenAI(api_key=api_key)
                model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a research assistant. Summarize academic papers concisely in 2-3 sentences, focusing on the main contribution and methodology. Be specific about what makes this paper unique."},
                    {"role": "user", "content": f"Summarize this paper:\n\nTitle: {title}\n\nAbstract: {abstract}"}
                ],
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
    except Exception:
        pass  # Fall back to heuristic
    
    # Heuristic fallback: extract key sentences from abstract
    sentences = abstract.replace('\n', ' ').split('. ')
    
    # Try to find the most informative sentences
    key_sentences = []
    
    # Look for contribution/result sentence
    contribution_keywords = ['propose', 'present', 'introduce', 'develop', 'demonstrate', 'show', 'achieve', 'outperform', 'novel', 'new approach']
    result_keywords = ['results', 'experiments', 'evaluation', 'performance', 'accuracy', 'improvement']
    
    contribution_sentence = None
    result_sentence = None
    
    for sent in sentences:
        sent_lower = sent.lower()
        if not contribution_sentence and any(kw in sent_lower for kw in contribution_keywords):
            contribution_sentence = sent.strip()
        elif not result_sentence and any(kw in sent_lower for kw in result_keywords):
            result_sentence = sent.strip()
    
    # Build summary from found sentences
    if contribution_sentence:
        key_sentences.append(contribution_sentence)
    elif sentences:
        # Use first sentence as fallback
        key_sentences.append(sentences[0].strip())
    
    if result_sentence and result_sentence != contribution_sentence:
        key_sentences.append(result_sentence)
    elif len(sentences) > 1 and sentences[-1].strip() and sentences[-1] != sentences[0]:
        # Use last sentence as fallback for results
        key_sentences.append(sentences[-1].strip())
    
    # Join and clean up
    summary = '. '.join(key_sentences)
    if not summary.endswith('.'):
        summary += '.'
    
    # Truncate if too long
    if len(summary) > 400:
        summary = summary[:397].rsplit(' ', 1)[0] + '...'
    
    return summary


def _generate_agent_email_summary(papers: List[Dict[str, Any]], user_name: str) -> str:
    """
    Generate an agent-style concise summary of papers.
    
    The ResearchPulse agent generates intelligent summaries:
    - For a single paper: detailed summary with key contribution
    - For multiple papers: bullet-point list with mini-summaries
    """
    lines = []
    lines.append(f"Hi {user_name},")
    lines.append("")
    
    if len(papers) == 1:
        # Single paper - detailed summary
        p = papers[0]
        lines.append(f"Here's a summary of the paper you selected:")
        lines.append("")
        lines.append(f"📄 **{p['title']}**")
        lines.append("")
        if p['authors']:
            author_str = ", ".join(p['authors'][:3])
            if len(p['authors']) > 3:
                author_str += f" et al."
            lines.append(f"👤 Authors: {author_str}")
        if p['categories']:
            lines.append(f"🏷️ Categories: {', '.join(p['categories'][:3])}")
        lines.append(f"⭐ Importance: {p['importance'].upper()}")
        lines.append("")
        
        # Generate intelligent summary
        summary = _generate_paper_summary(p.get('abstract', ''), p['title'])
        lines.append("📝 **Summary:**")
        lines.append(summary)
        lines.append("")
        lines.append(f"🔗 Read the full paper: {p['url']}")
        if p.get('pdf_url'):
            lines.append(f"📥 Download PDF: {p['pdf_url']}")
    else:
        # Multiple papers - compact list with mini-summaries
        lines.append(f"Here's a summary of the {len(papers)} papers you selected:")
        lines.append("")
        
        # Group by importance
        high_importance = [p for p in papers if p['importance'] in ('high', 'critical')]
        medium_importance = [p for p in papers if p['importance'] == 'medium']
        low_importance = [p for p in papers if p['importance'] == 'low']
        
        paper_groups = [
            ("🔴 HIGH PRIORITY", high_importance),
            ("🟡 MEDIUM PRIORITY", medium_importance),
            ("🟢 OTHER PAPERS", low_importance),
        ]
        
        for group_name, group_papers in paper_groups:
            if group_papers:
                lines.append(f"{group_name}:")
                lines.append("")
                for p in group_papers:
                    lines.append(f"  📄 **{p['title']}**")
                    # Generate short summary for each paper
                    summary = _generate_paper_summary(p.get('abstract', ''), p['title'])
                    # Truncate for multi-paper emails
                    if len(summary) > 200:
                        summary = summary[:197].rsplit(' ', 1)[0] + '...'
                    lines.append(f"     {summary}")
                    lines.append(f"     🔗 Abstract: {p['url']}")
                    if p.get('pdf_url'):
                        lines.append(f"     📥 PDF: {p['pdf_url']}")
                    lines.append("")
    
    lines.append("---")
    lines.append("This summary was generated by ResearchPulse at your request.")
    lines.append("")
    
    return "\n".join(lines)


@router.post("/reindex")
async def reindex_all(background_tasks: BackgroundTasks):
    """Re-index all papers in Pinecone."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        async def do_reindex():
            try:
                papers = store.list_papers(user_id, limit=10000)
                from ..rag.vector_store import upsert_paper_vector
                
                count = 0
                for view in papers:
                    paper = view.get("paper", {})
                    if paper:
                        upsert_paper_vector(paper, scope="user", user_id=str(user_id))
                        store.update_paper_view(user_id, UUID(view["paper_id"]), {"embedded_in_pinecone": True})
                        count += 1
                
                return count
            except Exception as e:
                print(f"Reindex error: {e}")
                return 0
        
        background_tasks.add_task(do_reindex)
        return {"status": "started", "message": "Reindex started in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-seen-papers")
async def reset_seen_papers():
    """Reset the papers_state.json file to clear all seen papers."""
    try:
        from pathlib import Path
        import json as json_module
        
        papers_state_path = Path("data/papers_state.json")
        
        if papers_state_path.exists():
            # Create backup
            backup_path = Path("data/papers_state_backup.json")
            with open(papers_state_path) as f:
                old_data = json_module.load(f)
            with open(backup_path, "w") as f:
                json_module.dump(old_data, f, indent=2)
            
            # Reset to empty
            new_data = {"papers": []}
            with open(papers_state_path, "w") as f:
                json_module.dump(new_data, f, indent=2)
            
            papers_cleared = len(old_data.get("papers", []))
            return {
                "status": "success",
                "message": f"Cleared {papers_cleared} papers from history. Backup saved to papers_state_backup.json"
            }
        else:
            return {"status": "success", "message": "No papers state file found - nothing to reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Stats Endpoint
# =============================================================================

@router.get("/stats")
async def get_stats():
    """Get dashboard statistics."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        # Get counts
        all_papers = store.list_papers(user_id, limit=10000, include_not_relevant=True)
        emails = store.list_emails(user_id, limit=10000)
        calendar = store.list_calendar_events(user_id, limit=10000)
        shares = store.list_shares(user_id, limit=10000)
        colleagues = store.list_colleagues(user_id)
        runs = store.list_runs(user_id, limit=100)
        
        # Count by decision
        decisions = {}
        for p in all_papers:
            d = p.get("decision", "unknown")
            decisions[d] = decisions.get(d, 0) + 1
        
        # Count by importance
        importance = {}
        for p in all_papers:
            i = p.get("importance", "unknown")
            importance[i] = importance.get(i, 0) + 1
        
        return {
            "papers": {
                "total": len(all_papers),
                "by_decision": decisions,
                "by_importance": importance,
            },
            "emails": {
                "total": len(emails),
                "sent": len([e for e in emails if e.get("status") == "sent"]),
                "failed": len([e for e in emails if e.get("status") == "failed"]),
                "queued": len([e for e in emails if e.get("status") == "queued"]),
            },
            "calendar": {"total": len(calendar)},
            "shares": {"total": len(shares)},
            "colleagues": {"total": len(colleagues), "enabled": len([c for c in colleagues if c.get("enabled")])},
            "runs": {
                "total": len(runs),
                "last_run": runs[0] if runs else None,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# User Feedback Signals for Agent
# =============================================================================

@router.get("/user/feedback-signals")
async def get_user_feedback_signals():
    """
    Get aggregated user feedback signals for agent use.
    
    Returns data that helps the agent understand user preferences:
    - Papers marked not relevant (with authors, categories, keywords)
    - Papers marked relevant 
    - Starred papers (high interest)
    - Read vs unread patterns
    - Feedback history summary
    
    The agent should call this at the start of each run to adjust recommendations.
    """
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        signals = store.get_user_feedback_signals(user_id)
        return signals
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/feedback-history")
async def get_all_feedback_history(
    limit: int = Query(100, le=500),
    offset: int = Query(0),
):
    """Get complete feedback history for the current user."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        history = store.get_feedback_history(user_id, limit=limit, offset=offset)
        return {"history": history, "count": len(history), "offset": offset, "limit": limit}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Email Management Endpoints
# =============================================================================

@router.get("/emails/{email_id}")
async def get_email(email_id: UUID):
    """Get a single email with full details."""
    try:
        store = get_default_store()
        email = store.get_email(email_id)
        if not email:
            raise HTTPException(status_code=404, detail="Email not found")
        return email
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/emails/{email_id}")
async def delete_email(email_id: UUID):
    """Delete a single email."""
    try:
        store = get_default_store()
        deleted = store.delete_email(email_id)
        return {"deleted": deleted, "email_id": str(email_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emails/{email_id}/resend")
async def resend_email(email_id: UUID):
    """Resend an email (creates a new email with the same content)."""
    try:
        from ..tools.calendar_invite_sender import send_calendar_invite_email
        
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        user_name = user.get("name", "Researcher")
        
        # Get original email
        original = store.get_email(email_id)
        if not original:
            raise HTTPException(status_code=404, detail="Email not found")
        
        recipient = original.get("recipient_email", "")
        subject = original.get("subject", "")
        body_text = original.get("body_text", "")
        
        if not recipient:
            raise HTTPException(status_code=400, detail="No recipient email in original")
        
        # Send the email
        email_sent, message_id, send_error = send_calendar_invite_email(
            to_email=recipient,
            to_name=user_name,
            subject=f"[Resent] {subject}",
            body_text=body_text,
            body_html=None,
            ics_content=None,
            ics_method=None,
        )
        
        # Create new email record
        new_email = store.create_email(
            user_id=user_id,
            paper_id=original.get("paper_id"),
            recipient_email=recipient,
            subject=f"[Resent] {subject}",
            body_text=body_text,
            body_preview=original.get("body_preview", ""),
            triggered_by="user",
            paper_ids=original.get("paper_ids", []),
        )
        
        new_email_id = UUID(new_email["id"])
        if email_sent:
            store.update_email_status(new_email_id, status="sent")
        else:
            store.update_email_status(new_email_id, status="failed", error=send_error)
        
        return {
            "success": email_sent,
            "new_email_id": str(new_email_id),
            "message": "Email resent successfully" if email_sent else f"Failed to resend: {send_error}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk/emails/delete")
async def bulk_delete_emails(email_ids: List[str] = Body(embed=False)):
    """Delete multiple emails at once."""
    try:
        store = get_default_store()
        
        deleted_count = 0
        failed_ids = []
        for email_id in email_ids:
            try:
                email_uuid = UUID(email_id)
                deleted = store.delete_email(email_uuid)
                if deleted:
                    deleted_count += 1
                else:
                    failed_ids.append(email_id)
            except Exception as e:
                failed_ids.append(email_id)
        
        return {"deleted_count": deleted_count, "failed_ids": failed_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk/emails/resend")
async def bulk_resend_emails(email_ids: List[str] = Body(embed=False)):
    """Resend multiple emails."""
    try:
        from ..tools.calendar_invite_sender import send_calendar_invite_email
        
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        user_name = user.get("name", "Researcher")
        
        sent_count = 0
        failed_ids = []
        for email_id in email_ids:
            try:
                email_uuid = UUID(email_id)
                original = store.get_email(email_uuid)
                if not original:
                    failed_ids.append(email_id)
                    continue
                
                recipient = original.get("recipient_email", "")
                if not recipient:
                    failed_ids.append(email_id)
                    continue
                
                subject = original.get("subject", "")
                body_text = original.get("body_text", "")
                
                email_sent, _, _ = send_calendar_invite_email(
                    to_email=recipient,
                    to_name=user_name,
                    subject=f"[Resent] {subject}",
                    body_text=body_text,
                    body_html=None,
                    ics_content=None,
                    ics_method=None,
                )
                
                if email_sent:
                    sent_count += 1
                else:
                    failed_ids.append(email_id)
            except Exception:
                failed_ids.append(email_id)
        
        return {"sent_count": sent_count, "failed_ids": failed_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bulk/emails/export")
async def bulk_export_emails(email_ids: Optional[str] = Query(None, description="Comma-separated email IDs")):
    """Export emails to CSV format."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        # Get emails
        if email_ids:
            id_list = [UUID(eid.strip()) for eid in email_ids.split(",")]
            emails = [store.get_email(eid) for eid in id_list]
            emails = [e for e in emails if e]
        else:
            emails = store.list_emails(user_id, limit=1000)
        
        # Create CSV
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["ID", "Recipient", "Subject", "Status", "Created At", "Sent At", "Body Preview"])
        
        for email in emails:
            writer.writerow([
                email.get("id", ""),
                email.get("recipient_email", ""),
                email.get("subject", ""),
                email.get("status", ""),
                email.get("created_at", ""),
                email.get("sent_at", ""),
                email.get("body_preview", "")[:100],
            ])
        
        output.seek(0)
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=emails_export.csv"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Calendar Event Management Endpoints
# =============================================================================

@router.delete("/calendar/{event_id}")
async def delete_calendar_event(event_id: UUID):
    """Delete a single calendar event."""
    try:
        store = get_default_store()
        deleted = store.delete_calendar_event(event_id)
        return {"deleted": deleted, "event_id": str(event_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calendar/{event_id}/ics")
async def download_calendar_ics(event_id: UUID):
    """Download the ICS file for a calendar event."""
    try:
        store = get_default_store()
        event = store.get_calendar_event(event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Calendar event not found")
        
        ics_content = event.get("ics_text", "")
        if not ics_content:
            # Generate ICS if not stored
            from ..tools.ics_generator import generate_uid, generate_reading_reminder_ics
            
            ics_uid = event.get("ics_uid") or generate_uid()
            start_time = event.get("start_time")
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            
            duration = event.get("duration_minutes") or 30
            
            ics_content = generate_reading_reminder_ics(
                uid=ics_uid,
                papers=[{"title": event.get("title", "Reading Reminder"), "url": "", "importance": "medium"}],
                start_time=start_time,
                duration_minutes=duration,
                organizer_email=os.getenv("SMTP_FROM_EMAIL", "researchpulse@example.com"),
                attendee_email="",
                reminder_minutes=15,
                sequence=event.get("sequence_number") or 0,
            )
        
        title = event.get("title", "event")[:30].replace(" ", "_")
        filename = f"researchpulse_{title}.ics"
        
        return Response(
            content=ics_content,
            media_type="text/calendar",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CreateEventRequest(BaseModel):
    title: str
    start_time: str  # ISO format datetime
    duration_minutes: int = 30
    description: Optional[str] = None
    reminder_minutes: int = 15
    send_invite: bool = True
    triggered_by: Optional[str] = None


@router.post("/calendar/create")
@router.post("/calendar")
async def create_calendar_event_endpoint(data: CreateEventRequest):
    """Create a new calendar event manually."""
    try:
        from ..tools.ics_generator import generate_uid, generate_reading_reminder_ics
        from ..tools.calendar_invite_sender import send_calendar_invite_email
        
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        user_email = user.get("email", "")
        user_name = user.get("name", "Researcher")
        
        # Parse start time
        try:
            start_time = datetime.fromisoformat(data.start_time.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid datetime format")
        
        # Generate ICS
        ics_uid = generate_uid()
        ics_content = generate_reading_reminder_ics(
            uid=ics_uid,
            papers=[{"title": data.title, "url": "", "importance": "medium"}],
            start_time=start_time,
            duration_minutes=data.duration_minutes,
            organizer_email=os.getenv("SMTP_FROM_EMAIL", "researchpulse@example.com"),
            attendee_email=user_email,
            reminder_minutes=data.reminder_minutes,
            sequence=0,
        )
        
        # Create calendar event
        event = store.create_calendar_event(
            user_id=user_id,
            paper_id=None,
            title=data.title,
            start_time=start_time,
            duration_minutes=data.duration_minutes,
            ics_text=ics_content,
            description=data.description,
            reminder_minutes=data.reminder_minutes,
            triggered_by="user",
            paper_ids=[],
        )
        
        event_id = UUID(event["id"])
        store.update_calendar_event(event_id, {"ics_uid": ics_uid, "sequence_number": 0})
        
        # Send invite email
        email_sent = False
        if data.send_invite and user_email and user_email != "user@researchpulse.local":
            try:
                email_subject = f"📅 {data.title}"
                email_body = f"""Calendar Event Created

📅 {data.title}
🕐 {start_time.strftime('%A, %B %d, %Y at %I:%M %p')}
⏱️ Duration: {data.duration_minutes} minutes

{data.description or ''}

---
Created by ResearchPulse
"""
                email_sent, message_id, _ = send_calendar_invite_email(
                    to_email=user_email,
                    to_name=user_name,
                    subject=email_subject,
                    body_text=email_body,
                    body_html=None,
                    ics_content=ics_content,
                    ics_method="REQUEST",
                )
            except Exception:
                pass
        
        return {
            "success": True,
            "event": event,
            "email_sent": email_sent,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class UpdateEventRequest(BaseModel):
    title: Optional[str] = None
    start_time: Optional[str] = None
    duration_minutes: Optional[int] = None
    description: Optional[str] = None
    reminder_minutes: Optional[int] = None


@router.put("/calendar/{event_id}")
async def update_calendar_event_endpoint(event_id: UUID, data: UpdateEventRequest):
    """Update a calendar event."""
    try:
        store = get_default_store()
        
        update_data = data.model_dump(exclude_none=True)
        
        # Parse start_time if provided
        if "start_time" in update_data:
            try:
                update_data["start_time"] = datetime.fromisoformat(
                    update_data["start_time"].replace('Z', '+00:00')
                )
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid datetime format")
        
        updated = store.update_calendar_event(event_id, update_data)
        if not updated:
            raise HTTPException(status_code=404, detail="Calendar event not found")
        
        return updated
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk/calendar/delete")
async def bulk_delete_calendar_events(event_ids: List[str] = Body(embed=False)):
    """Delete multiple calendar events at once."""
    try:
        store = get_default_store()
        
        deleted_count = 0
        failed_ids = []
        for event_id in event_ids:
            try:
                event_uuid = UUID(event_id)
                deleted = store.delete_calendar_event(event_uuid)
                if deleted:
                    deleted_count += 1
                else:
                    failed_ids.append(event_id)
            except Exception:
                failed_ids.append(event_id)
        
        return {"deleted_count": deleted_count, "failed_ids": failed_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bulk/calendar/download-ics")
async def bulk_download_ics(event_ids: Optional[str] = Query(None, description="Comma-separated event IDs")):
    """Download ICS files for multiple events as a combined file."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        # Get events
        if event_ids:
            id_list = [UUID(eid.strip()) for eid in event_ids.split(",")]
            events = [store.get_calendar_event(eid) for eid in id_list]
            events = [e for e in events if e]
        else:
            events = store.list_calendar_events(user_id, limit=100)
        
        if not events:
            raise HTTPException(status_code=404, detail="No events found")
        
        # Combine ICS files
        from ..tools.ics_generator import generate_uid
        
        ics_lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//ResearchPulse//Calendar//EN",
            "METHOD:PUBLISH",
            "X-WR-CALNAME:ResearchPulse Events",
        ]
        
        for event in events:
            ics_content = event.get("ics_text", "")
            if ics_content:
                # Extract VEVENT from ICS
                lines = ics_content.split("\n")
                in_event = False
                for line in lines:
                    line = line.strip()
                    if line == "BEGIN:VEVENT":
                        in_event = True
                    if in_event:
                        ics_lines.append(line)
                    if line == "END:VEVENT":
                        in_event = False
            else:
                # Generate minimal VEVENT
                ics_uid = event.get("ics_uid") or generate_uid()
                start_time = event.get("start_time")
                if isinstance(start_time, str):
                    start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                
                duration = event.get("duration_minutes") or 30
                end_time = start_time + timedelta(minutes=duration)
                
                ics_lines.extend([
                    "BEGIN:VEVENT",
                    f"UID:{ics_uid}",
                    f"DTSTART:{start_time.strftime('%Y%m%dT%H%M%S')}",
                    f"DTEND:{end_time.strftime('%Y%m%dT%H%M%S')}",
                    f"SUMMARY:{event.get('title', 'Event')}",
                    f"DESCRIPTION:{event.get('description', '')}",
                    "END:VEVENT",
                ])
        
        ics_lines.append("END:VCALENDAR")
        ics_content = "\r\n".join(ics_lines)
        
        return Response(
            content=ics_content,
            media_type="text/calendar",
            headers={"Content-Disposition": "attachment; filename=researchpulse_events.ics"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BulkRescheduleRequest(BaseModel):
    event_ids: List[str]
    new_start_time: str  # ISO format datetime
    new_duration_minutes: Optional[int] = None


@router.post("/bulk/calendar/reschedule")
async def bulk_reschedule_events(data: BulkRescheduleRequest):
    """Reschedule multiple calendar events to the same time."""
    try:
        from datetime import timedelta
        
        store = get_default_store()
        user = store.get_or_create_default_user()
        
        # Parse new start time
        try:
            new_start_time = datetime.fromisoformat(data.new_start_time.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid datetime format")
        
        updated_count = 0
        failed_ids = []
        
        for event_id in data.event_ids:
            try:
                event_uuid = UUID(event_id)
                existing = store.get_calendar_event(event_uuid)
                if not existing:
                    failed_ids.append(event_id)
                    continue
                
                new_duration = data.new_duration_minutes or existing.get("duration_minutes", 30)
                
                store.reschedule_calendar_event(
                    event_id=event_uuid,
                    new_start_time=new_start_time,
                    reschedule_note="Bulk rescheduled by user",
                    new_duration_minutes=new_duration,
                )
                updated_count += 1
                
                # Offset each subsequent event by its duration to avoid overlap
                new_start_time = new_start_time + timedelta(minutes=new_duration + 15)
            except Exception:
                failed_ids.append(event_id)
        
        return {"updated_count": updated_count, "failed_ids": failed_ids}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Inbox Settings API
# =============================================================================

class InboxSettingsUpdate(BaseModel):
    """Request model for updating inbox settings."""
    enabled: bool
    frequency_seconds: Optional[int] = None  # None/null = disabled


class ColleagueJoinCodeRequest(BaseModel):
    """Request model for setting join code."""
    code: str  # Plain text code (will be hashed)


@router.get("/settings/inbox")
async def get_inbox_settings():
    """
    Get current inbox polling settings.
    
    Returns:
        - enabled: Whether inbox polling is enabled
        - frequency_seconds: Polling interval (null if disabled)
        - last_check_at: Last time inbox was checked
        - has_join_code: Whether a colleague join code is configured
    """
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        settings = store.get_or_create_user_settings(user_id)
        
        return {
            "success": True,
            "settings": settings,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/settings/inbox")
async def update_inbox_settings(data: InboxSettingsUpdate):
    """
    Update inbox polling settings.
    
    Valid frequency_seconds values:
    - null/None: Disabled
    - 10: Every 10 seconds (aggressive)
    - 30: Every 30 seconds
    - 60: Every minute
    - 300: Every 5 minutes
    - 900: Every 15 minutes
    - 3600: Every hour (power saving)
    """
    try:
        # Validate frequency
        valid_frequencies = [None, 10, 30, 60, 300, 900, 3600]
        if data.frequency_seconds not in valid_frequencies:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid frequency. Valid values: {valid_frequencies}"
            )
        
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        settings = store.update_inbox_settings(
            user_id=user_id,
            enabled=data.enabled,
            frequency_seconds=data.frequency_seconds,
        )
        
        # Notify scheduler of change (if implemented)
        try:
            from ..tools.inbox_scheduler import update_scheduler
            update_scheduler(user_id, data.enabled, data.frequency_seconds)
        except ImportError:
            pass  # Scheduler not yet implemented
        except Exception as e:
            # Log but don't fail the request
            import logging
            logging.warning(f"Could not update scheduler: {e}")
        
        return {
            "success": True,
            "settings": settings,
            "message": "Inbox settings updated" if data.enabled else "Inbox polling disabled",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/settings/inbox/check-now")
async def trigger_inbox_check():
    """
    Manually trigger an inbox check (regardless of schedule).
    
    Useful for testing or immediate processing.
    """
    try:
        from ..tools.inbound_processor import InboundEmailProcessor
        
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = user["id"]
        
        processor = InboundEmailProcessor(store, user_id)
        results = processor.process_all(since_hours=48)
        
        return {
            "success": True,
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/settings/colleague-join-code")
async def set_colleague_join_code(data: ColleagueJoinCodeRequest):
    """
    Set or rotate the colleague join code.
    
    The code is encrypted (AES) before storage for display-back capability.
    Also stores bcrypt hash for backward-compatible verification.
    """
    try:
        from ..tools.inbound_processor import hash_join_code
        from ..tools.join_code_crypto import encrypt_join_code
        
        if len(data.code) < 4:
            raise HTTPException(status_code=400, detail="Code must be at least 4 characters")
        if len(data.code) > 32:
            raise HTTPException(status_code=400, detail="Code must be at most 32 characters")
        
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        # Store encrypted version (for display-back) and hash (for verification)
        encrypted = encrypt_join_code(data.code)
        code_hash = hash_join_code(data.code)
        
        # Save both in a single call: hash for verification, encrypted for display-back
        settings = store.set_colleague_join_code_encrypted(user_id, encrypted, code_hash=code_hash)
        
        return {
            "success": True,
            "message": "Join code updated successfully",
            "settings": settings,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/settings/colleague-join-code/current")
async def get_current_join_code():
    """
    Get the current join code (decrypted) for display to the owner.
    
    Only the owner can see the current code.
    """
    try:
        from ..tools.join_code_crypto import decrypt_join_code
        
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        encrypted = store.get_colleague_join_code_encrypted(user_id)
        if encrypted:
            plaintext = decrypt_join_code(encrypted)
            return {
                "success": True,
                "has_code": True,
                "code": plaintext,  # Decrypted for display only
            }
        else:
            return {
                "success": True,
                "has_code": False,
                "code": None,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/settings/colleague-join-code")
async def clear_colleague_join_code():
    """
    Clear the colleague join code (disable email signups).
    """
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        settings = store.clear_colleague_join_code(user_id)
        
        return {
            "success": True,
            "message": "Join code cleared",
            "settings": settings,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/settings/inbox/history")
async def get_inbox_processing_history(
    limit: int = Query(default=50, ge=1, le=500),
    email_type: Optional[str] = Query(default=None),
):
    """
    Get history of processed inbound emails.
    
    Useful for debugging and audit.
    """
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        history = store.list_processed_emails(
            user_id=user_id,
            limit=limit,
            email_type=email_type,
        )
        
        return {
            "success": True,
            "history": history,
            "count": len(history),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/settings/inbox/diagnostics")
async def run_inbox_diagnostics_endpoint(
    since_hours: int = Query(default=48, ge=1, le=168),
):
    """
    Run inbox diagnostics - fetch and analyze recent emails.
    
    Returns a table of recent emails with detected intent, useful for debugging
    why emails are not being processed.
    """
    try:
        from src.tools.inbound_processor import run_inbox_diagnostics
        
        results = run_inbox_diagnostics(since_hours=since_hours)
        
        return {
            "success": True,
            **results,
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# =============================================================================
# Execution Settings API (Papers per run + Execution Mode)
# =============================================================================

class ExecutionSettingsUpdate(BaseModel):
    """Request model for updating execution settings."""
    retrieval_max_results: Optional[int] = None
    execution_mode: Optional[str] = None  # 'manual' | 'scheduled'
    scheduled_frequency: Optional[str] = None  # 'daily' | 'every_x_days' | 'weekly' | 'monthly'
    scheduled_every_x_days: Optional[int] = None


@router.get("/settings/execution")
async def get_execution_settings():
    """
    Get current execution and retrieval settings.
    
    Returns:
        - retrieval_max_results: Papers per run
        - execution_mode: 'manual' | 'scheduled'
        - scheduled_frequency: Schedule cadence
        - scheduled_every_x_days: Days for every_x_days mode
        - last_run_at: Last execution timestamp
        - next_run_at: Next scheduled execution timestamp
    """
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        settings = store.get_or_create_user_settings(user_id)
        
        return {
            "success": True,
            "settings": {
                "retrieval_max_results": settings.get("retrieval_max_results", 7),
                "execution_mode": settings.get("execution_mode", "manual"),
                "scheduled_frequency": settings.get("scheduled_frequency"),
                "scheduled_every_x_days": settings.get("scheduled_every_x_days"),
                "last_run_at": settings.get("last_run_at"),
                "next_run_at": settings.get("next_run_at"),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/settings/execution")
async def update_execution_settings(data: ExecutionSettingsUpdate):
    """
    Update execution and retrieval settings.
    
    Validates input ranges:
    - retrieval_max_results: 1-50
    - execution_mode: 'manual' | 'scheduled'
    - scheduled_frequency: 'daily' | 'every_x_days' | 'weekly' | 'monthly'
    - scheduled_every_x_days: 1-30
    """
    try:
        # Validate inputs
        if data.retrieval_max_results is not None:
            if data.retrieval_max_results < 1 or data.retrieval_max_results > 50:
                raise HTTPException(status_code=400, detail="retrieval_max_results must be between 1 and 50")
        
        if data.execution_mode is not None:
            if data.execution_mode not in ("manual", "scheduled"):
                raise HTTPException(status_code=400, detail="execution_mode must be 'manual' or 'scheduled'")
        
        if data.scheduled_frequency is not None:
            valid_freqs = ("daily", "every_x_days", "weekly", "monthly", "")
            if data.scheduled_frequency not in valid_freqs:
                raise HTTPException(status_code=400, detail=f"Invalid frequency. Valid: {valid_freqs}")
        
        if data.scheduled_every_x_days is not None:
            if data.scheduled_every_x_days < 1 or data.scheduled_every_x_days > 30:
                raise HTTPException(status_code=400, detail="scheduled_every_x_days must be between 1 and 30")
        
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        settings = store.update_execution_settings(
            user_id=user_id,
            retrieval_max_results=data.retrieval_max_results,
            execution_mode=data.execution_mode,
            scheduled_frequency=data.scheduled_frequency,
            scheduled_every_x_days=data.scheduled_every_x_days,
        )
        
        import logging
        logging.info(
            "[SETTINGS] Updated execution settings: retrieval_max_results=%s mode=%s freq=%s",
            settings.get("retrieval_max_results"), settings.get("execution_mode"),
            settings.get("scheduled_frequency"),
        )
        
        return {
            "success": True,
            "message": "Execution settings updated",
            "settings": settings,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Autonomous Components API (Feature-Flagged)
# =============================================================================

# --- Audit Logs ---

@router.get("/audit-logs")
async def list_audit_logs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """
    List run audit logs.
    
    Feature flag: AUDIT_LOG_ENABLED (must be enabled)
    
    Returns list of audit logs ordered by created_at desc.
    """
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        # Check feature flag with user_id for DB-backed flags
        try:
            from ..config.feature_flags import is_feature_enabled
            if not is_feature_enabled("AUDIT_LOG", user_id):
                raise HTTPException(status_code=404, detail="Audit log feature is not enabled")
        except ImportError:
            raise HTTPException(status_code=404, detail="Feature flags not available")
        
        from ..tools.audit_log import get_audit_logs
        
        logs = get_audit_logs(user_id=user_id, limit=limit, offset=offset)
        
        return {
            "success": True,
            "audit_logs": logs,
            "count": len(logs),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit-logs/{run_id}")
async def get_audit_log(run_id: str):
    """
    Get a specific audit log by run_id.
    
    Feature flag: AUDIT_LOG_ENABLED (must be enabled)
    """
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        # Check feature flag with user_id for DB-backed flags
        try:
            from ..config.feature_flags import is_feature_enabled
            if not is_feature_enabled("AUDIT_LOG", user_id):
                raise HTTPException(status_code=404, detail="Audit log feature is not enabled")
        except ImportError:
            raise HTTPException(status_code=404, detail="Feature flags not available")
        
        from ..tools.audit_log import get_audit_log_by_run_id
        
        log = get_audit_log_by_run_id(run_id)
        
        if not log:
            raise HTTPException(status_code=404, detail="Audit log not found")
        
        return {
            "success": True,
            "audit_log": log,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/audit-logs")
async def reset_audit_logs():
    """
    Delete all audit logs for the current user.
    """
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])

        from ..db.database import is_database_configured, get_db_session
        from ..db.orm_models import RunAuditLog

        if not is_database_configured():
            raise HTTPException(status_code=500, detail="Database not configured")

        with get_db_session() as db:
            deleted = db.query(RunAuditLog).filter_by(user_id=user_id).delete()
            db.commit()

        return {"success": True, "deleted_count": deleted}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Profile Evolution Suggestions ---

class SuggestionStatusUpdate(BaseModel):
    """Request model for updating suggestion status."""
    status: str = Field(..., description="New status: accepted, rejected, or expired")


@router.get("/profile-suggestions")
async def list_profile_suggestions(
    status: Optional[str] = Query(default="pending", regex="^(pending|accepted|rejected|expired|all)$"),
    limit: int = Query(default=20, ge=1, le=100),
):
    """
    List profile evolution suggestions.
    
    Feature flag: PROFILE_EVOLUTION_ENABLED (must be enabled)
    
    Args:
        status: Filter by status (pending, accepted, rejected, expired, all)
        limit: Maximum suggestions to return
    
    Returns list of suggestions ordered by created_at desc.
    """
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        # Check feature flag with user_id for DB-backed flags
        try:
            from ..config.feature_flags import is_feature_enabled
            if not is_feature_enabled("PROFILE_EVOLUTION", user_id):
                raise HTTPException(status_code=404, detail="Profile evolution feature is not enabled")
        except ImportError:
            raise HTTPException(status_code=404, detail="Feature flags not available")
        
        from ..agent.profile_evolution import get_pending_suggestions
        from ..db.database import is_database_configured, get_db_session
        from ..db.orm_models import ProfileEvolutionSuggestion, profile_evolution_suggestion_to_dict
        
        if status == "pending":
            suggestions = get_pending_suggestions(user_id)
        else:
            # Query with optional status filter
            if not is_database_configured():
                return {"success": True, "suggestions": [], "count": 0}
            
            with get_db_session() as db:
                query = db.query(ProfileEvolutionSuggestion).filter_by(user_id=user_id)
                if status != "all":
                    query = query.filter_by(status=status)
                
                results = query.order_by(
                    ProfileEvolutionSuggestion.created_at.desc()
                ).limit(limit).all()
                
                suggestions = [profile_evolution_suggestion_to_dict(s) for s in results]
        
        return {
            "success": True,
            "suggestions": suggestions[:limit],
            "count": len(suggestions),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/profile-suggestions/{suggestion_id}")
async def update_profile_suggestion(suggestion_id: str, data: SuggestionStatusUpdate):
    """
    Update a profile suggestion's status (accept/reject).
    
    Feature flag: PROFILE_EVOLUTION_ENABLED (must be enabled)
    
    When accepted, the suggestion is automatically applied to the user's
    research profile (e.g., adding a topic or arXiv category).
    
    Args:
        suggestion_id: UUID of the suggestion
        data: New status (accepted, rejected, expired)
    """
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        # Check feature flag with user_id for DB-backed flags
        try:
            from ..config.feature_flags import is_feature_enabled
            if not is_feature_enabled("PROFILE_EVOLUTION", user_id):
                raise HTTPException(status_code=404, detail="Profile evolution feature is not enabled")
        except ImportError:
            raise HTTPException(status_code=404, detail="Feature flags not available")
        
        # Validate status
        if data.status not in ("accepted", "rejected", "expired"):
            raise HTTPException(status_code=400, detail="Status must be: accepted, rejected, or expired")
        
        from ..agent.profile_evolution import update_suggestion_status
        
        success = update_suggestion_status(
            suggestion_id=UUID(suggestion_id),
            status=data.status,
            reviewed_by=user.get("id"),
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        
        # If accepted, apply the suggestion to the user's profile
        applied = False
        if data.status == "accepted":
            applied = _apply_suggestion_to_profile(store, user_id, UUID(suggestion_id))
        
        return {
            "success": True,
            "message": f"Suggestion marked as {data.status}",
            "applied": applied,
        }
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid suggestion ID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _apply_suggestion_to_profile(store, user_id: UUID, suggestion_id: UUID) -> bool:
    """
    Apply an accepted suggestion to the user's profile.
    
    Reads the suggestion_data from DB and modifies the user's profile
    accordingly (add/remove topics, categories, etc.).
    """
    try:
        from ..db.database import get_db_session
        from ..db.orm_models import ProfileEvolutionSuggestion, User
        
        with get_db_session() as db:
            suggestion = db.query(ProfileEvolutionSuggestion).filter_by(
                id=suggestion_id
            ).first()
            
            if not suggestion or not suggestion.suggestion_data:
                return False
            
            sd = suggestion.suggestion_data
            action = sd.get("action", suggestion.suggestion_type)
            
            user = db.query(User).filter_by(id=user_id).first()
            if not user:
                return False
            
            # Get current profile fields
            interests = user.interests_include or ""
            topics = list(user.research_topics or [])
            cats_include = list(user.arxiv_categories_include or [])
            cats_exclude = list(user.arxiv_categories_exclude or [])
            
            if action == "add_topic":
                new_topic = sd.get("topic", "")
                if new_topic:
                    # Append to interests_include free text
                    if interests:
                        interests = interests.rstrip(", ") + ", " + new_topic
                    else:
                        interests = new_topic
                    # Also add to research_topics list
                    if new_topic not in topics:
                        topics.append(new_topic)
                        
            elif action == "remove_topic":
                old_topic = sd.get("topic", "")
                if old_topic:
                    topics = [t for t in topics if t.lower() != old_topic.lower()]
                    # Also remove from interests_include free text
                    if old_topic in (interests or ""):
                        import re
                        interests = re.sub(
                            r',?\s*' + re.escape(old_topic) + r'\s*,?',
                            ', ', interests, flags=re.IGNORECASE
                        ).strip(', ')
                    
            elif action == "refine_topic":
                old_topic = sd.get("topic", "")
                new_topic = sd.get("new_topic", "")
                if old_topic and new_topic:
                    topics = [new_topic if t.lower() == old_topic.lower() else t for t in topics]
                    if interests:
                        import re
                        interests = re.sub(
                            re.escape(old_topic), new_topic, interests, flags=re.IGNORECASE
                        )
                        
            elif action == "add_category":
                cat = sd.get("category", "")
                if cat and cat not in cats_include:
                    cats_include.append(cat)
                    # Also add human-readable name to interests & topics
                    human_name = None
                    try:
                        from ..tools.arxiv_categories import get_all_categories
                        all_cats = get_all_categories()
                        if cat in all_cats:
                            human_name = all_cats[cat].name
                    except Exception:
                        pass
                    if not human_name:
                        # Fallback: derive from category code
                        human_name = cat.split(".")[-1] if "." in cat else cat
                    if human_name:
                        if interests:
                            interests = interests.rstrip(", ") + ", " + human_name
                        else:
                            interests = human_name
                        if human_name not in topics:
                            topics.append(human_name)
                    
            elif action == "remove_category":
                cat = sd.get("category", "")
                if cat:
                    cats_include = [c for c in cats_include if c != cat]
                    # Also remove human-readable name from interests & topics
                    human_name = None
                    try:
                        from ..tools.arxiv_categories import get_all_categories
                        all_cats = get_all_categories()
                        if cat in all_cats:
                            human_name = all_cats[cat].name
                    except Exception:
                        pass
                    if human_name:
                        topics = [t for t in topics if t != human_name]
                        if human_name in (interests or ""):
                            interests = interests.replace(", " + human_name, "").replace(human_name + ", ", "").replace(human_name, "")
                    
            elif action == "merge_topics":
                to_merge = sd.get("topics_to_merge", [])
                merged = sd.get("merged_topic", "")
                if to_merge and merged:
                    topics = [t for t in topics if t.lower() not in
                              [m.lower() for m in to_merge]]
                    if merged not in topics:
                        topics.append(merged)
                    # Update interests_include: remove merged topics, add merged result
                    import re
                    for old_t in to_merge:
                        interests = re.sub(
                            r',?\s*' + re.escape(old_t) + r'\s*,?',
                            ', ', interests, flags=re.IGNORECASE
                        ).strip(', ')
                    if merged not in (interests or ""):
                        interests = (interests.rstrip(', ') + ', ' + merged) if interests else merged
            
            # Apply updates — derive research_topics from interests_include (single source of truth)
            user.interests_include = interests
            user.research_topics = _parse_interests_to_topics(interests)
            user.arxiv_categories_include = cats_include
            user.arxiv_categories_exclude = cats_exclude
            user.updated_at = __import__('datetime').datetime.utcnow()
            db.commit()

        # Re-score existing papers with the new profile
        try:
            from ..tools.rescore_papers import rescore_papers_for_user
            rescore_papers_for_user(user_id)
        except Exception as rescore_err:
            import logging
            logging.getLogger(__name__).warning(
                f"Post-suggestion paper re-scoring failed (non-critical): {rescore_err}"
            )

        return True
        
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to apply suggestion: {e}")
        return False


# --- Live Document ---

def _filter_live_doc_papers(doc: "LiveDocumentData", user_id: "UUID") -> "LiveDocumentData":
    """Filter out deleted papers and keep only the 10 newest by date."""
    try:
        from ..db.database import is_database_configured, get_db_session
        from ..db.orm_models import Paper, PaperView
        if is_database_configured():
            arxiv_ids = {p.arxiv_id for p in doc.top_papers}
            arxiv_ids.update(p.arxiv_id for p in doc.recent_papers)
            if arxiv_ids:
                with get_db_session() as db:
                    # Query PaperView joined with Paper for this user:
                    # only keep papers that still have an active (non-deleted) view
                    active = (
                        db.query(Paper.external_id)
                        .join(PaperView, PaperView.paper_id == Paper.id)
                        .filter(
                            PaperView.user_id == user_id,
                            Paper.external_id.in_(arxiv_ids),
                            PaperView.is_deleted.is_(False),
                        )
                        .all()
                    )
                    active_ids = {row[0] for row in active}
                before_top = len(doc.top_papers)
                before_recent = len(doc.recent_papers)
                doc.top_papers = [p for p in doc.top_papers if p.arxiv_id in active_ids]
                doc.recent_papers = [p for p in doc.recent_papers if p.arxiv_id in active_ids]
                removed = (before_top + before_recent) - (len(doc.top_papers) + len(doc.recent_papers))
                if removed:
                    logger.info(
                        f"Live document filter: removed {removed} deleted/missing papers for user {user_id}"
                    )
    except Exception as e:
        logger.warning(f"Live document paper filter failed: {e}")  # log instead of silently swallowing

    # Merge top_papers + recent_papers, deduplicate, sort by combined score (high to low), take 10
    seen = set()
    all_papers = []
    for p in list(doc.top_papers) + list(doc.recent_papers):
        if p.arxiv_id not in seen:
            seen.add(p.arxiv_id)
            all_papers.append(p)

    def _combined_score(p):
        novelty = (p.llm_novelty_score / 100.0) if p.llm_novelty_score else (p.novelty_score or 0)
        return p.relevance_score + novelty

    all_papers.sort(key=_combined_score, reverse=True)
    doc.top_papers = all_papers[:10]
    doc.recent_papers = []

    # Patch executive summary so paper count matches reality
    import re
    actual_count = len(doc.top_papers)
    doc.executive_summary = re.sub(
        r"\bthe\s+\d+\s+most\s+relevant",
        f"the {actual_count} most relevant",
        doc.executive_summary,
    )
    # Clean stale trending-topics references from executive summary
    doc.executive_summary = re.sub(
        r"\s+and\s+\d+\s+trending\s+topics?", "", doc.executive_summary
    )

    return doc


@router.get("/live-document")
async def get_live_document(format: str = Query(default="json", regex="^(json|markdown|html|txt|pdf)$")):
    """
    Get the user's live research briefing document.
    
    Feature flag: LIVE_DOCUMENT_ENABLED (must be enabled)
    
    Args:
        format: Response format (json, markdown, html, txt, pdf).
                'pdf' returns an actual PDF binary.
    
    Returns the latest version of the live document.
    """
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        # Check feature flag with user_id for DB-backed flags
        try:
            from ..config.feature_flags import is_feature_enabled
            if not is_feature_enabled("LIVE_DOCUMENT", user_id):
                raise HTTPException(status_code=404, detail="Live document feature is not enabled")
        except ImportError:
            raise HTTPException(status_code=404, detail="Feature flags not available")
        
        from ..tools.live_document import get_live_document, LiveDocumentManager, LiveDocumentData
        
        doc_data = get_live_document(user_id)
        
        if not doc_data:
            raise HTTPException(status_code=404, detail="No live document found. Run a researchPulse first.")
        
        manager = LiveDocumentManager()
        doc = LiveDocumentData(**doc_data.get("document_data", {}))
        doc = _filter_live_doc_papers(doc, user_id)

        logger.info(f"Live document request: format={format}, papers={len(doc.top_papers)}")

        if format == "markdown":
            return Response(
                content=manager.render_markdown(doc),
                media_type="text/markdown",
            )
        elif format == "html":
            return Response(
                content=manager.render_html(doc),
                media_type="text/html",
            )
        elif format == "txt":
            return Response(
                content=manager.render_text(doc),
                media_type="text/plain",
                headers={"Content-Disposition": "attachment; filename=live_document.txt"},
            )
        elif format == "pdf":
            try:
                pdf_bytes = manager.render_pdf(doc)
            except Exception as pdf_err:
                logger.warning(f"Server-side PDF generation unavailable: {pdf_err}")
                raise HTTPException(
                    status_code=501,
                    detail="Server-side PDF generation is not available. Use browser print instead.",
                )
            return Response(
                content=bytes(pdf_bytes),
                media_type="application/pdf",
                headers={"Content-Disposition": 'attachment; filename="live-document.pdf"'},
            )
        else:
            return {
                "success": True,
                "document": doc_data,
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Live document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/live-document/history")
async def get_live_document_history(limit: int = Query(default=10, ge=1, le=50)):
    """
    Get version history of the live document.
    
    Feature flag: LIVE_DOCUMENT_ENABLED (must be enabled)
    
    Returns list of previous document versions.
    """
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        # Check feature flag with user_id for DB-backed flags
        try:
            from ..config.feature_flags import is_feature_enabled
            if not is_feature_enabled("LIVE_DOCUMENT", user_id):
                raise HTTPException(status_code=404, detail="Live document feature is not enabled")
        except ImportError:
            raise HTTPException(status_code=404, detail="Feature flags not available")
        
        from ..tools.live_document import get_live_document, get_document_history
        
        # First get the current document to find its ID
        doc_data = get_live_document(user_id)
        
        if not doc_data:
            raise HTTPException(status_code=404, detail="No live document found")
        
        document_id = doc_data.get("id")
        if not document_id:
            return {"success": True, "history": [], "count": 0}
        
        history = get_document_history(UUID(document_id), limit=limit)
        
        return {
            "success": True,
            "history": history,
            "count": len(history),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/live-document")
async def reset_live_document():
    """
    Delete the live document and its history for the current user.
    """
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])

        from ..db.database import is_database_configured, get_db_session
        from ..db.orm_models import LiveDocument, LiveDocumentHistory

        if not is_database_configured():
            raise HTTPException(status_code=500, detail="Database not configured")

        with get_db_session() as db:
            doc = db.query(LiveDocument).filter_by(user_id=user_id).first()
            history_deleted = 0
            if doc:
                history_deleted = db.query(LiveDocumentHistory).filter_by(
                    document_id=doc.id
                ).delete()
                db.delete(doc)
            db.commit()

        return {"success": True, "document_deleted": doc is not None, "history_deleted": history_deleted}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Feature Flags Status ---

class FeatureFlagUpdate(BaseModel):
    """Request model for updating feature flags."""
    llm_novelty: Optional[dict] = None
    audit_log: Optional[dict] = None
    profile_evolution: Optional[dict] = None
    live_document: Optional[dict] = None


@router.get("/feature-flags")
async def get_feature_flags_status():
    """
    Get status of all autonomous component feature flags.
    
    First tries to load from database, then falls back to environment variables.
    Returns which features are enabled/disabled.
    """
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        # Try to load from DB first
        try:
            from ..config.feature_flags import get_feature_flags_from_db, get_feature_config, reload_feature_config_from_db
            
            # Load from DB and merge with global config
            reload_feature_config_from_db(user_id)
            
            db_flags = get_feature_flags_from_db(user_id)
            if db_flags:
                return {
                    "success": True,
                    "features": db_flags,
                    "source": "database",
                }
            
            # Fall back to environment config
            config = get_feature_config()
            return {
                "success": True,
                "features": {
                    "llm_novelty": {
                        "enabled": config.llm_novelty.enabled,
                        "model": config.llm_novelty.model,
                    },
                    "audit_log": {
                        "enabled": config.audit_log.enabled,
                    },
                    "profile_evolution": {
                        "enabled": config.profile_evolution.enabled,
                        "cooldown_hours": config.profile_evolution.cooldown_hours,
                    },
                    "live_document": {
                        "enabled": config.live_document.enabled,
                        "max_top_papers": config.live_document.max_top_papers,
                    },
                },
                "source": "environment",
            }
        except ImportError:
            return {
                "success": True,
                "features": {
                    "llm_novelty": {"enabled": False, "model": "gpt-4o-mini"},
                    "audit_log": {"enabled": False},
                    "profile_evolution": {"enabled": False, "cooldown_hours": 24},
                    "live_document": {"enabled": False, "max_top_papers": 10},
                },
                "source": "default",
                "message": "Feature flags module not available",
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/feature-flags")
async def update_feature_flags(data: FeatureFlagUpdate):
    """
    Update feature flag settings and save to database.
    
    Args:
        data: Feature flags to update. Only provided fields will be updated.
        
    Example body:
    {
        "llm_novelty": {"enabled": true, "model": "gpt-4o-mini"},
        "audit_log": {"enabled": true},
        "profile_evolution": {"enabled": true, "cooldown_hours": 24},
        "live_document": {"enabled": true, "max_top_papers": 10}
    }
    """
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        from ..config.feature_flags import update_feature_flags_in_db
        
        # Build flags dict from provided data
        flags = {}
        if data.llm_novelty is not None:
            flags["llm_novelty"] = data.llm_novelty
        if data.audit_log is not None:
            flags["audit_log"] = data.audit_log
        if data.profile_evolution is not None:
            flags["profile_evolution"] = data.profile_evolution
        if data.live_document is not None:
            flags["live_document"] = data.live_document
        
        if not flags:
            raise HTTPException(status_code=400, detail="No feature flags provided to update")
        
        success = update_feature_flags_in_db(user_id, flags)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save feature flags to database")
        
        return {
            "success": True,
            "message": "Feature flags updated successfully",
            "updated": list(flags.keys()),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
