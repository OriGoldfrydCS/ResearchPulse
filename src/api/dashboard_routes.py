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
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from uuid import UUID

# Load .env file if present (must be done before any os.getenv calls)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from fastapi import APIRouter, HTTPException, Query, Response, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError

from ..db.store import get_default_store
from ..db.database import check_connection, is_database_configured


# =============================================================================
# Router Setup
# =============================================================================

router = APIRouter(prefix="/api", tags=["dashboard"])


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
    name: str
    email: str
    affiliation: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    sharing_preference: str = "weekly"
    enabled: bool = True
    notes: Optional[str] = None


class ColleagueUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    affiliation: Optional[str] = None
    keywords: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    sharing_preference: Optional[str] = None
    enabled: Optional[bool] = None
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
    keywords_include: Optional[List[str]] = None
    keywords_exclude: Optional[List[str]] = None
    colleague_ids_to_always_share: Optional[List[str]] = None


class HealthResponse(BaseModel):
    status: str
    database: Dict[str, Any]
    pinecone: Dict[str, Any]
    email: Dict[str, Any]
    timestamp: str


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
            "keywords_include": user.get("keywords_include", []),
            "keywords_exclude": user.get("keywords_exclude", []),
            "colleague_ids_to_always_share": []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/profile")
async def update_research_profile(data: ProfileUpdate):
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
        if data.interests_exclude is not None:
            update_dict["interests_exclude"] = data.interests_exclude
        if data.keywords_include is not None:
            update_dict["keywords_include"] = data.keywords_include
        if data.keywords_exclude is not None:
            update_dict["keywords_exclude"] = data.keywords_exclude
        
        # Update user in database
        updated_user = store.update_user(user_id, update_dict)
        
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
):
    """List papers with filters."""
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
        )
        return {"papers": papers, "count": len(papers), "offset": offset, "limit": limit}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/papers/{paper_id}")
async def get_paper(paper_id: str):
    """Get a single paper with view details."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        paper_uuid = UUID(paper_id)
        
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
    paper_id: str,
    delete_vectors: bool = Query(True, description="Also delete Pinecone vectors"),
):
    """Delete a paper view and optionally Pinecone vectors."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        paper_uuid = UUID(paper_id)
        
        # Delete the paper view
        deleted = store.delete_paper_view(user_id, paper_uuid)
        
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
async def mark_paper_unseen(paper_id: str):
    """Mark a paper as unseen (delete view record)."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        paper_uuid = UUID(paper_id)
        
        deleted = store.delete_paper_view(user_id, paper_uuid)
        return {"success": deleted, "paper_id": paper_id}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid paper ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/papers/{paper_id}")
async def update_paper_view(paper_id: str, data: PaperViewUpdate):
    """Update paper view (notes, tags, importance, decision)."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        paper_uuid = UUID(paper_id)
        
        update_data = data.model_dump(exclude_none=True)
        updated = store.update_paper_view(user_id, paper_uuid, update_data)
        return updated
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
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


# =============================================================================
# Colleague Endpoints
# =============================================================================

@router.get("/colleagues")
async def list_colleagues(
    enabled_only: bool = Query(False, description="Only show enabled colleagues"),
):
    """List all colleagues."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        colleagues = store.list_colleagues(user_id, enabled_only=enabled_only)
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
    """Create a new colleague."""
    try:
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        colleague_data = data.model_dump()
        # Ensure proper defaults
        colleague_data.setdefault("keywords", [])
        colleague_data.setdefault("categories", [])
        colleague_data.setdefault("topics", [])
        colleague_data.setdefault("sharing_preference", "weekly")
        colleague_data.setdefault("enabled", True)
        
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
    """Update a colleague."""
    try:
        colleague_uuid = UUID(colleague_id)
        store = get_default_store()
        
        update_data = data.model_dump(exclude_none=True)
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
                            paper_record = store.upsert_paper({
                                "source": "arxiv",
                                "external_id": external_id,
                                "title": paper.get("title", "Untitled"),
                                "abstract": paper.get("abstract", ""),
                                "authors": paper.get("authors", []),
                                "categories": paper.get("categories", []),
                                "url": paper.get("url"),
                                "pdf_url": paper.get("pdf_url"),
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
# Bulk Actions
# =============================================================================

@router.post("/papers/bulk/delete")
async def bulk_delete_papers(paper_ids: List[str]):
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


@router.post("/papers/bulk/mark-unseen")
async def bulk_mark_unseen(paper_ids: List[str]):
    """Mark multiple papers as unseen."""
    return await bulk_delete_papers(paper_ids)


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
        all_papers = store.list_papers(user_id, limit=10000)
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
