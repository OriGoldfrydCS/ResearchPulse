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
from typing import Optional, List, Dict, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response, BackgroundTasks
from pydantic import BaseModel, Field

from ..db.store import get_default_store
from ..db.database import check_connection, is_database_configured


# =============================================================================
# Router Setup
# =============================================================================

router = APIRouter(prefix="/api", tags=["dashboard"])


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
        
        colleague = store.create_colleague(user_id, data.model_dump())
        return colleague
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        from ..agent.react_agent import run_agent
        import uuid as uuid_module
        
        store = get_default_store()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        run_id = str(uuid_module.uuid4())
        prompt = data.prompt or "Process today's papers and make delivery decisions."
        
        # Create run record
        run = store.create_run(user_id, run_id, prompt)
        
        # Run agent in background
        async def execute_run():
            try:
                result = await run_agent(prompt, run_id=run_id)
                store.update_run(run_id, {
                    "status": "done",
                    "report": result,
                    "papers_processed": result.get("papers_processed", 0),
                    "decisions_made": result.get("decisions_made", 0),
                })
            except Exception as e:
                store.update_run(run_id, {
                    "status": "error",
                    "error_message": str(e),
                })
        
        background_tasks.add_task(execute_run)
        
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
        health["database"] = {"status": "not_configured", "message": "DATABASE_URL not set"}
        # In production, this is an error
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
        health["pinecone"] = {"status": "not_configured", "message": "PINECONE_API_KEY not set"}
    
    # Check email provider
    smtp_host = os.getenv("SMTP_HOST")
    resend_key = os.getenv("RESEND_API_KEY")
    sendgrid_key = os.getenv("SENDGRID_API_KEY")
    
    if smtp_host or resend_key or sendgrid_key:
        health["email"] = {"status": "configured", "message": "Email provider configured"}
    else:
        health["email"] = {"status": "not_configured", "message": "No email provider configured"}
    
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
