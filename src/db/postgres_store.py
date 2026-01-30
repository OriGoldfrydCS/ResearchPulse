"""
PostgresStore - PostgreSQL/Supabase implementation of the Store interface.

Uses SQLAlchemy with DATABASE_URL for cloud-safe persistence.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID

from sqlalchemy import and_, or_, func
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import IntegrityError

from .store import Store
from .database import get_db_session, check_connection, is_database_configured
from .orm_models import (
    User, Paper, PaperView, Colleague, Run, Action, Email, CalendarEvent, Share, DeliveryPolicy,
    user_to_dict, paper_to_dict, paper_view_to_dict, colleague_to_dict,
    run_to_dict, email_to_dict, calendar_event_to_dict, share_to_dict, policy_to_dict,
)


class PostgresStore(Store):
    """PostgreSQL implementation of the Store interface."""
    
    def __init__(self):
        if not is_database_configured():
            raise RuntimeError("DATABASE_URL not configured. Set DATABASE_URL environment variable.")
    
    # =========================================================================
    # User Operations
    # =========================================================================
    
    def get_or_create_default_user(self) -> Dict[str, Any]:
        """Get or create the default user for single-user mode."""
        with get_db_session() as db:
            # Look for existing user
            user = db.query(User).first()
            if user:
                return user_to_dict(user)
            
            # Create default user from research_profile.json if exists
            default_data = self._load_default_profile()
            
            user = User(
                name=default_data.get("researcher_name", "Default User"),
                email=default_data.get("email", "user@researchpulse.local"),
                affiliation=default_data.get("affiliation"),
                research_topics=default_data.get("research_topics", []),
                my_papers=default_data.get("my_papers", []),
                preferred_venues=default_data.get("preferred_venues", []),
                avoid_topics=default_data.get("avoid_topics", []),
                time_budget_per_week_minutes=default_data.get("time_budget_per_week_minutes", 120),
                arxiv_categories_include=default_data.get("arxiv_categories_include", []),
                arxiv_categories_exclude=default_data.get("arxiv_categories_exclude", []),
                stop_policy=default_data.get("stop_policy", {}),
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            return user_to_dict(user)
    
    def _load_default_profile(self) -> Dict[str, Any]:
        """Load default profile from research_profile.json if it exists."""
        import json
        from pathlib import Path
        
        profile_path = Path("data/research_profile.json")
        if profile_path.exists():
            try:
                with open(profile_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def get_user(self, user_id: UUID) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        with get_db_session() as db:
            user = db.query(User).filter(User.id == user_id).first()
            return user_to_dict(user) if user else None
    
    def update_user(self, user_id: UUID, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user profile."""
        with get_db_session() as db:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError(f"User not found: {user_id}")
            
            for key, value in data.items():
                if hasattr(user, key) and key not in ("id", "created_at"):
                    setattr(user, key, value)
            
            user.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(user)
            return user_to_dict(user)
    
    def upsert_user(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update user from data dict."""
        with get_db_session() as db:
            # Try to find existing user by email
            email = data.get("email")
            user = db.query(User).filter(User.email == email).first() if email else None
            
            if not user:
                # Create new user
                user = User(
                    name=data.get("researcher_name") or data.get("name", "User"),
                    email=email or "user@researchpulse.local",
                    affiliation=data.get("affiliation"),
                    research_topics=data.get("research_topics", []),
                    my_papers=data.get("my_papers", []),
                    preferred_venues=data.get("preferred_venues", []),
                    avoid_topics=data.get("avoid_topics", []),
                    time_budget_per_week_minutes=data.get("time_budget_per_week_minutes", 120),
                    arxiv_categories_include=data.get("arxiv_categories_include", []),
                    arxiv_categories_exclude=data.get("arxiv_categories_exclude", []),
                    stop_policy=data.get("stop_policy", {}),
                )
                db.add(user)
            else:
                # Update existing user
                for key in ["affiliation", "research_topics", "my_papers", "preferred_venues", 
                           "avoid_topics", "time_budget_per_week_minutes", "arxiv_categories_include",
                           "arxiv_categories_exclude", "stop_policy"]:
                    if key in data:
                        setattr(user, key, data[key])
                if "researcher_name" in data:
                    user.name = data["researcher_name"]
                user.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(user)
            return user_to_dict(user)
    
    # =========================================================================
    # Paper Operations
    # =========================================================================
    
    def upsert_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Upsert a paper by source + external_id."""
        source = paper_data.get("source", "arxiv")
        external_id = paper_data.get("external_id") or paper_data.get("paper_id")
        
        if not external_id:
            raise ValueError("external_id or paper_id is required")
        
        with get_db_session() as db:
            paper = db.query(Paper).filter(
                and_(Paper.source == source, Paper.external_id == external_id)
            ).first()
            
            if paper:
                # Update existing
                for key in ["title", "abstract", "authors", "categories", "url", "pdf_url", "published_at"]:
                    if key in paper_data and paper_data[key] is not None:
                        setattr(paper, key, paper_data[key])
            else:
                # Create new
                paper = Paper(
                    source=source,
                    external_id=external_id,
                    title=paper_data.get("title", "Untitled"),
                    abstract=paper_data.get("abstract"),
                    authors=paper_data.get("authors", []),
                    categories=paper_data.get("categories", []),
                    url=paper_data.get("url"),
                    pdf_url=paper_data.get("pdf_url"),
                    published_at=paper_data.get("published_at"),
                )
                db.add(paper)
            
            db.commit()
            db.refresh(paper)
            return paper_to_dict(paper)
    
    def get_paper(self, paper_id: UUID) -> Optional[Dict[str, Any]]:
        """Get paper by ID."""
        with get_db_session() as db:
            paper = db.query(Paper).filter(Paper.id == paper_id).first()
            return paper_to_dict(paper) if paper else None
    
    def get_paper_by_external_id(self, source: str, external_id: str) -> Optional[Dict[str, Any]]:
        """Get paper by source and external ID."""
        with get_db_session() as db:
            paper = db.query(Paper).filter(
                and_(Paper.source == source, Paper.external_id == external_id)
            ).first()
            return paper_to_dict(paper) if paper else None
    
    def list_papers(
        self,
        user_id: UUID,
        seen: Optional[bool] = None,
        decision: Optional[str] = None,
        importance: Optional[str] = None,
        category: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List papers with filters."""
        with get_db_session() as db:
            q = db.query(PaperView).options(joinedload(PaperView.paper)).filter(
                PaperView.user_id == user_id
            )
            
            if decision:
                q = q.filter(PaperView.decision == decision)
            
            if importance:
                q = q.filter(PaperView.importance == importance)
            
            if query:
                # Search in title and abstract
                q = q.join(Paper).filter(
                    or_(
                        Paper.title.ilike(f"%{query}%"),
                        Paper.abstract.ilike(f"%{query}%"),
                    )
                )
            
            if category:
                # Filter by category in JSON array
                q = q.join(Paper).filter(
                    Paper.categories.contains([category])
                )
            
            q = q.order_by(PaperView.last_seen_at.desc())
            q = q.offset(offset).limit(limit)
            
            views = q.all()
            return [paper_view_to_dict(v, include_paper=True) for v in views]
    
    def delete_paper_view(self, user_id: UUID, paper_id: UUID) -> bool:
        """Delete a paper view (mark as unseen)."""
        with get_db_session() as db:
            view = db.query(PaperView).filter(
                and_(PaperView.user_id == user_id, PaperView.paper_id == paper_id)
            ).first()
            
            if view:
                db.delete(view)
                db.commit()
                return True
            return False
    
    def delete_paper(self, paper_id: UUID) -> bool:
        """Delete a paper and all related data."""
        with get_db_session() as db:
            paper = db.query(Paper).filter(Paper.id == paper_id).first()
            if paper:
                db.delete(paper)
                db.commit()
                return True
            return False
    
    # =========================================================================
    # Paper View Operations
    # =========================================================================
    
    def upsert_paper_view(self, user_id: UUID, paper_id: UUID, view_data: Dict[str, Any]) -> Dict[str, Any]:
        """Upsert a paper view."""
        with get_db_session() as db:
            view = db.query(PaperView).filter(
                and_(PaperView.user_id == user_id, PaperView.paper_id == paper_id)
            ).first()
            
            if view:
                # Update existing
                view.last_seen_at = datetime.utcnow()
                view.seen_count = (view.seen_count or 0) + 1
                for key in ["decision", "importance", "relevance_score", "novelty_score", "heuristic_score", "notes", "tags", "embedded_in_pinecone"]:
                    if key in view_data:
                        setattr(view, key, view_data[key])
            else:
                # Create new
                view = PaperView(
                    user_id=user_id,
                    paper_id=paper_id,
                    decision=view_data.get("decision", "logged"),
                    importance=view_data.get("importance", "low"),
                    relevance_score=view_data.get("relevance_score"),
                    novelty_score=view_data.get("novelty_score"),
                    heuristic_score=view_data.get("heuristic_score"),
                    notes=view_data.get("notes"),
                    tags=view_data.get("tags", []),
                    embedded_in_pinecone=view_data.get("embedded_in_pinecone", False),
                )
                db.add(view)
            
            db.commit()
            db.refresh(view)
            return paper_view_to_dict(view)
    
    def get_paper_view(self, user_id: UUID, paper_id: UUID) -> Optional[Dict[str, Any]]:
        """Get a paper view."""
        with get_db_session() as db:
            view = db.query(PaperView).options(joinedload(PaperView.paper)).filter(
                and_(PaperView.user_id == user_id, PaperView.paper_id == paper_id)
            ).first()
            return paper_view_to_dict(view, include_paper=True) if view else None
    
    def is_paper_seen(self, user_id: UUID, source: str, external_id: str) -> bool:
        """Check if a paper has been seen by user."""
        with get_db_session() as db:
            result = db.query(PaperView).join(Paper).filter(
                and_(
                    PaperView.user_id == user_id,
                    Paper.source == source,
                    Paper.external_id == external_id,
                )
            ).first()
            return result is not None
    
    def update_paper_view(self, user_id: UUID, paper_id: UUID, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update paper view (notes, tags, etc.)."""
        with get_db_session() as db:
            view = db.query(PaperView).filter(
                and_(PaperView.user_id == user_id, PaperView.paper_id == paper_id)
            ).first()
            
            if not view:
                raise ValueError(f"Paper view not found: user={user_id}, paper={paper_id}")
            
            for key, value in data.items():
                if hasattr(view, key) and key not in ("id", "user_id", "paper_id", "first_seen_at"):
                    setattr(view, key, value)
            
            view.last_seen_at = datetime.utcnow()
            db.commit()
            db.refresh(view)
            return paper_view_to_dict(view)
    
    # =========================================================================
    # Colleague Operations
    # =========================================================================
    
    def list_colleagues(self, user_id: UUID, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """List all colleagues for a user."""
        with get_db_session() as db:
            q = db.query(Colleague).filter(Colleague.user_id == user_id)
            if enabled_only:
                q = q.filter(Colleague.enabled == True)
            q = q.order_by(Colleague.name)
            return [colleague_to_dict(c) for c in q.all()]
    
    def get_colleague(self, colleague_id: UUID) -> Optional[Dict[str, Any]]:
        """Get colleague by ID."""
        with get_db_session() as db:
            colleague = db.query(Colleague).filter(Colleague.id == colleague_id).first()
            return colleague_to_dict(colleague) if colleague else None
    
    def create_colleague(self, user_id: UUID, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new colleague."""
        with get_db_session() as db:
            colleague = Colleague(
                user_id=user_id,
                name=data["name"],
                email=data["email"],
                affiliation=data.get("affiliation"),
                keywords=data.get("keywords", []),
                categories=data.get("categories", []),
                topics=data.get("topics", []),
                sharing_preference=data.get("sharing_preference", "weekly"),
                enabled=data.get("enabled", True),
                notes=data.get("notes"),
            )
            db.add(colleague)
            db.commit()
            db.refresh(colleague)
            return colleague_to_dict(colleague)
    
    def update_colleague(self, colleague_id: UUID, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update colleague."""
        with get_db_session() as db:
            colleague = db.query(Colleague).filter(Colleague.id == colleague_id).first()
            if not colleague:
                raise ValueError(f"Colleague not found: {colleague_id}")
            
            for key, value in data.items():
                if hasattr(colleague, key) and key not in ("id", "user_id", "created_at"):
                    setattr(colleague, key, value)
            
            colleague.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(colleague)
            return colleague_to_dict(colleague)
    
    def delete_colleague(self, colleague_id: UUID) -> bool:
        """Delete colleague."""
        with get_db_session() as db:
            colleague = db.query(Colleague).filter(Colleague.id == colleague_id).first()
            if colleague:
                db.delete(colleague)
                db.commit()
                return True
            return False
    
    def upsert_colleague(self, user_id: UUID, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update colleague by email."""
        email = data.get("email")
        if not email:
            raise ValueError("Colleague email is required")
        
        with get_db_session() as db:
            colleague = db.query(Colleague).filter(
                and_(Colleague.user_id == user_id, Colleague.email == email)
            ).first()
            
            if colleague:
                # Update existing
                for key in ["name", "affiliation", "keywords", "categories", "topics", 
                           "sharing_preference", "enabled", "notes"]:
                    if key in data:
                        setattr(colleague, key, data[key])
                colleague.updated_at = datetime.utcnow()
            else:
                # Create new
                colleague = Colleague(
                    user_id=user_id,
                    name=data.get("name", email.split("@")[0]),
                    email=email,
                    affiliation=data.get("affiliation"),
                    keywords=data.get("keywords", []),
                    categories=data.get("categories", []),
                    topics=data.get("topics", []),
                    sharing_preference=data.get("sharing_preference", "weekly"),
                    enabled=data.get("enabled", True),
                    notes=data.get("notes"),
                )
                db.add(colleague)
            
            db.commit()
            db.refresh(colleague)
            return colleague_to_dict(colleague)
    
    # =========================================================================
    # Run Operations
    # =========================================================================
    
    def create_run(self, user_id: UUID, run_id: str, prompt: str) -> Dict[str, Any]:
        """Create a new run record."""
        with get_db_session() as db:
            run = Run(
                user_id=user_id,
                run_id=run_id,
                user_prompt=prompt,
                status="running",
                started_at=datetime.utcnow(),
            )
            db.add(run)
            db.commit()
            db.refresh(run)
            return run_to_dict(run)
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run by run_id."""
        with get_db_session() as db:
            run = db.query(Run).filter(Run.run_id == run_id).first()
            return run_to_dict(run) if run else None
    
    def update_run(self, run_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update run status, metrics, etc."""
        with get_db_session() as db:
            run = db.query(Run).filter(Run.run_id == run_id).first()
            if not run:
                raise ValueError(f"Run not found: {run_id}")
            
            for key, value in data.items():
                if hasattr(run, key) and key not in ("id", "run_id", "user_id", "created_at"):
                    setattr(run, key, value)
            
            if data.get("status") in ("done", "error"):
                run.ended_at = datetime.utcnow()
            
            db.commit()
            db.refresh(run)
            return run_to_dict(run)
    
    def list_runs(self, user_id: UUID, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List runs for a user."""
        with get_db_session() as db:
            runs = db.query(Run).filter(Run.user_id == user_id).order_by(
                Run.started_at.desc()
            ).offset(offset).limit(limit).all()
            return [run_to_dict(r) for r in runs]
    
    # =========================================================================
    # Action Operations
    # =========================================================================
    
    def create_action(
        self,
        run_id: UUID,
        user_id: UUID,
        action_type: str,
        paper_id: Optional[UUID] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record an action taken during a run."""
        with get_db_session() as db:
            # Get run by run_id string to get its UUID
            run = db.query(Run).filter(Run.run_id == str(run_id)).first()
            if not run:
                # Try treating run_id as UUID directly
                try:
                    run = db.query(Run).filter(Run.id == run_id).first()
                except:
                    pass
            
            actual_run_id = run.id if run else None
            
            action = Action(
                run_id=actual_run_id,
                user_id=user_id,
                paper_id=paper_id,
                action_type=action_type,
                payload=payload or {},
            )
            db.add(action)
            db.commit()
            db.refresh(action)
            
            return {
                "id": str(action.id),
                "run_id": str(action.run_id) if action.run_id else None,
                "user_id": str(action.user_id),
                "paper_id": str(action.paper_id) if action.paper_id else None,
                "action_type": action.action_type,
                "payload": action.payload,
                "status": action.status,
                "created_at": action.created_at.isoformat() if action.created_at else None,
            }
    
    # =========================================================================
    # Email Operations
    # =========================================================================
    
    def create_email(
        self,
        user_id: UUID,
        paper_id: Optional[UUID],
        recipient_email: str,
        subject: str,
        body_text: str,
        body_preview: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create an email record."""
        with get_db_session() as db:
            email = Email(
                user_id=user_id,
                paper_id=paper_id,
                recipient_email=recipient_email,
                subject=subject,
                body_text=body_text,
                body_preview=body_preview or body_text[:500] if body_text else None,
                status="queued",
            )
            db.add(email)
            try:
                db.commit()
                db.refresh(email)
                return email_to_dict(email)
            except IntegrityError:
                db.rollback()
                # Already exists (idempotency)
                existing = db.query(Email).filter(
                    and_(Email.paper_id == paper_id, Email.recipient_email == recipient_email)
                ).first()
                return email_to_dict(existing) if existing else {"error": "duplicate"}
    
    def update_email_status(self, email_id: UUID, status: str, error: Optional[str] = None, provider_id: Optional[str] = None) -> Dict[str, Any]:
        """Update email status after send attempt."""
        with get_db_session() as db:
            email = db.query(Email).filter(Email.id == email_id).first()
            if not email:
                raise ValueError(f"Email not found: {email_id}")
            
            email.status = status
            email.error = error
            email.provider_message_id = provider_id
            if status == "sent":
                email.sent_at = datetime.utcnow()
            
            db.commit()
            db.refresh(email)
            return email_to_dict(email)
    
    def list_emails(self, user_id: UUID, status: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List emails for a user."""
        with get_db_session() as db:
            q = db.query(Email).filter(Email.user_id == user_id)
            if status:
                q = q.filter(Email.status == status)
            q = q.order_by(Email.created_at.desc())
            q = q.offset(offset).limit(limit)
            return [email_to_dict(e) for e in q.all()]
    
    def email_exists(self, paper_id: UUID, recipient_email: str) -> bool:
        """Check if email already sent for paper to recipient (idempotency)."""
        with get_db_session() as db:
            result = db.query(Email).filter(
                and_(Email.paper_id == paper_id, Email.recipient_email == recipient_email)
            ).first()
            return result is not None
    
    # =========================================================================
    # Calendar Event Operations
    # =========================================================================
    
    def create_calendar_event(
        self,
        user_id: UUID,
        paper_id: Optional[UUID],
        title: str,
        start_time: datetime,
        duration_minutes: int = 30,
        ics_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a calendar event record."""
        with get_db_session() as db:
            event = CalendarEvent(
                user_id=user_id,
                paper_id=paper_id,
                title=title,
                start_time=start_time,
                duration_minutes=duration_minutes,
                ics_text=ics_text,
                status="created",
            )
            db.add(event)
            try:
                db.commit()
                db.refresh(event)
                return calendar_event_to_dict(event)
            except IntegrityError:
                db.rollback()
                # Already exists (idempotency)
                existing = db.query(CalendarEvent).filter(
                    and_(CalendarEvent.paper_id == paper_id, CalendarEvent.start_time == start_time)
                ).first()
                return calendar_event_to_dict(existing) if existing else {"error": "duplicate"}
    
    def list_calendar_events(self, user_id: UUID, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List calendar events for a user."""
        with get_db_session() as db:
            events = db.query(CalendarEvent).filter(CalendarEvent.user_id == user_id).order_by(
                CalendarEvent.start_time.desc()
            ).offset(offset).limit(limit).all()
            return [calendar_event_to_dict(e) for e in events]
    
    def calendar_event_exists(self, paper_id: UUID, start_time: datetime) -> bool:
        """Check if calendar event exists (idempotency)."""
        with get_db_session() as db:
            result = db.query(CalendarEvent).filter(
                and_(CalendarEvent.paper_id == paper_id, CalendarEvent.start_time == start_time)
            ).first()
            return result is not None
    
    def insert_calendar_event(
        self,
        user_id: UUID,
        paper_id: Optional[UUID],
        title: str,
        start_time: datetime,
        duration_minutes: int = 30,
        ics_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Alias for create_calendar_event for backward compatibility."""
        return self.create_calendar_event(
            user_id=user_id,
            paper_id=paper_id,
            title=title,
            start_time=start_time,
            duration_minutes=duration_minutes,
            ics_text=ics_text,
        )
    
    # =========================================================================
    # Share Operations
    # =========================================================================
    
    def create_share(
        self,
        user_id: UUID,
        paper_id: UUID,
        colleague_id: UUID,
        reason: Optional[str] = None,
        match_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create a share record."""
        with get_db_session() as db:
            share = Share(
                user_id=user_id,
                paper_id=paper_id,
                colleague_id=colleague_id,
                reason=reason,
                match_score=match_score,
                status="pending",
            )
            db.add(share)
            try:
                db.commit()
                db.refresh(share)
                return share_to_dict(share)
            except IntegrityError:
                db.rollback()
                # Already exists (idempotency)
                existing = db.query(Share).filter(
                    and_(Share.paper_id == paper_id, Share.colleague_id == colleague_id)
                ).first()
                return share_to_dict(existing) if existing else {"error": "duplicate"}
    
    def update_share_status(self, share_id: UUID, status: str, error: Optional[str] = None) -> Dict[str, Any]:
        """Update share status after send attempt."""
        with get_db_session() as db:
            share = db.query(Share).filter(Share.id == share_id).first()
            if not share:
                raise ValueError(f"Share not found: {share_id}")
            
            share.status = status
            share.error = error
            db.commit()
            db.refresh(share)
            return share_to_dict(share)
    
    def list_shares(self, user_id: UUID, colleague_id: Optional[UUID] = None, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List shares for a user."""
        with get_db_session() as db:
            q = db.query(Share).options(
                joinedload(Share.colleague),
                joinedload(Share.paper)
            ).filter(Share.user_id == user_id)
            
            if colleague_id:
                q = q.filter(Share.colleague_id == colleague_id)
            
            q = q.order_by(Share.created_at.desc())
            q = q.offset(offset).limit(limit)
            return [share_to_dict(s, include_details=True) for s in q.all()]
    
    def share_exists(self, paper_id: UUID, colleague_id: UUID) -> bool:
        """Check if share already exists (idempotency)."""
        with get_db_session() as db:
            result = db.query(Share).filter(
                and_(Share.paper_id == paper_id, Share.colleague_id == colleague_id)
            ).first()
            return result is not None
    
    # =========================================================================
    # Delivery Policy Operations
    # =========================================================================
    
    def get_delivery_policy(self, user_id: UUID) -> Optional[Dict[str, Any]]:
        """Get delivery policy for user."""
        with get_db_session() as db:
            policy = db.query(DeliveryPolicy).filter(DeliveryPolicy.user_id == user_id).first()
            return policy_to_dict(policy) if policy else None
    
    def upsert_delivery_policy(self, user_id: UUID, policy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update delivery policy."""
        with get_db_session() as db:
            policy = db.query(DeliveryPolicy).filter(DeliveryPolicy.user_id == user_id).first()
            
            if policy:
                policy.policy_json = policy_data
                policy.colleague_share_enabled = policy_data.get("colleague_share_enabled", True)
                policy.colleague_share_min_score = policy_data.get("colleague_share_min_score", 0.5)
                policy.digest_mode = policy_data.get("digest_mode", False)
                policy.updated_at = datetime.utcnow()
            else:
                policy = DeliveryPolicy(
                    user_id=user_id,
                    policy_json=policy_data,
                    colleague_share_enabled=policy_data.get("colleague_share_enabled", True),
                    colleague_share_min_score=policy_data.get("colleague_share_min_score", 0.5),
                    digest_mode=policy_data.get("digest_mode", False),
                )
                db.add(policy)
            
            db.commit()
            db.refresh(policy)
            return policy_to_dict(policy)
    
    # =========================================================================
    # Health Check
    # =========================================================================
    
    def health_check(self) -> tuple[bool, str]:
        """Check store health. Returns (healthy, message)."""
        return check_connection()
