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

from sqlalchemy import and_, or_, func, case, desc, asc
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import IntegrityError

from .store import Store
from .database import get_db_session, check_connection, is_database_configured
from .orm_models import (
    User, Paper, PaperView, Colleague, Run, Action, Email, CalendarEvent, Share, DeliveryPolicy,
    PaperFeedbackHistory, CalendarInviteEmail, InboundEmailReply, UserSettings, ProcessedInboundEmail,
    user_to_dict, paper_to_dict, paper_view_to_dict, colleague_to_dict,
    run_to_dict, email_to_dict, calendar_event_to_dict, share_to_dict, policy_to_dict,
    feedback_history_to_dict, calendar_invite_email_to_dict, inbound_email_reply_to_dict,
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
        sort_by: str = "added_at",
        sort_dir: str = "desc",
        is_starred: Optional[bool] = None,
        is_read: Optional[bool] = None,
        relevance_state: Optional[str] = None,
        include_not_relevant: bool = False,
    ) -> List[Dict[str, Any]]:
        """List papers with filters and sorting.
        
        Args:
            user_id: User ID to filter by
            seen: Filter by seen status
            decision: Filter by decision status
            importance: Filter by importance level
            category: Filter by arXiv category
            query: Text search in title/abstract
            limit: Max results to return
            offset: Results offset for pagination
            sort_by: Sort field ('added_at', 'published_at', 'importance', 'title')
            sort_dir: Sort direction ('asc' or 'desc')
            is_starred: Filter by starred status
            is_read: Filter by read status
            relevance_state: Filter by relevance state ('relevant', 'not_relevant')
            include_not_relevant: Whether to include not_relevant papers (default False)
        """
        with get_db_session() as db:
            q = db.query(PaperView).options(joinedload(PaperView.paper)).filter(
                PaperView.user_id == user_id
            )
            
            # By default, exclude not_relevant papers unless explicitly requested
            if not include_not_relevant and relevance_state != 'not_relevant':
                q = q.filter(or_(
                    PaperView.relevance_state == None,
                    PaperView.relevance_state != 'not_relevant'
                ))
            
            if relevance_state:
                q = q.filter(PaperView.relevance_state == relevance_state)
            
            if decision:
                q = q.filter(PaperView.decision == decision)
            
            if importance:
                q = q.filter(PaperView.importance == importance)
            
            if is_starred is not None:
                q = q.filter(PaperView.is_starred == is_starred)
            
            if is_read is not None:
                q = q.filter(PaperView.is_read == is_read)
            
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
            
            # Dynamic sorting
            is_descending = sort_dir.lower() == "desc"
            
            if sort_by == "added_at":
                sort_col = PaperView.first_seen_at
                q = q.order_by(desc(sort_col) if is_descending else asc(sort_col))
            elif sort_by == "published_at":
                q = q.join(Paper, isouter=True)
                sort_col = func.coalesce(Paper.published_at, PaperView.first_seen_at)
                q = q.order_by(desc(sort_col) if is_descending else asc(sort_col))
            elif sort_by == "importance":
                # Order by importance level: critical > high > medium > low
                importance_order = case(
                    (PaperView.importance == 'critical', 4),
                    (PaperView.importance == 'high', 3),
                    (PaperView.importance == 'medium', 2),
                    else_=1
                )
                q = q.order_by(desc(importance_order) if is_descending else asc(importance_order))
            elif sort_by == "title":
                q = q.join(Paper, isouter=True)
                q = q.order_by(desc(Paper.title) if is_descending else asc(Paper.title))
            else:
                # Default fallback
                sort_col = PaperView.last_seen_at
                q = q.order_by(desc(sort_col) if is_descending else asc(sort_col))
            
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
    
    def record_feedback_history(
        self,
        db: Session,
        paper_view_id: UUID,
        user_id: UUID,
        paper_id: UUID,
        action_type: str,
        old_value: Optional[str],
        new_value: Optional[str],
        note: Optional[str] = None,
    ) -> PaperFeedbackHistory:
        """Record a feedback change in history. Must be called within an active session."""
        history = PaperFeedbackHistory(
            paper_view_id=paper_view_id,
            user_id=user_id,
            paper_id=paper_id,
            action_type=action_type,
            old_value=old_value,
            new_value=new_value,
            note=note,
        )
        db.add(history)
        return history
    
    def toggle_star(self, user_id: UUID, paper_id: UUID) -> Dict[str, Any]:
        """Toggle starred status for a paper."""
        with get_db_session() as db:
            view = db.query(PaperView).filter(
                and_(PaperView.user_id == user_id, PaperView.paper_id == paper_id)
            ).first()
            
            if not view:
                raise ValueError(f"Paper view not found: user={user_id}, paper={paper_id}")
            
            old_value = str(view.is_starred) if view.is_starred is not None else None
            view.is_starred = not view.is_starred
            new_value = str(view.is_starred)
            view.starred_at = datetime.utcnow() if view.is_starred else None
            
            # Record in feedback history
            self.record_feedback_history(
                db=db,
                paper_view_id=view.id,
                user_id=user_id,
                paper_id=paper_id,
                action_type="star_toggle",
                old_value=old_value,
                new_value=new_value,
            )
            
            db.commit()
            db.refresh(view)
            return paper_view_to_dict(view)
    
    def toggle_read(self, user_id: UUID, paper_id: UUID) -> Dict[str, Any]:
        """Toggle read status for a paper."""
        with get_db_session() as db:
            view = db.query(PaperView).filter(
                and_(PaperView.user_id == user_id, PaperView.paper_id == paper_id)
            ).first()
            
            if not view:
                raise ValueError(f"Paper view not found: user={user_id}, paper={paper_id}")
            
            old_value = str(view.is_read) if view.is_read is not None else None
            view.is_read = not view.is_read
            new_value = str(view.is_read)
            view.read_at = datetime.utcnow() if view.is_read else None
            
            # Record in feedback history
            self.record_feedback_history(
                db=db,
                paper_view_id=view.id,
                user_id=user_id,
                paper_id=paper_id,
                action_type="read_toggle",
                old_value=old_value,
                new_value=new_value,
            )
            
            db.commit()
            db.refresh(view)
            return paper_view_to_dict(view)
    
    def mark_read(self, user_id: UUID, paper_id: UUID) -> Dict[str, Any]:
        """Mark a paper as read (idempotent)."""
        with get_db_session() as db:
            view = db.query(PaperView).filter(
                and_(PaperView.user_id == user_id, PaperView.paper_id == paper_id)
            ).first()
            
            if not view:
                raise ValueError(f"Paper view not found: user={user_id}, paper={paper_id}")
            
            if not view.is_read:
                old_value = str(view.is_read) if view.is_read is not None else None
                view.is_read = True
                view.read_at = datetime.utcnow()
                
                # Record in feedback history
                self.record_feedback_history(
                    db=db,
                    paper_view_id=view.id,
                    user_id=user_id,
                    paper_id=paper_id,
                    action_type="read_toggle",
                    old_value=old_value,
                    new_value="True",
                )
                
                db.commit()
                db.refresh(view)
            
            return paper_view_to_dict(view)
    
    def toggle_relevance(self, user_id: UUID, paper_id: UUID, note: Optional[str] = None) -> Dict[str, Any]:
        """Toggle relevance state for a paper (relevant <-> not_relevant)."""
        with get_db_session() as db:
            view = db.query(PaperView).filter(
                and_(PaperView.user_id == user_id, PaperView.paper_id == paper_id)
            ).first()
            
            if not view:
                raise ValueError(f"Paper view not found: user={user_id}, paper={paper_id}")
            
            old_value = view.relevance_state
            
            # Toggle: if not_relevant -> relevant, otherwise -> not_relevant
            if view.relevance_state == "not_relevant":
                view.relevance_state = "relevant"
                view.is_relevant = True
            else:
                view.relevance_state = "not_relevant"
                view.is_relevant = False
            
            view.feedback_timestamp = datetime.utcnow()
            view.feedback_reason = note
            
            # Record in feedback history
            self.record_feedback_history(
                db=db,
                paper_view_id=view.id,
                user_id=user_id,
                paper_id=paper_id,
                action_type="relevance_change",
                old_value=old_value,
                new_value=view.relevance_state,
                note=note,
            )
            
            db.commit()
            db.refresh(view)
            return paper_view_to_dict(view)
    
    def set_relevance(self, user_id: UUID, paper_id: UUID, relevance_state: str, note: Optional[str] = None) -> Dict[str, Any]:
        """Set explicit relevance state for a paper."""
        with get_db_session() as db:
            view = db.query(PaperView).filter(
                and_(PaperView.user_id == user_id, PaperView.paper_id == paper_id)
            ).first()
            
            if not view:
                raise ValueError(f"Paper view not found: user={user_id}, paper={paper_id}")
            
            old_value = view.relevance_state
            view.relevance_state = relevance_state
            view.is_relevant = relevance_state == "relevant"
            view.feedback_timestamp = datetime.utcnow()
            view.feedback_reason = note
            
            # Record in feedback history
            self.record_feedback_history(
                db=db,
                paper_view_id=view.id,
                user_id=user_id,
                paper_id=paper_id,
                action_type="relevance_change",
                old_value=old_value,
                new_value=relevance_state,
                note=note,
            )
            
            db.commit()
            db.refresh(view)
            return paper_view_to_dict(view)
    
    def bulk_toggle_read(self, user_id: UUID, paper_ids: List[UUID]) -> List[Dict[str, Any]]:
        """Toggle read status for multiple papers."""
        results = []
        for paper_id in paper_ids:
            try:
                result = self.toggle_read(user_id, paper_id)
                results.append(result)
            except ValueError:
                pass  # Skip papers that don't exist
        return results
    
    def bulk_toggle_star(self, user_id: UUID, paper_ids: List[UUID]) -> List[Dict[str, Any]]:
        """Toggle starred status for multiple papers."""
        results = []
        for paper_id in paper_ids:
            try:
                result = self.toggle_star(user_id, paper_id)
                results.append(result)
            except ValueError:
                pass  # Skip papers that don't exist
        return results
    
    def bulk_set_relevance(self, user_id: UUID, paper_ids: List[UUID], relevance_state: str, note: Optional[str] = None) -> List[Dict[str, Any]]:
        """Set relevance state for multiple papers."""
        results = []
        for paper_id in paper_ids:
            try:
                result = self.set_relevance(user_id, paper_id, relevance_state, note)
                results.append(result)
            except ValueError:
                pass  # Skip papers that don't exist
        return results
    
    def get_feedback_history(self, user_id: UUID, paper_id: Optional[UUID] = None, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get feedback history for a user, optionally filtered by paper."""
        with get_db_session() as db:
            q = db.query(PaperFeedbackHistory).filter(PaperFeedbackHistory.user_id == user_id)
            
            if paper_id:
                q = q.filter(PaperFeedbackHistory.paper_id == paper_id)
            
            q = q.order_by(PaperFeedbackHistory.created_at.desc())
            q = q.offset(offset).limit(limit)
            
            history = q.all()
            return [feedback_history_to_dict(h) for h in history]
    
    def get_user_feedback_signals(self, user_id: UUID) -> Dict[str, Any]:
        """
        Get aggregated user feedback signals for agent use.
        
        Returns data that helps the agent understand user preferences:
        - Papers marked not relevant (with authors, categories, keywords)
        - Papers marked relevant 
        - Starred papers (high interest)
        - Read vs unread patterns
        - Feedback history summary
        """
        with get_db_session() as db:
            # Get papers marked as not relevant
            not_relevant_views = db.query(PaperView).options(
                joinedload(PaperView.paper)
            ).filter(
                and_(
                    PaperView.user_id == user_id,
                    PaperView.relevance_state == "not_relevant"
                )
            ).all()
            
            # Get papers marked as relevant
            relevant_views = db.query(PaperView).options(
                joinedload(PaperView.paper)
            ).filter(
                and_(
                    PaperView.user_id == user_id,
                    PaperView.relevance_state == "relevant"
                )
            ).all()
            
            # Get starred papers
            starred_views = db.query(PaperView).options(
                joinedload(PaperView.paper)
            ).filter(
                and_(
                    PaperView.user_id == user_id,
                    PaperView.is_starred == True
                )
            ).all()
            
            # Get overall read/unread stats
            read_count = db.query(PaperView).filter(
                and_(PaperView.user_id == user_id, PaperView.is_read == True)
            ).count()
            
            unread_count = db.query(PaperView).filter(
                and_(PaperView.user_id == user_id, PaperView.is_read == False)
            ).count()
            
            # Extract patterns from not relevant papers
            not_relevant_authors = []
            not_relevant_categories = []
            not_relevant_paper_ids = []
            
            for view in not_relevant_views:
                if view.paper:
                    not_relevant_paper_ids.append(str(view.paper.external_id))
                    if view.paper.authors:
                        not_relevant_authors.extend(view.paper.authors)
                    if view.paper.categories:
                        not_relevant_categories.extend(view.paper.categories)
            
            # Extract patterns from starred papers (positive signals)
            starred_authors = []
            starred_categories = []
            
            for view in starred_views:
                if view.paper:
                    if view.paper.authors:
                        starred_authors.extend(view.paper.authors)
                    if view.paper.categories:
                        starred_categories.extend(view.paper.categories)
            
            # Get recent feedback history (last 100 entries)
            recent_feedback = db.query(PaperFeedbackHistory).filter(
                PaperFeedbackHistory.user_id == user_id
            ).order_by(
                PaperFeedbackHistory.created_at.desc()
            ).limit(100).all()
            
            # Count feedback by type
            feedback_counts = {}
            for fb in recent_feedback:
                key = fb.action_type
                feedback_counts[key] = feedback_counts.get(key, 0) + 1
            
            return {
                "not_relevant": {
                    "count": len(not_relevant_views),
                    "paper_ids": not_relevant_paper_ids,
                    "authors": list(set(not_relevant_authors)),
                    "categories": list(set(not_relevant_categories)),
                },
                "relevant": {
                    "count": len(relevant_views),
                },
                "starred": {
                    "count": len(starred_views),
                    "authors": list(set(starred_authors)),
                    "categories": list(set(starred_categories)),
                },
                "read_stats": {
                    "read": read_count,
                    "unread": unread_count,
                },
                "feedback_summary": feedback_counts,
            }
    
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
                research_interests=data.get("research_interests"),
                keywords=data.get("keywords", []),
                categories=data.get("categories", []),
                topics=data.get("topics", []),
                sharing_preference=data.get("sharing_preference", "weekly"),
                enabled=data.get("enabled", True),
                added_by=data.get("added_by", "manual"),
                auto_send_emails=data.get("auto_send_emails", True),
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
    
    def log_colleague_email(
        self,
        colleague_id: UUID,
        user_id: UUID,
        subject: str,
        email_type: str = "paper_recommendation",
        snippet: Optional[str] = None,
        paper_id: Optional[UUID] = None,
        paper_arxiv_id: Optional[str] = None,
        message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log an outbound email sent to a colleague.
        
        This enables tracking email activity in the colleague details view.
        """
        from .orm_models import ColleagueEmailLog
        
        with get_db_session() as db:
            log_entry = ColleagueEmailLog(
                colleague_id=colleague_id,
                user_id=user_id,
                subject=subject,
                email_type=email_type,
                snippet=snippet[:500] if snippet else None,  # Limit snippet length
                paper_id=paper_id,
                paper_arxiv_id=paper_arxiv_id,
                message_id=message_id,
                sent_at=datetime.utcnow(),
                extra_data=metadata or {}  # Store extra metadata
            )
            db.add(log_entry)
            db.commit()
            db.refresh(log_entry)
            
            return {
                "id": str(log_entry.id),
                "colleague_id": str(log_entry.colleague_id),
                "user_id": str(log_entry.user_id),
                "subject": log_entry.subject,
                "email_type": log_entry.email_type,
                "sent_at": log_entry.sent_at.isoformat() if log_entry.sent_at else None,
            }

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
                for key in ["name", "affiliation", "research_interests", "keywords", "categories", "topics", 
                           "sharing_preference", "enabled", "added_by", "auto_send_emails", "notes"]:
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
                    research_interests=data.get("research_interests"),
                    keywords=data.get("keywords", []),
                    categories=data.get("categories", []),
                    topics=data.get("topics", []),
                    sharing_preference=data.get("sharing_preference", "weekly"),
                    enabled=data.get("enabled", True),
                    added_by=data.get("added_by", "manual"),
                    auto_send_emails=data.get("auto_send_emails", True),
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
        triggered_by: str = 'agent',
        paper_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create an email record.
        
        Args:
            user_id: User ID
            paper_id: Single paper ID (for single-paper emails)
            recipient_email: Email recipient
            subject: Email subject
            body_text: Email body text
            body_preview: Short preview of email body
            triggered_by: Who triggered this email - 'agent' or 'user'
            paper_ids: List of paper IDs (for bulk emails)
        """
        with get_db_session() as db:
            email = Email(
                user_id=user_id,
                paper_id=paper_id,
                recipient_email=recipient_email,
                subject=subject,
                body_text=body_text,
                body_preview=body_preview or body_text[:500] if body_text else None,
                status="queued",
                triggered_by=triggered_by,
                paper_ids=paper_ids,
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
    
    def get_email(self, email_id: UUID) -> Optional[Dict[str, Any]]:
        """Get a single email by ID."""
        with get_db_session() as db:
            email = db.query(Email).filter(Email.id == email_id).first()
            return email_to_dict(email) if email else None
    
    def delete_email(self, email_id: UUID) -> bool:
        """Delete an email by ID."""
        with get_db_session() as db:
            email = db.query(Email).filter(Email.id == email_id).first()
            if email:
                db.delete(email)
                db.commit()
                return True
            return False
    
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
        description: Optional[str] = None,
        reminder_minutes: int = 15,
        triggered_by: str = 'agent',
        paper_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a calendar event record.
        
        Args:
            user_id: User ID
            paper_id: Single paper ID (for single-paper reminders)
            title: Event title
            start_time: Event start time
            duration_minutes: Event duration in minutes
            ics_text: ICS calendar text
            description: Event description with paper details
            reminder_minutes: Minutes before event to send reminder
            triggered_by: Who triggered this event - 'agent' or 'user'
            paper_ids: List of paper IDs (for bulk reminders)
        """
        with get_db_session() as db:
            event = CalendarEvent(
                user_id=user_id,
                paper_id=paper_id,
                title=title,
                start_time=start_time,
                duration_minutes=duration_minutes,
                ics_text=ics_text,
                description=description,
                reminder_minutes=reminder_minutes,
                status="created",
                triggered_by=triggered_by,
                paper_ids=paper_ids,
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
        description: Optional[str] = None,
        reminder_minutes: int = 15,
        triggered_by: str = 'agent',
        paper_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Alias for create_calendar_event for backward compatibility."""
        return self.create_calendar_event(
            user_id=user_id,
            paper_id=paper_id,
            title=title,
            start_time=start_time,
            duration_minutes=duration_minutes,
            ics_text=ics_text,
            description=description,
            reminder_minutes=reminder_minutes,
            triggered_by=triggered_by,
            paper_ids=paper_ids,
        )
    
    def get_calendar_event(self, event_id: UUID) -> Optional[Dict[str, Any]]:
        """Get a calendar event by ID."""
        with get_db_session() as db:
            event = db.query(CalendarEvent).filter(CalendarEvent.id == event_id).first()
            return calendar_event_to_dict(event) if event else None
    
    def update_calendar_event(self, event_id: UUID, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a calendar event."""
        with get_db_session() as db:
            event = db.query(CalendarEvent).filter(CalendarEvent.id == event_id).first()
            if not event:
                raise ValueError(f"Calendar event not found: {event_id}")
            
            for key, value in data.items():
                if hasattr(event, key) and key not in ("id", "user_id", "created_at"):
                    setattr(event, key, value)
            
            event.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(event)
            return calendar_event_to_dict(event)
    
    def reschedule_calendar_event(
        self,
        event_id: UUID,
        new_start_time: datetime,
        reschedule_note: str,
        new_duration_minutes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Reschedule a calendar event to a new time.
        
        Updates the event's start_time, increments sequence_number,
        and adds a reschedule note.
        """
        with get_db_session() as db:
            event = db.query(CalendarEvent).filter(CalendarEvent.id == event_id).first()
            if not event:
                raise ValueError(f"Calendar event not found: {event_id}")
            
            # Update event
            event.start_time = new_start_time
            if new_duration_minutes:
                event.duration_minutes = new_duration_minutes
            event.sequence_number = (event.sequence_number or 0) + 1
            event.reschedule_note = reschedule_note
            event.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(event)
            return calendar_event_to_dict(event)
    
    def mark_calendar_event_superseded(
        self,
        original_event_id: UUID,
        new_event_id: UUID,
    ) -> Dict[str, Any]:
        """
        Mark an original calendar event as superseded by a new event.
        
        Used when creating a new event for rescheduling instead of
        updating the existing one.
        """
        with get_db_session() as db:
            original = db.query(CalendarEvent).filter(CalendarEvent.id == original_event_id).first()
            if not original:
                raise ValueError(f"Original calendar event not found: {original_event_id}")
            
            original.status = "superseded"
            original.updated_at = datetime.utcnow()
            
            # Set the new event's original_event_id
            new_event = db.query(CalendarEvent).filter(CalendarEvent.id == new_event_id).first()
            if new_event:
                new_event.original_event_id = original_event_id
            
            db.commit()
            db.refresh(original)
            return calendar_event_to_dict(original)
    
    def delete_calendar_event(self, event_id: UUID) -> bool:
        """Delete a calendar event by ID."""
        with get_db_session() as db:
            event = db.query(CalendarEvent).filter(CalendarEvent.id == event_id).first()
            if event:
                db.delete(event)
                db.commit()
                return True
            return False
    
    # =========================================================================
    # Calendar Invite Email Operations
    # =========================================================================
    
    def create_calendar_invite_email(
        self,
        calendar_event_id: UUID,
        user_id: UUID,
        message_id: str,
        recipient_email: str,
        subject: str,
        ics_uid: str,
        email_id: Optional[UUID] = None,
        thread_id: Optional[str] = None,
        reminder_token: Optional[str] = None,
        ics_sequence: int = 0,
        ics_method: str = "REQUEST",
        triggered_by: str = "agent",
    ) -> Dict[str, Any]:
        """
        Create a calendar invite email record.
        
        Tracks sent calendar invitations for reply processing.
        
        Args:
            calendar_event_id: Associated calendar event
            user_id: Owner user ID  
            message_id: RFC 2822 Message-ID for email threading
            recipient_email: Email address of recipient
            subject: Email subject
            ics_uid: ICS calendar UID
            email_id: Optional link to Email record
            thread_id: Optional Gmail thread ID
            reminder_token: Unique token embedded in email body for reply matching
            ics_sequence: Sequence number for calendar updates
            ics_method: ICS method (REQUEST, CANCEL, etc.)
            triggered_by: 'user' or 'agent'
        """
        with get_db_session() as db:
            invite_email = CalendarInviteEmail(
                calendar_event_id=calendar_event_id,
                email_id=email_id,
                user_id=user_id,
                message_id=message_id,
                thread_id=thread_id,
                reminder_token=reminder_token,
                recipient_email=recipient_email,
                subject=subject,
                ics_uid=ics_uid,
                ics_sequence=ics_sequence,
                ics_method=ics_method,
                triggered_by=triggered_by,
                status="sent",
                is_latest=True,
                sent_at=datetime.utcnow(),
            )
            db.add(invite_email)
            
            # Mark any previous invites for this event as not latest
            db.query(CalendarInviteEmail).filter(
                and_(
                    CalendarInviteEmail.calendar_event_id == calendar_event_id,
                    CalendarInviteEmail.id != invite_email.id,
                )
            ).update({"is_latest": False})
            
            db.commit()
            db.refresh(invite_email)
            return calendar_invite_email_to_dict(invite_email)
    
    def get_calendar_invite_by_message_id(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get calendar invite email by message_id for reply matching."""
        with get_db_session() as db:
            invite = db.query(CalendarInviteEmail).filter(
                CalendarInviteEmail.message_id == message_id
            ).first()
            return calendar_invite_email_to_dict(invite) if invite else None
    
    def get_calendar_invite_by_ics_uid(self, ics_uid: str) -> Optional[Dict[str, Any]]:
        """Get calendar invite email by ICS UID."""
        with get_db_session() as db:
            invite = db.query(CalendarInviteEmail).filter(
                CalendarInviteEmail.ics_uid == ics_uid
            ).first()
            return calendar_invite_email_to_dict(invite) if invite else None
    
    def list_calendar_invite_emails(
        self,
        user_id: UUID,
        calendar_event_id: Optional[UUID] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List calendar invite emails for a user."""
        with get_db_session() as db:
            q = db.query(CalendarInviteEmail).filter(
                CalendarInviteEmail.user_id == user_id
            )
            
            if calendar_event_id:
                q = q.filter(CalendarInviteEmail.calendar_event_id == calendar_event_id)
            
            q = q.order_by(CalendarInviteEmail.created_at.desc())
            q = q.offset(offset).limit(limit)
            
            return [calendar_invite_email_to_dict(e) for e in q.all()]
    
    def get_calendar_invite_by_token(self, reminder_token: str) -> Optional[Dict[str, Any]]:
        """Get calendar invite email by embedded reminder token.
        
        This is the primary matching method for reschedule replies.
        Token format: [RP_REMINDER_ID: <uuid>]
        """
        with get_db_session() as db:
            invite = db.query(CalendarInviteEmail).filter(
                CalendarInviteEmail.reminder_token == reminder_token
            ).first()
            return calendar_invite_email_to_dict(invite) if invite else None
    
    def get_calendar_invite_by_thread_id(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get calendar invite email by Gmail thread_id.
        
        Fallback method for matching reschedule replies when token not found.
        """
        with get_db_session() as db:
            invite = db.query(CalendarInviteEmail).filter(
                CalendarInviteEmail.thread_id == thread_id
            ).first()
            return calendar_invite_email_to_dict(invite) if invite else None
    
    def get_recent_calendar_invites(
        self,
        user_id: UUID,
        days: int = 7,
        subject_contains: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent calendar invites for fuzzy matching fallback.
        
        Args:
            user_id: Owner user ID
            days: Number of days to look back
            subject_contains: Optional subject filter (case-insensitive)
            
        Returns:
            List of recent invite records, newest first
        """
        with get_db_session() as db:
            since = datetime.utcnow() - timedelta(days=days)
            q = db.query(CalendarInviteEmail).filter(
                and_(
                    CalendarInviteEmail.user_id == user_id,
                    CalendarInviteEmail.created_at >= since,
                    CalendarInviteEmail.is_latest == True,
                )
            )
            
            if subject_contains:
                q = q.filter(
                    CalendarInviteEmail.subject.ilike(f"%{subject_contains}%")
                )
            
            q = q.order_by(CalendarInviteEmail.created_at.desc())
            return [calendar_invite_email_to_dict(e) for e in q.all()]

    # =========================================================================
    # Inbound Email Reply Operations
    # =========================================================================
    
    def create_inbound_email_reply(
        self,
        user_id: UUID,
        original_invite_id: UUID,
        message_id: str,
        from_email: str,
        subject: Optional[str] = None,
        body_text: Optional[str] = None,
        body_html: Optional[str] = None,
        in_reply_to: Optional[str] = None,
        references: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create an inbound email reply record.
        
        Stores user replies to calendar invitation emails.
        """
        with get_db_session() as db:
            reply = InboundEmailReply(
                user_id=user_id,
                original_invite_id=original_invite_id,
                message_id=message_id,
                from_email=from_email,
                subject=subject,
                body_text=body_text,
                body_html=body_html,
                in_reply_to=in_reply_to,
                references=references,
                processed=False,
                received_at=datetime.utcnow(),
            )
            db.add(reply)
            db.commit()
            db.refresh(reply)
            return inbound_email_reply_to_dict(reply)
    
    def get_inbound_reply(self, reply_id: UUID) -> Optional[Dict[str, Any]]:
        """Get inbound email reply by ID."""
        with get_db_session() as db:
            reply = db.query(InboundEmailReply).filter(
                InboundEmailReply.id == reply_id
            ).first()
            return inbound_email_reply_to_dict(reply) if reply else None
    
    def get_inbound_reply_by_message_id(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get inbound email reply by message ID."""
        with get_db_session() as db:
            reply = db.query(InboundEmailReply).filter(
                InboundEmailReply.message_id == message_id
            ).first()
            return inbound_email_reply_to_dict(reply) if reply else None
    
    def list_unprocessed_replies(
        self,
        user_id: Optional[UUID] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List unprocessed inbound email replies."""
        with get_db_session() as db:
            q = db.query(InboundEmailReply).filter(
                InboundEmailReply.processed == False
            )
            
            if user_id:
                q = q.filter(InboundEmailReply.user_id == user_id)
            
            q = q.order_by(InboundEmailReply.received_at.asc())
            q = q.limit(limit)
            
            return [inbound_email_reply_to_dict(r) for r in q.all()]
    
    def update_inbound_reply_processing(
        self,
        reply_id: UUID,
        intent: str,
        extracted_datetime: Optional[datetime] = None,
        extracted_datetime_text: Optional[str] = None,
        confidence_score: Optional[float] = None,
        action_taken: Optional[str] = None,
        new_event_id: Optional[UUID] = None,
        processing_result: Optional[Dict[str, Any]] = None,
        processing_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update inbound reply with processing results.
        
        Called after the agent processes the reply.
        """
        with get_db_session() as db:
            reply = db.query(InboundEmailReply).filter(
                InboundEmailReply.id == reply_id
            ).first()
            
            if not reply:
                raise ValueError(f"Inbound reply not found: {reply_id}")
            
            reply.intent = intent
            reply.extracted_datetime = extracted_datetime
            reply.extracted_datetime_text = extracted_datetime_text
            reply.confidence_score = confidence_score
            reply.processed = True
            reply.processed_at = datetime.utcnow()
            reply.action_taken = action_taken
            reply.new_event_id = new_event_id
            reply.processing_result = processing_result
            reply.processing_error = processing_error
            
            db.commit()
            db.refresh(reply)
            return inbound_email_reply_to_dict(reply)
    
    def get_reply_history_for_event(
        self,
        calendar_event_id: UUID,
    ) -> List[Dict[str, Any]]:
        """
        Get all replies related to a calendar event.
        
        Includes replies to any invite email for the event.
        """
        with get_db_session() as db:
            # Get all invite emails for the event
            invite_ids = db.query(CalendarInviteEmail.id).filter(
                CalendarInviteEmail.calendar_event_id == calendar_event_id
            ).all()
            invite_id_list = [i[0] for i in invite_ids]
            
            if not invite_id_list:
                return []
            
            # Get all replies to those invites
            replies = db.query(InboundEmailReply).filter(
                InboundEmailReply.original_invite_id.in_(invite_id_list)
            ).order_by(InboundEmailReply.received_at.asc()).all()
            
            return [inbound_email_reply_to_dict(r) for r in replies]
    
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
    
    def delete_share(self, share_id: UUID) -> bool:
        """Delete a share record."""
        with get_db_session() as db:
            share = db.query(Share).filter(Share.id == share_id).first()
            if not share:
                return False
            db.delete(share)
            db.commit()
            return True
    
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
    
    # =========================================================================
    # User Settings Operations
    # =========================================================================
    
    def get_user_settings(self, user_id: UUID) -> Optional[Dict[str, Any]]:
        """Get user settings by user ID."""
        with get_db_session() as db:
            settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
            return settings.to_dict() if settings else None
    
    def get_or_create_user_settings(self, user_id: UUID) -> Dict[str, Any]:
        """Get or create user settings with defaults for new fields."""
        with get_db_session() as db:
            settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
            if not settings:
                settings = UserSettings(
                    user_id=user_id,
                    inbox_check_enabled=False,
                    inbox_check_frequency_seconds=None,
                    retrieval_max_results=7,
                    execution_mode="manual",
                )
                db.add(settings)
                db.commit()
                db.refresh(settings)
            else:
                # Ensure defaults for new fields if null (backward compat)
                dirty = False
                if settings.retrieval_max_results is None:
                    settings.retrieval_max_results = 7
                    dirty = True
                if settings.execution_mode is None:
                    settings.execution_mode = "manual"
                    dirty = True
                if dirty:
                    db.commit()
                    db.refresh(settings)
            return settings.to_dict()
    
    def update_inbox_settings(
        self,
        user_id: UUID,
        enabled: bool,
        frequency_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update inbox polling settings."""
        with get_db_session() as db:
            settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
            if not settings:
                settings = UserSettings(user_id=user_id)
                db.add(settings)
            
            settings.inbox_check_enabled = enabled
            settings.inbox_check_frequency_seconds = frequency_seconds
            settings.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(settings)
            return settings.to_dict()
    
    def update_last_inbox_check(self, user_id: UUID) -> None:
        """Update the last inbox check timestamp."""
        with get_db_session() as db:
            settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
            if settings:
                settings.last_inbox_check_at = datetime.utcnow()
                db.commit()
    
    def set_colleague_join_code(self, user_id: UUID, code_hash: str) -> Dict[str, Any]:
        """Set or update the colleague join code hash."""
        with get_db_session() as db:
            settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
            if not settings:
                settings = UserSettings(user_id=user_id)
                db.add(settings)
            
            settings.colleague_join_code_hash = code_hash
            settings.colleague_join_code_updated_at = datetime.utcnow()
            settings.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(settings)
            return settings.to_dict()
    
    def get_colleague_join_code_hash(self, user_id: UUID) -> Optional[str]:
        """Get the colleague join code hash (for validation)."""
        with get_db_session() as db:
            settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
            return settings.colleague_join_code_hash if settings else None
    
    def clear_colleague_join_code(self, user_id: UUID) -> Dict[str, Any]:
        """Clear the colleague join code."""
        with get_db_session() as db:
            settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
            if settings:
                settings.colleague_join_code_hash = None
                settings.colleague_join_code_encrypted = None
                settings.colleague_join_code_updated_at = datetime.utcnow()
                settings.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(settings)
                return settings.to_dict()
            return self.get_or_create_user_settings(user_id)
    
    def set_colleague_join_code_encrypted(self, user_id: UUID, encrypted_code: str, code_hash: Optional[str] = None) -> Dict[str, Any]:
        """Set or update the colleague join code (encrypted for display-back).
        
        Args:
            user_id: The user ID
            encrypted_code: The AES-encrypted join code for display-back
            code_hash: Optional bcrypt hash for verification (if not provided, existing hash is kept)
        """
        with get_db_session() as db:
            settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
            if not settings:
                settings = UserSettings(user_id=user_id)
                db.add(settings)
            settings.colleague_join_code_encrypted = encrypted_code
            # Only update hash if explicitly provided; do NOT clear it
            if code_hash is not None:
                settings.colleague_join_code_hash = code_hash
            settings.colleague_join_code_updated_at = datetime.utcnow()
            settings.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(settings)
            return settings.to_dict()
    
    def get_colleague_join_code_encrypted(self, user_id: UUID) -> Optional[str]:
        """Get the encrypted colleague join code."""
        with get_db_session() as db:
            settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
            return settings.colleague_join_code_encrypted if settings else None
    
    # =========================================================================
    # Execution & Retrieval Settings Operations
    # =========================================================================
    
    def get_retrieval_max_results(self, user_id: UUID) -> int:
        """Get retrieval_max_results from DB, defaulting to 7."""
        with get_db_session() as db:
            settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
            if settings and settings.retrieval_max_results is not None:
                return settings.retrieval_max_results
            return 7
    
    def update_execution_settings(
        self,
        user_id: UUID,
        retrieval_max_results: Optional[int] = None,
        execution_mode: Optional[str] = None,
        scheduled_frequency: Optional[str] = None,
        scheduled_every_x_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update execution/retrieval settings."""
        with get_db_session() as db:
            settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
            if not settings:
                settings = UserSettings(user_id=user_id)
                db.add(settings)
            
            if retrieval_max_results is not None:
                settings.retrieval_max_results = max(1, min(50, retrieval_max_results))
            if execution_mode is not None:
                if execution_mode in ("manual", "scheduled"):
                    settings.execution_mode = execution_mode
            if scheduled_frequency is not None:
                if scheduled_frequency in ("daily", "every_x_days", "weekly", "monthly", ""):
                    settings.scheduled_frequency = scheduled_frequency if scheduled_frequency else None
            if scheduled_every_x_days is not None:
                settings.scheduled_every_x_days = max(1, min(30, scheduled_every_x_days))
            
            # Compute next_run_at when switching to scheduled mode
            if settings.execution_mode == "scheduled" and settings.scheduled_frequency:
                from datetime import timedelta
                now = datetime.utcnow()
                freq = settings.scheduled_frequency
                if freq == "daily":
                    settings.next_run_at = now + timedelta(days=1)
                elif freq == "every_x_days":
                    days = settings.scheduled_every_x_days or 1
                    settings.next_run_at = now + timedelta(days=days)
                elif freq == "weekly":
                    settings.next_run_at = now + timedelta(weeks=1)
                elif freq == "monthly":
                    settings.next_run_at = now + timedelta(days=30)
            elif settings.execution_mode == "manual":
                settings.next_run_at = None
            
            settings.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(settings)
            return settings.to_dict()
    
    def record_run_completed(self, user_id: UUID) -> None:
        """Record that a run just completed and compute next_run_at."""
        with get_db_session() as db:
            settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
            if not settings:
                return
            from datetime import timedelta
            now = datetime.utcnow()
            settings.last_run_at = now
            
            if settings.execution_mode == "scheduled" and settings.scheduled_frequency:
                freq = settings.scheduled_frequency
                if freq == "daily":
                    settings.next_run_at = now + timedelta(days=1)
                elif freq == "every_x_days":
                    days = settings.scheduled_every_x_days or 1
                    settings.next_run_at = now + timedelta(days=days)
                elif freq == "weekly":
                    settings.next_run_at = now + timedelta(weeks=1)
                elif freq == "monthly":
                    settings.next_run_at = now + timedelta(days=30)
            else:
                settings.next_run_at = None
            
            settings.updated_at = now
            db.commit()
    
    # =========================================================================
    # Processed Inbound Email Operations (Idempotency)
    # =========================================================================
    
    def is_email_processed(self, user_id: UUID, gmail_message_id: str) -> bool:
        """Check if an email has already been processed."""
        with get_db_session() as db:
            result = db.query(ProcessedInboundEmail).filter(
                and_(
                    ProcessedInboundEmail.user_id == user_id,
                    ProcessedInboundEmail.gmail_message_id == gmail_message_id
                )
            ).first()
            return result is not None
    
    def get_processed_email_info(self, user_id: UUID, gmail_message_id: str) -> Optional[Dict[str, Any]]:
        """Get processing info for a specific email, or None if not processed."""
        with get_db_session() as db:
            result = db.query(ProcessedInboundEmail).filter(
                and_(
                    ProcessedInboundEmail.user_id == user_id,
                    ProcessedInboundEmail.gmail_message_id == gmail_message_id
                )
            ).first()
            return result.to_dict() if result else None
    
    def mark_email_processed(
        self,
        user_id: UUID,
        gmail_message_id: str,
        email_type: str,
        processing_result: str,
        from_email: Optional[str] = None,
        subject: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mark an email as processed for idempotency."""
        with get_db_session() as db:
            # Check if already exists
            existing = db.query(ProcessedInboundEmail).filter(
                and_(
                    ProcessedInboundEmail.user_id == user_id,
                    ProcessedInboundEmail.gmail_message_id == gmail_message_id
                )
            ).first()
            
            if existing:
                return existing.to_dict()
            
            processed = ProcessedInboundEmail(
                user_id=user_id,
                gmail_message_id=gmail_message_id,
                email_type=email_type,
                processing_result=processing_result,
                from_email=from_email,
                subject=subject,
                error_message=error_message,
            )
            db.add(processed)
            db.commit()
            db.refresh(processed)
            return processed.to_dict()
    
    def list_processed_emails(
        self,
        user_id: UUID,
        limit: int = 100,
        email_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List processed emails for a user."""
        with get_db_session() as db:
            q = db.query(ProcessedInboundEmail).filter(ProcessedInboundEmail.user_id == user_id)
            
            if email_type:
                q = q.filter(ProcessedInboundEmail.email_type == email_type)
            
            q = q.order_by(ProcessedInboundEmail.processed_at.desc()).limit(limit)
            return [p.to_dict() for p in q.all()]

    def has_sent_instructions_to(
        self,
        user_id: UUID,
        from_email: str,
    ) -> bool:
        """Check if we have already sent signup instructions to this sender.

        Returns True if any processed email from this sender has a
        processing_result that indicates an instruction reply was sent.
        This prevents sending repeated instruction emails to the same person.
        """
        instruction_results = (
            "rejected_no_code_replied",
            "rejected_invalid_code_replied",
            "rejected_not_configured_replied",
            "instruction_sent",
        )
        with get_db_session() as db:
            result = db.query(ProcessedInboundEmail).filter(
                and_(
                    ProcessedInboundEmail.user_id == user_id,
                    ProcessedInboundEmail.from_email == from_email,
                    ProcessedInboundEmail.processing_result.in_(instruction_results),
                )
            ).first()
            return result is not None

