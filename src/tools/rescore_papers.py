"""
Re-score existing papers after user interests change.

When a user adds/updates Research Interests (via accepting a profile suggestion
or manual edit), this module recomputes relevance for already-saved papers and
updates stored scores, labels, and downstream behaviour (live doc inclusion).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


def rescore_papers_for_user(user_id: UUID) -> Dict[str, Any]:
    """Re-score all saved papers for *user_id* using the current profile.

    Steps
    -----
    1. Load user profile (research_topics, categories, avoid_topics …).
    2. Query all PaperView rows for this user (with joined Paper data).
    3. For each paper, call ``score_relevance_and_importance`` with the
       **updated** profile.
    4. If the new importance differs from the stored one, update the
       PaperView row (relevance_score, importance, explanation).
    5. If any paper's importance changed away from / to ``very_low``,
       mark the live document as stale so it regenerates on next access.

    Returns a summary dict suitable for API responses / logging.
    """
    from ..db.database import is_database_configured, get_db_session
    from ..db.orm_models import User, Paper, PaperView

    if not is_database_configured():
        return {"rescored": 0, "changed": 0, "error": "database not configured"}

    try:
        from ..tools.score_relevance import score_relevance_and_importance
    except ImportError:
        from tools.score_relevance import score_relevance_and_importance

    changed = 0
    rescored = 0
    upgrades: List[str] = []   # paper titles that improved
    downgrades: List[str] = []  # paper titles that worsened
    live_doc_stale = False

    try:
        with get_db_session() as db:
            user = db.query(User).filter_by(id=user_id).first()
            if not user:
                return {"rescored": 0, "changed": 0, "error": "user not found"}

            # Build the research profile dict expected by scoring
            research_profile: Dict[str, Any] = {
                "research_topics": list(user.research_topics or []),
                "avoid_topics": list(user.avoid_topics or []),
                "arxiv_categories_include": list(user.arxiv_categories_include or []),
                "arxiv_categories_exclude": list(user.arxiv_categories_exclude or []),
                "preferred_venues": list(user.preferred_venues or []),
                "my_paper_titles": [p.get("title", "") for p in (user.my_papers or []) if isinstance(p, dict)],
            }

            # Fetch all paper views for this user (batch load papers)
            views = (
                db.query(PaperView)
                .join(Paper, PaperView.paper_id == Paper.id)
                .filter(
                    PaperView.user_id == user_id,
                    PaperView.is_deleted.is_(False),
                )
                .all()
            )

            IMPORTANCE_ORDER = {"very_low": 0, "low": 1, "medium": 2, "high": 3}

            for view in views:
                paper: Paper = view.paper
                if paper is None:
                    continue

                paper_dict: Dict[str, Any] = {
                    "arxiv_id": paper.external_id,
                    "title": paper.title or "",
                    "abstract": paper.abstract or "",
                    "categories": list(paper.categories or []),
                    "authors": list(paper.authors or []),
                    "publication_date": paper.published_at.isoformat() if paper.published_at else None,
                    "link": paper.url,
                }

                result = score_relevance_and_importance(
                    paper=paper_dict,
                    research_profile=research_profile,
                )

                rescored += 1

                old_importance = view.importance or "low"
                new_importance = result.importance
                new_relevance = result.relevance_score

                # Apply the very-low gate (same threshold as react_agent)
                VERY_LOW_RELEVANCE_THRESHOLD = 0.20
                if new_relevance < VERY_LOW_RELEVANCE_THRESHOLD:
                    new_importance = "very_low"

                if (
                    new_importance != old_importance
                    or _score_differs(view.relevance_score, new_relevance)
                ):
                    # Track live-doc staleness
                    if old_importance == "very_low" or new_importance == "very_low":
                        live_doc_stale = True

                    # Update the view
                    view.relevance_score = new_relevance
                    view.novelty_score = result.novelty_score
                    view.importance = new_importance
                    view.last_seen_at = datetime.utcnow()

                    title_short = (paper.title or "")[:80]
                    if IMPORTANCE_ORDER.get(new_importance, 0) > IMPORTANCE_ORDER.get(old_importance, 0):
                        upgrades.append(title_short)
                    else:
                        downgrades.append(title_short)
                    changed += 1

            if changed:
                db.commit()

        # Invalidate live document if papers crossed the very_low boundary
        if live_doc_stale:
            _invalidate_live_document(user_id)

        summary = {
            "rescored": rescored,
            "changed": changed,
            "upgrades": len(upgrades),
            "downgrades": len(downgrades),
        }
        if changed:
            logger.info(
                "Rescored %d papers for user %s: %d changed (%d upgrades, %d downgrades)",
                rescored, user_id, changed, len(upgrades), len(downgrades),
            )
        else:
            logger.debug("Rescored %d papers for user %s — no changes", rescored, user_id)

        return summary

    except Exception as exc:
        logger.error("rescore_papers_for_user failed: %s", exc, exc_info=True)
        return {"rescored": rescored, "changed": changed, "error": str(exc)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_differs(old: Optional[float], new: float, tol: float = 0.005) -> bool:
    """Return True when the relevance score changed meaningfully."""
    if old is None:
        return True
    return abs(old - new) > tol


def _invalidate_live_document(user_id: UUID) -> None:
    """Delete the cached live document so it regenerates on next request."""
    try:
        from ..db.database import get_db_session
        from ..db.orm_models import LiveDocument

        with get_db_session() as db:
            doc = db.query(LiveDocument).filter_by(user_id=user_id).first()
            if doc:
                db.delete(doc)
                db.commit()
                logger.info("Invalidated live document for user %s", user_id)
    except Exception as exc:
        logger.warning("Could not invalidate live document: %s", exc)
