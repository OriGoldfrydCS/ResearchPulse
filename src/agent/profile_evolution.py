"""
Module: profile_evolution - Autonomous profile evolution suggestions.

This module analyzes high-relevance and high-novelty papers to generate
advisory suggestions for refining the owner's research profile.

**Key principle: ADVISORY ONLY**
- Suggestions are generated but NEVER auto-applied
- User must explicitly accept or reject each suggestion
- All suggestions are stored for review

**Suggestion types:**
- add_topic: Add a new research topic
- remove_topic: Remove an obsolete topic
- refine_topic: Make a topic more specific
- add_category: Add a new arXiv category
- remove_category: Remove an unused category
- merge_topics: Combine overlapping topics

**Trigger conditions:**
- At least N high-relevance papers in the run (configurable)
- At least 1 paper with high novelty score
- Cooldown period since last analysis (configurable)

Feature flag: PROFILE_EVOLUTION_ENABLED (defaults to False)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Add parent to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class SupportingPaper(BaseModel):
    """Paper that supports a suggestion."""
    arxiv_id: str
    title: str
    relevance_score: float
    novelty_score: float
    llm_novelty_score: Optional[float] = None


class ProfileSuggestion(BaseModel):
    """A single profile evolution suggestion."""
    suggestion_type: str = Field(
        ..., 
        description="Type: add_topic, remove_topic, refine_topic, add_category, remove_category, merge_topics"
    )
    suggestion_text: str = Field(..., description="Human-readable suggestion")
    reasoning: str = Field(..., description="Why this suggestion is being made")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    
    # Structured data for applying the suggestion
    suggestion_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured data: {action, topic, category, related_to, etc.}"
    )
    
    # Supporting evidence
    supporting_papers: List[SupportingPaper] = Field(
        default_factory=list,
        description="Papers that support this suggestion"
    )


class ProfileEvolutionAnalysis(BaseModel):
    """Result of profile evolution analysis."""
    run_id: str
    user_id: str
    analyzed_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    papers_analyzed: int = Field(0)
    high_relevance_count: int = Field(0)
    high_novelty_count: int = Field(0)
    suggestions: List[ProfileSuggestion] = Field(default_factory=list)
    skipped: bool = Field(False, description="True if analysis was skipped")
    skip_reason: Optional[str] = Field(None)


# =============================================================================
# Prompt Templates
# =============================================================================

PROFILE_EVOLUTION_SYSTEM_PROMPT = """You are a research profile optimization expert. Your task is to analyze papers a researcher found highly relevant and novel, then suggest refinements to their research profile.

Current profile structure:
- research_topics: List of research interest areas
- avoid_topics: Topics to exclude from recommendations
- arxiv_categories_include: arXiv categories to search
- arxiv_categories_exclude: arXiv categories to exclude

Suggestion types you can make:
1. add_topic: Suggest adding a new research topic (emerging interest detected)
2. remove_topic: Suggest removing an obsolete topic (no recent matches)
3. refine_topic: Suggest making a topic more specific (too broad)
4. add_category: Suggest adding a new arXiv category
5. merge_topics: Suggest combining similar overlapping topics

Guidelines:
- Only suggest changes with strong evidence from papers
- Be conservative - suggest 1-3 changes maximum
- Consider the user's current profile context
- Provide clear reasoning for each suggestion

Output format (JSON array):
[
    {
        "suggestion_type": "add_topic",
        "topic": "vision-language models",
        "related_to": "language models",
        "confidence": 0.85,
        "reasoning": "5 highly relevant papers in this area detected"
    }
]"""


PROFILE_EVOLUTION_USER_PROMPT = """Analyze these high-relevance and high-novelty papers and suggest profile refinements.

**CURRENT USER PROFILE:**
Research Topics: {research_topics}
Avoid Topics: {avoid_topics}
arXiv Categories (include): {categories_include}
arXiv Categories (exclude): {categories_exclude}

**HIGH-RELEVANCE PAPERS FROM THIS RUN:**
{papers}

Based on these papers, suggest 0-3 profile refinements.
Only suggest changes if there's clear evidence. If the profile looks well-tuned, return an empty array."""


# =============================================================================
# Profile Evolution Analyzer
# =============================================================================

class ProfileEvolutionAnalyzer:
    """
    Analyzes run data to generate profile evolution suggestions.
    
    IMPORTANT: This analyzer ONLY generates suggestions.
    It does NOT automatically modify the user's profile.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        min_high_relevance_papers: int = 3,
        min_novelty_threshold: float = 0.7,
        max_suggestions: int = 3,
        cooldown_hours: int = 24,
    ):
        """
        Initialize the analyzer.
        
        Args:
            model: OpenAI model to use
            min_high_relevance_papers: Minimum papers to trigger analysis
            min_novelty_threshold: Minimum novelty for consideration
            max_suggestions: Maximum suggestions per run
            cooldown_hours: Hours between analyses
        """
        self.model = model
        self.min_high_relevance_papers = min_high_relevance_papers
        self.min_novelty_threshold = min_novelty_threshold
        self.max_suggestions = max_suggestions
        self.cooldown_hours = cooldown_hours
        
        self._last_analysis_time: Optional[datetime] = None
    
    def _check_cooldown(self, user_id: str) -> Optional[str]:
        """Check if we're in cooldown period."""
        try:
            from db.database import is_database_configured, get_db_session
            from db.orm_models import ProfileEvolutionSuggestion
            
            if not is_database_configured():
                # Without DB, use in-memory tracking
                if self._last_analysis_time:
                    elapsed = datetime.utcnow() - self._last_analysis_time
                    if elapsed < timedelta(hours=self.cooldown_hours):
                        return f"Cooldown: {self.cooldown_hours - elapsed.total_seconds() // 3600:.0f}h remaining"
                return None
            
            with get_db_session() as db:
                recent = db.query(ProfileEvolutionSuggestion).filter_by(
                    user_id=uuid.UUID(user_id)
                ).order_by(
                    ProfileEvolutionSuggestion.created_at.desc()
                ).first()
                
                if recent and recent.created_at:
                    elapsed = datetime.utcnow() - recent.created_at
                    if elapsed < timedelta(hours=self.cooldown_hours):
                        remaining = self.cooldown_hours - (elapsed.total_seconds() / 3600)
                        return f"Cooldown: {remaining:.0f}h remaining since last analysis"
            
            return None
            
        except Exception as e:
            logger.warning(f"Cooldown check failed: {e}")
            return None
    
    def _filter_high_relevance_papers(
        self,
        papers: List[Dict],
        relevance_threshold: float = 0.6,
    ) -> List[Dict]:
        """Filter to high-relevance papers."""
        return [
            p for p in papers
            if p.get("relevance_score", 0) >= relevance_threshold
        ]
    
    def _filter_high_novelty_papers(
        self,
        papers: List[Dict],
    ) -> List[Dict]:
        """Filter to high-novelty papers."""
        return [
            p for p in papers
            if (p.get("novelty_score", 0) >= self.min_novelty_threshold or
                p.get("llm_novelty_score", 0) >= self.min_novelty_threshold * 100)
        ]
    
    def _format_papers_for_prompt(self, papers: List[Dict]) -> str:
        """Format papers for the LLM prompt."""
        lines = []
        for i, paper in enumerate(papers[:10], 1):
            lines.append(f"{i}. {paper.get('title', 'Untitled')}")
            lines.append(f"   Relevance: {paper.get('relevance_score', 0):.2f}")
            
            novelty = paper.get("novelty_score", 0)
            llm_novelty = paper.get("llm_novelty_score")
            if llm_novelty:
                lines.append(f"   Novelty: {novelty:.2f} (LLM: {llm_novelty:.0f}/100)")
            else:
                lines.append(f"   Novelty: {novelty:.2f}")
            
            lines.append(f"   Categories: {', '.join(paper.get('categories', []))}")
            
            abstract = paper.get("abstract", "")[:200]
            if abstract:
                lines.append(f"   Abstract: {abstract}...")
            lines.append("")
        
        return "\n".join(lines)
    
    def _heuristic_suggestions(
        self,
        profile: Dict[str, Any],
        high_relevance: List[Dict],
        high_novelty: List[Dict],
    ) -> List[Dict[str, Any]]:
        """
        Generate suggestions from paper metadata when LLM is unavailable.
        
        Detects categories and topics in papers that don't appear in the 
        user's profile and suggests adding them.
        """
        suggestions = []
        user_topics = set(t.lower() for t in profile.get("research_topics", []))
        user_cats_include = set(profile.get("arxiv_categories_include", []))
        
        # Find new categories from papers not in user's included list
        seen_cats = set()
        for paper in high_relevance + high_novelty:
            for cat in paper.get("categories", []):
                if isinstance(cat, str) and cat not in user_cats_include and cat not in seen_cats:
                    seen_cats.add(cat)
                    try:
                        from tools.arxiv_categories import get_category_display_name
                        cat_label = get_category_display_name(cat)
                    except Exception:
                        cat_label = cat
                    suggestions.append({
                        "suggestion_type": "add_category",
                        "category": cat,
                        "reasoning": f"Category {cat_label} appeared in recent high-relevance papers but is not in your tracked categories.",
                        "confidence": 0.6,
                    })
        
        # Extract potential new topics from paper titles
        import re
        topic_words = {}
        for paper in high_relevance:
            title = paper.get("title", "").lower()
            # Split into meaningful phrases (2-3 word combinations)
            words = re.findall(r'\b[a-z]{3,}\b', title)
            for w in words:
                if w not in {"the", "and", "for", "with", "from", "that", "this", "are", "was", "not",
                             "our", "its", "can", "has", "have", "been", "more", "than", "also", "into",
                             "using", "based", "via", "novel", "new", "approach", "method", "paper",
                             "show", "results", "model", "models", "data", "learning", "neural",
                             "network", "networks", "deep", "large", "language", "training"}:
                    topic_words[w] = topic_words.get(w, 0) + 1
        
        # Find repeated meaningful words not already in user topics
        for word, count in sorted(topic_words.items(), key=lambda x: -x[1]):
            if count >= 2 and word not in user_topics and len(suggestions) < self.max_suggestions:
                suggestions.append({
                    "suggestion_type": "add_topic",
                    "topic": word,
                    "reasoning": f"The term '{word}' appeared in {count} recent papers and may represent an emerging interest.",
                    "confidence": 0.5,
                })
        
        return suggestions[:self.max_suggestions]
    
    def _call_llm(
        self,
        profile: Dict[str, Any],
        papers: List[Dict],
    ) -> List[Dict[str, Any]]:
        """Call LLM to generate suggestions."""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            user_prompt = PROFILE_EVOLUTION_USER_PROMPT.format(
                research_topics=", ".join(profile.get("research_topics", [])),
                avoid_topics=", ".join(profile.get("avoid_topics", [])),
                categories_include=", ".join(profile.get("arxiv_categories_include", [])),
                categories_exclude=", ".join(profile.get("arxiv_categories_exclude", [])),
                papers=self._format_papers_for_prompt(papers),
            )
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PROFILE_EVOLUTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            
            content = response.choices[0].message.content
            
            try:
                result = json.loads(content)
                # Handle both {"suggestions": [...]} and [...] formats
                if isinstance(result, dict):
                    return result.get("suggestions", [])
                elif isinstance(result, list):
                    return result
                return []
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM response: {content[:200]}")
                return []
                
        except ImportError:
            logger.error("OpenAI package not installed")
            return []
        except Exception as e:
            logger.error(f"Profile evolution LLM call failed: {e}")
            return []
    
    def _parse_suggestions(
        self,
        llm_suggestions: List[Dict],
        high_relevance_papers: List[Dict],
    ) -> List[ProfileSuggestion]:
        """Parse LLM suggestions into structured format."""
        suggestions = []
        
        for raw in llm_suggestions[:self.max_suggestions]:
            suggestion_type = raw.get("suggestion_type", "")
            
            if suggestion_type not in (
                "add_topic", "remove_topic", "refine_topic",
                "add_category", "remove_category", "merge_topics"
            ):
                continue
            
            # Build structured suggestion data
            suggestion_data = {
                "action": suggestion_type,
            }
            
            if suggestion_type in ("add_topic", "remove_topic", "refine_topic"):
                suggestion_data["topic"] = raw.get("topic", "")
                if "new_topic" in raw:
                    suggestion_data["new_topic"] = raw["new_topic"]
                if "related_to" in raw:
                    suggestion_data["related_to"] = raw["related_to"]
            
            if suggestion_type in ("add_category", "remove_category"):
                suggestion_data["category"] = raw.get("category", "")
            
            if suggestion_type == "merge_topics":
                suggestion_data["topics_to_merge"] = raw.get("topics_to_merge", [])
                suggestion_data["merged_topic"] = raw.get("merged_topic", "")
            
            # Build human-readable text
            if suggestion_type == "add_topic":
                text = f"Add research topic: '{raw.get('topic', '')}'"
            elif suggestion_type == "remove_topic":
                text = f"Remove research topic: '{raw.get('topic', '')}'"
            elif suggestion_type == "refine_topic":
                text = f"Refine topic '{raw.get('topic', '')}' to '{raw.get('new_topic', '')}'"
            elif suggestion_type == "add_category":
                _cat_code = raw.get('category', '')
                try:
                    from tools.arxiv_categories import get_category_display_name
                    _cat_display = get_category_display_name(_cat_code)
                except Exception:
                    _cat_display = _cat_code
                text = f"Add arXiv category: {_cat_display}"
            elif suggestion_type == "remove_category":
                _cat_code = raw.get('category', '')
                try:
                    from tools.arxiv_categories import get_category_display_name
                    _cat_display = get_category_display_name(_cat_code)
                except Exception:
                    _cat_display = _cat_code
                text = f"Remove arXiv category: {_cat_display}"
            elif suggestion_type == "merge_topics":
                text = f"Merge topics: {', '.join(raw.get('topics_to_merge', []))} → '{raw.get('merged_topic', '')}'"
            else:
                text = raw.get("reasoning", "Unknown suggestion")
            
            # Find supporting papers
            supporting = []
            for paper in high_relevance_papers[:5]:
                supporting.append(SupportingPaper(
                    arxiv_id=paper.get("arxiv_id", ""),
                    title=paper.get("title", ""),
                    relevance_score=paper.get("relevance_score", 0),
                    novelty_score=paper.get("novelty_score", 0),
                    llm_novelty_score=paper.get("llm_novelty_score"),
                ))
            
            suggestion = ProfileSuggestion(
                suggestion_type=suggestion_type,
                suggestion_text=text,
                reasoning=raw.get("reasoning", "Based on recent paper patterns"),
                confidence=float(raw.get("confidence", 0.5)),
                suggestion_data=suggestion_data,
                supporting_papers=supporting,
            )
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def analyze(
        self,
        run_id: str,
        user_id: str,
        user_profile: Dict[str, Any],
        scored_papers: List[Dict],
    ) -> ProfileEvolutionAnalysis:
        """
        Analyze run data and generate profile suggestions.
        
        Args:
            run_id: Current run ID
            user_id: User UUID string
            user_profile: User's current research profile
            scored_papers: Papers that were scored in this run
        
        Returns:
            ProfileEvolutionAnalysis with suggestions (may be empty)
        """
        analysis = ProfileEvolutionAnalysis(
            run_id=run_id,
            user_id=user_id,
            papers_analyzed=len(scored_papers),
        )
        
        # Check cooldown
        cooldown_msg = self._check_cooldown(user_id)
        if cooldown_msg:
            analysis.skipped = True
            analysis.skip_reason = cooldown_msg
            logger.info(f"Profile evolution skipped: {cooldown_msg}")
            return analysis
        
        # Filter papers
        high_relevance = self._filter_high_relevance_papers(scored_papers)
        high_novelty = self._filter_high_novelty_papers(scored_papers)
        
        analysis.high_relevance_count = len(high_relevance)
        analysis.high_novelty_count = len(high_novelty)
        
        # Check minimum thresholds
        if len(high_relevance) < self.min_high_relevance_papers:
            analysis.skipped = True
            analysis.skip_reason = f"Insufficient high-relevance papers ({len(high_relevance)} < {self.min_high_relevance_papers})"
            logger.info(f"Profile evolution skipped: {analysis.skip_reason}")
            return analysis
        
        if len(high_novelty) == 0:
            analysis.skipped = True
            analysis.skip_reason = "No high-novelty papers found"
            logger.info(f"Profile evolution skipped: {analysis.skip_reason}")
            return analysis
        
        # Call LLM for suggestions
        llm_suggestions = self._call_llm(user_profile, high_relevance)
        
        if not llm_suggestions:
            # Heuristic fallback: generate suggestions from paper metadata
            llm_suggestions = self._heuristic_suggestions(user_profile, high_relevance, high_novelty)
        
        if not llm_suggestions:
            analysis.skipped = False
            analysis.skip_reason = "No suggestions generated (profile may be well-tuned)"
            return analysis
        
        # Parse suggestions
        analysis.suggestions = self._parse_suggestions(llm_suggestions, high_relevance)
        
        # Update tracking
        self._last_analysis_time = datetime.utcnow()
        
        logger.info(f"Profile evolution generated {len(analysis.suggestions)} suggestions")
        
        return analysis


# =============================================================================
# Database Operations
# =============================================================================

def save_profile_suggestions(
    analysis: ProfileEvolutionAnalysis,
) -> Dict[str, Any]:
    """
    Save profile evolution suggestions to database.
    
    Args:
        analysis: The analysis result with suggestions
    
    Returns:
        Dict with success status and saved suggestion IDs
    """
    if not analysis.suggestions:
        return {"success": True, "saved": 0, "ids": []}
    
    try:
        from db.database import is_database_configured, get_db_session
        from db.orm_models import ProfileEvolutionSuggestion
        
        if not is_database_configured():
            logger.warning("Database not configured, skipping suggestion save")
            return {"success": False, "error": "database_not_configured"}
        
        saved_ids = []
        skipped_dupes = 0
        
        with get_db_session() as db:
            for suggestion in analysis.suggestions:
                # Deduplicate: skip if an identical pending suggestion exists
                existing = (
                    db.query(ProfileEvolutionSuggestion)
                    .filter_by(
                        user_id=uuid.UUID(analysis.user_id),
                        suggestion_type=suggestion.suggestion_type,
                        suggestion_text=suggestion.suggestion_text,
                        status="pending",
                    )
                    .first()
                )
                if existing:
                    skipped_dupes += 1
                    continue

                record = ProfileEvolutionSuggestion(
                    user_id=uuid.UUID(analysis.user_id),
                    run_id=analysis.run_id,
                    suggestion_type=suggestion.suggestion_type,
                    suggestion_text=suggestion.suggestion_text,
                    reasoning=suggestion.reasoning,
                    confidence=suggestion.confidence,
                    supporting_papers=[p.model_dump() for p in suggestion.supporting_papers],
                    suggestion_data=suggestion.suggestion_data,
                    status="pending",
                )
                db.add(record)
                db.flush()
                saved_ids.append(str(record.id))
            
            db.commit()
        
        logger.info(f"Saved {len(saved_ids)} profile evolution suggestions (skipped {skipped_dupes} duplicates)")
        
        return {"success": True, "saved": len(saved_ids), "skipped_duplicates": skipped_dupes, "ids": saved_ids}
        
    except Exception as e:
        logger.error(f"Failed to save profile suggestions: {e}")
        return {"success": False, "error": str(e)}


def get_pending_suggestions(user_id: uuid.UUID) -> List[Dict[str, Any]]:
    """Get all pending suggestions for a user."""
    try:
        from db.database import is_database_configured, get_db_session
        from db.orm_models import ProfileEvolutionSuggestion, profile_evolution_suggestion_to_dict
        
        if not is_database_configured():
            return []
        
        with get_db_session() as db:
            suggestions = db.query(ProfileEvolutionSuggestion).filter_by(
                user_id=user_id,
                status="pending"
            ).order_by(
                ProfileEvolutionSuggestion.created_at.desc()
            ).all()
            
            return [profile_evolution_suggestion_to_dict(s) for s in suggestions]
            
    except Exception as e:
        logger.error(f"Failed to get pending suggestions: {e}")
        return []


def update_suggestion_status(
    suggestion_id: uuid.UUID,
    status: str,
    reviewed_by: Optional[str] = None,
) -> bool:
    """
    Update a suggestion's status.
    
    Args:
        suggestion_id: Suggestion UUID
        status: New status (accepted, rejected, expired)
        reviewed_by: Who reviewed it (for audit)
    
    Returns:
        True if updated successfully
    """
    try:
        from db.database import is_database_configured, get_db_session
        from db.orm_models import ProfileEvolutionSuggestion
        
        if not is_database_configured():
            return False
        
        with get_db_session() as db:
            suggestion = db.query(ProfileEvolutionSuggestion).filter_by(
                id=suggestion_id
            ).first()
            
            if suggestion:
                suggestion.status = status
                suggestion.reviewed_at = datetime.utcnow()
                suggestion.reviewed_by = reviewed_by
                db.commit()
                return True
            
            return False
            
    except Exception as e:
        logger.error(f"Failed to update suggestion status: {e}")
        return False


# =============================================================================
# Integration Helper
# =============================================================================

def analyze_and_suggest_profile_evolution(
    run_id: str,
    user_id: str,
    user_profile: Dict[str, Any],
    scored_papers: List[Dict],
) -> Dict[str, Any]:
    """
    Main integration point for profile evolution analysis.
    
    This function:
    1. Checks if feature is enabled
    2. Creates analyzer with config
    3. Runs analysis
    4. Saves suggestions to DB
    
    Args:
        run_id: Current run ID
        user_id: User UUID string
        user_profile: User's research profile
        scored_papers: Papers scored in this run
    
    Returns:
        Dict with analysis results and save status
    """
    from config.feature_flags import is_feature_enabled, get_feature_config
    
    if not is_feature_enabled("PROFILE_EVOLUTION", user_id):
        return {
            "enabled": False,
            "suggestions_count": 0,
        }
    
    try:
        config = get_feature_config().profile_evolution
        
        analyzer = ProfileEvolutionAnalyzer(
            model=config.model,
            min_high_relevance_papers=config.min_high_relevance_papers,
            min_novelty_threshold=config.min_novelty_for_analysis,
            max_suggestions=config.max_suggestions_per_run,
            cooldown_hours=config.cooldown_hours,
        )
        
        analysis = analyzer.analyze(
            run_id=run_id,
            user_id=user_id,
            user_profile=user_profile,
            scored_papers=scored_papers,
        )
        
        # Save suggestions
        save_result = save_profile_suggestions(analysis)
        
        return {
            "enabled": True,
            "analyzed": not analysis.skipped,
            "skip_reason": analysis.skip_reason,
            "papers_analyzed": analysis.papers_analyzed,
            "high_relevance_count": analysis.high_relevance_count,
            "high_novelty_count": analysis.high_novelty_count,
            "suggestions_count": len(analysis.suggestions),
            "suggestions": [s.model_dump() for s in analysis.suggestions],
            "save_result": save_result,
        }
        
    except Exception as e:
        logger.error(f"Profile evolution analysis failed: {e}")
        return {
            "enabled": True,
            "error": str(e),
            "suggestions_count": 0,
        }
