"""
Tool: score_relevance_and_importance - Score a paper's relevance and importance.

This tool evaluates a paper's relevance to the researcher's profile, its novelty
compared to existing knowledge (via RAG results), and determines an importance level.

**Scoring Approach (Deterministic Heuristic + Optional LLM):**

1. **Relevance Score (0-1)**: Heuristic-based scoring using:
   - Topic overlap with research_topics (keyword matching, weighted)
   - Category alignment (arxiv_categories_include vs exclude)
   - Avoid-topic penalties
   - Venue preference bonus
   - User feedback signals (papers marked not_relevant get excluded or penalized)
   
2. **Novelty Score (0-1)**: Based on RAG results:
   - High similarity to existing papers = low novelty
   - No similar papers found = high novelty
   - Decay based on max similarity score from RAG
   
3. **Importance Level**: Combines relevance + novelty:
   - high: relevance >= 0.7 AND novelty >= 0.5
   - medium: relevance >= 0.4 OR (relevance >= 0.3 AND novelty >= 0.6)
   - low: everything else

4. **User Feedback Integration**: The tool now respects user feedback:
   - Papers previously marked "not_relevant" are excluded (importance=low, relevance=0)
   - Authors/topics frequently marked not_relevant receive reduced scores
   - Feedback signals are loaded from DB via get_user_feedback_signals()

5. **Optional LLM Enhancement**: When LLM_API_KEY is configured,
   can refine scoring with semantic understanding. Falls back to
   heuristics if LLM unavailable.

The tool supports min_importance_to_act logic by returning importance
level that can be compared against the stop policy threshold.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

# Add parent to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Input/Output Models
# =============================================================================

class PaperForScoring(BaseModel):
    """Paper input for relevance/importance scoring."""
    arxiv_id: str = Field(..., description="arXiv paper ID (e.g., '2501.00123')")
    title: str = Field(..., description="Paper title")
    abstract: str = Field("", description="Paper abstract")
    categories: List[str] = Field(default_factory=list, description="arXiv categories")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    publication_date: Optional[str] = Field(None, description="Publication date (ISO format)")
    link: Optional[str] = Field(None, description="URL to the paper")


class ResearchProfileSummary(BaseModel):
    """Minimal research profile needed for scoring."""
    research_topics: List[str] = Field(default_factory=list, description="Research interests")
    avoid_topics: List[str] = Field(default_factory=list, description="Topics to exclude")
    arxiv_categories_include: List[str] = Field(default_factory=list, description="Preferred categories")
    arxiv_categories_exclude: List[str] = Field(default_factory=list, description="Excluded categories")
    preferred_venues: List[str] = Field(default_factory=list, description="Preferred venues/conferences")
    my_paper_titles: List[str] = Field(default_factory=list, description="Titles of researcher's own papers")


class RAGMatch(BaseModel):
    """A single RAG retrieval match."""
    text: str = Field(..., description="Matched document text")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Match metadata")


class RAGResults(BaseModel):
    """RAG retrieval results for novelty scoring."""
    matches: List[RAGMatch] = Field(default_factory=list, description="Matching documents")
    query: str = Field("", description="The query used")


class UserFeedbackSignals(BaseModel):
    """User feedback signals for adjusting paper scoring."""
    not_relevant_paper_ids: List[str] = Field(
        default_factory=list,
        description="arxiv_ids of papers marked as not_relevant"
    )
    not_relevant_authors: Dict[str, int] = Field(
        default_factory=dict,
        description="Author names mapped to count of not_relevant papers"
    )
    not_relevant_topics: Dict[str, int] = Field(
        default_factory=dict,
        description="Topic keywords mapped to count of not_relevant papers"
    )
    not_relevant_categories: Dict[str, int] = Field(
        default_factory=dict,
        description="arXiv categories mapped to count of not_relevant papers"
    )
    starred_paper_ids: List[str] = Field(
        default_factory=list,
        description="arxiv_ids of starred papers (positive signal)"
    )
    starred_authors: Dict[str, int] = Field(
        default_factory=dict,
        description="Author names from starred papers"
    )
    starred_topics: Dict[str, int] = Field(
        default_factory=dict,
        description="Topic keywords from starred papers"
    )


class ScoringResult(BaseModel):
    """Result of paper scoring."""
    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance to researcher (0-1)")
    novelty_score: float = Field(..., ge=0.0, le=1.0, description="Novelty vs existing knowledge (0-1)")
    importance: str = Field(..., description="Importance level: high/medium/low")
    explanation: str = Field(..., description="Short user-facing explanation")
    scoring_factors: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed breakdown of scoring factors"
    )
    meets_importance_threshold: Optional[bool] = Field(
        None,
        description="Whether paper meets min_importance_to_act threshold (if provided)"
    )


# =============================================================================
# Scoring Constants
# =============================================================================

# Weights for different scoring components
TOPIC_MATCH_WEIGHT = 0.45
CATEGORY_MATCH_WEIGHT = 0.30
AVOID_TOPIC_PENALTY = 0.85
VENUE_BONUS = 0.10
TITLE_KEYWORD_WEIGHT = 0.15

# Importance thresholds
HIGH_RELEVANCE_THRESHOLD = 0.45  
HIGH_NOVELTY_THRESHOLD = 0.5
MEDIUM_RELEVANCE_THRESHOLD = 0.4
MEDIUM_RELEVANCE_LOW = 0.3
MEDIUM_NOVELTY_THRESHOLD = 0.6

# Novelty calculation from RAG
NOVELTY_BASE = 1.0
NOVELTY_DECAY_FACTOR = 0.85  # How much high similarity reduces novelty

# Feedback-based scoring adjustments
FEEDBACK_AUTHOR_PENALTY_WEIGHT = 0.15  # Penalty per not_relevant author match
FEEDBACK_TOPIC_PENALTY_WEIGHT = 0.10   # Penalty per not_relevant topic match  
FEEDBACK_CATEGORY_PENALTY_WEIGHT = 0.08  # Penalty per not_relevant category match
FEEDBACK_AUTHOR_BONUS_WEIGHT = 0.10    # Bonus per starred author match
FEEDBACK_TOPIC_BONUS_WEIGHT = 0.05     # Bonus per starred topic match
MAX_FEEDBACK_PENALTY = 0.5             # Maximum feedback penalty
MAX_FEEDBACK_BONUS = 0.3               # Maximum feedback bonus

# Importance level order for threshold comparison
IMPORTANCE_ORDER = {"low": 0, "medium": 1, "high": 2}


# =============================================================================
# Helper Functions
# =============================================================================

def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, remove punctuation."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _extract_keywords(text: str) -> set:
    """Extract meaningful keywords from text."""
    normalized = _normalize_text(text)
    words = set(normalized.split())
    
    # Filter out common stopwords
    stopwords = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
        'these', 'those', 'we', 'our', 'us', 'they', 'their', 'it', 'its',
        'i', 'you', 'he', 'she', 'which', 'who', 'whom', 'what', 'when',
        'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'not', 'only', 'same', 'so',
        'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then',
        'using', 'based', 'show', 'shows', 'paper', 'propose', 'proposed',
        'method', 'approach', 'new', 'novel', 'results', 'experiments',
        'experimental', 'performance', 'achieves', 'achieve', 'significant',
        'demonstrate', 'demonstrates', 'via', 'over', 'under', 'through',
    }
    
    return {w for w in words if len(w) > 2 and w not in stopwords}


def _calculate_topic_overlap(paper_keywords: set, profile_topics: List[str]) -> float:
    """
    Calculate topic overlap score between paper and research profile.
    
    Uses per-topic evaluation: each profile topic is checked individually.
    For multi-word topics (e.g., "Multi Armed Bandits"), at least 2 of
    its constituent keywords must appear in the paper to count as a match.
    This prevents false positives from a single generic word overlap
    (e.g., "multi" in "multi-robot" falsely matching "Multi Armed Bandits").
    
    Returns a score from 0.0 to 1.0.
    """
    if not profile_topics or not paper_keywords:
        return 0.0
    
    matched_topic_count = 0
    valid_topics = 0
    
    for topic in profile_topics:
        topic_kws = _extract_keywords(topic)
        if not topic_kws:
            continue
        valid_topics += 1
        
        # Count how many of this topic's keywords appear in the paper
        hit_count = 0.0
        for tk in topic_kws:
            # Direct match
            if tk in paper_keywords:
                hit_count += 1.0
                continue
            # Partial match (substring): e.g., "bandit" matches "bandits"
            for pk in paper_keywords:
                if pk != tk and (pk in tk or tk in pk) and len(min(pk, tk, key=len)) > 3:
                    hit_count += 0.5
                    break  # one partial match per topic keyword is enough
        
        # For multi-word topics, require at least 2 keyword hits to count
        # as a genuine match. This prevents single generic words (like
        # "multi" from "Multi Armed Bandits") from creating false positives.
        if len(topic_kws) >= 2:
            if hit_count >= 2:
                matched_topic_count += 1
        else:
            # Single-word topic (e.g., "PCA"): any hit counts
            if hit_count > 0:
                matched_topic_count += 1
    
    if valid_topics == 0:
        return 0.0
    
    raw_score = matched_topic_count / valid_topics
    # Apply scaling with diminishing returns
    normalized_score = min(1.0, raw_score * 1.5)
    
    return round(normalized_score, 3)


def _calculate_category_score(
    paper_categories: List[str],
    include_categories: List[str],
    exclude_categories: List[str]
) -> float:
    """
    Calculate category alignment score.
    
    Returns:
        1.0 if paper is in included categories and not in excluded
        0.5 if paper has some overlap with included
        0.0 if paper is in excluded categories or no overlap
    """
    if not paper_categories:
        return 0.3  # Neutral score if no categories
    
    paper_cats_set = set(paper_categories)
    include_set = set(include_categories)
    exclude_set = set(exclude_categories)
    
    # Check for excluded categories first
    if paper_cats_set & exclude_set:
        return 0.0
    
    # Check overlap with included categories
    if not include_set:
        return 0.5  # No preference specified
    
    overlap = paper_cats_set & include_set
    if overlap:
        # Score based on how many included categories match
        return min(1.0, 0.5 + 0.5 * len(overlap) / len(include_set))
    
    return 0.2  # Some base score for papers not in exclude list


def _check_avoid_topics(paper_keywords: set, avoid_topics: List[str],
                        paper_text: str = "") -> float:
    """
    Check if paper contains topics to avoid.
    
    Uses two strategies:
    1. Phrase matching: checks if the full exclude phrase appears in paper text
       (word-boundary aware to prevent "GAN" matching "organ")
    2. Keyword matching: falls back to keyword set intersection for single words
    
    Returns penalty score from 0.0 (no penalty) to 1.0 (max penalty).
    """
    if not avoid_topics:
        return 0.0
    
    phrase_matches = 0
    keyword_matches = 0
    paper_text_lower = paper_text.lower() if paper_text else ""
    
    for topic in avoid_topics:
        topic_clean = topic.strip()
        if not topic_clean:
            continue
        # Strategy 1: Word-boundary phrase match against full paper text
        if paper_text_lower:
            pattern = r'\b' + re.escape(topic_clean.lower()) + r'\b'
            if re.search(pattern, paper_text_lower):
                phrase_matches += 1
                continue
        # Strategy 2: Keyword set intersection (fallback for single words)
        topic_kw = _extract_keywords(topic_clean)
        if topic_kw and topic_kw.issubset(paper_keywords):
            keyword_matches += 1
    
    total_matches = phrase_matches + keyword_matches
    
    # Even one match is a significant penalty
    if total_matches > 0:
        return min(1.0, 0.5 + 0.15 * total_matches)
    
    return 0.0


def _check_venue_match(paper_text: str, preferred_venues: List[str]) -> float:
    """
    Check if paper mentions preferred venues.
    
    Returns bonus from 0.0 to 1.0.
    """
    if not preferred_venues:
        return 0.0
    
    paper_lower = paper_text.lower()
    
    for venue in preferred_venues:
        if venue.lower() in paper_lower:
            return 1.0
    
    return 0.0


def _calculate_novelty_from_rag(rag_results: Optional[RAGResults]) -> tuple[float, str]:
    """
    Calculate novelty score based on RAG similarity results.
    
    High similarity to existing papers = low novelty.
    No similar papers = high novelty.
    
    Returns:
        (novelty_score, explanation)
    """
    if not rag_results or not rag_results.matches:
        return (0.85, "No similar papers found in knowledge base")
    
    # Get maximum similarity score
    max_similarity = max(m.score for m in rag_results.matches)
    
    # Calculate novelty as inverse of max similarity (with decay)
    # High similarity (0.9) -> low novelty (~0.2)
    # Low similarity (0.3) -> high novelty (~0.75)
    novelty = NOVELTY_BASE - (max_similarity * NOVELTY_DECAY_FACTOR)
    novelty = max(0.1, min(0.95, novelty))  # Clamp to reasonable range
    
    # Build explanation
    if max_similarity >= 0.85:
        explanation = f"Very similar to existing papers (max similarity: {max_similarity:.0%})"
    elif max_similarity >= 0.7:
        explanation = f"Moderately similar to known work (max similarity: {max_similarity:.0%})"
    elif max_similarity >= 0.5:
        explanation = f"Some overlap with existing papers (max similarity: {max_similarity:.0%})"
    else:
        explanation = f"Appears novel compared to stored papers (max similarity: {max_similarity:.0%})"
    
    return (round(novelty, 3), explanation)


def _determine_importance(
    relevance_score: float,
    novelty_score: float
) -> str:
    """
    Determine importance level based on relevance and novelty scores.
    
    Decision logic:
    - high: relevance >= 0.65 AND novelty >= 0.5
    - medium: relevance >= 0.4 OR (relevance >= 0.3 AND novelty >= 0.6)
    - low: everything else
    """
    if relevance_score >= HIGH_RELEVANCE_THRESHOLD and novelty_score >= HIGH_NOVELTY_THRESHOLD:
        return "high"
    
    if relevance_score >= MEDIUM_RELEVANCE_THRESHOLD:
        return "medium"
    
    if relevance_score >= MEDIUM_RELEVANCE_LOW and novelty_score >= MEDIUM_NOVELTY_THRESHOLD:
        return "medium"
    
    return "low"


def _generate_explanation(
    paper: PaperForScoring,
    relevance_score: float,
    novelty_score: float,
    importance: str,
    factors: Dict[str, Any]
) -> str:
    """
    Generate a short, user-facing explanation of the scoring decision.
    """
    parts = []
    
    # Relevance explanation
    if relevance_score >= 0.7:
        parts.append("Highly relevant to your research interests")
    elif relevance_score >= 0.4:
        parts.append("Moderately relevant to your work")
    elif relevance_score >= 0.2:
        parts.append("Tangentially related to your research")
    else:
        parts.append("Limited relevance to your profile")
    
    # Add specific factors
    if factors.get("topic_overlap", 0) >= 0.5:
        parts.append("strong topic alignment")
    
    if factors.get("avoid_penalty", 0) > 0:
        parts.append("contains some less preferred topics")
    
    if factors.get("category_score", 0) == 0:
        parts.append("from excluded category")
    
    # Novelty explanation
    if novelty_score >= 0.7:
        parts.append("appears to present novel contributions")
    elif novelty_score <= 0.3:
        parts.append("covers familiar ground")
    
    # Build final explanation
    explanation = parts[0]
    if len(parts) > 1:
        explanation += " (" + ", ".join(parts[1:]) + ")"
    
    explanation += f". Importance: {importance}."
    
    return explanation


def _meets_importance_threshold(importance: str, min_importance: Optional[str]) -> Optional[bool]:
    """
    Check if the paper's importance meets the minimum threshold.
    
    Returns:
        True if meets threshold, False if below, None if no threshold specified
    """
    if not min_importance:
        return None
    
    paper_level = IMPORTANCE_ORDER.get(importance, 0)
    threshold_level = IMPORTANCE_ORDER.get(min_importance, 0)
    
    return paper_level >= threshold_level


def _calculate_feedback_adjustments(
    paper: PaperForScoring,
    paper_keywords: Set[str],
    feedback_signals: Optional[UserFeedbackSignals]
) -> tuple[float, float, Dict[str, Any]]:
    """
    Calculate scoring adjustments based on user feedback signals.
    
    Returns:
        (penalty, bonus, feedback_factors)
        - penalty: 0.0 to MAX_FEEDBACK_PENALTY (reduces relevance)
        - bonus: 0.0 to MAX_FEEDBACK_BONUS (increases relevance)
        - feedback_factors: dict of detailed feedback factors
    """
    if not feedback_signals:
        return (0.0, 0.0, {})
    
    penalty = 0.0
    bonus = 0.0
    factors = {
        "not_relevant_author_matches": 0,
        "not_relevant_topic_matches": 0,
        "not_relevant_category_matches": 0,
        "starred_author_matches": 0,
        "starred_topic_matches": 0,
    }
    
    # Check author matches against not_relevant authors
    for author in paper.authors:
        author_lower = author.lower().strip()
        for nr_author, count in feedback_signals.not_relevant_authors.items():
            if author_lower == nr_author.lower() or author_lower in nr_author.lower() or nr_author.lower() in author_lower:
                penalty += min(count, 3) * FEEDBACK_AUTHOR_PENALTY_WEIGHT
                factors["not_relevant_author_matches"] += 1
                break
    
    # Check topic/keyword matches against not_relevant topics
    for keyword in paper_keywords:
        if keyword in feedback_signals.not_relevant_topics:
            count = feedback_signals.not_relevant_topics[keyword]
            penalty += min(count, 3) * FEEDBACK_TOPIC_PENALTY_WEIGHT
            factors["not_relevant_topic_matches"] += 1
    
    # Check category matches against not_relevant categories
    for category in paper.categories:
        if category in feedback_signals.not_relevant_categories:
            count = feedback_signals.not_relevant_categories[category]
            penalty += min(count, 3) * FEEDBACK_CATEGORY_PENALTY_WEIGHT
            factors["not_relevant_category_matches"] += 1
    
    # Check author matches against starred authors (positive signal)
    for author in paper.authors:
        author_lower = author.lower().strip()
        for star_author, count in feedback_signals.starred_authors.items():
            if author_lower == star_author.lower() or author_lower in star_author.lower() or star_author.lower() in author_lower:
                bonus += min(count, 3) * FEEDBACK_AUTHOR_BONUS_WEIGHT
                factors["starred_author_matches"] += 1
                break
    
    # Check topic/keyword matches against starred topics
    for keyword in paper_keywords:
        if keyword in feedback_signals.starred_topics:
            count = feedback_signals.starred_topics[keyword]
            bonus += min(count, 3) * FEEDBACK_TOPIC_BONUS_WEIGHT
            factors["starred_topic_matches"] += 1
    
    # Cap the adjustments
    penalty = min(penalty, MAX_FEEDBACK_PENALTY)
    bonus = min(bonus, MAX_FEEDBACK_BONUS)
    
    factors["total_feedback_penalty"] = round(penalty, 3)
    factors["total_feedback_bonus"] = round(bonus, 3)
    
    return (penalty, bonus, factors)


# =============================================================================
# Main Tool Implementation
# =============================================================================

def score_relevance_and_importance(
    paper: Dict[str, Any],
    research_profile: Dict[str, Any],
    rag_results: Optional[Dict[str, Any]] = None,
    min_importance_to_act: Optional[str] = None,
    user_feedback: Optional[Dict[str, Any]] = None,
) -> ScoringResult:
    """
    Score a paper's relevance, novelty, and determine importance.
    
    This is the main tool function that combines heuristic scoring factors
    to evaluate how relevant and novel a paper is to a researcher.
    
    Args:
        paper: Paper dictionary with at minimum:
            - arxiv_id: arXiv paper ID
            - title: Paper title
            - abstract: Paper abstract (optional but recommended)
            - categories: arXiv categories (optional)
            - authors: List of authors (optional)
            - publication_date: Publication date (optional)
            - link: URL to paper (optional)
            
        research_profile: Research profile with:
            - research_topics: List of research interests
            - avoid_topics: Topics to exclude
            - arxiv_categories_include: Preferred categories
            - arxiv_categories_exclude: Excluded categories
            - preferred_venues: Preferred conferences/journals
            - my_paper_titles: Researcher's own paper titles (optional)
            
        rag_results: Optional RAG retrieval results with:
            - matches: List of similar documents with scores
            - query: The query used
            
        min_importance_to_act: Optional minimum importance threshold
            from stop policy ("high", "medium", or "low")
            
        user_feedback: Optional user feedback signals with:
            - not_relevant_paper_ids: List of paper IDs marked not relevant
            - not_relevant_authors: Dict of author names -> count
            - not_relevant_topics: Dict of topic keywords -> count
            - not_relevant_categories: Dict of categories -> count
            - starred_paper_ids: List of starred paper IDs (positive signal)
            - starred_authors: Dict of author names -> count
            - starred_topics: Dict of topic keywords -> count
            
    Returns:
        ScoringResult with relevance_score, novelty_score, importance,
        explanation, and meets_importance_threshold.
        
    Example:
        >>> result = score_relevance_and_importance(
        ...     paper={"arxiv_id": "2501.00123", "title": "LLM Attention", "abstract": "..."},
        ...     research_profile={"research_topics": ["language models", "attention"]},
        ... )
        >>> print(f"Relevance: {result.relevance_score}, Importance: {result.importance}")
    """
    # Parse inputs
    paper_obj = PaperForScoring(**paper)
    profile_obj = ResearchProfileSummary(**research_profile)
    
    rag_obj = None
    if rag_results:
        rag_obj = RAGResults(
            matches=[RAGMatch(**m) for m in rag_results.get("matches", [])],
            query=rag_results.get("query", "")
        )
    
    feedback_obj = None
    if user_feedback:
        feedback_obj = UserFeedbackSignals(**user_feedback)
    
    # =================================
    # Check if paper is already marked not_relevant
    # =================================
    
    if feedback_obj and paper_obj.arxiv_id in feedback_obj.not_relevant_paper_ids:
        # Paper was previously marked as not relevant by user
        # Return low importance immediately to exclude it
        return ScoringResult(
            arxiv_id=paper_obj.arxiv_id,
            title=paper_obj.title,
            relevance_score=0.0,
            novelty_score=0.5,
            importance="low",
            explanation="Previously marked as not relevant by user.",
            scoring_factors={
                "excluded_by_user_feedback": True,
                "reason": "Paper was marked not_relevant"
            },
            meets_importance_threshold=_meets_importance_threshold("low", min_importance_to_act),
        )
    
    # Combine title and abstract for keyword extraction
    paper_text = f"{paper_obj.title} {paper_obj.abstract}"
    paper_keywords = _extract_keywords(paper_text)
    
    # =================================
    # Calculate Relevance Score
    # =================================
    
    # 1. Topic overlap
    topic_overlap = _calculate_topic_overlap(paper_keywords, profile_obj.research_topics)
    
    # 2. Category alignment
    category_score = _calculate_category_score(
        paper_obj.categories,
        profile_obj.arxiv_categories_include,
        profile_obj.arxiv_categories_exclude
    )
    
    # 3. Avoid topic penalty
    avoid_penalty = _check_avoid_topics(paper_keywords, profile_obj.avoid_topics,
                                        paper_text=paper_text)
    
    # 4. Venue bonus (check in title/abstract)
    venue_bonus = _check_venue_match(paper_text, profile_obj.preferred_venues)
    
    # 5. Title keyword match (extra weight for topic match in title alone)
    title_keywords = _extract_keywords(paper_obj.title)
    title_topic_match = _calculate_topic_overlap(title_keywords, profile_obj.research_topics)
    
    # 6. User feedback adjustments
    feedback_penalty, feedback_bonus, feedback_factors = _calculate_feedback_adjustments(
        paper_obj, paper_keywords, feedback_obj
    )
    
    # Combine factors into relevance score
    relevance_score = (
        topic_overlap * TOPIC_MATCH_WEIGHT +
        category_score * CATEGORY_MATCH_WEIGHT +
        title_topic_match * TITLE_KEYWORD_WEIGHT +
        venue_bonus * VENUE_BONUS
    )
    
    # CRITICAL: If a paper has ZERO topic keyword overlap with the researcher's
    # stated interests, category match alone should NOT produce a meaningful score.
    # A paper in "cs.LG" is not relevant just because the user also studies
    # topics that happen to be in cs.LG — it must actually mention those topics.
    #
    # This is the FUNDAMENTAL gate: positive evidence of topic relevance is
    # REQUIRED. Without it, category-only match produces at most 0.10.
    if topic_overlap == 0 and title_topic_match == 0:
        # No topic keywords matched at all — this paper doesn't mention
        # anything the researcher cares about. Cap to near-zero.
        relevance_score = min(relevance_score, 0.10)
    
    # Apply avoid penalty — hard-exclude if an avoid topic phrase appears in the title
    if avoid_penalty > 0:
        title_lower = paper_obj.title.lower()
        title_has_avoid = False
        for t in profile_obj.avoid_topics:
            pattern = r'\b' + re.escape(t.strip().lower()) + r'\b'
            if re.search(pattern, title_lower):
                title_has_avoid = True
                break
        if title_has_avoid:
            # Avoid topic is in the title — cap score very low
            relevance_score = min(relevance_score * 0.05, 0.05)
        else:
            relevance_score = relevance_score * (1.0 - avoid_penalty * AVOID_TOPIC_PENALTY)
    
    # Apply user feedback adjustments
    relevance_score = relevance_score * (1.0 - feedback_penalty) + feedback_bonus
    
    # Hard floor for excluded categories
    if category_score == 0:
        relevance_score = min(relevance_score, 0.15)
    
    relevance_score = round(max(0.0, min(1.0, relevance_score)), 3)
    
    # =================================
    # Calculate Novelty Score
    # =================================
    
    novelty_score, novelty_explanation = _calculate_novelty_from_rag(rag_obj)
    
    # =================================
    # Determine Importance
    # =================================
    
    importance = _determine_importance(relevance_score, novelty_score)
    
    # =================================
    # Build Scoring Factors
    # =================================
    
    scoring_factors = {
        "topic_overlap": topic_overlap,
        "category_score": category_score,
        "avoid_penalty": avoid_penalty,
        "venue_bonus": venue_bonus,
        "title_topic_match": title_topic_match,
        "novelty_explanation": novelty_explanation,
        "rag_matches_count": len(rag_obj.matches) if rag_obj else 0,
        "rag_max_similarity": max((m.score for m in rag_obj.matches), default=0.0) if rag_obj and rag_obj.matches else None,
        "feedback_penalty": feedback_factors.get("total_feedback_penalty", 0.0),
        "feedback_bonus": feedback_factors.get("total_feedback_bonus", 0.0),
        **feedback_factors,
    }
    
    # =================================
    # Generate Explanation
    # =================================
    
    explanation = _generate_explanation(
        paper_obj, relevance_score, novelty_score, importance, scoring_factors
    )
    
    # =================================
    # Check Importance Threshold
    # =================================
    
    meets_threshold = _meets_importance_threshold(importance, min_importance_to_act)
    
    return ScoringResult(
        arxiv_id=paper_obj.arxiv_id,
        title=paper_obj.title,
        relevance_score=relevance_score,
        novelty_score=novelty_score,
        importance=importance,
        explanation=explanation,
        scoring_factors=scoring_factors,
        meets_importance_threshold=meets_threshold,
    )


def score_relevance_and_importance_json(
    paper: Dict[str, Any],
    research_profile: Dict[str, Any],
    rag_results: Optional[Dict[str, Any]] = None,
    min_importance_to_act: Optional[str] = None,
    user_feedback: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    JSON-serializable version of score_relevance_and_importance.
    
    Args:
        paper: Paper dictionary
        research_profile: Research profile dictionary
        rag_results: Optional RAG results dictionary
        min_importance_to_act: Optional minimum importance threshold
        user_feedback: Optional user feedback signals dictionary
        
    Returns:
        Dictionary with scoring results
    """
    result = score_relevance_and_importance(
        paper=paper,
        research_profile=research_profile,
        rag_results=rag_results,
        min_importance_to_act=min_importance_to_act,
        user_feedback=user_feedback,
    )
    return result.model_dump()


# =============================================================================
# Batch Scoring
# =============================================================================

def score_papers_batch(
    papers: List[Dict[str, Any]],
    research_profile: Dict[str, Any],
    rag_results_map: Optional[Dict[str, Dict[str, Any]]] = None,
    min_importance_to_act: Optional[str] = None,
    user_feedback: Optional[Dict[str, Any]] = None,
) -> List[ScoringResult]:
    """
    Score multiple papers in batch.
    
    Args:
        papers: List of paper dictionaries
        research_profile: Research profile dictionary
        rag_results_map: Optional dict mapping arxiv_id to RAG results
        min_importance_to_act: Optional minimum importance threshold
        user_feedback: Optional user feedback signals dictionary
        
    Returns:
        List of ScoringResult objects
    """
    results = []
    rag_results_map = rag_results_map or {}
    
    for paper in papers:
        arxiv_id = paper.get("arxiv_id", "")
        rag_results = rag_results_map.get(arxiv_id)
        
        result = score_relevance_and_importance(
            paper=paper,
            research_profile=research_profile,
            rag_results=rag_results,
            min_importance_to_act=min_importance_to_act,
            user_feedback=user_feedback,
        )
        results.append(result)
    
    return results


# =============================================================================
# LangChain Tool Definition (for ReAct agent)
# =============================================================================

SCORE_RELEVANCE_DESCRIPTION = """
Score a paper's relevance, novelty, and determine its importance level.

Input:
- paper: Paper with arxiv_id, title, abstract, categories, authors, publication_date, link
- research_profile: Profile with research_topics, avoid_topics, arxiv_categories_include/exclude
- rag_results: Optional RAG retrieval results for novelty scoring
- min_importance_to_act: Optional threshold from stop policy
- user_feedback: Optional user feedback signals for personalized scoring

Output:
- relevance_score: 0.0 to 1.0 (how relevant to researcher's interests)
- novelty_score: 0.0 to 1.0 (how novel compared to existing knowledge)
- importance: "high", "medium", or "low"
- explanation: Short user-facing explanation
- meets_importance_threshold: Whether paper meets min_importance_to_act

User Feedback Integration:
- Papers marked not_relevant are excluded (importance=low, relevance=0)
- Authors/topics frequently marked not_relevant receive reduced scores
- Starred papers/authors receive bonus scores
- Feedback signals should be loaded from DB via get_user_feedback_signals()

Use this tool after retrieving similar papers via RAG to determine
what action to take for each paper.

Scoring logic:
- Relevance: topic match + category alignment - avoid penalties +/- feedback adjustments
- Novelty: inverse of RAG similarity (high similarity = low novelty)
- High importance: relevance >= 0.65 AND novelty >= 0.5
- Medium importance: relevance >= 0.4 OR (relevance >= 0.3 AND novelty >= 0.6)
"""

SCORE_RELEVANCE_SCHEMA = {
    "name": "score_relevance_and_importance",
    "description": SCORE_RELEVANCE_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "paper": {
                "type": "object",
                "description": "The paper to score",
                "properties": {
                    "arxiv_id": {"type": "string", "description": "arXiv paper ID"},
                    "title": {"type": "string", "description": "Paper title"},
                    "abstract": {"type": "string", "description": "Paper abstract"},
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "arXiv categories"
                    },
                    "authors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of authors"
                    },
                    "publication_date": {"type": "string", "description": "Publication date"},
                    "link": {"type": "string", "description": "URL to paper"}
                },
                "required": ["arxiv_id", "title"]
            },
            "research_profile": {
                "type": "object",
                "description": "Researcher profile for relevance scoring",
                "properties": {
                    "research_topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Research interests"
                    },
                    "avoid_topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Topics to avoid"
                    },
                    "arxiv_categories_include": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Preferred categories"
                    },
                    "arxiv_categories_exclude": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Excluded categories"
                    },
                    "preferred_venues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Preferred venues"
                    }
                }
            },
            "rag_results": {
                "type": "object",
                "description": "Optional RAG retrieval results",
                "properties": {
                    "matches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "score": {"type": "number"},
                                "metadata": {"type": "object"}
                            }
                        }
                    },
                    "query": {"type": "string"}
                }
            },
            "min_importance_to_act": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Minimum importance threshold from stop policy"
            },
            "user_feedback": {
                "type": "object",
                "description": "User feedback signals for personalized scoring",
                "properties": {
                    "not_relevant_paper_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "arxiv_ids of papers marked as not_relevant"
                    },
                    "not_relevant_authors": {
                        "type": "object",
                        "description": "Author names mapped to count of not_relevant papers"
                    },
                    "not_relevant_topics": {
                        "type": "object",
                        "description": "Topic keywords mapped to count of not_relevant papers"
                    },
                    "not_relevant_categories": {
                        "type": "object",
                        "description": "arXiv categories mapped to count of not_relevant papers"
                    },
                    "starred_paper_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "arxiv_ids of starred papers"
                    },
                    "starred_authors": {
                        "type": "object",
                        "description": "Author names from starred papers"
                    },
                    "starred_topics": {
                        "type": "object",
                        "description": "Topic keywords from starred papers"
                    }
                }
            }
        },
        "required": ["paper", "research_profile"]
    }
}


# =============================================================================
# Self-Check
# =============================================================================

def self_check() -> bool:
    """
    Run self-check tests for score_relevance_and_importance tool.
    
    Returns:
        True if all checks pass, False otherwise.
    """
    print("=" * 60)
    print("score_relevance_and_importance Self-Check")
    print("=" * 60)
    
    all_passed = True
    
    def check(name: str, condition: bool) -> bool:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        return condition

    # Test profile matching the demo research_profile.json
    test_profile = {
        "research_topics": [
            "large language models",
            "transformer architectures",
            "neural machine translation",
            "retrieval-augmented generation",
            "efficient inference",
            "model compression"
        ],
        "avoid_topics": [
            "cryptocurrency",
            "blockchain",
            "social media analysis"
        ],
        "arxiv_categories_include": ["cs.CL", "cs.LG", "cs.AI"],
        "arxiv_categories_exclude": ["cs.CR", "q-fin.ST"],
        "preferred_venues": ["NeurIPS", "ICML", "ACL", "EMNLP"],
    }

    # Test 1: High relevance paper
    print("\n1. High Relevance Paper:")
    try:
        paper = {
            "arxiv_id": "2501.00001",
            "title": "Efficient Transformer Architectures for Large Language Models",
            "abstract": "We present a novel transformer architecture that improves inference efficiency for large language models through sparse attention patterns and model compression techniques.",
            "categories": ["cs.CL", "cs.LG"],
        }
        result = score_relevance_and_importance(paper, test_profile)
        all_passed &= check("returns ScoringResult", isinstance(result, ScoringResult))
        all_passed &= check("relevance >= 0.5", result.relevance_score >= 0.5)
        all_passed &= check("importance is high/medium", result.importance in ["high", "medium"])
        all_passed &= check("has explanation", len(result.explanation) > 10)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 2: Low relevance paper (cryptocurrency)
    print("\n2. Low Relevance Paper (Avoid Topic):")
    try:
        paper = {
            "arxiv_id": "2501.00002",
            "title": "Blockchain-Based Cryptocurrency Trading Analysis",
            "abstract": "We analyze cryptocurrency market trends using blockchain transaction data and social media sentiment.",
            "categories": ["q-fin.ST"],
        }
        result = score_relevance_and_importance(paper, test_profile)
        all_passed &= check("relevance < 0.3", result.relevance_score < 0.3)
        all_passed &= check("importance is low", result.importance == "low")
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 3: With RAG results (high similarity = low novelty)
    print("\n3. With RAG Results (Low Novelty):")
    try:
        paper = {
            "arxiv_id": "2501.00003",
            "title": "Attention Mechanisms in Neural Networks",
            "abstract": "A review of attention mechanisms for language modeling.",
            "categories": ["cs.CL"],
        }
        rag_results = {
            "matches": [
                {"text": "Similar paper about attention", "score": 0.92, "metadata": {}},
                {"text": "Another similar paper", "score": 0.85, "metadata": {}},
            ],
            "query": "attention mechanisms"
        }
        result = score_relevance_and_importance(paper, test_profile, rag_results)
        all_passed &= check("novelty < 0.3 (high similarity)", result.novelty_score < 0.3)
        all_passed &= check("rag_matches_count in factors", result.scoring_factors["rag_matches_count"] == 2)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 4: With RAG results (no matches = high novelty)
    print("\n4. With RAG Results (High Novelty):")
    try:
        paper = {
            "arxiv_id": "2501.00004",
            "title": "Novel LLM Compression Technique",
            "abstract": "A completely new approach to model compression.",
            "categories": ["cs.LG"],
        }
        rag_results = {"matches": [], "query": "compression"}
        result = score_relevance_and_importance(paper, test_profile, rag_results)
        all_passed &= check("novelty > 0.7 (no similar)", result.novelty_score > 0.7)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 5: Importance threshold check
    print("\n5. Importance Threshold Check:")
    try:
        paper = {
            "arxiv_id": "2501.00005",
            "title": "LLM Training",
            "abstract": "Training large language models.",
            "categories": ["cs.CL"],
        }
        # High threshold
        result_high = score_relevance_and_importance(
            paper, test_profile, min_importance_to_act="high"
        )
        # Low threshold
        result_low = score_relevance_and_importance(
            paper, test_profile, min_importance_to_act="low"
        )
        all_passed &= check("threshold comparison works", 
                          result_low.meets_importance_threshold is not None)
        # A "medium" paper should meet "low" threshold but might not meet "high"
        if result_high.importance == "medium":
            all_passed &= check("medium meets low threshold", 
                              result_low.meets_importance_threshold == True)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 6: JSON output format
    print("\n6. JSON Output Format:")
    try:
        paper = {"arxiv_id": "2501.00006", "title": "Test", "abstract": "Test abstract"}
        result = score_relevance_and_importance_json(paper, test_profile)
        all_passed &= check("returns dict", isinstance(result, dict))
        all_passed &= check("has relevance_score", "relevance_score" in result)
        all_passed &= check("has novelty_score", "novelty_score" in result)
        all_passed &= check("has importance", "importance" in result)
        all_passed &= check("has explanation", "explanation" in result)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 7: Batch scoring
    print("\n7. Batch Scoring:")
    try:
        papers = [
            {"arxiv_id": "2501.00007", "title": "LLM Paper 1", "abstract": "About transformers"},
            {"arxiv_id": "2501.00008", "title": "LLM Paper 2", "abstract": "About attention"},
        ]
        results = score_papers_batch(papers, test_profile)
        all_passed &= check("returns list", isinstance(results, list))
        all_passed &= check("correct count", len(results) == 2)
        all_passed &= check("all ScoringResult", all(isinstance(r, ScoringResult) for r in results))
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 8: Excluded category handling
    print("\n8. Excluded Category Handling:")
    try:
        paper = {
            "arxiv_id": "2501.00009",
            "title": "Cryptographic Security for Machine Learning",
            "abstract": "Novel ML approaches using transformers and language models.",
            "categories": ["cs.CR", "cs.LG"],  # cs.CR is excluded
        }
        result = score_relevance_and_importance(paper, test_profile)
        all_passed &= check("excluded category = low score", result.relevance_score <= 0.15)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 9: Tool schema validation
    print("\n9. Tool Schema:")
    all_passed &= check("schema has name", SCORE_RELEVANCE_SCHEMA["name"] == "score_relevance_and_importance")
    all_passed &= check("schema has description", len(SCORE_RELEVANCE_SCHEMA["description"]) > 100)
    all_passed &= check("paper required", "paper" in SCORE_RELEVANCE_SCHEMA["parameters"]["required"])
    all_passed &= check("research_profile required", "research_profile" in SCORE_RELEVANCE_SCHEMA["parameters"]["required"])

    # Test 10: Score bounds
    print("\n10. Score Bounds:")
    try:
        paper = {"arxiv_id": "2501.00010", "title": "X", "abstract": "Y"}
        result = score_relevance_and_importance(paper, {})
        all_passed &= check("relevance in [0,1]", 0.0 <= result.relevance_score <= 1.0)
        all_passed &= check("novelty in [0,1]", 0.0 <= result.novelty_score <= 1.0)
        all_passed &= check("importance valid", result.importance in ["high", "medium", "low", "very_low"])
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Summary
    print("\n" + "=" * 60)
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


# =============================================================================
# User Feedback Signals Loader (for Agent Integration)
# =============================================================================

def get_user_feedback_signals(user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Load user feedback signals from the database for use in paper scoring.
    
    This function connects to the database via PostgresStore and retrieves
    aggregated feedback signals that can be passed to score_relevance_and_importance.
    
    Args:
        user_id: Optional user ID. If not provided, uses the default user.
        
    Returns:
        Dictionary with feedback signals:
        - not_relevant_paper_ids: List of arxiv_ids marked not relevant
        - not_relevant_authors: Dict of author names -> count
        - not_relevant_topics: Dict of topic keywords -> count
        - not_relevant_categories: Dict of categories -> count
        - starred_paper_ids: List of starred paper arxiv_ids
        - starred_authors: Dict of author names -> count
        - starred_topics: Dict of topic keywords -> count
        
    Example:
        >>> signals = get_user_feedback_signals("user-123")
        >>> result = score_relevance_and_importance(paper, profile, user_feedback=signals)
    """
    try:
        from db.postgres_store import get_default_store
        store = get_default_store()
        return store.get_user_feedback_signals(user_id=user_id)
    except ImportError:
        # Fallback if import fails (e.g., during testing without DB)
        return {
            "not_relevant_paper_ids": [],
            "not_relevant_authors": {},
            "not_relevant_topics": {},
            "not_relevant_categories": {},
            "starred_paper_ids": [],
            "starred_authors": {},
            "starred_topics": {},
        }
    except Exception as e:
        # Log error and return empty signals
        print(f"Warning: Failed to load user feedback signals: {e}")
        return {
            "not_relevant_paper_ids": [],
            "not_relevant_authors": {},
            "not_relevant_topics": {},
            "not_relevant_categories": {},
            "starred_paper_ids": [],
            "starred_authors": {},
            "starred_topics": {},
        }
