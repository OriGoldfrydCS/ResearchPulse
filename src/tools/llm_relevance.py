"""
Tool: llm_relevance - LLM-powered relevance filtering for papers.

This module uses an LLM to semantically evaluate whether a paper is
relevant to the researcher's interests and whether it matches any
excluded topics. It complements the heuristic keyword-based scoring
with genuine semantic understanding.

**Why this exists:**
Heuristic keyword matching can't understand semantic relationships.
For example, it can't know that "AttentionRetriever" is about the
Attention mechanism (an excluded topic) without the word appearing
in a keyword set. An LLM understands that instantly.

**How it works:**
1. Paper title + abstract are sent to the LLM along with the
   researcher's interests and excluded topics
2. LLM returns structured JSON with:
   - is_relevant: bool (does this match the researcher's interests?)
   - is_excluded: bool (does this match an excluded topic?)
   - relevance_score: 0.0-1.0 (how relevant to interests)
   - reasoning: short explanation
3. Papers judged irrelevant or excluded are filtered out before
   heuristic scoring even runs

**Cost controls:**
- Uses gpt-4o-mini by default (~$0.15/1M input tokens)
- Typical paper evaluation: ~400-600 tokens → ~$0.0001/paper
- Results are cached in-memory (configurable TTL)
- Fallback to heuristic-only on any error
- Feature flag: LLM_RELEVANCE_ENABLED (defaults to False)

**Integration:**
Called in the react_agent scoring loop, BEFORE heuristic scoring.
Papers that the LLM judges as excluded or irrelevant are skipped.
Papers that pass get an LLM relevance score that can optionally
augment the heuristic score.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# Add parent to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.feature_flags import is_feature_enabled, get_feature_config

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class LLMRelevanceResult(BaseModel):
    """Result of LLM relevance evaluation."""
    arxiv_id: str = Field(..., description="Paper arXiv ID")
    is_relevant: bool = Field(True, description="Whether the paper is relevant to research interests")
    is_excluded: bool = Field(False, description="Whether the paper matches an excluded topic")
    relevance_score: float = Field(0.5, ge=0.0, le=1.0, description="LLM-assessed relevance 0-1")
    excluded_topic_match: str = Field("", description="Which excluded topic matched, if any")
    reasoning: str = Field("", description="LLM's reasoning for the judgment")
    is_cached: bool = Field(False, description="Whether result was from cache")
    model_used: str = Field("", description="LLM model used")
    tokens_used: int = Field(0, description="Tokens consumed")


# =============================================================================
# Prompt Templates
# =============================================================================

RELEVANCE_SYSTEM_PROMPT = """You are a research paper relevance filter. Your task is to determine whether a paper is relevant to a researcher's interests and whether it should be EXCLUDED based on their excluded topics.

You must be STRICT about exclusions. If a paper is primarily about an excluded topic, it must be excluded even if it has tangential overlap with an interest.

You must also be HONEST about relevance. A paper is only relevant if its core contribution directly relates to one or more of the researcher's interests. Superficial keyword overlap does not count.

Output format (JSON only, no markdown):
{
    "is_relevant": true/false,
    "is_excluded": true/false,
    "relevance_score": 0.0-1.0,
    "excluded_topic_match": "" or "the matched topic",
    "reasoning": "1-2 sentence explanation"
}

Guidelines:
- is_excluded=true takes priority: if a paper matches an excluded topic, it should be filtered out regardless of relevance
- relevance_score should reflect how directly the paper's CORE contribution aligns with the interests (not superficial keyword overlap)
- A score of 0.0-0.2 means no meaningful connection
- A score of 0.3-0.5 means tangential or weak connection
- A score of 0.6-0.8 means moderate direct relevance
- A score of 0.9-1.0 means the paper is squarely in the researcher's area"""


RELEVANCE_USER_PROMPT = """Evaluate this paper for the researcher:

**RESEARCHER'S INTERESTS:**
{interests}

**TOPICS TO EXCLUDE (papers about these should be filtered out):**
{exclude_topics}

**PAPER TO EVALUATE:**
Title: {title}
Abstract: {abstract}

Respond with JSON only."""


# =============================================================================
# LLM Relevance Filter
# =============================================================================

class LLMRelevanceFilter:
    """
    LLM-powered relevance filter for research papers.

    Uses an LLM to semantically judge whether each paper is relevant
    to the researcher's interests and whether it should be excluded.
    """

    def __init__(
        self,
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 300,
        cache_ttl_hours: int = 48,
    ):
        self.model = model or os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_ttl_hours = cache_ttl_hours

        # In-memory cache: key -> (result, timestamp)
        self._cache: Dict[str, Tuple[LLMRelevanceResult, datetime]] = {}

    # -----------------------------------------------------------------
    # Cache helpers
    # -----------------------------------------------------------------

    def _cache_key(
        self,
        arxiv_id: str,
        interests: List[str],
        exclude_topics: List[str],
    ) -> str:
        """Deterministic cache key from paper + profile."""
        content = (
            f"{arxiv_id}:"
            + ",".join(sorted(t.lower() for t in interests))
            + "|"
            + ",".join(sorted(t.lower() for t in exclude_topics))
        )
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _check_cache(self, key: str) -> Optional[LLMRelevanceResult]:
        if key not in self._cache:
            return None
        result, ts = self._cache[key]
        if datetime.utcnow() - ts > timedelta(hours=self.cache_ttl_hours):
            del self._cache[key]
            return None
        result.is_cached = True
        return result

    def _set_cache(self, key: str, result: LLMRelevanceResult) -> None:
        self._cache[key] = (result, datetime.utcnow())

    # -----------------------------------------------------------------
    # LLM call
    # -----------------------------------------------------------------

    def _call_llm(
        self,
        title: str,
        abstract: str,
        interests: List[str],
        exclude_topics: List[str],
    ) -> Tuple[Dict[str, Any], int]:
        """Call OpenAI to evaluate relevance. Returns (parsed_json, tokens)."""
        try:
            from openai import OpenAI

            api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
            api_base = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
            if not api_key:
                logger.error("LLM relevance: neither LLM_API_KEY nor OPENAI_API_KEY is set")
                return {}, 0
            client = OpenAI(api_key=api_key, base_url=api_base)

            user_prompt = RELEVANCE_USER_PROMPT.format(
                interests=", ".join(interests) if interests else "(none specified)",
                exclude_topics=", ".join(exclude_topics) if exclude_topics else "(none specified)",
                title=title,
                abstract=abstract[:1500],
            )

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": RELEVANCE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0

            try:
                return json.loads(content), tokens_used
            except json.JSONDecodeError:
                logger.warning(f"LLM relevance: bad JSON: {content[:200]}")
                return {}, tokens_used

        except ImportError:
            logger.error("openai package not installed")
            return {}, 0
        except Exception as e:
            logger.error(f"LLM relevance call failed: {e}")
            return {}, 0

    # -----------------------------------------------------------------
    # Main evaluate method
    # -----------------------------------------------------------------

    def evaluate(
        self,
        paper: Dict[str, Any],
        research_topics: List[str],
        avoid_topics: List[str],
    ) -> Optional[LLMRelevanceResult]:
        """
        Evaluate a single paper's relevance using the LLM.

        Args:
            paper: dict with arxiv_id, title, abstract
            research_topics: researcher's interest topics
            avoid_topics: topics to exclude

        Returns:
            LLMRelevanceResult or None on failure
        """
        arxiv_id = paper.get("arxiv_id", "unknown")
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")

        # Cache check
        key = self._cache_key(arxiv_id, research_topics, avoid_topics)
        cached = self._check_cache(key)
        if cached:
            return cached

        # Call LLM
        response, tokens = self._call_llm(title, abstract, research_topics, avoid_topics)
        if not response:
            return None

        # Parse
        result = LLMRelevanceResult(
            arxiv_id=arxiv_id,
            is_relevant=bool(response.get("is_relevant", True)),
            is_excluded=bool(response.get("is_excluded", False)),
            relevance_score=float(response.get("relevance_score", 0.5)),
            excluded_topic_match=str(response.get("excluded_topic_match", "")),
            reasoning=str(response.get("reasoning", "")),
            is_cached=False,
            model_used=self.model,
            tokens_used=tokens,
        )

        self._set_cache(key, result)
        logger.info(
            f"LLM relevance for {arxiv_id}: relevant={result.is_relevant}, "
            f"excluded={result.is_excluded}, score={result.relevance_score:.2f}"
        )
        return result


# =============================================================================
# Singleton & convenience
# =============================================================================

_filter_instance: Optional[LLMRelevanceFilter] = None


def get_llm_relevance_filter() -> LLMRelevanceFilter:
    """Get or create the singleton LLM relevance filter."""
    global _filter_instance
    if _filter_instance is None:
        config = get_feature_config()
        llm_rel = getattr(config, "llm_relevance", None)

        _filter_instance = LLMRelevanceFilter(
            model=getattr(llm_rel, "model", "gpt-4o-mini") if llm_rel else "gpt-4o-mini",
            temperature=getattr(llm_rel, "temperature", 0.1) if llm_rel else 0.1,
            max_tokens=getattr(llm_rel, "max_tokens", 300) if llm_rel else 300,
            cache_ttl_hours=getattr(llm_rel, "cache_ttl_hours", 48) if llm_rel else 48,
        )
    return _filter_instance


def evaluate_paper_relevance_with_llm(
    paper: Dict[str, Any],
    research_topics: List[str],
    avoid_topics: List[str],
    user_id: Optional[str] = None,
    fallback_on_error: bool = True,
) -> Dict[str, Any]:
    """
    High-level integration point for the agent workflow.

    Returns a dict with is_relevant, is_excluded, relevance_score, reasoning.
    Returns empty dict if the feature is disabled or if scoring fails
    without fallback.
    """
    if not is_feature_enabled("LLM_RELEVANCE", user_id):
        return {}

    try:
        filt = get_llm_relevance_filter()
        result = filt.evaluate(paper, research_topics, avoid_topics)

        if result:
            return {
                "is_relevant": result.is_relevant,
                "is_excluded": result.is_excluded,
                "relevance_score": result.relevance_score,
                "excluded_topic_match": result.excluded_topic_match,
                "reasoning": result.reasoning,
                "tokens_used": result.tokens_used,
                "model_used": result.model_used,
                "is_cached": result.is_cached,
            }

        # LLM call failed
        if fallback_on_error:
            return {
                "is_relevant": True,
                "is_excluded": False,
                "relevance_score": 0.5,
                "excluded_topic_match": "",
                "reasoning": "Fallback: LLM unavailable, allowing paper through for heuristic scoring",
                "tokens_used": 0,
                "model_used": "fallback",
                "is_cached": False,
            }

        return {}

    except Exception as e:
        logger.error(f"LLM relevance evaluation error: {e}")
        if fallback_on_error:
            return {
                "is_relevant": True,
                "is_excluded": False,
                "relevance_score": 0.5,
                "excluded_topic_match": "",
                "reasoning": f"Fallback: error ({str(e)[:80]})",
                "tokens_used": 0,
                "model_used": "fallback",
                "is_cached": False,
            }
        return {}
