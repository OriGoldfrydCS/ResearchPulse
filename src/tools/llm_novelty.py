"""
Tool: llm_novelty - LLM-powered novelty scoring for papers.

This module provides LLM-based novelty assessment for papers that pass
the relevance threshold. It complements the existing heuristic novelty
scoring with deeper semantic analysis.

**How it works:**
1. Paper passes relevance threshold (configurable, default 0.4)
2. Top-N similar papers from Pinecone are retrieved
3. LLM compares the paper against similar papers
4. Returns structured novelty score (0-100) with sub-scores and reasoning

**Cost controls:**
- Only invoked for papers above relevance threshold
- Results are cached (configurable TTL)
- Uses efficient model by default (gpt-4o-mini)
- Fallback to heuristic scoring on any error

**Sub-scores:**
- methodology_novelty: How novel is the methodology?
- application_novelty: How novel is the application domain?
- theoretical_novelty: How novel are the theoretical contributions?
- dataset_novelty: How novel are the datasets/benchmarks?

Feature flag: LLM_NOVELTY_ENABLED (defaults to False)
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

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class NoveltySubScores(BaseModel):
    """Sub-component scores for novelty assessment."""
    methodology: float = Field(0.5, ge=0.0, le=1.0, description="Methodology novelty")
    application: float = Field(0.5, ge=0.0, le=1.0, description="Application domain novelty")
    theoretical: float = Field(0.5, ge=0.0, le=1.0, description="Theoretical contribution novelty")
    dataset: float = Field(0.5, ge=0.0, le=1.0, description="Dataset/benchmark novelty")


class LLMNoveltyResult(BaseModel):
    """Result of LLM novelty scoring."""
    arxiv_id: str = Field(..., description="Paper arXiv ID")
    llm_novelty_score: float = Field(..., ge=0.0, le=100.0, description="Overall novelty score 0-100")
    normalized_score: float = Field(..., ge=0.0, le=1.0, description="Normalized score 0-1")
    sub_scores: NoveltySubScores = Field(default_factory=NoveltySubScores)
    reasoning: str = Field(..., description="LLM's reasoning for the score")
    similar_papers_count: int = Field(0, description="Number of similar papers analyzed")
    is_cached: bool = Field(False, description="Whether result was from cache")
    model_used: str = Field("", description="LLM model used")
    tokens_used: int = Field(0, description="Tokens consumed")
    
    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# Prompt Templates
# =============================================================================

NOVELTY_SCORING_SYSTEM_PROMPT = """You are a research paper novelty assessment expert. Your task is to evaluate how novel a paper is compared to existing work in the field.

Novelty assessment criteria:
1. METHODOLOGY (methodology_novelty): Does the paper introduce new methods, algorithms, or techniques?
2. APPLICATION (application_novelty): Does the paper apply known methods to new domains or problems?
3. THEORETICAL (theoretical_novelty): Does the paper provide new theoretical insights or proofs?
4. DATASET (dataset_novelty): Does the paper introduce new datasets, benchmarks, or evaluation methods?

Scoring guidelines:
- 0-20: Incremental work, very similar to existing papers
- 21-40: Minor novelty, small improvements over prior work
- 41-60: Moderate novelty, meaningful contributions in one area
- 61-80: Significant novelty, clear advances in multiple areas
- 81-100: Highly novel, breakthrough contributions

Output format (JSON):
{
    "overall_score": <0-100>,
    "methodology_novelty": <0.0-1.0>,
    "application_novelty": <0.0-1.0>,
    "theoretical_novelty": <0.0-1.0>,
    "dataset_novelty": <0.0-1.0>,
    "reasoning": "<2-3 sentence explanation>"
}"""


NOVELTY_SCORING_USER_PROMPT = """Assess the novelty of this paper compared to the similar papers retrieved from the research corpus.

**TARGET PAPER:**
Title: {title}
Abstract: {abstract}

**SIMILAR EXISTING PAPERS (from corpus):**
{similar_papers}

Based on the comparison, provide your novelty assessment in the required JSON format."""


# =============================================================================
# LLM Novelty Scorer
# =============================================================================

class LLMNoveltyScorer:
    """
    LLM-powered novelty scorer for research papers.
    
    Analyzes papers against similar existing work to produce
    a structured novelty assessment with sub-scores and reasoning.
    """
    
    def __init__(
        self,
        model: str = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        cache_ttl_days: int = 7,
        min_relevance_threshold: float = 0.4,
    ):
        """
        Initialize the LLM novelty scorer.
        
        Args:
            model: OpenAI model to use
            temperature: LLM temperature (lower = more consistent)
            max_tokens: Maximum tokens for response
            cache_ttl_days: How long to cache results
            min_relevance_threshold: Minimum relevance to trigger scoring
        """
        self.model = model or os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_ttl_days = cache_ttl_days
        self.min_relevance_threshold = min_relevance_threshold
        
        # In-memory cache (could be replaced with Redis/DB in production)
        self._cache: Dict[str, Tuple[LLMNoveltyResult, datetime]] = {}
    
    def _get_cache_key(self, arxiv_id: str, similar_ids: List[str]) -> str:
        """Generate cache key from paper and similar papers."""
        content = f"{arxiv_id}:" + ",".join(sorted(similar_ids))
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def _check_cache(self, cache_key: str) -> Optional[LLMNoveltyResult]:
        """Check if result is in cache and not expired."""
        if cache_key not in self._cache:
            return None
        
        result, timestamp = self._cache[cache_key]
        if datetime.utcnow() - timestamp > timedelta(days=self.cache_ttl_days):
            del self._cache[cache_key]
            return None
        
        # Mark as cached
        result.is_cached = True
        return result
    
    def _set_cache(self, cache_key: str, result: LLMNoveltyResult) -> None:
        """Store result in cache."""
        self._cache[cache_key] = (result, datetime.utcnow())
    
    def _format_similar_papers(self, similar_papers: List[Dict]) -> str:
        """Format similar papers for the prompt."""
        if not similar_papers:
            return "No similar papers found in the corpus."
        
        lines = []
        for i, paper in enumerate(similar_papers[:5], 1):
            title = paper.get("title", paper.get("text", "")[:100])
            score = paper.get("score", 0)
            metadata = paper.get("metadata", {})
            
            lines.append(f"{i}. Title: {title}")
            lines.append(f"   Similarity: {score:.2%}")
            if "abstract" in metadata:
                lines.append(f"   Abstract: {metadata['abstract'][:200]}...")
            lines.append("")
        
        return "\n".join(lines)
    
    def _call_llm(
        self,
        paper_title: str,
        paper_abstract: str,
        similar_papers_text: str,
    ) -> Tuple[Dict[str, Any], int]:
        """
        Call the LLM to get novelty assessment.
        
        Returns:
            Tuple of (parsed JSON response, tokens used)
        """
        try:
            from openai import OpenAI
            
            api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
            api_base = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
            if not api_key:
                logger.error("LLM novelty: neither LLM_API_KEY nor OPENAI_API_KEY is set")
                return {}, 0
            client = OpenAI(api_key=api_key, base_url=api_base)
            
            user_prompt = NOVELTY_SCORING_USER_PROMPT.format(
                title=paper_title,
                abstract=paper_abstract[:1500],  # Truncate long abstracts
                similar_papers=similar_papers_text,
            )
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": NOVELTY_SCORING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            
            # Parse response
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            try:
                result = json.loads(content)
                return result, tokens_used
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM response as JSON: {content[:200]}")
                return {}, tokens_used
                
        except ImportError:
            logger.error("OpenAI package not installed")
            return {}, 0
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {}, 0
    
    def _parse_llm_response(
        self,
        arxiv_id: str,
        response: Dict[str, Any],
        tokens_used: int,
        similar_count: int,
    ) -> LLMNoveltyResult:
        """Parse LLM response into structured result."""
        overall_score = float(response.get("overall_score", 50))
        
        sub_scores = NoveltySubScores(
            methodology=float(response.get("methodology_novelty", 0.5)),
            application=float(response.get("application_novelty", 0.5)),
            theoretical=float(response.get("theoretical_novelty", 0.5)),
            dataset=float(response.get("dataset_novelty", 0.5)),
        )
        
        reasoning = response.get("reasoning", "No reasoning provided.")
        
        return LLMNoveltyResult(
            arxiv_id=arxiv_id,
            llm_novelty_score=overall_score,
            normalized_score=overall_score / 100.0,
            sub_scores=sub_scores,
            reasoning=reasoning,
            similar_papers_count=similar_count,
            is_cached=False,
            model_used=self.model,
            tokens_used=tokens_used,
        )
    
    def score(
        self,
        paper: Dict[str, Any],
        similar_papers: List[Dict[str, Any]],
        relevance_score: Optional[float] = None,
    ) -> Optional[LLMNoveltyResult]:
        """
        Score paper novelty using LLM.
        
        Args:
            paper: Paper dict with arxiv_id, title, abstract
            similar_papers: List of similar papers from RAG
            relevance_score: Paper's relevance score (for threshold check)
        
        Returns:
            LLMNoveltyResult or None if skipped/failed
        """
        arxiv_id = paper.get("arxiv_id", "")
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        
        # Check relevance threshold
        if relevance_score is not None and relevance_score < self.min_relevance_threshold:
            logger.debug(f"Skipping LLM novelty for {arxiv_id}: relevance {relevance_score} < {self.min_relevance_threshold}")
            return None
        
        # Check cache
        similar_ids = [p.get("id", p.get("arxiv_id", "")) for p in similar_papers]
        cache_key = self._get_cache_key(arxiv_id, similar_ids)
        
        cached = self._check_cache(cache_key)
        if cached:
            logger.debug(f"Using cached novelty score for {arxiv_id}")
            return cached
        
        # Format similar papers
        similar_text = self._format_similar_papers(similar_papers)
        
        # Call LLM
        response, tokens_used = self._call_llm(title, abstract, similar_text)
        
        if not response:
            logger.warning(f"LLM novelty scoring failed for {arxiv_id}")
            return None
        
        # Parse and cache result
        result = self._parse_llm_response(
            arxiv_id=arxiv_id,
            response=response,
            tokens_used=tokens_used,
            similar_count=len(similar_papers),
        )
        
        self._set_cache(cache_key, result)
        
        logger.info(f"LLM novelty score for {arxiv_id}: {result.llm_novelty_score}/100")
        
        return result


# =============================================================================
# Factory and Integration Functions
# =============================================================================

_scorer_instance: Optional[LLMNoveltyScorer] = None


def get_llm_novelty_scorer() -> LLMNoveltyScorer:
    """Get or create the singleton LLM novelty scorer."""
    global _scorer_instance
    
    if _scorer_instance is None:
        from config.feature_flags import get_feature_config
        config = get_feature_config().llm_novelty
        
        _scorer_instance = LLMNoveltyScorer(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            cache_ttl_days=config.cache_ttl_days,
            min_relevance_threshold=config.min_relevance_threshold,
        )
    
    return _scorer_instance


def score_paper_novelty_with_llm(
    paper: Dict[str, Any],
    similar_papers: List[Dict[str, Any]],
    relevance_score: Optional[float] = None,
    fallback_on_error: bool = True,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Score paper novelty using LLM with fallback to heuristic.
    
    This is the main integration point for the agent workflow.
    
    Args:
        paper: Paper dict with arxiv_id, title, abstract
        similar_papers: Similar papers from RAG (with score and metadata)
        relevance_score: Paper's relevance score
        fallback_on_error: If True, return heuristic score on LLM failure
        user_id: Optional user ID for DB-backed feature flag check
    
    Returns:
        Dict with llm_novelty_score, llm_novelty_reasoning, novelty_sub_scores
        or empty dict if scoring was skipped/failed without fallback
    """
    from config.feature_flags import is_feature_enabled
    
    # Check feature flag with user_id for DB-backed flag check
    if not is_feature_enabled("LLM_NOVELTY", user_id):
        return {}
    
    try:
        scorer = get_llm_novelty_scorer()
        result = scorer.score(paper, similar_papers, relevance_score)
        
        if result:
            return {
                "llm_novelty_score": result.llm_novelty_score,
                "llm_novelty_reasoning": result.reasoning,
                "novelty_sub_scores": result.sub_scores.model_dump(),
                "llm_tokens_used": result.tokens_used,
                "llm_model_used": result.model_used,
                "llm_novelty_cached": result.is_cached,
            }
        
        # Scoring was skipped (below threshold) or failed
        if fallback_on_error:
            # Return heuristic-based fallback
            heuristic_novelty = paper.get("novelty_score", 0.5)
            return {
                "llm_novelty_score": heuristic_novelty * 100,
                "llm_novelty_reasoning": "Fallback to heuristic scoring (LLM skipped or unavailable)",
                "novelty_sub_scores": {},
                "llm_tokens_used": 0,
                "llm_model_used": "heuristic",
                "llm_novelty_cached": False,
            }
        
        return {}
        
    except Exception as e:
        logger.error(f"LLM novelty scoring error: {e}")
        
        if fallback_on_error:
            heuristic_novelty = paper.get("novelty_score", 0.5)
            return {
                "llm_novelty_score": heuristic_novelty * 100,
                "llm_novelty_reasoning": f"Fallback to heuristic (error: {str(e)[:100]})",
                "novelty_sub_scores": {},
                "llm_tokens_used": 0,
                "llm_model_used": "heuristic",
                "llm_novelty_cached": False,
            }
        
        return {}
