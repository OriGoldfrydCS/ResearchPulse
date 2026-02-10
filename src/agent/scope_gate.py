"""
Scope Gate - Classifies user messages to enforce arXiv-only paper discovery scope.

ResearchPulse is a focused paper-finding assistant for arXiv.
This module ensures the chat stays on-topic by classifying every user
message before it reaches the heavy agent pipeline.

Classification:
    IN_SCOPE                  → proceed to normal agent flow
    OUT_OF_SCOPE_ARXIV_ONLY   → research-adjacent but not arXiv-accessible
    OUT_OF_SCOPE_GENERAL      → completely unrelated to research paper discovery
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Scope classification enum
# =============================================================================

class ScopeClass(str, Enum):
    """Result of classifying a user message."""
    IN_SCOPE = "IN_SCOPE"
    OUT_OF_SCOPE_ARXIV_ONLY = "OUT_OF_SCOPE_ARXIV_ONLY"
    OUT_OF_SCOPE_GENERAL = "OUT_OF_SCOPE_GENERAL"


# =============================================================================
# Classification result
# =============================================================================

class ScopeResult:
    """Container for a scope-gate classification decision."""

    __slots__ = ("scope", "reason", "suggested_rewrite", "response")

    def __init__(
        self,
        scope: ScopeClass,
        reason: str,
        suggested_rewrite: Optional[str] = None,
        response: Optional[str] = None,
    ):
        self.scope = scope
        self.reason = reason
        self.suggested_rewrite = suggested_rewrite
        self.response = response

    def __repr__(self) -> str:
        return f"ScopeResult(scope={self.scope.value}, reason={self.reason!r})"


# =============================================================================
# Response templates (constants – consistent tone, never hallucinate sources)
# =============================================================================

RESPONSE_OUT_OF_SCOPE_GENERAL = (
    "ResearchPulse is focused on finding, filtering, and summarizing relevant "
    "scientific papers from arXiv. It is an autonomous agent that perceives its "
    "environment and reasons about paper relevance, novelty, and delivery actions — "
    "including email summaries, calendar reminders, colleague sharing, and reading-list "
    "management. I can't help with general requests in this version. "
)

RESPONSE_OUT_OF_SCOPE_ARXIV_ONLY = (
    "Right now I can only search and summarize papers on arXiv. "
    "If you want, I can look for arXiv preprints related to your topic. "
    "Tell me the keywords (and optional category like cs.CL, cs.LG, stat.ML)."
)

RESPONSE_MISSING_TOPIC = (
    "I can help you find arXiv papers, but I need a topic. "
    "What keywords should I use (and any category or time window)?"
)

RESPONSE_NEED_ARXIV_LINK = (
    "I can summarize arXiv papers. Please provide an arXiv link or ID "
    "(e.g. arXiv:2301.00001 or https://arxiv.org/abs/2301.00001)."
)

RESPONSE_NON_ARXIV_VENUE = (
    "I can't search that venue directly, but I can look for related "
    "preprints on arXiv. What keywords should I use?"
)


# =============================================================================
# Keyword / pattern banks (lowercase, compiled once)
# =============================================================================

# ---- In-scope signals -------------------------------------------------------
_ARXIV_PATTERNS = [
    r"arxiv",
    r"arxiv\.org",
    r"\d{4}\.\d{4,5}",                       # arXiv ID like 2301.12345
    r"paper[s]?",
    r"preprint[s]?",
    r"find\s.*paper",
    r"search\s.*paper",
    r"recent\s.*paper",
    r"latest\s.*paper",
    r"new\s.*paper",
    r"top\s.*paper",
    r"show\s.*paper",
    r"get\s.*paper",
    r"fetch\s.*paper",
    r"look\s.*paper",
    r"recommend\s.*paper",
    r"suggest\s.*paper",
    r"summarize\s.*paper",
    r"summarise\s.*paper",
    r"abstract",
    r"research\s.*interest",
    r"research\s.*topic",
    r"what.s new",
    r"track\s.*author",
    r"track\s.*keyword",
    r"alert\s+me",
    r"notify\s+me",
    r"reading\s*list",
    r"diffusion\s*model",
    r"neural\s*network",
    r"deep\s*learning",
    r"machine\s*learning",
    r"reinforcement\s*learning",
    r"computer\s*vision",
    r"natural\s*language",
    r"cs\.[A-Z]{2}",                           # arXiv category codes
    r"stat\.\w+",
    r"math\.\w+",
    r"eess\.\w+",
    r"q-bio\.\w+",
    r"q-fin\.\w+",
    r"quant-ph",
    r"hep-\w+",
    r"cond-mat",
    r"astro-ph",
]

# ResearchPulse operational features that are always in-scope
_OPERATIONAL_KEYWORDS = [
    "colleague", "colleagues", "share", "sharing",
    "email", "inbox", "reminder", "calendar",
    "delivery", "policy", "setting", "settings",
    "profile", "schedule", "run", "execute",
    "report", "artifact", "reading list",
    "category", "categories", "subscribe",
    "health", "status",
]

# ---- Out-of-scope: non-arXiv venues -----------------------------------------
_NON_ARXIV_VENUES = [
    "google scholar", "pubmed", "ieee", "acm",
    "scopus", "web of science", "nature", "science",
    "springer", "elsevier", "wiley", "jstor",
    "semantic scholar", "crossref", "dblp",
    "biorxiv", "medrxiv", "ssrn", "repec",
]

# ---- Out-of-scope: general requests -----------------------------------------
_GENERAL_OFF_TOPIC = [
    r"write\s+(my|me|a)\s+(code|email|essay|homework|report|letter|resume|cv)",
    r"help\s+me\s+(write|code|program|debug|fix)",
    r"explain\s+(to\s+me\s+)?(what|how|why)\s+(?!.*paper)",
    r"tell\s+me\s+a\s+(joke|story|fact)",
    r"what\s+is\s+the\s+(weather|time|date|news)",
    r"who\s+(is|are|was|were)\b",
    r"recipe[s]?",
    r"play\s+(a\s+)?(game|song|music)",
    r"translate\s",
    r"personal\s+advice",
    r"relationship",
    r"movie[s]?",
    r"travel",
    r"restaurant[s]?",
    r"shopping",
    r"buy\s",
    r"price\s",
    r"stock\s+(market|price)",
    r"crypto",
    r"bitcoin",
    r"sports?(\s|$)",
    r"score[s]?\s",
    r"celebrity",
    r"gossip",
    r"horoscope",
]

# Compile once for performance
_ARXIV_RE = re.compile("|".join(_ARXIV_PATTERNS), re.IGNORECASE)
_GENERAL_RE = re.compile("|".join(_GENERAL_OFF_TOPIC), re.IGNORECASE)


# =============================================================================
# "Explain X" detector — research redirect
# =============================================================================

_EXPLAIN_RE = re.compile(
    r"^(explain|describe|what\s+is|what\s+are|define|tell\s+me\s+about|how\s+does)\b",
    re.IGNORECASE,
)

_SUMMARIZE_RE = re.compile(
    r"^(summarize|summarise|summary\s+of|give\s+me\s+a\s+summary)\b",
    re.IGNORECASE,
)

_ARXIV_ID_RE = re.compile(
    r"(arxiv[:\s]?\d{4}\.\d{4,5}|https?://arxiv\.org/\w+/\d{4}\.\d{4,5})",
    re.IGNORECASE,
)


# =============================================================================
# Core classification function
# =============================================================================

def classify_user_request(
    message: str,
    conversation_context: Optional[str] = None,
) -> ScopeResult:
    """
    Classify a user message for scope-gating.

    Args:
        message: The raw user message text.
        conversation_context: Optional prior conversation summary (unused
            in v1 but reserved for future multi-turn awareness).

    Returns:
        A ``ScopeResult`` with scope, reason, optional rewrite, and response.
    """
    text = message.strip()
    text_lower = text.lower()

    # ------------------------------------------------------------------
    # 0. Empty / trivially short messages → ask for topic
    # ------------------------------------------------------------------
    if len(text) < 3:
        return ScopeResult(
            scope=ScopeClass.IN_SCOPE,
            reason="message_too_short",
            response=RESPONSE_MISSING_TOPIC,
        )

    # ------------------------------------------------------------------
    # 1. Check for clearly general / off-topic requests FIRST
    # ------------------------------------------------------------------
    if _GENERAL_RE.search(text_lower):
        # But if the message ALSO mentions papers / arXiv, keep in-scope
        if _ARXIV_RE.search(text_lower):
            pass  # fall through to in-scope checks
        else:
            logger.info("[SCOPE_GATE] scope=OUT_OF_SCOPE_GENERAL, reason=general_off_topic")
            return ScopeResult(
                scope=ScopeClass.OUT_OF_SCOPE_GENERAL,
                reason="general_off_topic",
                response=RESPONSE_OUT_OF_SCOPE_GENERAL,
            )

    # ------------------------------------------------------------------
    # 2. Non-arXiv venue mentioned?
    # ------------------------------------------------------------------
    for venue in _NON_ARXIV_VENUES:
        if venue in text_lower:
            # If they also mention arXiv, keep in scope
            if "arxiv" in text_lower:
                break
            logger.info(
                "[SCOPE_GATE] scope=OUT_OF_SCOPE_ARXIV_ONLY, "
                f"reason=non_arxiv_venue:{venue}"
            )
            return ScopeResult(
                scope=ScopeClass.OUT_OF_SCOPE_ARXIV_ONLY,
                reason=f"non_arxiv_venue:{venue}",
                suggested_rewrite=(
                    f"I can't search {venue}, but I can look for arXiv "
                    f"preprints related to your topic. What keywords should I use?"
                ),
                response=RESPONSE_NON_ARXIV_VENUE,
            )

    # ------------------------------------------------------------------
    # 3. "Explain X" without asking for papers → redirect
    #    (checked BEFORE broad arXiv keyword match so "What is a CNN?"
    #     hits the explain branch, not the neural-network keyword)
    # ------------------------------------------------------------------
    if _EXPLAIN_RE.search(text_lower):
        # If they explicitly mention paper(s) / arXiv, keep in scope
        if not re.search(r"paper|arxiv|preprint", text_lower):
            topic = _EXPLAIN_RE.sub("", text).strip().rstrip("?.!")
            logger.info(
                "[SCOPE_GATE] scope=OUT_OF_SCOPE_ARXIV_ONLY, reason=explain_without_papers"
            )
            return ScopeResult(
                scope=ScopeClass.OUT_OF_SCOPE_ARXIV_ONLY,
                reason="explain_without_papers",
                suggested_rewrite=(
                    f"Find arXiv papers that explain {topic}" if topic else None
                ),
                response=(
                    "I can find arXiv papers that explain it. "
                    "What subtopic and level are you interested in?"
                ),
            )

    # ------------------------------------------------------------------
    # 4. "Summarize" without an arXiv ID/link → ask for link
    # ------------------------------------------------------------------
    if _SUMMARIZE_RE.search(text_lower):
        if not _ARXIV_ID_RE.search(text):
            # Check if general paper mention (still in scope, but prompt for ID)
            if _ARXIV_RE.search(text_lower):
                pass  # has paper-related keywords, let it proceed
            else:
                logger.info(
                    "[SCOPE_GATE] scope=IN_SCOPE, reason=summarize_missing_arxiv_id"
                )
                return ScopeResult(
                    scope=ScopeClass.IN_SCOPE,
                    reason="summarize_missing_arxiv_id",
                    response=RESPONSE_NEED_ARXIV_LINK,
                )

    # ------------------------------------------------------------------
    # 5. Vague "find papers" / "latest papers" without a real topic → ask
    #    (checked BEFORE broad arXiv match so bare "find papers" triggers
    #     the missing-topic follow-up instead of silently proceeding)
    # ------------------------------------------------------------------
    _vague_paper = re.search(
        r"^(find|search|get|show|fetch|look\s+for|latest|recent|new)\s+(papers?|articles?|preprints?|research)\s*$",
        text_lower,
    )
    if _vague_paper:
        logger.info("[SCOPE_GATE] scope=IN_SCOPE, reason=missing_topic")
        return ScopeResult(
            scope=ScopeClass.IN_SCOPE,
            reason="missing_topic",
            response=RESPONSE_MISSING_TOPIC,
        )

    # ------------------------------------------------------------------
    # 6. Positive match: arXiv / paper-related keywords
    # ------------------------------------------------------------------
    if _ARXIV_RE.search(text_lower):
        logger.info("[SCOPE_GATE] scope=IN_SCOPE, reason=arxiv_keyword_match")
        return ScopeResult(
            scope=ScopeClass.IN_SCOPE,
            reason="arxiv_keyword_match",
        )

    # ------------------------------------------------------------------
    # 7. Operational keywords (colleague, email, settings, etc.)
    # ------------------------------------------------------------------
    for kw in _OPERATIONAL_KEYWORDS:
        if kw in text_lower:
            logger.info(
                f"[SCOPE_GATE] scope=IN_SCOPE, reason=operational_keyword:{kw}"
            )
            return ScopeResult(
                scope=ScopeClass.IN_SCOPE,
                reason=f"operational_keyword:{kw}",
            )

    # ------------------------------------------------------------------
    # 8. Fallback heuristic: if the message contains research-y words
    #    that didn't match earlier patterns, give benefit of the doubt
    #    but only if it's short enough to be a query
    # ------------------------------------------------------------------
    _RESEARCH_HINTS = {
        "model", "models", "algorithm", "algorithms", "benchmark",
        "dataset", "training", "inference", "evaluation", "attention",
        "transformer", "bert", "gpt", "llm", "cnn", "rnn", "gan",
        "vae", "rl", "optimization", "gradient", "architecture",
        "network", "embedding", "fine-tuning", "finetune", "pretrain",
        "pretraining", "survey", "review",
    }
    words = set(text_lower.split())
    if words & _RESEARCH_HINTS:
        logger.info("[SCOPE_GATE] scope=IN_SCOPE, reason=research_hint_fallback")
        return ScopeResult(
            scope=ScopeClass.IN_SCOPE,
            reason="research_hint_fallback",
        )

    # ------------------------------------------------------------------
    # 9. Nothing matched → treat as general out-of-scope
    # ------------------------------------------------------------------
    logger.info("[SCOPE_GATE] scope=OUT_OF_SCOPE_GENERAL, reason=no_matching_signal")
    return ScopeResult(
        scope=ScopeClass.OUT_OF_SCOPE_GENERAL,
        reason="no_matching_signal",
        response=RESPONSE_OUT_OF_SCOPE_GENERAL,
    )
