"""
Prompt Controller - System-level controller for standardized research prompts.

This module implements:
1. Prompt Template Library - recognizes and maps user queries to templates
2. Retrieval vs Output Rules - separates internal retrieval from user-facing output
3. Output Enforcement Logic - ensures exact compliance with requested paper counts
4. Database Persistence - ALL prompts are saved to database for audit/compliance

CRITICAL: The number of papers shown to the user MUST EXACTLY match the
requested number. Internal retrieval can fetch up to MAX_RETRIEVAL_RESULTS,
but output is strictly bounded by user's requested K.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default number of papers to fetch from arXiv per run.
# NOTE: This constant is a fallback default only.
# Production reads from DB via get_arxiv_fetch_count() (dashboard "Papers per Run").
DEFAULT_ARXIV_FETCH_COUNT = 7

# Backward-compatible alias (used by tests and legacy references)
MAX_RETRIEVAL_RESULTS = DEFAULT_ARXIV_FETCH_COUNT

# Default number of papers to show the user if they don't specify
DEFAULT_OUTPUT_COUNT = 5

# Extra papers to retrieve from Pinecone beyond what the user requested,
# so the agent has a small buffer for ranking/filtering.
PINECONE_RETRIEVAL_BUFFER = 2


def get_arxiv_fetch_count(user_id=None) -> int:
    """
    Load the arXiv fetch count from DB (preferred) or fall back to constant.

    This is the dashboard "Papers per Run" setting — it controls how many
    papers to fetch from the arXiv API each run.

    Returns:
        int: The max papers to fetch from arXiv per run.
    """
    try:
        from db.database import is_database_configured
        if not is_database_configured():
            logger.info("[SETTINGS] arxiv_fetch_count=%d source=default (no DB)", DEFAULT_ARXIV_FETCH_COUNT)
            return DEFAULT_ARXIV_FETCH_COUNT

        from db.store import get_default_store
        store = get_default_store()

        if user_id is None:
            user = store.get_or_create_default_user()
            user_id_val = user["id"]
        else:
            user_id_val = user_id

        from uuid import UUID
        uid = UUID(str(user_id_val)) if not isinstance(user_id_val, UUID) else user_id_val
        value = store.get_retrieval_max_results(uid)
        logger.info("[SETTINGS] arxiv_fetch_count=%d source=db", value)
        return value
    except Exception as e:
        logger.warning("[SETTINGS] arxiv_fetch_count=%d source=default (error: %s)", DEFAULT_ARXIV_FETCH_COUNT, e)
        return DEFAULT_ARXIV_FETCH_COUNT


# Backward-compatible alias
def get_retrieval_max_results(user_id=None) -> int:
    """Deprecated alias for get_arxiv_fetch_count()."""
    return get_arxiv_fetch_count(user_id)


# =============================================================================
# Prompt Template Definitions
# =============================================================================

class PromptTemplate(str, Enum):
    """Enumeration of all recognized prompt templates."""
    TOPIC_VENUE_TIME = "topic_venue_time"          # Template 1
    TOPIC_TIME = "topic_time"                       # Template 2
    TOPIC_ONLY = "topic_only"                       # Template 3
    TOP_K_PAPERS = "top_k_papers"                   # Template 4
    TOP_K_TIME = "top_k_time"                       # Template 5
    SURVEY_REVIEW = "survey_review"                 # Template 6
    METHOD_FOCUSED = "method_focused"               # Template 7
    APPLICATION_FOCUSED = "application_focused"    # Template 8
    EMERGING_TRENDS = "emerging_trends"            # Template 9
    STRUCTURED_OUTPUT = "structured_output"        # Template 10
    FETCH_BY_ID = "fetch_by_id"                     # Template 11 — single paper lookup
    UNRECOGNIZED = "unrecognized"                   # Fallback


@dataclass
class ParsedPrompt:
    """
    Structured representation of a parsed user prompt.
    
    Contains all extracted parameters from the user's query,
    including the detected template and requested output count.
    """
    template: PromptTemplate = PromptTemplate.UNRECOGNIZED
    topic: str = ""
    venue: Optional[str] = None
    time_period: Optional[str] = None
    time_days: Optional[int] = None  # Converted time period in days
    requested_count: Optional[int] = None  # K from "top K papers"
    method_or_approach: Optional[str] = None
    application_domain: Optional[str] = None
    is_survey_request: bool = False
    is_trends_request: bool = False
    is_structured_output: bool = False
    exclude_topics: List[str] = field(default_factory=list)  # Topics to exclude from results
    interests_only: str = ""  # ONLY the research interest keywords, no exclude/meta text
    arxiv_id: Optional[str] = None  # arXiv paper ID for single-paper lookup
    raw_prompt: str = ""
    
    @property
    def output_count(self) -> int:
        """
        Get the number of papers to output.
        
        Returns the explicitly requested count (K) if the user specified one.
        Otherwise, falls back to the dashboard "Papers per Run" setting
        (arxiv_fetch_count) so that the system respects the user's configured
        limit.  DEFAULT_OUTPUT_COUNT is only used when neither is available.
        """
        if self.requested_count:
            return self.requested_count
        if self._arxiv_fetch_count is not None:
            return self._arxiv_fetch_count
        return DEFAULT_OUTPUT_COUNT
    
    _arxiv_fetch_count: Optional[int] = None  # Injected DB value (dashboard "Papers per Run")

    @property
    def arxiv_fetch_count(self) -> int:
        """
        How many papers to fetch from the arXiv API.
        
        Uses DB-loaded value (dashboard setting) if set, else DEFAULT_ARXIV_FETCH_COUNT.
        """
        if self._arxiv_fetch_count is not None:
            return self._arxiv_fetch_count
        return DEFAULT_ARXIV_FETCH_COUNT

    # Backward-compatible alias
    _retrieval_max: Optional[int] = None

    @property
    def retrieval_count(self) -> int:
        """Deprecated alias for arxiv_fetch_count (backward compat)."""
        if self._retrieval_max is not None:
            return self._retrieval_max
        return self.arxiv_fetch_count

    @property
    def pinecone_retrieval_count(self) -> int:
        """
        How many papers to retrieve from Pinecone for ranking.
        
        Automatically set to output_count + PINECONE_RETRIEVAL_BUFFER
        so the agent has a small surplus for ranking/filtering.
        """
        return self.output_count + PINECONE_RETRIEVAL_BUFFER
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "template": self.template.value,
            "topic": self.topic,
            "venue": self.venue,
            "time_period": self.time_period,
            "time_days": self.time_days,
            "requested_count": self.requested_count,
            "output_count": self.output_count,
            "arxiv_fetch_count": self.arxiv_fetch_count,
            "pinecone_retrieval_count": self.pinecone_retrieval_count,
            "method_or_approach": self.method_or_approach,
            "application_domain": self.application_domain,
            "is_survey_request": self.is_survey_request,
            "is_trends_request": self.is_trends_request,
            "is_structured_output": self.is_structured_output,
            "raw_prompt": self.raw_prompt,
        }


# =============================================================================
# Pattern Definitions for Template Matching
# =============================================================================

# Pattern to extract numbers (for "top 5", "5 papers", etc.)
NUMBER_PATTERNS = [
    r'\btop\s+(\d+)\b',                    # "top 5"
    r'\b(\d+)\s+(?:papers?|articles?)\b',  # "5 papers", "10 articles"
    r'\bfind\s+(\d+)\b',                   # "find 5"
    r'\bget\s+(\d+)\b',                    # "get 5"
    r'\bshow\s+(?:me\s+)?(\d+)\b',         # "show me 5", "show 5"
    r'\b(\d+)\s+most\b',                   # "5 most relevant"
    r'\b(\d+)\s+recent\b',                 # "5 recent papers"
]

# Pattern to extract time periods
TIME_PATTERNS = [
    (r'\b(?:last|past)\s+(\d+)\s+days?\b', lambda x: int(x)),
    (r'\b(?:last|past)\s+(\d+)\s+weeks?\b', lambda x: int(x) * 7),
    (r'\b(?:last|past)\s+(\d+)\s+months?\b', lambda x: int(x) * 30),
    (r'\b(?:last|past)\s+year\b', lambda _: 365),
    (r'\b(?:last|past)\s+week\b', lambda _: 7),      # singular week without number
    (r'\b(?:last|past)\s+month\b', lambda _: 30),    # singular month without number
    (r'\bthis\s+week\b', lambda _: 7),
    (r'\bthis\s+month\b', lambda _: 30),
    (r'\brecent(?:ly)?\b', lambda _: 7),  # Default "recent" to 7 days
    (r'\bwithin\s+(\d+)\s+days?\b', lambda x: int(x)),
    (r'\bfrom\s+(?:the\s+)?(?:last|past)\s+(\d+)\s+days?\b', lambda x: int(x)),
    (r'\bfrom\s+(?:the\s+)?(?:last|past)\s+week\b', lambda _: 7),   # "from the last week"
    (r'\bfrom\s+(?:the\s+)?(?:last|past)\s+month\b', lambda _: 30), # "from the past month"
]

# Keywords for template detection
SURVEY_KEYWORDS = ['survey', 'review', 'overview', 'comprehensive', 'systematic review']
TRENDS_KEYWORDS = ['trends', 'trending', 'emerging', 'cutting-edge', 'frontier', 'state-of-the-art']
STRUCTURED_KEYWORDS = ['including title', 'with authors', 'include summary', 'structured', 'formatted']
METHOD_KEYWORDS = ['using', 'based on', 'focus on', 'that uses', 'utilizing', 'employing', 'with method']
APPLICATION_KEYWORDS = ['applied to', 'for', 'in the field of', 'in domain', 'in the area of']
VENUE_KEYWORDS = ['from', 'in', 'published in', 'appearing in']
VENUE_NAMES = ['arxiv', 'neurips', 'icml', 'iclr', 'acl', 'emnlp', 'cvpr', 'iccv', 'eccv', 'aaai', 'ijcai']


# =============================================================================
# Prompt Parser
# =============================================================================

class PromptParser:
    """
    Parses user prompts to extract structured parameters and detect templates.
    
    The parser identifies:
    - Topic/subject of interest
    - Time constraints (last week, past month, etc.)
    - Requested paper count (K)
    - Venue/publication filters
    - Method or approach focus
    - Application domain focus
    - Survey/review requests
    - Trend analysis requests
    """
    
    def __init__(self):
        self._number_patterns = [re.compile(p, re.IGNORECASE) for p in NUMBER_PATTERNS]
        self._time_patterns = [(re.compile(p, re.IGNORECASE), fn) for p, fn in TIME_PATTERNS]
    
    def parse(self, prompt: str) -> ParsedPrompt:
        """
        Parse a user prompt and extract all relevant parameters.
        
        Args:
            prompt: The raw user prompt string
            
        Returns:
            ParsedPrompt with all extracted parameters
        """
        result = ParsedPrompt(raw_prompt=prompt)
        
        # Extract exclude topics from prompt text
        result.exclude_topics = self._extract_exclude_topics(prompt)
        prompt_lower = prompt.lower()
        
        # Extract requested count (K)
        result.requested_count = self._extract_count(prompt)
        
        # Extract time period
        result.time_period, result.time_days = self._extract_time_period(prompt)
        
        # Check for special request types
        result.is_survey_request = any(kw in prompt_lower for kw in SURVEY_KEYWORDS)
        result.is_trends_request = any(kw in prompt_lower for kw in TRENDS_KEYWORDS)
        result.is_structured_output = any(kw in prompt_lower for kw in STRUCTURED_KEYWORDS)
        
        # Extract venue if mentioned
        result.venue = self._extract_venue(prompt)
        
        # Extract method/approach focus
        result.method_or_approach = self._extract_method(prompt)
        
        # Extract application domain
        result.application_domain = self._extract_application(prompt)
        
        # Extract clean interest keywords (ONLY the research interests, no exclude/meta text)
        result.interests_only = self._extract_interests_only(prompt, result)
        
        # Extract topic (what remains after removing meta-information)
        result.topic = self._extract_topic(prompt, result)
        
        # Extract arXiv ID for direct paper lookup
        result.arxiv_id = self._extract_arxiv_id(prompt)

        # Determine template
        result.template = self._determine_template(result)
        
        return result
    
    def _extract_count(self, prompt: str) -> Optional[int]:
        """Extract the requested paper count from the prompt."""
        for pattern in self._number_patterns:
            match = pattern.search(prompt)
            if match:
                count = int(match.group(1))
                # Sanity check: limit to reasonable range
                if 1 <= count <= 100:
                    return count
        return None
    
    def _extract_time_period(self, prompt: str) -> Tuple[Optional[str], Optional[int]]:
        """Extract time period from the prompt."""
        for pattern, converter in self._time_patterns:
            match = pattern.search(prompt)
            if match:
                raw_period = match.group(0)
                # Extract the numeric part if present
                groups = match.groups()
                if groups and groups[0]:
                    days = converter(groups[0])
                else:
                    days = converter(None)
                return raw_period, days
        return None, None
    
    def _extract_venue(self, prompt: str) -> Optional[str]:
        """Extract venue/publication from the prompt."""
        prompt_lower = prompt.lower()
        for venue in VENUE_NAMES:
            if venue in prompt_lower:
                return venue
        return None
    
    def _extract_method(self, prompt: str) -> Optional[str]:
        """Extract method or approach focus from the prompt."""
        for keyword in METHOD_KEYWORDS:
            # Use word boundaries to avoid matching within words
            pattern = re.compile(rf'\b{re.escape(keyword)}\s+(.+?)(?:published|from the|[.,;]|$)', re.IGNORECASE)
            match = pattern.search(prompt)
            if match:
                after = match.group(1).strip()
                if after:
                    return after
        return None
    
    def _extract_application(self, prompt: str) -> Optional[str]:
        """Extract application domain focus from the prompt."""
        for keyword in APPLICATION_KEYWORDS:
            # Use word boundaries to avoid matching within words (e.g., "for" in "transformers")
            pattern = re.compile(rf'\b{re.escape(keyword)}\s+(.+?)(?:using|with|[.,;]|$)', re.IGNORECASE)
            match = pattern.search(prompt)
            if match:
                after = match.group(1).strip()
                if after:
                    return after
        return None
    
    def _extract_exclude_topics(self, prompt: str) -> List[str]:
        """Extract topics to exclude from the prompt text.
        
        Carefully handles sentence boundaries so we don't greedily
        capture unrelated text (e.g., "Focus on papers...") as an
        exclude topic.
        """
        import re
        # Match patterns like "Exclude the following topics...: <topics>"
        # or "Exclude: <topics>" or "exclude <topics>"
        # Key fix: stop at sentence-boundary words like "Focus", "Find",
        # newline-period, or double-newline — not just period.
        # Use a lookahead for common follow-up phrases.
        BOUNDARY = r'(?=\s*(?:Focus|Find|Search|Look|Priorit|Include|Published|$)|\n\n|\.$)'
        patterns = [
            r'[Ee]xclude\s+(?:the\s+following\s+topics?\s*(?:if\s+applicable)?\s*[:\n]+)\s*(.+?)' + BOUNDARY,
            r'[Ee]xclude\s*:\s*(.+?)' + BOUNDARY,
            r'[Tt]opics?\s+to\s+exclude\s*[:\n]+\s*(.+?)' + BOUNDARY,
        ]
        for pattern in patterns:
            m = re.search(pattern, prompt, re.DOTALL)
            if m:
                raw = m.group(1).strip().rstrip('.')
                # Split by comma, newline, or semicolon
                topics = [t.strip() for t in re.split(r'[,;\n]+', raw) if t.strip()]
                return topics
        return []

    def _extract_interests_only(self, prompt: str, parsed: ParsedPrompt) -> str:
        """
        Extract ONLY the research interest keywords from the prompt.
        
        This specifically isolates the interest portion and strips out
        all meta-text (exclude clauses, time instructions, etc.).
        Returns a clean comma-separated list like:
            "Multi Armed Bandits, PCA, TSNE, Behavioral Economics"
            
        This is the AUTHORITATIVE source for category mapping and
        arXiv query construction.
        """
        # Strategy 1: Look for explicit "research interests:" pattern
        interest_patterns = [
            r'(?:research\s+)?interests?\s*:\s*(.+?)(?:\.\s*[Ee]xclude|\.\s*[Ff]ocus|\.\s*[Ff]ind|\.\s*[Pp]ublished|\.\s*$|$)',
            r'related\s+to\s+(?:the\s+following\s+(?:research\s+)?interests?\s*:\s*)?(.+?)(?:\.\s*[Ee]xclude|\.\s*[Ff]ocus|\.\s*[Ff]ind|\.\s*[Pp]ublished|\.\s*$|$)',
            r'(?:papers?|articles?|research)\s+(?:on|about|regarding)\s+(.+?)(?:\.\s*[Ee]xclude|\.\s*[Ff]ocus|\.\s*[Ff]ind|\.\s*[Pp]ublished|\.\s*$|$)',
        ]
        
        for pattern in interest_patterns:
            m = re.search(pattern, prompt, re.IGNORECASE | re.DOTALL)
            if m:
                raw = m.group(1).strip().rstrip('.')
                # Remove any remaining meta text
                raw = re.sub(r'\s*[Ee]xclude\s+.*$', '', raw, flags=re.DOTALL).strip()
                if raw:
                    return raw
        
        # Strategy 2: If topic is available, strip the exclude portion from it
        # This is a fallback for prompts that don't match explicit patterns
        topic = parsed.topic or prompt
        # Remove everything from "Exclude" onwards
        topic = re.sub(r'\s*[Ee]xclude\s+.*$', '', topic, flags=re.DOTALL).strip()
        # Remove common instruction words
        topic = re.sub(r'\s*(?:Focus on|Published|Find|Search).*$', '', topic, flags=re.IGNORECASE | re.DOTALL).strip()
        return topic.rstrip('.')

    def _extract_topic(self, prompt: str, parsed: ParsedPrompt) -> str:
        """
        Extract the main topic from the prompt.
        
        Removes meta-information (counts, time periods, exclude clauses, etc.)
        to get the core topic.
        """
        topic = prompt
        
        # ==================================================================
        # CRITICAL: Strip the entire "Exclude..." clause FIRST so exclude
        # terms don't leak into the topic text and contaminate downstream
        # processing (category mapping, keyword matching, etc.)
        # ==================================================================
        topic = re.sub(
            r'\s*[Ee]xclude\s+(?:the\s+following\s+)?(?:topics?\s*)?(?:if\s+applicable\s*)?[:\s].*?(?=\s*(?:Focus|Find|Search|Priorit|Published|\.\s|$))',
            '',
            topic,
            flags=re.DOTALL | re.IGNORECASE,
        )
        
        # Remove common prefixes
        prefixes = [
            r'^(?:find|get|show|provide|give|list|search for|look for|retrieve)\s+',
            r'^(?:me\s+)?',
            r'^(?:recent|latest|new|top\s+\d+)\s+',
            r'^\d+\s+',
            r'^(?:papers?|articles?|research)\s+(?:on|about|regarding|related to)\s+',
        ]
        for prefix in prefixes:
            topic = re.sub(prefix, '', topic, flags=re.IGNORECASE)
        
        # Remove time period mentions
        if parsed.time_period:
            topic = topic.replace(parsed.time_period, '')
        
        # Remove venue mentions
        if parsed.venue:
            topic = re.sub(rf'\b(?:from|in|published in)\s+{parsed.venue}\b', '', topic, flags=re.IGNORECASE)
        
        # Remove method/approach mentions
        if parsed.method_or_approach:
            for kw in METHOD_KEYWORDS:
                topic = re.sub(rf'\b{kw}\s+{re.escape(parsed.method_or_approach)}\b', '', topic, flags=re.IGNORECASE)
        
        # Remove application domain mentions
        if parsed.application_domain:
            for kw in APPLICATION_KEYWORDS:
                topic = re.sub(rf'\b{kw}\s+{re.escape(parsed.application_domain)}\b', '', topic, flags=re.IGNORECASE)
        
        # Clean up
        topic = re.sub(r'\s+', ' ', topic).strip()
        topic = re.sub(r'^(?:papers?|articles?|research)\s*(?:on|about)?\s*', '', topic, flags=re.IGNORECASE)
        
        return topic.strip()
    
    # Regex for arXiv IDs: standard (2301.12345) or old-style (hep-th/9901001)
    _ARXIV_ID_RE = re.compile(r'\b(\d{4}\.\d{4,5}(?:v\d+)?)\b')
    _ARXIV_OLD_ID_RE = re.compile(r'\b([a-z-]+/\d{7})\b')

    def _extract_arxiv_id(self, prompt: str) -> Optional[str]:
        """Extract an arXiv paper ID from the prompt, if present."""
        m = self._ARXIV_ID_RE.search(prompt)
        if m:
            return m.group(1)
        m = self._ARXIV_OLD_ID_RE.search(prompt)
        if m:
            return m.group(1)
        return None

    def _determine_template(self, parsed: ParsedPrompt) -> PromptTemplate:
        """Determine which template the parsed prompt matches."""
        # Template 11: Fetch by arXiv ID (highest priority)
        if parsed.arxiv_id:
            return PromptTemplate.FETCH_BY_ID

        has_topic = bool(parsed.topic)
        has_venue = bool(parsed.venue)
        has_time = bool(parsed.time_period)
        has_count = parsed.requested_count is not None
        
        # Template 10: Structured Output
        if parsed.is_structured_output:
            return PromptTemplate.STRUCTURED_OUTPUT
        
        # Template 9: Emerging Trends
        if parsed.is_trends_request:
            return PromptTemplate.EMERGING_TRENDS
        
        # Template 6: Survey / Review
        if parsed.is_survey_request:
            return PromptTemplate.SURVEY_REVIEW
        
        # Template 7: Method-Focused
        if parsed.method_or_approach:
            return PromptTemplate.METHOD_FOCUSED
        
        # Template 8: Application-Focused
        if parsed.application_domain:
            return PromptTemplate.APPLICATION_FOCUSED
        
        # Template 1: Topic + Venue + Time
        if has_topic and has_venue and has_time:
            return PromptTemplate.TOPIC_VENUE_TIME
        
        # Template 5: Top-K + Time
        if has_count and has_time:
            return PromptTemplate.TOP_K_TIME
        
        # Template 4: Top-K Papers
        if has_count:
            return PromptTemplate.TOP_K_PAPERS
        
        # Template 2: Topic + Time
        if has_topic and has_time:
            return PromptTemplate.TOPIC_TIME
        
        # Template 3: Topic Only
        if has_topic:
            return PromptTemplate.TOPIC_ONLY
        
        return PromptTemplate.UNRECOGNIZED


# =============================================================================
# Output Enforcer
# =============================================================================

@dataclass
class OutputEnforcementResult:
    """Result of output enforcement operation."""
    papers: List[Dict[str, Any]] = field(default_factory=list)
    requested_count: int = 0
    actual_count: int = 0
    truncated: bool = False
    insufficient: bool = False
    message: Optional[str] = None


class OutputEnforcer:
    """
    Enforces strict output limits based on user's requested paper count.
    
    CRITICAL RULES:
    - If user requests K papers, output EXACTLY K papers (or fewer if unavailable)
    - Never output more than requested
    - Clearly state if fewer papers available than requested
    """
    
    def enforce(
        self,
        papers: List[Dict[str, Any]],
        parsed_prompt: ParsedPrompt,
        sort_key: str = "relevance_score",
        sort_descending: bool = True,
    ) -> OutputEnforcementResult:
        """
        Enforce output limits on a list of papers.
        
        Args:
            papers: List of paper dictionaries
            parsed_prompt: The parsed user prompt with requested count
            sort_key: Key to sort papers by before truncation
            sort_descending: Whether to sort in descending order
            
        Returns:
            OutputEnforcementResult with exactly K papers (or fewer if unavailable)
        """
        result = OutputEnforcementResult(
            requested_count=parsed_prompt.output_count,
        )
        
        if not papers:
            result.insufficient = True
            result.message = f"No papers found matching your criteria. You requested {result.requested_count} papers."
            return result
        
        # Sort papers by the specified key
        sorted_papers = self._sort_papers(papers, sort_key, sort_descending)
        
        # Truncate to exactly K papers
        k = parsed_prompt.output_count
        
        if len(sorted_papers) >= k:
            # Normal case: we have enough papers
            result.papers = sorted_papers[:k]
            result.actual_count = k
            result.truncated = len(sorted_papers) > k
        else:
            # Insufficient papers case
            result.papers = sorted_papers
            result.actual_count = len(sorted_papers)
            result.insufficient = True
            result.message = (
                f"You requested {k} papers, but only {len(sorted_papers)} "
                f"high-quality papers were found matching your criteria."
            )
        
        return result
    
    def _sort_papers(
        self,
        papers: List[Dict[str, Any]],
        sort_key: str,
        descending: bool = True,
    ) -> List[Dict[str, Any]]:
        """Sort papers by specified key."""
        # Handle papers that might not have the sort key
        def get_sort_value(paper: Dict[str, Any]) -> float:
            value = paper.get(sort_key)
            if value is None:
                # Fallback to other scoring keys
                for fallback in ["importance_score", "novelty_score", "score"]:
                    if paper.get(fallback) is not None:
                        return float(paper[fallback])
                return 0.0
            return float(value) if not isinstance(value, str) else 0.0
        
        return sorted(papers, key=get_sort_value, reverse=descending)


# =============================================================================
# Prompt Controller (Main Interface)
# =============================================================================

class PromptController:
    """
    Main system-level controller for research prompts.
    
    Provides unified interface for:
    1. Parsing prompts to standardized templates
    2. Determining retrieval limits
    3. Enforcing output limits
    """
    
    def __init__(self):
        self.parser = PromptParser()
        self.enforcer = OutputEnforcer()
    
    def parse_prompt(self, prompt: str) -> ParsedPrompt:
        """Parse a user prompt into structured parameters."""
        return self.parser.parse(prompt)
    
    def get_arxiv_fetch_count(self, parsed: ParsedPrompt) -> int:
        """Get the number of papers to fetch from arXiv (dashboard setting)."""
        return parsed.arxiv_fetch_count

    def get_retrieval_count(self, parsed: ParsedPrompt) -> int:
        """Deprecated alias for get_arxiv_fetch_count."""
        return parsed.arxiv_fetch_count
    
    def get_output_count(self, parsed: ParsedPrompt) -> int:
        """Get the number of papers to show to the user."""
        return parsed.output_count
    
    def enforce_output(
        self,
        papers: List[Dict[str, Any]],
        parsed: ParsedPrompt,
        sort_key: str = "relevance_score",
    ) -> OutputEnforcementResult:
        """
        Enforce output limits on retrieved papers.
        
        Returns exactly K papers (or fewer if unavailable).
        """
        return self.enforcer.enforce(papers, parsed, sort_key)
    
    def validate_output_count(
        self,
        papers: List[Dict[str, Any]],
        parsed: ParsedPrompt,
    ) -> Tuple[bool, str]:
        """
        Validate that output count matches user's request.
        
        Returns (is_valid, error_message).
        """
        expected = parsed.output_count
        actual = len(papers)
        
        if actual > expected:
            return False, f"CRITICAL: Output has {actual} papers but user requested {expected}"
        
        return True, ""
    
    # =========================================================================
    # Database Persistence Methods
    # =========================================================================
    
    def save_prompt_to_db(
        self,
        parsed: ParsedPrompt,
        run_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Save a parsed prompt to the database for audit/compliance tracking.
        
        Args:
            parsed: ParsedPrompt object from parse_prompt()
            run_id: Optional agent run ID
            
        Returns:
            Prompt ID (UUID string) if saved, None if failed
        """
        try:
            from db.data_service import save_prompt_request
            
            parsed_data = {
                "template": parsed.template.value if hasattr(parsed.template, 'value') else str(parsed.template),
                "topic": parsed.topic,
                "venue": parsed.venue,
                "time_period": parsed.time_period,
                "time_days": parsed.time_days,
                "requested_count": parsed.requested_count,
                "output_count": parsed.output_count,
                "arxiv_fetch_count": parsed.arxiv_fetch_count,
            "pinecone_retrieval_count": parsed.pinecone_retrieval_count,
                "method_or_approach": parsed.method_or_approach,
                "application_domain": parsed.application_domain,
                "is_survey_request": parsed.is_survey_request,
                "is_trends_request": parsed.is_trends_request,
                "is_structured_output": parsed.is_structured_output,
            }
            
            result = save_prompt_request(
                raw_prompt=parsed.raw_prompt,
                parsed_data=parsed_data,
                run_id=run_id,
            )
            
            if result:
                logger.info(f"Saved prompt to DB: {result.get('id')} (template: {parsed.template})")
                return result.get("id")
            return None
            
        except Exception as e:
            logger.error(f"Failed to save prompt to DB: {e}")
            return None
    
    def update_prompt_compliance(
        self,
        prompt_id: str,
        papers_retrieved: int,
        papers_returned: int,
        output_enforced: bool = False,
        output_insufficient: bool = False,
        compliance_message: Optional[str] = None,
    ) -> bool:
        """
        Update the compliance status of a saved prompt after output enforcement.
        
        Args:
            prompt_id: UUID of the prompt request (from save_prompt_to_db)
            papers_retrieved: Number of papers retrieved internally
            papers_returned: Number of papers returned to user
            output_enforced: Whether output was truncated
            output_insufficient: Whether fewer papers than requested
            compliance_message: Any compliance notes
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            from db.data_service import update_prompt_compliance
            
            result = update_prompt_compliance(
                prompt_id=prompt_id,
                papers_retrieved=papers_retrieved,
                papers_returned=papers_returned,
                output_enforced=output_enforced,
                output_insufficient=output_insufficient,
                compliance_message=compliance_message,
            )
            
            if result:
                logger.info(f"Updated prompt compliance: {result.get('compliance_status')}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to update prompt compliance: {e}")
            return False
    
    def parse_and_save(
        self,
        prompt: str,
        run_id: Optional[str] = None,
    ) -> Tuple[ParsedPrompt, Optional[str]]:
        """
        Parse a prompt and save it to the database in one call.
        
        Args:
            prompt: The raw user prompt
            run_id: Optional agent run ID
            
        Returns:
            Tuple of (ParsedPrompt, prompt_id or None)
        """
        parsed = self.parse_prompt(prompt)
        prompt_id = self.save_prompt_to_db(parsed, run_id)
        return parsed, prompt_id


# =============================================================================
# Global Instance
# =============================================================================

# Global prompt controller instance
prompt_controller = PromptController()


# =============================================================================
# Self-Check
# =============================================================================

def self_check() -> bool:
    """
    Run self-check tests for the PromptController.
    
    Returns:
        True if all checks pass, False otherwise.
    """
    print("=" * 60)
    print("PromptController Self-Check")
    print("=" * 60)
    
    all_passed = True
    
    def check(name: str, condition: bool) -> bool:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        return condition
    
    controller = PromptController()
    
    # Test 1: Basic parsing
    print("\n1. Basic Prompt Parsing:")
    parsed = controller.parse_prompt("Find top 5 papers on transformers")
    all_passed &= check("extracts count=5", parsed.requested_count == 5)
    all_passed &= check("output_count=5", parsed.output_count == 5)
    all_passed &= check("has topic", "transformer" in parsed.topic.lower())
    all_passed &= check("template=TOP_K_PAPERS", parsed.template == PromptTemplate.TOP_K_PAPERS)
    
    # Test 2: Time period extraction
    print("\n2. Time Period Extraction:")
    parsed = controller.parse_prompt("Papers from the last 7 days on NLP")
    all_passed &= check("extracts days=7", parsed.time_days == 7)
    all_passed &= check("has time_period", parsed.time_period is not None)
    all_passed &= check("template=TOPIC_TIME", parsed.template == PromptTemplate.TOPIC_TIME)
    
    # Test 3: Default output count
    print("\n3. Default Output Count:")
    parsed = controller.parse_prompt("Find recent papers on machine learning")
    all_passed &= check("requested_count is None", parsed.requested_count is None)
    all_passed &= check(f"output_count={DEFAULT_OUTPUT_COUNT}", parsed.output_count == DEFAULT_OUTPUT_COUNT)
    
    # Test 4: Retrieval count always MAX
    print("\n4. Retrieval Count:")
    parsed = controller.parse_prompt("Give me 3 papers on LLMs")
    all_passed &= check(f"retrieval_count={MAX_RETRIEVAL_RESULTS}", parsed.retrieval_count == MAX_RETRIEVAL_RESULTS)
    all_passed &= check("output_count=3", parsed.output_count == 3)
    
    # Test 5: Survey detection
    print("\n5. Survey Detection:")
    parsed = controller.parse_prompt("Find survey papers on deep learning")
    all_passed &= check("is_survey_request=True", parsed.is_survey_request)
    all_passed &= check("template=SURVEY_REVIEW", parsed.template == PromptTemplate.SURVEY_REVIEW)
    
    # Test 6: Trends detection
    print("\n6. Trends Detection:")
    parsed = controller.parse_prompt("What are the emerging trends in AI?")
    all_passed &= check("is_trends_request=True", parsed.is_trends_request)
    all_passed &= check("template=EMERGING_TRENDS", parsed.template == PromptTemplate.EMERGING_TRENDS)
    
    # Test 7: Output enforcement with truncation
    print("\n7. Output Enforcement (Truncation):")
    papers = [
        {"arxiv_id": f"2501.{i:05d}", "title": f"Paper {i}", "relevance_score": 0.9 - i*0.1}
        for i in range(10)
    ]
    parsed = controller.parse_prompt("Top 5 papers on AI")
    result = controller.enforce_output(papers, parsed)
    all_passed &= check("returns 5 papers", result.actual_count == 5)
    all_passed &= check("truncated=True", result.truncated)
    all_passed &= check("papers sorted by relevance", result.papers[0]["relevance_score"] == 0.9)
    
    # Test 8: Output enforcement with insufficient papers
    print("\n8. Output Enforcement (Insufficient):")
    papers = [
        {"arxiv_id": f"2501.{i:05d}", "title": f"Paper {i}", "relevance_score": 0.9 - i*0.1}
        for i in range(3)
    ]
    parsed = controller.parse_prompt("Top 10 papers on quantum computing")
    result = controller.enforce_output(papers, parsed)
    all_passed &= check("returns 3 papers (all available)", result.actual_count == 3)
    all_passed &= check("insufficient=True", result.insufficient)
    all_passed &= check("has message", result.message is not None)
    
    # Test 9: Validation of output
    print("\n9. Output Validation:")
    parsed = controller.parse_prompt("Top 5 papers on robotics")
    papers = [{"id": i} for i in range(5)]
    is_valid, _ = controller.validate_output_count(papers, parsed)
    all_passed &= check("valid when count matches", is_valid)
    
    papers_too_many = [{"id": i} for i in range(10)]
    is_valid, error = controller.validate_output_count(papers_too_many, parsed)
    all_passed &= check("invalid when too many", not is_valid)
    all_passed &= check("has error message", "CRITICAL" in error)
    
    # Test 10: Various count formats
    print("\n10. Various Count Formats:")
    test_cases = [
        ("Show me 7 papers on NLP", 7),
        ("Find 10 articles on CV", 10),
        ("Get 3 most relevant papers", 3),
        ("15 recent papers on RL", 15),
        ("Top 20 papers", 20),
    ]
    for prompt, expected in test_cases:
        parsed = controller.parse_prompt(prompt)
        all_passed &= check(f'"{prompt[:30]}..." -> {expected}', parsed.requested_count == expected)
    
    # Test 11: Week/month time periods
    print("\n11. Week/Month Periods:")
    test_cases = [
        ("Papers from last week", 7),
        ("Papers from the past month", 30),
        ("Papers from last 2 weeks", 14),
        ("Papers from past 3 months", 90),
    ]
    for prompt, expected in test_cases:
        parsed = controller.parse_prompt(prompt)
        all_passed &= check(f'"{prompt}" -> {expected} days', parsed.time_days == expected)
    
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
