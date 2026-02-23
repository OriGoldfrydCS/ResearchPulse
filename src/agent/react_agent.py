"""
LangChain ReAct Agent - Research Awareness and Sharing Agent.

This module implements the ReAct (Reasoning + Acting) pattern for the
episodic research paper discovery and sharing agent.

**ReAct Loop:**
1. Thought: Agent reasons about what to do next
2. Action: Agent selects and invokes a tool
3. Observation: Agent receives tool output
4. Repeat until termination condition or TERMINATE action

**Stop Controller Integration:**
- Checked before each tool call
- Checked after each observation
- Enforces bounded execution (max runtime, max papers, etc.)

**Tools:**
- fetch_arxiv_papers: Retrieve papers from arXiv
- check_seen_papers: Identify unseen papers
- retrieve_similar_from_pinecone: RAG similarity search
- score_relevance_and_importance: Score paper relevance/novelty
- decide_delivery_action: Determine delivery actions
- persist_state: Save paper decisions
- generate_report: Create final run report
- terminate_run: End the episode
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# Add parent directories to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.stop_controller import StopController, StopPolicy, RunMetrics, StopReason
from agent.prompt_controller import (
    PromptController,
    ParsedPrompt,
    PromptTemplate,
    DEFAULT_ARXIV_FETCH_COUNT,
    MAX_RETRIEVAL_RESULTS,  # backward compat alias
    DEFAULT_OUTPUT_COUNT,
    prompt_controller,
    get_arxiv_fetch_count,
    get_retrieval_max_results,  # backward compat alias
)


# =============================================================================
# Tool Import and Registration
# =============================================================================

# Import all tools
from tools.fetch_arxiv import fetch_arxiv_papers, fetch_arxiv_papers_json
from tools.check_seen import check_seen_papers, check_seen_papers_json
from tools.retrieve_similar import retrieve_similar_from_pinecone, retrieve_similar_from_pinecone_json
from tools.score_relevance import score_relevance_and_importance, score_relevance_and_importance_json, score_papers_batch
from tools.decide_delivery import decide_delivery_action, decide_delivery_action_json, write_artifact_files, process_colleague_surplus
from tools.persist_state import persist_paper_decision, persist_paper_decisions_batch, reset_run_tracker
from tools.generate_report import generate_report, generate_report_json
from tools.terminate_run import terminate_run, terminate_run_json

# Import DB utilities - use data_service for DB-first access
from db.data_service import get_research_profile, get_colleagues, get_delivery_policy, is_db_available, save_artifacts_to_db

# Import autonomous components (with graceful degradation)
try:
    from config.feature_flags import is_feature_enabled, get_feature_config
    FEATURE_FLAGS_AVAILABLE = True
except ImportError:
    FEATURE_FLAGS_AVAILABLE = False


# =============================================================================
# Interest to arXiv Category Mapping
# =============================================================================

# Mapping of keywords/interests to arXiv categories (agent auto-maps these)
INTEREST_TO_CATEGORY_MAP = {
    # NLP / Language
    "natural language processing": ["cs.CL"],
    "nlp": ["cs.CL"],
    "language models": ["cs.CL", "cs.LG"],
    "large language models": ["cs.CL", "cs.LG"],
    "llm": ["cs.CL", "cs.LG"],
    "transformers": ["cs.CL", "cs.LG"],
    "text generation": ["cs.CL"],
    "machine translation": ["cs.CL"],
    "question answering": ["cs.CL", "cs.IR"],
    "text understanding": ["cs.CL"],
    "dialogue systems": ["cs.CL"],
    "chatbots": ["cs.CL", "cs.AI"],
    "conversation": ["cs.CL"],
    "sentiment analysis": ["cs.CL"],
    "named entity recognition": ["cs.CL"],
    
    # Machine Learning
    "machine learning": ["cs.LG", "stat.ML"],
    "deep learning": ["cs.LG"],
    "neural networks": ["cs.LG", "cs.NE"],
    "reinforcement learning": ["cs.LG", "cs.AI"],
    "supervised learning": ["cs.LG"],
    "unsupervised learning": ["cs.LG"],
    "transfer learning": ["cs.LG"],
    "few-shot learning": ["cs.LG", "cs.CL"],
    "meta-learning": ["cs.LG"],
    "optimization": ["cs.LG", "math.OC"],
    "multi-armed bandits": ["cs.LG", "stat.ML"],
    "multi armed bandits": ["cs.LG", "stat.ML"],
    "bandits": ["cs.LG", "stat.ML"],
    "bandit": ["cs.LG", "stat.ML"],
    "contextual bandits": ["cs.LG", "stat.ML"],
    "bayesian optimization": ["cs.LG", "stat.ML"],
    "gaussian processes": ["cs.LG", "stat.ML"],
    "causal inference": ["stat.ME", "stat.ML", "cs.LG"],
    "pca": ["stat.ML", "cs.LG", "stat.ME"],
    "principal component analysis": ["stat.ML", "cs.LG", "stat.ME"],
    "tsne": ["cs.LG", "stat.ML"],
    "t-sne": ["cs.LG", "stat.ML"],
    "dimensionality reduction": ["cs.LG", "stat.ML"],
    "clustering": ["cs.LG", "stat.ML"],
    "anomaly detection": ["cs.LG", "stat.ML"],
    "time series": ["cs.LG", "stat.ML", "stat.ME"],
    "regression": ["stat.ML", "stat.ME"],
    "classification": ["cs.LG", "stat.ML"],
    "ensemble methods": ["cs.LG", "stat.ML"],
    "random forests": ["cs.LG", "stat.ML"],
    "gradient boosting": ["cs.LG", "stat.ML"],
    "feature selection": ["cs.LG", "stat.ML"],
    "active learning": ["cs.LG", "stat.ML"],
    "online learning": ["cs.LG", "stat.ML"],
    "representation learning": ["cs.LG", "stat.ML"],
    "contrastive learning": ["cs.LG", "cs.CV"],
    
    # AI / Agents
    "artificial intelligence": ["cs.AI"],
    "ai agents": ["cs.AI", "cs.LG"],
    "autonomous agents": ["cs.AI", "cs.RO"],
    "multi-agent": ["cs.AI", "cs.MA"],
    "reasoning": ["cs.AI", "cs.CL"],
    "planning": ["cs.AI"],
    "knowledge representation": ["cs.AI"],
    
    # Information Retrieval / RAG
    "information retrieval": ["cs.IR"],
    "retrieval": ["cs.IR", "cs.CL"],
    "rag": ["cs.IR", "cs.CL"],
    "retrieval-augmented generation": ["cs.IR", "cs.CL"],
    "search": ["cs.IR"],
    "recommendation": ["cs.IR"],
    
    # Computer Vision
    "computer vision": ["cs.CV"],
    "image processing": ["cs.CV"],
    "object detection": ["cs.CV"],
    "image classification": ["cs.CV"],
    "image generation": ["cs.CV"],
    "video": ["cs.CV"],
    
    # Robotics
    "robotics": ["cs.RO"],
    "robot learning": ["cs.RO", "cs.LG"],
    
    # Security
    "security": ["cs.CR"],
    "cryptography": ["cs.CR"],
    "privacy": ["cs.CR", "cs.LG"],
    
    # Networking
    "networking": ["cs.NI"],
    "distributed systems": ["cs.DC"],
    
    # Other
    "databases": ["cs.DB"],
    "software engineering": ["cs.SE"],
    "human-computer interaction": ["cs.HC"],
    "hci": ["cs.HC"],
    "statistics": ["stat.TH", "stat.ML"],
    "graph neural networks": ["cs.LG"],
    "gnns": ["cs.LG"],
    "knowledge graphs": ["cs.AI", "cs.CL"],
    "embeddings": ["cs.CL", "cs.LG"],
    "attention mechanisms": ["cs.LG", "cs.CL"],
    "prompt engineering": ["cs.CL"],
    "in-context learning": ["cs.CL", "cs.LG"],
    "fine-tuning": ["cs.CL", "cs.LG"],
    
    # Biology & Life Sciences
    "biology": ["q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.MN", "q-bio.NC", "q-bio.OT", "q-bio.PE", "q-bio.QM", "q-bio.SC", "q-bio.TO"],
    "bioinformatics": ["q-bio.BM", "q-bio.GN", "q-bio.QM"],
    "genomics": ["q-bio.GN"],
    "genetics": ["q-bio.GN", "q-bio.PE"],
    "neuroscience": ["q-bio.NC"],
    "molecular biology": ["q-bio.BM"],
    "cell biology": ["q-bio.CB"],
    "computational biology": ["q-bio.QM", "q-bio.BM"],
    "life sciences": ["q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.NC"],
    
    # Physics
    "physics": ["physics.comp-ph", "physics.data-an", "hep-ph", "hep-th", "cond-mat.stat-mech"],
    "quantum": ["quant-ph"],
    "quantum computing": ["quant-ph", "cs.ET"],
    "quantum physics": ["quant-ph"],
    "high energy physics": ["hep-ph", "hep-th"],
    "condensed matter": ["cond-mat.stat-mech", "cond-mat.str-el"],
    "astrophysics": ["astro-ph.GA", "astro-ph.CO", "astro-ph.SR"],
    
    # Mathematics
    "mathematics": ["math.ST", "math.OC", "math.NA", "math.CO", "math.PR"],
    "math": ["math.ST", "math.OC", "math.NA"],
    "applied math": ["math.NA", "math.OC"],
    "probability": ["math.PR", "stat.TH"],
    "combinatorics": ["math.CO"],
    
    # Economics & Finance
    "economics": ["econ.EM", "econ.GN", "econ.TH"],
    "behavioral economics": ["econ.TH", "econ.GN"],
    "experimental economics": ["econ.TH", "econ.GN"],
    "game theory": ["cs.GT", "econ.TH"],
    "mechanism design": ["cs.GT", "econ.TH"],
    "auction": ["cs.GT", "econ.TH"],
    "decision making": ["cs.AI", "econ.TH", "stat.ML"],
    "finance": ["q-fin.ST", "q-fin.RM", "q-fin.PM", "q-fin.CP"],
    "econometrics": ["econ.EM", "stat.ME"],
    "financial mathematics": ["q-fin.MF", "q-fin.CP"],
    
    # Electrical Engineering
    "electrical engineering": ["eess.SP", "eess.SY"],
    "signal processing": ["eess.SP"],
    "control systems": ["eess.SY", "cs.SY"],
}

def map_interests_to_categories(interests_text: str, exclude: bool = False) -> List[str]:
    """
    Map user's free-text interests to arXiv categories.
    
    Uses word-boundary matching to avoid false positives
    (e.g., 'search' should not match inside 'research').
    
    Args:
        interests_text: Free-text description of research interests
        exclude: If True, these are topics to exclude
        
    Returns:
        List of arXiv category codes
    """
    if not interests_text:
        return []
    
    interests_lower = interests_text.lower()
    categories = set()
    
    for keyword, cats in INTEREST_TO_CATEGORY_MAP.items():
        # Use word-boundary matching to prevent partial matches
        # e.g., "search" should NOT match inside "research"
        pattern = r'(?:^|\b)' + re.escape(keyword) + r'(?:\b|$)'
        if re.search(pattern, interests_lower):
            for cat in cats:
                categories.add(cat)
    
    return list(categories)


# =============================================================================
# Agent State Models
# =============================================================================

class ToolCall(BaseModel):
    """Record of a single tool call."""
    tool_name: str = Field(..., description="Name of the tool called")
    input_args: Dict[str, Any] = Field(default_factory=dict, description="Tool input arguments")
    output: Any = Field(None, description="Tool output")
    timestamp: str = Field(..., description="When the call was made")
    duration_ms: float = Field(0, description="Call duration in milliseconds")
    success: bool = Field(True, description="Whether the call succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")


class AgentStep(BaseModel):
    """A single step in the ReAct loop."""
    step_number: int = Field(..., description="Step sequence number")
    thought: str = Field("", description="Agent's reasoning")
    action: Optional[str] = Field(None, description="Tool name to call")
    action_input: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    observation: Any = Field(None, description="Tool output")
    timestamp: str = Field(..., description="Step timestamp")


class AgentEpisode(BaseModel):
    """Complete record of an agent episode."""
    run_id: str = Field(..., description="Unique run identifier")
    start_time: str = Field(..., description="Episode start time")
    end_time: Optional[str] = Field(None, description="Episode end time")
    user_message: str = Field(..., description="Original user request")
    detected_template: Optional[str] = Field(None, description="Detected prompt template")
    requested_paper_count: Optional[int] = Field(None, description="Number of papers requested by user")
    output_paper_count: Optional[int] = Field(None, description="Actual number of papers in output")
    steps: List[AgentStep] = Field(default_factory=list, description="All steps taken")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="All tool calls made")
    stop_reason: Optional[str] = Field(None, description="Why the episode ended")
    final_report: Optional[Dict[str, Any]] = Field(None, description="Final run report")
    papers_processed: List[Dict[str, Any]] = Field(default_factory=list, description="Papers processed")
    decisions_made: List[Dict[str, Any]] = Field(default_factory=list, description="Decisions made")
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list, description="Actions taken")
    artifacts_generated: List[Dict[str, Any]] = Field(default_factory=list, description="Artifacts generated")
    
    class Config:
        arbitrary_types_allowed = True


class AgentConfig(BaseModel):
    """Configuration for the ReAct agent."""
    max_steps: int = Field(50, description="Maximum steps before forced termination")
    stop_policy: StopPolicy = Field(default_factory=StopPolicy, description="Stop policy configuration")
    use_mock_arxiv: bool = Field(True, description="Use mock arXiv data for demo")
    verbose: bool = Field(True, description="Enable verbose logging")
    initial_prompt: Optional[str] = Field(None, description="The user's initial prompt/message")
    parsed_prompt: Optional[ParsedPrompt] = Field(None, description="Parsed prompt with template and constraints")
    
    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# Tool Registry
# =============================================================================

class ToolRegistry:
    """Registry of available tools for the agent."""
    
    def __init__(self, research_profile: Dict[str, Any] = None):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, Dict] = {}
        self._research_profile = research_profile or {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register all default tools."""
        # Tool: fetch_arxiv_papers
        self.register(
            name="fetch_arxiv_papers",
            func=self._wrap_fetch_arxiv,
            schema={
                "description": "Fetch recent papers from arXiv matching category/keyword criteria",
                "parameters": ["categories_include", "categories_exclude", "max_results", "query"]
            }
        )
        
        # Tool: check_seen_papers
        self.register(
            name="check_seen_papers",
            func=self._wrap_check_seen,
            schema={
                "description": "Identify which papers have been seen before",
                "parameters": ["papers"]
            }
        )
        
        # Tool: retrieve_similar_from_pinecone
        self.register(
            name="retrieve_similar_from_pinecone",
            func=self._wrap_retrieve_similar,
            schema={
                "description": "Query Pinecone for similar documents via RAG",
                "parameters": ["query_text", "top_k", "similarity_threshold"]
            }
        )
        
        # Tool: score_relevance_and_importance
        self.register(
            name="score_relevance_and_importance",
            func=self._wrap_score_relevance,
            schema={
                "description": "Score a paper's relevance, novelty, and importance",
                "parameters": ["paper", "research_profile", "rag_results"]
            }
        )
        
        # Tool: decide_delivery_action
        self.register(
            name="decide_delivery_action",
            func=self._wrap_decide_delivery,
            schema={
                "description": "Decide delivery actions for a scored paper",
                "parameters": ["scored_paper", "delivery_policy", "colleagues"]
            }
        )
        
        # Tool: persist_state
        self.register(
            name="persist_state",
            func=self._wrap_persist_state,
            schema={
                "description": "Persist paper decisions to database",
                "parameters": ["paper_decision"]
            }
        )
        
        # Tool: generate_report
        self.register(
            name="generate_report",
            func=self._wrap_generate_report,
            schema={
                "description": "Generate final run report",
                "parameters": ["run_id", "start_time", "stop_reason", "papers", "decisions"]
            }
        )
        
        # Tool: terminate_run
        self.register(
            name="terminate_run",
            func=self._wrap_terminate_run,
            schema={
                "description": "Terminate the current run",
                "parameters": ["run_id", "stop_reason", "final_metrics"]
            }
        )
    
    def register(self, name: str, func: Callable, schema: Dict):
        """Register a tool."""
        self._tools[name] = func
        self._schemas[name] = schema
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all available tool names."""
        return list(self._tools.keys())
    
    def get_tools_description(self) -> str:
        """Get a formatted description of all tools."""
        lines = ["Available tools:"]
        for name, schema in self._schemas.items():
            desc = schema.get("description", "No description")
            lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)
    
    # Wrapper methods that normalize inputs/outputs
    
    def _wrap_fetch_arxiv(self, **kwargs) -> Dict[str, Any]:
        result = fetch_arxiv_papers(**kwargs, use_mock=False)
        return {
            "success": result.success,
            "papers": [p.model_dump() for p in result.papers],
            "total_found": result.total_found,
            "error": result.error
        }
    
    def _wrap_check_seen(self, **kwargs) -> Dict[str, Any]:
        return check_seen_papers_json(**kwargs)
    
    def _wrap_retrieve_similar(self, **kwargs) -> Dict[str, Any]:
        return retrieve_similar_from_pinecone_json(**kwargs)
    
    def _wrap_score_relevance(self, **kwargs) -> Dict[str, Any]:
        return score_relevance_and_importance_json(**kwargs)
    
    def _wrap_decide_delivery(self, **kwargs) -> Dict[str, Any]:
        # Pass researcher name and email from profile
        researcher_name = self._research_profile.get("researcher_name", "Researcher")
        researcher_email = self._research_profile.get("researcher_email", "")
        return decide_delivery_action_json(
            researcher_name=researcher_name,
            researcher_email=researcher_email,
            **kwargs
        )
    
    def _wrap_persist_state(self, **kwargs) -> Dict[str, Any]:
        result = persist_paper_decision(**kwargs)
        return result.model_dump()
    
    def _wrap_generate_report(self, **kwargs) -> Dict[str, Any]:
        return generate_report_json(**kwargs)
    
    def _wrap_terminate_run(self, **kwargs) -> Dict[str, Any]:
        return terminate_run_json(**kwargs)


# =============================================================================
# ReAct Agent Implementation
# =============================================================================

class ResearchReActAgent:
    """
    ReAct agent for research paper discovery and sharing.
    
    Implements the Thought -> Action -> Observation loop with
    StopController integration for bounded episodic execution.
    """
    
    def __init__(
        self,
        run_id: str,
        config: Optional[AgentConfig] = None,
        log_callback: Optional[Callable[[str, str, str], None]] = None,
        cancellation_check: Optional[Callable[[], bool]] = None,
    ):
        """
        Initialize the ReAct agent.
        
        Args:
            run_id: Unique identifier for this run
            config: Agent configuration (uses defaults if None)
            log_callback: Optional callback for logging (level, message, timestamp)
            cancellation_check: Optional callback to check if run was cancelled externally
        """
        self.run_id = run_id
        self.config = config or AgentConfig()
        self.log_callback = log_callback
        self._cancellation_check = cancellation_check
        
        # Load research profile and colleagues
        self._research_profile = get_research_profile()
        self._colleagues = get_colleagues()
        self._delivery_policy = get_delivery_policy()
        
        # Initialize components (after loading profile so we can pass it)
        self.tool_registry = ToolRegistry(research_profile=self._research_profile)
        self.stop_controller = StopController(policy=self.config.stop_policy)
        
        # Episode state
        self.episode: Optional[AgentEpisode] = None
        self.current_step = 0
        self._terminated = False
        self._cancelled = False
        
        # Working state (accumulated during run)
        self._fetched_papers: List[Dict] = []
        self._unseen_papers: List[Dict] = []
        self._scored_papers: List[Dict] = []
        self._decisions: List[Dict] = []
        self._actions: List[Dict] = []
        self._artifacts: List[Dict] = []
        
        # Prompt controller for template matching and output enforcement
        self._prompt_controller = prompt_controller
        self._parsed_prompt: Optional[ParsedPrompt] = None
        self._prompt_id: Optional[str] = None  # Database ID for prompt tracking
    
    def _log(self, level: str, message: str):
        """Log a message."""
        ts = datetime.utcnow().isoformat() + "Z"
        if self.log_callback:
            self.log_callback(level, message, ts)
        if self.config.verbose:
            print(f"[{level}] {message}")
    
    def _create_tool_call_record(
        self,
        tool_name: str,
        input_args: Dict,
        output: Any,
        duration_ms: float,
        success: bool,
        error: Optional[str] = None
    ) -> ToolCall:
        """Create a tool call record."""
        return ToolCall(
            tool_name=tool_name,
            input_args=input_args,
            output=output,
            timestamp=datetime.utcnow().isoformat() + "Z",
            duration_ms=duration_ms,
            success=success,
            error=error
        )
    
    def _invoke_tool(self, tool_name: str, **kwargs) -> Tuple[Any, bool, Optional[str]]:
        """
        Invoke a tool and return (output, success, error).
        """
        # Check for cancellation before every tool call
        if self._cancellation_check and self._cancellation_check():
            self._terminated = True
            self._cancelled = True
            self._log("INFO", "Run cancelled by user before tool execution")
            return None, False, "cancelled_by_user"
        
        start_time = datetime.utcnow()
        
        tool_func = self.tool_registry.get_tool(tool_name)
        if not tool_func:
            return None, False, f"Tool not found: {tool_name}"
        
        try:
            output = tool_func(**kwargs)
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Record the tool call
            record = self._create_tool_call_record(
                tool_name, kwargs, output, duration_ms, True, None
            )
            self.episode.tool_calls.append(record)
            
            self._log("INFO", f"Tool {tool_name} completed in {duration_ms:.1f}ms")
            return output, True, None
            
        except Exception as e:
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            error_msg = str(e)
            
            # Record failed call
            record = self._create_tool_call_record(
                tool_name, kwargs, None, duration_ms, False, error_msg
            )
            self.episode.tool_calls.append(record)
            
            self._log("ERROR", f"Tool {tool_name} failed: {error_msg}")
            return None, False, error_msg
    
    def _check_stop_condition(self) -> Tuple[bool, str]:
        """Check if we should stop the agent."""
        if self._terminated:
            return True, self.stop_controller.stop_reason
        
        # Check for external cancellation (user clicked Stop button)
        if self._cancellation_check and self._cancellation_check():
            self._terminated = True
            self._cancelled = True
            self._log("INFO", "Run cancelled by user")
            return True, "cancelled_by_user"
        
        should_stop, reason = self.stop_controller.should_stop()
        if should_stop:
            self._terminated = True
            return True, reason
        
        # Check max steps
        if self.current_step >= self.config.max_steps:
            self._terminated = True
            return True, "max_steps reached"
        
        return False, ""
    
    def _run_workflow(self, user_message: str) -> AgentEpisode:
        """
        Execute the full agent workflow.
        
        This implements a structured workflow rather than free-form ReAct:
        1. Fetch papers from arXiv
        2. Check which are unseen
        3. For each unseen paper: query RAG, score, decide, persist
        4. Generate report
        5. Terminate
        
        SYSTEM CONTROLLER INTEGRATION:
        - Dashboard "Papers per Run" controls how many papers to fetch from arXiv
        - Pinecone retrieval is automatic (output_count + buffer)
        - Final output is truncated to EXACTLY user's requested count (K)
        """
        self._log("INFO", f"Starting workflow for: {user_message[:100]}...")
        
        # Load DB-backed arXiv fetch count (dashboard "Papers per Run" setting)
        db_arxiv_fetch_count = get_arxiv_fetch_count()
        
        # Parse prompt for template matching and output constraints (CRITICAL)
        # Also saves prompt to database for audit/compliance tracking
        self._parsed_prompt, self._prompt_id = self._prompt_controller.parse_and_save(
            user_message, run_id=self.run_id
        )
        # Inject DB-backed arXiv fetch count into parsed prompt
        self._parsed_prompt._arxiv_fetch_count = db_arxiv_fetch_count
        self._parsed_prompt._retrieval_max = db_arxiv_fetch_count  # backward compat
        
        self._log("INFO", f"Detected prompt template: {self._parsed_prompt.template.value}")
        if self._prompt_id:
            self._log("INFO", f"Prompt saved to DB with ID: {self._prompt_id}")
        self._log("INFO", f"Requested output count: {self._parsed_prompt.output_count} papers")
        self._log("INFO", f"Internal retrieval limit: {self._parsed_prompt.retrieval_count} papers")
        if self._parsed_prompt.time_period:
            self._log("INFO", f"Time filter: {self._parsed_prompt.time_period} ({self._parsed_prompt.time_days} days)")
        if self._parsed_prompt.topic:
            self._log("INFO", f"Topic extracted: {self._parsed_prompt.topic}")
        
        # Merge exclude topics from prompt into the research profile's avoid_topics
        if self._parsed_prompt.exclude_topics:
            existing_avoid = self._research_profile.get("avoid_topics", [])
            merged = list(set(existing_avoid + self._parsed_prompt.exclude_topics))
            self._research_profile["avoid_topics"] = merged
            self._log("INFO", f"Exclude topics from prompt merged into avoid_topics: {merged}")
        
        # Store parsed info in episode for reporting
        self.episode.detected_template = self._parsed_prompt.template.value
        self.episode.requested_paper_count = self._parsed_prompt.output_count
        
        # Route: single-paper lookup by arXiv ID
        from agent.prompt_controller import PromptTemplate
        if self._parsed_prompt.template == PromptTemplate.FETCH_BY_ID:
            return self._run_single_paper_workflow(self._parsed_prompt.arxiv_id)

        # Reset run tracker for idempotency
        reset_run_tracker()
        
        # Step 1: Fetch papers from arXiv
        self._log("INFO", "Step 1: Fetching papers from arXiv...")
        should_stop, reason = self._check_stop_condition()
        if should_stop:
            return self._finalize_episode(reason)
        
        # Get arXiv categories - prefer explicitly set categories, otherwise map from user interests
        categories_include = self._research_profile.get("arxiv_categories_include", [])
        categories_exclude = self._research_profile.get("arxiv_categories_exclude", [])
        
        # If no explicit categories, map from user's free-text interests
        if not categories_include:
            interests_include = self._research_profile.get("interests_include", "")
            if interests_include:
                categories_include = map_interests_to_categories(interests_include)
                self._log("INFO", f"Mapped interests to categories: {categories_include}")
        
        # Fall back to broad categories if still empty.
        # Use general ML/AI/Stats rather than hardcoding NLP-specific categories,
        # since the researcher's interests could be in any field.
        if not categories_include:
            categories_include = ["cs.LG", "stat.ML", "cs.AI"]
            self._log("INFO", "Using broad default categories: cs.LG, stat.ML, cs.AI")
        
        # Merge categories derived from the INTERESTS portion of the prompt.
        # IMPORTANT: We use `self._parsed_prompt.interests_only` which contains
        # ONLY the extracted research interests (e.g. "Multi Armed Bandits, PCA,
        # TSNE, Behavioral Economics") — NOT exclude terms or meta-text.
        prompt_interests_text = getattr(self._parsed_prompt, 'interests_only', '') or ''
        if prompt_interests_text:
            user_categories = map_interests_to_categories(prompt_interests_text)
        else:
            user_categories = []
        if user_categories:
            merged = list(set(categories_include) | set(user_categories))
            self._log("INFO", f"Merged prompt-interest categories {user_categories} with profile categories -> {merged}")
            categories_include = merged
        
        # Map exclude interests if provided
        if not categories_exclude:
            interests_exclude = self._research_profile.get("interests_exclude", "")
            if interests_exclude:
                categories_exclude = map_interests_to_categories(interests_exclude, exclude=True)
                self._log("INFO", f"Mapped exclude interests to categories: {categories_exclude}")
        
        # CRITICAL: Remove any exclude categories that also appear in include.
        # This prevents situations where e.g. "Bandits" -> cs.LG (include) and
        # "Transformers" -> cs.LG (exclude) would incorrectly filter out wanted papers.
        if categories_exclude and categories_include:
            conflict = set(categories_include) & set(categories_exclude)
            if conflict:
                categories_exclude = [c for c in categories_exclude if c not in conflict]
                self._log("WARN", f"Removed conflicting categories from exclude (also in include): {conflict}")
        
        # Build a keyword query from research topics for more targeted arXiv search.
        # Prefer the clean interests_only text from prompt parsing, which guarantees
        # that only the user's actual research interests are used in the query
        # (no exclude terms, no meta-text).
        research_topics = self._research_profile.get("research_topics", [])
        topic_query = None
        
        # Primary: use prompt interests (clean, exclude-free)
        prompt_interests = getattr(self._parsed_prompt, 'interests_only', '') if self._parsed_prompt else ''
        if prompt_interests:
            # Split comma-separated interests into individual query terms
            interest_terms = [t.strip() for t in prompt_interests.split(',') if t.strip()]
            if interest_terms:
                topic_query = " OR ".join(interest_terms)
                self._log("INFO", f"Using interest keywords for arXiv query: {topic_query}")
        
        # Fallback to research_topics from profile
        if not topic_query and research_topics:
            topic_query = " OR ".join(research_topics)
            self._log("INFO", f"Using profile topics for arXiv query: {topic_query}")
        
        # Determine how many papers to request from the arXiv API.
        # Request a larger pool (3x the output count) so that after
        # filtering out already-delivered papers, enough fresh candidates
        # remain for the user's requested output count.
        desired_output = (
            self._parsed_prompt.arxiv_fetch_count
            if self._parsed_prompt
            else DEFAULT_ARXIV_FETCH_COUNT
        )
        candidate_pool_size = min(desired_output * 3, 100)

        fetch_result, success, error = self._invoke_tool(
            "fetch_arxiv_papers",
            categories_include=categories_include,
            categories_exclude=categories_exclude,
            query=topic_query,
            max_results=candidate_pool_size,
        )
        
        if not success or not fetch_result.get("success"):
            return self._finalize_episode(f"fetch_arxiv_papers failed: {error or fetch_result.get('error')}")
        
        self._fetched_papers = fetch_result.get("papers", [])
        # NOTE: Don't increment papers_checked here - it should track papers actually PROCESSED (scored),
        # not papers fetched. We'll increment it as we process each paper.
        self._log("INFO", f"Fetched {len(self._fetched_papers)} papers from arXiv")
        
        # Step 2: Check which papers are unseen
        self._log("INFO", "Step 2: Checking seen/unseen papers...")
        # NOTE: Don't check stop condition here - we need to run check_seen_papers first
        # to determine if there are new papers before evaluating stop_if_no_new_papers
        
        check_result, success, error = self._invoke_tool(
            "check_seen_papers",
            papers=self._fetched_papers
        )
        
        if not success:
            return self._finalize_episode(f"check_seen_papers failed: {error}")
        
        self._unseen_papers = check_result.get("unseen_papers", [])
        seen_count = check_result.get("summary", {}).get("seen", 0)
        self.stop_controller.set_new_papers_found(len(self._unseen_papers))
        
        self._log("INFO", f"Found {len(self._unseen_papers)} unseen papers, {seen_count} already seen")
        
        # Check stop condition for no new papers
        should_stop, reason = self._check_stop_condition()
        if should_stop:
            return self._finalize_episode(reason)
        
        # Step 3: Process each unseen paper
        self._log("INFO", "Step 3: Processing unseen papers...")
        
        for i, paper in enumerate(self._unseen_papers):
            self._log("INFO", f"Processing paper {i+1}/{len(self._unseen_papers)}: {paper.get('title', '')[:50]}...")
            
            should_stop, reason = self._check_stop_condition()
            if should_stop:
                return self._finalize_episode(reason)
            
            # Pre-filter: skip papers whose title or abstract contains an excluded keyword.
            # Uses word-boundary matching to avoid false positives
            # (e.g., excluding "GAN" should NOT skip "Elegant" or "Organ").
            avoid_topics = self._research_profile.get("avoid_topics", [])
            if avoid_topics:
                paper_text = ((paper.get("title") or "") + " " + (paper.get("abstract") or "")).lower()
                skip = False
                for topic in avoid_topics:
                    # Build a word-boundary regex for the excluded topic
                    pattern = r'\b' + re.escape(topic.lower()) + r'\b'
                    if re.search(pattern, paper_text):
                        self._log("INFO", f"Skipping paper (matches excluded topic '{topic}'): {paper.get('title', '')[:80]}")
                        skip = True
                        break
                if skip:
                    continue
            
            # LLM Relevance Filter: use LLM reasoning to determine if paper
            # is truly relevant and not about an excluded topic.
            # This catches semantic matches that keyword regex misses
            # (e.g., a paper about "Attention mechanisms" when "Attention" is excluded).
            research_topics = self._research_profile.get("research_topics", [])
            try:
                from tools.llm_relevance import evaluate_paper_relevance_with_llm

                user_uuid = self._research_profile.get("user_uuid")
                llm_rel = evaluate_paper_relevance_with_llm(
                    paper=paper,
                    research_topics=research_topics,
                    avoid_topics=avoid_topics,
                    user_id=user_uuid,
                )
                if llm_rel:  # non-empty means feature is enabled
                    if llm_rel.get("is_excluded"):
                        matched = llm_rel.get("excluded_topic_match", "")
                        reason_text = llm_rel.get("reasoning", "")
                        self._log(
                            "INFO",
                            f"LLM excluded paper (topic='{matched}'): "
                            f"{paper.get('title', '')[:80]} — {reason_text}",
                        )
                        continue
                    if not llm_rel.get("is_relevant", True):
                        reason_text = llm_rel.get("reasoning", "")
                        self._log(
                            "INFO",
                            f"LLM judged irrelevant: "
                            f"{paper.get('title', '')[:80]} — {reason_text}",
                        )
                        continue
                    # Store LLM relevance score for later use in heuristic blending
                    paper["_llm_relevance"] = llm_rel
            except Exception as e:
                self._log("DEBUG", f"LLM relevance filter unavailable: {e}")
            
            # 3a: Query RAG for similar papers
            rag_results = None
            if self.stop_controller.metrics.rag_queries < self.config.stop_policy.max_rag_queries:
                rag_result, success, _ = self._invoke_tool(
                    "retrieve_similar_from_pinecone",
                    query_text=f"{paper.get('title', '')} {paper.get('abstract', '')[:500]}",
                    top_k=5
                )
                if success:
                    rag_results = rag_result
                    self.stop_controller.increment_rag_queries(1)
            
            # 3b: Score relevance and importance
            score_result, success, error = self._invoke_tool(
                "score_relevance_and_importance",
                paper={
                    "arxiv_id": paper.get("arxiv_id"),
                    "title": paper.get("title"),
                    "abstract": paper.get("abstract", ""),
                    "categories": paper.get("categories", []),
                    "authors": paper.get("authors", []),
                    "publication_date": paper.get("published"),
                    "link": paper.get("link"),
                },
                research_profile={
                    "research_topics": self._research_profile.get("research_topics", []),
                    "avoid_topics": avoid_topics,
                    "arxiv_categories_include": self._research_profile.get("arxiv_categories_include", []),
                    "arxiv_categories_exclude": self._research_profile.get("arxiv_categories_exclude", []),
                    "preferred_venues": self._research_profile.get("preferred_venues", []),
                },
                rag_results=rag_results if rag_results and rag_results.get("success") else None,
                min_importance_to_act=self.config.stop_policy.min_importance_to_act,
            )
            
            if not success:
                self._log("WARN", f"Scoring failed for {paper.get('arxiv_id')}: {error}")
                continue
            
            # Blend LLM relevance score with heuristic score when available.
            # The LLM score captures semantic understanding that keywords miss;
            # taking a weighted average gives the best of both worlds.
            llm_rel = paper.get("_llm_relevance")
            if llm_rel and "relevance_score" in llm_rel:
                heuristic_score = score_result.get("relevance_score", 0)
                llm_score = llm_rel["relevance_score"]
                # 40% heuristic + 60% LLM — LLM gets higher weight because
                # it understands semantic meaning the heuristics can't capture
                blended = 0.4 * heuristic_score + 0.6 * llm_score
                score_result["relevance_score"] = round(blended, 3)
                score_result["llm_relevance_score"] = llm_score
                score_result["llm_relevance_reasoning"] = llm_rel.get("reasoning", "")
                # Re-derive importance from blended score
                novelty = score_result.get("novelty_score", 0.5)
                if blended >= 0.45 and novelty >= 0.5:
                    score_result["importance"] = "high"
                elif blended >= 0.4 or (blended >= 0.3 and novelty >= 0.6):
                    score_result["importance"] = "medium"
                else:
                    score_result["importance"] = "low"
            
            # ==================================================================
            # HARD RELEVANCE GATE (FUNDAMENTAL CHANGE)
            # ==================================================================
            # Papers must demonstrate POSITIVE relevance to the researcher's
            # stated interests. If a paper has zero topic keyword overlap AND
            # gets a low LLM score (or no LLM), it is DROPPED entirely — not
            # delivered as LOW, but removed from results.
            #
            # This prevents the system from flooding the user with papers that
            # happen to share a broad arXiv category (e.g., cs.LG) but have
            # nothing to do with the user's actual interests.
            # ==================================================================
            HARD_RELEVANCE_THRESHOLD = 0.20
            final_relevance = score_result.get("relevance_score", 0)
            if final_relevance < HARD_RELEVANCE_THRESHOLD:
                self._log("INFO",
                    f"DROPPED paper below relevance gate ({final_relevance:.3f} < {HARD_RELEVANCE_THRESHOLD}): "
                    f"{paper.get('title', '')[:80]}")
                continue
            
            # Increment papers_checked now that we've actually processed (scored) this paper
            self.stop_controller.increment_papers_checked(1)
            
            # Update highest importance
            importance = score_result.get("importance", "low")
            self.stop_controller.update_highest_importance(importance)
            
            scored_paper = {
                **paper,
                "relevance_score": score_result.get("relevance_score", 0),
                "novelty_score": score_result.get("novelty_score", 0),
                "importance": importance,
                "explanation": score_result.get("explanation", ""),
            }
            self._scored_papers.append(scored_paper)
            
            # 3c: Decide delivery actions
            delivery_result, success, error = self._invoke_tool(
                "decide_delivery_action",
                scored_paper=scored_paper,
                delivery_policy=self._delivery_policy,
                colleagues=self._colleagues,
            )
            
            if success:
                # Record actions
                researcher_actions = delivery_result.get("researcher_actions", [])
                colleague_actions = delivery_result.get("colleague_actions", [])
                files_to_write = delivery_result.get("files_to_write", [])
                
                self._actions.extend(researcher_actions)
                self._actions.extend(colleague_actions)
                self._artifacts.extend(files_to_write)
                
                # Save artifact files - to DB when available, otherwise local files
                if files_to_write:
                    if is_db_available():
                        # Save to database
                        db_result = save_artifacts_to_db(files_to_write)
                        if not db_result.get("success"):
                            self._log("WARN", f"Some artifacts failed to save to DB: {db_result.get('errors', [])}")
                    else:
                        # Fallback to local files
                        from tools.decide_delivery import FileToWrite
                        file_objects = [FileToWrite(**f) for f in files_to_write]
                        write_artifact_files(file_objects)
                
                # Record decision
                decision = {
                    "paper_id": paper.get("arxiv_id"),
                    "paper_title": paper.get("title"),
                    "importance": importance,
                    "decision": "saved" if importance in ["high", "medium"] else "logged",
                    "actions": [a.get("action_type") for a in researcher_actions if a.get("action_type") != "log"],
                }
                self._decisions.append(decision)
            
            # Derive agent email/calendar decisions from delivery actions
            agent_email_decision = None
            agent_calendar_decision = None
            if success and researcher_actions:
                action_types = [a.get("action_type") for a in researcher_actions]
                agent_email_decision = "email" in action_types
                agent_calendar_decision = "calendar" in action_types

            # 3d: Persist state
            persist_result, success, error = self._invoke_tool(
                "persist_state",
                paper_decision={
                    "paper_id": paper.get("arxiv_id"),
                    "title": paper.get("title"),
                    "decision": "saved" if importance in ["high", "medium"] else "logged",
                    "importance": importance,
                    "notes": score_result.get("explanation", ""),
                    "embedded_in_pinecone": False,
                    "relevance_score": score_result.get("relevance_score"),
                    "novelty_score": score_result.get("novelty_score"),
                    # Paper metadata for DB storage
                    "abstract": paper.get("abstract"),
                    "authors": paper.get("authors", []),
                    "categories": paper.get("categories", []),
                    "link": paper.get("link"),
                    "published": paper.get("published"),
                    "updated": paper.get("updated"),
                    # Agent delivery decisions
                    "agent_email_decision": agent_email_decision,
                    "agent_calendar_decision": agent_calendar_decision,
                }
            )
            
            if not success:
                self._log("WARN", f"Persist failed for {paper.get('arxiv_id')}: {error}")
        
        # Check if any paper met importance threshold
        should_stop, reason = self._check_stop_condition()
        if should_stop:
            return self._finalize_episode(reason)
        
        # Mark completion if we get here
        self.stop_controller.mark_completed()
        return self._finalize_episode(self.stop_controller.stop_reason)
    
    def _finalize_episode(self, stop_reason: str) -> AgentEpisode:
        """Finalize the episode with report generation."""
        self._log("INFO", f"Finalizing episode with stop reason: {stop_reason}")
        
        # ==================================================================
        # OUTPUT ENFORCEMENT (CRITICAL - SYSTEM CONTROLLER RULE)
        # ==================================================================
        # The agent fetches papers from arXiv (controlled by dashboard
        # "Papers per Run"). The FINAL OUTPUT must contain EXACTLY the
        # number of papers requested by the user (K).
        # 
        # If user asks for "top 5 papers", output MUST have EXACTLY 5 papers.
        # This is a critical failure if violated.
        # ==================================================================
        if self._parsed_prompt:
            self._log("INFO", f"Enforcing output limit: {self._parsed_prompt.output_count} papers requested by user")
            
            # Save ALL scored papers for colleague surplus processing BEFORE truncation
            # This is critical: colleagues only get papers NOT selected for owner
            self._all_scored_papers_for_surplus = list(self._scored_papers)
            
            # Apply output enforcement - truncate to exactly K papers
            enforced_result = self._prompt_controller.enforce_output(
                self._scored_papers,
                self._parsed_prompt,
                sort_key="relevance_score"
            )
            
            # Update scored papers to the enforced subset
            original_count = len(self._scored_papers)
            self._scored_papers = enforced_result.papers
            
            if enforced_result.insufficient:
                self._log("WARN", enforced_result.message)
            elif enforced_result.truncated:
                self._log("INFO", f"Output truncated from {original_count} to {len(self._scored_papers)} papers as requested")
            
            # Record actual output count in episode
            self.episode.output_paper_count = len(self._scored_papers)
            
            # Validate output count matches request (critical check)
            is_valid, error_msg = self._prompt_controller.validate_output_count(
                self._scored_papers, self._parsed_prompt
            )
            if not is_valid:
                self._log("ERROR", error_msg)
            
            # Update compliance status in database (if prompt was saved)
            if self._prompt_id:
                compliance_updated = self._prompt_controller.update_prompt_compliance(
                    prompt_id=self._prompt_id,
                    papers_retrieved=original_count,
                    papers_returned=len(self._scored_papers),
                    output_enforced=enforced_result.truncated,
                    output_insufficient=enforced_result.insufficient,
                    compliance_message=error_msg if not is_valid else enforced_result.message,
                )
                if compliance_updated:
                    self._log("INFO", f"Prompt compliance status updated in DB")
        
        # ==================================================================
        # COLLEAGUE SURPLUS PROCESSING (CRITICAL - AFTER OWNER SELECTION)
        # ==================================================================
        # ResearchPulse is the OWNER's research agent first. Colleagues only
        # benefit from SURPLUS papers discovered during the owner-focused
        # workflow. This processing happens AFTER owner's papers are selected.
        # 
        # Principle: Owner gets their top K papers. Only remaining papers
        # that weren't selected for the owner may be forwarded to colleagues.
        # ==================================================================
        if self._colleagues and hasattr(self, '_all_scored_papers_for_surplus'):
            owner_paper_ids = [p.get("arxiv_id") for p in self._scored_papers]
            self._log("INFO", f"Processing colleague surplus: {len(self._all_scored_papers_for_surplus)} total papers, {len(owner_paper_ids)} for owner")
            
            try:
                surplus_result = process_colleague_surplus(
                    all_scored_papers=self._all_scored_papers_for_surplus,
                    owner_paper_ids=owner_paper_ids,
                    colleagues=self._colleagues,
                    delivery_policy=self._delivery_policy,
                    researcher_name=self._research_profile.get("name", "Researcher"),
                    artifacts_dir="artifacts",
                )
                
                if surplus_result.get("success"):
                    surplus_count = surplus_result.get("surplus_count", 0)
                    total_shares = surplus_result.get("total_shares", 0)
                    self._log("INFO", f"Colleague surplus: {surplus_count} surplus papers, {total_shares} shares queued")
                    
                    # Add colleague actions and artifacts
                    colleague_actions = surplus_result.get("colleague_actions", [])
                    surplus_files = surplus_result.get("files_to_write", [])
                    
                    self._actions.extend(colleague_actions)
                    self._artifacts.extend(surplus_files)
                    
                    # Save surplus artifacts to DB
                    if surplus_files and is_db_available():
                        from tools.decide_delivery import FileToWrite
                        file_objects = [FileToWrite(**f) for f in surplus_files]
                        db_result = save_artifacts_to_db(surplus_files)
                        if not db_result.get("success"):
                            self._log("WARN", f"Some surplus artifacts failed to save: {db_result.get('errors', [])}")
                else:
                    self._log("WARN", f"Colleague surplus processing failed: {surplus_result.get('message', 'Unknown error')}")
            except Exception as e:
                self._log("ERROR", f"Colleague surplus processing error: {str(e)}")
        
        # Generate report
        report_result, success, error = self._invoke_tool(
            "generate_report",
            run_id=self.run_id,
            start_time=self.episode.start_time,
            stop_reason=stop_reason,
            papers=[{
                "arxiv_id": p.get("arxiv_id"),
                "title": p.get("title"),
                "importance": p.get("importance"),
                "relevance_score": p.get("relevance_score"),
                "novelty_score": p.get("novelty_score"),
                "decision": next((d.get("decision") for d in self._decisions if d.get("paper_id") == p.get("arxiv_id")), "logged"),
                "is_unseen": True,
            } for p in self._scored_papers],
            decisions=self._decisions,
            actions=self._actions,
            artifacts=self._artifacts,
            rag_query_count=self.stop_controller.metrics.rag_queries,
            unseen_count=len(self._unseen_papers),
            seen_count=len(self._fetched_papers) - len(self._unseen_papers),
            highest_importance=self.stop_controller.metrics.highest_importance,
        )
        
        # Terminate run
        terminate_result, _, _ = self._invoke_tool(
            "terminate_run",
            run_id=self.run_id,
            stop_reason=stop_reason,
            final_metrics=self.stop_controller.metrics.to_dict(),
            success=True,
        )
        
        # Update episode
        self.episode.end_time = datetime.utcnow().isoformat() + "Z"
        self.episode.stop_reason = stop_reason
        self.episode.papers_processed = self._scored_papers
        self.episode.decisions_made = self._decisions
        self.episode.actions_taken = self._actions
        self.episode.artifacts_generated = self._artifacts
        
        if success and report_result:
            self.episode.final_report = report_result.get("report_json", {})
        
        self._log("INFO", f"Episode completed. Processed {len(self._scored_papers)} papers, made {len(self._decisions)} decisions")
        
        # ==================================================================
        # AUTONOMOUS COMPONENTS (Feature-flagged, advisory-only)
        # ==================================================================
        # These components enhance the run with additional insights but
        # NEVER block or modify core functionality. All are feature-flagged.
        # ==================================================================
        self._run_autonomous_components()
        
        return self.episode
    
    def run(self, user_message: str) -> AgentEpisode:
        """
        Execute a full agent episode.
        
        Args:
            user_message: The user's request/message to process
            
        Returns:
            AgentEpisode with complete run information
        """
        # Store the user message in config for use during workflow
        self.config.initial_prompt = user_message
        
        # Initialize episode
        self.episode = AgentEpisode(
            run_id=self.run_id,
            start_time=datetime.utcnow().isoformat() + "Z",
            user_message=user_message,
        )
        
        self._log("INFO", f"Starting agent episode: {self.run_id}")
        
        try:
            return self._run_workflow(user_message)
        except Exception as e:
            self._log("ERROR", f"Agent episode failed: {str(e)}")
            self.episode.end_time = datetime.utcnow().isoformat() + "Z"
            self.episode.stop_reason = f"error: {str(e)}"
            return self.episode
    
    def _run_single_paper_workflow(self, arxiv_id: str) -> AgentEpisode:
        """Fetch exactly one paper by arXiv ID and persist it."""
        self._log("INFO", f"Single paper lookup: {arxiv_id}")
        from tools.fetch_arxiv import fetch_single_paper
        result = fetch_single_paper(arxiv_id)

        if not result.success or not result.papers:
            return self._finalize_episode(f"Paper {arxiv_id} not found on arXiv")

        paper = result.papers[0].model_dump()
        self._fetched_papers = [paper]
        self._scored_papers = [paper]
        self.stop_controller.increment_papers_checked(1)

        # Persist the paper
        persist_result, success, error = self._invoke_tool(
            "persist_state",
            paper_decision={
                "paper_id": paper.get("arxiv_id"),
                "title": paper.get("title"),
                "decision": "saved",
                "importance": "high",
                "notes": f"Directly requested by arXiv ID {arxiv_id}",
                "embedded_in_pinecone": False,
                "abstract": paper.get("abstract"),
                "authors": paper.get("authors", []),
                "categories": paper.get("categories", []),
                "link": paper.get("link"),
                "published": paper.get("published"),
                "updated": paper.get("updated"),
            },
        )
        if not success:
            self._log("WARN", f"Persist failed for {arxiv_id}: {error}")

        # Generate report for the single paper
        report_result, success, error = self._invoke_tool(
            "generate_report",
            research_profile=self._research_profile,
            papers=self._scored_papers,
            decisions=[],
            actions=[],
        )
        if success:
            self.episode.final_report = report_result.get("report", "")

        self.stop_controller.mark_completed()
        return self._finalize_episode(f"Single paper {arxiv_id} fetched and saved")

    def _run_autonomous_components(self) -> None:
        """
        Run all autonomous components after the main workflow completes.
        
        This method is called at the end of _finalize_episode and includes:
        1. Run Audit Log - Structured logging of run data
        2. LLM Novelty Scoring - Enhanced novelty assessment (if not already done)
        3. Profile Evolution - Advisory suggestions for profile refinement
        4. Live Document - Update living research briefing
        
        **Important guarantees:**
        - ALL calls are wrapped in try/except for graceful degradation
        - ALL features are behind feature flags (default OFF)
        - Failures here NEVER affect the core run results
        - User's episode.papers_processed is NOT modified by these calls
        """
        if not FEATURE_FLAGS_AVAILABLE:
            self._log("DEBUG", "Feature flags not available, skipping autonomous components")
            return
        
        user_id = self._research_profile.get("user_id", "")
        
        # Convert user_id to UUID for feature flag checks
        try:
            from uuid import UUID
            user_uuid = UUID(user_id) if user_id else None
        except (ValueError, TypeError):
            user_uuid = None
        
        # Combine scored papers with remaining fetched papers for analysis.
        # Scored papers have real relevance/novelty scores from the pipeline.
        # Fetched-but-unseen papers that weren't scored (already seen) get defaults
        # since they matched the user's query and are valid for profile analysis.
        scored_ids = set()
        analysis_papers = []
        
        if self._scored_papers:
            for p in self._scored_papers:
                analysis_papers.append(p)
                scored_ids.add(p.get("arxiv_id", ""))
        
        if self._fetched_papers:
            for p in self._fetched_papers:
                if p.get("arxiv_id", "") not in scored_ids:
                    paper = dict(p)  # shallow copy
                    paper.setdefault("relevance_score", 0.7)
                    paper.setdefault("novelty_score", 0.75)
                    analysis_papers.append(paper)
        
        # 1. Run Audit Log
        try:
            if is_feature_enabled("AUDIT_LOG", user_uuid):
                from tools.audit_log import build_audit_log_from_episode, save_audit_log
                
                self._log("DEBUG", "Building run audit log...")
                audit_log = build_audit_log_from_episode(
                    run_id=self.run_id,
                    user_id=user_id,
                    research_profile=self._research_profile,
                    fetched_papers=self._fetched_papers,
                    scored_papers=self._scored_papers,
                    decisions=self._decisions,
                    actions=self._actions,
                    colleagues=self._colleagues if hasattr(self, '_colleagues') else [],
                    stop_reason=self.episode.stop_reason or "",
                )
                
                save_result = save_audit_log(audit_log)
                if save_result.get("success"):
                    self._log("INFO", f"Audit log saved: {save_result.get('audit_log_id', save_result.get('log_id', 'unknown'))}")
                else:
                    self._log("DEBUG", f"Audit log save skipped: {save_result.get('error', 'unknown')}")
        except Exception as e:
            self._log("WARNING", f"Audit log failed (non-critical): {e}")
        
        # 2. LLM Novelty Scoring (enhance scored papers)
        try:
            if is_feature_enabled("LLM_NOVELTY", user_uuid) and analysis_papers:
                from tools.llm_novelty import score_paper_novelty_with_llm
                
                self._log("DEBUG", "Running LLM novelty scoring...")
                enhanced_count = 0
                
                for paper in analysis_papers:
                    # Skip if already has LLM novelty score
                    if paper.get("llm_novelty_score") is not None:
                        continue
                    
                    try:
                        novelty_result = score_paper_novelty_with_llm(
                            paper=paper,
                            similar_papers=[],
                            relevance_score=paper.get("relevance_score"),
                            user_id=user_id,
                        )
                        
                        if novelty_result.get("llm_novelty_score") is not None:
                            paper["llm_novelty_score"] = novelty_result["llm_novelty_score"]
                            paper["llm_novelty_reasoning"] = novelty_result.get("llm_novelty_reasoning", "")
                            paper["novelty_sub_scores"] = novelty_result.get("novelty_sub_scores", {})
                            enhanced_count += 1
                    except Exception:
                        pass  # Individual paper failures don't block others
                
                if enhanced_count > 0:
                    self._log("INFO", f"LLM novelty scoring: enhanced {enhanced_count} papers")
        except Exception as e:
            self._log("WARNING", f"LLM novelty scoring failed (non-critical): {e}")
        
        # 3. Profile Evolution Suggestions (ADVISORY ONLY)
        try:
            if is_feature_enabled("PROFILE_EVOLUTION", user_uuid) and analysis_papers:
                from agent.profile_evolution import analyze_and_suggest_profile_evolution
                
                self._log("DEBUG", "Running profile evolution analysis...")
                evolution_result = analyze_and_suggest_profile_evolution(
                    run_id=self.run_id,
                    user_id=user_id,
                    user_profile=self._research_profile,
                    scored_papers=analysis_papers,
                )
                
                suggestions_count = evolution_result.get("suggestions_count", 0)
                if suggestions_count > 0:
                    self._log("INFO", f"Profile evolution: {suggestions_count} suggestions generated (pending review)")
                elif evolution_result.get("skip_reason"):
                    self._log("DEBUG", f"Profile evolution skipped: {evolution_result.get('skip_reason')}")
        except Exception as e:
            self._log("WARNING", f"Profile evolution failed (non-critical): {e}")
        
        # 4. Live Document Update
        # Use only _scored_papers (papers that passed the scoring pipeline)
        # rather than analysis_papers (which includes all fetched papers with
        # inflated default scores).
        try:
            if is_feature_enabled("LIVE_DOCUMENT", user_uuid) and self._scored_papers:
                from tools.live_document import update_live_document_from_run
                
                self._log("DEBUG", "Updating live document...")
                doc_result = update_live_document_from_run(
                    run_id=self.run_id,
                    user_id=user_id,
                    user_profile=self._research_profile,
                    scored_papers=self._scored_papers,
                )
                
                if doc_result.get("save_result", {}).get("success"):
                    self._log("INFO", f"Live document updated: {doc_result.get('top_papers_count', 0)} top papers")
                elif doc_result.get("error"):
                    self._log("DEBUG", f"Live document update failed: {doc_result.get('error')}")
        except Exception as e:
            self._log("WARNING", f"Live document failed (non-critical): {e}")


# =============================================================================
# Convenience Functions
# =============================================================================

def create_agent(
    run_id: str,
    stop_policy: Optional[StopPolicy] = None,
    log_callback: Optional[Callable] = None,
    verbose: bool = True,
    cancellation_check: Optional[Callable[[], bool]] = None,
) -> ResearchReActAgent:
    """
    Create a configured ReAct agent.
    
    Args:
        run_id: Unique identifier for this run
        stop_policy: Optional custom stop policy
        log_callback: Optional logging callback
        verbose: Enable verbose output
        cancellation_check: Optional callback to check if run was cancelled externally
        
    Returns:
        Configured ResearchReActAgent instance
    """
    config = AgentConfig(
        stop_policy=stop_policy or StopPolicy(),
        verbose=verbose,
    )
    return ResearchReActAgent(
        run_id=run_id,
        config=config,
        log_callback=log_callback,
        cancellation_check=cancellation_check,
    )


def run_agent_episode(
    run_id: str,
    user_message: str,
    stop_policy: Optional[StopPolicy] = None,
    log_callback: Optional[Callable] = None,
    cancellation_check: Optional[Callable[[], bool]] = None,
) -> AgentEpisode:
    """
    Run a complete agent episode.
    
    Convenience function that creates an agent and runs it.
    
    Args:
        run_id: Unique identifier for this run
        user_message: The user's request
        stop_policy: Optional custom stop policy
        log_callback: Optional logging callback
        cancellation_check: Optional callback to check if run was cancelled externally
        
    Returns:
        AgentEpisode with complete run information
    """
    agent = create_agent(
        run_id=run_id,
        stop_policy=stop_policy,
        log_callback=log_callback,
        cancellation_check=cancellation_check,
    )
    return agent.run(user_message)


# =============================================================================
# Self-Check
# =============================================================================

def self_check() -> bool:
    """
    Run self-check tests for the ReAct agent.
    
    Returns:
        True if all checks pass, False otherwise.
    """
    print("=" * 60)
    print("ResearchReActAgent Self-Check")
    print("=" * 60)
    
    all_passed = True
    
    def check(name: str, condition: bool) -> bool:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        return condition

    # Test 1: Create agent
    print("\n1. Create Agent:")
    try:
        agent = create_agent("test-001", verbose=False)
        all_passed &= check("agent created", agent is not None)
        all_passed &= check("has run_id", agent.run_id == "test-001")
        all_passed &= check("has tool_registry", agent.tool_registry is not None)
        all_passed &= check("has stop_controller", agent.stop_controller is not None)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 2: Tool registry
    print("\n2. Tool Registry:")
    try:
        registry = ToolRegistry()
        tools = registry.list_tools()
        all_passed &= check("has fetch_arxiv_papers", "fetch_arxiv_papers" in tools)
        all_passed &= check("has check_seen_papers", "check_seen_papers" in tools)
        all_passed &= check("has retrieve_similar_from_pinecone", "retrieve_similar_from_pinecone" in tools)
        all_passed &= check("has score_relevance_and_importance", "score_relevance_and_importance" in tools)
        all_passed &= check("has decide_delivery_action", "decide_delivery_action" in tools)
        all_passed &= check("has persist_state", "persist_state" in tools)
        all_passed &= check("has generate_report", "generate_report" in tools)
        all_passed &= check("has terminate_run", "terminate_run" in tools)
        all_passed &= check("total 8 tools", len(tools) == 8)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 3: Agent config
    print("\n3. Agent Config:")
    try:
        config = AgentConfig()
        all_passed &= check("default max_steps = 50", config.max_steps == 50)
        all_passed &= check("default use_mock_arxiv = True", config.use_mock_arxiv is True)
        all_passed &= check("has stop_policy", config.stop_policy is not None)
        all_passed &= check("stop_policy max_runtime = 6", config.stop_policy.max_runtime_minutes == 6)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 4: Run short episode
    print("\n4. Run Short Episode:")
    try:
        # Create agent with strict limits for fast test
        policy = StopPolicy(
            max_papers_checked=5,
            max_rag_queries=3,
            max_runtime_minutes=1,
        )
        episode = run_agent_episode(
            run_id="test-002",
            user_message="Find recent papers on transformers",
            stop_policy=policy,
            log_callback=None,
        )
        all_passed &= check("episode returned", episode is not None)
        all_passed &= check("has run_id", episode.run_id == "test-002")
        all_passed &= check("has start_time", bool(episode.start_time))
        all_passed &= check("has end_time", bool(episode.end_time))
        all_passed &= check("has stop_reason", bool(episode.stop_reason))
        all_passed &= check("has tool_calls", len(episode.tool_calls) > 0)
        all_passed &= check("has final_report", episode.final_report is not None)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 5: Episode structure
    print("\n5. Episode Structure:")
    try:
        episode = AgentEpisode(
            run_id="test-003",
            start_time="2026-01-08T10:00:00Z",
            user_message="Test message"
        )
        all_passed &= check("episode created", episode is not None)
        all_passed &= check("steps list empty", len(episode.steps) == 0)
        all_passed &= check("tool_calls list empty", len(episode.tool_calls) == 0)
        all_passed &= check("papers_processed empty", len(episode.papers_processed) == 0)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 6: Tool call record
    print("\n6. Tool Call Record:")
    try:
        call = ToolCall(
            tool_name="test_tool",
            input_args={"arg1": "value1"},
            output={"result": "success"},
            timestamp="2026-01-08T10:00:00Z",
            duration_ms=150.5,
            success=True
        )
        all_passed &= check("tool call created", call is not None)
        all_passed &= check("has tool_name", call.tool_name == "test_tool")
        all_passed &= check("has duration_ms", call.duration_ms == 150.5)
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
