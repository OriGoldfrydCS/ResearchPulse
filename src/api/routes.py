"""
FastAPI Routes - API endpoints for the ResearchPulse agent.

Endpoints:
- GET /api/team_info: Team information
- GET /api/agent_info: Agent configuration and capabilities
- GET /api/model_architecture: System architecture diagram
- POST /api/execute: Execute agent (main endpoint)
- GET /api/status: Poll for run status and logs
- GET /api/health: Health check
"""

from __future__ import annotations

import sys
import asyncio
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .run_manager import RunManager, RunStatus, run_manager
from .schema_guard import validate_team_info, validate_agent_info, validate_execute_response

# Add src to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.stop_controller import StopController, StopPolicy, RunMetrics
from agent.react_agent import ResearchReActAgent, AgentConfig, run_agent_episode
from agent.scope_gate import classify_user_request, ScopeClass

router = APIRouter()


# =============================================================================
# Team & Agent Configuration (per Course Project requirements - EXACT FORMAT)
# =============================================================================

# GET /api/team_info - EXACT format from Course Project
TEAM_INFO = {
    "group_batch_order_number": "2_6",
    "team_name": "Ori_Gaya_Ron",
    "students": [
        {
            "name": "Ori Goldfryd",
            "email": "origoldmsc@gmail.com"
        },
        {
            "name": "Gaya Brodsky",
            "email": "gayabaron@gmail.com"
        },
        {
            "name": "Ron Libman",
            "email": "libman0207@gmail.com"
        }
    ]
}

# GET /api/agent_info - EXACT format from Course Project
AGENT_INFO = {
    "description": (
        "ResearchPulse is a fully autonomous AI research agent that continuously monitors "
        "the arXiv research repository and acts independently on behalf of researchers, "
        "requiring no runtime user interaction. The agent autonomously reasons, plans, and "
        "executes multi-step workflows end-to-end without human intervention. It is equipped "
        "with 28 specialized tools, inter alia, arXiv paper fetching, LLM-based relevance "
        "and novelty scoring, email summarization, calendar invite generation (.ics), colleague "
        "sharing, paper summarization, vector-based retrieval (RAG), and persistent state "
        "management. The agent exhibits three core dimensions: "
        "(1) Perception: the agent autonomously observes its environment by fetching newly "
        "published papers from arXiv across configurable categories and keywords, querying its "
        "long-term vector-store memory (Pinecone) for semantic similarity to previously encountered "
        "work, reading the researcher's evolving profile and preferences from Supabase, and "
        "detecting inbound email replies to adapt its behavior accordingly; "
        "(2) Reasoning: the agent employs a ReAct (Reasoning + Acting) loop powered by a large "
        "language model to score each paper on relevance, novelty, and impact, classify papers "
        "into importance tiers (high, medium, low), determine which delivery actions to execute, "
        "and autonomously decide when to stop processing based on configurable stop conditions; "
        "(3) Action: the agent independently generates and sends email summaries, creates calendar "
        "reminders (.ics) for high-priority papers, updates personalized reading lists, shares "
        "discoveries with colleagues, produces concise paper summaries, rescores papers when the "
        "researcher's profile evolves, and persists all decisions and artifacts to the database. "
        "The agent operates as a bounded episodic system: each run is triggered by a user prompt, "
        "the agent then autonomously executes its full pipeline through multiple reasoning-action "
        "cycles, and terminates when a configurable stop condition is met."
    ),
    "purpose": (
        "Researchers, academics, and students struggle with information overload when trying to "
        "keep up with new work. In fast-moving fields that rely on preprints, the daily stream of "
        "papers creates a paradox of choice: more items to consider makes it harder to decide what "
        "truly deserves attention. ResearchPulse addresses this pain point as an autonomous "
        "ReAct-based research agent that helps researchers deal with information overload by "
        "continuously collecting, ranking, and selecting newly published academic papers. The agent "
        "runs an ongoing reasoning loop to judge how useful each paper is based on relevance, "
        "novelty, and impact, and independently decides when to alert the user, create short "
        "summaries, or schedule dedicated time for reading. By autonomously fetching papers from "
        "arXiv, scoring them against the researcher's profile and existing knowledge (via "
        "RAG/Pinecone), and organizing high-importance papers through email summaries, calendar "
        "reminders, reading lists, and colleague sharing, ResearchPulse eliminates the manual "
        "effort of literature monitoring and ensures researchers never miss critical work in "
        "their field."
    ),
    "prompt_template": {
        "template": "Find recent research papers related to the following research interests: {topic}. Exclude the following topics if applicable: {topics_to_exclude}. Focus on papers published within the last {time_period}.",
        "all_templates": [
            {"name": "Template 1: Topic + Time", "text": "Provide recent research papers on <TOPIC> published within the last <TIME_PERIOD>."},
            {"name": "Template 2: Topic Only", "text": "Provide the most recent research papers on <TOPIC>."},
            {"name": "Template 3: Time Only", "text": "Provide recent research papers published within <TIME_RANGE>."},
            {"name": "Template 4: Key Papers", "text": "List key and influential research papers that help understand the field of <TOPIC>."},
            {"name": "Template 5: Emerging Trends", "text": "Identify emerging research trends based on recent papers on <TOPIC>."},
        ]
    },
    "prompt_examples": []  # loaded from data/prompt_examples.json below
}

# Load real prompt examples from JSON (generated by running all 9 prompts against the live agent)
_PROMPT_EXAMPLES_PATH = Path(__file__).parent.parent.parent / "data" / "prompt_examples.json"
if _PROMPT_EXAMPLES_PATH.exists():
    import json as _json
    with open(_PROMPT_EXAMPLES_PATH, encoding="utf-8") as _f:
        AGENT_INFO["prompt_examples"] = _json.load(_f)

# Architecture diagram — try multiple paths (Vercel CWD differs from local __file__ resolution)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
ARCHITECTURE_PNG_CANDIDATES = [
    _PROJECT_ROOT / "static" / "public" / "architecture.png",       # local dev
    Path("static") / "public" / "architecture.png",                  # Vercel CWD-relative
    _PROJECT_ROOT / "data" / "architecture.png",                     # alternative
]

# Module name mapping: snake_case (runtime) → PascalCase (architecture/AGENT_INFO)
# Course Project requires consistency across architecture diagram, steps, and descriptions
MODULE_NAME_MAP = {
    "fetch_arxiv_papers": "FetchArxivPapers",
    "check_seen_papers": "CheckSeenPapers",
    "retrieve_similar_from_pinecone": "RetrieveSimilarFromPinecone",
    "score_relevance_and_importance": "ScoreRelevanceAndImportance",
    "decide_delivery_action": "DecideDeliveryAction",
    "persist_state": "PersistState",
    "generate_report": "GenerateReport",
    "terminate_run": "TerminateRun",
}

# Thread pool for running agent in background
_executor = ThreadPoolExecutor(max_workers=2)


# =============================================================================
# Request/Response Models
# =============================================================================

class ExecuteRequest(BaseModel):
    """Request body for POST /api/execute endpoint (per Course Project requirements)."""
    prompt: str = Field(
        ...,
        max_length=2000,
        description="The user's prompt to trigger an agent run",
        examples=["Find recent papers on transformer architectures"]
    )
    run_id: Optional[str] = Field(
        None,
        description="Optional run ID for cancellation tracking. If provided, will be used instead of generating a new one."
    )


class ExecuteStepLog(BaseModel):
    """Step log format for POST /api/execute (per Course Project requirements)."""
    module: str = Field(..., description="Module/tool name that was invoked")
    prompt: Dict[str, Any] = Field(default_factory=dict, description="Input to the module")
    response: Dict[str, Any] = Field(default_factory=dict, description="Response from the module")


class ExecuteResponse(BaseModel):
    """Response body for POST /api/execute endpoint (per Course Project requirements)."""
    status: str = Field(..., description="'ok' on success, 'error' on failure")
    error: Optional[str] = Field(None, description="Error message if status is 'error', null otherwise")
    response: Optional[str] = Field(None, description="The agent's final response if successful")
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="List of execution steps")


class StepLog(BaseModel):
    """ReAct step log format (per Course Project requirements)."""
    step: int = Field(..., description="Step number")
    thought: str = Field(..., description="Agent's reasoning")
    action: str = Field(..., description="Tool name to execute")
    action_input: Dict[str, Any] = Field(default_factory=dict, description="Input to the tool")
    observation: str = Field(..., description="Result from tool execution")


class ChatRequest(BaseModel):
    """Request body for POST /chat endpoint (legacy, use /execute instead)."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's message to trigger an agent run",
        examples=["Find recent papers on transformer architectures"]
    )


class ChatResponse(BaseModel):
    """Response body for POST /chat endpoint (legacy, use /execute instead)."""
    run_id: str = Field(..., description="Unique identifier for this run")
    status: str = Field(..., description="Current status of the run")
    message: str = Field(..., description="Confirmation message")


class LogEntryResponse(BaseModel):
    """A single log entry in the status response."""
    ts: str = Field(..., description="Timestamp in ISO format")
    level: str = Field(..., description="Log level (INFO, WARN, ERROR, DEBUG)")
    msg: str = Field(..., description="Log message")


class StatusResponse(BaseModel):
    """Response body for GET /status endpoint."""
    run_id: str = Field(..., description="Unique identifier for this run")
    status: str = Field(..., description="Current status: running, done, error")
    start_time: str = Field(..., description="Run start time in ISO format")
    message: str = Field(..., description="Original user message")
    logs: list[LogEntryResponse] = Field(default_factory=list, description="Log entries")
    report: Optional[dict] = Field(None, description="Final report when status is done")
    error: Optional[str] = Field(None, description="Error message if status is error")


# =============================================================================
# Agent Execution
# =============================================================================

def _run_agent_sync(run_id: str, message: str) -> None:
    """
    Run the ReAct agent synchronously.
    Called in a background thread to not block the API.
    """
    try:
        # Create log callback that writes to RunManager
        def log_callback(level: str, msg: str, ts: str):
            run_manager.add_log(run_id, level, msg)
        
        # Create cancellation check callback that checks RunManager
        def cancellation_check() -> bool:
            return run_manager.is_cancelled(run_id)
        
        # Configure stop policy per SPEC.md Section 2
        stop_policy = StopPolicy(
            max_runtime_minutes=6,
            max_papers_checked=30,
            stop_if_no_new_papers=True,
            max_rag_queries=50,
            min_importance_to_act="medium",
        )
        
        # Run the agent episode with cancellation support
        episode = run_agent_episode(
            run_id=run_id,
            user_message=message,
            stop_policy=stop_policy,
            log_callback=log_callback,
            cancellation_check=cancellation_check,
        )
        
        # Build final report for RunManager
        report = {
            "run_id": run_id,
            "start_time": episode.start_time,
            "end_time": episode.end_time,
            "stop_reason": episode.stop_reason,
            "papers_processed": len(episode.papers_processed),
            "decisions_made": len(episode.decisions_made),
            "actions_taken": len(episode.actions_taken),
            "artifacts_generated": len(episode.artifacts_generated),
            "tool_calls_count": len(episode.tool_calls),
            "papers": episode.papers_processed,
            "decisions": episode.decisions_made,
            "actions": episode.actions_taken,
            "artifacts": episode.artifacts_generated,
            "final_report": episode.final_report,
        }
        
        run_manager.set_report(run_id, report)
        run_manager.update_status(run_id, RunStatus.DONE)
        run_manager.add_log(run_id, "INFO", f"Agent episode completed: {episode.stop_reason}")
        
    except Exception as e:
        error_msg = str(e)
        run_manager.add_log(run_id, "ERROR", f"Agent failed: {error_msg}")
        run_manager.set_error(run_id, error_msg)


def _run_agent_background(run_id: str, message: str) -> None:
    """
    Schedule agent execution in background thread pool.
    """
    _executor.submit(_run_agent_sync, run_id, message)


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def start_chat(request: ChatRequest) -> ChatResponse:
    """
    Start a new episodic agent run.
    
    Creates a new run with a unique ID, initializes the run state,
    and begins the ReAct agent episode processing the user's message.
    
    The agent runs in the background. Use GET /status?run_id=<id> to poll
    for progress and final results.
    
    Returns the run_id which can be used to poll for status.
    """
    # --- Scope Gate: classify before running the heavy agent pipeline ---
    scope_result = classify_user_request(request.message)

    if scope_result.scope != ScopeClass.IN_SCOPE:
        # Return the predefined response without invoking the agent
        import uuid as _uuid
        return ChatResponse(
            run_id=str(_uuid.uuid4()),
            status="done",
            message=scope_result.response or "",
        )

    # If in-scope but has an early response (e.g. missing topic), still
    # create a run but surface the follow-up immediately
    if scope_result.response:
        import uuid as _uuid
        return ChatResponse(
            run_id=str(_uuid.uuid4()),
            status="done",
            message=scope_result.response,
        )

    # Create a new run
    run_state = run_manager.create_run(request.message)
    run_id = run_state.run_id
    
    # Log initial state
    run_manager.add_log(run_id, "INFO", "Agent run created")
    run_manager.add_log(run_id, "INFO", f"User message: {request.message[:100]}...")
    
    # Start agent in background
    _run_agent_background(run_id, request.message)
    
    return ChatResponse(
        run_id=run_id,
        status=run_state.status.value,
        message="Agent run initiated. Poll GET /status?run_id=<id> for progress."
    )


@router.get("/status", response_model=StatusResponse, tags=["Agent"])
async def get_status(
    run_id: str = Query(..., description="The run ID to check status for")
) -> StatusResponse:
    """
    Get the current status of an agent run.
    
    Poll this endpoint to track the progress of a run started via POST /chat.
    Returns the current status, all log entries, and the final report when complete.
    """
    run_state = run_manager.get_run(run_id)
    
    if run_state is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run not found: {run_id}"
        )
    
    state_dict = run_state.to_dict()
    
    return StatusResponse(
        run_id=state_dict["run_id"],
        status=state_dict["status"],
        start_time=state_dict["start_time"],
        message=state_dict["message"],
        logs=[LogEntryResponse(**log) for log in state_dict["logs"]],
        report=state_dict["report"],
        error=state_dict["error"]
    )


class CancelRequest(BaseModel):
    """Request body for POST /api/cancel endpoint."""
    run_id: str = Field(..., description="The run ID to cancel")
    reason: Optional[str] = Field("User cancelled", description="Reason for cancellation")


class CancelResponse(BaseModel):
    """Response body for POST /api/cancel endpoint."""
    success: bool = Field(..., description="Whether cancellation was successful")
    message: str = Field(..., description="Status message")
    run_id: str = Field(..., description="The run ID that was cancelled")


@router.post("/cancel", response_model=CancelResponse, tags=["Agent"])
async def cancel_run(request: CancelRequest) -> CancelResponse:
    """
    Cancel a running agent execution.
    
    This endpoint allows users to stop an in-progress agent run.
    The agent will stop at the next opportunity and mark the run as cancelled.
    
    Args:
        request: Contains run_id and optional reason for cancellation
        
    Returns:
        Success status and message
    """
    import logging
    logging.warning(f"[DEBUG] cancel_run called with run_id: {request.run_id}")
    
    run_state = run_manager.get_run(request.run_id)
    
    if run_state is None:
        logging.warning(f"[DEBUG] Run not found: {request.run_id}")
        logging.warning(f"[DEBUG] Available runs: {run_manager.list_runs()}")
        raise HTTPException(
            status_code=404,
            detail=f"Run not found: {request.run_id}"
        )
    
    logging.warning(f"[DEBUG] Run found, status: {run_state.status}")
    
    if run_state.status != RunStatus.RUNNING:
        return CancelResponse(
            success=False,
            message=f"Run is not running (status: {run_state.status.value})",
            run_id=request.run_id
        )
    
    success = run_manager.cancel_run(request.run_id, request.reason or "User cancelled")
    logging.warning(f"[DEBUG] cancel_run result: {success}")
    
    if success:
        return CancelResponse(
            success=True,
            message="Run cancelled successfully",
            run_id=request.run_id
        )
    else:
        return CancelResponse(
            success=False,
            message="Failed to cancel run",
            run_id=request.run_id
        )


# Health check is in dashboard_routes.py with comprehensive service checks


# =============================================================================
# Required Endpoints (per Course Project requirements)
# =============================================================================

@router.get("/team_info", tags=["Info"])
async def get_team_info() -> dict:
    """
    Get team information.
    
    Returns the team name and team members with their details.
    Required by Course Project specification.
    """
    return validate_team_info(TEAM_INFO)


@router.get("/agent_info", tags=["Info"])
async def get_agent_info() -> dict:
    """
    Get agent information.
    
    Returns agent name, description, models used, tools available,
    prompts used, and other relevant data.
    Required by Course Project specification.
    """
    return validate_agent_info(AGENT_INFO)


@router.get("/model_architecture", tags=["Info"])
async def get_model_architecture():
    """
    Get model architecture diagram.
    
    Returns the architecture diagram as a PNG image.
    Content-Type: image/png (per Course Project specification).
    """
    from fastapi.responses import Response

    # Try each candidate path (covers local dev + Vercel serverless)
    for candidate in ARCHITECTURE_PNG_CANDIDATES:
        if candidate.exists():
            return Response(
                content=candidate.read_bytes(),
                media_type="image/png",
                headers={"Content-Disposition": "inline; filename=architecture.png"}
            )

    # Fallback: generate on-the-fly if no static file found
    try:
        from tools.architecture_diagram import generate_architecture_png
        png_bytes = generate_architecture_png()
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=architecture.png"}
        )
    except Exception as e:
        tried = [str(p) for p in ARCHITECTURE_PNG_CANDIDATES]
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Architecture PNG not found and fallback failed: {e}",
                "tried_paths": tried,
            }
        )


@router.post("/execute", response_model=ExecuteResponse, tags=["Agent"])
async def execute_agent(request: ExecuteRequest) -> ExecuteResponse:
    """
    Execute the agent with the given prompt.
    
    This is the main endpoint for running the ResearchPulse agent.
    It processes the user's prompt, runs the ReAct agent workflow,
    and returns the agent's output.
    
    Required by Course Project specification.
    
    Input: { "prompt": "user prompt string" }
    Output: { "status": "ok", "error": null, "response": "...", "steps": [...] }
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    import logging

    # --- Validate prompt is not empty (return spec-compliant error JSON) ---
    if not request.prompt or not request.prompt.strip():
        return ExecuteResponse(
            status="error",
            error="Prompt cannot be empty. Please provide a research query.",
            response=None,
            steps=[],
        )

    # --- Scope Gate: classify before running the heavy agent pipeline ---
    scope_result = classify_user_request(request.prompt)

    if scope_result.scope != ScopeClass.IN_SCOPE:
        return ExecuteResponse(
            status="ok",
            error=None,
            response=scope_result.response or "",
            steps=[],
        )

    # If in-scope but has an early response (e.g. missing topic / need arXiv link)
    if scope_result.response:
        return ExecuteResponse(
            status="ok",
            error=None,
            response=scope_result.response,
            steps=[],
        )

    # Use provided run_id or create a new one
    if request.run_id:
        run_id = request.run_id
        run_state = run_manager.create_run_with_id(run_id, request.prompt)
        logging.warning(f"[DEBUG] execute_agent using provided run_id: {run_id}")
    else:
        run_state = run_manager.create_run(request.prompt)
        run_id = run_state.run_id
        logging.warning(f"[DEBUG] execute_agent generated run_id: {run_id}")
    
    # Log initial state
    run_manager.add_log(run_id, "INFO", "Agent execution started")
    run_manager.add_log(run_id, "INFO", f"User prompt: {request.prompt[:100]}...")
    
    # Track execution steps in the required format
    execution_steps: List[Dict[str, Any]] = []
    
    def run_agent_in_thread():
        """Run agent in a separate thread so cancel requests can be processed."""
        # Create log callback that writes to RunManager
        def log_callback(level: str, msg: str, ts: str):
            run_manager.add_log(run_id, level, msg)
        
        # Configure stop policy
        stop_policy = StopPolicy(
            max_runtime_minutes=6,
            max_papers_checked=30,
            stop_if_no_new_papers=True,
            max_rag_queries=50,
            min_importance_to_act="medium",
        )
        
        # Create cancellation check callback that checks RunManager
        def cancellation_check() -> bool:
            return run_manager.is_cancelled(run_id)
        
        # Run the agent episode with cancellation support
        return run_agent_episode(
            run_id=run_id,
            user_message=request.prompt,
            stop_policy=stop_policy,
            log_callback=log_callback,
            cancellation_check=cancellation_check,
        )
    
    try:
        # Run agent in thread pool so event loop stays free for cancel requests
        loop = asyncio.get_event_loop()
        episode = await loop.run_in_executor(_executor, run_agent_in_thread)
        
        # Check if was cancelled
        if run_manager.is_cancelled(run_id):
            return ExecuteResponse(
                status="ok",
                error=None,
                response="⏹ Run cancelled by user.",
                steps=[]
            )
        
        # Build execution steps from tool_calls
        for i, tool_call in enumerate(episode.tool_calls):
            # Map snake_case tool names to PascalCase module names
            # for consistency with architecture diagram and AGENT_INFO (Course Project req)
            raw_name = tool_call.tool_name if tool_call.tool_name else f"Step_{i+1}"
            module_name = MODULE_NAME_MAP.get(raw_name, raw_name)
            execution_steps.append({
                "module": module_name,
                "prompt": tool_call.input_args,
                "response": {"result": tool_call.output, "success": tool_call.success}
            })
        
        # Build the agent response from the episode
        output_parts = []
        output_parts.append(f"ResearchPulse Agent Run Complete")
        output_parts.append(f"=" * 40)
        output_parts.append(f"Run ID: {run_id}")
        output_parts.append(f"Stop Reason: {episode.stop_reason}")
        output_parts.append(f"Papers Processed: {len(episode.papers_processed)}")
        output_parts.append(f"Decisions Made: {len(episode.decisions_made)}")
        output_parts.append(f"Artifacts Generated: {len(episode.artifacts_generated)}")
        output_parts.append("")
        
        # Check if no new papers were found and provide a clear message
        if episode.final_report:
            report = episode.final_report if isinstance(episode.final_report, dict) else {}
            stats = report.get("stats", {}) if isinstance(report, dict) else {}
            seen_count = stats.get("seen_papers_count", 0)
            unseen_count = stats.get("unseen_papers_count", 0)
            filtered_count = stats.get("papers_filtered_count", 0)
            total_fetched = stats.get("total_fetched_count", 0)
            
            if len(episode.papers_processed) == 0 and unseen_count > 0:
                # CRITICAL: unseen papers were found but ALL were filtered by
                # quality/relevance.  Do NOT claim "No New Papers Found".
                output_parts.append("🔍 New Papers Found but Filtered")
                output_parts.append("-" * 40)
                output_parts.append(f"Found {unseen_count} new (unseen) papers from {total_fetched} fetched,")
                output_parts.append(f"but none met the relevance/quality criteria for delivery.")
                if filtered_count:
                    output_parts.append(f"({filtered_count} paper(s) filtered out by quality thresholds.)")
                output_parts.append("")
                output_parts.append("💡 Tip: Adjust your research topics or relevance thresholds in Settings")
                output_parts.append("   to broaden the match criteria.")
                output_parts.append("")
            elif len(episode.papers_processed) == 0 and unseen_count == 0 and seen_count > 0:
                output_parts.append("📭 No New Papers Found")
                output_parts.append("-" * 40)
                output_parts.append(f"All {seen_count} papers from arXiv have already been processed.")
                output_parts.append("This means you're up to date with the latest research!")
                output_parts.append("")
                output_parts.append("💡 Tip: New papers are typically published on arXiv weekdays.")
                output_parts.append("   Try running the agent again later for fresh content.")
                output_parts.append("")
            elif len(episode.papers_processed) == 0 and seen_count == 0 and unseen_count == 0:
                if getattr(episode, 'topic_not_in_categories', False):
                    searched = getattr(episode, 'searched_topic', None) or "the requested topic"
                    output_parts.append("")
                    output_parts.append(
                        f"You asked to search for research papers about '{searched}'.\n\n"
                        "arXiv primarily provides papers in fields such as computer science, "
                        "mathematics, physics, statistics, quantitative biology, quantitative "
                        "finance, and related technical domains.\n\n"
                        "There are currently no relevant arXiv papers for the requested topic, "
                        "and RESEARCHPULSE cannot assist with this topic in its current version.\n\n"
                        "This may be supported in the future — stay tuned 🙂"
                    )
                    output_parts.append("")
                else:
                    output_parts.append("⚠️ No Papers Retrieved")
                    output_parts.append("-" * 40)
                    output_parts.append("Could not fetch papers from arXiv.")
                    output_parts.append("This may be due to network issues or arXiv API limits.")
                    output_parts.append("")
            
            output_parts.append("Summary:")
            output_parts.append(str(episode.final_report))
        
        # The topic_not_in_categories flag with papers > 0 should no longer
        # occur (we return early for out-of-scope topics), but handle it
        # defensively in case the taxonomy mapper found categories via fallback.
        if getattr(episode, 'topic_not_in_categories', False) and len(episode.papers_processed) > 0:
            searched = getattr(episode, 'searched_topic', None) or "the requested topic"
            output_parts.append("")
            output_parts.append(
                f"You asked to search for research papers about '{searched}'.\n\n"
                "arXiv primarily provides papers in fields such as computer science, "
                "mathematics, physics, statistics, quantitative biology, quantitative "
                "finance, and related technical domains.\n\n"
                "There are currently no relevant arXiv papers for the requested topic, "
                "and RESEARCHPULSE cannot assist with this topic in its current version.\n\n"
                "This may be supported in the future — stay tuned 🙂"
            )
            output_parts.append("")

        if episode.artifacts_generated:
            output_parts.append("")
            output_parts.append("Generated Artifacts:")
            for artifact in episode.artifacts_generated:
                output_parts.append(f"  - {artifact}")
        
        agent_response = "\n".join(output_parts)
        
        # Update run manager
        run_manager.set_report(run_id, {
            "run_id": run_id,
            "stop_reason": episode.stop_reason,
            "papers_processed": len(episode.papers_processed),
            "decisions_made": len(episode.decisions_made),
            "artifacts_generated": len(episode.artifacts_generated),
        })
        run_manager.update_status(run_id, RunStatus.DONE)
        
        # Auto-summarize high-priority papers in background
        try:
            from tools.summarize_paper import auto_summarize_high_priority
            from db.postgres_store import PostgresStore
            store = PostgresStore()
            user = store.get_or_create_default_user()
            if user:
                from uuid import UUID as _UUID
                user_id = _UUID(user["id"])
                _executor.submit(auto_summarize_high_priority, user_id)
                import logging as _logging
                _logging.getLogger(__name__).info(
                    f"[AUTO_SUMMARIZE] Queued auto-summarization for high-priority papers"
                )
        except Exception as auto_sum_err:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                f"[AUTO_SUMMARIZE] Failed to queue auto-summarization: {auto_sum_err}"
            )
        
        return ExecuteResponse(
            status="ok",
            error=None,
            response=agent_response,
            steps=execution_steps
        )
        
    except Exception as e:
        error_msg = str(e)
        run_manager.add_log(run_id, "ERROR", f"Agent failed: {error_msg}")
        run_manager.set_error(run_id, error_msg)
        return ExecuteResponse(
            status="error",
            error=error_msg,
            response=None,
            steps=execution_steps
        )


# =============================================================================
# Saved Prompts API - CRUD for Quick Prompt Builder templates
# =============================================================================

class SavedPromptRequest(BaseModel):
    """Request body for saving a prompt template."""
    name: str = Field(..., description="User-given name for the prompt")
    prompt_text: str = Field(..., description="The generated prompt text")
    template_type: Optional[str] = Field(None, description="Type of template")
    areas: Optional[List[str]] = Field(default=[], description="Selected research areas")
    topics: Optional[List[str]] = Field(default=[], description="Focus topics")
    time_period: Optional[str] = Field(None, description="Time period selection")
    paper_count: Optional[int] = Field(None, description="Number of papers requested")


class SavedPromptResponse(BaseModel):
    """Response for saved prompt operations."""
    success: bool
    message: str
    prompt_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


@router.get("/saved-prompts")
async def get_saved_prompts_endpoint() -> JSONResponse:
    """
    Get all saved prompt templates for the current user.
    
    Returns:
        List of saved prompt templates
    """
    try:
        from db.data_service import get_saved_prompts
        prompts = get_saved_prompts(limit=100)
        return JSONResponse(content={
            "success": True,
            "prompts": prompts
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@router.post("/saved-prompts")
async def save_prompt_endpoint(request: SavedPromptRequest) -> SavedPromptResponse:
    """
    Save a new prompt template.
    
    Args:
        request: Prompt template details
        
    Returns:
        Success status and prompt ID
    """
    try:
        from db.data_service import save_prompt_template
        prompt_id = save_prompt_template(
            name=request.name,
            prompt_text=request.prompt_text,
            template_type=request.template_type,
            areas=request.areas or [],
            topics=request.topics or [],
            time_period=request.time_period,
            paper_count=request.paper_count,
        )
        
        if prompt_id:
            return SavedPromptResponse(
                success=True,
                message="Prompt saved successfully",
                prompt_id=prompt_id
            )
        else:
            return SavedPromptResponse(
                success=False,
                message="Failed to save prompt - database may be unavailable"
            )
    except Exception as e:
        return SavedPromptResponse(
            success=False,
            message=f"Error saving prompt: {str(e)}"
        )


@router.get("/saved-prompts/{prompt_id}")
async def get_saved_prompt_endpoint(prompt_id: str) -> JSONResponse:
    """
    Get a specific saved prompt by ID.
    
    Args:
        prompt_id: The prompt UUID
        
    Returns:
        Saved prompt details
    """
    try:
        from db.data_service import get_saved_prompt_by_id
        prompt = get_saved_prompt_by_id(prompt_id)
        
        if prompt:
            return JSONResponse(content={
                "success": True,
                "prompt": prompt
            })
        else:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "Prompt not found"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@router.delete("/saved-prompts/{prompt_id}")
async def delete_saved_prompt_endpoint(prompt_id: str) -> SavedPromptResponse:
    """
    Delete a saved prompt template.
    
    Args:
        prompt_id: The prompt UUID to delete
        
    Returns:
        Success status
    """
    try:
        from db.data_service import delete_saved_prompt
        success = delete_saved_prompt(prompt_id)
        
        if success:
            return SavedPromptResponse(
                success=True,
                message="Prompt deleted successfully"
            )
        else:
            return SavedPromptResponse(
                success=False,
                message="Prompt not found or could not be deleted"
            )
    except Exception as e:
        return SavedPromptResponse(
            success=False,
            message=f"Error deleting prompt: {str(e)}"
        )


# =============================================================================
# Prompt Templates API - CRUD for reusable prompt templates
# =============================================================================

class TemplateRequest(BaseModel):
    """Request body for creating a prompt template."""
    name: str = Field(..., description="Template name")
    text: str = Field(..., description="Template text content")


@router.get("/templates")
async def get_templates_endpoint() -> JSONResponse:
    """
    Get all prompt templates ordered by: builtin first, then custom by name.
    
    Returns:
        List of templates with id, name, text, is_builtin fields
    """
    try:
        from db.data_service import get_prompt_templates
        templates = get_prompt_templates()
        return JSONResponse(content={"success": True, "templates": templates})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@router.post("/templates")
async def create_template_endpoint(request: TemplateRequest) -> JSONResponse:
    """
    Create a new custom prompt template (non-builtin).
    
    Args:
        request: Template name and text
        
    Returns:
        Template ID if successful
    """
    try:
        from db.data_service import create_prompt_template
        template_id = create_prompt_template(request.name, request.text)
        if template_id:
            return JSONResponse(content={"success": True, "id": template_id})
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Failed to create template"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@router.delete("/templates/{template_id}")
async def delete_template_endpoint(template_id: str) -> JSONResponse:
    """
    Delete a custom prompt template (builtin templates cannot be deleted).
    
    Args:
        template_id: UUID of the template to delete
        
    Returns:
        Success status
    """
    try:
        from db.data_service import delete_prompt_template
        success = delete_prompt_template(template_id)
        if success:
            return JSONResponse(content={"success": True})
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Cannot delete builtin template or template not found"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
