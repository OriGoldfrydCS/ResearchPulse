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

# Add src to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.stop_controller import StopController, StopPolicy, RunMetrics
from agent.react_agent import ResearchReActAgent, AgentConfig, run_agent_episode

router = APIRouter()


# =============================================================================
# Team & Agent Configuration (per Course Project requirements - EXACT FORMAT)
# =============================================================================

# GET /api/team_info - EXACT format from Course Project
TEAM_INFO = {
    "group_batch_order_number": "1_1",  # Update with your actual batch#_order#
    "team_name": "ResearchPulse Team",
    "students": [
        {
            "name": "Ori Goldfryd",
            "email": "origoldmsc@gmail.com"
        }
        # Add more team members as needed
    ]
}

# GET /api/agent_info - EXACT format from Course Project
AGENT_INFO = {
    "description": "ResearchPulse is an autonomous AI agent designed to help researchers stay up-to-date with scientific literature. It monitors arXiv for new papers, evaluates their relevance using RAG-based novelty detection and heuristic scoring, and takes appropriate actions such as generating email summaries, creating calendar reminders, and sharing discoveries with colleagues.",
    "purpose": "To automate the discovery and filtering of relevant research papers from arXiv, saving researchers time by automatically scoring papers for relevance and novelty, and organizing them through email summaries, calendar reminders, and reading lists.",
    "prompt_template": {
        "template": "Find recent papers on {topic} from arXiv. Filter by relevance to my research interests: {interests}. Check novelty against my existing knowledge. Suggest delivery actions for important papers."
    },
    "prompt_examples": [
        {
            "prompt": "Find recent papers on transformer architectures and large language models",
            "full_response": "ResearchPulse Agent Run Complete\n========================================\nProcessed 15 papers from arXiv\nHigh importance: 3 papers (email + calendar)\nMedium importance: 5 papers (reading list)\nLow importance: 7 papers (logged only)\n\nTop papers found:\n1. 'Efficient Attention Mechanisms for LLMs' - High relevance (0.85)\n2. 'Scaling Laws for Neural Language Models' - High novelty (0.72)\n3. 'Multi-Modal Transformers Survey' - Added to reading list",
            "steps": [
                {
                    "module": "FetchArxivPapers",
                    "prompt": {"action": "fetch_arxiv_papers", "categories": ["cs.CL", "cs.LG"], "keywords": ["transformer", "large language model"], "max_results": 30},
                    "response": {"papers_fetched": 30, "status": "success"}
                },
                {
                    "module": "CheckSeenPapers",
                    "prompt": {"action": "check_seen_papers", "paper_ids": ["2601.05167", "2601.05171", "..."]},
                    "response": {"new_papers": 15, "already_seen": 15}
                },
                {
                    "module": "RetrieveSimilarFromPinecone",
                    "prompt": {"action": "retrieve_similar", "query": "Efficient Attention Mechanisms for LLMs", "top_k": 5},
                    "response": {"matches": 3, "max_similarity": 0.65}
                },
                {
                    "module": "ScoreRelevanceAndImportance",
                    "prompt": {"action": "score_paper", "paper_id": "2601.05167", "title": "Efficient Attention Mechanisms"},
                    "response": {"relevance": 0.85, "novelty": 0.72, "importance": "high"}
                },
                {
                    "module": "DecideDeliveryAction",
                    "prompt": {"action": "decide_action", "importance": "high", "paper_id": "2601.05167"},
                    "response": {"actions": ["email_summary", "calendar_event", "reading_list"]}
                },
                {
                    "module": "GenerateReport",
                    "prompt": {"action": "generate_report"},
                    "response": {"papers_processed": 15, "decisions_made": 15, "artifacts_generated": 8}
                }
            ]
        }
    ]
}

# Architecture diagram path (for SVG endpoint)
ARCHITECTURE_SVG_PATH = Path(__file__).parent.parent.parent / "static" / "public" / "architecture.svg"

# Thread pool for running agent in background
_executor = ThreadPoolExecutor(max_workers=2)


# =============================================================================
# Request/Response Models
# =============================================================================

class ExecuteRequest(BaseModel):
    """Request body for POST /api/execute endpoint (per Course Project requirements)."""
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's prompt to trigger an agent run",
        examples=["Find recent papers on transformer architectures"]
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
        
        # Configure stop policy per SPEC.md Section 2
        stop_policy = StopPolicy(
            max_runtime_minutes=6,
            max_papers_checked=30,
            stop_if_no_new_papers=True,
            max_rag_queries=50,
            min_importance_to_act="medium",
        )
        
        # Run the agent episode
        episode = run_agent_episode(
            run_id=run_id,
            user_message=message,
            stop_policy=stop_policy,
            log_callback=log_callback,
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
        run_manager.update_status(run_id, RunStatus.ERROR, error=error_msg)


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


@router.get("/health", tags=["System"])
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ResearchPulse",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


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
    return TEAM_INFO


@router.get("/agent_info", tags=["Info"])
async def get_agent_info() -> dict:
    """
    Get agent information.
    
    Returns agent name, description, models used, tools available,
    prompts used, and other relevant data.
    Required by Course Project specification.
    """
    return AGENT_INFO


@router.get("/model_architecture", tags=["Info"])
async def get_model_architecture():
    """
    Get model architecture diagram.
    
    Returns the architecture diagram as an image (SVG format).
    Required by Course Project specification.
    """
    from fastapi.responses import FileResponse
    
    if ARCHITECTURE_SVG_PATH.exists():
        return FileResponse(
            path=str(ARCHITECTURE_SVG_PATH),
            media_type="image/svg+xml",
            filename="architecture.svg"
        )
    else:
        # Return a JSON fallback if SVG doesn't exist
        return {
            "error": "Architecture diagram not found",
            "description": "ResearchPulse uses a ReAct (Reasoning + Acting) architecture with FastAPI backend, Pinecone vector store, Supabase database, and LLMod.ai for LLM calls.",
            "expected_path": str(ARCHITECTURE_SVG_PATH)
        }


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
    # Create a new run
    run_state = run_manager.create_run(request.prompt)
    run_id = run_state.run_id
    
    # Log initial state
    run_manager.add_log(run_id, "INFO", "Agent execution started")
    run_manager.add_log(run_id, "INFO", f"User prompt: {request.prompt[:100]}...")
    
    # Track execution steps in the required format
    execution_steps: List[Dict[str, Any]] = []
    
    try:
        # Create log callback that writes to RunManager and tracks steps
        def log_callback(level: str, msg: str, ts: str):
            run_manager.add_log(run_id, level, msg)
            # Track tool calls as steps
            if "Tool:" in msg:
                # Parse tool execution from log message
                execution_steps.append({
                    "module": "LogCapture",
                    "prompt": {"message": msg},
                    "response": {"level": level, "timestamp": ts}
                })
        
        # Configure stop policy
        stop_policy = StopPolicy(
            max_runtime_minutes=6,
            max_papers_checked=30,
            stop_if_no_new_papers=True,
            max_rag_queries=50,
            min_importance_to_act="medium",
        )
        
        # Run the agent episode
        episode = run_agent_episode(
            run_id=run_id,
            user_message=request.prompt,
            stop_policy=stop_policy,
            log_callback=log_callback,
        )
        
        # Build execution steps from tool_calls
        for i, tool_call in enumerate(episode.tool_calls):
            execution_steps.append({
                "module": tool_call.tool_name if tool_call.tool_name else f"Step_{i+1}",
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
            
            if len(episode.papers_processed) == 0 and seen_count > 0:
                output_parts.append("📭 No New Papers Found")
                output_parts.append("-" * 40)
                output_parts.append(f"All {seen_count} papers from arXiv have already been processed.")
                output_parts.append("This means you're up to date with the latest research!")
                output_parts.append("")
                output_parts.append("💡 Tip: New papers are typically published on arXiv weekdays.")
                output_parts.append("   Try running the agent again later for fresh content.")
                output_parts.append("")
            elif len(episode.papers_processed) == 0 and seen_count == 0:
                output_parts.append("⚠️ No Papers Retrieved")
                output_parts.append("-" * 40)
                output_parts.append("Could not fetch papers from arXiv.")
                output_parts.append("This may be due to network issues or arXiv API limits.")
                output_parts.append("")
            
            output_parts.append("Summary:")
            output_parts.append(str(episode.final_report))
        
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
