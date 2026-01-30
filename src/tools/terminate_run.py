"""
Tool: terminate_run - Set run status to done and record stop reason.

This tool terminates an episodic agent run, marking it as complete and
recording the reason for termination. It integrates with the RunManager
and StopController to properly close out a run.

**Termination Flow:**
1. Update run status to "done" in RunManager
2. Record the stop_reason from StopController
3. Optionally trigger report generation
4. Return termination result with final metrics

**Stop Reasons (from StopController):**
- max_runtime_minutes exceeded
- max_papers_checked reached  
- stop_if_no_new_papers: no unseen papers detected
- max_rag_queries reached
- no paper exceeds min_importance_to_act
- agent emitted TERMINATE action
- run completed successfully
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

# Add parent to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Input/Output Models
# =============================================================================

class TerminateRunInput(BaseModel):
    """Input for terminate_run tool."""
    run_id: str = Field(..., description="Unique run identifier")
    stop_reason: str = Field(..., description="Reason for termination")
    final_metrics: Optional[Dict[str, Any]] = Field(
        None, description="Final run metrics to record"
    )
    success: bool = Field(True, description="Whether the run completed successfully")
    error_message: Optional[str] = Field(None, description="Error message if run failed")


class TerminateRunResult(BaseModel):
    """Result of terminate_run tool."""
    success: bool = Field(..., description="Whether termination succeeded")
    run_id: str = Field(..., description="Run identifier")
    status: Literal["done", "error"] = Field(..., description="Final run status")
    stop_reason: str = Field(..., description="Recorded stop reason")
    end_time: str = Field(..., description="Run end timestamp (ISO format)")
    final_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Final metrics snapshot"
    )
    message: str = Field("", description="Human-readable status message")


class RunTerminator:
    """
    Helper class for terminating runs with proper state management.
    
    This class coordinates termination across the RunManager and StopController,
    ensuring all state is properly recorded.
    """
    
    def __init__(self, run_id: str):
        """Initialize terminator for a specific run."""
        self.run_id = run_id
        self.end_time = datetime.utcnow().isoformat() + "Z"
        self._terminated = False
        self._result: Optional[TerminateRunResult] = None
    
    @property
    def is_terminated(self) -> bool:
        """Check if this run has been terminated."""
        return self._terminated
    
    @property
    def result(self) -> Optional[TerminateRunResult]:
        """Get the termination result if terminated."""
        return self._result
    
    def terminate(
        self,
        stop_reason: str,
        final_metrics: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> TerminateRunResult:
        """
        Terminate the run with the given reason.
        
        Args:
            stop_reason: Why the run is stopping
            final_metrics: Final metrics to record
            success: Whether this is a successful completion
            error_message: Error message if not successful
            
        Returns:
            TerminateRunResult with termination details
        """
        if self._terminated:
            return self._result
        
        status: Literal["done", "error"] = "done" if success else "error"
        
        # Build message
        if error_message:
            message = f"Run {self.run_id} terminated with error: {error_message}"
        elif success:
            message = f"Run {self.run_id} completed successfully: {stop_reason}"
        else:
            message = f"Run {self.run_id} terminated: {stop_reason}"
        
        self._result = TerminateRunResult(
            success=True,  # termination itself succeeded
            run_id=self.run_id,
            status=status,
            stop_reason=stop_reason,
            end_time=self.end_time,
            final_metrics=final_metrics or {},
            message=message,
        )
        
        self._terminated = True
        return self._result


# =============================================================================
# Helper Functions
# =============================================================================

def _get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


def _build_final_metrics(
    metrics: Optional[Dict[str, Any]] = None,
    papers_checked: int = 0,
    rag_queries: int = 0,
    new_papers_found: int = 0,
    highest_importance: Optional[str] = None,
    decisions_made: int = 0,
    actions_taken: int = 0,
    artifacts_generated: int = 0,
) -> Dict[str, Any]:
    """
    Build a complete final metrics dictionary.
    
    Merges provided metrics with defaults.
    """
    default_metrics = {
        "papers_checked": papers_checked,
        "rag_queries": rag_queries,
        "new_papers_found": new_papers_found,
        "highest_importance": highest_importance,
        "decisions_made": decisions_made,
        "actions_taken": actions_taken,
        "artifacts_generated": artifacts_generated,
    }
    
    if metrics:
        default_metrics.update(metrics)
    
    return default_metrics


# =============================================================================
# Main Tool Implementation
# =============================================================================

def terminate_run(
    run_id: str,
    stop_reason: str,
    final_metrics: Optional[Dict[str, Any]] = None,
    success: bool = True,
    error_message: Optional[str] = None,
    update_run_manager: bool = True,
) -> TerminateRunResult:
    """
    Terminate an episodic agent run and record the stop reason.
    
    This tool should be called when the StopController determines the run
    should end, or when the agent explicitly decides to terminate.
    
    Args:
        run_id: Unique identifier for the run to terminate
        
        stop_reason: Human-readable reason for termination. Common values:
            - "max_runtime_minutes exceeded"
            - "max_papers_checked reached"
            - "stop_if_no_new_papers: no unseen papers detected"
            - "max_rag_queries reached"
            - "no paper exceeds min_importance_to_act"
            - "agent emitted TERMINATE action"
            - "run completed successfully"
            
        final_metrics: Dictionary of final run metrics. Can include:
            - papers_checked: Number of papers processed
            - rag_queries: Number of RAG queries made
            - new_papers_found: Number of unseen papers found
            - highest_importance: Highest importance level found
            - decisions_made: Number of decisions made
            - actions_taken: Number of actions executed
            - artifacts_generated: Number of artifact files created
            - elapsed_minutes: Total run duration
            
        success: Whether this is a successful completion (True) or
            an error/early termination (False). Default: True
            
        error_message: Error message if success=False. Will be included
            in the result message and logged.
            
        update_run_manager: Whether to update the RunManager state.
            Set to False for testing without RunManager. Default: True
            
    Returns:
        TerminateRunResult with:
        - success: Whether termination operation succeeded
        - run_id: The run identifier
        - status: Final status ("done" or "error")
        - stop_reason: The recorded stop reason
        - end_time: ISO timestamp when run ended
        - final_metrics: Snapshot of final metrics
        - message: Human-readable termination message
        
    Example:
        >>> result = terminate_run(
        ...     run_id="abc-123",
        ...     stop_reason="max_papers_checked reached",
        ...     final_metrics={
        ...         "papers_checked": 30,
        ...         "rag_queries": 45,
        ...         "new_papers_found": 12,
        ...         "highest_importance": "high",
        ...     },
        ... )
        >>> print(result.status)  # "done"
        >>> print(result.stop_reason)  # "max_papers_checked reached"
        
    Notes:
        - This tool should be called ONCE per run
        - Call generate_report AFTER terminate_run for the final report
        - The stop_reason will be included in the final report
    """
    end_time = _get_timestamp()
    
    # Determine final status
    status: Literal["done", "error"] = "done" if success else "error"
    
    # Build final metrics
    metrics = final_metrics or {}
    if "end_time" not in metrics:
        metrics["end_time"] = end_time
    
    # Build message
    if error_message:
        message = f"Run {run_id} terminated with error: {error_message}"
    elif success:
        message = f"Run {run_id} completed: {stop_reason}"
    else:
        message = f"Run {run_id} terminated: {stop_reason}"
    
    # Update RunManager if available and requested
    if update_run_manager:
        try:
            from api.run_manager import run_manager, RunStatus
            
            run_state = run_manager.get_run(run_id)
            if run_state:
                # Update status
                if success:
                    run_manager.update_status(run_id, RunStatus.DONE)
                else:
                    run_manager.set_error(run_id, error_message or stop_reason)
                
                # Log termination
                run_manager.add_log(
                    run_id, 
                    "INFO" if success else "ERROR",
                    f"Run terminated: {stop_reason}"
                )
                
                # Store final metrics in report
                if run_state.report is None:
                    run_state.report = {}
                run_state.report["stop_reason"] = stop_reason
                run_state.report["final_metrics"] = metrics
                run_state.report["end_time"] = end_time
                run_state.report["status"] = status
                
        except ImportError:
            # RunManager not available, skip update
            pass
        except Exception as e:
            # Log but don't fail termination
            message += f" (RunManager update warning: {str(e)})"
    
    return TerminateRunResult(
        success=True,  # termination operation itself succeeded
        run_id=run_id,
        status=status,
        stop_reason=stop_reason,
        end_time=end_time,
        final_metrics=metrics,
        message=message,
    )


def terminate_run_from_controller(
    run_id: str,
    stop_controller: Any,  # Type: StopController, but avoid import for flexibility
    additional_metrics: Optional[Dict[str, Any]] = None,
) -> TerminateRunResult:
    """
    Terminate a run using state from a StopController instance.
    
    This convenience function extracts the stop reason and metrics
    from a StopController, useful when termination is triggered by
    stop condition checks.
    
    Args:
        run_id: Run identifier
        stop_controller: StopController instance with termination state
        additional_metrics: Extra metrics to include
        
    Returns:
        TerminateRunResult with termination details
    """
    # Extract from controller
    stop_reason = stop_controller.stop_reason
    metrics_dict = stop_controller.metrics.to_dict()
    
    # Merge additional metrics
    if additional_metrics:
        metrics_dict.update(additional_metrics)
    
    return terminate_run(
        run_id=run_id,
        stop_reason=stop_reason,
        final_metrics=metrics_dict,
        success=True,  # Controlled termination is successful
    )


def terminate_run_json(
    run_id: str,
    stop_reason: str,
    final_metrics: Optional[Dict[str, Any]] = None,
    success: bool = True,
    error_message: Optional[str] = None,
) -> dict:
    """JSON-serializable version of terminate_run."""
    result = terminate_run(
        run_id=run_id,
        stop_reason=stop_reason,
        final_metrics=final_metrics,
        success=success,
        error_message=error_message,
        update_run_manager=False,  # Don't update for JSON-only calls
    )
    return result.model_dump()


# =============================================================================
# LangChain Tool Definition (for ReAct agent)
# =============================================================================

TERMINATE_RUN_DESCRIPTION = """
Terminate the current episodic agent run and record the stop reason.

Use this tool when:
- StopController indicates a stop condition is met
- Agent decides to explicitly terminate (TERMINATE action)
- All papers have been processed
- An error requires early termination

Input:
- run_id (required): Unique run identifier
- stop_reason (required): Why the run is stopping. Common values:
  - "max_runtime_minutes exceeded"
  - "max_papers_checked reached"
  - "stop_if_no_new_papers: no unseen papers detected"
  - "max_rag_queries reached"
  - "no paper exceeds min_importance_to_act"
  - "agent emitted TERMINATE action"
  - "run completed successfully"
- final_metrics: Dictionary with final counts (papers_checked, rag_queries, etc.)
- success: True for normal completion, False for error (default: True)
- error_message: Error details if success=False

Output:
- success: Whether termination succeeded
- run_id: The run identifier
- status: "done" or "error"
- stop_reason: Recorded stop reason
- end_time: ISO timestamp when run ended
- final_metrics: Final metrics snapshot
- message: Human-readable status

After calling terminate_run, call generate_report to create the final run report.
"""

TERMINATE_RUN_SCHEMA = {
    "name": "terminate_run",
    "description": TERMINATE_RUN_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "Unique run identifier"
            },
            "stop_reason": {
                "type": "string",
                "description": "Human-readable reason for termination"
            },
            "final_metrics": {
                "type": "object",
                "description": "Final run metrics",
                "properties": {
                    "papers_checked": {"type": "integer"},
                    "rag_queries": {"type": "integer"},
                    "new_papers_found": {"type": "integer"},
                    "highest_importance": {
                        "type": "string",
                        "enum": ["high", "medium", "low"]
                    },
                    "decisions_made": {"type": "integer"},
                    "actions_taken": {"type": "integer"},
                    "artifacts_generated": {"type": "integer"},
                    "elapsed_minutes": {"type": "number"}
                }
            },
            "success": {
                "type": "boolean",
                "description": "Whether run completed successfully",
                "default": True
            },
            "error_message": {
                "type": "string",
                "description": "Error message if success=False"
            }
        },
        "required": ["run_id", "stop_reason"]
    }
}


# =============================================================================
# Self-Check
# =============================================================================

def self_check() -> bool:
    """
    Run self-check tests for terminate_run tool.
    
    Returns:
        True if all checks pass, False otherwise.
    """
    print("=" * 60)
    print("terminate_run Self-Check")
    print("=" * 60)
    
    all_passed = True
    
    def check(name: str, condition: bool) -> bool:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        return condition

    # Test 1: Basic successful termination
    print("\n1. Basic Successful Termination:")
    try:
        result = terminate_run(
            run_id="test-run-001",
            stop_reason="run completed successfully",
            update_run_manager=False,
        )
        all_passed &= check("returns TerminateRunResult", isinstance(result, TerminateRunResult))
        all_passed &= check("success is True", result.success)
        all_passed &= check("status is done", result.status == "done")
        all_passed &= check("run_id matches", result.run_id == "test-run-001")
        all_passed &= check("stop_reason matches", result.stop_reason == "run completed successfully")
        all_passed &= check("end_time is set", len(result.end_time) > 0)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 2: Termination with metrics
    print("\n2. Termination with Metrics:")
    try:
        metrics = {
            "papers_checked": 25,
            "rag_queries": 40,
            "new_papers_found": 10,
            "highest_importance": "high",
            "decisions_made": 10,
            "actions_taken": 15,
        }
        result = terminate_run(
            run_id="test-run-002",
            stop_reason="max_papers_checked reached",
            final_metrics=metrics,
            update_run_manager=False,
        )
        all_passed &= check("success is True", result.success)
        all_passed &= check("final_metrics has papers_checked", result.final_metrics.get("papers_checked") == 25)
        all_passed &= check("final_metrics has rag_queries", result.final_metrics.get("rag_queries") == 40)
        all_passed &= check("final_metrics has highest_importance", result.final_metrics.get("highest_importance") == "high")
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 3: Error termination
    print("\n3. Error Termination:")
    try:
        result = terminate_run(
            run_id="test-run-003",
            stop_reason="API connection failed",
            success=False,
            error_message="Connection timeout to arXiv API",
            update_run_manager=False,
        )
        all_passed &= check("success is True (termination succeeded)", result.success)
        all_passed &= check("status is error", result.status == "error")
        all_passed &= check("message has error info", "error" in result.message.lower())
        all_passed &= check("message has error detail", "Connection timeout" in result.message)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 4: Stop reason: max_runtime_minutes
    print("\n4. Stop Reason - Max Runtime:")
    try:
        result = terminate_run(
            run_id="test-run-004",
            stop_reason="max_runtime_minutes exceeded",
            final_metrics={"elapsed_minutes": 6.5},
            update_run_manager=False,
        )
        all_passed &= check("stop_reason recorded", result.stop_reason == "max_runtime_minutes exceeded")
        all_passed &= check("elapsed_minutes in metrics", result.final_metrics.get("elapsed_minutes") == 6.5)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 5: Stop reason: no new papers
    print("\n5. Stop Reason - No New Papers:")
    try:
        result = terminate_run(
            run_id="test-run-005",
            stop_reason="stop_if_no_new_papers: no unseen papers detected",
            final_metrics={"new_papers_found": 0, "papers_checked": 20},
            update_run_manager=False,
        )
        all_passed &= check("stop_reason recorded", "no unseen papers" in result.stop_reason)
        all_passed &= check("new_papers_found is 0", result.final_metrics.get("new_papers_found") == 0)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 6: Stop reason: no important papers
    print("\n6. Stop Reason - No Important Papers:")
    try:
        result = terminate_run(
            run_id="test-run-006",
            stop_reason="no paper exceeds min_importance_to_act",
            final_metrics={"highest_importance": "low", "papers_checked": 15},
            update_run_manager=False,
        )
        all_passed &= check("stop_reason recorded", "min_importance_to_act" in result.stop_reason)
        all_passed &= check("highest_importance is low", result.final_metrics.get("highest_importance") == "low")
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 7: Stop reason: agent TERMINATE
    print("\n7. Stop Reason - Agent TERMINATE:")
    try:
        result = terminate_run(
            run_id="test-run-007",
            stop_reason="agent emitted TERMINATE action",
            final_metrics={"decisions_made": 5, "actions_taken": 8},
            update_run_manager=False,
        )
        all_passed &= check("stop_reason recorded", "TERMINATE" in result.stop_reason)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 8: JSON output function
    print("\n8. JSON Output Function:")
    try:
        result = terminate_run_json(
            run_id="test-run-008",
            stop_reason="test termination",
        )
        all_passed &= check("returns dict", isinstance(result, dict))
        all_passed &= check("has success", "success" in result)
        all_passed &= check("has status", "status" in result)
        all_passed &= check("has stop_reason", "stop_reason" in result)
        all_passed &= check("has end_time", "end_time" in result)
        all_passed &= check("has final_metrics", "final_metrics" in result)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 9: RunTerminator class
    print("\n9. RunTerminator Class:")
    try:
        terminator = RunTerminator("test-run-009")
        all_passed &= check("initial state not terminated", not terminator.is_terminated)
        all_passed &= check("initial result is None", terminator.result is None)
        
        result = terminator.terminate(
            stop_reason="test termination",
            final_metrics={"test": 123},
        )
        all_passed &= check("after terminate: is_terminated", terminator.is_terminated)
        all_passed &= check("after terminate: result not None", terminator.result is not None)
        all_passed &= check("result matches", result.stop_reason == "test termination")
        
        # Second call should return same result
        result2 = terminator.terminate(stop_reason="should be ignored")
        all_passed &= check("idempotent: same result", result2 == result)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 10: Tool schema
    print("\n10. Tool Schema:")
    all_passed &= check("schema has name", TERMINATE_RUN_SCHEMA["name"] == "terminate_run")
    all_passed &= check("schema has description", len(TERMINATE_RUN_SCHEMA["description"]) > 100)
    all_passed &= check("run_id required", "run_id" in TERMINATE_RUN_SCHEMA["parameters"]["required"])
    all_passed &= check("stop_reason required", "stop_reason" in TERMINATE_RUN_SCHEMA["parameters"]["required"])

    # Test 11: End time format
    print("\n11. End Time Format:")
    try:
        result = terminate_run(
            run_id="test-run-011",
            stop_reason="test",
            update_run_manager=False,
        )
        # Check ISO format with Z suffix
        all_passed &= check("end_time has T separator", "T" in result.end_time)
        all_passed &= check("end_time ends with Z", result.end_time.endswith("Z"))
        all_passed &= check("end_time parseable", len(result.end_time.split("T")) == 2)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 12: Message generation
    print("\n12. Message Generation:")
    try:
        # Success message
        result1 = terminate_run(
            run_id="msg-test-1",
            stop_reason="completed",
            success=True,
            update_run_manager=False,
        )
        all_passed &= check("success message has run_id", "msg-test-1" in result1.message)
        all_passed &= check("success message has reason", "completed" in result1.message)
        
        # Error message
        result2 = terminate_run(
            run_id="msg-test-2",
            stop_reason="failed",
            success=False,
            error_message="Test error",
            update_run_manager=False,
        )
        all_passed &= check("error message has run_id", "msg-test-2" in result2.message)
        all_passed &= check("error message has error", "Test error" in result2.message)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 13: Empty metrics
    print("\n13. Empty Metrics:")
    try:
        result = terminate_run(
            run_id="test-run-013",
            stop_reason="test",
            final_metrics=None,
            update_run_manager=False,
        )
        all_passed &= check("final_metrics is dict", isinstance(result.final_metrics, dict))
        all_passed &= check("end_time added to metrics", "end_time" in result.final_metrics)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 14: Build final metrics helper
    print("\n14. Build Final Metrics Helper:")
    try:
        metrics = _build_final_metrics(
            papers_checked=10,
            rag_queries=20,
            new_papers_found=5,
            highest_importance="medium",
        )
        all_passed &= check("has papers_checked", metrics["papers_checked"] == 10)
        all_passed &= check("has rag_queries", metrics["rag_queries"] == 20)
        all_passed &= check("has new_papers_found", metrics["new_papers_found"] == 5)
        all_passed &= check("has highest_importance", metrics["highest_importance"] == "medium")
        
        # With override
        metrics2 = _build_final_metrics(
            metrics={"papers_checked": 100, "custom": "value"},
            papers_checked=10,
        )
        all_passed &= check("override works", metrics2["papers_checked"] == 100)
        all_passed &= check("custom value preserved", metrics2["custom"] == "value")
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
