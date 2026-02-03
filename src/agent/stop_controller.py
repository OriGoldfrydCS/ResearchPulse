"""
Stop Controller - Enforces bounded episodic execution for the ReAct agent.

The Stop Controller is checked before each tool call and after each observation
to ensure the agent terminates within defined limits.

Stop conditions (from SPEC.md section 2):
- max_runtime_minutes exceeded
- max_papers_checked reached
- stop_if_no_new_papers is true AND no unseen papers were detected
- max_rag_queries reached
- No paper exceeds min_importance_to_act
- ReAct agent emits TERMINATE action
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Literal, Optional


class StopReason(str, Enum):
    """Enumeration of all possible stop reasons."""
    MAX_RUNTIME_EXCEEDED = "max_runtime_minutes exceeded"
    MAX_PAPERS_CHECKED = "max_papers_checked reached"
    NO_NEW_PAPERS = "stop_if_no_new_papers: no unseen papers detected"
    MAX_RAG_QUERIES = "max_rag_queries reached"
    NO_IMPORTANT_PAPERS = "no paper exceeds min_importance_to_act"
    AGENT_TERMINATE = "agent emitted TERMINATE action"
    COMPLETED_SUCCESSFULLY = "run completed successfully"
    NOT_STOPPED = ""


@dataclass
class StopPolicy:
    """
    Configuration for stop conditions.
    
    Default values match SPEC.md section 2 demo defaults.
    NOTE: max_papers_checked set to 5 for faster testing (original was 30).
    """
    max_runtime_minutes: int = 6
    max_papers_checked: int = 7
    stop_if_no_new_papers: bool = True
    max_rag_queries: int = 50
    min_importance_to_act: Literal["high", "medium", "low"] = "medium"

    @classmethod
    def from_dict(cls, data: dict) -> StopPolicy:
        """Create StopPolicy from a dictionary (e.g., from JSON config)."""
        return cls(
            max_runtime_minutes=data.get("max_runtime_minutes", 6),
            max_papers_checked=data.get("max_papers_checked", 30),
            stop_if_no_new_papers=data.get("stop_if_no_new_papers", True),
            max_rag_queries=data.get("max_rag_queries", 50),
            min_importance_to_act=data.get("min_importance_to_act", "medium"),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "max_runtime_minutes": self.max_runtime_minutes,
            "max_papers_checked": self.max_papers_checked,
            "stop_if_no_new_papers": self.stop_if_no_new_papers,
            "max_rag_queries": self.max_rag_queries,
            "min_importance_to_act": self.min_importance_to_act,
        }


@dataclass
class RunMetrics:
    """
    Tracks runtime metrics for stop condition evaluation.
    
    Updated by tools during execution and checked by StopController.
    """
    start_time: datetime = field(default_factory=datetime.utcnow)
    papers_checked: int = 0
    rag_queries: int = 0
    new_papers_found: int = 0
    highest_importance: Optional[Literal["high", "medium", "low"]] = None
    agent_requested_terminate: bool = False

    def elapsed_minutes(self) -> float:
        """Calculate elapsed time since run started."""
        delta = datetime.utcnow() - self.start_time
        return delta.total_seconds() / 60.0

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/reporting."""
        return {
            "start_time": self.start_time.isoformat() + "Z",
            "elapsed_minutes": round(self.elapsed_minutes(), 2),
            "papers_checked": self.papers_checked,
            "rag_queries": self.rag_queries,
            "new_papers_found": self.new_papers_found,
            "highest_importance": self.highest_importance,
            "agent_requested_terminate": self.agent_requested_terminate,
        }


# Importance level ordering for comparison
IMPORTANCE_ORDER = {"low": 1, "medium": 2, "high": 3}


class StopController:
    """
    Controls when the ReAct agent should stop execution.
    
    Call should_stop() before each tool invocation and after each observation.
    The controller evaluates all stop conditions and returns the appropriate
    stop reason if any condition is met.
    
    Usage:
        controller = StopController(policy, metrics)
        
        # Before/after each tool call:
        should_stop, reason = controller.should_stop()
        if should_stop:
            return terminate_with_reason(reason)
            
        # Update metrics as agent runs:
        controller.metrics.papers_checked += 1
        controller.metrics.rag_queries += 1
    """

    def __init__(
        self,
        policy: Optional[StopPolicy] = None,
        metrics: Optional[RunMetrics] = None
    ):
        """
        Initialize the stop controller.
        
        Args:
            policy: Stop policy configuration. Uses defaults if not provided.
            metrics: Runtime metrics tracker. Creates new one if not provided.
        """
        self.policy = policy or StopPolicy()
        self.metrics = metrics or RunMetrics()
        self._stop_reason: StopReason = StopReason.NOT_STOPPED
        self._stopped: bool = False

    @property
    def stop_reason(self) -> str:
        """Get the stop reason as a string."""
        return self._stop_reason.value

    @property
    def is_stopped(self) -> bool:
        """Check if the controller has determined a stop."""
        return self._stopped

    def should_stop(self) -> tuple[bool, str]:
        """
        Evaluate all stop conditions.
        
        This method should be called before each tool call and after
        each observation in the ReAct loop.
        
        Returns:
            Tuple of (should_stop: bool, stop_reason: str)
        """
        # Already stopped - return cached reason
        if self._stopped:
            return True, self._stop_reason.value

        # Check each condition in order of precedence

        # 1. Agent explicitly requested termination
        if self.metrics.agent_requested_terminate:
            self._stopped = True
            self._stop_reason = StopReason.AGENT_TERMINATE
            return True, self._stop_reason.value

        # 2. Max runtime exceeded
        if self.metrics.elapsed_minutes() >= self.policy.max_runtime_minutes:
            self._stopped = True
            self._stop_reason = StopReason.MAX_RUNTIME_EXCEEDED
            return True, self._stop_reason.value

        # 3. Max papers checked
        if self.metrics.papers_checked >= self.policy.max_papers_checked:
            self._stopped = True
            self._stop_reason = StopReason.MAX_PAPERS_CHECKED
            return True, self._stop_reason.value

        # 4. No new papers (only check if papers have been fetched)
        if (
            self.policy.stop_if_no_new_papers
            and self.metrics.papers_checked > 0
            and self.metrics.new_papers_found == 0
        ):
            self._stopped = True
            self._stop_reason = StopReason.NO_NEW_PAPERS
            return True, self._stop_reason.value

        # 5. Max RAG queries
        if self.metrics.rag_queries >= self.policy.max_rag_queries:
            self._stopped = True
            self._stop_reason = StopReason.MAX_RAG_QUERIES
            return True, self._stop_reason.value

        # 6. No paper exceeds minimum importance threshold
        # Only check AFTER all papers have been processed (papers_checked >= new_papers_found)
        # This prevents stopping early before giving all papers a chance
        if (
            self.metrics.papers_checked > 0
            and self.metrics.new_papers_found > 0
            and self.metrics.papers_checked >= self.metrics.new_papers_found
            and self.metrics.highest_importance is not None
        ):
            min_required = IMPORTANCE_ORDER[self.policy.min_importance_to_act]
            highest_found = IMPORTANCE_ORDER.get(self.metrics.highest_importance, 0)
            if highest_found < min_required:
                self._stopped = True
                self._stop_reason = StopReason.NO_IMPORTANT_PAPERS
                return True, self._stop_reason.value

        # No stop condition met
        return False, ""

    def mark_terminate(self) -> None:
        """Mark that the agent has explicitly requested termination."""
        self.metrics.agent_requested_terminate = True

    def mark_completed(self) -> None:
        """Mark the run as successfully completed."""
        self._stopped = True
        self._stop_reason = StopReason.COMPLETED_SUCCESSFULLY

    def increment_papers_checked(self, count: int = 1) -> None:
        """Increment the papers checked counter."""
        self.metrics.papers_checked += count

    def increment_rag_queries(self, count: int = 1) -> None:
        """Increment the RAG queries counter."""
        self.metrics.rag_queries += count

    def set_new_papers_found(self, count: int) -> None:
        """Set the number of new (unseen) papers found."""
        self.metrics.new_papers_found = count

    def update_highest_importance(
        self, importance: Literal["high", "medium", "low"]
    ) -> None:
        """Update the highest importance level seen."""
        if self.metrics.highest_importance is None:
            self.metrics.highest_importance = importance
        else:
            current = IMPORTANCE_ORDER[self.metrics.highest_importance]
            new = IMPORTANCE_ORDER[importance]
            if new > current:
                self.metrics.highest_importance = importance

    def get_status_report(self) -> dict:
        """
        Generate a status report of the stop controller state.
        
        Returns:
            Dictionary with policy, metrics, and stop status.
        """
        return {
            "policy": self.policy.to_dict(),
            "metrics": self.metrics.to_dict(),
            "is_stopped": self._stopped,
            "stop_reason": self._stop_reason.value,
        }

    def reset(self) -> None:
        """Reset the controller for a new run."""
        self.metrics = RunMetrics()
        self._stop_reason = StopReason.NOT_STOPPED
        self._stopped = False


# =============================================================================
# Self-Check / Verification
# =============================================================================

def self_check() -> bool:
    """
    Run self-check tests for the StopController.
    
    Returns:
        True if all checks pass, False otherwise.
    """
    print("=" * 60)
    print("StopController Self-Check")
    print("=" * 60)
    
    all_passed = True
    
    def check(name: str, condition: bool) -> bool:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        return condition

    # Test 1: Default policy values
    print("\n1. Default Policy Values:")
    policy = StopPolicy()
    all_passed &= check("max_runtime_minutes = 6", policy.max_runtime_minutes == 6)
    all_passed &= check("max_papers_checked = 5", policy.max_papers_checked == 5)
    all_passed &= check("stop_if_no_new_papers = True", policy.stop_if_no_new_papers is True)
    all_passed &= check("max_rag_queries = 50", policy.max_rag_queries == 50)
    all_passed &= check("min_importance_to_act = 'medium'", policy.min_importance_to_act == "medium")

    # Test 2: No stop initially
    print("\n2. Initial State (no stop):")
    controller = StopController()
    should_stop, reason = controller.should_stop()
    all_passed &= check("should_stop() returns False initially", should_stop is False)
    all_passed &= check("stop_reason is empty", reason == "")

    # Test 3: Max papers checked
    print("\n3. Max Papers Checked Stop:")
    controller = StopController(StopPolicy(max_papers_checked=5))
    controller.metrics.papers_checked = 5
    controller.metrics.new_papers_found = 3  # Has new papers, so won't stop for that
    should_stop, reason = controller.should_stop()
    all_passed &= check("stops when papers_checked >= max", should_stop is True)
    all_passed &= check("reason is MAX_PAPERS_CHECKED", "max_papers_checked" in reason)

    # Test 4: No new papers stop
    print("\n4. No New Papers Stop:")
    controller = StopController(StopPolicy(stop_if_no_new_papers=True))
    controller.metrics.papers_checked = 10
    controller.metrics.new_papers_found = 0
    should_stop, reason = controller.should_stop()
    all_passed &= check("stops when no new papers found", should_stop is True)
    all_passed &= check("reason is NO_NEW_PAPERS", "no unseen papers" in reason)

    # Test 5: Max RAG queries
    print("\n5. Max RAG Queries Stop:")
    controller = StopController(StopPolicy(max_rag_queries=10))
    controller.metrics.rag_queries = 10
    should_stop, reason = controller.should_stop()
    all_passed &= check("stops when rag_queries >= max", should_stop is True)
    all_passed &= check("reason is MAX_RAG_QUERIES", "max_rag_queries" in reason)

    # Test 6: Importance threshold
    print("\n6. Importance Threshold Stop:")
    controller = StopController(StopPolicy(min_importance_to_act="medium"))
    controller.metrics.papers_checked = 5
    controller.metrics.new_papers_found = 5
    controller.metrics.highest_importance = "low"
    should_stop, reason = controller.should_stop()
    all_passed &= check("stops when highest_importance < min_required", should_stop is True)
    all_passed &= check("reason is NO_IMPORTANT_PAPERS", "min_importance_to_act" in reason)

    # Test 7: Agent terminate
    print("\n7. Agent Terminate Stop:")
    controller = StopController()
    controller.mark_terminate()
    should_stop, reason = controller.should_stop()
    all_passed &= check("stops when agent requests terminate", should_stop is True)
    all_passed &= check("reason is AGENT_TERMINATE", "TERMINATE" in reason)

    # Test 8: Policy from dict
    print("\n8. Policy from Dict:")
    policy_dict = {
        "max_runtime_minutes": 10,
        "max_papers_checked": 50,
        "stop_if_no_new_papers": False,
        "max_rag_queries": 100,
        "min_importance_to_act": "high"
    }
    policy = StopPolicy.from_dict(policy_dict)
    all_passed &= check("from_dict loads max_runtime_minutes", policy.max_runtime_minutes == 10)
    all_passed &= check("from_dict loads min_importance_to_act", policy.min_importance_to_act == "high")

    # Test 9: Continues when conditions not met
    print("\n9. Continues When OK:")
    controller = StopController(StopPolicy(max_papers_checked=30))
    controller.metrics.papers_checked = 10
    controller.metrics.new_papers_found = 5
    controller.metrics.highest_importance = "high"
    should_stop, reason = controller.should_stop()
    all_passed &= check("does not stop when all conditions OK", should_stop is False)

    # Test 10: Status report
    print("\n10. Status Report:")
    controller = StopController()
    controller.metrics.papers_checked = 15
    controller.metrics.rag_queries = 25
    report = controller.get_status_report()
    all_passed &= check("report contains policy", "policy" in report)
    all_passed &= check("report contains metrics", "metrics" in report)
    all_passed &= check("metrics.papers_checked = 15", report["metrics"]["papers_checked"] == 15)

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("All checks PASSED!")
    else:
        print("Some checks FAILED!")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    # Run self-check when executed directly
    import sys
    success = self_check()
    sys.exit(0 if success else 1)
