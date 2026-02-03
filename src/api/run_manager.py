"""
Run Manager - In-memory state management for agent runs.

Tracks run_id, status, start_time, and logs for each episodic agent execution.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class RunStatus(str, Enum):
    """Possible states for an agent run."""
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class LogEntry:
    """A single log entry for a run."""
    ts: str
    level: str
    msg: str

    def to_dict(self) -> dict:
        return {"ts": self.ts, "level": self.level, "msg": self.msg}


@dataclass
class RunState:
    """State container for a single agent run."""
    run_id: str
    start_time: datetime
    status: RunStatus = RunStatus.RUNNING
    logs: List[LogEntry] = field(default_factory=list)
    message: str = ""
    report: Optional[dict] = None
    error: Optional[str] = None

    def add_log(self, level: str, msg: str) -> None:
        """Add a log entry with current timestamp."""
        entry = LogEntry(
            ts=datetime.utcnow().isoformat() + "Z",
            level=level,
            msg=msg
        )
        self.logs.append(entry)

    def to_dict(self) -> dict:
        """Convert run state to dictionary for API response."""
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() + "Z",
            "message": self.message,
            "logs": [log.to_dict() for log in self.logs],
            "report": self.report,
            "error": self.error,
        }


class RunManager:
    """
    Manages in-memory state for all agent runs.
    
    Thread-safe singleton pattern for storing run states.
    In production, this would be backed by Redis or a database.
    """

    _instance: Optional[RunManager] = None
    _runs: Dict[str, RunState]

    def __new__(cls) -> RunManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._runs = {}
        return cls._instance

    def create_run(self, message: str) -> RunState:
        """
        Create a new run with a unique ID.
        
        Args:
            message: The user's input message that triggered the run.
            
        Returns:
            The newly created RunState.
        """
        run_id = str(uuid.uuid4())
        run_state = RunState(
            run_id=run_id,
            start_time=datetime.utcnow(),
            message=message
        )
        run_state.add_log("INFO", f"Run created with message: {message[:100]}...")
        self._runs[run_id] = run_state
        return run_state

    def get_run(self, run_id: str) -> Optional[RunState]:
        """
        Retrieve a run by its ID.
        
        Args:
            run_id: The unique identifier for the run.
            
        Returns:
            The RunState if found, None otherwise.
        """
        return self._runs.get(run_id)

    def update_status(self, run_id: str, status: RunStatus) -> bool:
        """
        Update the status of a run.
        
        Args:
            run_id: The unique identifier for the run.
            status: The new status to set.
            
        Returns:
            True if the run was found and updated, False otherwise.
        """
        run = self.get_run(run_id)
        if run:
            run.status = status
            run.add_log("INFO", f"Status changed to: {status.value}")
            return True
        return False

    def set_error(self, run_id: str, error: str) -> bool:
        """
        Mark a run as errored with an error message.
        
        Args:
            run_id: The unique identifier for the run.
            error: The error message.
            
        Returns:
            True if the run was found and updated, False otherwise.
        """
        run = self.get_run(run_id)
        if run:
            run.status = RunStatus.ERROR
            run.error = error
            run.add_log("ERROR", error)
            return True
        return False

    def set_report(self, run_id: str, report: dict) -> bool:
        """
        Set the final report for a completed run.
        
        Args:
            run_id: The unique identifier for the run.
            report: The report dictionary.
            
        Returns:
            True if the run was found and updated, False otherwise.
        """
        run = self.get_run(run_id)
        if run:
            run.report = report
            run.add_log("INFO", "Report generated")
            return True
        return False

    def add_log(self, run_id: str, level: str, msg: str) -> bool:
        """
        Add a log entry to a run.
        
        Args:
            run_id: The unique identifier for the run.
            level: Log level (INFO, WARN, ERROR, DEBUG).
            msg: The log message.
            
        Returns:
            True if the run was found and log added, False otherwise.
        """
        run = self.get_run(run_id)
        if run:
            run.add_log(level, msg)
            return True
        return False

    def list_runs(self) -> List[str]:
        """Return all run IDs."""
        return list(self._runs.keys())

    def clear_runs(self) -> None:
        """Clear all runs (useful for testing)."""
        self._runs.clear()

    def cancel_run(self, run_id: str, reason: str = "User cancelled") -> bool:
        """
        Cancel a running run.
        
        Args:
            run_id: The unique identifier for the run.
            reason: The reason for cancellation.
            
        Returns:
            True if the run was found and cancelled, False otherwise.
        """
        run = self.get_run(run_id)
        if run and run.status == RunStatus.RUNNING:
            run.status = RunStatus.CANCELLED
            run.add_log("INFO", f"Run cancelled: {reason}")
            return True
        return False

    def is_cancelled(self, run_id: str) -> bool:
        """
        Check if a run has been cancelled.
        
        Args:
            run_id: The unique identifier for the run.
            
        Returns:
            True if the run is cancelled, False otherwise.
        """
        run = self.get_run(run_id)
        return run is not None and run.status == RunStatus.CANCELLED


# Global instance
run_manager = RunManager()
