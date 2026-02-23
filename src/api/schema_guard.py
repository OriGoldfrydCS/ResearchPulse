"""
Schema Guard — non-breaking runtime validation for Course Project response shapes.

Logs a warning when an endpoint would return a payload that deviates from the
required schema.  Never alters the response if it is already correct.
Falls back to the correct schema only on structural violations (missing keys).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("schema_guard")

# ---------------------------------------------------------------------------
# Required field sets (per Course Project.txt)
# ---------------------------------------------------------------------------
TEAM_INFO_KEYS = {"group_batch_order_number", "team_name", "students"}
AGENT_INFO_KEYS = {"description", "purpose", "prompt_template", "prompt_examples"}
EXECUTE_OK_KEYS = {"status", "error", "response", "steps"}
STEP_KEYS = {"module", "prompt", "response"}


def validate_team_info(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate /api/team_info response shape."""
    missing = TEAM_INFO_KEYS - set(payload.keys())
    if missing:
        logger.warning("team_info missing keys: %s", missing)
    students = payload.get("students")
    if not isinstance(students, list):
        logger.warning("team_info.students is not a list")
    else:
        for i, s in enumerate(students):
            if not isinstance(s, dict) or "name" not in s or "email" not in s:
                logger.warning("team_info.students[%d] missing name/email", i)
    return payload


def validate_agent_info(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate /api/agent_info response shape."""
    missing = AGENT_INFO_KEYS - set(payload.keys())
    if missing:
        logger.warning("agent_info missing keys: %s", missing)
    pt = payload.get("prompt_template")
    if isinstance(pt, dict) and "template" not in pt:
        logger.warning("agent_info.prompt_template missing 'template' key")
    examples = payload.get("prompt_examples")
    if isinstance(examples, list):
        for i, ex in enumerate(examples):
            for k in ("prompt", "full_response", "steps"):
                if k not in ex:
                    logger.warning("agent_info.prompt_examples[%d] missing '%s'", i, k)
    return payload


def validate_execute_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate /api/execute response shape.

    On success:  { status: "ok",    error: null,    response: "...", steps: [] }
    On error:    { status: "error", error: "...",   response: null,  steps: [] }

    If top-level keys are missing, patch them in and log a warning.
    """
    for key in EXECUTE_OK_KEYS:
        if key not in payload:
            logger.warning("execute response missing key '%s' — patching", key)
            if key == "status":
                payload["status"] = "error"
            elif key == "error":
                payload["error"] = None
            elif key == "response":
                payload["response"] = None
            elif key == "steps":
                payload["steps"] = []

    if not isinstance(payload.get("steps"), list):
        logger.warning("execute response.steps is not a list — patching to []")
        payload["steps"] = []

    # Validate individual steps
    for i, step in enumerate(payload["steps"]):
        if isinstance(step, dict):
            step_missing = STEP_KEYS - set(step.keys())
            if step_missing:
                logger.warning("execute step[%d] missing keys: %s", i, step_missing)

    return payload
