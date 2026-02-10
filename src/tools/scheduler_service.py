"""
Scheduler Service - Background scheduler for automatic ResearchPulse runs.

Provides:
- run_researchpulse_search(user_id, trigger_source): Shared core search function
- SchedulerService: Background loop that checks DB settings and triggers runs
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional, Callable
from uuid import UUID

logger = logging.getLogger(__name__)

# Lock to prevent overlapping runs
_run_lock = threading.Lock()
_scheduler_thread: Optional[threading.Thread] = None
_scheduler_stop_event = threading.Event()


def run_researchpulse_search(
    user_id: Optional[str] = None,
    trigger_source: str = "manual",
    prompt: Optional[str] = None,
    log_callback: Optional[Callable] = None,
    cancellation_check: Optional[Callable[[], bool]] = None,
) -> dict:
    """
    Shared core search function used by both manual UI and scheduler.
    
    Args:
        user_id: User UUID string (uses default user if None)
        trigger_source: "manual" or "scheduled"
        prompt: Custom prompt (uses default if None)
        log_callback: Optional log callback for RunManager
        cancellation_check: Optional cancellation check
        
    Returns:
        dict with run results
    """
    import uuid as uuid_mod
    
    logger.info("[RUN] trigger=%s started", trigger_source)
    
    # Prevent overlapping runs
    if not _run_lock.acquire(blocking=False):
        logger.info("[RUN] trigger=%s skipped reason=overlapping_run", trigger_source)
        return {"status": "skipped", "reason": "overlapping_run"}
    
    try:
        from db.store import get_default_store
        from agent.stop_controller import StopPolicy
        from agent.react_agent import run_agent_episode
        from api.run_manager import run_manager, RunStatus
        
        store = get_default_store()
        
        # Resolve user
        if user_id is None:
            user = store.get_or_create_default_user()
            user_id = user["id"]
        
        uid = UUID(str(user_id))
        
        # Build default prompt if not provided
        if not prompt:
            user_data = store.get_user(uid)
            interests = ""
            if user_data:
                interests = user_data.get("interests_include", "") or ""
                topics = user_data.get("research_topics", []) or []
                if topics and not interests:
                    interests = ", ".join(topics)
            
            prompt = f"Find recent papers on {interests}" if interests else "Find recent papers on my research interests"
        
        run_id = str(uuid_mod.uuid4())
        
        # Create run in RunManager
        run_state = run_manager.create_run_with_id(run_id, prompt) if hasattr(run_manager, 'create_run_with_id') else run_manager.create_run(prompt)
        actual_run_id = run_state.run_id if hasattr(run_state, 'run_id') else run_id
        
        run_manager.add_log(actual_run_id, "INFO", f"[RUN] trigger={trigger_source}")
        
        # Create log callback wrapper
        def _log_cb(level: str, msg: str, ts: str):
            run_manager.add_log(actual_run_id, level, msg)
            if log_callback:
                log_callback(level, msg, ts)
        
        stop_policy = StopPolicy(
            max_runtime_minutes=6,
            max_papers_checked=30,
            stop_if_no_new_papers=True,
            max_rag_queries=50,
            min_importance_to_act="medium",
        )
        
        episode = run_agent_episode(
            run_id=actual_run_id,
            user_message=prompt,
            stop_policy=stop_policy,
            log_callback=_log_cb,
            cancellation_check=cancellation_check,
        )
        
        # Record completion in DB for scheduler tracking
        try:
            store.record_run_completed(uid)
        except Exception as e:
            logger.warning("[RUN] Failed to record run completion: %s", e)
        
        run_manager.update_status(actual_run_id, RunStatus.DONE)
        
        logger.info("[RUN] trigger=%s completed run_id=%s papers=%d",
                     trigger_source, actual_run_id, len(episode.papers_processed))
        
        return {
            "status": "completed",
            "run_id": actual_run_id,
            "trigger_source": trigger_source,
            "papers_processed": len(episode.papers_processed),
            "stop_reason": episode.stop_reason,
        }
        
    except Exception as e:
        logger.error("[RUN] trigger=%s failed: %s", trigger_source, e)
        return {"status": "error", "error": str(e), "trigger_source": trigger_source}
    finally:
        _run_lock.release()


class SchedulerService:
    """
    Background scheduler that periodically checks DB settings and triggers runs.
    
    Checks every 60 seconds whether a scheduled run is due.
    Safe and idempotent: uses next_run_at from DB.
    """
    
    CHECK_INTERVAL_SECONDS = 60
    
    def __init__(self, check_interval_seconds: int = 60):
        self.CHECK_INTERVAL_SECONDS = check_interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
    
    @property
    def _running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()
    
    def start(self):
        """Start the scheduler in a background thread."""
        if self._thread and self._thread.is_alive():
            logger.info("[SCHEDULE] Scheduler already running")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="rp-scheduler")
        self._thread.start()
        logger.info("[SCHEDULE] Scheduler started")
    
    def stop(self):
        """Stop the scheduler."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[SCHEDULE] Scheduler stopped")
    
    def _loop(self):
        """Main scheduler loop."""
        while not self._stop_event.is_set():
            try:
                self._check_and_run()
            except Exception as e:
                logger.error("[SCHEDULE] Error in scheduler loop: %s", e)
            
            self._stop_event.wait(timeout=self.CHECK_INTERVAL_SECONDS)
    
    def _check_and_run(self):
        """Check if any scheduled run is due and trigger it."""
        try:
            from db.store import get_default_store
            from db.database import is_database_configured
            
            if not is_database_configured():
                return
            
            store = get_default_store()
            user = store.get_or_create_default_user()
            user_id = user["id"]
            uid = UUID(str(user_id))
            
            settings = store.get_or_create_user_settings(uid)
            mode = settings.get("execution_mode", "manual")
            
            if mode != "scheduled":
                logger.debug("[SCHEDULE] mode=manual => no scheduled run")
                return
            
            freq = settings.get("scheduled_frequency")
            if not freq:
                logger.debug("[SCHEDULE] mode=scheduled but no frequency set")
                return
            
            next_run_at_str = settings.get("next_run_at")
            now = datetime.utcnow()
            
            if next_run_at_str:
                # Parse ISO format
                try:
                    next_run_at = datetime.fromisoformat(next_run_at_str.replace("Z", "+00:00").replace("+00:00", ""))
                except (ValueError, AttributeError):
                    next_run_at = now  # If parse fails, run now
                
                if now < next_run_at:
                    logger.debug("[SCHEDULE] mode=scheduled freq=%s next_run_at=%s (not due yet)",
                                freq, next_run_at_str)
                    return
            
            # Time to run!
            logger.info("[SCHEDULE] mode=scheduled freq=%s triggering run", freq)
            result = run_researchpulse_search(
                user_id=user_id,
                trigger_source="scheduled",
            )
            logger.info("[SCHEDULE] Run result: %s", result.get("status"))
            
        except Exception as e:
            logger.error("[SCHEDULE] Check failed: %s", e)


# Global scheduler instance
_scheduler: Optional[SchedulerService] = None


def get_scheduler() -> SchedulerService:
    """Get or create the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = SchedulerService()
    return _scheduler


def start_scheduler():
    """Start the background scheduler."""
    scheduler = get_scheduler()
    scheduler.start()


def stop_scheduler():
    """Stop the background scheduler."""
    global _scheduler
    if _scheduler:
        _scheduler.stop()
