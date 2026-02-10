"""
Inbox Scheduler for ResearchPulse.

This module provides background scheduling for inbox polling.
It manages a per-user polling loop based on their configured frequency.

The scheduler:
- Runs as a background asyncio task
- Reads polling settings from the database
- Calls the inbound email processor at configured intervals
- Handles frequency changes dynamically
- Gracefully handles errors without crashing
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional
from uuid import UUID

logger = logging.getLogger(__name__)

# Global scheduler state
_scheduler_task: Optional[asyncio.Task] = None
_scheduler_running: bool = False
_current_frequency: Optional[int] = None
_stop_event: Optional[asyncio.Event] = None


async def _poll_inbox_once(store, user_id: UUID) -> Dict:
    """
    Perform a single inbox poll.
    
    Returns processing results.
    """
    try:
        from .inbound_processor import InboundEmailProcessor
        
        processor = InboundEmailProcessor(store, str(user_id))
        results = processor.process_all(since_hours=48)
        
        return results
    except Exception as e:
        logger.error(f"[SCHEDULER] Error during inbox poll: {e}")
        return {"error": str(e)}


async def _scheduler_loop(store, user_id: UUID, stop_event: asyncio.Event):
    """
    Main scheduler loop that runs continuously.
    
    Checks settings periodically and polls inbox based on configured frequency.
    """
    global _current_frequency
    
    logger.info(f"[SCHEDULER] Starting inbox scheduler for user {user_id}")
    
    last_check = datetime.utcnow()
    check_settings_interval = 30  # Re-check settings every 30 seconds
    tick_count = 0
    
    while not stop_event.is_set():
        try:
            tick_count += 1
            now = datetime.utcnow()
            
            # Get current settings
            settings = store.get_user_settings(user_id)
            
            if not settings:
                logger.debug(f"[SCHEDULER_TICK][#{tick_count}] No settings found - waiting {check_settings_interval}s")
                await asyncio.sleep(check_settings_interval)
                continue
            
            enabled = settings.get("inbox_check_enabled", False)
            frequency = settings.get("inbox_check_frequency_seconds")
            
            if not enabled or not frequency:
                # Polling disabled, check again later
                _current_frequency = None
                logger.debug(f"[SCHEDULER_TICK][#{tick_count}] disabled (enabled={enabled}, freq={frequency}), sleeping {check_settings_interval}s")
                await asyncio.sleep(check_settings_interval)
                continue
            
            _current_frequency = frequency
            
            # Calculate time since last check
            elapsed = (now - last_check).total_seconds()
            time_until_next = max(0, frequency - elapsed)
            
            logger.debug(f"[SCHEDULER_TICK][#{tick_count}] elapsed={elapsed:.1f}s, freq={frequency}s, until_next={time_until_next:.1f}s")
            
            if elapsed >= frequency:
                # Time to poll
                logger.info(f"[SCHEDULER_TICK][#{tick_count}] FIRING poll (elapsed={elapsed:.1f}s >= freq={frequency}s)")
                logger.info(f"[SCHEDULER] timestamp={now.isoformat()}, next_run_in={frequency}s")
                
                results = await _poll_inbox_once(store, user_id)
                
                last_check = datetime.utcnow()
                
                # Log detailed summary
                reschedule_processed = results.get("reschedule_replies", {}).get("processed", 0)
                join_processed = results.get("colleague_joins", {}).get("processed", 0)
                join_succeeded = results.get("colleague_joins", {}).get("succeeded", 0)
                total_scanned = results.get("total_emails_scanned", 0)
                already_processed = results.get("already_processed", 0)
                errors = results.get("errors", [])
                
                logger.info(
                    f"[SCHEDULER] Poll complete: scanned={total_scanned}, "
                    f"reschedule_processed={reschedule_processed}, "
                    f"join_processed={join_processed}, join_succeeded={join_succeeded}, "
                    f"already_processed={already_processed}, errors={len(errors)}"
                )
                
                if errors:
                    for err in errors[:3]:  # Log first 3 errors
                        logger.error(f"[SCHEDULER] Error: {err}")
            
            # Sleep for a short interval before checking again
            sleep_time = min(frequency, check_settings_interval)
            await asyncio.sleep(sleep_time)
            
        except asyncio.CancelledError:
            logger.info("[SCHEDULER] Scheduler cancelled, stopping...")
            break
        except Exception as e:
            logger.error(f"[SCHEDULER] Error in scheduler loop: {e}", exc_info=True)
            await asyncio.sleep(check_settings_interval)
    
    logger.info("[SCHEDULER] Inbox scheduler stopped")


def start_scheduler(store, user_id: UUID) -> bool:
    """
    Start the inbox scheduler as a background task.
    
    Returns True if scheduler was started successfully.
    """
    global _scheduler_task, _scheduler_running, _stop_event
    
    if _scheduler_running and _scheduler_task and not _scheduler_task.done():
        logger.warning("[SCHEDULER] Scheduler already running")
        return False
    
    try:
        _stop_event = asyncio.Event()
        
        # Get or create the event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        _scheduler_task = loop.create_task(_scheduler_loop(store, user_id, _stop_event))
        _scheduler_running = True
        
        logger.info(f"[SCHEDULER] Inbox scheduler started for user {user_id}")
        return True
        
    except Exception as e:
        logger.error(f"[SCHEDULER] Failed to start scheduler: {e}")
        return False


def stop_scheduler():
    """Stop the inbox scheduler."""
    global _scheduler_task, _scheduler_running, _stop_event
    
    if _stop_event:
        _stop_event.set()
    
    if _scheduler_task and not _scheduler_task.done():
        _scheduler_task.cancel()
        logger.info("[SCHEDULER] Scheduler stop requested")
    
    _scheduler_running = False


def update_scheduler(user_id: UUID, enabled: bool, frequency_seconds: Optional[int]):
    """
    Update scheduler settings dynamically.
    
    The scheduler loop will pick up the new settings on its next iteration.
    This function just logs the change - the loop reads from DB.
    """
    logger.info(f"[SCHEDULER] Settings updated: enabled={enabled}, frequency={frequency_seconds}s")
    # The scheduler loop will automatically pick up the new settings
    # from the database on its next iteration


def get_scheduler_status() -> Dict:
    """Get current scheduler status."""
    return {
        "running": _scheduler_running,
        "current_frequency_seconds": _current_frequency,
        "task_done": _scheduler_task.done() if _scheduler_task else True,
    }


# =============================================================================
# FastAPI Lifespan Integration
# =============================================================================

async def startup_inbox_scheduler():
    """
    Start the inbox scheduler during FastAPI startup.
    
    Call this from the FastAPI lifespan or startup event.
    """
    try:
        from ..db.postgres_store import PostgresStore
        from ..db.database import is_database_configured
        
        if not is_database_configured():
            logger.info("[SCHEDULER] Database not configured, skipping scheduler startup")
            return
        
        store = PostgresStore()
        user = store.get_or_create_default_user()
        user_id = UUID(user["id"])
        
        # Check if inbox polling is enabled
        settings = store.get_user_settings(user_id)
        
        if settings and settings.get("inbox_check_enabled") and settings.get("inbox_check_frequency_seconds"):
            start_scheduler(store, user_id)
            logger.info(f"[SCHEDULER] Auto-started inbox scheduler (frequency: {settings['inbox_check_frequency_seconds']}s)")
        else:
            logger.info("[SCHEDULER] Inbox polling not enabled, scheduler not started")
            
    except Exception as e:
        logger.error(f"[SCHEDULER] Error during scheduler startup: {e}")


async def shutdown_inbox_scheduler():
    """
    Stop the inbox scheduler during FastAPI shutdown.
    
    Call this from the FastAPI lifespan or shutdown event.
    """
    stop_scheduler()
    logger.info("[SCHEDULER] Inbox scheduler shutdown complete")
