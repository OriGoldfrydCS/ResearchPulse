"""
Inbound Email Processor for ResearchPulse.

This module handles incoming email processing with two main flows:
1. Owner reminder reschedule - Replies to calendar invite emails
2. Colleague join requests - Requests to join with a join code

Features:
- Scheduled polling based on user settings
- Idempotent processing (tracks processed message IDs)
- Owner email validation for reschedule requests
- Secure join code validation (bcrypt hashed)
- Structured logging for debugging
"""

import os
import re
import logging
import hashlib
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID

from .email_poller import (
    get_imap_config,
    fetch_recent_replies,
    fetch_colleague_signup_emails,
    decode_email_subject,
    get_email_body,
    extract_original_message_id,
    is_calendar_invite_reply,
)

# Import unified outbound email module for consistent sender name
try:
    from .outbound_email import (
        send_outbound_email,
        EmailType,
    )
except ImportError:
    from outbound_email import (
        send_outbound_email,
        EmailType,
    )

logger = logging.getLogger(__name__)

# Configure structured logging for inbound email pipeline
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] %(message)s'
)


# =============================================================================
# Join Code Utilities
# =============================================================================

def hash_join_code(code: str) -> str:
    """Hash a join code using bcrypt for secure storage."""
    # bcrypt expects bytes
    code_bytes = code.encode('utf-8')
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(code_bytes, salt)
    return hashed.decode('utf-8')


def verify_join_code(code: str, code_hash: str) -> bool:
    """Verify a join code against its stored hash."""
    try:
        code_bytes = code.encode('utf-8')
        hash_bytes = code_hash.encode('utf-8')
        result = bcrypt.checkpw(code_bytes, hash_bytes)
        
        # Log hash prefixes for debugging (safe - doesn't reveal code)
        stored_prefix = code_hash[:10] if code_hash else "None"
        logger.info(f"[JOIN_CODE] Verification: stored_hash_prefix={stored_prefix}..., code_length={len(code)}, match={result}")
        
        return result
    except Exception as e:
        logger.error(f"[JOIN_CODE] Error verifying join code: {e}")
        return False


def extract_join_code_from_email(body: str, subject: str) -> Optional[str]:
    """
    Extract a join code from email body or subject.
    
    Looks for patterns like:
    - "Code: ABC123" or "code:ABC123"
    - "code=ABC123"
    - "Join code: ABC123" or "join code ABC123"
    - "My code is ABC123"
    - "#ABC123"
    """
    text = f"{subject}\n{body}".strip()
    
    # Diagnostic: log what we're searching through
    text_preview = text[:200].replace('\n', '\\n') if text else "(empty)"
    logger.info(f"[JOIN_CODE] Searching text (length={len(text)}): {text_preview}")
    
    # Common patterns for join codes (case-insensitive, whitespace tolerant)
    patterns = [
        r'(?:join\s*)?code[:\s=]+([A-Za-z0-9-_]{4,32})',  # code: 123456, code=123456, join code 123456
        r'#([A-Za-z0-9-_]{4,32})',  # #123456
        r'(?:my|the)\s+code\s+(?:is\s+)?([A-Za-z0-9-_]{4,32})',  # my code is 123456
        r'use\s+code[:\s]+([A-Za-z0-9-_]{4,32})',  # use code 123456
    ]
    
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            # Filter out common false positives
            if code.lower() not in ['the', 'is', 'code', 'join', 'my']:
                logger.info(f"[JOIN_CODE] Extracted code (length={len(code)}) using pattern #{i+1}")
                return code
    
    logger.info(f"[JOIN_CODE] No code found in email (text_length={len(text)})")
    return None


def extract_signup_template_fields(body: str, subject: str) -> Dict[str, Optional[str]]:
    """
    Extract name and interests from the structured signup template format.
    
    Expected format (as given in our instruction email):
        Code: ABC123
        Name: Jane Smith
        Research interests: machine learning, NLP, transformers
    
    Returns:
        Dict with 'name' and 'interests' (either can be None if not found)
    """
    text = f"{subject}\n{body}"
    result = {"name": None, "interests": None}
    
    # Extract Name: field
    name_match = re.search(r'name[:\s]+([^\n\r]+)', text, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).strip()
        # Filter out placeholder values
        if name.lower() not in ('your full name', 'your name', 'jane smith', ''):
            result["name"] = name
            logger.info(f"[TEMPLATE_PARSE] Extracted name: {name}")
    
    # Extract Research interests: field (or just interests:)
    interests_match = re.search(r'(?:research\s+)?interests?[:\s]+([^\n\r]+)', text, re.IGNORECASE)
    if interests_match:
        interests = interests_match.group(1).strip()
        # Filter out placeholder values
        if interests.lower() not in ('topic1, topic2, topic3', 'your interests', ''):
            result["interests"] = interests
            logger.info(f"[TEMPLATE_PARSE] Extracted interests: {interests}")
    
    return result


def is_colleague_join_request(subject: str, body: str) -> bool:
    """
    Check if an email is a colleague join request.
    
    Keywords indicating join intent:
    - "add me", "join", "subscribe", "sign up"
    - "research updates", "paper updates"
    """
    text = (subject + " " + body).lower()
    
    join_keywords = [
        "add me",
        "join",
        "subscribe",
        "sign me up",
        "sign up",
        "send me papers",
        "research updates",
        "paper updates",
        "colleague",
        "include me",
        "add to your list",
        "want to receive",
        "interested in receiving",
    ]
    
    researchpulse_keywords = [
        "researchpulse",
        "research pulse",
        "paper recommendation",
    ]
    
    # Must mention ResearchPulse or be very explicit about joining
    mentions_rp = any(kw in text for kw in researchpulse_keywords)
    has_join_intent = any(kw in text for kw in join_keywords)
    
    # Require both indicators for safety
    return mentions_rp and has_join_intent


# =============================================================================
# Reschedule Intent Detection
# =============================================================================

def is_owner_email(from_email: str, owner_email: str) -> bool:
    """Check if the sender is the owner (for trusted reschedule requests)."""
    if not from_email or not owner_email:
        return False
    return from_email.lower().strip() == owner_email.lower().strip()


def extract_reschedule_datetime(body: str) -> Tuple[Optional[datetime], Optional[str]]:
    """
    Extract a requested datetime from reschedule email body.
    
    Strips quoted content first, then uses the reply_parser module.
    """
    try:
        from ..agent.reply_parser import parse_reply, strip_email_quotes
        cleaned_body = strip_email_quotes(body)
        logger.debug(f"[RESCHEDULE_PARSE] cleaned_body_length={len(cleaned_body)} (original={len(body)})")

        result = parse_reply(cleaned_body, use_llm=False)

        if result.extracted_datetime:
            logger.info(
                f"[RESCHEDULE_PARSE] extracted_text=\"{result.extracted_datetime_text}\" "
                f"parsed_dt={result.extracted_datetime.isoformat()}"
            )
            return result.extracted_datetime, result.extracted_datetime_text

        # If cleaned body failed, try the full body (sometimes context matters)
        if cleaned_body != body:
            result = parse_reply(body, use_llm=False)
            if result.extracted_datetime:
                logger.info(
                    f"[RESCHEDULE_PARSE] extracted_text=\"{result.extracted_datetime_text}\" "
                    f"parsed_dt={result.extracted_datetime.isoformat()} (full body fallback)"
                )
                return result.extracted_datetime, result.extracted_datetime_text

        logger.warning(f"[RESCHEDULE_PARSE] No datetime found in body (intent={result.intent.value})")
    except ImportError:
        logger.warning("[RESCHEDULE_PARSE] reply_parser import failed, falling back to basic patterns")
    
    # Fallback: Basic pattern matching
    return None, None


# =============================================================================
# Main Inbound Email Processor
# =============================================================================

class InboundEmailProcessor:
    """
    Processes inbound emails with routing to appropriate handlers.
    
    Supports:
    1. Owner reschedule requests (reply to calendar invites)
    2. Colleague join requests (with join code validation)
    """
    
    def __init__(self, store, user_id: str):
        """
        Initialize the processor.
        
        Args:
            store: Database store instance
            user_id: User ID (owner) for this processor
        """
        self.store = store
        self.user_id = UUID(user_id) if isinstance(user_id, str) else user_id
        self._owner_email = None
        self._join_code_hash = None
    
    @property
    def owner_email(self) -> Optional[str]:
        """Get the owner's email address."""
        if self._owner_email is None:
            user = self.store.get_user(self.user_id)
            self._owner_email = user.get("email") if user else None
        return self._owner_email
    
    @property
    def join_code_hash(self) -> Optional[str]:
        """Get the stored join code hash."""
        if self._join_code_hash is None:
            self._join_code_hash = self.store.get_colleague_join_code_hash(self.user_id)
        return self._join_code_hash
    
    def process_all(self, since_hours: int = 48) -> Dict[str, Any]:
        """
        Process all inbound emails.
        
        Returns:
            Summary of processing results
        """
        import time
        
        logger.info(f"[INBOUND] Starting inbound email processing for user {self.user_id}")
        
        results = {
            "reschedule_replies": {"processed": 0, "succeeded": 0, "failed": 0, "skipped": 0},
            "colleague_joins": {"processed": 0, "succeeded": 0, "failed": 0, "rejected": 0},
            "total_emails_scanned": 0,
            "already_processed": 0,
            "errors": [],
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Process reschedule replies
        try:
            reschedule_results = self._process_reschedule_replies(since_hours)
            results["reschedule_replies"] = reschedule_results
            results["total_emails_scanned"] += reschedule_results.get("scanned", 0)
            results["already_processed"] += reschedule_results.get("already_processed", 0)
        except Exception as e:
            error_msg = f"Error processing reschedule replies: {str(e)}"
            logger.error(f"[INBOUND] {error_msg}")
            results["errors"].append(error_msg)
        
        # Add a small delay between IMAP connections to avoid Gmail rate limiting
        # Gmail may throttle rapid consecutive IMAP connections from the same IP
        logger.debug("[INBOUND] Waiting 2s before colleague join fetch to avoid Gmail rate limiting...")
        time.sleep(2)
        
        # Process colleague join requests
        try:
            join_results = self._process_colleague_joins(since_hours)
            results["colleague_joins"] = join_results
            results["total_emails_scanned"] += join_results.get("scanned", 0)
            results["already_processed"] += join_results.get("already_processed", 0)
        except Exception as e:
            error_msg = f"Error processing colleague joins: {str(e)}"
            logger.error(f"[INBOUND] {error_msg}")
            results["errors"].append(error_msg)
        
        # Update last check timestamp
        try:
            self.store.update_last_inbox_check(self.user_id)
        except Exception as e:
            logger.warning(f"[INBOUND] Failed to update last inbox check timestamp: {e}")
        
        logger.info(f"[INBOUND] Processing complete: {results}")
        return results
    
    def _process_reschedule_replies(self, since_hours: int) -> Dict[str, Any]:
        """Process calendar invite reply emails for reschedule requests."""
        logger.info(f"[INBOUND] Fetching reschedule replies (last {since_hours}h)")
        
        results = {
            "scanned": 0,
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "skipped": 0,
            "already_processed": 0,
        }
        
        # Fetch recent replies
        replies = fetch_recent_replies(since_hours=since_hours)
        results["scanned"] = len(replies)
        
        logger.info(f"[INBOUND] Found {len(replies)} potential reschedule replies")
        
        for reply in replies:
            message_id = reply.get("message_id", "")
            from_email = reply.get("from_email", "")
            subject = reply.get("subject", "")
            
            # Check idempotency
            if self.store.is_email_processed(self.user_id, message_id):
                logger.debug(f"[INBOUND] Skipping already processed: {message_id[:20]}...")
                results["already_processed"] += 1
                continue
            
            results["processed"] += 1
            
            # Validate sender is owner
            if not is_owner_email(from_email, self.owner_email):
                logger.warning(f"[INBOUND] Reschedule request from non-owner: {from_email}")
                self.store.mark_email_processed(
                    user_id=self.user_id,
                    gmail_message_id=message_id,
                    email_type="reschedule_reply",
                    processing_result="rejected_not_owner",
                    from_email=from_email,
                    subject=subject,
                )
                results["skipped"] += 1
                continue
            
            # Process the reschedule
            try:
                success = self._handle_reschedule_reply(reply)
                if success:
                    results["succeeded"] += 1
                    self.store.mark_email_processed(
                        user_id=self.user_id,
                        gmail_message_id=message_id,
                        email_type="reschedule_reply",
                        processing_result="success",
                        from_email=from_email,
                        subject=subject,
                    )
                else:
                    results["failed"] += 1
                    # Mark as match_failed so we don't keep retrying indefinitely
                    self.store.mark_email_processed(
                        user_id=self.user_id,
                        gmail_message_id=message_id,
                        email_type="reschedule_reply",
                        processing_result="match_failed",
                        from_email=from_email,
                        subject=subject,
                    )
            except Exception as e:
                logger.error(f"[INBOUND] Error handling reschedule reply: {e}", exc_info=True)
                results["failed"] += 1
                # Do NOT mark as processed — allows retry on next poll cycle
        
        return results
    
    def _extract_reminder_token(self, body: str) -> Optional[str]:
        """Extract RP_REMINDER_ID token from email body.
        
        Looks for pattern: [RP_REMINDER_ID: <token>]
        Token is typically a UUID or truncated UUID (letters, numbers, hyphens).
        """
        import re
        # Match alphanumeric tokens with hyphens (UUIDs, truncated UUIDs)
        pattern = r'\[RP_REMINDER_ID:\s*([a-zA-Z0-9\-]+)\]'
        match = re.search(pattern, body, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
    
    def _find_invite_with_fallback(
        self, 
        reply: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Find the original calendar invite using multi-step matching.
        
        Priority:
        1. Extract RP_REMINDER_ID token from body (most reliable)
        2. Match by Gmail thread_id (if available)
        3. Match by In-Reply-To header (with format variations)
        4. Fuzzy match: subject + time proximity (last resort)
        
        Returns:
            Tuple of (invite_dict or None, match_method: str)
        """
        body = reply.get("body_text", "")
        in_reply_to = reply.get("in_reply_to", "")
        thread_id = reply.get("thread_id", "")
        subject = reply.get("subject", "")
        
        # Step 1: Try token-based matching (preferred)
        token = self._extract_reminder_token(body)
        if token:
            logger.debug(f"[RESCHEDULE_MATCH] Found RP_REMINDER_ID token: {token[:16]}...")
            invite = self.store.get_calendar_invite_by_token(token)
            if invite:
                logger.info(f"[RESCHEDULE_MATCH] Matched by token: {token[:16]}...")
                return invite, "token"
            else:
                logger.warning(f"[RESCHEDULE_MATCH] Token found but no matching invite: {token[:16]}...")
        
        # Step 2: Try thread_id matching (Gmail)
        if thread_id:
            logger.debug(f"[RESCHEDULE_MATCH] Trying thread_id: {thread_id[:20] if len(thread_id) > 20 else thread_id}")
            invite = self.store.get_calendar_invite_by_thread_id(thread_id)
            if invite:
                logger.info(f"[RESCHEDULE_MATCH] Matched by thread_id")
                return invite, "thread_id"
        
        # Step 3: Try In-Reply-To matching (with format variations)
        if in_reply_to:
            # Try exact match
            invite = self.store.get_calendar_invite_by_message_id(in_reply_to)
            if invite:
                logger.info(f"[RESCHEDULE_MATCH] Matched by In-Reply-To (exact)")
                return invite, "in_reply_to"
            
            # Try with angle brackets
            if not in_reply_to.startswith("<"):
                invite = self.store.get_calendar_invite_by_message_id(f"<{in_reply_to}>")
                if invite:
                    logger.info(f"[RESCHEDULE_MATCH] Matched by In-Reply-To (with brackets)")
                    return invite, "in_reply_to_brackets"
            
            # Try without angle brackets
            stripped = in_reply_to.strip("<>")
            if stripped != in_reply_to:
                invite = self.store.get_calendar_invite_by_message_id(stripped)
                if invite:
                    logger.info(f"[RESCHEDULE_MATCH] Matched by In-Reply-To (stripped)")
                    return invite, "in_reply_to_stripped"
        
        # Step 4: Fuzzy fallback - subject + time proximity
        # Only if subject contains "Reading Reminder" and from owner
        subject_lower = (subject or "").lower()
        if "reading reminder" in subject_lower or "researchpulse" in subject_lower:
            logger.debug(f"[RESCHEDULE_MATCH] Trying fuzzy subject match")
            recent_invites = self.store.get_recent_calendar_invites(
                user_id=self.user_id,
                days=7,
                subject_contains="Reading Reminder"
            )
            
            if len(recent_invites) == 1:
                # Only one candidate - use it
                logger.info(f"[RESCHEDULE_MATCH] Matched by fuzzy (single recent invite)")
                return recent_invites[0], "fuzzy_single"
            elif len(recent_invites) > 1:
                # Multiple candidates - log for debugging but don't auto-match
                logger.warning(
                    f"[RESCHEDULE_MATCH] Multiple candidates ({len(recent_invites)}) for fuzzy match - "
                    f"cannot auto-select"
                )
        
        return None, "none"
    
    def _handle_reschedule_reply(self, reply: Dict[str, Any]) -> bool:
        """
        Handle a single reschedule reply with robust matching.
        
        Returns True if successfully rescheduled.
        
        Matching strategy (in order):
        1. RP_REMINDER_ID token in body
        2. Gmail thread_id
        3. In-Reply-To header (with format variations)
        4. Fuzzy subject + time proximity (single candidate only)
        """
        from_email = reply.get("from_email", "")
        body = reply.get("body_text", "")
        in_reply_to = reply.get("in_reply_to", "")
        message_id = reply.get("message_id", "")
        subject = reply.get("subject", "")
        
        logger.info(f"[_handle_reschedule_reply] Processing reschedule from {from_email}")
        logger.debug(f"[_handle_reschedule_reply] Subject: {subject[:50] if subject else 'None'}...")
        logger.debug(f"[_handle_reschedule_reply] In-Reply-To: {in_reply_to[:50] if in_reply_to else 'None'}...")
        
        # Find the original calendar invite using fallback chain
        invite, match_method = self._find_invite_with_fallback(reply)
        
        if not invite:
            # Log detailed failure info for debugging
            token = self._extract_reminder_token(body)
            recent_count = len(self.store.get_recent_calendar_invites(self.user_id, days=7))
            
            logger.warning(
                f"[RESCHEDULE_MATCH_FAILED] Could not match reschedule reply:\n"
                f"  - In-Reply-To: {in_reply_to[:50] if in_reply_to else 'None'}...\n"
                f"  - Token found: {token[:16] + '...' if token else 'No'}\n"
                f"  - Thread ID available: {bool(reply.get('thread_id'))}\n"
                f"  - Subject: {subject[:40] if subject else 'None'}...\n"
                f"  - Recent invites (7d): {recent_count}\n"
                f"  - Steps tried: token, thread_id, in_reply_to (3 formats), fuzzy"
            )
            return False
        
        logger.info(f"[RESCHEDULE] Invite matched via '{match_method}'")
        
        # Parse the requested new datetime
        new_datetime, datetime_text = extract_reschedule_datetime(body)
        
        if not new_datetime:
            logger.warning(f"[RESCHEDULE_PARSE] No datetime found in reschedule request body")
            # Record the reply for manual review
            try:
                self.store.create_inbound_email_reply(
                    user_id=self.user_id,
                    original_invite_id=UUID(invite["id"]),
                    message_id=message_id,
                    from_email=from_email,
                    subject=reply.get("subject"),
                    body_text=body,
                    in_reply_to=in_reply_to,
                )
            except Exception as e:
                logger.error(f"[INBOUND] Error storing reply: {e}")
            # Send clarification email (once per thread)
            self._send_reschedule_clarification(from_email, subject, message_id)
            return False
        
        # Validate parsed datetime is in the future
        now = datetime.utcnow()
        if new_datetime < now:
            logger.warning(
                f"[RESCHEDULE_PARSE] Parsed datetime is in the past: "
                f"{new_datetime.isoformat()} < {now.isoformat()}"
            )
            self._send_reschedule_clarification(from_email, subject, message_id)
            return False
        
        logger.info(
            f"[RESCHEDULE_PARSE] extracted_text=\"{datetime_text}\" "
            f"parsed_dt={new_datetime.isoformat()}"
        )
        
        # Get the calendar event
        event = self.store.get_calendar_event(UUID(invite["calendar_event_id"]))
        if not event:
            logger.error(f"[INBOUND] Calendar event not found: {invite['calendar_event_id']}")
            return False
        
        old_time = event.get("start_time")
        logger.info(
            f"[RESCHEDULE_UPDATE] reminder_id={invite['id']} "
            f"old_dt={old_time} new_dt={new_datetime.isoformat()}"
        )
        
        # Reschedule the event
        self.store.reschedule_calendar_event(
            event_id=UUID(event["id"]),
            new_start_time=new_datetime,
            reschedule_note=f"Rescheduled via email reply: {datetime_text or 'user request'}",
        )
        
        # Record the reply
        self.store.create_inbound_email_reply(
            user_id=self.user_id,
            original_invite_id=UUID(invite["id"]),
            message_id=message_id,
            from_email=from_email,
            subject=reply.get("subject"),
            body_text=body,
            in_reply_to=in_reply_to,
        )
        
        # Log success with details
        logger.info(
            f"[RESCHEDULE_SUCCESS] reminder_id={invite['id']}, "
            f"old_time={old_time}, new_time={new_datetime.isoformat()}, "
            f"match_method={match_method}"
        )
        
        # Send updated calendar invite (pass invite for ics_uid fallback)
        sent_ok = self._send_reschedule_confirmation(event, new_datetime, invite)
        logger.info(
            f"[RESCHEDULE_EMAIL] sent_update_invite={sent_ok} "
            f"reminder_id={invite['id']}"
        )
        
        return True
    
    def _send_reschedule_confirmation(
        self,
        event: Dict[str, Any],
        new_time: datetime,
        invite: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send a confirmation calendar invite for the reschedule.
        
        Looks up ics_uid from event, then invite record, then CalendarInviteEmail table.
        Returns True if the email was sent successfully.
        """
        try:
            from .calendar_invite_sender import send_reschedule_invite
            
            user = self.store.get_user(self.user_id)
            if user and user.get("email"):
                # Get papers for the event (handles both UUID and arXiv-ID formats)
                papers = []
                paper_ids = event.get("paper_ids") or []
                if event.get("paper_id"):
                    paper_ids.append(str(event["paper_id"]))
                
                for pid in paper_ids:
                    paper = None
                    # Try UUID lookup first
                    try:
                        paper = self.store.get_paper(UUID(pid))
                    except (ValueError, AttributeError):
                        pass
                    # Fall back to arXiv external_id lookup
                    if not paper:
                        try:
                            paper = self.store.get_paper_by_external_id("arxiv", pid)
                        except Exception:
                            pass
                    if paper:
                        papers.append(paper)
                
                if not papers:
                    logger.warning(
                        f"[RESCHEDULE_EMAIL] No papers resolved from paper_ids={paper_ids}, "
                        f"paper_id={event.get('paper_id')}"
                    )
                
                # Resolve ics_uid: event → invite record → DB lookup
                ics_uid = event.get("ics_uid")
                if not ics_uid and invite:
                    ics_uid = invite.get("ics_uid")
                    if ics_uid:
                        logger.info("[RESCHEDULE_EMAIL] ics_uid resolved from invite record")
                if not ics_uid:
                    # Last resort: look up from CalendarInviteEmail table
                    try:
                        invite_emails = self.store.list_calendar_invite_emails(
                            user_id=self.user_id,
                            calendar_event_id=UUID(event["id"]),
                            limit=1,
                        )
                        if invite_emails:
                            ics_uid = invite_emails[0].get("ics_uid")
                            if ics_uid:
                                logger.info("[RESCHEDULE_EMAIL] ics_uid resolved from invite email table")
                    except Exception as lookup_err:
                        logger.debug(f"[RESCHEDULE_EMAIL] ics_uid DB lookup failed: {lookup_err}")
                
                if not ics_uid:
                    logger.warning("[RESCHEDULE_EMAIL] Event missing ics_uid, cannot send update invite")
                    return False
                
                # Backfill ics_uid to event for future reschedules
                if not event.get("ics_uid") and ics_uid:
                    try:
                        self.store.update_calendar_event(UUID(event["id"]), {"ics_uid": ics_uid})
                        logger.info(f"[RESCHEDULE_EMAIL] Backfilled ics_uid to event {event['id']}")
                    except Exception:
                        pass

                old_sequence = event.get("sequence_number") or 0
                duration = event.get("duration_minutes") or 30

                result = send_reschedule_invite(
                    user_email=user["email"],
                    user_name=user.get("name", user["email"]),
                    papers=papers,
                    new_start_time=new_time,
                    duration_minutes=duration,
                    ics_uid=ics_uid,
                    sequence=old_sequence + 1,
                    reschedule_reason="Rescheduled via email reply",
                )
                if result.get("success"):
                    logger.info(f"[RESCHEDULE_EMAIL] Sent reschedule confirmation to {user['email']}")
                    return True
                else:
                    logger.error(f"[RESCHEDULE_EMAIL] Send failed: {result.get('error')}")
        except Exception as e:
            logger.error(f"[RESCHEDULE_EMAIL] Error sending reschedule confirmation: {e}")
        return False
    
    def _send_reschedule_clarification(
        self, to_email: str, original_subject: str, in_reply_to_id: str
    ) -> None:
        """Send a clarification email when we can't parse the requested datetime.
        
        Only sends one clarification per thread (checks DB before sending).
        """
        try:
            # Anti-spam: check if we already sent a clarification for this thread
            if self.store.is_email_processed(
                self.user_id, f"clarification_{in_reply_to_id}"
            ):
                logger.debug(
                    f"[RESCHEDULE_CLARIFY] Skipping duplicate clarification for {in_reply_to_id[:30]}"
                )
                return

            body = (
                "Hi,\n\n"
                "I understood you'd like to reschedule, but I couldn't figure out "
                "the new date and time from your reply.\n\n"
                "Could you reply again with a clear date and time? For example:\n\n"
                '  "Reschedule to February 15, 2026 at 3:00 PM"\n\n'
                "Thanks!\n"
                "— ResearchPulse"
            )

            success, msg_id, error = send_outbound_email(
                to_email=to_email,
                subject=f"Re: {original_subject}" if original_subject else "Reschedule clarification",
                body=body,
                email_type=EmailType.RESCHEDULE,
                skip_tag=True,
            )
            if success:
                # Mark so we don't send again for this thread
                self.store.mark_email_processed(
                    user_id=self.user_id,
                    gmail_message_id=f"clarification_{in_reply_to_id}",
                    email_type="reschedule_clarification",
                    processing_result="clarification_sent",
                    from_email="system",
                    subject=original_subject or "",
                )
                logger.info(f"[RESCHEDULE_CLARIFY] Sent clarification to {to_email}")
            else:
                logger.error(f"[RESCHEDULE_CLARIFY] Failed to send clarification: {error}")
        except Exception as e:
            logger.error(f"[RESCHEDULE_CLARIFY] Error sending clarification email: {e}")
    
    def _process_colleague_joins(self, since_hours: int) -> Dict[str, Any]:
        """Process colleague join request emails."""
        logger.info(f"[INBOUND] Fetching colleague join requests (last {since_hours}h)")
        
        results = {
            "scanned": 0,
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "rejected": 0,
            "already_processed": 0,
            "onboarding_continued": 0,
        }
        
        # Check if join code is configured
        join_hash = self.join_code_hash
        if not join_hash:
            logger.info("[INBOUND] No join code configured in DB - will send instructions to new senders")
        else:
            logger.info(f"[INBOUND] Join code configured (hash_prefix={join_hash[:10]}...)")
        
        # Fetch potential signup emails
        signups = fetch_colleague_signup_emails(since_hours=since_hours)
        results["scanned"] = len(signups)
        
        # Pre-load colleagues who need onboarding info for efficient lookup
        # Include colleagues with:
        # - Explicit onboarding status (needs_interests, needs_name, pending)
        # - OR missing interests (even if status is wrong/complete due to earlier bugs)
        # - OR name is "Pending" placeholder
        all_colleagues = self.store.list_colleagues(self.user_id)
        onboarding_colleagues = {}
        for c in all_colleagues:
            email_key = c.get("email", "").lower()
            if not email_key:
                continue
            
            onboarding_status = c.get("onboarding_status", "")
            interests = c.get("interests") or c.get("research_interests") or ""
            name = c.get("name") or ""
            
            needs_onboarding = (
                onboarding_status in ("needs_interests", "needs_name", "pending", "awaiting_clarification") or
                not interests.strip() or  # Missing interests
                name.lower() == "pending"  # Placeholder name
            )
            
            if needs_onboarding:
                onboarding_colleagues[email_key] = c
                logger.debug(f"[INBOUND] Colleague {email_key} needs onboarding: "
                           f"status={onboarding_status}, has_interests={bool(interests.strip())}, name={name}")
        
        logger.info(f"[INBOUND] Found {len(onboarding_colleagues)} colleagues needing onboarding info")
        
        logger.info(f"[INBOUND] Processing {len(signups)} potential colleague join emails")
        
        # Per-batch dedup: track senders we've already sent instruction
        # replies to within THIS polling cycle to avoid duplicate replies
        # for multiple emails from the same person in one scan.
        instructed_this_batch: set[str] = set()
        
        for signup in signups:
            message_id = signup.get("message_id", "")
            thread_id = signup.get("thread_id", "")
            from_email = signup.get("from_email", "")
            subject = signup.get("subject", "")
            body = signup.get("body_text", "")
            from_name = signup.get("from_name", "")
            corr_id = signup.get("correlation_id", f"{message_id[:15]}")
            
            logger.info(f"[JOIN_PROC][{corr_id}] Processing from={from_email}, name={from_name}")
            
            # PRIORITY CHECK: Is this an onboarding continuation?
            # If the sender is already a colleague who needs onboarding info, treat this as
            # an onboarding reply (providing interests/name) rather than a new signup.
            # 
            # IMPORTANT: Only treat as onboarding continuation if the email does NOT contain
            # a join code! If it has a join code, it's an initial signup attempt, not a
            # reply to an onboarding question.
            onboarding_colleague = onboarding_colleagues.get(from_email.lower())
            provided_code = extract_join_code_from_email(body, subject)
            
            if onboarding_colleague and not provided_code:
                # This is an onboarding continuation (reply without join code from existing colleague)
                # Check if THIS SPECIFIC email was already processed as an onboarding_continuation.
                processed_info = self.store.get_processed_email_info(self.user_id, message_id)
                already_handled_as_onboarding = (
                    processed_info and 
                    processed_info.get("email_type") == "onboarding_continuation" and
                    processed_info.get("processing_result", "").startswith("updated_to_")
                )
                
                if already_handled_as_onboarding:
                    logger.info(f"[IDEMPOTENT][{corr_id}] SKIP: already processed as onboarding continuation")
                    results["already_processed"] += 1
                    continue
                
                logger.info(f"[JOIN_PROC][{corr_id}] Detected ONBOARDING CONTINUATION for colleague {onboarding_colleague.get('id')} (no join code in email)")
                results["processed"] += 1
                
                try:
                    success = self._process_onboarding_continuation(
                        colleague=onboarding_colleague,
                        email_body=body,
                        from_name=from_name,
                        message_id=message_id,
                        subject=subject,
                        from_email=from_email,
                        corr_id=corr_id,
                    )
                    if success:
                        results["onboarding_continued"] += 1
                        results["succeeded"] += 1
                        # Remove from onboarding map if complete
                        del onboarding_colleagues[from_email.lower()]
                    else:
                        results["failed"] += 1
                except Exception as e:
                    logger.error(f"[JOIN_PROC][{corr_id}] Error in onboarding continuation: {e}", exc_info=True)
                    results["failed"] += 1
                continue
            
            # Not an onboarding continuation - check normal idempotency
            if self.store.is_email_processed(self.user_id, message_id):
                logger.info(f"[IDEMPOTENT][{corr_id}] SKIP: already_processed=true")
                results["already_processed"] += 1
                continue
            
            logger.debug(f"[IDEMPOTENT][{corr_id}] not_processed_yet=true")
            results["processed"] += 1
            
            # ── Join code gating ──
            # A join code is ALWAYS required for colleague signups.
            # Per-batch dedup: only send one instruction email per sender per poll cycle.
            sender_key = from_email.lower()
            already_instructed_this_batch = sender_key in instructed_this_batch
            
            if not provided_code:
                if already_instructed_this_batch:
                    logger.info(f"[JOIN_PROC][{corr_id}] No code & already instructed {from_email} this batch - silent skip")
                    self.store.mark_email_processed(
                        user_id=self.user_id,
                        gmail_message_id=message_id,
                        email_type="colleague_join",
                        processing_result="repeat_ignored",
                        from_email=from_email,
                        subject=subject,
                    )
                    results["rejected"] += 1
                    continue
                
                # Send instruction reply
                logger.info(f"[JOIN_PROC][{corr_id}] No code found - sending instruction reply")
                self._send_join_code_required_reply(from_email, from_name)
                instructed_this_batch.add(sender_key)
                self.store.mark_email_processed(
                    user_id=self.user_id,
                    gmail_message_id=message_id,
                    email_type="colleague_join",
                    processing_result="rejected_no_code_replied",
                    from_email=from_email,
                    subject=subject,
                )
                results["rejected"] += 1
                continue
            
            # Code provided but owner hasn't configured a join code yet
            if not join_hash:
                if already_instructed_this_batch:
                    logger.info(f"[JOIN_PROC][{corr_id}] Code provided but no join code in DB & already instructed this batch - silent skip")
                    self.store.mark_email_processed(
                        user_id=self.user_id,
                        gmail_message_id=message_id,
                        email_type="colleague_join",
                        processing_result="repeat_ignored",
                        from_email=from_email,
                        subject=subject,
                    )
                    results["rejected"] += 1
                    continue
                
                logger.info(f"[JOIN_PROC][{corr_id}] Code provided but no join code configured - sending not-configured reply")
                self._send_not_configured_reply(from_email, from_name)
                instructed_this_batch.add(sender_key)
                self.store.mark_email_processed(
                    user_id=self.user_id,
                    gmail_message_id=message_id,
                    email_type="colleague_join",
                    processing_result="rejected_not_configured_replied",
                    from_email=from_email,
                    subject=subject,
                )
                results["rejected"] += 1
                continue
            
            # Code provided - validate against the stored join hash
            # (always try, even if previously instructed - they followed instructions!)
            code_valid = verify_join_code(provided_code, join_hash)
            if not code_valid:
                if already_instructed_this_batch:
                    logger.info(f"[JOIN_PROC][{corr_id}] Invalid code & already instructed this batch - silent skip")
                    self.store.mark_email_processed(
                        user_id=self.user_id,
                        gmail_message_id=message_id,
                        email_type="colleague_join",
                        processing_result="repeat_ignored",
                        from_email=from_email,
                        subject=subject,
                    )
                    results["rejected"] += 1
                    continue
                
                logger.warning(f"[JOIN_PROC][{corr_id}] Invalid code - sending instruction reply")
                self._send_invalid_code_reply(from_email, from_name)
                instructed_this_batch.add(sender_key)
                self.store.mark_email_processed(
                    user_id=self.user_id,
                    gmail_message_id=message_id,
                    email_type="colleague_join",
                    processing_result="rejected_invalid_code_replied",
                    from_email=from_email,
                    subject=subject,
                )
                results["rejected"] += 1
                continue
            
            # ── Valid code - process signup ──
            logger.info(f"[JOIN_PROC][{corr_id}] Code VALID - analyzing email with LLM reasoning")
            
            # PRIORITY: First try direct template extraction (matches our instruction format)
            template_fields = extract_signup_template_fields(body, subject)
            template_name = template_fields.get("name")
            template_interests = template_fields.get("interests")
            
            if template_name or template_interests:
                logger.info(f"[JOIN_PROC][{corr_id}] Template extraction found: name={template_name}, interests={template_interests}")
            
            # Use LLM reasoning to understand the email content
            try:
                # Step 1: Analyze the email with reasoning
                analysis = self._analyze_signup_email_with_reasoning(body, subject, from_name, corr_id)
                intent = analysis.get("intent", "unclear")
                response_type = analysis.get("response_type", "clarify_intent")
                extracted_name = template_name or analysis.get("name") or from_name
                extracted_interests = template_interests or analysis.get("interests")
                custom_message = analysis.get("custom_message")
                
                logger.info(f"[JOIN_PROC][{corr_id}] Reasoning analysis: intent={intent}, response_type={response_type}")
                
                # Override intent if we got clear data from template parsing
                # (user followed our instruction format = clear signup intent)
                if template_interests:
                    logger.info(f"[JOIN_PROC][{corr_id}] Template interests found - overriding intent to clear_signup")
                    intent = "clear_signup"
                    response_type = "welcome_complete" if extracted_name and extracted_interests else "ask_all"
                
                # Step 2: Handle based on analysis result
                if intent in ("just_code", "unclear") or response_type == "clarify_intent":
                    # The email doesn't show clear signup intent - ask for clarification
                    # Still add them as a pending colleague so we can track the conversation
                    colleague_id, _ = self._add_colleague_from_email_with_analysis(
                        signup, extracted_name, extracted_interests, "awaiting_clarification"
                    )
                    
                    if colleague_id:
                        results["succeeded"] += 1
                        self._send_clarify_intent_reply(from_email, extracted_name, custom_message, thread_id)
                        logger.info(f"[REPLY][{corr_id}] Sent clarify intent email (unclear signup)")
                        
                        self.store.mark_email_processed(
                            user_id=self.user_id,
                            gmail_message_id=message_id,
                            email_type="colleague_join",
                            processing_result="awaiting_clarification",
                            from_email=from_email,
                            subject=subject,
                        )
                    else:
                        results["failed"] += 1
                        self.store.mark_email_processed(
                            user_id=self.user_id,
                            gmail_message_id=message_id,
                            email_type="colleague_join",
                            processing_result="failed",
                            from_email=from_email,
                            subject=subject,
                        )
                    continue
                
                # Step 3: Clear signup intent - proceed with adding colleague
                colleague_id, onboarding_status = self._add_colleague_from_email_with_analysis(
                    signup, extracted_name, extracted_interests, None
                )
                
                if colleague_id:
                    results["succeeded"] += 1
                    logger.info(f"[DB_WRITE][{corr_id}] Colleague upsert SUCCESS - id={colleague_id}, onboarding_status={onboarding_status}")
                    
                    # Send appropriate reply based on onboarding status
                    if onboarding_status == "complete":
                        self._send_join_success_reply(from_email, extracted_name, interests=extracted_interests)
                        logger.info(f"[REPLY][{corr_id}] Sent welcome email (complete)")
                    elif onboarding_status == "needs_interests":
                        self._send_onboarding_interests_reply(from_email, extracted_name, thread_id)
                        logger.info(f"[REPLY][{corr_id}] Sent interests request email")
                    elif onboarding_status == "needs_name":
                        self._send_onboarding_name_reply(from_email, extracted_name, thread_id)
                        logger.info(f"[REPLY][{corr_id}] Sent name request email")
                    else:  # pending (both missing)
                        self._send_onboarding_questions_reply(from_email, extracted_name, thread_id)
                        logger.info(f"[REPLY][{corr_id}] Sent full onboarding questions email")
                    
                    self.store.mark_email_processed(
                        user_id=self.user_id,
                        gmail_message_id=message_id,
                        email_type="colleague_join",
                        processing_result=f"success_{onboarding_status}",
                        from_email=from_email,
                        subject=subject,
                    )
                else:
                    results["failed"] += 1
                    logger.error(f"[DB_WRITE][{corr_id}] Colleague upsert FAILED")
                    self.store.mark_email_processed(
                        user_id=self.user_id,
                        gmail_message_id=message_id,
                        email_type="colleague_join",
                        processing_result="failed",
                        from_email=from_email,
                        subject=subject,
                    )
            except Exception as e:
                logger.error(f"[JOIN_PROC][{corr_id}] Exception adding colleague: {e}", exc_info=True)
                results["failed"] += 1
                self.store.mark_email_processed(
                    user_id=self.user_id,
                    gmail_message_id=message_id,
                    email_type="colleague_join",
                    processing_result="error",
                    from_email=from_email,
                    subject=subject,
                    error_message=str(e),
                )
        
        return results
    
    def _process_onboarding_continuation(
        self,
        colleague: Dict[str, Any],
        email_body: str,
        from_name: str,
        message_id: str,
        subject: str,
        from_email: str,
        corr_id: str,
    ) -> bool:
        """
        Process an onboarding continuation email where a colleague provides missing info.
        
        This handles replies to onboarding emails where the colleague provides:
        - Research interests (if status was needs_interests)
        - Their name (if status was needs_name)
        - Both (if status was pending)
        
        Returns:
            True if onboarding was successfully updated, False otherwise
        """
        from src.tools.arxiv_categories import derive_arxiv_categories_from_interests
        
        colleague_id = colleague.get("id")
        current_status = colleague.get("onboarding_status", "")
        current_name = colleague.get("name", "")
        current_interests = colleague.get("interests") or colleague.get("research_interests") or ""
        
        logger.info(f"[ONBOARD_CONT][{corr_id}] Processing continuation: "
                   f"colleague_id={colleague_id}, current_status={current_status}")
        
        # PRIMARY: Use LLM to intelligently extract name and interests from the email
        # This handles multi-language content, reply markers, and email thread garbage
        logger.info(f"[ONBOARD_CONT][{corr_id}] Attempting LLM extraction for name and interests...")
        llm_result = self._extract_colleague_info_with_llm(email_body, corr_id)
        llm_extracted_name = llm_result.get("name")
        llm_extracted_interests = llm_result.get("interests")
        logger.info(f"[ONBOARD_CONT][{corr_id}] LLM extraction result: name={llm_extracted_name}, interests={llm_extracted_interests[:100] if llm_extracted_interests else 'None'}")
        
        # FALLBACK: Pattern-based extraction if LLM fails or is unavailable
        clean_body = self._extract_new_content_from_reply(email_body)
        logger.info(f"[ONBOARD_CONT][{corr_id}] Pattern-cleaned body length: {len(clean_body)} chars")
        
        # Try to extract name - prefer LLM, fallback to patterns
        extracted_name = llm_extracted_name or self._extract_name_from_body(clean_body)
        
        # Determine interests - prefer LLM extraction
        if llm_extracted_interests:
            interests_text = llm_extracted_interests  # Already cleaned by LLM extraction
            has_meaningful_interests = True
            logger.info(f"[ONBOARD_CONT][{corr_id}] Using LLM-extracted interests: {interests_text[:100]}...")
        else:
            # Fallback to pattern matching
            interests_text, has_meaningful_interests = self._extract_interests_from_body(clean_body)
            # ALWAYS clean the interests text
            interests_text = self._clean_interests_text(interests_text)
            logger.info(f"[ONBOARD_CONT][{corr_id}] Pattern extraction: name={extracted_name}, "
                       f"has_interests={has_meaningful_interests}")
            
            # Last resort: if no keywords found but reply looks like a list
            if not has_meaningful_interests and len(clean_body.strip()) > 10:
                if ',' in clean_body or '\n' in clean_body:
                    logger.info(f"[ONBOARD_CONT][{corr_id}] Treating reply as interest list (comma/newline separated)")
                    interests_text = self._clean_interests_text(clean_body.strip()[:500])
                    has_meaningful_interests = bool(interests_text.strip())
        
        # Determine what we have now
        has_name_now = bool(extracted_name) or (bool(current_name) and current_name.lower() != "pending")
        has_interests_now = has_meaningful_interests or bool(current_interests.strip())
        
        # Check if we're getting NEW information
        new_name = extracted_name if extracted_name and (not current_name or current_name.lower() == "pending") else None
        new_interests = interests_text if has_meaningful_interests and not current_interests.strip() else None
        
        if not new_name and not new_interests:
            # Reply doesn't contain the information we need
            logger.warning(f"[ONBOARD_CONT][{corr_id}] Reply doesn't contain needed info. "
                          f"needed={current_status}, found_name={bool(extracted_name)}, found_interests={has_meaningful_interests}")
            
            # Send a follow-up asking for the specific info again
            if current_status == "needs_interests":
                self._send_onboarding_interests_reply(from_email, from_name or current_name, "")
            elif current_status == "needs_name":
                self._send_onboarding_name_reply(from_email, from_name or current_name, "")
            elif current_status == "awaiting_clarification":
                # They replied to our clarification - ask for interests now
                self._send_onboarding_interests_reply(from_email, from_name or current_name, "")
            else:
                self._send_onboarding_questions_reply(from_email, from_name or current_name, "")
            
            self.store.mark_email_processed(
                user_id=self.user_id,
                gmail_message_id=message_id,
                email_type="onboarding_continuation",
                processing_result="needs_more_info",
                from_email=from_email,
                subject=subject,
            )
            return False
        
        # Build update data
        update_data = {}
        
        if new_name:
            update_data["name"] = new_name
            has_name_now = True
            logger.info(f"[ONBOARD_CONT][{corr_id}] Will update name to: {new_name}")
        
        if new_interests:
            update_data["interests"] = new_interests
            update_data["research_interests"] = new_interests
            has_interests_now = True
            logger.info(f"[ONBOARD_CONT][{corr_id}] Will update interests: {new_interests[:100]}...")
            
            # Derive arXiv categories
            try:
                derived_categories = derive_arxiv_categories_from_interests(new_interests)
                update_data["derived_arxiv_categories"] = derived_categories
                update_data["categories"] = (
                    derived_categories.get("primary", []) + 
                    derived_categories.get("secondary", [])
                )
                logger.info(f"[ONBOARD_CONT][{corr_id}] Derived categories: {derived_categories}")
            except Exception as e:
                logger.warning(f"[ONBOARD_CONT][{corr_id}] Failed to derive categories: {e}")
        
        # Compute new status
        if has_name_now and has_interests_now:
            new_status = "complete"
            update_data["enabled"] = True
        elif not has_interests_now:
            new_status = "needs_interests"
        elif not has_name_now:
            new_status = "needs_name"
        else:
            new_status = "pending"
        
        update_data["onboarding_status"] = new_status
        
        logger.info(f"[ONBOARD_CONT][{corr_id}] New status: {current_status} -> {new_status}")
        
        # Update the colleague
        try:
            self.store.update_colleague(UUID(str(colleague_id)), update_data)
            logger.info(f"[DB_WRITE][{corr_id}] Colleague update SUCCESS")
        except Exception as e:
            logger.error(f"[DB_WRITE][{corr_id}] Colleague update FAILED: {e}")
            self.store.mark_email_processed(
                user_id=self.user_id,
                gmail_message_id=message_id,
                email_type="onboarding_continuation",
                processing_result="db_error",
                from_email=from_email,
                subject=subject,
            )
            return False
        
        # Send appropriate response
        if new_status == "complete":
            final_interests = new_interests if new_interests else None
            self._send_join_success_reply(from_email, new_name or from_name or current_name, interests=final_interests)
            logger.info(f"[REPLY][{corr_id}] Sent onboarding complete welcome email")
        elif new_status == "needs_interests":
            self._send_onboarding_interests_reply(from_email, new_name or from_name or current_name, "")
            logger.info(f"[REPLY][{corr_id}] Sent interests request (still needed)")
        elif new_status == "needs_name":
            self._send_onboarding_name_reply(from_email, new_name or from_name or current_name, "")
            logger.info(f"[REPLY][{corr_id}] Sent name request (still needed)")
        
        self.store.mark_email_processed(
            user_id=self.user_id,
            gmail_message_id=message_id,
            email_type="onboarding_continuation",
            processing_result=f"updated_to_{new_status}",
            from_email=from_email,
            subject=subject,
        )
        
        return new_status == "complete"
    
    def _clean_interests_text(self, interests: str) -> str:
        """
        Clean extracted interests text by removing non-English garbage.
        
        Removes:
        - Hebrew/Arabic/RTL text and markers
        - Email reply markers in any language  
        - Date/time patterns
        - Email addresses
        - Anything after common reply markers
        """
        if not interests:
            return ""
        
        # First, cut off at common reply markers (any language)
        reply_markers = [
            r'\u202b',  # RTL embedding character
            r'\u200f',  # RTL mark
            r'\u200e',  # LTR mark
            'בתאריך',  # Hebrew "On date"
            'ב-',  # Hebrew time prefix
            'מאת',  # Hebrew "from"
            'On ', 
            'wrote:',
            'From:',
            'Sent:',
            '----',
            '___',
        ]
        
        result = interests
        for marker in reply_markers:
            if marker in result:
                result = result.split(marker)[0]
        
        # Remove any remaining RTL/Hebrew/Arabic characters (Unicode ranges)
        # Hebrew: \u0590-\u05FF, Arabic: \u0600-\u06FF, RTL marks: \u200F, \u202B, etc.
        import re
        result = re.sub(r'[\u0590-\u05FF\u0600-\u06FF\u200E-\u200F\u202A-\u202E]+', '', result)
        
        # Remove email addresses
        result = re.sub(r'<[^>]+@[^>]+>', '', result)
        result = re.sub(r'\S+@\S+\.\S+', '', result)
        
        # Remove date patterns
        result = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', '', result)
        result = re.sub(r'\d{4}[/-]\d{1,2}[/-]\d{1,2}', '', result)
        
        # Clean up whitespace and punctuation
        result = re.sub(r'\s+', ' ', result)
        result = result.strip().strip(',').strip()
        
        return result
    
    def _analyze_signup_email_with_reasoning(self, email_body: str, subject: str, from_name: str, corr_id: str) -> Dict[str, Any]:
        """
        Use LLM reasoning to analyze a signup email and determine the appropriate response.
        
        This provides adaptive, intelligent handling of signup emails by understanding:
        - Whether the email shows clear intent to receive research paper updates
        - Whether the content is meaningful or just noise (e.g., "GOOD MORNING")
        - What information (name, interests) can be extracted
        - What response is most appropriate
        
        Args:
            email_body: The email body text
            subject: The email subject
            from_name: The sender's name from email header
            corr_id: Correlation ID for logging
            
        Returns:
            Dict with:
                - 'intent': 'clear_signup' | 'unclear' | 'just_code' | 'other'
                - 'name': Extracted name or None
                - 'interests': Extracted interests or None
                - 'response_type': 'welcome_complete' | 'ask_interests' | 'clarify_intent' | 'ask_all'
                - 'custom_message': Optional personalized message based on content
        """
        import os
        import json
        
        result = {
            "intent": "unclear",
            "name": None,
            "interests": None,
            "response_type": "clarify_intent",
            "custom_message": None,
        }
        
        try:
            import openai
            
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                logger.warning(f"[LLM_REASON][{corr_id}] No OpenAI API key - using basic analysis")
                # Fallback: if email has research keywords, treat as clear intent
                body_lower = email_body.lower()
                research_keywords = ['machine learning', 'nlp', 'ai', 'research', 'papers', 'deep learning', 
                                    'computer vision', 'neural', 'transformer', 'llm']
                if any(kw in body_lower for kw in research_keywords):
                    result["intent"] = "clear_signup"
                    result["response_type"] = "ask_interests"
                return result
            
            client = openai.OpenAI(api_key=api_key)
            
            prompt = f"""You are ResearchPulse, a service that sends personalized research paper recommendations.

Someone sent an email with a valid join code. Analyze this email to understand their intent.

EMAIL SUBJECT: {subject}
SENDER NAME: {from_name or 'Unknown'}
EMAIL BODY:
---
{email_body[:3000]}
---

ANALYZE this email and determine:

1. INTENT - What does this person want?
   - "clear_signup": They clearly want to receive research paper updates and provided useful info
   - "just_code": They only sent a code with no meaningful content (e.g., just "code: 123456" or "GOOD MORNING")
   - "unclear": The email doesn't clearly indicate they understand what ResearchPulse does
   - "other": Something else entirely

2. NAME - Can you extract their name? (from signature, "my name is...", etc.)

3. INTERESTS - Did they mention ANY research interests or topics? 
   - Look for academic fields, technologies, research areas
   - e.g., "machine learning", "NLP", "computer vision", "biology", "physics"

4. RESPONSE_TYPE - What response is most appropriate?
   - "welcome_complete": They provided name AND interests - just welcome them
   - "ask_interests": We have their name but need interests
   - "ask_all": We need both name and interests
   - "clarify_intent": The email is unclear - ask if they want research paper updates

5. CUSTOM_MESSAGE - Write a SHORT, friendly, personalized note (1-2 sentences) based on what they wrote.
   - If they said "GOOD MORNING", acknowledge it warmly
   - If they mentioned a topic vaguely, show interest
   - Be conversational and human
   - If nothing to personalize, leave as null

OUTPUT FORMAT - respond with ONLY this JSON:
{{"intent": "...", "name": "... or null", "interests": "topic1, topic2 or null", "response_type": "...", "custom_message": "... or null"}}

JSON:"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400,
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.info(f"[LLM_REASON][{corr_id}] Raw response: {response_text[:300]}...")
            
            # Parse JSON response
            try:
                # Handle potential markdown code blocks
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                response_text = response_text.strip()
                
                parsed = json.loads(response_text)
                
                # Extract and validate fields
                if parsed.get("intent") in ("clear_signup", "just_code", "unclear", "other"):
                    result["intent"] = parsed["intent"]
                
                if parsed.get("name") and parsed["name"] not in ("null", "None", None):
                    result["name"] = parsed["name"].strip()
                
                if parsed.get("interests") and parsed["interests"] not in ("null", "None", None):
                    interests = parsed["interests"].strip()
                    # Clean the interests
                    interests = self._clean_interests_text(interests)
                    if interests:
                        result["interests"] = interests
                
                if parsed.get("response_type") in ("welcome_complete", "ask_interests", "ask_all", "clarify_intent"):
                    result["response_type"] = parsed["response_type"]
                
                if parsed.get("custom_message") and parsed["custom_message"] not in ("null", "None", None):
                    result["custom_message"] = parsed["custom_message"].strip()
                
                logger.info(f"[LLM_REASON][{corr_id}] Analysis result: intent={result['intent']}, "
                           f"response_type={result['response_type']}, has_name={bool(result['name'])}, "
                           f"has_interests={bool(result['interests'])}")
                
            except json.JSONDecodeError as je:
                logger.warning(f"[LLM_REASON][{corr_id}] Failed to parse JSON: {je}")
                
        except Exception as e:
            logger.error(f"[LLM_REASON][{corr_id}] Error in reasoning analysis: {e}")
        
        return result
    
    def _extract_colleague_info_with_llm(self, email_body: str, corr_id: str, extract_name: bool = True, extract_interests: bool = True) -> Dict[str, Optional[str]]:
        """
        Use LLM to intelligently extract colleague information from an email.
        
        This handles:
        - Multi-language content and reply markers (English, Hebrew, Arabic, etc.)
        - Various name formats (Dr. John Snow, Prof. Jane Doe, etc.)
        - Various interest formats (comma-separated, bullet points, prose)
        - Email thread garbage (quoted text, signatures, headers)
        
        Args:
            email_body: The full email body text
            corr_id: Correlation ID for logging
            extract_name: Whether to extract name from the email
            extract_interests: Whether to extract interests from the email
            
        Returns:
            Dict with 'name' and 'interests' keys (values may be None)
        """
        import os
        result = {"name": None, "interests": None}
        
        try:
            import openai
            
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                logger.warning(f"[LLM_EXTRACT][{corr_id}] No OpenAI API key - falling back to pattern matching")
                return result
            
            client = openai.OpenAI(api_key=api_key)
            
            prompt = f"""Extract information from this email signup for research paper updates.

EMAIL:
---
{email_body[:4000]}
---

CRITICAL RULES:
1. ONLY extract from the USER'S NEW message - the FIRST part before any reply markers
2. STOP reading at ANY of these patterns (in ANY language):
   - "On ... wrote:" or similar reply headers
   - Hebrew text like "בתאריך" (means "on date")
   - Lines starting with ">"
   - Email signatures
   - Previous conversation text
3. For INTERESTS: Return ONLY English research topic keywords (e.g., "machine learning, computer vision")
   - Do NOT include dates, email addresses, or non-English text
   - Do NOT include anything after a reply marker

OUTPUT FORMAT - respond with ONLY this JSON:
{{"name": "extracted name or null", "interests": "topic1, topic2, topic3 or null"}}

JSON:"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.info(f"[LLM_EXTRACT][{corr_id}] Raw LLM response: {response_text[:200]}...")
            
            # Parse JSON response
            import json
            try:
                # Handle potential markdown code blocks
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                response_text = response_text.strip()
                
                parsed = json.loads(response_text)
                
                # Extract name if present and valid
                if extract_name and parsed.get("name") and parsed["name"] != "null":
                    name = parsed["name"].strip()
                    # Basic validation
                    if 2 <= len(name) <= 50 and not name.lower() in ['null', 'none', 'unknown']:
                        result["name"] = name
                        logger.info(f"[LLM_EXTRACT][{corr_id}] Extracted name: {name}")
                
                # Extract interests if present and valid
                if extract_interests and parsed.get("interests") and parsed["interests"] != "null":
                    interests = parsed["interests"].strip()
                    # ALWAYS clean interests to remove any garbage that slipped through
                    interests = self._clean_interests_text(interests)
                    if len(interests) >= 3 and not interests.lower() in ['null', 'none']:
                        result["interests"] = interests
                        logger.info(f"[LLM_EXTRACT][{corr_id}] Extracted interests (cleaned): {interests[:100]}...")
                        
            except json.JSONDecodeError as je:
                logger.warning(f"[LLM_EXTRACT][{corr_id}] Failed to parse JSON response: {je}")
                # Try to extract from non-JSON response as fallback
                if extract_interests and 'interest' in response_text.lower():
                    # Look for interests after colon
                    for pattern in [r'interests?[:\s]+([^"\n]+)', r'topics?[:\s]+([^"\n]+)']:
                        match = re.search(pattern, response_text, re.IGNORECASE)
                        if match:
                            result["interests"] = match.group(1).strip().strip('",')
                            break
                            
        except Exception as e:
            logger.error(f"[LLM_EXTRACT][{corr_id}] Error using LLM extraction: {e}")
        
        return result
    
    def _extract_interests_with_llm(self, email_body: str, corr_id: str) -> Optional[str]:
        """
        Use LLM to intelligently extract research interests from an email reply.
        
        This handles:
        - Multi-language reply markers (English, Hebrew, etc.)
        - Email thread garbage (quoted text, signatures)
        - Various formats (comma-separated, bullet points, prose)
        
        Returns:
            Extracted interests string, or None if extraction fails
        """
        import os
        try:
            import openai
            
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                logger.warning(f"[LLM_EXTRACT][{corr_id}] No OpenAI API key - falling back to pattern matching")
                return None
            
            client = openai.OpenAI(api_key=api_key)
            
            prompt = f"""You are extracting research interests from an email reply.

The email is a reply to a question asking "What research topics interest you?"
The user's NEW reply content should contain their research interests.
Ignore any quoted text, email headers, signatures, or previous messages in the thread.

EMAIL CONTENT:
---
{email_body[:3000]}
---

TASK: Extract ONLY the research interests/topics the user mentioned in their NEW reply.
- Look for topics like: machine learning, NLP, computer vision, transformers, etc.
- Ignore reply markers in ANY language (e.g., "On ... wrote:", "בתאריך...", "El día...", etc.)
- Ignore quoted text (lines starting with ">")
- Ignore email signatures and previous conversation
- Return ONLY the comma-separated list of research topics

If you find research interests, return them as a clean comma-separated list.
If you cannot find any research interests in the NEW content, respond with exactly: NO_INTERESTS_FOUND

EXTRACTED INTERESTS:"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
            )
            
            result = response.choices[0].message.content.strip()
            
            if result == "NO_INTERESTS_FOUND" or not result:
                logger.info(f"[LLM_EXTRACT][{corr_id}] LLM found no interests in email")
                return None
            
            logger.info(f"[LLM_EXTRACT][{corr_id}] LLM extracted interests: {result[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"[LLM_EXTRACT][{corr_id}] Error using LLM extraction: {e}")
            return None
    
    def _extract_new_content_from_reply(self, body: str) -> str:
        """
        Extract only the new content from a reply email, removing quoted text.
        
        This removes:
        - Lines starting with ">" (quoted text)
        - Content after "On ... wrote:" patterns
        - Content after "---" or "___" separators
        - Email signatures
        """
        if not body:
            return ""
        
        lines = body.split('\n')
        new_lines = []
        
        for line in lines:
            # Stop at common reply markers
            line_lower = line.lower().strip()
            
            # End markers
            if line_lower.startswith('on ') and ' wrote:' in line_lower:
                break
            if line.strip().startswith('>'):
                continue  # Skip quoted lines
            if line.strip() in ('---', '___', '---', '- - -', '_ _ _'):
                break
            if 'researchpulse' in line_lower and ('best regards' in line_lower or 'welcome' in line_lower):
                # Likely our previous email
                break
            if line_lower.startswith('from:') or line_lower.startswith('sent:'):
                break
            
            new_lines.append(line)
        
        return '\n'.join(new_lines).strip()
    
    def _extract_name_from_body(self, body: str) -> Optional[str]:
        """Extract name from email body using common patterns."""
        import re
        
        # Title prefix pattern (Dr., Prof., Mr., Mrs., Ms., etc.)
        title_prefix = r'(?:Dr\.?|Prof\.?|Mr\.?|Mrs\.?|Ms\.?|Professor)?\s*'
        # Name word pattern - capitalized word
        name_word = r'[A-Z][a-zA-Z\'\-]+'
        # Full name - title (optional) + 1-4 name words
        full_name = title_prefix + name_word + r'(?:\s+' + name_word + r'){0,3}'
        
        patterns = [
            r'my\s+name\s+is\s+(' + full_name + r')',  # My name is Dr. John Snow
            r'i\s+am\s+(' + full_name + r')',  # I am Dr. John Snow
            r'name[:\s]+(' + full_name + r')',  # name: Dr. John Snow
        ]
        
        # Common words to filter out
        common_words = {
            'the', 'i', 'we', 'you', 'me', 'my', 'your', 'please', 'just', 'add',
            'thanks', 'cheers', 'regards', 'best', 'sincerely', 'hello', 'hi',
            'interested', 'want', 'would', 'like', 'list', 'code', 'update'
        }
        
        for pattern in patterns:
            match = re.search(pattern, body, re.IGNORECASE | re.MULTILINE)
            if match:
                name = match.group(1).strip()
                # Basic validation: 2-30 chars, not common words
                if (2 <= len(name) <= 30 and 
                    name.lower() not in common_words and
                    not all(w.lower() in common_words for w in name.split())):
                    return name
        return None
    
    def _extract_interests_from_body(self, body: str) -> Tuple[str, bool]:
        """
        Extract research interests from email body.
        
        Returns:
            Tuple of (interests_text, is_meaningful)
            - interests_text: Raw text that may contain interests
            - is_meaningful: True if we found actual research topics
        """
        if not body:
            return ("", False)
        
        # Research topic keywords that indicate meaningful interests
        # NOTE: Short keywords (<=4 chars) require word boundary matching to avoid false positives
        # e.g., 'ai' should not match 'said', 'paid', 'wait'
        long_keywords = [
            'machine learning', 'deep learning', 'neural network',
            'natural language', 'computer vision', 'reinforcement learning',
            'transformer', 'large language', 'artificial intelligence',
            'robotics', 'optimization', 'recommendation',
            'classification', 'detection', 'segmentation', 'generation',
            'representation learning', 'self-supervised', 'contrastive',
            'attention', 'diffusion', 'generative',
            'multimodal', 'vision-language', 'speech', 'audio',
            'knowledge graph', 'reasoning', 'question answering',
            'summarization', 'translation', 'sentiment', 'information retrieval',
            'physics', 'quantum', 'biology', 'chemistry', 'mathematics',
            'statistics', 'economics', 'finance', 'healthcare', 'medical',
        ]
        
        # Short keywords need word boundary matching to avoid false positives
        # e.g., 'ai' could match 'said', 'paid', etc. if we just do 'ai' in text
        short_keywords = ['nlp', 'llm', 'ai', 'bert', 'gpt', 'qa', 'image', 'graph']
        
        body_lower = body.lower()
        
        # Find all interest keywords present
        # For long keywords, simple substring match is fine (they're specific enough)
        found_interests = [kw for kw in long_keywords if kw in body_lower]
        
        # For short keywords, use word boundary matching to avoid false positives
        for kw in short_keywords:
            # Match the keyword only as a complete word (with word boundaries)
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, body_lower):
                found_interests.append(kw)
        
        if found_interests:
            # Clean the body to extract just the interests portion
            # Remove common join request phrases
            clean_body = body
            noise_phrases = [
                'add me to the colleagues list',
                'add me to your list',
                'code:', 'join code',
                'my name is',
            ]
            for phrase in noise_phrases:
                clean_body = re.sub(phrase, '', clean_body, flags=re.IGNORECASE)
            
            # Return cleaned text and indicate it has meaningful interests
            return (clean_body.strip()[:500], True)
        
        return (body[:500], False)
    
    def _add_colleague_from_email(self, signup: Dict[str, Any]) -> Tuple[Optional[str], str]:
        """
        Add a new colleague from an email signup with reasoning-based onboarding.
        
        Required fields for complete colleague record:
        - email (always present from sender)
        - name (extracted or from header)
        - interests (at least 1 keyword/topic)
        
        Returns:
            Tuple of (colleague_id or None, onboarding_status: str)
            onboarding_status: 'complete', 'needs_interests', 'needs_name', 'pending'
        """
        from src.tools.arxiv_categories import derive_arxiv_categories_from_interests
        
        from_email = signup.get("from_email", "")
        from_name = signup.get("from_name", "")
        body = signup.get("body_text", "")
        thread_id = signup.get("thread_id", "")
        
        logger.info(f"[JOIN_LOGIC] Starting colleague add: email={from_email}, header_name={from_name}")
        
        # PRIMARY: Use LLM to extract name and interests intelligently
        llm_result = self._extract_colleague_info_with_llm(body, f"join_{from_email[:20]}")
        llm_name = llm_result.get("name")
        llm_interests = llm_result.get("interests")
        
        logger.info(f"[JOIN_LOGIC] LLM extraction: name={llm_name}, interests={llm_interests[:50] if llm_interests else None}...")
        
        # Step 1: Extract name - prefer LLM, fallback to patterns, then header
        if llm_name:
            name = llm_name
            logger.info(f"[JOIN_LOGIC] Using LLM-extracted name: {name}")
        else:
            extracted_name = self._extract_name_from_body(body)
            if extracted_name:
                name = extracted_name
                logger.info(f"[JOIN_LOGIC] Using pattern-extracted name: {name}")
            elif from_name and from_name.lower() not in ['unknown', '']:
                name = from_name
                logger.info(f"[JOIN_LOGIC] Using name from header: {name}")
            else:
                name = ""
                logger.info(f"[JOIN_LOGIC] No name found")
        
        # Step 2: Extract interests - prefer LLM, fallback to patterns
        if llm_interests:
            interests_text = llm_interests  # Already cleaned by LLM extraction
            has_meaningful_interests = True
            logger.info(f"[JOIN_LOGIC] Using LLM-extracted interests: {interests_text[:100]}")
        else:
            interests_text, has_meaningful_interests = self._extract_interests_from_body(body)
            # ALWAYS clean the interests to remove any garbage
            interests_text = self._clean_interests_text(interests_text)
            logger.info(f"[JOIN_LOGIC] Pattern interests extraction: has_meaningful={has_meaningful_interests}, text_len={len(interests_text)}")
        
        # Step 3: Derive arXiv categories from interests
        derived_categories = {}
        if has_meaningful_interests:
            try:
                derived_categories = derive_arxiv_categories_from_interests(interests_text)
                logger.info(f"[JOIN_LOGIC] Derived categories: primary={derived_categories.get('primary', [])}, "
                           f"secondary={derived_categories.get('secondary', [])}")
            except Exception as e:
                logger.warning(f"[JOIN_LOGIC] Failed to derive categories: {e}")
        
        # Step 4: Determine onboarding status based on what's missing
        missing_fields = []
        if not name:
            missing_fields.append("name")
        if not has_meaningful_interests:
            missing_fields.append("interests")
        
        if not missing_fields:
            onboarding_status = "complete"
        elif "interests" in missing_fields and "name" in missing_fields:
            onboarding_status = "pending"  # Missing both
        elif "interests" in missing_fields:
            onboarding_status = "needs_interests"
        else:
            onboarding_status = "needs_name"
        
        logger.info(f"[JOIN_LOGIC] Status determination: missing={missing_fields}, status={onboarding_status}")
        
        # Step 5: Check if colleague already exists
        existing_colleagues = self.store.list_colleagues(self.user_id)
        for colleague in existing_colleagues:
            if colleague.get("email", "").lower() == from_email.lower():
                colleague_id = str(colleague.get("id", ""))
                existing_status = colleague.get("onboarding_status", "")
                existing_interests = colleague.get("interests") or colleague.get("research_interests") or ""
                existing_name = colleague.get("name") or ""
                
                # Determine the ACTUAL status based on what's really missing
                # Don't trust the stored status - recalculate based on actual data
                has_interests_now = has_meaningful_interests or bool(existing_interests.strip())
                has_name_now = bool(name) or (bool(existing_name.strip()) and existing_name.lower() != "pending")
                
                actual_missing = []
                if not has_name_now:
                    actual_missing.append("name")
                if not has_interests_now:
                    actual_missing.append("interests")
                
                # Compute the real onboarding status based on actual data
                if not actual_missing:
                    computed_status = "complete"
                elif "interests" in actual_missing and "name" in actual_missing:
                    computed_status = "pending"
                elif "interests" in actual_missing:
                    computed_status = "needs_interests"
                else:
                    computed_status = "needs_name"
                
                logger.info(f"[JOIN_LOGIC] Existing colleague found: id={colleague_id}, stored_status={existing_status}, "
                           f"has_interests={has_interests_now}, has_name={has_name_now}, computed_status={computed_status}")
                
                # Update colleague with any new info we have
                update_data = {}
                if name and (not existing_name or existing_name.lower() == "pending"):
                    update_data["name"] = name
                if has_meaningful_interests and not existing_interests.strip():
                    update_data["interests"] = interests_text
                    update_data["research_interests"] = interests_text
                    update_data["derived_arxiv_categories"] = derived_categories
                
                # Update status and enabled flag based on computed status
                if computed_status != existing_status:
                    update_data["onboarding_status"] = computed_status
                if computed_status == "complete":
                    update_data["enabled"] = True
                
                if update_data:
                    logger.info(f"[JOIN_LOGIC] Updating colleague with: {list(update_data.keys())}")
                    self.store.update_colleague(UUID(colleague_id), update_data)
                
                return (colleague_id, computed_status)
        
        # Step 6: Create new colleague record
        try:
            colleague_data = {
                "name": name or "Pending",  # Use placeholder if no name
                "email": from_email,
                "research_interests": interests_text if has_meaningful_interests else "",
                "interests": interests_text if has_meaningful_interests else "",
                "derived_arxiv_categories": derived_categories,
                "keywords": [],
                "categories": derived_categories.get("primary", []) + derived_categories.get("secondary", []),
                "added_by": "email",
                "auto_send_emails": True,
                "enabled": onboarding_status == "complete",  # Only enable if complete
                "onboarding_status": onboarding_status,
                "onboarding_thread_id": thread_id,
                "join_verified": True,  # Join code was verified at this point
                "notes": f"Added via email join on {datetime.utcnow().isoformat()}. Status: {onboarding_status}",
            }
            
            new_colleague = self.store.create_colleague(self.user_id, colleague_data)
            colleague_id = str(new_colleague.get("id", "")) if new_colleague else None
            
            if colleague_id:
                logger.info(f"[DB_WRITE] Created colleague: id={colleague_id}, status={onboarding_status}")
            
            return (colleague_id, onboarding_status)
            
        except Exception as e:
            logger.error(f"[DB_WRITE] Error creating colleague: {e}", exc_info=True)
            return (None, "error")
    
    def _add_colleague_from_email_with_analysis(
        self, 
        signup: Dict[str, Any], 
        extracted_name: Optional[str], 
        extracted_interests: Optional[str],
        override_status: Optional[str] = None
    ) -> Tuple[Optional[str], str]:
        """
        Add a new colleague using pre-analyzed data from LLM reasoning.
        
        This is similar to _add_colleague_from_email but uses the already-extracted
        name and interests from the reasoning analysis instead of re-extracting.
        
        Args:
            signup: The signup email data
            extracted_name: Name extracted by reasoning (or None)
            extracted_interests: Interests extracted by reasoning (or None)
            override_status: If set, use this status instead of computing it
            
        Returns:
            Tuple of (colleague_id or None, onboarding_status: str)
        """
        from src.tools.arxiv_categories import derive_arxiv_categories_from_interests
        
        from_email = signup.get("from_email", "")
        from_name = signup.get("from_name", "")
        thread_id = signup.get("thread_id", "")
        
        logger.info(f"[JOIN_LOGIC_V2] Starting colleague add: email={from_email}, "
                   f"extracted_name={extracted_name}, has_interests={bool(extracted_interests)}")
        
        # Use extracted data
        name = extracted_name or from_name or ""
        interests_text = extracted_interests or ""
        has_meaningful_interests = bool(interests_text.strip())
        
        # Derive arXiv categories from interests
        derived_categories = {}
        if has_meaningful_interests:
            try:
                derived_categories = derive_arxiv_categories_from_interests(interests_text)
                logger.info(f"[JOIN_LOGIC_V2] Derived categories: primary={derived_categories.get('primary', [])}")
            except Exception as e:
                logger.warning(f"[JOIN_LOGIC_V2] Failed to derive categories: {e}")
        
        # Determine onboarding status
        if override_status:
            onboarding_status = override_status
            logger.info(f"[JOIN_LOGIC_V2] Using override status: {override_status}")
        else:
            missing_fields = []
            if not name:
                missing_fields.append("name")
            if not has_meaningful_interests:
                missing_fields.append("interests")
            
            if not missing_fields:
                onboarding_status = "complete"
            elif "interests" in missing_fields and "name" in missing_fields:
                onboarding_status = "pending"
            elif "interests" in missing_fields:
                onboarding_status = "needs_interests"
            else:
                onboarding_status = "needs_name"
        
        logger.info(f"[JOIN_LOGIC_V2] Status: {onboarding_status}")
        
        # Check if colleague already exists
        existing_colleagues = self.store.list_colleagues(self.user_id)
        for colleague in existing_colleagues:
            if colleague.get("email", "").lower() == from_email.lower():
                colleague_id = str(colleague.get("id", ""))
                existing_interests = colleague.get("interests") or colleague.get("research_interests") or ""
                existing_name = colleague.get("name") or ""
                
                # Update colleague with any new info
                update_data = {}
                if name and (not existing_name or existing_name.lower() == "pending"):
                    update_data["name"] = name
                if has_meaningful_interests and not existing_interests.strip():
                    update_data["interests"] = interests_text
                    update_data["research_interests"] = interests_text
                    update_data["derived_arxiv_categories"] = derived_categories
                    update_data["categories"] = (
                        derived_categories.get("primary", []) + 
                        derived_categories.get("secondary", [])
                    )
                
                update_data["onboarding_status"] = onboarding_status
                if onboarding_status == "complete":
                    update_data["enabled"] = True
                
                if update_data:
                    self.store.update_colleague(UUID(colleague_id), update_data)
                    logger.info(f"[JOIN_LOGIC_V2] Updated existing colleague: {colleague_id}")
                
                return (colleague_id, onboarding_status)
        
        # Create new colleague record
        try:
            colleague_data = {
                "name": name or "Pending",
                "email": from_email,
                "research_interests": interests_text,
                "interests": interests_text,
                "derived_arxiv_categories": derived_categories,
                "keywords": [],
                "categories": derived_categories.get("primary", []) + derived_categories.get("secondary", []),
                "added_by": "email",
                "auto_send_emails": True,
                "enabled": onboarding_status == "complete",
                "onboarding_status": onboarding_status,
                "onboarding_thread_id": thread_id,
                "join_verified": True,
                "notes": f"Added via email join on {datetime.utcnow().isoformat()}. Status: {onboarding_status}",
            }
            
            new_colleague = self.store.create_colleague(self.user_id, colleague_data)
            colleague_id = str(new_colleague.get("id", "")) if new_colleague else None
            
            if colleague_id:
                logger.info(f"[DB_WRITE_V2] Created colleague: id={colleague_id}, status={onboarding_status}")
            
            return (colleague_id, onboarding_status)
            
        except Exception as e:
            logger.error(f"[DB_WRITE_V2] Error creating colleague: {e}", exc_info=True)
            return (None, "error")
    
    def _send_join_code_required_reply(self, to_email: str, name: str):
        """Send a reply requesting the join code with HTML template."""
        from .email_templates import render_onboarding_instruction_email
        subject, plain, html = render_onboarding_instruction_email(name or "", reason="missing_code")
        self._send_join_reply(
            to_email=to_email,
            subject=subject,
            body=plain,
            email_type=EmailType.COLLEAGUE_JOIN,
            html_body=html,
        )
    
    def _send_invalid_code_reply(self, to_email: str, name: str):
        """Send a reply indicating the join code was invalid with HTML template."""
        from .email_templates import render_onboarding_instruction_email
        subject, plain, html = render_onboarding_instruction_email(name or "", reason="invalid_code")
        self._send_join_reply(
            to_email=to_email,
            subject=subject,
            body=plain,
            email_type=EmailType.COLLEAGUE_JOIN,
            html_body=html,
        )
    
    def _send_not_configured_reply(self, to_email: str, name: str):
        """Send a reply when no join code is configured yet."""
        from .email_templates import render_onboarding_instruction_email
        subject, plain, html = render_onboarding_instruction_email(name or "", reason="not_configured")
        self._send_join_reply(
            to_email=to_email,
            subject=subject,
            body=plain,
            email_type=EmailType.COLLEAGUE_JOIN,
            html_body=html,
        )
    
    def _send_join_success_reply(self, to_email: str, name: str, interests: list = None):
        """Send a confirmation that the colleague was added, with HTML template."""
        from .email_templates import render_colleague_confirmation_email
        from .colleague_tokens import generate_remove_url, generate_update_url
        import os
        base_url = os.getenv("RESEARCHPULSE_BASE_URL", "https://researchpulse.app")
        owner_id = str(self.user_id)
        remove_url = generate_remove_url(base_url, owner_id, to_email)
        update_url = generate_update_url(base_url, owner_id, to_email)
        # Normalize interests to a list (may come as a comma-separated string)
        if isinstance(interests, str):
            interests_list = [i.strip() for i in interests.split(",") if i.strip()]
        else:
            interests_list = interests or []
        subject, plain, html = render_colleague_confirmation_email(
            colleague_name=name or "",
            interests=interests_list,
            remove_url=remove_url,
            update_url=update_url,
        )
        self._send_join_reply(
            to_email=to_email,
            subject=subject,
            body=plain,
            email_type=EmailType.WELCOME,
            html_body=html,
        )
    
    def _send_clarify_intent_reply(self, to_email: str, name: str, custom_message: str = None, thread_id: str = ""):
        """
        Send a clarifying reply when the signup intent is unclear.
        
        This is used when someone sends a valid code but with unclear content
        (e.g., just "GOOD MORNING" with a code).
        """
        logger.info(f"[REPLY] Sending clarify intent reply to {to_email}")
        from .email_templates import render_clarify_intent_email
        subject, plain, html = render_clarify_intent_email(name or "", custom_message=custom_message)
        self._send_join_reply(
            to_email=to_email,
            subject=subject,
            body=plain,
            email_type=EmailType.COLLEAGUE_JOIN,
            html_body=html,
        )
    
    def _send_onboarding_questions_reply(self, to_email: str, name: str, thread_id: str = ""):
        """Send onboarding questions to collect both name and research interests."""
        logger.info(f"[REPLY] Sending full onboarding questions to {to_email}")
        from .email_templates import render_onboarding_questions_email
        subject, plain, html = render_onboarding_questions_email(name or "")
        self._send_join_reply(
            to_email=to_email,
            subject=subject,
            body=plain,
            email_type=EmailType.ONBOARDING,
            html_body=html,
        )
    
    def _send_onboarding_interests_reply(self, to_email: str, name: str, thread_id: str = ""):
        """Send onboarding question to collect only research interests."""
        logger.info(f"[REPLY] Sending interests request to {to_email}")
        from .email_templates import render_onboarding_interests_email
        subject, plain, html = render_onboarding_interests_email(name or "")
        self._send_join_reply(
            to_email=to_email,
            subject=subject,
            body=plain,
            email_type=EmailType.ONBOARDING,
            html_body=html,
        )
    
    def _send_onboarding_name_reply(self, to_email: str, name: str, thread_id: str = ""):
        """Send onboarding question to collect only name."""
        logger.info(f"[REPLY] Sending name request to {to_email}")
        from .email_templates import render_onboarding_name_email
        subject, plain, html = render_onboarding_name_email(name or "")
        self._send_join_reply(
            to_email=to_email,
            subject=subject,
            body=plain,
            email_type=EmailType.ONBOARDING,
            html_body=html,
        )
    
    def _send_join_reply(self, to_email: str, subject: str, body: str, email_type: EmailType = EmailType.ONBOARDING, html_body: str = ""):
        """Send an email reply for join flow using unified outbound module."""
        try:
            # Use unified outbound email module for consistent sender name and tagging
            success, message_id, error = send_outbound_email(
                to_email=to_email,
                subject=subject,
                body=body,
                email_type=email_type,
                html_body=html_body or None,
            )
            
            if success:
                logger.info(f"[INBOUND] Sent reply email to {to_email} (message_id={message_id})")
            else:
                logger.warning(f"[INBOUND] Failed to send reply email to {to_email}: {error}")
            
        except Exception as e:
            logger.error(f"[INBOUND] Error sending reply email: {e}")


# =============================================================================
# Convenience function for scheduled polling
# =============================================================================

async def run_inbox_check(store, user_id: str) -> Dict[str, Any]:
    """
    Run a single inbox check cycle.
    
    This is called by the background scheduler.
    """
    logger.info(f"[SCHEDULER] Running inbox check for user {user_id}")
    
    processor = InboundEmailProcessor(store, user_id)
    results = processor.process_all(since_hours=48)
    
    return results


# =============================================================================
# Inbox Diagnostics Mode
# =============================================================================

def run_inbox_diagnostics(since_hours: int = 48) -> Dict[str, Any]:
    """
    Run inbox diagnostics - fetch recent emails and analyze them.
    
    This is a debug tool to verify the email pipeline is working.
    Prints a table of recent emails with detected intent.
    
    Returns:
        Dict with diagnostic results
    """
    import imaplib
    import email as email_module
    from datetime import datetime, timedelta
    
    from .email_poller import (
        get_imap_config,
        decode_email_subject,
        get_email_body,
        is_colleague_signup_email,
        is_calendar_invite_reply,
        has_join_code_pattern,
    )
    
    logger.info("=" * 70)
    logger.info("[DIAGNOSTICS] Starting inbox diagnostics")
    logger.info("=" * 70)
    
    config = get_imap_config()
    
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "config_valid": bool(config["user"] and config["password"]),
        "imap_host": config["host"],
        "emails_scanned": 0,
        "emails": [],
        "errors": [],
    }
    
    if not config["user"] or not config["password"]:
        logger.error("[DIAGNOSTICS] IMAP credentials NOT configured (SMTP_USER/SMTP_PASSWORD)")
        results["errors"].append("IMAP credentials not configured")
        return results
    
    try:
        # Connect
        logger.info(f"[DIAGNOSTICS] Connecting to {config['host']}:{config['port']}")
        mail = imaplib.IMAP4_SSL(config["host"], config["port"])
        mail.login(config["user"], config["password"])
        logger.info("[DIAGNOSTICS] Login successful")
        
        mail.select("INBOX")
        
        # Search
        since_date = (datetime.now() - timedelta(hours=since_hours)).strftime("%d-%b-%Y")
        search_criteria = f'(SINCE "{since_date}")'
        logger.info(f"[DIAGNOSTICS] Query: {search_criteria}")
        
        result, data = mail.search(None, search_criteria)
        if result != "OK":
            logger.error(f"[DIAGNOSTICS] Search failed: {result}")
            results["errors"].append(f"Search failed: {result}")
            return results
        
        email_ids = data[0].split()
        # Get last 20
        email_ids = email_ids[-20:] if len(email_ids) > 20 else email_ids
        
        logger.info(f"[DIAGNOSTICS] Processing {len(email_ids)} most recent emails")
        logger.info("-" * 70)
        logger.info(f"{'MsgID':<15} | {'From':<25} | {'Intent':<15} | {'Subject':<30}")
        logger.info("-" * 70)
        
        for email_id in email_ids:
            try:
                result, msg_data = mail.fetch(email_id, "(RFC822)")
                if result != "OK":
                    continue
                
                raw_email = msg_data[0][1]
                msg = email_module.message_from_bytes(raw_email)
                
                message_id = msg.get("Message-ID", "")[:15]
                subject = decode_email_subject(msg.get("Subject", ""))
                from_addr = msg.get("From", "")
                date_str = msg.get("Date", "")
                
                # Parse from
                import re
                from_match = re.search(r'[\w\.-]+@[\w\.-]+', from_addr)
                from_email = from_match.group(0)[:25] if from_match else from_addr[:25]
                
                body = get_email_body(msg)
                snippet = body[:50].replace('\n', ' ') if body else ""
                
                # Detect intent
                has_code = has_join_code_pattern((subject + " " + body).lower())
                is_signup = is_colleague_signup_email(subject, body)
                is_reschedule = is_calendar_invite_reply(subject)
                
                if is_signup:
                    intent = "JOIN_REQUEST" + (" +CODE" if has_code else "")
                elif is_reschedule:
                    intent = "RESCHEDULE"
                else:
                    intent = "UNKNOWN"
                
                logger.info(f"{message_id:<15} | {from_email:<25} | {intent:<15} | {subject[:30]:<30}")
                
                results["emails"].append({
                    "message_id": message_id,
                    "from": from_email,
                    "subject": subject[:50],
                    "date": date_str,
                    "snippet": snippet,
                    "intent": intent,
                    "has_code": has_code,
                })
                
            except Exception as e:
                logger.error(f"[DIAGNOSTICS] Error processing email: {e}")
        
        results["emails_scanned"] = len(results["emails"])
        mail.logout()
        
        logger.info("-" * 70)
        logger.info(f"[DIAGNOSTICS] Complete: {results['emails_scanned']} emails analyzed")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"[DIAGNOSTICS] Error: {e}", exc_info=True)
        results["errors"].append(str(e))
    
    return results
