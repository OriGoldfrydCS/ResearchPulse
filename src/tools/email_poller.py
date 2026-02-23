"""Email poller for fetching incoming replies via IMAP.

This module polls the configured Gmail inbox for replies to calendar invitations
and processes them to reschedule events based on user requests.
"""

import os
import imaplib
import email
from email.header import decode_header
from email.message import Message
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from uuid import uuid4
import re
import logging

logger = logging.getLogger(__name__)


def get_imap_config() -> Dict[str, Any]:
    """Get IMAP configuration from environment variables."""
    return {
        "host": os.getenv("IMAP_HOST", "imap.gmail.com"),
        "port": int(os.getenv("IMAP_PORT", "993")),
        "user": os.getenv("SMTP_USER", ""),  # Reuse SMTP credentials
        "password": os.getenv("SMTP_PASSWORD", ""),
    }


def decode_email_subject(subject: str) -> str:
    """Decode email subject handling various encodings."""
    if subject is None:
        return ""
    decoded_parts = decode_header(subject)
    result = []
    for part, encoding in decoded_parts:
        if isinstance(part, bytes):
            result.append(part.decode(encoding or 'utf-8', errors='replace'))
        else:
            result.append(part)
    return ''.join(result)


def _strip_html_tags(html: str) -> str:
    """Convert HTML to plain text by stripping tags and decoding entities."""
    import html as html_module
    # Replace <br>, <br/>, <p>, <div> with newlines for readability
    text = re.sub(r'<br\s*/?\s*>', '\n', html, flags=re.IGNORECASE)
    text = re.sub(r'</(p|div|tr|li)>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<(p|div|tr|li|h[1-6])[^>]*>', '\n', text, flags=re.IGNORECASE)
    # Remove all remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Decode HTML entities (&amp; &lt; &#39; etc.)
    text = html_module.unescape(text)
    # Collapse excessive whitespace/newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def get_email_body(msg: Message) -> str:
    """Extract plain text body from email message.
    
    Prefers text/plain part; falls back to text/html (stripped of tags)
    when no plain text part is available.
    """
    body = ""
    html_body = ""
    
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))
            
            # Skip attachments
            if "attachment" in content_disposition:
                continue
                
            if content_type == "text/plain" and not body:
                try:
                    payload = part.get_payload(decode=True)
                    charset = part.get_content_charset() or 'utf-8'
                    body = payload.decode(charset, errors='replace')
                except Exception as e:
                    logger.warning(f"Error decoding text/plain part: {e}")
            elif content_type == "text/html" and not html_body:
                try:
                    payload = part.get_payload(decode=True)
                    charset = part.get_content_charset() or 'utf-8'
                    html_body = payload.decode(charset, errors='replace')
                except Exception as e:
                    logger.warning(f"Error decoding text/html part: {e}")
        
        # Prefer plain text; fall back to stripped HTML
        if not body and html_body:
            logger.info("[EMAIL_BODY] No text/plain part found — falling back to stripped HTML")
            body = _strip_html_tags(html_body)
    else:
        try:
            payload = msg.get_payload(decode=True)
            charset = msg.get_content_charset() or 'utf-8'
            raw = payload.decode(charset, errors='replace')
            content_type = msg.get_content_type()
            if content_type == "text/html":
                logger.info("[EMAIL_BODY] Non-multipart HTML email — stripping tags")
                body = _strip_html_tags(raw)
            else:
                body = raw
        except Exception as e:
            logger.warning(f"Error decoding email body: {e}")
    
    return body


def extract_original_message_id(msg: Message) -> Optional[str]:
    """Extract the In-Reply-To or References header to find original message."""
    in_reply_to = msg.get("In-Reply-To", "").strip()
    if in_reply_to:
        # Clean up message ID
        in_reply_to = in_reply_to.strip("<>")
        return in_reply_to
    
    references = msg.get("References", "").strip()
    if references:
        # Get the first (original) message ID from references
        refs = references.split()
        if refs:
            return refs[0].strip("<>")
    
    return None


def is_calendar_invite_reply(subject: str) -> bool:
    """Check if this email appears to be a reply to a calendar invite."""
    subject_lower = subject.lower()
    
    # Check for ResearchPulse-related keywords
    keywords = [
        "researchpulse",
        "reading reminder",
        "read:",
        "research papers",
        "scheduled reading"
    ]
    
    # Check for Re: prefix indicating a reply
    is_reply = subject_lower.startswith("re:") or subject_lower.startswith("re :")
    
    return is_reply and any(kw in subject_lower for kw in keywords)


def fetch_recent_replies(since_hours: int = 24) -> List[Dict[str, Any]]:
    """Fetch recent email replies from the inbox.
    
    Args:
        since_hours: Look for emails from the last N hours
        
    Returns:
        List of parsed email reply data
    """
    import socket
    import time
    import traceback
    
    IMAP_SOCKET_TIMEOUT = 20  # seconds
    
    config = get_imap_config()
    cycle_start = time.time()
    
    logger.info(f"[EMAIL_POLLER] ===== START fetch_recent_replies (since_hours={since_hours}) =====")
    logger.debug(f"[EMAIL_POLLER] IMAP host={config['host']}, port={config['port']}, user={'configured' if config['user'] else 'NOT SET'}")
    
    if not config["user"] or not config["password"]:
        logger.warning("[EMAIL_POLLER] IMAP credentials not configured - SMTP_USER and SMTP_PASSWORD required")
        return []
    
    replies = []
    mail = None
    
    try:
        # Set socket timeout for all IMAP operations
        socket.setdefaulttimeout(IMAP_SOCKET_TIMEOUT)
        
        # Connect to IMAP server
        step_start = time.time()
        logger.debug(f"[EMAIL_POLLER] CONNECT start - host={config['host']}, port={config['port']}")
        mail = imaplib.IMAP4_SSL(config["host"], config["port"])
        logger.debug(f"[EMAIL_POLLER] CONNECT ok - elapsed={time.time() - step_start:.2f}s")
        
        # Login
        step_start = time.time()
        logger.debug("[EMAIL_POLLER] LOGIN start")
        mail.login(config["user"], config["password"])
        logger.info(f"[EMAIL_POLLER] IMAP login successful - elapsed={time.time() - step_start:.2f}s")
        
        # Select inbox
        step_start = time.time()
        mail.select("INBOX")
        logger.debug(f"[EMAIL_POLLER] INBOX selected - elapsed={time.time() - step_start:.2f}s")
        
        # Search for recent emails
        since_date = (datetime.now() - timedelta(hours=since_hours)).strftime("%d-%b-%Y")
        search_criteria = f'(SINCE "{since_date}")'
        logger.debug(f"[EMAIL_POLLER] Searching with criteria: {search_criteria}")
        
        result, data = mail.search(None, search_criteria)
        
        if result != "OK":
            logger.error(f"[EMAIL_POLLER] IMAP search failed: {result}")
            return []
        
        email_ids = data[0].split() if data[0] else []
        logger.info(f"Found {len(email_ids)} emails in last {since_hours} hours")
        
        for email_id in email_ids:
            try:
                result, msg_data = mail.fetch(email_id, "(RFC822)")
                if result != "OK":
                    continue
                
                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)
                
                # Get headers
                message_id = msg.get("Message-ID", "").strip("<>")
                subject = decode_email_subject(msg.get("Subject", ""))
                from_addr = msg.get("From", "")
                date_str = msg.get("Date", "")
                
                # Check if this looks like a reply to our calendar invite
                if not is_calendar_invite_reply(subject):
                    continue
                
                # Get the original message ID this is replying to
                in_reply_to = extract_original_message_id(msg)
                if not in_reply_to:
                    logger.debug(f"Reply without In-Reply-To header: {subject}")
                    continue
                
                # Get body
                body = get_email_body(msg)
                
                # Parse from address
                from_match = re.search(r'[\w\.-]+@[\w\.-]+', from_addr)
                from_email = from_match.group(0) if from_match else from_addr
                
                reply_data = {
                    "message_id": message_id or str(uuid4()),
                    "in_reply_to": in_reply_to,
                    "from_email": from_email,
                    "subject": subject,
                    "body_text": body,
                    "received_at": date_str,
                }
                
                replies.append(reply_data)
                logger.info(f"Found calendar invite reply: {subject[:50]}...")
                
            except socket.timeout:
                logger.error(f"[EMAIL_POLLER] Fetch timeout for email {email_id}")
                continue
            except Exception as e:
                logger.error(f"Error processing email {email_id}: {e}")
                continue
        
        mail.logout()
        total_elapsed = time.time() - cycle_start
        logger.info(f"[EMAIL_POLLER] ===== END SUCCESS: found={len(replies)} replies, total_time={total_elapsed:.2f}s =====")
        
    except socket.timeout as e:
        logger.error(f"[EMAIL_POLLER] Socket timeout: {e}")
        logger.error(f"[EMAIL_POLLER] Stack trace:\n{traceback.format_exc()}")
        if mail:
            try:
                mail.logout()
            except:
                pass
    except imaplib.IMAP4.error as e:
        logger.error(f"IMAP error: {e}")
        if mail:
            try:
                mail.logout()
            except:
                pass
    except Exception as e:
        logger.error(f"Error fetching emails: {e}")
        logger.error(f"[EMAIL_POLLER] Stack trace:\n{traceback.format_exc()}")
        if mail:
            try:
                mail.logout()
            except:
                pass
    finally:
        # Reset socket timeout to default
        socket.setdefaulttimeout(None)
    
    return replies


async def poll_and_process_replies(store, user_id: str) -> Dict[str, Any]:
    """Poll for email replies and process them.
    
    Args:
        store: Database store instance
        user_id: User ID to associate replies with
        
    Returns:
        Summary of processing results
    """
    from src.agent.reply_parser import parse_reply, ReplyIntent
    from src.tools.calendar_invite_sender import send_reschedule_invite
    from uuid import UUID
    
    results = {
        "emails_found": 0,
        "replies_matched": 0,
        "events_rescheduled": 0,
        "errors": []
    }
    
    # Fetch recent replies
    replies = fetch_recent_replies(since_hours=48)
    results["emails_found"] = len(replies)
    
    for reply in replies:
        try:
            # Find the original calendar invite by message_id
            invite = store.get_calendar_invite_by_message_id(reply["in_reply_to"])
            
            if not invite:
                # Try to find by partial match in thread
                logger.debug(f"No invite found for message_id: {reply['in_reply_to']}")
                continue
            
            results["replies_matched"] += 1
            
            # Check if we already processed this reply
            existing = store.get_inbound_reply_by_message_id(reply["message_id"])
            if existing:
                logger.debug(f"Reply already processed: {reply['message_id']}")
                continue
            
            # Store the inbound reply
            inbound_reply = store.create_inbound_email_reply(
                user_id=UUID(user_id),
                original_invite_id=invite["id"],
                message_id=reply["message_id"],
                in_reply_to=reply["in_reply_to"],
                from_email=reply["from_email"],
                subject=reply["subject"],
                body_text=reply["body_text"],
            )
            
            # Parse the reply intent
            parse_result = parse_reply(reply["body_text"])
            
            # Update with parsed intent
            store.update_inbound_reply_processing(
                reply_id=inbound_reply["id"],
                intent=parse_result.intent.value,
                extracted_datetime=parse_result.extracted_datetime,
                extracted_datetime_text=parse_result.extracted_datetime_text,
                confidence=parse_result.confidence,
            )
            
            # If reschedule intent with valid datetime, reschedule the event
            if parse_result.intent == ReplyIntent.RESCHEDULE and parse_result.extracted_datetime:
                event = store.get_calendar_event(invite["calendar_event_id"])
                if event:
                    # Reschedule the event
                    new_event = store.reschedule_calendar_event(
                        event_id=event["id"],
                        new_start_time=parse_result.extracted_datetime,
                        reschedule_note="Rescheduled after user email reply"
                    )
                    
                    if new_event:
                        # Update reply with action taken
                        store.update_inbound_reply_processing(
                            reply_id=inbound_reply["id"],
                            action_taken="rescheduled",
                            new_event_id=new_event["id"],
                        )
                        
                        # Send new calendar invite
                        try:
                            user = store.get_user(UUID(user_id))
                            if user and user.get("email"):
                                # Get papers for description (handles UUID and arXiv-ID formats)
                                papers = []
                                for pid in (new_event.get("paper_ids") or []):
                                    paper = None
                                    try:
                                        paper = store.get_paper(UUID(pid))
                                    except (ValueError, AttributeError):
                                        pass
                                    if not paper:
                                        try:
                                            paper = store.get_paper_by_external_id("arxiv", pid)
                                        except Exception:
                                            pass
                                    if paper:
                                        papers.append(paper)
                                
                                ics_uid = new_event.get("ics_uid")
                                if ics_uid:
                                    send_reschedule_invite(
                                        user_email=user["email"],
                                        user_name=user.get("name", user["email"]),
                                        papers=papers,
                                        new_start_time=parse_result.extracted_datetime,
                                        duration_minutes=new_event.get("duration_minutes") or 30,
                                        ics_uid=ics_uid,
                                        sequence=(new_event.get("sequence_number") or 0) + 1,
                                        reschedule_reason="Rescheduled after user email reply",
                                    )
                                else:
                                    logger.warning("Event missing ics_uid, cannot send reschedule invite")
                        except Exception as e:
                            logger.error(f"Error sending reschedule invite: {e}")
                        
                        results["events_rescheduled"] += 1
                        logger.info(f"Rescheduled event to {parse_result.extracted_datetime}")
            
        except Exception as e:
            error_msg = f"Error processing reply {reply.get('message_id', 'unknown')}: {str(e)}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
    
    return results


def get_inbound_reply_by_message_id_wrapper(store, message_id: str):
    """Wrapper to check if reply already exists."""
    try:
        # This will be added to the store
        from sqlalchemy import select
        from src.db.orm_models import InboundEmailReply
        
        with store.Session() as session:
            stmt = select(InboundEmailReply).where(InboundEmailReply.message_id == message_id)
            result = session.execute(stmt).scalar_one_or_none()
            return result
    except Exception:
        return None


# =============================================================================
# Colleague Signup via Email
# =============================================================================

def has_join_code_pattern(text: str) -> bool:
    """Check if text contains a join code pattern.
    
    Patterns matched:
    - "code: 123456" or "code:123456"
    - "code=123456"
    - "join code 123456" or "join code: 123456"
    - "#123456"
    """
    import re
    patterns = [
        r'(?:join\s*)?code[:\s=]+[A-Za-z0-9-_]{4,32}',
        r'#[A-Za-z0-9-_]{4,32}',
        r'(?:my|the)\s+code\s+(?:is\s+)?[A-Za-z0-9-_]{4,32}',
        r'use\s+code[:\s]+[A-Za-z0-9-_]{4,32}',
    ]
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


# Automated / no-reply senders that should never be treated as colleague emails
_AUTOMATED_SENDER_PATTERNS = [
    "noreply", "no-reply", "no_reply",
    "mailer-daemon", "postmaster",
    "notifications@", "notification@",
    "donotreply", "do-not-reply", "do_not_reply",
    "bounce", "auto-reply", "autoreply",
    "daemon@", "system@",
]


def _is_automated_email(from_email: str, subject: str) -> bool:
    """Return True if the email appears to be automated / not from a real person."""
    addr = from_email.lower()
    subj = subject.lower()

    # Check sender patterns
    for pat in _AUTOMATED_SENDER_PATTERNS:
        if pat in addr:
            return True

    # Common automated subject indicators
    auto_subjects = [
        "auto-reply", "automatic reply", "out of office",
        "delivery status", "undeliverable", "returned mail",
        "newsletter", "promotion", "verify your email",
    ]
    for pat in auto_subjects:
        if pat in subj:
            return True

    return False


def is_colleague_signup_email(subject: str, body: str, from_email: str = "") -> bool:
    """Check if this email is a potential colleague interaction.
    
    Classification priority:
    1. Contains a join code pattern → definite signup attempt
    2. Mentions ResearchPulse + signup intent → definite signup
    3. Any non-automated personal email → potential interaction
       (the processing layer will check format and send instructions)
    
    Only filters OUT clearly automated / system-generated emails.
    """
    text = (subject + " " + body).lower()
    
    # FILTER OUT: Automated / no-reply / system emails
    if from_email and _is_automated_email(from_email, subject):
        logger.debug(f"[EMAIL_POLLER] Automated email from {from_email[:30]} - skipping")
        return False
    
    # PRIORITY 1: If email contains a join code pattern, it's a signup attempt
    if has_join_code_pattern(text):
        logger.info("[EMAIL_POLLER] Detected join code pattern in email - treating as signup")
        return True
    
    signup_keywords = [
        "subscribe",
        "sign me up",
        "sign up",
        "add me",
        "send me papers",
        "send me research",
        "interested in receiving",
        "want to receive",
        "research updates",
        "paper updates",
        "research papers",
        "share papers with me",
        "recommendation",
        "colleague list",
        "add to your",
        "colleagues list",
    ]
    
    researchpulse_keywords = [
        "researchpulse",
        "research pulse",
    ]
    
    # PRIORITY 2: Mentions ResearchPulse and has signup intent
    mentions_rp = any(kw in text for kw in researchpulse_keywords)
    has_signup_intent = any(kw in text for kw in signup_keywords)
    
    if mentions_rp and has_signup_intent:
        logger.info("[EMAIL_POLLER] Email matches ResearchPulse + signup intent")
        return True
    
    # PRIORITY 3: Treat any non-automated email as a potential interaction
    # The inbound processor will check the format and send instructions once
    logger.info(f"[EMAIL_POLLER] Non-automated email detected - treating as potential interaction")
    return True


def extract_name_from_email_header(from_header: str) -> str:
    """Extract displayable name from email From header."""
    # Format: "John Doe <john@example.com>" or just "john@example.com"
    if "<" in from_header:
        name_part = from_header.split("<")[0].strip()
        if name_part:
            # Remove quotes if present
            name_part = name_part.strip('"\'')
            return name_part
    
    # Fall back to email username
    email_match = re.search(r'([\w\.-]+)@', from_header)
    if email_match:
        return email_match.group(1).replace(".", " ").title()
    
    return "Unknown"


async def extract_research_interests_from_email(body: str) -> str:
    """Use LLM to extract research interests from email body."""
    try:
        import openai
        
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return body[:500]  # Just use the body text as-is
        
        client = openai.AsyncOpenAI(api_key=api_key)
        
        prompt = f"""Extract the research interests from this email where someone is signing up for research paper updates.

Email body:
---
{body[:2000]}
---

Extract and summarize their research interests in 1-3 sentences. Focus on:
- Areas of research they mention
- Topics they're interested in
- Specific keywords or domains

If no specific interests are mentioned, return a generic summary based on context.

Response: Just the research interests summary, nothing else."""

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Error extracting research interests: {e}")
        return body[:500] if body else ""


def fetch_colleague_signup_emails(since_hours: int = 48, diagnostic_mode: bool = False) -> List[Dict[str, Any]]:
    """Fetch recent emails that appear to be colleague signup requests.
    
    Args:
        since_hours: Look for emails from the last N hours
        diagnostic_mode: If True, print the 10 newest messages in INBOX for debugging
        
    Returns:
        List of parsed signup email data
    """
    import socket
    import time
    import traceback
    from threading import Timer
    
    IMAP_SOCKET_TIMEOUT = 20  # seconds - per-operation timeout
    CYCLE_TIMEOUT = 30  # seconds - overall timeout for entire fetch
    
    config = get_imap_config()
    cycle_start = time.time()
    
    logger.info(f"[GMAIL_FETCH] ===== START fetch_colleague_signup_emails (since_hours={since_hours}, diagnostic={diagnostic_mode}) =====")
    logger.info(f"[GMAIL_FETCH] IMAP config: host={config['host']}, port={config['port']}, user={'***' + config['user'][-10:] if config['user'] else 'NOT SET'}")
    
    if not config["user"] or not config["password"]:
        logger.warning("[GMAIL_FETCH] IMAP credentials not configured - SMTP_USER and SMTP_PASSWORD required")
        logger.info("[GMAIL_FETCH] ===== END (no credentials) =====")
        return []
    
    signups = []
    total_scanned = 0
    filtered_out = 0
    mail = None
    
    def check_cycle_timeout():
        elapsed = time.time() - cycle_start
        if elapsed > CYCLE_TIMEOUT:
            raise TimeoutError(f"Cycle timeout exceeded: {elapsed:.1f}s > {CYCLE_TIMEOUT}s")
        return elapsed
    
    try:
        # Step 1: Connect to IMAP server
        step_start = time.time()
        logger.info(f"[GMAIL_FETCH] [STEP 1/6] CONNECT start - host={config['host']}, port={config['port']}")
        
        # Set default socket timeout for all IMAP operations
        socket.setdefaulttimeout(IMAP_SOCKET_TIMEOUT)
        
        try:
            mail = imaplib.IMAP4_SSL(config["host"], config["port"])
            logger.info(f"[GMAIL_FETCH] [STEP 1/6] CONNECT ok - elapsed={time.time() - step_start:.2f}s")
        except socket.timeout:
            logger.error(f"[GMAIL_FETCH] [STEP 1/6] CONNECT TIMEOUT after {IMAP_SOCKET_TIMEOUT}s")
            raise
        except Exception as e:
            logger.error(f"[GMAIL_FETCH] [STEP 1/6] CONNECT FAILED: {e}")
            raise
        
        check_cycle_timeout()
        
        # Step 2: Login
        step_start = time.time()
        logger.info(f"[GMAIL_FETCH] [STEP 2/6] LOGIN start")
        try:
            mail.login(config["user"], config["password"])
            logger.info(f"[GMAIL_FETCH] [STEP 2/6] LOGIN ok - elapsed={time.time() - step_start:.2f}s")
        except socket.timeout:
            logger.error(f"[GMAIL_FETCH] [STEP 2/6] LOGIN TIMEOUT after {IMAP_SOCKET_TIMEOUT}s")
            raise
        except imaplib.IMAP4.error as e:
            logger.error(f"[GMAIL_FETCH] [STEP 2/6] LOGIN FAILED: {e}")
            raise
        
        check_cycle_timeout()
        
        # Step 3: Select INBOX explicitly (critical - must select INBOX for colleague joins)
        step_start = time.time()
        logger.info(f"[GMAIL_FETCH] [STEP 3/6] SELECT mailbox='INBOX' start")
        try:
            result, data = mail.select("INBOX")
            if result != "OK":
                logger.error(f"[GMAIL_FETCH] [STEP 3/6] SELECT INBOX FAILED: result={result}")
                raise Exception(f"IMAP SELECT failed: {result}")
            message_count = data[0].decode() if data and data[0] else "unknown"
            logger.info(f"[GMAIL_FETCH] [STEP 3/6] SELECT ok - mailbox=INBOX, total_messages={message_count}, elapsed={time.time() - step_start:.2f}s")
        except socket.timeout:
            logger.error(f"[GMAIL_FETCH] [STEP 3/6] SELECT TIMEOUT after {IMAP_SOCKET_TIMEOUT}s")
            raise
        
        check_cycle_timeout()
        
        # Step 4: Search for recent emails (NOT just UNSEEN - include all for complete search)
        step_start = time.time()
        since_date = (datetime.now() - timedelta(hours=since_hours)).strftime("%d-%b-%Y")
        # Use broad search criteria - DO NOT filter by UNSEEN or specific TO address
        # This ensures we see all emails in the timeframe
        search_criteria = f'(SINCE "{since_date}")'
        
        logger.info(f"[GMAIL_FETCH] [STEP 4/6] SEARCH start - criteria={search_criteria}")
        try:
            result, data = mail.search(None, search_criteria)
            if result != "OK":
                logger.error(f"[GMAIL_FETCH] [STEP 4/6] SEARCH FAILED: result={result}")
                raise Exception(f"IMAP SEARCH failed: {result}")
            
            email_ids = data[0].split() if data[0] else []
            logger.info(f"[GMAIL_FETCH] [STEP 4/6] SEARCH ok - hits={len(email_ids)}, elapsed={time.time() - step_start:.2f}s")
        except socket.timeout:
            logger.error(f"[GMAIL_FETCH] [STEP 4/6] SEARCH TIMEOUT after {IMAP_SOCKET_TIMEOUT}s")
            raise
        
        check_cycle_timeout()
        
        # Diagnostic mode: print the 10 newest messages
        if diagnostic_mode and email_ids:
            logger.info(f"[GMAIL_FETCH] [DIAGNOSTIC] Printing newest 10 messages...")
            newest_ids = email_ids[-10:]  # Get last 10 (newest)
            for eid in reversed(newest_ids):
                try:
                    # Fetch headers only for diagnostic (faster)
                    result, msg_data = mail.fetch(eid, "(UID FLAGS BODY.PEEK[HEADER.FIELDS (FROM TO SUBJECT DATE MESSAGE-ID)])")
                    if result == "OK" and msg_data and msg_data[0]:
                        uid_match = re.search(rb'UID\s+(\d+)', msg_data[0][0] if isinstance(msg_data[0], tuple) else msg_data[0])
                        uid = uid_match.group(1).decode() if uid_match else "?"
                        flags_match = re.search(rb'FLAGS\s+\(([^)]*)\)', msg_data[0][0] if isinstance(msg_data[0], tuple) else msg_data[0])
                        flags = flags_match.group(1).decode() if flags_match else "?"
                        
                        header_data = msg_data[0][1] if isinstance(msg_data[0], tuple) else b""
                        header_msg = email.message_from_bytes(header_data)
                        
                        logger.info(f"[GMAIL_FETCH] [DIAGNOSTIC] UID={uid}, FLAGS=({flags}), FROM={header_msg.get('From', '?')[:40]}, TO={header_msg.get('To', '?')[:40]}, SUBJECT={header_msg.get('Subject', '?')[:50]}, DATE={header_msg.get('Date', '?')}")
                except Exception as diag_e:
                    logger.warning(f"[GMAIL_FETCH] [DIAGNOSTIC] Error reading email {eid}: {diag_e}")
        
        # Step 5: Fetch and parse each email
        logger.info(f"[GMAIL_FETCH] [STEP 5/6] FETCH/PARSE start - processing {len(email_ids)} emails")
        
        for i, email_id in enumerate(email_ids):
            # Check cycle timeout periodically
            if i % 10 == 0:
                elapsed = check_cycle_timeout()
                if elapsed > CYCLE_TIMEOUT * 0.8:  # Warn if approaching timeout
                    logger.warning(f"[GMAIL_FETCH] Approaching cycle timeout ({elapsed:.1f}s), {len(email_ids) - i} emails remaining")
            
            total_scanned += 1
            eid_str = email_id.decode() if isinstance(email_id, bytes) else str(email_id)
            
            try:
                # Fetch with BODY.PEEK to avoid marking as seen
                fetch_start = time.time()
                result, msg_data = mail.fetch(email_id, "(UID BODY.PEEK[])")
                fetch_elapsed = time.time() - fetch_start
                
                if result != "OK":
                    logger.warning(f"[GMAIL_FETCH] FETCH failed for idx={eid_str}: result={result}")
                    continue
                
                if fetch_elapsed > 5:
                    logger.warning(f"[GMAIL_FETCH] Slow fetch for idx={eid_str}: {fetch_elapsed:.1f}s")
                
                # Extract UID from response
                uid = "?"
                if msg_data and msg_data[0]:
                    uid_match = re.search(rb'UID\s+(\d+)', msg_data[0][0] if isinstance(msg_data[0], tuple) else msg_data[0])
                    if uid_match:
                        uid = uid_match.group(1).decode()
                
                raw_email = msg_data[0][1] if isinstance(msg_data[0], tuple) else msg_data[0]
                msg = email.message_from_bytes(raw_email)
                
                # Get headers
                message_id = msg.get("Message-ID", "").strip("<>")
                thread_id = msg.get("Thread-Id", message_id[:20] if message_id else "unknown")
                subject = decode_email_subject(msg.get("Subject", ""))
                from_addr = msg.get("From", "")
                to_addr = msg.get("To", "")
                date_str = msg.get("Date", "")
                
                # Parse from address early for logging
                from_match = re.search(r'[\w\.-]+@[\w\.-]+', from_addr)
                from_email = from_match.group(0) if from_match else from_addr
                
                # Skip emails from ourselves (ResearchPulse's own outbound replies)
                system_email = config["user"].lower()
                if system_email and from_email.lower() == system_email:
                    filtered_out += 1
                    logger.debug(f"[MSG_ROUTE] Skipping self-sent email from {from_email}")
                    continue
                
                # Create correlation ID for this message
                corr_id = f"UID{uid}|{message_id[:12]}" if message_id else f"UID{uid}|idx{eid_str}"
                
                logger.debug(f"[MSG_READ][{corr_id}] from={from_email}, to={to_addr[:30]}, subject={subject[:50]}, date={date_str}")
                
                # Get body
                parse_start = time.time()
                body = get_email_body(msg)
                body_preview = body[:100].replace('\n', ' ') if body else "(empty)"
                logger.debug(f"[MSG_READ][{corr_id}] body_preview={body_preview}, parse_time={time.time() - parse_start:.2f}s")
                
                # Check if this is a potential colleague interaction
                if not is_colleague_signup_email(subject, body, from_email):
                    filtered_out += 1
                    logger.debug(f"[MSG_ROUTE][{corr_id}] intent=AUTOMATED (filtered out)")
                    continue
                
                logger.info(f"[MSG_ROUTE][{corr_id}] intent=JOIN_REQUEST from={from_email}")
                
                from_name = extract_name_from_email_header(from_addr)
                
                signup_data = {
                    "message_id": message_id or str(uuid4()),
                    "thread_id": thread_id,
                    "from_email": from_email,
                    "from_name": from_name,
                    "subject": subject,
                    "body_text": body,
                    "received_at": date_str,
                    "correlation_id": corr_id,
                    "uid": uid,
                }
                
                signups.append(signup_data)
                logger.info(f"[GMAIL_FETCH][{corr_id}] Queued colleague signup email from: {from_email}")
                
            except socket.timeout:
                logger.error(f"[GMAIL_FETCH] FETCH TIMEOUT for idx={eid_str} after {IMAP_SOCKET_TIMEOUT}s - skipping")
                continue
            except Exception as e:
                logger.error(f"[GMAIL_FETCH] Error processing email idx={eid_str}: {e}")
                continue
        
        # Step 6: Logout
        step_start = time.time()
        logger.info(f"[GMAIL_FETCH] [STEP 6/6] LOGOUT start")
        try:
            mail.logout()
            logger.info(f"[GMAIL_FETCH] [STEP 6/6] LOGOUT ok - elapsed={time.time() - step_start:.2f}s")
        except Exception as e:
            logger.warning(f"[GMAIL_FETCH] [STEP 6/6] LOGOUT warning: {e}")
        
        total_elapsed = time.time() - cycle_start
        logger.info(f"[GMAIL_FETCH] ===== END SUCCESS: scanned={total_scanned}, filtered_out={filtered_out}, signups_found={len(signups)}, total_time={total_elapsed:.2f}s =====")
        
    except socket.timeout as e:
        logger.error(f"[GMAIL_FETCH] Socket timeout: {e}")
        logger.error(f"[GMAIL_FETCH] Stack trace:\n{traceback.format_exc()}")
        if mail:
            try:
                mail.logout()
            except:
                pass
        logger.info(f"[GMAIL_FETCH] ===== END (socket timeout) =====")
        
    except TimeoutError as e:
        logger.error(f"[GMAIL_FETCH] Cycle timeout: {e}")
        if mail:
            try:
                mail.logout()
            except:
                pass
        logger.info(f"[GMAIL_FETCH] ===== END (cycle timeout) =====")
        
    except imaplib.IMAP4.error as e:
        logger.error(f"[GMAIL_FETCH] IMAP protocol error: {e}")
        logger.error(f"[GMAIL_FETCH] Stack trace:\n{traceback.format_exc()}")
        if mail:
            try:
                mail.logout()
            except:
                pass
        logger.info(f"[GMAIL_FETCH] ===== END (IMAP error) =====")
        
    except Exception as e:
        logger.error(f"[GMAIL_FETCH] Unexpected error: {e}")
        logger.error(f"[GMAIL_FETCH] Stack trace:\n{traceback.format_exc()}")
        if mail:
            try:
                mail.logout()
            except:
                pass
        logger.info(f"[GMAIL_FETCH] ===== END (unexpected error) =====")
    
    finally:
        # Reset socket timeout to default
        socket.setdefaulttimeout(None)
    
    return signups


async def process_colleague_signups(store, user_id: str) -> Dict[str, Any]:
    """Poll for colleague signup emails and create colleagues.
    
    Args:
        store: Database store instance
        user_id: User ID to associate colleagues with
        
    Returns:
        Summary of processing results
    """
    from uuid import UUID
    
    results = {
        "emails_scanned": 0,
        "signups_found": 0,
        "colleagues_created": 0,
        "colleagues_updated": 0,
        "errors": []
    }
    
    # Fetch signup emails
    signups = fetch_colleague_signup_emails(since_hours=48)
    results["signups_found"] = len(signups)
    
    for signup in signups:
        try:
            email_addr = signup["from_email"]
            
            # Check if colleague already exists
            existing_colleagues = store.get_colleagues(UUID(user_id))
            existing = next(
                (c for c in existing_colleagues if c.get("email", "").lower() == email_addr.lower()),
                None
            )
            
            if existing:
                logger.info(f"Colleague already exists for {email_addr}, skipping")
                results["colleagues_updated"] += 1
                continue
            
            # Extract research interests from the email body
            research_interests = await extract_research_interests_from_email(signup["body_text"])
            
            # Use LLM to infer keywords and categories
            from src.api.dashboard_routes import infer_research_keywords_categories
            inferred = await infer_research_keywords_categories(research_interests)
            
            # Create new colleague
            colleague_data = {
                "name": signup["from_name"],
                "email": email_addr,
                "research_interests": research_interests,
                "keywords": inferred.get("keywords", []),
                "categories": inferred.get("categories", []),
                "added_by": "email",  # Mark as added via email
                "auto_send_emails": True,
                "enabled": True,
                "notes": f"Signed up via email on {signup['received_at']}. Original subject: {signup['subject'][:100]}"
            }
            
            colleague = store.create_colleague(UUID(user_id), colleague_data)
            results["colleagues_created"] += 1
            logger.info(f"Created new colleague from email signup: {email_addr}")
            
        except Exception as e:
            error_msg = f"Error processing signup from {signup.get('from_email', 'unknown')}: {str(e)}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
    
    return results
