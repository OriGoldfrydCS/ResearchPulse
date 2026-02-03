"""Email poller for fetching incoming replies via IMAP.

This module polls the configured Gmail inbox for replies to calendar invitations
and processes them to reschedule events based on user requests.
"""

import os
import imaplib
import email
from email.header import decode_header
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


def get_email_body(msg: email.message.Message) -> str:
    """Extract plain text body from email message."""
    body = ""
    
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))
            
            # Skip attachments
            if "attachment" in content_disposition:
                continue
                
            if content_type == "text/plain":
                try:
                    payload = part.get_payload(decode=True)
                    charset = part.get_content_charset() or 'utf-8'
                    body = payload.decode(charset, errors='replace')
                    break
                except Exception as e:
                    logger.warning(f"Error decoding email part: {e}")
    else:
        try:
            payload = msg.get_payload(decode=True)
            charset = msg.get_content_charset() or 'utf-8'
            body = payload.decode(charset, errors='replace')
        except Exception as e:
            logger.warning(f"Error decoding email body: {e}")
    
    return body


def extract_original_message_id(msg: email.message.Message) -> Optional[str]:
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
    config = get_imap_config()
    
    if not config["user"] or not config["password"]:
        logger.warning("IMAP credentials not configured, skipping email polling")
        return []
    
    replies = []
    
    try:
        # Connect to IMAP server
        mail = imaplib.IMAP4_SSL(config["host"], config["port"])
        mail.login(config["user"], config["password"])
        
        # Select inbox
        mail.select("INBOX")
        
        # Search for recent emails
        since_date = (datetime.now() - timedelta(hours=since_hours)).strftime("%d-%b-%Y")
        search_criteria = f'(SINCE "{since_date}")'
        
        result, data = mail.search(None, search_criteria)
        
        if result != "OK":
            logger.error(f"IMAP search failed: {result}")
            return []
        
        email_ids = data[0].split()
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
                
            except Exception as e:
                logger.error(f"Error processing email {email_id}: {e}")
                continue
        
        mail.logout()
        
    except imaplib.IMAP4.error as e:
        logger.error(f"IMAP error: {e}")
    except Exception as e:
        logger.error(f"Error fetching emails: {e}")
    
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
                extracted_datetime_text=parse_result.raw_datetime_text,
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
                                # Get papers for description
                                papers = []
                                for pid in (new_event.get("paper_ids") or []):
                                    paper = store.get_paper(UUID(pid))
                                    if paper:
                                        papers.append(paper)
                                
                                send_reschedule_invite(
                                    event=new_event,
                                    recipient_email=user["email"],
                                    papers=papers,
                                    old_start_time=event["start_time"],
                                    triggered_by="user"
                                )
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
