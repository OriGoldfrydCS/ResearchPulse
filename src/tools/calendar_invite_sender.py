"""
Calendar Invite Email Sender for ResearchPulse.

Sends calendar invitation emails with ICS attachments for:
- New reading reminders
- Rescheduled events
- Event cancellations

Uses SMTP or configured email provider (Resend, SendGrid).
Tracks sent invites in the database for reply processing.
"""

from __future__ import annotations

import os
import sys
import uuid
import smtplib
import importlib.util
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.utils import formataddr, make_msgid
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

# Import ics_generator functions using importlib to avoid relative import issues
_ics_generator_path = Path(__file__).parent / "ics_generator.py"

def _load_ics_generator():
    """Load ics_generator module, handling different import contexts."""
    global generate_uid, generate_reading_reminder_ics, generate_reschedule_ics, generate_cancel_ics
    
    try:
        # First try: direct import if path is in sys.path
        import sys
        tools_path = str(Path(__file__).parent)
        if tools_path not in sys.path:
            sys.path.insert(0, tools_path)
        import ics_generator as _ics
        return _ics.generate_uid, _ics.generate_reading_reminder_ics, _ics.generate_reschedule_ics, _ics.generate_cancel_ics
    except ImportError:
        pass
    
    # Fallback: use importlib with file location
    spec = importlib.util.spec_from_file_location("ics_generator", _ics_generator_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.generate_uid, module.generate_reading_reminder_ics, module.generate_reschedule_ics, module.generate_cancel_ics
    
    raise ImportError(f"Could not load ics_generator from {_ics_generator_path}")

generate_uid, generate_reading_reminder_ics, generate_reschedule_ics, generate_cancel_ics = _load_ics_generator()


def _get_smtp_config() -> Dict[str, Any]:
    """Get SMTP configuration from environment variables."""
    return {
        "host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
        "port": int(os.getenv("SMTP_PORT", "587")),
        "user": os.getenv("SMTP_USER", ""),
        "password": os.getenv("SMTP_PASSWORD", ""),
        "from_email": os.getenv("SMTP_FROM_EMAIL", os.getenv("SMTP_USER", "noreply@researchpulse.app")),
        "from_name": os.getenv("SMTP_FROM_NAME", "ResearchPulse"),
        "use_tls": os.getenv("SMTP_USE_TLS", "true").lower() == "true",
    }


def _is_email_configured() -> bool:
    """Check if email sending is configured."""
    config = _get_smtp_config()
    return bool(config["host"] and config["user"] and config["password"])


def send_calendar_invite_email(
    to_email: str,
    to_name: str,
    subject: str,
    body_text: str,
    body_html: Optional[str],
    ics_content: Optional[str] = None,
    ics_method: Optional[str] = "REQUEST",
    ics_filename: str = "invite.ics",
) -> Tuple[bool, str, str]:
    """
    Send a calendar invitation email with ICS attachment.
    
    Args:
        to_email: Recipient email address
        to_name: Recipient name
        subject: Email subject
        body_text: Plain text body
        body_html: Optional HTML body
        ics_content: ICS calendar content
        ics_method: ICS method (REQUEST, CANCEL)
        ics_filename: Filename for the ICS attachment
        
    Returns:
        Tuple of (success: bool, message_id: str, error: str)
    """
    config = _get_smtp_config()
    
    if not _is_email_configured():
        # Return simulated success for development
        fake_message_id = f"<{uuid.uuid4()}@researchpulse.local>"
        print(f"[DEV MODE] Would send calendar invite to {to_email}")
        print(f"[DEV MODE] Subject: {subject}")
        print(f"[DEV MODE] ICS Method: {ics_method}")
        return True, fake_message_id, ""
    
    try:
        # Generate unique message ID for threading
        message_id = make_msgid(domain="researchpulse.app")
        
        # Create multipart message
        msg = MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"] = formataddr((config["from_name"], config["from_email"]))
        msg["To"] = formataddr((to_name, to_email))
        msg["Message-ID"] = message_id
        
        # Add calendar-specific headers for email clients
        msg["X-Mailer"] = "ResearchPulse Calendar"
        
        # Create alternative part for text/html
        msg_alt = MIMEMultipart("alternative")
        
        # Plain text part
        part_text = MIMEText(body_text, "plain", "utf-8")
        msg_alt.attach(part_text)
        
        # HTML part (if provided)
        if body_html:
            part_html = MIMEText(body_html, "html", "utf-8")
            msg_alt.attach(part_html)
        
        # Calendar part (inline for Outlook/Apple Mail compatibility) - only if ICS content provided
        if ics_content:
            part_calendar = MIMEText(ics_content, "calendar", "utf-8")
            part_calendar.add_header("Content-Disposition", "inline")
            if ics_method:
                part_calendar.set_param("method", ics_method)
            msg_alt.attach(part_calendar)
        
        msg.attach(msg_alt)
        
        # ICS file attachment (for email clients that prefer attachments) - only if ICS content provided
        if ics_content:
            part_ics = MIMEBase("application", "ics")
            part_ics.set_payload(ics_content.encode("utf-8"))
            encoders.encode_base64(part_ics)
            part_ics.add_header(
                "Content-Disposition",
                f'attachment; filename="{ics_filename}"'
            )
            part_ics.add_header("Content-class", "urn:content-classes:calendarmessage")
            msg.attach(part_ics)
        
        # Send via SMTP
        with smtplib.SMTP(config["host"], config["port"]) as server:
            if config["use_tls"]:
                server.starttls()
            server.login(config["user"], config["password"])
            server.sendmail(
                config["from_email"],
                [to_email],
                msg.as_string()
            )
        
        return True, message_id, ""
        
    except smtplib.SMTPAuthenticationError as e:
        return False, "", f"SMTP authentication failed: {e}"
    except smtplib.SMTPConnectError as e:
        return False, "", f"SMTP connection failed: {e}"
    except Exception as e:
        return False, "", f"Failed to send email: {e}"


def send_reading_reminder_invite(
    user_email: str,
    user_name: str,
    papers: List[Dict[str, Any]],
    start_time: datetime,
    duration_minutes: int,
    reminder_minutes: int = 15,
    triggered_by: str = "user",
    agent_note: Optional[str] = None,
    existing_ics_uid: Optional[str] = None,
    sequence: int = 0,
) -> Dict[str, Any]:
    """
    Send a reading reminder calendar invitation.
    
    Args:
        user_email: Recipient email address
        user_name: Recipient name
        papers: List of paper dicts with 'title', 'url', 'importance' keys
        start_time: Event start time
        duration_minutes: Duration in minutes
        reminder_minutes: Minutes before event to trigger reminder
        triggered_by: 'user' or 'agent'
        agent_note: Optional note from the agent
        existing_ics_uid: Use existing UID for updates (rescheduling)
        sequence: Sequence number for updates
        
    Returns:
        Dict with:
        - success: bool
        - message_id: str (for email threading)
        - ics_uid: str (for calendar updates)
        - error: str (if failed)
    """
    # Generate or use existing UID
    ics_uid = existing_ics_uid or generate_uid()
    
    # Get organizer email from config
    config = _get_smtp_config()
    organizer_email = config["from_email"]
    
    # Generate ICS content
    ics_content = generate_reading_reminder_ics(
        uid=ics_uid,
        papers=papers,
        start_time=start_time,
        duration_minutes=duration_minutes,
        organizer_email=organizer_email,
        attendee_email=user_email,
        attendee_name=user_name,
        reminder_minutes=reminder_minutes,
        sequence=sequence,
        method="REQUEST",
        agent_note=agent_note,
    )
    
    # Generate email subject
    if len(papers) == 1:
        subject = f"📅 Reading Reminder: {papers[0].get('title', 'Research Paper')[:40]}..."
    else:
        subject = f"📅 Reading Reminder: {len(papers)} research papers – ResearchPulse"
    
    # Generate email body
    body_text = _generate_invite_body_text(papers, start_time, duration_minutes, agent_note)
    body_html = _generate_invite_body_html(papers, start_time, duration_minutes, agent_note)
    
    # Send the email
    success, message_id, error = send_calendar_invite_email(
        to_email=user_email,
        to_name=user_name,
        subject=subject,
        body_text=body_text,
        body_html=body_html,
        ics_content=ics_content,
        ics_method="REQUEST",
        ics_filename="reading_reminder.ics",
    )
    
    return {
        "success": success,
        "message_id": message_id,
        "ics_uid": ics_uid,
        "ics_content": ics_content,
        "subject": subject,
        "error": error,
        "triggered_by": triggered_by,
        "sequence": sequence,
    }


def send_reschedule_invite(
    user_email: str,
    user_name: str,
    papers: List[Dict[str, Any]],
    new_start_time: datetime,
    duration_minutes: int,
    ics_uid: str,
    sequence: int,
    reschedule_reason: str = "User requested reschedule",
    reminder_minutes: int = 15,
) -> Dict[str, Any]:
    """
    Send a rescheduled calendar invitation.
    
    Args:
        user_email: Recipient email address
        user_name: Recipient name
        papers: List of paper dicts
        new_start_time: New event start time
        duration_minutes: Duration in minutes
        ics_uid: SAME UID as the original event (required for updates)
        sequence: Incremented sequence number (must be > original)
        reschedule_reason: Reason for the reschedule
        reminder_minutes: Minutes before event to trigger reminder
        
    Returns:
        Dict with success, message_id, error, etc.
    """
    config = _get_smtp_config()
    organizer_email = config["from_email"]
    
    # Generate ICS content for reschedule
    ics_content = generate_reschedule_ics(
        uid=ics_uid,
        papers=papers,
        new_start_time=new_start_time,
        duration_minutes=duration_minutes,
        organizer_email=organizer_email,
        attendee_email=user_email,
        attendee_name=user_name,
        reminder_minutes=reminder_minutes,
        sequence=sequence,
        reschedule_reason=reschedule_reason,
    )
    
    # Generate email subject
    subject = f"📅 Rescheduled: Reading Reminder – ResearchPulse"
    
    # Generate email body
    body_text = _generate_reschedule_body_text(papers, new_start_time, duration_minutes, reschedule_reason)
    body_html = _generate_reschedule_body_html(papers, new_start_time, duration_minutes, reschedule_reason)
    
    # Send the email
    success, message_id, error = send_calendar_invite_email(
        to_email=user_email,
        to_name=user_name,
        subject=subject,
        body_text=body_text,
        body_html=body_html,
        ics_content=ics_content,
        ics_method="REQUEST",
        ics_filename="rescheduled_reminder.ics",
    )
    
    return {
        "success": success,
        "message_id": message_id,
        "ics_uid": ics_uid,
        "ics_content": ics_content,
        "subject": subject,
        "error": error,
        "triggered_by": "user",  # Reschedules are always user-triggered
        "sequence": sequence,
    }


def _generate_invite_body_text(
    papers: List[Dict[str, Any]],
    start_time: datetime,
    duration_minutes: int,
    agent_note: Optional[str] = None,
) -> str:
    """Generate plain text email body for calendar invite."""
    lines = []
    lines.append("📚 ResearchPulse Reading Reminder")
    lines.append("")
    lines.append(f"Scheduled for: {start_time.strftime('%B %d, %Y at %I:%M %p')}")
    lines.append(f"Duration: {duration_minutes} minutes")
    lines.append("")
    
    if agent_note:
        lines.append(f"🤖 Agent note: {agent_note}")
        lines.append("")
    
    lines.append("Papers to read:")
    lines.append("")
    
    for i, paper in enumerate(papers, 1):
        title = paper.get('title', 'Untitled')
        url = paper.get('url', '')
        importance = paper.get('importance', 'medium').upper()
        lines.append(f"{i}. [{importance}] {title}")
        if url:
            lines.append(f"   {url}")
        lines.append("")
    
    lines.append("---")
    lines.append("This calendar invitation was sent by ResearchPulse.")
    lines.append("Reply to this email to reschedule.")
    lines.append("")
    
    return "\n".join(lines)


def _generate_invite_body_html(
    papers: List[Dict[str, Any]],
    start_time: datetime,
    duration_minutes: int,
    agent_note: Optional[str] = None,
) -> str:
    """Generate HTML email body for calendar invite."""
    html = f"""
    <html>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px 10px 0 0;">
            <h1 style="margin: 0; font-size: 24px;">📚 ResearchPulse Reading Reminder</h1>
        </div>
        
        <div style="background: #f8f9fa; padding: 20px; border: 1px solid #e9ecef;">
            <p style="margin: 0 0 10px 0;">
                <strong>📅 Scheduled for:</strong> {start_time.strftime('%B %d, %Y at %I:%M %p')}
            </p>
            <p style="margin: 0;">
                <strong>⏱️ Duration:</strong> {duration_minutes} minutes
            </p>
        </div>
    """
    
    if agent_note:
        html += f"""
        <div style="background: #e3f2fd; padding: 15px; margin: 10px 0; border-left: 4px solid #2196f3; border-radius: 4px;">
            <strong>🤖 Agent note:</strong> {agent_note}
        </div>
        """
    
    html += """
        <div style="padding: 20px 0;">
            <h2 style="font-size: 18px; margin: 0 0 15px 0;">Papers to read:</h2>
    """
    
    for i, paper in enumerate(papers, 1):
        title = paper.get('title', 'Untitled')
        url = paper.get('url', '')
        importance = paper.get('importance', 'medium').upper()
        
        # Color coding for importance
        importance_colors = {
            'CRITICAL': '#d32f2f',
            'HIGH': '#f57c00',
            'MEDIUM': '#fbc02d',
            'LOW': '#4caf50',
        }
        color = importance_colors.get(importance, '#757575')
        
        html += f"""
            <div style="margin-bottom: 15px; padding: 15px; background: white; border: 1px solid #e0e0e0; border-radius: 8px;">
                <span style="background: {color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold;">{importance}</span>
                <h3 style="margin: 10px 0 5px 0; font-size: 16px;">{i}. {title}</h3>
        """
        if url:
            html += f"""
                <a href="{url}" style="color: #667eea; text-decoration: none;">Read paper →</a>
            """
        html += """
            </div>
        """
    
    html += """
        </div>
        
        <div style="background: #fff3e0; padding: 15px; border-radius: 8px; margin-top: 20px;">
            <p style="margin: 0; font-size: 14px; color: #e65100;">
                💡 <strong>Tip:</strong> Reply to this email to reschedule if this time doesn't work for you.
            </p>
        </div>
        
        <div style="text-align: center; padding: 20px; color: #9e9e9e; font-size: 12px;">
            <p>This calendar invitation was sent by ResearchPulse.</p>
        </div>
    </body>
    </html>
    """
    
    return html


def _generate_reschedule_body_text(
    papers: List[Dict[str, Any]],
    new_start_time: datetime,
    duration_minutes: int,
    reschedule_reason: str,
) -> str:
    """Generate plain text email body for rescheduled invite."""
    lines = []
    lines.append("📅 Your reading reminder has been RESCHEDULED")
    lines.append("")
    lines.append(f"Reason: {reschedule_reason}")
    lines.append("")
    lines.append(f"NEW TIME: {new_start_time.strftime('%B %d, %Y at %I:%M %p')}")
    lines.append(f"Duration: {duration_minutes} minutes")
    lines.append("")
    lines.append("Papers to read:")
    
    for i, paper in enumerate(papers, 1):
        title = paper.get('title', 'Untitled')
        lines.append(f"{i}. {title}")
    
    lines.append("")
    lines.append("---")
    lines.append("ResearchPulse has updated your calendar automatically.")
    
    return "\n".join(lines)


def _generate_reschedule_body_html(
    papers: List[Dict[str, Any]],
    new_start_time: datetime,
    duration_minutes: int,
    reschedule_reason: str,
) -> str:
    """Generate HTML email body for rescheduled invite."""
    paper_list = "".join([
        f"<li>{p.get('title', 'Untitled')}</li>" for p in papers
    ])
    
    return f"""
    <html>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: #ff9800; color: white; padding: 20px; border-radius: 10px 10px 0 0;">
            <h1 style="margin: 0; font-size: 24px;">📅 Reading Reminder Rescheduled</h1>
        </div>
        
        <div style="background: #fff3e0; padding: 15px; border-left: 4px solid #ff9800;">
            <strong>Reason:</strong> {reschedule_reason}
        </div>
        
        <div style="background: #e8f5e9; padding: 20px; margin: 15px 0; border-radius: 8px;">
            <h2 style="margin: 0 0 10px 0; color: #2e7d32;">✅ New Time</h2>
            <p style="margin: 0; font-size: 18px;">
                <strong>{new_start_time.strftime('%B %d, %Y at %I:%M %p')}</strong>
            </p>
            <p style="margin: 5px 0 0 0; color: #666;">
                Duration: {duration_minutes} minutes
            </p>
        </div>
        
        <div style="padding: 15px 0;">
            <h3>Papers to read:</h3>
            <ol>{paper_list}</ol>
        </div>
        
        <div style="text-align: center; padding: 20px; color: #9e9e9e; font-size: 12px;">
            <p>ResearchPulse has updated your calendar automatically.</p>
        </div>
    </body>
    </html>
    """
