"""
Unified Outbound Email Module for ResearchPulse.

This module provides a single entry point for ALL outbound emails, ensuring:
1. Consistent sender display name ("ResearchPulse")
2. Subject type tags ([REMINDER], [DIGEST], etc.)
3. Diagnostic logging for debugging
4. Optional database logging of sent emails

ALL email sending in ResearchPulse should route through this module.
"""

from __future__ import annotations

import logging
import os
import smtplib
from dataclasses import dataclass
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, make_msgid
from email import encoders
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Display name for all outbound emails
SENDER_DISPLAY_NAME = "ResearchPulse"


# =============================================================================
# Email Types and Subject Tags
# =============================================================================

class EmailType(str, Enum):
    """
    Enumeration of all email types sent by ResearchPulse.
    
    Each type maps to a subject tag prefix.
    """
    # Calendar/Reminder related
    REMINDER = "reminder"
    RESCHEDULE = "reschedule"
    CANCEL = "cancel"
    
    # Research updates
    UPDATE = "update"
    DIGEST = "digest"
    PAPER = "paper"
    
    # Colleague management
    COLLEAGUE_ADD = "colleague_add"
    COLLEAGUE_JOIN = "colleague_join"
    
    # Onboarding
    ONBOARDING = "onboarding"
    WELCOME = "welcome"
    
    # General
    NOTIFICATION = "notification"
    OTHER = "other"


# Mapping from EmailType to subject tag prefix
EMAIL_TYPE_TAGS: Dict[EmailType, str] = {
    EmailType.REMINDER: "[REMINDER]",
    EmailType.RESCHEDULE: "[RESCHEDULE]",
    EmailType.CANCEL: "[CANCEL]",
    EmailType.UPDATE: "[UPDATE]",
    EmailType.DIGEST: "[DIGEST]",
    EmailType.PAPER: "[PAPER]",
    EmailType.COLLEAGUE_ADD: "[COLLEAGUE_ADD]",
    EmailType.COLLEAGUE_JOIN: "[COLLEAGUE_JOIN]",
    EmailType.ONBOARDING: "[ONBOARDING]",
    EmailType.WELCOME: "[WELCOME]",
    EmailType.NOTIFICATION: "[NOTIFICATION]",
    EmailType.OTHER: "[RESEARCHPULSE]",
}


# =============================================================================
# SMTP Configuration
# =============================================================================

@dataclass
class SMTPConfig:
    """SMTP configuration loaded from environment variables."""
    host: str
    port: int
    user: str
    password: str
    from_email: str
    from_name: str
    use_tls: bool

    @classmethod
    def from_env(cls) -> "SMTPConfig":
        """Load SMTP configuration from environment variables."""
        smtp_user = os.getenv("SMTP_USER", "")
        return cls(
            host=os.getenv("SMTP_HOST", "smtp.gmail.com"),
            port=int(os.getenv("SMTP_PORT", "587")),
            user=smtp_user,
            password=os.getenv("SMTP_PASSWORD", ""),
            from_email=os.getenv("SMTP_FROM_EMAIL", os.getenv("SMTP_FROM", smtp_user)),
            from_name=os.getenv("SMTP_FROM_NAME", SENDER_DISPLAY_NAME),
            use_tls=os.getenv("SMTP_USE_TLS", "true").lower() == "true",
        )
    
    def is_configured(self) -> bool:
        """Check if SMTP is properly configured."""
        return bool(self.host and self.user and self.password)


# =============================================================================
# Subject Tag Utilities
# =============================================================================

def get_subject_tag(email_type: EmailType) -> str:
    """Get the subject tag prefix for an email type."""
    return EMAIL_TYPE_TAGS.get(email_type, "[RESEARCHPULSE]")


def has_subject_tag(subject: str) -> bool:
    """Check if a subject already has a tag prefix."""
    if not subject:
        return False
    # Check for any tag pattern like [WORD] at the start
    subject_stripped = subject.strip()
    if subject_stripped.startswith("["):
        close_bracket = subject_stripped.find("]")
        if close_bracket > 1:
            return True
    return False


def apply_subject_tag(subject: str, email_type: EmailType) -> str:
    """
    Apply a subject tag if not already present.
    
    Args:
        subject: Original subject line
        email_type: Type of email being sent
        
    Returns:
        Subject with tag prefix added (or original if already tagged)
    """
    if has_subject_tag(subject):
        return subject
    
    tag = get_subject_tag(email_type)
    # Clean up subject - remove any "ResearchPulse:" prefix if we're adding a tag
    clean_subject = subject.strip()
    if clean_subject.lower().startswith("researchpulse:"):
        clean_subject = clean_subject[14:].strip()
    
    return f"{tag} {clean_subject}"


# =============================================================================
# Main Email Sending Function
# =============================================================================

def send_outbound_email(
    to_email: str,
    subject: str,
    body: str,
    email_type: EmailType,
    html_body: Optional[str] = None,
    to_name: Optional[str] = None,
    attachments: Optional[List[Dict[str, Any]]] = None,
    ics_content: Optional[str] = None,
    ics_method: Optional[str] = None,
    ics_filename: str = "invite.ics",
    skip_tag: bool = False,
) -> Tuple[bool, str, str]:
    """
    Send an outbound email with consistent formatting.
    
    This is the SINGLE entry point for all outbound emails in ResearchPulse.
    It ensures:
    - Sender display name is always "ResearchPulse"
    - Subject has appropriate type tag prefix
    - Diagnostic logging is performed
    
    Args:
        to_email: Recipient email address
        subject: Email subject (tag will be prepended if not present)
        body: Plain text body
        email_type: Type of email (determines subject tag)
        html_body: Optional HTML body for rich formatting
        to_name: Optional recipient name
        attachments: Optional list of attachments [{filename, content, mimetype}]
        ics_content: Optional ICS calendar content
        ics_method: ICS method (REQUEST, CANCEL) if ics_content provided
        ics_filename: Filename for ICS attachment
        skip_tag: If True, don't add subject tag (for already-tagged subjects)
        
    Returns:
        Tuple of (success: bool, message_id: str, error: str)
    """
    config = SMTPConfig.from_env()
    
    # Apply subject tag
    tagged_subject = subject if skip_tag else apply_subject_tag(subject, email_type)
    
    # Log outbound email (diagnostic)
    logger.info(
        f"[OUTBOUND] type={email_type.value} "
        f"from_display=\"{config.from_name}\" "
        f"from_email=\"{config.from_email}\" "
        f"to=\"{to_email}\" "
        f"subject=\"{tagged_subject[:50]}...\""
    )
    
    if not config.is_configured():
        # Development mode - return simulated success
        fake_message_id = f"<dev-{email_type.value}-{os.urandom(8).hex()}@researchpulse.local>"
        logger.warning(
            f"[OUTBOUND] SMTP not configured - simulating send. "
            f"Set SMTP_USER and SMTP_PASSWORD environment variables."
        )
        print(f"[DEV MODE] Would send {email_type.value} email to {to_email}")
        print(f"[DEV MODE] Subject: {tagged_subject}")
        return True, fake_message_id, ""
    
    try:
        # Generate unique message ID
        message_id = make_msgid(domain="researchpulse.app")
        
        # Create message based on content type
        if ics_content:
            # Calendar invitation email
            msg = MIMEMultipart("mixed")
        elif html_body:
            # HTML + plain text alternative
            msg = MIMEMultipart("alternative")
        else:
            # Plain text only
            msg = MIMEMultipart()
        
        # Set headers with proper display name
        msg["From"] = formataddr((config.from_name, config.from_email))
        if to_name:
            msg["To"] = formataddr((to_name, to_email))
        else:
            msg["To"] = to_email
        msg["Subject"] = tagged_subject
        msg["Message-ID"] = message_id
        msg["X-Mailer"] = "ResearchPulse"
        msg["X-ResearchPulse-Type"] = email_type.value
        
        # Handle calendar invitation emails
        if ics_content:
            # Create alternative part for text/html
            msg_alt = MIMEMultipart("alternative")
            
            # Plain text part
            msg_alt.attach(MIMEText(body, "plain", "utf-8"))
            
            # HTML part
            if html_body:
                msg_alt.attach(MIMEText(html_body, "html", "utf-8"))
            
            # Inline calendar part
            part_calendar = MIMEText(ics_content, "calendar", "utf-8")
            part_calendar.add_header("Content-Disposition", "inline")
            if ics_method:
                part_calendar.set_param("method", ics_method)
            msg_alt.attach(part_calendar)
            
            msg.attach(msg_alt)
            
            # ICS file attachment
            part_ics = MIMEBase("application", "ics")
            part_ics.set_payload(ics_content.encode("utf-8"))
            encoders.encode_base64(part_ics)
            part_ics.add_header(
                "Content-Disposition",
                f'attachment; filename="{ics_filename}"'
            )
            part_ics.add_header("Content-class", "urn:content-classes:calendarmessage")
            msg.attach(part_ics)
        else:
            # Standard email (text or HTML)
            # Clean up plain text body (remove header lines if present)
            body_lines = body.split("\n")
            body_start = 0
            for i, line in enumerate(body_lines):
                if line.startswith("=" * 20):
                    body_start = i
                    break
            clean_body = "\n".join(body_lines[body_start:])
            
            msg.attach(MIMEText(clean_body, "plain", "utf-8"))
            
            if html_body:
                msg.attach(MIMEText(html_body, "html", "utf-8"))
        
        # Add any additional attachments
        if attachments:
            for attachment in attachments:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.get("content", b""))
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f'attachment; filename="{attachment.get("filename", "attachment")}"'
                )
                if attachment.get("mimetype"):
                    part.set_type(attachment["mimetype"])
                msg.attach(part)
        
        # Send via SMTP
        with smtplib.SMTP(config.host, config.port) as server:
            if config.use_tls:
                server.starttls()
            server.login(config.user, config.password)
            server.send_message(msg)
        
        logger.info(f"[OUTBOUND] ✓ Email sent successfully to {to_email} (message_id={message_id})")
        print(f"✓ {email_type.value.upper()} email sent to {to_email}")
        return True, message_id, ""
        
    except smtplib.SMTPAuthenticationError as e:
        error_msg = f"SMTP authentication failed: {e}"
        logger.error(f"[OUTBOUND] {error_msg}")
        return False, "", error_msg
    except smtplib.SMTPConnectError as e:
        error_msg = f"SMTP connection failed: {e}"
        logger.error(f"[OUTBOUND] {error_msg}")
        return False, "", error_msg
    except Exception as e:
        error_msg = f"Failed to send email: {e}"
        logger.error(f"[OUTBOUND] {error_msg}")
        return False, "", error_msg


# =============================================================================
# Convenience Functions for Common Email Types
# =============================================================================

def send_reminder_email(
    to_email: str,
    subject: str,
    body: str,
    html_body: Optional[str] = None,
    to_name: Optional[str] = None,
    ics_content: Optional[str] = None,
    ics_method: str = "REQUEST",
) -> Tuple[bool, str, str]:
    """Send a reminder/calendar email."""
    return send_outbound_email(
        to_email=to_email,
        subject=subject,
        body=body,
        email_type=EmailType.REMINDER,
        html_body=html_body,
        to_name=to_name,
        ics_content=ics_content,
        ics_method=ics_method,
    )


def send_digest_email(
    to_email: str,
    subject: str,
    body: str,
    html_body: Optional[str] = None,
    to_name: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """Send a digest/summary email."""
    return send_outbound_email(
        to_email=to_email,
        subject=subject,
        body=body,
        email_type=EmailType.DIGEST,
        html_body=html_body,
        to_name=to_name,
    )


def send_update_email(
    to_email: str,
    subject: str,
    body: str,
    html_body: Optional[str] = None,
    to_name: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """Send a paper update email."""
    return send_outbound_email(
        to_email=to_email,
        subject=subject,
        body=body,
        email_type=EmailType.UPDATE,
        html_body=html_body,
        to_name=to_name,
    )


def send_onboarding_email(
    to_email: str,
    subject: str,
    body: str,
    html_body: Optional[str] = None,
    to_name: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """Send an onboarding email."""
    return send_outbound_email(
        to_email=to_email,
        subject=subject,
        body=body,
        email_type=EmailType.ONBOARDING,
        html_body=html_body,
        to_name=to_name,
    )


def send_colleague_email(
    to_email: str,
    subject: str,
    body: str,
    email_type: EmailType = EmailType.COLLEAGUE_JOIN,
    html_body: Optional[str] = None,
    to_name: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """Send a colleague-related email."""
    return send_outbound_email(
        to_email=to_email,
        subject=subject,
        body=body,
        email_type=email_type,
        html_body=html_body,
        to_name=to_name,
    )


# =============================================================================
# Legacy Compatibility Layer
# =============================================================================

def get_formatted_from_header() -> str:
    """
    Get properly formatted From header for use in legacy code.
    
    Returns:
        Formatted "Display Name <email>" string
    """
    config = SMTPConfig.from_env()
    return formataddr((config.from_name, config.from_email))


def get_smtp_config_dict() -> Dict[str, Any]:
    """
    Get SMTP configuration as dictionary for legacy code.
    
    Returns:
        Dict with host, port, user, password, from_email, from_name, use_tls
    """
    config = SMTPConfig.from_env()
    return {
        "host": config.host,
        "port": config.port,
        "user": config.user,
        "password": config.password,
        "from_email": config.from_email,
        "from_name": config.from_name,
        "use_tls": config.use_tls,
    }
