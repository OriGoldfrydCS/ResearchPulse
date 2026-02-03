"""
ICS (iCalendar) Generation Utility for ResearchPulse.

Generates RFC 5545 compliant iCalendar files (.ics) for:
- Reading reminder events
- Calendar invitations with VALARM reminders
- Update/reschedule events (using SEQUENCE)
- Event cancellations

The generated ICS files can be attached to emails as calendar invitations.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class ICSEvent:
    """Data class representing an iCalendar event."""
    uid: str  # Unique identifier for the event
    title: str
    description: str
    start_time: datetime
    duration_minutes: int
    organizer_email: str
    organizer_name: str = "ResearchPulse"
    attendee_email: Optional[str] = None
    attendee_name: Optional[str] = None
    location: Optional[str] = None
    url: Optional[str] = None
    reminder_minutes: int = 15
    sequence: int = 0  # Incremented for updates
    method: str = "REQUEST"  # REQUEST, CANCEL, REPLY
    

def generate_uid() -> str:
    """Generate a unique identifier for an ICS event."""
    return f"{uuid.uuid4()}@researchpulse.app"


def escape_ics_text(text: str) -> str:
    """
    Escape special characters in ICS text fields.
    
    According to RFC 5545, backslash, semicolon, comma, and newlines
    need to be escaped in text fields.
    """
    if not text:
        return ""
    # Escape backslashes first
    text = text.replace("\\", "\\\\")
    # Escape semicolons and commas
    text = text.replace(";", "\\;")
    text = text.replace(",", "\\,")
    # Convert newlines to \n
    text = text.replace("\r\n", "\\n")
    text = text.replace("\n", "\\n")
    text = text.replace("\r", "\\n")
    return text


def fold_ics_line(line: str, max_length: int = 75) -> str:
    """
    Fold long ICS lines according to RFC 5545.
    
    Lines longer than 75 octets should be folded by inserting a CRLF
    followed by a single whitespace character.
    """
    if len(line.encode('utf-8')) <= max_length:
        return line
    
    result = []
    current_line = ""
    
    for char in line:
        test_line = current_line + char
        if len(test_line.encode('utf-8')) > max_length:
            result.append(current_line)
            current_line = " " + char  # Continue with space prefix
        else:
            current_line = test_line
    
    if current_line:
        result.append(current_line)
    
    return "\r\n".join(result)


def format_datetime_utc(dt: datetime) -> str:
    """Format datetime as ICS UTC timestamp (YYYYMMDDTHHMMSSZ)."""
    # Convert to UTC if timezone-aware
    if dt.tzinfo is not None:
        dt = dt.utctimetuple()
        return f"{dt.tm_year:04d}{dt.tm_mon:02d}{dt.tm_mday:02d}T{dt.tm_hour:02d}{dt.tm_min:02d}{dt.tm_sec:02d}Z"
    else:
        return dt.strftime("%Y%m%dT%H%M%SZ")


def generate_ics_content(event: ICSEvent) -> str:
    """
    Generate RFC 5545 compliant iCalendar content for a single event.
    
    Args:
        event: ICSEvent dataclass with event details
        
    Returns:
        ICS file content as a string
    """
    lines = []
    
    # Calendar header
    lines.append("BEGIN:VCALENDAR")
    lines.append("VERSION:2.0")
    lines.append("PRODID:-//ResearchPulse//Calendar Reminder//EN")
    lines.append(f"METHOD:{event.method}")
    lines.append("CALSCALE:GREGORIAN")
    
    # Event
    lines.append("BEGIN:VEVENT")
    
    # Unique identifier and sequence
    lines.append(f"UID:{event.uid}")
    lines.append(f"SEQUENCE:{event.sequence}")
    
    # Timestamps
    now = datetime.utcnow()
    lines.append(f"DTSTAMP:{format_datetime_utc(now)}")
    lines.append(f"CREATED:{format_datetime_utc(now)}")
    lines.append(f"LAST-MODIFIED:{format_datetime_utc(now)}")
    
    # Event times
    lines.append(f"DTSTART:{format_datetime_utc(event.start_time)}")
    end_time = event.start_time + timedelta(minutes=event.duration_minutes)
    lines.append(f"DTEND:{format_datetime_utc(end_time)}")
    
    # Event details
    lines.append(fold_ics_line(f"SUMMARY:{escape_ics_text(event.title)}"))
    
    if event.description:
        lines.append(fold_ics_line(f"DESCRIPTION:{escape_ics_text(event.description)}"))
    
    if event.location:
        lines.append(fold_ics_line(f"LOCATION:{escape_ics_text(event.location)}"))
    
    if event.url:
        lines.append(fold_ics_line(f"URL:{event.url}"))
    
    # Organizer
    organizer_cn = escape_ics_text(event.organizer_name)
    lines.append(f"ORGANIZER;CN={organizer_cn}:mailto:{event.organizer_email}")
    
    # Attendee (the recipient of the invite)
    if event.attendee_email:
        attendee_cn = escape_ics_text(event.attendee_name or event.attendee_email)
        lines.append(f"ATTENDEE;CN={attendee_cn};RSVP=TRUE;PARTSTAT=NEEDS-ACTION:mailto:{event.attendee_email}")
    
    # Status
    if event.method == "CANCEL":
        lines.append("STATUS:CANCELLED")
    else:
        lines.append("STATUS:CONFIRMED")
    
    # Transparency (show as busy)
    lines.append("TRANSP:OPAQUE")
    
    # Priority (normal)
    lines.append("PRIORITY:5")
    
    # Categories
    lines.append("CATEGORIES:ResearchPulse,Reading,Research")
    
    # Alarm/Reminder
    if event.reminder_minutes > 0 and event.method != "CANCEL":
        lines.append("BEGIN:VALARM")
        lines.append("ACTION:DISPLAY")
        lines.append(fold_ics_line(f"DESCRIPTION:Reminder: {escape_ics_text(event.title)}"))
        lines.append(f"TRIGGER:-PT{event.reminder_minutes}M")
        lines.append("END:VALARM")
    
    lines.append("END:VEVENT")
    lines.append("END:VCALENDAR")
    
    # Join with CRLF as per RFC 5545
    return "\r\n".join(lines) + "\r\n"


def generate_reading_reminder_ics(
    uid: str,
    papers: List[Dict[str, Any]],
    start_time: datetime,
    duration_minutes: int,
    organizer_email: str,
    attendee_email: str,
    attendee_name: str = "Researcher",
    reminder_minutes: int = 15,
    sequence: int = 0,
    method: str = "REQUEST",
    agent_note: Optional[str] = None,
) -> str:
    """
    Generate an ICS calendar invite for reading research papers.
    
    Args:
        uid: Unique identifier for the event (use generate_uid() for new events)
        papers: List of paper dicts with 'title' and 'url' keys
        start_time: Event start time
        duration_minutes: Duration in minutes
        organizer_email: Email of the organizer (ResearchPulse)
        attendee_email: Email of the attendee (researcher)
        attendee_name: Name of the attendee
        reminder_minutes: Minutes before event to trigger reminder
        sequence: Sequence number (increment for updates)
        method: ICS method (REQUEST, CANCEL)
        agent_note: Optional note from the agent
        
    Returns:
        ICS file content as a string
    """
    # Generate title
    if len(papers) == 1:
        title = f"📖 Read: {papers[0].get('title', 'Research Paper')[:50]}..."
    else:
        title = f"📖 Read {len(papers)} research papers – ResearchPulse"
    
    # Generate description with paper list
    desc_lines = []
    desc_lines.append("📚 ResearchPulse Reading Session")
    desc_lines.append(f"Estimated time: {duration_minutes} minutes")
    desc_lines.append("")
    
    if agent_note:
        desc_lines.append(f"🤖 Agent note: {agent_note}")
        desc_lines.append("")
    
    desc_lines.append("Papers to read:")
    desc_lines.append("")
    
    for i, paper in enumerate(papers, 1):
        paper_title = paper.get('title', 'Untitled')
        paper_url = paper.get('url', '')
        importance = paper.get('importance', 'medium').upper()
        
        desc_lines.append(f"{i}. [{importance}] {paper_title}")
        if paper_url:
            desc_lines.append(f"   Link: {paper_url}")
        desc_lines.append("")
    
    desc_lines.append("---")
    desc_lines.append("This reminder was created by ResearchPulse.")
    desc_lines.append("Reply to this email to reschedule.")
    
    description = "\n".join(desc_lines)
    
    # Create ICSEvent
    event = ICSEvent(
        uid=uid,
        title=title,
        description=description,
        start_time=start_time,
        duration_minutes=duration_minutes,
        organizer_email=organizer_email,
        organizer_name="ResearchPulse",
        attendee_email=attendee_email,
        attendee_name=attendee_name,
        reminder_minutes=reminder_minutes,
        sequence=sequence,
        method=method,
    )
    
    return generate_ics_content(event)


def generate_reschedule_ics(
    uid: str,
    papers: List[Dict[str, Any]],
    new_start_time: datetime,
    duration_minutes: int,
    organizer_email: str,
    attendee_email: str,
    attendee_name: str = "Researcher",
    reminder_minutes: int = 15,
    sequence: int = 1,
    reschedule_reason: str = "User requested reschedule",
) -> str:
    """
    Generate an ICS update for a rescheduled event.
    
    Same as generate_reading_reminder_ics but with incremented sequence
    and note about the reschedule.
    
    Args:
        uid: SAME uid as the original event (required for updates)
        sequence: Incremented sequence number (must be > original)
        reschedule_reason: Reason for the reschedule
    """
    agent_note = f"⏰ Rescheduled: {reschedule_reason}"
    
    return generate_reading_reminder_ics(
        uid=uid,
        papers=papers,
        start_time=new_start_time,
        duration_minutes=duration_minutes,
        organizer_email=organizer_email,
        attendee_email=attendee_email,
        attendee_name=attendee_name,
        reminder_minutes=reminder_minutes,
        sequence=sequence,
        method="REQUEST",  # REQUEST is used for updates too
        agent_note=agent_note,
    )


def generate_cancel_ics(
    uid: str,
    title: str,
    start_time: datetime,
    duration_minutes: int,
    organizer_email: str,
    attendee_email: str,
    attendee_name: str = "Researcher",
    sequence: int = 1,
    cancel_reason: str = "Event cancelled",
) -> str:
    """
    Generate an ICS cancellation for an event.
    
    Args:
        uid: SAME uid as the original event
        sequence: Incremented sequence number
        cancel_reason: Reason for cancellation
    """
    event = ICSEvent(
        uid=uid,
        title=f"[CANCELLED] {title}",
        description=f"This event has been cancelled.\n\nReason: {cancel_reason}",
        start_time=start_time,
        duration_minutes=duration_minutes,
        organizer_email=organizer_email,
        organizer_name="ResearchPulse",
        attendee_email=attendee_email,
        attendee_name=attendee_name,
        reminder_minutes=0,  # No reminder for cancellation
        sequence=sequence,
        method="CANCEL",
    )
    
    return generate_ics_content(event)


# =============================================================================
# Testing/Debug Functions
# =============================================================================

def validate_ics_content(ics_content: str) -> Dict[str, Any]:
    """
    Basic validation of ICS content structure.
    
    Returns dict with 'valid' boolean and any 'errors' found.
    """
    errors = []
    
    required_fields = [
        "BEGIN:VCALENDAR",
        "END:VCALENDAR",
        "BEGIN:VEVENT",
        "END:VEVENT",
        "VERSION:2.0",
        "UID:",
        "DTSTART:",
        "DTEND:",
    ]
    
    for field in required_fields:
        if field not in ics_content:
            errors.append(f"Missing required field: {field}")
    
    # Check line endings
    if "\r\n" not in ics_content:
        errors.append("ICS should use CRLF line endings")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }
