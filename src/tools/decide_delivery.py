"""
Tool: decide_delivery_action - Determine delivery actions for scored papers.

This tool takes scored paper information, delivery policy, and colleagues database
to decide what actions should be taken for both the researcher and colleagues.

**Delivery Decision Logic:**

1. **Researcher Actions**: Based on paper importance and delivery policy:
   - high importance: email + calendar + reading list + colleague sharing
   - medium importance: reading list + colleague sharing
   - low importance: log only (configurable)
   - log_only: no actions

2. **Colleague Actions**: Based on:
   - Paper importance allows colleague sharing
   - Topic overlap between paper and colleague interests
   - Colleague sharing preference (immediate/daily/weekly/on_request)
   - arXiv category alignment

3. **Simulated Outputs**:
   - Email files (.txt or .md) in artifacts/emails/
   - Calendar .ics files in artifacts/calendar/
   - Reading list append (artifacts/reading_list.md)
   - Colleague share files in artifacts/shares/
"""

from __future__ import annotations

import os
import sys
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# Add parent to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Input/Output Models
# =============================================================================

class ScoredPaper(BaseModel):
    """A paper with scoring results from score_relevance_and_importance."""
    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    abstract: str = Field("", description="Paper abstract")
    link: Optional[str] = Field(None, description="URL to the paper")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    categories: List[str] = Field(default_factory=list, description="arXiv categories")
    publication_date: Optional[str] = Field(None, description="Publication date")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score 0-1")
    novelty_score: float = Field(..., ge=0.0, le=1.0, description="Novelty score 0-1")
    importance: Literal["high", "medium", "low"] = Field(..., description="Importance level")
    explanation: str = Field("", description="Scoring explanation")


class ColleagueInfo(BaseModel):
    """Colleague information for sharing decisions."""
    id: str = Field(..., description="Unique colleague identifier")
    name: str = Field(..., description="Colleague's full name")
    email: str = Field(..., description="Email address")
    affiliation: Optional[str] = Field(None, description="Institution")
    topics: List[str] = Field(default_factory=list, description="Research interests")
    sharing_preference: Literal["immediate", "daily_digest", "weekly_digest", "on_request", "daily", "weekly", "monthly", "never"] = Field(
        "daily_digest", description="Preferred sharing frequency"
    )
    arxiv_categories_interest: List[str] = Field(
        default_factory=list, description="Interested arXiv categories"
    )


class ResearcherAction(BaseModel):
    """An action to take for the researcher."""
    action_type: Literal["email", "calendar", "reading_list", "log"] = Field(
        ..., description="Type of action"
    )
    paper_id: str = Field(..., description="arXiv paper ID")
    paper_title: str = Field(..., description="Paper title")
    priority: str = Field("normal", description="Priority label")
    details: Dict[str, Any] = Field(default_factory=dict, description="Action-specific details")


class ColleagueAction(BaseModel):
    """An action to share with a colleague."""
    action_type: Literal["share_immediate", "share_daily", "share_weekly", "skip"] = Field(
        ..., description="Type of sharing action"
    )
    colleague_id: str = Field(..., description="Colleague identifier")
    colleague_name: str = Field(..., description="Colleague name")
    colleague_email: str = Field(..., description="Colleague email")
    paper_id: str = Field(..., description="arXiv paper ID")
    paper_title: str = Field(..., description="Paper title")
    relevance_reason: str = Field("", description="Why this paper is relevant to colleague")
    details: Dict[str, Any] = Field(default_factory=dict, description="Action-specific details")


class FileToWrite(BaseModel):
    """A file to write as simulated output."""
    file_type: Literal["email", "calendar", "reading_list", "share"] = Field(
        ..., description="Type of file"
    )
    file_path: str = Field(..., description="Relative path from artifacts/")
    content: str = Field(..., description="File content")
    description: str = Field("", description="Description of the file")


class DeliveryDecisionResult(BaseModel):
    """Result of decide_delivery_action tool."""
    paper_id: str = Field(..., description="arXiv paper ID")
    paper_title: str = Field(..., description="Paper title")
    importance: str = Field(..., description="Paper importance level")
    researcher_actions: List[ResearcherAction] = Field(
        default_factory=list, description="Actions for the researcher"
    )
    colleague_actions: List[ColleagueAction] = Field(
        default_factory=list, description="Actions for colleagues"
    )
    files_to_write: List[FileToWrite] = Field(
        default_factory=list, description="Files to write as simulated outputs"
    )
    summary: str = Field("", description="Summary of decisions made")


# =============================================================================
# Helper Functions
# =============================================================================

def _get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _get_date_str() -> str:
    """Get current date string."""
    return datetime.now().strftime("%Y-%m-%d")


def _get_file_timestamp() -> str:
    """Get timestamp suitable for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_for_matching(text: str) -> str:
    """Normalize text for topic matching."""
    return text.lower().strip()


def _topics_overlap(topics1: List[str], topics2: List[str]) -> tuple[bool, List[str]]:
    """
    Check if two topic lists have meaningful overlap.
    
    Returns:
        (has_overlap: bool, matching_topics: List[str])
    """
    if not topics1 or not topics2:
        return False, []
    
    # Normalize topics
    norm1 = {_normalize_for_matching(t) for t in topics1}
    norm2 = {_normalize_for_matching(t) for t in topics2}
    
    # Direct matches
    direct = norm1 & norm2
    
    # Partial matches (one contains the other)
    partial = set()
    for t1 in norm1:
        for t2 in norm2:
            if t1 != t2:
                if t1 in t2 or t2 in t1:
                    partial.add(t2 if t2 in topics2 else t1)
    
    matching = list(direct | partial)
    return len(matching) > 0, matching


def _categories_overlap(cats1: List[str], cats2: List[str]) -> bool:
    """Check if two category lists overlap."""
    if not cats1 or not cats2:
        return False
    return bool(set(cats1) & set(cats2))


def _send_email_smtp(
    to_email: str,
    subject: str,
    body: str,
) -> bool:
    """
    Send email via SMTP using environment variables for configuration.
    
    Required environment variables:
        SMTP_HOST: SMTP server hostname (default: smtp.gmail.com)
        SMTP_PORT: SMTP server port (default: 587)
        SMTP_USER: SMTP username/email
        SMTP_PASSWORD: SMTP password or app-specific password
        SMTP_FROM: From email address (defaults to SMTP_USER)
    
    Returns:
        True if email was sent successfully, False otherwise.
    """
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    smtp_from = os.environ.get("SMTP_FROM", smtp_user)
    
    if not smtp_user or not smtp_password:
        print("SMTP credentials not configured. Set SMTP_USER and SMTP_PASSWORD environment variables.")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg["From"] = smtp_from
        msg["To"] = to_email
        msg["Subject"] = subject
        
        # Remove the email headers from body (they're duplicated in MIME)
        body_lines = body.split("\n")
        body_start = 0
        for i, line in enumerate(body_lines):
            if line.startswith("=" * 20):  # Start of content
                body_start = i
                break
        clean_body = "\n".join(body_lines[body_start:])
        
        msg.attach(MIMEText(clean_body, "plain"))
        
        # Connect and send
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        print(f"✓ Email sent successfully to {to_email}")
        return True
        
    except smtplib.SMTPAuthenticationError:
        print("SMTP authentication failed. Check SMTP_USER and SMTP_PASSWORD.")
        return False
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False


def _generate_email_content(
    paper: ScoredPaper,
    priority: str,
    include_abstract: bool = True,
    include_explanation: bool = True,
    researcher_name: str = "Researcher",
) -> str:
    """Generate email content for a paper notification."""
    lines = [
        f"Subject: [ResearchPulse] {priority.upper()}: New paper - {paper.title}",
        f"From: researchpulse@localhost",
        f"Date: {_get_timestamp()}",
        "",
        f"Dear {researcher_name},",
        "",
        "=" * 60,
        f"NEW PAPER ALERT ({priority.upper()} PRIORITY)",
        "=" * 60,
        "",
        f"Title: {paper.title}",
        f"arXiv ID: {paper.arxiv_id}",
    ]
    
    if paper.authors:
        lines.append(f"Authors: {', '.join(paper.authors[:5])}")
        if len(paper.authors) > 5:
            lines[-1] += f" (+{len(paper.authors) - 5} more)"
    
    if paper.link:
        lines.append(f"Link: {paper.link}")
    
    if paper.categories:
        lines.append(f"Categories: {', '.join(paper.categories)}")
    
    if paper.publication_date:
        lines.append(f"Published: {paper.publication_date}")
    
    lines.extend([
        "",
        "-" * 40,
        "RELEVANCE ASSESSMENT",
        "-" * 40,
        f"Relevance Score: {paper.relevance_score:.0%}",
        f"Novelty Score: {paper.novelty_score:.0%}",
        f"Importance: {paper.importance.upper()}",
    ])
    
    if include_explanation and paper.explanation:
        lines.extend(["", f"Assessment: {paper.explanation}"])
    
    if include_abstract and paper.abstract:
        lines.extend([
            "",
            "-" * 40,
            "ABSTRACT",
            "-" * 40,
            paper.abstract[:1500] + ("..." if len(paper.abstract) > 1500 else ""),
        ])
    
    lines.extend([
        "",
        "=" * 60,
        "This email was generated by ResearchPulse.",
        "",
    ])
    
    return "\n".join(lines)


def _generate_digest_email_content(
    papers: List[ScoredPaper],
    researcher_name: str = "Researcher",
    include_abstracts: bool = True,
    max_papers: int = 10,
) -> str:
    """
    Generate a digest email with multiple papers.
    
    Args:
        papers: List of scored papers to include in digest
        researcher_name: Name of the researcher
        include_abstracts: Whether to include paper abstracts
        max_papers: Maximum papers to include in the digest
    
    Returns:
        Formatted email content
    """
    # Sort papers by importance and relevance
    importance_order = {"high": 0, "medium": 1, "low": 2}
    sorted_papers = sorted(
        papers,
        key=lambda p: (importance_order.get(p.importance, 2), -p.relevance_score)
    )[:max_papers]
    
    high_count = sum(1 for p in sorted_papers if p.importance == "high")
    medium_count = sum(1 for p in sorted_papers if p.importance == "medium")
    
    lines = [
        f"Subject: [ResearchPulse] Daily Digest: {len(sorted_papers)} important papers",
        f"From: researchpulse@localhost",
        f"Date: {_get_timestamp()}",
        "",
        f"Dear {researcher_name},",
        "",
        "=" * 60,
        "RESEARCH PULSE DAILY DIGEST",
        "=" * 60,
        "",
        f"We found {len(sorted_papers)} papers that may interest you:",
        f"  • {high_count} HIGH priority papers",
        f"  • {medium_count} MEDIUM priority papers",
        "",
        "-" * 60,
        "PAPER SUMMARIES",
        "-" * 60,
    ]
    
    for i, paper in enumerate(sorted_papers, 1):
        lines.extend([
            "",
            f"[{i}] {paper.title}",
            f"    Importance: {paper.importance.upper()} | Relevance: {paper.relevance_score:.0%}",
            f"    arXiv ID: {paper.arxiv_id}",
        ])
        
        if paper.authors:
            author_str = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                author_str += f" (+{len(paper.authors) - 3} more)"
            lines.append(f"    Authors: {author_str}")
        
        if paper.link:
            lines.append(f"    Link: {paper.link}")
        
        if paper.categories:
            lines.append(f"    Categories: {', '.join(paper.categories[:3])}")
        
        if paper.explanation:
            lines.append(f"    Why: {paper.explanation[:200]}...")
        
        if include_abstracts and paper.abstract:
            abstract_preview = paper.abstract[:300] + ("..." if len(paper.abstract) > 300 else "")
            lines.extend([
                f"    Abstract: {abstract_preview}",
            ])
    
    lines.extend([
        "",
        "=" * 60,
        "This digest was generated by ResearchPulse.",
        f"To adjust your preferences, visit the ResearchPulse dashboard.",
        "",
    ])
    
    return "\n".join(lines)


def send_digest_email(
    papers: List[ScoredPaper],
    researcher_email: str,
    researcher_name: str = "Researcher",
    email_settings: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Send a single digest email with all important papers.
    
    This should be called at the end of a run instead of sending
    individual emails per paper.
    
    Args:
        papers: List of scored papers to include
        researcher_email: Recipient email address
        researcher_name: Name of the researcher
        email_settings: Email settings from delivery policy
        
    Returns:
        True if email was sent successfully
    """
    if not papers:
        print("No papers to include in digest email.")
        return False
    
    settings = email_settings or {}
    max_papers = settings.get("max_papers_per_email", 10)
    include_abstracts = settings.get("include_abstract", True)
    
    # Filter to only high and medium importance papers
    important_papers = [p for p in papers if p.importance in ("high", "medium")]
    
    if not important_papers:
        print("No important papers to send in digest.")
        return False
    
    # Generate digest content
    email_content = _generate_digest_email_content(
        papers=important_papers,
        researcher_name=researcher_name,
        include_abstracts=include_abstracts,
        max_papers=max_papers,
    )
    
    # Send via SMTP
    high_count = sum(1 for p in important_papers if p.importance == "high")
    subject = f"[ResearchPulse] Daily Digest: {len(important_papers)} papers ({high_count} high priority)"
    
    return _send_email_smtp(
        to_email=researcher_email,
        subject=subject,
        body=email_content,
    )


def _generate_calendar_ics(
    paper: ScoredPaper,
    event_duration_minutes: int = 30,
    reminder_minutes: int = 60,
    schedule_within_days: int = 7,
) -> str:
    """Generate iCalendar (.ics) content for a reading event."""
    # Generate unique ID for the event
    uid = f"{paper.arxiv_id}-{uuid.uuid4().hex[:8]}@researchpulse"
    
    # Schedule within the next N days
    now = datetime.now()
    # Schedule for tomorrow at 10am
    event_start = now.replace(hour=10, minute=0, second=0, microsecond=0) + timedelta(days=1)
    event_end = event_start + timedelta(minutes=event_duration_minutes)
    
    # Format dates for iCalendar (UTC format)
    dtstart = event_start.strftime("%Y%m%dT%H%M%S")
    dtend = event_end.strftime("%Y%m%dT%H%M%S")
    dtstamp = now.strftime("%Y%m%dT%H%M%SZ")
    
    # Truncate title if too long
    title = paper.title[:75] + "..." if len(paper.title) > 75 else paper.title
    
    # Build description
    description_parts = [
        f"Paper: {paper.title}",
        f"arXiv: {paper.arxiv_id}",
        f"Importance: {paper.importance.upper()}",
        f"Relevance: {paper.relevance_score:.0%}",
    ]
    if paper.link:
        description_parts.append(f"Link: {paper.link}")
    if paper.explanation:
        description_parts.append(f"\\nAssessment: {paper.explanation}")
    
    description = "\\n".join(description_parts)
    
    ics_content = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//ResearchPulse//Paper Reading//EN
CALSCALE:GREGORIAN
METHOD:PUBLISH
BEGIN:VEVENT
UID:{uid}
DTSTAMP:{dtstamp}
DTSTART:{dtstart}
DTEND:{dtend}
SUMMARY:Read: {title}
DESCRIPTION:{description}
STATUS:CONFIRMED
CATEGORIES:ResearchPulse,Paper,Reading
PRIORITY:{"1" if paper.importance == "high" else "5" if paper.importance == "medium" else "9"}
BEGIN:VALARM
TRIGGER:-PT{reminder_minutes}M
ACTION:DISPLAY
DESCRIPTION:Reminder: Read paper - {paper.title[:50]}
END:VALARM
END:VEVENT
END:VCALENDAR
"""
    return ics_content


def _generate_reading_list_entry(paper: ScoredPaper) -> str:
    """Generate a markdown entry for the reading list."""
    lines = [
        f"## [{paper.title}]({paper.link or f'https://arxiv.org/abs/{paper.arxiv_id}'})",
        "",
        f"- **arXiv ID**: {paper.arxiv_id}",
        f"- **Date Added**: {_get_date_str()}",
        f"- **Importance**: {paper.importance.upper()}",
        f"- **Relevance**: {paper.relevance_score:.0%}",
        f"- **Novelty**: {paper.novelty_score:.0%}",
    ]
    
    if paper.authors:
        authors_str = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors_str += f" et al. ({len(paper.authors)} authors)"
        lines.append(f"- **Authors**: {authors_str}")
    
    if paper.categories:
        lines.append(f"- **Categories**: {', '.join(paper.categories)}")
    
    if paper.explanation:
        lines.append(f"- **Assessment**: {paper.explanation}")
    
    lines.append("")
    
    return "\n".join(lines)


def _generate_share_content(
    paper: ScoredPaper,
    colleague: ColleagueInfo,
    relevance_reason: str,
) -> str:
    """Generate share notification content for a colleague."""
    lines = [
        f"Subject: Paper recommendation: {paper.title}",
        f"To: {colleague.name} <{colleague.email}>",
        f"From: ResearchPulse <researchpulse@localhost>",
        f"Date: {_get_timestamp()}",
        "",
        "=" * 60,
        "PAPER RECOMMENDATION",
        "=" * 60,
        "",
        f"Hi {colleague.name.split()[0]},",
        "",
        "I found a paper that might interest you based on your research in",
        f"{', '.join(colleague.topics[:3]) if colleague.topics else 'your area'}.",
        "",
        "-" * 40,
        "",
        f"Title: {paper.title}",
        f"arXiv ID: {paper.arxiv_id}",
    ]
    
    if paper.authors:
        lines.append(f"Authors: {', '.join(paper.authors[:5])}")
    
    if paper.link:
        lines.append(f"Link: {paper.link}")
    
    lines.extend([
        "",
        f"Why this might be relevant: {relevance_reason}",
        "",
    ])
    
    if paper.abstract:
        lines.extend([
            "-" * 40,
            "Abstract (excerpt):",
            paper.abstract[:800] + ("..." if len(paper.abstract) > 800 else ""),
            "",
        ])
    
    lines.extend([
        "=" * 60,
        "This recommendation was generated by ResearchPulse.",
        "",
    ])
    
    return "\n".join(lines)


def _get_colleague_relevance_reason(
    paper: ScoredPaper,
    colleague: ColleagueInfo,
    matching_topics: List[str],
    has_category_match: bool,
) -> str:
    """Generate a relevance reason for why a paper is being shared."""
    reasons = []
    
    if matching_topics:
        topics_str = ", ".join(matching_topics[:3])
        reasons.append(f"matches your interest in {topics_str}")
    
    if has_category_match:
        overlap = set(paper.categories) & set(colleague.arxiv_categories_interest)
        if overlap:
            reasons.append(f"in categories you follow ({', '.join(overlap)})")
    
    if paper.importance == "high":
        reasons.append("assessed as high importance")
    
    if paper.novelty_score > 0.7:
        reasons.append("presents novel contributions")
    
    if not reasons:
        reasons.append("may be of general interest")
    
    return "; ".join(reasons).capitalize()


# =============================================================================
# Main Tool Implementation
# =============================================================================

def decide_delivery_action(
    scored_paper: Dict[str, Any],
    delivery_policy: Dict[str, Any],
    colleagues: List[Dict[str, Any]],
    artifacts_dir: str = "artifacts",
    researcher_name: str = "Researcher",
    researcher_email: str = "",
) -> DeliveryDecisionResult:
    """
    Decide delivery actions for a scored paper.
    
    This tool determines what actions to take based on:
    1. Paper's importance level
    2. Delivery policy settings
    3. Colleague interests and sharing preferences
    
    Args:
        scored_paper: Paper with scoring results. Required fields:
            - arxiv_id: arXiv paper ID
            - title: Paper title
            - importance: "high", "medium", or "low"
            - relevance_score: 0-1 relevance score
            - novelty_score: 0-1 novelty score
            Optional: abstract, link, authors, categories, explanation
            
        delivery_policy: Policy configuration with:
            - importance_policies: Dict mapping importance to action flags
            - email_settings: Email delivery settings
            - calendar_settings: Calendar settings
            - reading_list_settings: Reading list settings
            - colleague_sharing_settings: Sharing settings
            
        colleagues: List of colleague dictionaries with:
            - id, name, email
            - topics: Research interests
            - sharing_preference: "immediate", "daily_digest", etc.
            - arxiv_categories_interest: Categories they follow
            
        artifacts_dir: Base directory for simulated outputs (default: "artifacts")
            
    Returns:
        DeliveryDecisionResult with researcher_actions, colleague_actions,
        and files_to_write for simulated outputs.
        
    Example:
        >>> result = decide_delivery_action(
        ...     scored_paper={"arxiv_id": "2501.00123", "title": "LLM Paper", 
        ...                   "importance": "high", "relevance_score": 0.8,
        ...                   "novelty_score": 0.7},
        ...     delivery_policy=policy_dict,
        ...     colleagues=colleagues_list,
        ... )
        >>> print(result.researcher_actions)
    """
    # Parse inputs
    paper = ScoredPaper(**scored_paper)
    colleague_objs = [ColleagueInfo(**c) for c in colleagues]
    
    # Extract policy settings
    importance_policies = delivery_policy.get("importance_policies", {})
    email_settings = delivery_policy.get("email_settings", {})
    calendar_settings = delivery_policy.get("calendar_settings", {})
    reading_list_settings = delivery_policy.get("reading_list_settings", {})
    colleague_sharing_settings = delivery_policy.get("colleague_sharing_settings", {})
    
    # Get policy for this importance level
    policy = importance_policies.get(paper.importance, {})
    priority_label = policy.get("priority_label", "normal")
    
    # Initialize result
    researcher_actions: List[ResearcherAction] = []
    colleague_actions: List[ColleagueAction] = []
    files_to_write: List[FileToWrite] = []
    
    timestamp = _get_file_timestamp()
    
    # =================================
    # Researcher Actions
    # =================================
    
    # 1. Email notification
    if policy.get("send_email", False) and email_settings.get("enabled", True):
        action = ResearcherAction(
            action_type="email",
            paper_id=paper.arxiv_id,
            paper_title=paper.title,
            priority=priority_label,
            details={
                "subject": f"[ResearchPulse] {priority_label}: {paper.title[:50]}",
                "include_abstract": email_settings.get("include_abstract", True),
                "digest_mode": email_settings.get("digest_mode", False),
            }
        )
        researcher_actions.append(action)
        
        # Generate email file
        if email_settings.get("simulate_output", True):
            email_content = _generate_email_content(
                paper,
                priority_label,
                include_abstract=email_settings.get("include_abstract", True),
                include_explanation=email_settings.get("include_relevance_explanation", True),
                researcher_name=researcher_name,
            )
            file_path = f"emails/email_{paper.arxiv_id.replace('.', '_')}_{timestamp}.txt"
            files_to_write.append(FileToWrite(
                file_type="email",
                file_path=file_path,
                content=email_content,
                description=f"Email notification for paper {paper.arxiv_id}",
            ))
            
            # Send real email ONLY if NOT in digest mode
            # In digest mode, emails are collected and sent as a single digest at the end of the run
            digest_mode = email_settings.get("digest_mode", False)
            if researcher_email and not digest_mode:
                try:
                    _send_email_smtp(
                        to_email=researcher_email,
                        subject=f"[ResearchPulse] {priority_label.upper()}: New paper - {paper.title}",
                        body=email_content,
                    )
                except Exception as e:
                    print(f"Warning: Could not send email: {e}")
    
    # 2. Calendar entry
    if policy.get("create_calendar_entry", False) and calendar_settings.get("enabled", True):
        action = ResearcherAction(
            action_type="calendar",
            paper_id=paper.arxiv_id,
            paper_title=paper.title,
            priority=priority_label,
            details={
                "event_title": f"Read: {paper.title[:50]}",
                "duration_minutes": calendar_settings.get("event_duration_minutes", 30),
            }
        )
        researcher_actions.append(action)
        
        # Generate .ics file
        if calendar_settings.get("simulate_output", True):
            ics_content = _generate_calendar_ics(
                paper,
                event_duration_minutes=calendar_settings.get("event_duration_minutes", 30),
                reminder_minutes=calendar_settings.get("default_reminder_minutes", 60),
                schedule_within_days=calendar_settings.get("schedule_within_days", 7),
            )
            file_path = f"calendar/event_{paper.arxiv_id.replace('.', '_')}_{timestamp}.ics"
            files_to_write.append(FileToWrite(
                file_type="calendar",
                file_path=file_path,
                content=ics_content,
                description=f"Calendar event for reading paper {paper.arxiv_id}",
            ))
    
    # 3. Reading list
    if policy.get("add_to_reading_list", False) and reading_list_settings.get("enabled", True):
        action = ResearcherAction(
            action_type="reading_list",
            paper_id=paper.arxiv_id,
            paper_title=paper.title,
            priority=priority_label,
            details={
                "include_link": reading_list_settings.get("include_link", True),
                "include_importance": reading_list_settings.get("include_importance", True),
            }
        )
        researcher_actions.append(action)
        
        # Generate reading list entry
        reading_list_entry = _generate_reading_list_entry(paper)
        file_path = "reading_list.md"
        files_to_write.append(FileToWrite(
            file_type="reading_list",
            file_path=file_path,
            content=reading_list_entry,
            description=f"Reading list entry for paper {paper.arxiv_id}",
        ))
    
    # 4. Always log the decision
    researcher_actions.append(ResearcherAction(
        action_type="log",
        paper_id=paper.arxiv_id,
        paper_title=paper.title,
        priority=priority_label,
        details={
            "importance": paper.importance,
            "relevance_score": paper.relevance_score,
            "novelty_score": paper.novelty_score,
            "timestamp": _get_timestamp(),
        }
    ))
    
    # =================================
    # Colleague Actions
    # =================================
    
    if (policy.get("allow_colleague_sharing", False) and 
        colleague_sharing_settings.get("enabled", True)):
        
        # Extract paper topics from title and abstract for matching
        paper_text = f"{paper.title} {paper.abstract}".lower()
        paper_topics = []
        # Simple keyword extraction from paper
        for word in paper_text.split():
            word = word.strip(".,!?;:()[]{}\"'")
            if len(word) > 4:
                paper_topics.append(word)
        
        for colleague in colleague_objs:
            # Skip on_request colleagues unless explicitly requested
            if (colleague.sharing_preference == "on_request" and 
                colleague_sharing_settings.get("respect_sharing_preferences", True)):
                colleague_actions.append(ColleagueAction(
                    action_type="skip",
                    colleague_id=colleague.id,
                    colleague_name=colleague.name,
                    colleague_email=colleague.email,
                    paper_id=paper.arxiv_id,
                    paper_title=paper.title,
                    relevance_reason="Colleague prefers on-request sharing only",
                ))
                continue
            
            # Check topic overlap
            has_topic_match, matching_topics = _topics_overlap(
                paper_topics, colleague.topics
            )
            
            # Check category overlap
            has_category_match = _categories_overlap(
                paper.categories, colleague.arxiv_categories_interest
            )
            
            # Decide if we should share
            should_share = has_topic_match or has_category_match
            
            # For high importance papers, share even with weak matches
            if paper.importance == "high" and colleague.topics:
                should_share = True
            
            if not should_share:
                colleague_actions.append(ColleagueAction(
                    action_type="skip",
                    colleague_id=colleague.id,
                    colleague_name=colleague.name,
                    colleague_email=colleague.email,
                    paper_id=paper.arxiv_id,
                    paper_title=paper.title,
                    relevance_reason="No significant topic or category overlap",
                ))
                continue
            
            # Map sharing preference to action type
            action_map = {
                "immediate": "share_immediate",
                "daily_digest": "share_daily",
                "weekly_digest": "share_weekly",
                "daily": "share_daily",
                "weekly": "share_weekly",
                "monthly": "share_weekly",  # treat monthly as weekly
                "never": "skip",
            }
            action_type = action_map.get(colleague.sharing_preference, "share_daily")
            
            # Generate relevance reason
            relevance_reason = _get_colleague_relevance_reason(
                paper, colleague, matching_topics, has_category_match
            )
            
            colleague_actions.append(ColleagueAction(
                action_type=action_type,
                colleague_id=colleague.id,
                colleague_name=colleague.name,
                colleague_email=colleague.email,
                paper_id=paper.arxiv_id,
                paper_title=paper.title,
                relevance_reason=relevance_reason,
                details={
                    "matching_topics": matching_topics,
                    "has_category_match": has_category_match,
                }
            ))
            
            # Generate share file for immediate shares
            if (action_type == "share_immediate" and 
                colleague_sharing_settings.get("simulate_output", True)):
                share_content = _generate_share_content(paper, colleague, relevance_reason)
                colleague_slug = colleague.id.replace("_", "-")
                file_path = f"shares/share_{colleague_slug}_{paper.arxiv_id.replace('.', '_')}_{timestamp}.txt"
                files_to_write.append(FileToWrite(
                    file_type="share",
                    file_path=file_path,
                    content=share_content,
                    description=f"Share notification to {colleague.name} for paper {paper.arxiv_id}",
                ))
    
    # =================================
    # Generate Summary
    # =================================
    
    action_counts = {
        "email": sum(1 for a in researcher_actions if a.action_type == "email"),
        "calendar": sum(1 for a in researcher_actions if a.action_type == "calendar"),
        "reading_list": sum(1 for a in researcher_actions if a.action_type == "reading_list"),
    }
    
    share_counts = {
        "immediate": sum(1 for a in colleague_actions if a.action_type == "share_immediate"),
        "daily": sum(1 for a in colleague_actions if a.action_type == "share_daily"),
        "weekly": sum(1 for a in colleague_actions if a.action_type == "share_weekly"),
        "skipped": sum(1 for a in colleague_actions if a.action_type == "skip"),
    }
    
    summary_parts = [f"Paper '{paper.title[:40]}...' ({paper.importance} importance):"]
    
    researcher_summary = []
    if action_counts["email"]:
        researcher_summary.append("email")
    if action_counts["calendar"]:
        researcher_summary.append("calendar")
    if action_counts["reading_list"]:
        researcher_summary.append("reading list")
    
    if researcher_summary:
        summary_parts.append(f"Researcher: {', '.join(researcher_summary)}")
    else:
        summary_parts.append("Researcher: log only")
    
    if share_counts["immediate"] + share_counts["daily"] + share_counts["weekly"] > 0:
        share_summary = []
        if share_counts["immediate"]:
            share_summary.append(f"{share_counts['immediate']} immediate")
        if share_counts["daily"]:
            share_summary.append(f"{share_counts['daily']} daily digest")
        if share_counts["weekly"]:
            share_summary.append(f"{share_counts['weekly']} weekly digest")
        summary_parts.append(f"Colleagues: {', '.join(share_summary)}")
    
    summary_parts.append(f"Files: {len(files_to_write)} to write")
    
    return DeliveryDecisionResult(
        paper_id=paper.arxiv_id,
        paper_title=paper.title,
        importance=paper.importance,
        researcher_actions=researcher_actions,
        colleague_actions=colleague_actions,
        files_to_write=files_to_write,
        summary=" | ".join(summary_parts),
    )


def decide_delivery_action_json(
    scored_paper: Dict[str, Any],
    delivery_policy: Dict[str, Any],
    colleagues: List[Dict[str, Any]],
    artifacts_dir: str = "artifacts",
    researcher_name: str = "Researcher",
    researcher_email: str = "",
) -> dict:
    """
    JSON-serializable version of decide_delivery_action.
    
    Returns:
        Dictionary with delivery decision results
    """
    result = decide_delivery_action(
        scored_paper=scored_paper,
        delivery_policy=delivery_policy,
        colleagues=colleagues,
        artifacts_dir=artifacts_dir,
        researcher_name=researcher_name,
        researcher_email=researcher_email,
    )
    return result.model_dump()


# =============================================================================
# File Writing Functions
# =============================================================================

def write_artifact_files(
    files_to_write: List[FileToWrite],
    base_dir: str = "artifacts",
) -> Dict[str, Any]:
    """
    Actually write the artifact files to disk.
    
    Args:
        files_to_write: List of FileToWrite objects
        base_dir: Base directory for artifacts (default: "artifacts")
        
    Returns:
        Dictionary with status and written file paths
    """
    base_path = Path(base_dir)
    written_files = []
    errors = []
    
    for file_info in files_to_write:
        try:
            # Build full path
            full_path = base_path / file_info.file_path
            
            # Create directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # For reading list, append instead of overwrite
            if file_info.file_type == "reading_list":
                # Add header if file doesn't exist
                if not full_path.exists():
                    header = f"# Reading List\n\nGenerated by ResearchPulse\n\n---\n\n"
                    full_path.write_text(header + file_info.content, encoding="utf-8")
                else:
                    # Append to existing file
                    with open(full_path, "a", encoding="utf-8") as f:
                        f.write("\n---\n\n" + file_info.content)
            else:
                # Write new file
                full_path.write_text(file_info.content, encoding="utf-8")
            
            written_files.append(str(full_path))
        except Exception as e:
            errors.append({"file": file_info.file_path, "error": str(e)})
    
    return {
        "success": len(errors) == 0,
        "written_files": written_files,
        "file_count": len(written_files),
        "errors": errors,
    }


def decide_and_write_artifacts(
    scored_paper: Dict[str, Any],
    delivery_policy: Dict[str, Any],
    colleagues: List[Dict[str, Any]],
    artifacts_dir: str = "artifacts",
    write_files: bool = True,
) -> Dict[str, Any]:
    """
    Combined function: decide delivery actions and optionally write files.
    
    Args:
        scored_paper: Paper with scoring results
        delivery_policy: Delivery policy configuration
        colleagues: List of colleagues
        artifacts_dir: Base directory for artifacts
        write_files: Whether to actually write files (default: True)
        
    Returns:
        Dictionary with decision results and write status
    """
    # Get delivery decision
    result = decide_delivery_action(
        scored_paper=scored_paper,
        delivery_policy=delivery_policy,
        colleagues=colleagues,
        artifacts_dir=artifacts_dir,
    )
    
    output = result.model_dump()
    
    # Optionally write files
    if write_files and result.files_to_write:
        write_result = write_artifact_files(result.files_to_write, artifacts_dir)
        output["write_status"] = write_result
    
    return output


# =============================================================================
# LangChain Tool Definition (for ReAct agent)
# =============================================================================

DECIDE_DELIVERY_DESCRIPTION = """
Decide delivery actions for a scored paper based on importance and policy.

Input:
- scored_paper: Paper with scoring results (arxiv_id, title, importance, relevance_score, novelty_score, etc.)
- delivery_policy: Policy configuration from delivery_policy.json
- colleagues: List of colleagues from colleagues.json

Output:
- researcher_actions: List of actions for the researcher (email, calendar, reading_list, log)
- colleague_actions: List of sharing actions for colleagues
- files_to_write: Simulated output files (emails, .ics, reading_list.md, shares)
- summary: Human-readable summary of decisions

Action Logic:
- High importance: email + calendar + reading list + colleague sharing
- Medium importance: reading list + colleague sharing
- Low importance: log only

Colleague sharing considers:
- Topic overlap between paper and colleague interests
- arXiv category alignment
- Colleague sharing preference (immediate/daily/weekly/on_request)

Use this tool after scoring papers to determine what notifications to generate.
"""

DECIDE_DELIVERY_SCHEMA = {
    "name": "decide_delivery_action",
    "description": DECIDE_DELIVERY_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "scored_paper": {
                "type": "object",
                "description": "Paper with scoring results",
                "properties": {
                    "arxiv_id": {"type": "string", "description": "arXiv paper ID"},
                    "title": {"type": "string", "description": "Paper title"},
                    "abstract": {"type": "string", "description": "Paper abstract"},
                    "link": {"type": "string", "description": "URL to paper"},
                    "authors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of authors"
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "arXiv categories"
                    },
                    "relevance_score": {"type": "number", "description": "Relevance 0-1"},
                    "novelty_score": {"type": "number", "description": "Novelty 0-1"},
                    "importance": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Importance level"
                    },
                    "explanation": {"type": "string", "description": "Scoring explanation"}
                },
                "required": ["arxiv_id", "title", "importance", "relevance_score", "novelty_score"]
            },
            "delivery_policy": {
                "type": "object",
                "description": "Delivery policy configuration"
            },
            "colleagues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                        "topics": {"type": "array", "items": {"type": "string"}},
                        "sharing_preference": {"type": "string"},
                        "arxiv_categories_interest": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "description": "List of colleagues"
            },
            "artifacts_dir": {
                "type": "string",
                "description": "Base directory for simulated outputs",
                "default": "artifacts"
            }
        },
        "required": ["scored_paper", "delivery_policy", "colleagues"]
    }
}


# =============================================================================
# Self-Check
# =============================================================================

def self_check() -> bool:
    """
    Run self-check tests for decide_delivery_action tool.
    
    Returns:
        True if all checks pass, False otherwise.
    """
    print("=" * 60)
    print("decide_delivery_action Self-Check")
    print("=" * 60)
    
    all_passed = True
    
    def check(name: str, condition: bool) -> bool:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        return condition

    # Test delivery policy (similar to demo)
    test_policy = {
        "importance_policies": {
            "high": {
                "notify_researcher": True,
                "send_email": True,
                "create_calendar_entry": True,
                "add_to_reading_list": True,
                "allow_colleague_sharing": True,
                "priority_label": "urgent"
            },
            "medium": {
                "notify_researcher": True,
                "send_email": False,
                "create_calendar_entry": False,
                "add_to_reading_list": True,
                "allow_colleague_sharing": True,
                "priority_label": "normal"
            },
            "low": {
                "notify_researcher": False,
                "send_email": False,
                "create_calendar_entry": False,
                "add_to_reading_list": False,
                "allow_colleague_sharing": False,
                "priority_label": "low"
            }
        },
        "email_settings": {
            "enabled": True,
            "simulate_output": True,
            "include_abstract": True,
            "include_relevance_explanation": True,
        },
        "calendar_settings": {
            "enabled": True,
            "simulate_output": True,
            "event_duration_minutes": 30,
            "default_reminder_minutes": 60,
        },
        "reading_list_settings": {
            "enabled": True,
            "include_link": True,
            "include_importance": True,
        },
        "colleague_sharing_settings": {
            "enabled": True,
            "simulate_output": True,
            "respect_sharing_preferences": True,
        }
    }

    # Test colleagues (similar to demo)
    test_colleagues = [
        {
            "id": "colleague_001",
            "name": "Dr. Wei Chen",
            "email": "wei.chen@berkeley.edu",
            "topics": ["GPU optimization", "efficient inference"],
            "sharing_preference": "immediate",
            "arxiv_categories_interest": ["cs.LG"],
        },
        {
            "id": "colleague_002",
            "name": "Dr. Sarah Kim",
            "email": "s.kim@mit.edu",
            "topics": ["transformers", "attention"],
            "sharing_preference": "daily_digest",
            "arxiv_categories_interest": ["cs.CL"],
        },
        {
            "id": "colleague_003",
            "name": "Dr. James Liu",
            "email": "jliu@princeton.edu",
            "topics": ["neural machine translation"],
            "sharing_preference": "on_request",
            "arxiv_categories_interest": ["cs.CL"],
        },
    ]

    # Test 1: High importance paper
    print("\n1. High Importance Paper:")
    try:
        paper = {
            "arxiv_id": "2501.00001",
            "title": "Efficient Transformer Architectures for LLMs",
            "abstract": "We present efficient inference methods using transformers and attention.",
            "link": "https://arxiv.org/abs/2501.00001",
            "authors": ["Alice Smith", "Bob Jones"],
            "categories": ["cs.CL", "cs.LG"],
            "relevance_score": 0.85,
            "novelty_score": 0.75,
            "importance": "high",
            "explanation": "Highly relevant to your research",
        }
        result = decide_delivery_action(paper, test_policy, test_colleagues)
        all_passed &= check("returns DeliveryDecisionResult", isinstance(result, DeliveryDecisionResult))
        all_passed &= check("has email action", any(a.action_type == "email" for a in result.researcher_actions))
        all_passed &= check("has calendar action", any(a.action_type == "calendar" for a in result.researcher_actions))
        all_passed &= check("has reading_list action", any(a.action_type == "reading_list" for a in result.researcher_actions))
        all_passed &= check("has email file", any(f.file_type == "email" for f in result.files_to_write))
        all_passed &= check("has ics file", any(f.file_type == "calendar" for f in result.files_to_write))
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 2: Medium importance paper
    print("\n2. Medium Importance Paper:")
    try:
        paper = {
            "arxiv_id": "2501.00002",
            "title": "LLM Training Improvements",
            "abstract": "Training large language models.",
            "categories": ["cs.CL"],
            "relevance_score": 0.55,
            "novelty_score": 0.60,
            "importance": "medium",
        }
        result = decide_delivery_action(paper, test_policy, test_colleagues)
        all_passed &= check("no email action", not any(a.action_type == "email" for a in result.researcher_actions))
        all_passed &= check("no calendar action", not any(a.action_type == "calendar" for a in result.researcher_actions))
        all_passed &= check("has reading_list action", any(a.action_type == "reading_list" for a in result.researcher_actions))
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 3: Low importance paper
    print("\n3. Low Importance Paper:")
    try:
        paper = {
            "arxiv_id": "2501.00003",
            "title": "Random Topic Paper",
            "abstract": "Something unrelated.",
            "categories": ["cs.DC"],
            "relevance_score": 0.15,
            "novelty_score": 0.30,
            "importance": "low",
        }
        result = decide_delivery_action(paper, test_policy, test_colleagues)
        all_passed &= check("only log action", 
            len([a for a in result.researcher_actions if a.action_type != "log"]) == 0)
        all_passed &= check("no colleague sharing", 
            all(a.action_type == "skip" for a in result.colleague_actions))
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 4: Colleague sharing - immediate
    print("\n4. Colleague Sharing - Immediate:")
    try:
        paper = {
            "arxiv_id": "2501.00004",
            "title": "GPU Optimization for Efficient Inference",
            "abstract": "Optimizing inference on GPUs.",
            "categories": ["cs.LG"],
            "relevance_score": 0.70,
            "novelty_score": 0.65,
            "importance": "high",
        }
        result = decide_delivery_action(paper, test_policy, test_colleagues)
        immediate_shares = [a for a in result.colleague_actions if a.action_type == "share_immediate"]
        all_passed &= check("has immediate share action", len(immediate_shares) > 0)
        all_passed &= check("immediate share to Wei Chen", 
            any(a.colleague_name == "Dr. Wei Chen" for a in immediate_shares))
        all_passed &= check("has share file for immediate", 
            any(f.file_type == "share" for f in result.files_to_write))
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 5: Colleague sharing - on_request skipped
    print("\n5. Colleague Sharing - On Request Skipped:")
    try:
        paper = {
            "arxiv_id": "2501.00005",
            "title": "Neural Machine Translation Advances",
            "abstract": "New NMT methods.",
            "categories": ["cs.CL"],
            "relevance_score": 0.65,
            "novelty_score": 0.60,
            "importance": "high",
        }
        result = decide_delivery_action(paper, test_policy, test_colleagues)
        # James Liu should be skipped (on_request)
        james_action = next((a for a in result.colleague_actions if a.colleague_name == "Dr. James Liu"), None)
        all_passed &= check("on_request colleague skipped", 
            james_action is not None and james_action.action_type == "skip")
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 6: Calendar ICS format
    print("\n6. Calendar ICS Format:")
    try:
        paper = {
            "arxiv_id": "2501.00006",
            "title": "Test Paper",
            "importance": "high",
            "relevance_score": 0.8,
            "novelty_score": 0.7,
        }
        result = decide_delivery_action(paper, test_policy, test_colleagues)
        ics_files = [f for f in result.files_to_write if f.file_type == "calendar"]
        all_passed &= check("has ics file", len(ics_files) > 0)
        if ics_files:
            content = ics_files[0].content
            all_passed &= check("ics has VCALENDAR", "BEGIN:VCALENDAR" in content)
            all_passed &= check("ics has VEVENT", "BEGIN:VEVENT" in content)
            all_passed &= check("ics has VALARM", "BEGIN:VALARM" in content)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 7: Reading list markdown format
    print("\n7. Reading List Markdown Format:")
    try:
        paper = {
            "arxiv_id": "2501.00007",
            "title": "Markdown Test Paper",
            "link": "https://arxiv.org/abs/2501.00007",
            "importance": "medium",
            "relevance_score": 0.5,
            "novelty_score": 0.6,
        }
        result = decide_delivery_action(paper, test_policy, test_colleagues)
        rl_files = [f for f in result.files_to_write if f.file_type == "reading_list"]
        all_passed &= check("has reading list file", len(rl_files) > 0)
        if rl_files:
            content = rl_files[0].content
            all_passed &= check("markdown has title", "## [" in content)
            all_passed &= check("markdown has importance", "**Importance**" in content)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 8: JSON output format
    print("\n8. JSON Output Format:")
    try:
        paper = {
            "arxiv_id": "2501.00008",
            "title": "JSON Test",
            "importance": "high",
            "relevance_score": 0.8,
            "novelty_score": 0.7,
        }
        result = decide_delivery_action_json(paper, test_policy, test_colleagues)
        all_passed &= check("returns dict", isinstance(result, dict))
        all_passed &= check("has researcher_actions", "researcher_actions" in result)
        all_passed &= check("has colleague_actions", "colleague_actions" in result)
        all_passed &= check("has files_to_write", "files_to_write" in result)
        all_passed &= check("has summary", "summary" in result)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 9: Tool schema validation
    print("\n9. Tool Schema:")
    all_passed &= check("schema has name", DECIDE_DELIVERY_SCHEMA["name"] == "decide_delivery_action")
    all_passed &= check("schema has description", len(DECIDE_DELIVERY_SCHEMA["description"]) > 100)
    all_passed &= check("scored_paper required", "scored_paper" in DECIDE_DELIVERY_SCHEMA["parameters"]["required"])

    # Test 10: Summary generation
    print("\n10. Summary Generation:")
    try:
        paper = {
            "arxiv_id": "2501.00010",
            "title": "Summary Test Paper with Long Title",
            "importance": "high",
            "relevance_score": 0.9,
            "novelty_score": 0.8,
        }
        result = decide_delivery_action(paper, test_policy, test_colleagues)
        all_passed &= check("summary not empty", len(result.summary) > 20)
        all_passed &= check("summary has paper title", "Summary Test" in result.summary)
        all_passed &= check("summary has importance", "high" in result.summary.lower())
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("All checks PASSED!")
    else:
        print("Some checks FAILED!")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = self_check()
    sys.exit(0 if success else 1)
