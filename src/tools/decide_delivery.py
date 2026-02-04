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
    added_at: Optional[str] = Field(None, description="ISO timestamp when paper was added")
    pdf_url: Optional[str] = Field(None, description="URL to PDF")
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
    added_by: Literal["manual", "email"] = Field(
        "manual", description="How the colleague was added: 'manual' by user, 'email' via email signup"
    )
    auto_send_emails: bool = Field(
        True, description="Whether to automatically send research update emails to this colleague"
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
    html_body: Optional[str] = None,
) -> bool:
    """
    Send email via SMTP using environment variables for configuration.
    
    Required environment variables:
        SMTP_HOST: SMTP server hostname (default: smtp.gmail.com)
        SMTP_PORT: SMTP server port (default: 587)
        SMTP_USER: SMTP username/email
        SMTP_PASSWORD: SMTP password or app-specific password
        SMTP_FROM: From email address (defaults to SMTP_USER)
    
    Args:
        to_email: Recipient email address
        subject: Email subject line
        body: Plain text body (fallback for email clients that don't support HTML)
        html_body: Optional HTML body for rich formatting
    
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
        # Create multipart message for HTML + plain text fallback
        if html_body:
            msg = MIMEMultipart("alternative")
        else:
            msg = MIMEMultipart()
        
        msg["From"] = smtp_from
        msg["To"] = to_email
        msg["Subject"] = subject
        
        # Clean up the plain text body (remove header lines if present)
        body_lines = body.split("\n")
        body_start = 0
        for i, line in enumerate(body_lines):
            if line.startswith("=" * 20):  # Start of content
                body_start = i
                break
        clean_body = "\n".join(body_lines[body_start:])
        
        # Attach plain text part (required for email clients that don't support HTML)
        msg.attach(MIMEText(clean_body, "plain", "utf-8"))
        
        # Attach HTML part if provided (preferred by modern email clients)
        if html_body:
            print(f"[DEBUG] SMTP: Attaching HTML part ({len(html_body)} chars)")
            msg.attach(MIMEText(html_body, "html", "utf-8"))
        else:
            print(f"[DEBUG] SMTP: NO HTML body provided, sending plain text only")
        
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


def _generate_email_content_html(
    paper: ScoredPaper,
    priority: str,
    include_abstract: bool = True,
    include_explanation: bool = True,
    researcher_name: str = "Researcher",
) -> str:
    """Generate beautiful HTML email content for a single paper notification."""
    from datetime import datetime
    
    # Determine priority styling
    priority_upper = priority.upper()
    priority_colors = {
        "HIGH": {"bg": "linear-gradient(135deg, #dc2626 0%, #991b1b 100%)", "badge": "#ef4444", "icon": "🔥", "label": "Must Read"},
        "MEDIUM": {"bg": "linear-gradient(135deg, #d97706 0%, #92400e 100%)", "badge": "#f59e0b", "icon": "📌", "label": "Recommended"},
        "LOW": {"bg": "linear-gradient(135deg, #059669 0%, #047857 100%)", "badge": "#10b981", "icon": "📖", "label": "For Later"},
    }
    style = priority_colors.get(priority_upper, priority_colors["MEDIUM"])
    
    # Format date
    current_date = datetime.now().strftime("%B %d, %Y")
    hour = datetime.now().hour
    greeting = "Good morning" if hour < 12 else ("Good afternoon" if hour < 17 else "Good evening")
    
    # Format authors
    authors_text = ""
    if paper.authors:
        if len(paper.authors) <= 3:
            authors_text = ", ".join(paper.authors)
        else:
            authors_text = f"{', '.join(paper.authors[:3])} and {len(paper.authors) - 3} others"
    
    # Format categories
    categories_html = ""
    if paper.categories:
        cats = paper.categories[:4]
        categories_html = " ".join([f'<span style="display:inline-block;background:#374151;color:#9ca3af;font-size:11px;padding:2px 8px;border-radius:10px;margin:2px;">{cat}</span>' for cat in cats])
    
    # Abstract section
    abstract_html = ""
    if include_abstract and paper.abstract:
        abstract_text = paper.abstract[:800] + ("..." if len(paper.abstract) > 800 else "")
        abstract_html = f'''
            <tr>
                <td style="padding:20px 30px;">
                    <table width="100%" cellpadding="0" cellspacing="0" border="0">
                        <tr>
                            <td style="background:#1e293b;border-radius:12px;padding:20px;border-left:4px solid {style["badge"]};">
                                <p style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:0 0 10px 0;">📄 Abstract</p>
                                <p style="color:#cbd5e1;font-size:14px;line-height:1.7;margin:0;">{abstract_text}</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        '''
    
    # Explanation section
    explanation_html = ""
    if include_explanation and paper.explanation:
        explanation_html = f'''
            <tr>
                <td style="padding:0 30px 20px 30px;">
                    <table width="100%" cellpadding="0" cellspacing="0" border="0">
                        <tr>
                            <td style="background:linear-gradient(135deg, #312e81 0%, #1e1b4b 100%);border-radius:12px;padding:20px;">
                                <p style="color:#a5b4fc;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:0 0 10px 0;">🤖 ResearchPulse Analysis</p>
                                <p style="color:#e0e7ff;font-size:14px;line-height:1.6;margin:0;">{paper.explanation}</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        '''
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>New Paper Alert - ResearchPulse</title>
    <!--[if mso]>
    <style type="text/css">
        body, table, td {{font-family: Arial, sans-serif !important;}}
    </style>
    <![endif]-->
</head>
<body style="margin:0;padding:0;background-color:#0f172a;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Oxygen,Ubuntu,sans-serif;">
    <div style="display:none;max-height:0;overflow:hidden;">{style["icon"]} {priority_upper} Priority: {paper.title[:60]}...</div>
    
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background-color:#0f172a;">
        <tr>
            <td align="center" style="padding:40px 20px;">
                <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="max-width:600px;background:#1e293b;border-radius:16px;overflow:hidden;box-shadow:0 25px 50px -12px rgba(0,0,0,0.5);">
                    
                    <!-- Header -->
                    <tr>
                        <td style="background:{style["bg"]};padding:30px;text-align:center;">
                            <h1 style="color:white;font-size:24px;margin:0 0 5px 0;">📚 ResearchPulse</h1>
                            <p style="color:rgba(255,255,255,0.8);font-size:13px;margin:0;">New Paper Alert • {current_date}</p>
                        </td>
                    </tr>
                    
                    <!-- Priority Badge -->
                    <tr>
                        <td style="padding:25px 30px 0 30px;text-align:center;">
                            <span style="display:inline-block;background:{style["badge"]};color:white;font-size:12px;font-weight:bold;padding:8px 20px;border-radius:20px;text-transform:uppercase;letter-spacing:1px;">
                                {style["icon"]} {style["label"]} • {priority_upper} PRIORITY
                            </span>
                        </td>
                    </tr>
                    
                    <!-- Greeting -->
                    <tr>
                        <td style="padding:25px 30px 15px 30px;">
                            <p style="color:#94a3b8;font-size:15px;margin:0;">{greeting}, <strong style="color:white;">{researcher_name}</strong>!</p>
                            <p style="color:#64748b;font-size:14px;margin:8px 0 0 0;">We found a paper that matches your research interests:</p>
                        </td>
                    </tr>
                    
                    <!-- Paper Card -->
                    <tr>
                        <td style="padding:0 30px 20px 30px;">
                            <table width="100%" cellpadding="0" cellspacing="0" border="0">
                                <tr>
                                    <td style="background:linear-gradient(135deg, #1e293b 0%, #0f172a 100%);border:1px solid #334155;border-left:4px solid {style["badge"]};border-radius:12px;padding:25px;">
                                        
                                        <!-- Title -->
                                        <h2 style="color:#f1f5f9;font-size:18px;line-height:1.4;margin:0 0 15px 0;">
                                            <a href="{paper.link or f'http://arxiv.org/abs/{paper.arxiv_id}'}" style="color:#f1f5f9;text-decoration:none;">
                                                {paper.title}
                                            </a>
                                        </h2>
                                        
                                        <!-- Meta info -->
                                        <p style="color:#94a3b8;font-size:13px;margin:0 0 12px 0;">
                                            👤 {authors_text or "Unknown authors"}
                                        </p>
                                        
                                        <p style="color:#64748b;font-size:12px;margin:0 0 15px 0;">
                                            📅 {paper.publication_date or "Recently published"} &nbsp;•&nbsp; 🏷️ {paper.arxiv_id}
                                        </p>
                                        
                                        <!-- Scores -->
                                        <table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom:15px;">
                                            <tr>
                                                <td width="50%" style="padding-right:10px;">
                                                    <div style="background:#0f172a;border-radius:8px;padding:12px;text-align:center;">
                                                        <p style="color:#94a3b8;font-size:10px;text-transform:uppercase;margin:0 0 4px 0;">Relevance</p>
                                                        <p style="color:#60a5fa;font-size:20px;font-weight:bold;margin:0;">{paper.relevance_score:.0%}</p>
                                                    </div>
                                                </td>
                                                <td width="50%" style="padding-left:10px;">
                                                    <div style="background:#0f172a;border-radius:8px;padding:12px;text-align:center;">
                                                        <p style="color:#94a3b8;font-size:10px;text-transform:uppercase;margin:0 0 4px 0;">Novelty</p>
                                                        <p style="color:#a78bfa;font-size:20px;font-weight:bold;margin:0;">{paper.novelty_score:.0%}</p>
                                                    </div>
                                                </td>
                                            </tr>
                                        </table>
                                        
                                        <!-- Categories -->
                                        {categories_html}
                                        
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    
                    {explanation_html}
                    
                    {abstract_html}
                    
                    <!-- CTA Button -->
                    <tr>
                        <td style="padding:10px 30px 30px 30px;text-align:center;">
                            <a href="{paper.link or f'http://arxiv.org/abs/{paper.arxiv_id}'}" style="display:inline-block;background:linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%);color:white;font-size:14px;font-weight:bold;padding:14px 35px;border-radius:25px;text-decoration:none;">
                                📄 Read Full Paper →
                            </a>
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td style="background:#0f172a;padding:25px 30px;border-top:1px solid #1e293b;text-align:center;">
                            <p style="color:#64748b;font-size:12px;margin:0 0 8px 0;">
                                🤖 Curated by <strong style="color:#94a3b8;">ResearchPulse AI</strong>
                            </p>
                            <p style="color:#475569;font-size:11px;margin:0;">
                                Keeping you at the forefront of research
                            </p>
                        </td>
                    </tr>
                    
                </table>
            </td>
        </tr>
    </table>
</body>
</html>'''
    
    return html


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


def generate_summary_email_html(
    papers: List[ScoredPaper],
    query_text: str = "",
    researcher_name: str = "Researcher",
    triggered_by: str = "agent",
    include_abstracts: bool = False,
) -> tuple:
    """
    Generate a beautifully formatted HTML summary email for all papers in a query.
    Groups papers by importance (HIGH, MEDIUM, LOW) with visual badges matching the web app.
    
    Args:
        papers: List of scored papers to include
        query_text: The original search query
        researcher_name: Name of the researcher
        triggered_by: "agent" for automatic or "user" for manual
        include_abstracts: Whether to include paper abstracts
        
    Returns:
        Tuple of (subject, plain_text_body, html_body)
    """
    from html import escape
    
    importance_order = {"high": 0, "medium": 1, "low": 2}
    sorted_papers = sorted(papers, key=lambda p: (importance_order.get(p.importance, 2), -p.relevance_score))
    
    groups = {"high": [], "medium": [], "low": []}
    for p in sorted_papers:
        groups[p.importance].append(p)
    
    high_count = len(groups["high"])
    medium_count = len(groups["medium"])
    low_count = len(groups["low"])
    total_count = len(papers)
    
    run_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    
    if high_count > 0:
        subject = f"[ResearchPulse] {total_count} papers found - {high_count} high importance"
    else:
        subject = f"[ResearchPulse] {total_count} papers found for your query"
    
    plain_lines = [
        "ResearchPulse Paper Summary",
        "===========================",
        "",
        f"Query: {query_text}" if query_text else "",
        f"Run: {run_date}",
        f"Total Papers: {total_count}",
        "",
    ]
    
    for importance in ["high", "medium", "low"]:
        if groups[importance]:
            plain_lines.append(f"\n{importance.upper()} IMPORTANCE ({len(groups[importance])} papers)")
            plain_lines.append("-" * 40)
            for p in groups[importance]:
                link = p.link or f"https://arxiv.org/abs/{p.arxiv_id}"
                plain_lines.append(f"\n• {p.title}")
                plain_lines.append(f"  Link: {link}")
                if p.explanation:
                    plain_lines.append(f"  Summary: {p.explanation[:150]}...")
    
    plain_lines.append("\n\nGenerated by ResearchPulse")
    plain_text = "\n".join(plain_lines)
    
    triggered_badge_text = "🤖 Sent by ResearchPulse" if triggered_by == "agent" else "👤 Sent by you"
    triggered_class = "agent" if triggered_by == "agent" else "user"
    
    # Get current date for newsletter edition
    edition_date = datetime.now().strftime("%B %d, %Y")
    greeting_time = "morning" if datetime.now().hour < 12 else "afternoon" if datetime.now().hour < 17 else "evening"
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ResearchPulse Newsletter</title>
    <!--[if mso]>
    <noscript>
        <xml>
            <o:OfficeDocumentSettings>
                <o:PixelsPerInch>96</o:PixelsPerInch>
            </o:OfficeDocumentSettings>
        </xml>
    </noscript>
    <![endif]-->
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #0f172a; color: #e2e8f0;">
    <!-- Preheader text (hidden) -->
    <div style="display: none; max-height: 0; overflow: hidden;">
        Your personalized research digest: {total_count} papers curated for you • {high_count} high importance finds
    </div>
    
    <!-- Main wrapper -->
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color: #0f172a;">
        <tr>
            <td align="center" style="padding: 40px 20px;">
                
                <!-- Newsletter container -->
                <table role="presentation" width="640" cellspacing="0" cellpadding="0" style="max-width: 640px; width: 100%;">
                    
                    <!-- Header with logo -->
                    <tr>
                        <td style="text-align: center; padding-bottom: 32px;">
                            <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                                <tr>
                                    <td style="text-align: center;">
                                        <div style="display: inline-block; background: linear-gradient(135deg, #a84370 0%, #ec4899 50%, #f472b6 100%); border-radius: 16px; padding: 16px 24px; margin-bottom: 16px;">
                                            <span style="font-size: 32px; vertical-align: middle;">📚</span>
                                            <span style="font-size: 28px; font-weight: 700; color: white; vertical-align: middle; margin-left: 8px; letter-spacing: -0.5px;">ResearchPulse</span>
                                        </div>
                                        <div style="color: #94a3b8; font-size: 13px; text-transform: uppercase; letter-spacing: 2px; margin-top: 8px;">
                                            Your AI Research Assistant
                                        </div>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    
                    <!-- Edition info bar -->
                    <tr>
                        <td style="background: linear-gradient(90deg, rgba(168, 67, 112, 0.2) 0%, rgba(236, 72, 153, 0.1) 50%, rgba(168, 67, 112, 0.2) 100%); border-radius: 8px; padding: 12px 24px; text-align: center; margin-bottom: 24px;">
                            <span style="color: #f472b6; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">
                                ✨ {edition_date} Edition
                            </span>
                        </td>
                    </tr>
                    
                    <!-- Main content card -->
                    <tr>
                        <td style="padding-top: 24px;">
                            <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background: linear-gradient(180deg, #1e293b 0%, #1a2234 100%); border-radius: 24px; overflow: hidden; border: 1px solid rgba(148, 163, 184, 0.1);">
                                
                                <!-- Hero section -->
                                <tr>
                                    <td style="padding: 40px 40px 32px; background: linear-gradient(135deg, rgba(168, 67, 112, 0.15) 0%, rgba(236, 72, 153, 0.05) 100%); border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                                        <h1 style="margin: 0 0 12px 0; font-size: 28px; font-weight: 700; color: #f8fafc; line-height: 1.3;">
                                            Good {greeting_time}, {escape(researcher_name)}! 👋
                                        </h1>
                                        <p style="margin: 0 0 20px 0; font-size: 16px; color: #94a3b8; line-height: 1.6;">
                                            I've analyzed the latest research and found <strong style="color: #f472b6;">{total_count} papers</strong> that match your interests. Here's your personalized digest.
                                        </p>
                                        
                                        <!-- Query info -->
                                        <div style="background: rgba(15, 23, 42, 0.5); border-radius: 12px; padding: 16px; border: 1px solid rgba(148, 163, 184, 0.1);">
                                            <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #64748b; margin-bottom: 6px;">Search Query</div>
                                            <div style="font-size: 15px; color: #e2e8f0; font-style: italic;">"{escape(query_text) if query_text else "Your research interests"}"</div>
                                        </div>
                                    </td>
                                </tr>
                                
                                <!-- Stats row -->
                                <tr>
                                    <td style="padding: 24px 40px; border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                                        <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                                            <tr>
                                                <td width="33%" style="text-align: center; padding: 16px;">
                                                    <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.05) 100%); border-radius: 16px; padding: 20px; border: 1px solid rgba(239, 68, 68, 0.2);">
                                                        <div style="font-size: 36px; font-weight: 700; color: #f87171; line-height: 1;">{high_count}</div>
                                                        <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px;">🔥 Must Read</div>
                                                    </div>
                                                </td>
                                                <td width="33%" style="text-align: center; padding: 16px;">
                                                    <div style="background: linear-gradient(135deg, rgba(251, 191, 36, 0.2) 0%, rgba(251, 191, 36, 0.05) 100%); border-radius: 16px; padding: 20px; border: 1px solid rgba(251, 191, 36, 0.2);">
                                                        <div style="font-size: 36px; font-weight: 700; color: #fbbf24; line-height: 1;">{medium_count}</div>
                                                        <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px;">📌 Worth Reading</div>
                                                    </div>
                                                </td>
                                                <td width="33%" style="text-align: center; padding: 16px;">
                                                    <div style="background: linear-gradient(135deg, rgba(74, 222, 128, 0.2) 0%, rgba(74, 222, 128, 0.05) 100%); border-radius: 16px; padding: 20px; border: 1px solid rgba(74, 222, 128, 0.2);">
                                                        <div style="font-size: 36px; font-weight: 700; color: #4ade80; line-height: 1;">{low_count}</div>
                                                        <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px;">📖 For Later</div>
                                                    </div>
                                                </td>
                                            </tr>
                                        </table>
                                    </td>
                                </tr>
'''
    
    importance_config = {
        "high": {
            "emoji": "🔥",
            "label": "Must Read",
            "icon": "⚡",
            "gradient": "linear-gradient(135deg, #ef4444 0%, #f87171 100%)",
            "bg_gradient": "linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.02) 100%)",
            "border_color": "rgba(239, 68, 68, 0.3)",
            "accent": "#f87171",
            "badge_text": "HIGH PRIORITY"
        },
        "medium": {
            "emoji": "📌",
            "label": "Worth Reading",
            "icon": "💡",
            "gradient": "linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%)",
            "bg_gradient": "linear-gradient(135deg, rgba(251, 191, 36, 0.1) 0%, rgba(251, 191, 36, 0.02) 100%)",
            "border_color": "rgba(251, 191, 36, 0.3)",
            "accent": "#fbbf24",
            "badge_text": "RECOMMENDED"
        },
        "low": {
            "emoji": "📖",
            "label": "For Later",
            "icon": "📚",
            "gradient": "linear-gradient(135deg, #22c55e 0%, #4ade80 100%)",
            "bg_gradient": "linear-gradient(135deg, rgba(74, 222, 128, 0.1) 0%, rgba(74, 222, 128, 0.02) 100%)",
            "border_color": "rgba(74, 222, 128, 0.3)",
            "accent": "#4ade80",
            "badge_text": "BROWSE"
        },
    }
    
    for importance in ["high", "medium", "low"]:
        if not groups[importance]:
            continue
        
        cfg = importance_config[importance]
        count = len(groups[importance])
        
        html += f'''
                                <!-- {importance.upper()} importance section -->
                                <tr>
                                    <td style="padding: 32px 40px;">
                                        <!-- Section header -->
                                        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="margin-bottom: 20px;">
                                            <tr>
                                                <td>
                                                    <div style="display: inline-block; background: {cfg["gradient"]}; border-radius: 24px; padding: 8px 20px;">
                                                        <span style="font-size: 14px; font-weight: 600; color: white; text-transform: uppercase; letter-spacing: 0.5px;">
                                                            {cfg["emoji"]} {cfg["label"]}
                                                        </span>
                                                    </div>
                                                    <span style="color: #64748b; font-size: 14px; margin-left: 12px;">{count} paper{"s" if count != 1 else ""}</span>
                                                </td>
                                            </tr>
                                        </table>
'''
        
        for idx, paper in enumerate(groups[importance]):
            link = paper.link or f"https://arxiv.org/abs/{paper.arxiv_id}"
            summary = escape(paper.explanation[:250] + "..." if paper.explanation and len(paper.explanation) > 250 else (paper.explanation or ""))
            authors_str = ", ".join(paper.authors[:3]) if paper.authors else ""
            if paper.authors and len(paper.authors) > 3:
                authors_str += f" et al."
            
            relevance_pct = int(paper.relevance_score * 100)
            
            html += f'''
                                        <!-- Paper card -->
                                        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background: {cfg["bg_gradient"]}; border-radius: 16px; margin-bottom: 16px; border: 1px solid {cfg["border_color"]};">
                                            <tr>
                                                <td style="padding: 24px;">
                                                    <!-- Paper number badge -->
                                                    <div style="display: inline-block; background: {cfg["gradient"]}; border-radius: 8px; padding: 4px 10px; margin-bottom: 12px;">
                                                        <span style="font-size: 11px; font-weight: 700; color: white; letter-spacing: 0.5px;">{cfg["badge_text"]}</span>
                                                    </div>
                                                    
                                                    <!-- Title -->
                                                    <h3 style="margin: 0 0 12px 0; font-size: 18px; font-weight: 600; line-height: 1.4;">
                                                        <a href="{link}" target="_blank" style="color: #f8fafc; text-decoration: none; border-bottom: 2px solid transparent;">{escape(paper.title)}</a>
                                                    </h3>
                                                    
                                                    <!-- Meta info -->
                                                    <div style="margin-bottom: 12px;">
                                                        {f'<span style="color: #94a3b8; font-size: 13px;">{escape(authors_str)}</span>' if authors_str else ''}
                                                        {f'<span style="color: #64748b; font-size: 13px;"> • {escape(paper.publication_date)}</span>' if paper.publication_date else ''}
                                                    </div>
                                                    
                                                    <!-- Summary from ResearchPulse -->
                                                    {f'<p style="margin: 0 0 16px 0; font-size: 14px; color: #cbd5e1; line-height: 1.6;">{summary}</p>' if summary else ''}
                                                    
                                                    <!-- Footer with relevance and CTA -->
                                                    <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                                                        <tr>
                                                            <td>
                                                                <span style="display: inline-block; background: rgba(15, 23, 42, 0.5); padding: 4px 12px; border-radius: 20px; font-size: 12px; color: #94a3b8;">
                                                                    📊 {relevance_pct}% match
                                                                </span>
                                                                {f'<span style="display: inline-block; background: rgba(15, 23, 42, 0.5); padding: 4px 12px; border-radius: 20px; font-size: 12px; color: #94a3b8; margin-left: 8px;">🏷️ {escape(", ".join(paper.categories[:2]))}</span>' if paper.categories else ''}
                                                            </td>
                                                            <td style="text-align: right;">
                                                                <a href="{link}" target="_blank" style="display: inline-block; background: {cfg["gradient"]}; color: white; font-size: 13px; font-weight: 600; padding: 8px 16px; border-radius: 8px; text-decoration: none;">
                                                                    Read Paper →
                                                                </a>
                                                            </td>
                                                        </tr>
                                                    </table>
                                                </td>
                                            </tr>
                                        </table>
'''
        
        # Add divider between sections (except after last)
        if importance != "low" and (groups.get("medium") or groups.get("low")):
            html += '''
                                        <!-- Section divider -->
                                        <div style="height: 1px; background: linear-gradient(90deg, transparent 0%, rgba(148, 163, 184, 0.2) 50%, transparent 100%); margin: 16px 0;"></div>
'''
        
        html += '''
                                    </td>
                                </tr>
'''
    
    html += f'''
                                <!-- Footer -->
                                <tr>
                                    <td style="padding: 32px 40px; background: linear-gradient(180deg, rgba(15, 23, 42, 0.5) 0%, rgba(15, 23, 42, 0.8) 100%); border-top: 1px solid rgba(148, 163, 184, 0.1);">
                                        <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                                            <tr>
                                                <td style="text-align: center; padding-bottom: 20px;">
                                                    <div style="display: inline-block; background: {'rgba(168, 67, 112, 0.2)' if triggered_by == 'agent' else 'rgba(59, 130, 246, 0.2)'}; border-radius: 20px; padding: 6px 16px; border: 1px solid {'rgba(168, 67, 112, 0.3)' if triggered_by == 'agent' else 'rgba(59, 130, 246, 0.3)'};">
                                                        <span style="font-size: 12px; color: {'#f472b6' if triggered_by == 'agent' else '#60a5fa'};">
                                                            {triggered_badge_text}
                                                        </span>
                                                    </div>
                                                </td>
                                            </tr>
                                            <tr>
                                                <td style="text-align: center;">
                                                    <p style="margin: 0 0 8px 0; font-size: 13px; color: #64748b;">
                                                        Curated with ❤️ by your AI research assistant
                                                    </p>
                                                    <p style="margin: 0; font-size: 12px; color: #475569;">
                                                        Visit your <a href="#" style="color: #f472b6; text-decoration: none;">ResearchPulse Dashboard</a> to manage preferences
                                                    </p>
                                                </td>
                                            </tr>
                                        </table>
                                    </td>
                                </tr>
                                
                            </table>
                        </td>
                    </tr>
                    
                    <!-- Bottom branding -->
                    <tr>
                        <td style="padding: 32px; text-align: center;">
                            <p style="margin: 0 0 12px 0; font-size: 13px; color: #64748b;">
                                📚 ResearchPulse • Your AI-Powered Research Companion
                            </p>
                            <p style="margin: 0; font-size: 11px; color: #475569;">
                                This is an automated newsletter based on your research interests.
                            </p>
                        </td>
                    </tr>
                    
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
'''
    
    return subject, plain_text, html


def send_summary_email_html(
    papers: List[ScoredPaper],
    researcher_email: str,
    query_text: str = "",
    researcher_name: str = "Researcher",
    triggered_by: str = "agent",
) -> bool:
    """
    Send a beautiful HTML summary email with all papers from a query.
    
    Args:
        papers: List of scored papers
        researcher_email: Recipient email
        query_text: The original query text
        researcher_name: Name of researcher
        triggered_by: "agent" for automatic or "user" for manual
        
    Returns:
        True if email was sent successfully
    """
    if not papers:
        print("No papers to include in summary email.")
        return False
    
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    smtp_from = os.environ.get("SMTP_FROM", smtp_user)
    
    if not smtp_user or not smtp_password:
        print("SMTP credentials not configured. Set SMTP_USER and SMTP_PASSWORD.")
        return False
    
    try:
        subject, plain_text, html_body = generate_summary_email_html(
            papers=papers,
            query_text=query_text,
            researcher_name=researcher_name,
            triggered_by=triggered_by,
        )
        
        msg = MIMEMultipart("alternative")
        msg["From"] = smtp_from
        msg["To"] = researcher_email
        msg["Subject"] = subject
        
        part1 = MIMEText(plain_text, "plain")
        part2 = MIMEText(html_body, "html")
        msg.attach(part1)
        msg.attach(part2)
        
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        print(f"✓ Summary email sent successfully to {researcher_email}")
        return True
        
    except smtplib.SMTPAuthenticationError:
        print("SMTP authentication failed.")
        return False
    except Exception as e:
        print(f"Failed to send summary email: {e}")
        return False


def send_digest_email(
    papers: List[ScoredPaper],
    researcher_email: str,
    researcher_name: str = "Researcher",
    email_settings: Optional[Dict[str, Any]] = None,
    query_text: str = "",
    triggered_by: str = "agent",
) -> bool:
    """
    Send a beautiful HTML newsletter-style digest email with all papers.
    
    This should be called at the end of a run instead of sending
    individual emails per paper. Uses the modern HTML template for
    professional newsletter appearance.
    
    Args:
        papers: List of scored papers to include
        researcher_email: Recipient email address
        researcher_name: Name of the researcher
        email_settings: Email settings from delivery policy
        query_text: The original search query (for context in email)
        triggered_by: "agent" for automatic or "user" for manual
        
    Returns:
        True if email was sent successfully
    """
    if not papers:
        print("No papers to include in digest email.")
        return False
    
    settings = email_settings or {}
    max_papers = settings.get("max_papers_per_email", 20)
    include_abstracts = settings.get("include_abstract", False)
    
    # Include all papers (high, medium, low) - let the template group them
    papers_to_send = papers[:max_papers]
    
    if not papers_to_send:
        print("No papers to send in digest.")
        return False
    
    # Generate beautiful HTML newsletter email
    subject, plain_text, html_body = generate_summary_email_html(
        papers=papers_to_send,
        query_text=query_text,
        researcher_name=researcher_name,
        triggered_by=triggered_by,
        include_abstracts=include_abstracts,
    )
    
    # Send via SMTP with HTML
    return _send_email_smtp(
        to_email=researcher_email,
        subject=subject,
        body=plain_text,
        html_body=html_body,
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
    owner_name: str = "a researcher",
) -> str:
    """Generate share notification content for a colleague.
    
    The tone of the email varies based on how the colleague was added:
    - 'email': The colleague signed up themselves, so we speak as ResearchPulse agent
    - 'manual': The owner added them, so we speak on behalf of the owner
    """
    first_name = colleague.name.split()[0] if colleague.name else "there"
    
    # Generate intro based on how colleague was added
    if colleague.added_by == "email":
        # Colleague signed up themselves via email
        intro_lines = [
            f"Hi {first_name},",
            "",
            f"This is ResearchPulse, the research paper assistant for {owner_name}.",
            "You asked me to share relevant papers with you, and I found one that",
            f"matches your interests in {', '.join(colleague.topics[:3]) if colleague.topics else 'your research area'}.",
        ]
    else:
        # Owner manually added this colleague
        intro_lines = [
            f"Hi {first_name},",
            "",
            f"{owner_name} wanted to share a relevant research paper with you.",
            "Based on your interests, this paper might be useful for your research",
            f"in {', '.join(colleague.topics[:3]) if colleague.topics else 'your area'}.",
        ]
    
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
    ]
    
    lines.extend(intro_lines)
    
    lines.extend([
        "",
        "-" * 40,
        "",
        f"Title: {paper.title}",
        f"arXiv ID: {paper.arxiv_id}",
    ])
    
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
    skip_colleague_sharing: bool = True,
) -> DeliveryDecisionResult:
    """
    Decide delivery actions for a scored paper.
    
    This tool determines what actions to take based on:
    1. Paper's importance level
    2. Delivery policy settings
    3. Colleague interests and sharing preferences
    
    IMPORTANT: ResearchPulse is the OWNER's research agent first.
    Colleagues only benefit from SURPLUS papers discovered during the 
    owner-focused workflow. By default, skip_colleague_sharing=True to
    defer colleague matching to post-processing (after owner's papers
    are selected). Use process_colleague_surplus() after owner selection.
    
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
        
        skip_colleague_sharing: If True (default), skip colleague sharing 
            in this call. Colleague sharing should happen in post-processing
            via process_colleague_surplus() after owner's papers are selected.
            
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
                    # Generate beautiful HTML version of the email
                    html_content = _generate_email_content_html(
                        paper,
                        priority_label,
                        include_abstract=email_settings.get("include_abstract", True),
                        include_explanation=email_settings.get("include_relevance_explanation", True),
                        researcher_name=researcher_name,
                    )
                    print(f"[DEBUG] Generated HTML email: {len(html_content)} chars, starts with: {html_content[:100]}")
                    _send_email_smtp(
                        to_email=researcher_email,
                        subject=f"[ResearchPulse] {priority_label.upper()}: New paper - {paper.title}",
                        body=email_content,
                        html_body=html_content,
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
    
    # IMPORTANT: ResearchPulse works for the OWNER first.
    # Colleague sharing is deferred to post-processing to ensure:
    # 1. Owner gets their requested top X papers
    # 2. Only SURPLUS papers (not selected for owner) go to colleagues
    # 3. Owner's experience is never degraded
    if skip_colleague_sharing:
        # Colleague sharing will happen in process_colleague_surplus() 
        # after owner's papers are finalized
        pass
    elif (policy.get("allow_colleague_sharing", False) and 
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
                share_content = _generate_share_content(paper, colleague, relevance_reason, owner_name=researcher_name)
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
# Colleague Sharing Processing (Post-Owner Selection)
# =============================================================================

def process_colleague_surplus(
    all_scored_papers: List[Dict[str, Any]],
    owner_paper_ids: List[str],
    colleagues: List[Dict[str, Any]],
    delivery_policy: Dict[str, Any],
    researcher_name: str = "Researcher",
    artifacts_dir: str = "artifacts",
) -> Dict[str, Any]:
    """
    Process papers for colleague sharing AFTER owner selection is finalized.
    
    CRITICAL PRINCIPLE:
    ResearchPulse is the OWNER's research agent first. This function is
    called AFTER the owner's papers are selected via enforce_output().
    
    KEY BEHAVIOR:
    - A paper CAN appear in both owner's results AND be sent to colleagues
    - Sharing with colleagues is NOT mutually exclusive with owner delivery
    - Owner's selection is NEVER affected - this runs AFTER owner selection
    - ALL papers (including owner's top X) can be shared with relevant colleagues
    
    Operational rules:
    1. Owner's paper selection is already finalized before this runs
    2. ALL scored papers are considered for colleague matching
    3. If a paper matches colleague interests, it's shared regardless of
       whether it was also selected for the owner
    4. This is opportunistic - no separate searches are done for colleagues
    
    Args:
        all_scored_papers: All papers that were scored during the run
        owner_paper_ids: List of arxiv_ids selected for the owner (for reference only)
        colleagues: List of colleague dictionaries with interests
        delivery_policy: Policy configuration with sharing settings
        researcher_name: Owner's name for email personalization
        artifacts_dir: Base directory for simulated outputs
        
    Returns:
        Dictionary with colleague_actions and files_to_write
    """
    # Process ALL papers for colleague matching (including owner's papers)
    # Owner sharing is NOT mutually exclusive with colleague sharing
    papers_to_check = all_scored_papers
    owner_ids_set = set(owner_paper_ids)
    
    if not papers_to_check:
        return {
            "success": True,
            "message": "No papers available for colleague sharing",
            "paper_count": 0,
            "colleague_actions": [],
            "files_to_write": [],
        }
    
    # Get colleague sharing settings
    colleague_sharing_settings = delivery_policy.get("colleague_sharing_settings", {})
    if not colleague_sharing_settings.get("enabled", True):
        return {
            "success": True,
            "message": "Colleague sharing is disabled in policy",
            "paper_count": len(papers_to_check),
            "colleague_actions": [],
            "files_to_write": [],
        }
    
    # Parse colleagues
    colleague_objs = [ColleagueInfo(**c) for c in colleagues]
    if not colleague_objs:
        return {
            "success": True,
            "message": "No colleagues configured",
            "paper_count": len(papers_to_check),
            "colleague_actions": [],
            "files_to_write": [],
        }
    
    # Filter to only auto-send enabled colleagues
    auto_send_colleagues = [
        c for c in colleague_objs 
        if getattr(c, 'auto_send_emails', True)
    ]
    
    if not auto_send_colleagues:
        return {
            "success": True,
            "message": "No colleagues have auto-send enabled",
            "paper_count": len(papers_to_check),
            "colleague_actions": [],
            "files_to_write": [],
        }
    
    all_colleague_actions: List[ColleagueAction] = []
    all_files_to_write: List[FileToWrite] = []
    timestamp = _get_file_timestamp()
    
    # Process ALL papers for colleague matching (including owner's papers)
    # A paper can be sent to BOTH owner AND colleagues - not mutually exclusive
    for paper_dict in papers_to_check:
        paper = ScoredPaper(**paper_dict)
        also_for_owner = paper.arxiv_id in owner_ids_set  # Track if also in owner's results
        
        # Extract paper topics from title and abstract for matching
        paper_text = f"{paper.title} {paper.abstract}".lower()
        paper_topics = []
        for word in paper_text.split():
            word = word.strip(".,!?;:()[]{}\"'")
            if len(word) > 4:
                paper_topics.append(word)
        
        for colleague in auto_send_colleagues:
            # Skip on_request colleagues unless explicitly requested
            if (colleague.sharing_preference == "on_request" and 
                colleague_sharing_settings.get("respect_sharing_preferences", True)):
                all_colleague_actions.append(ColleagueAction(
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
            
            # Decide if we should share this paper with colleagues
            should_share = has_topic_match or has_category_match
            
            # For high importance papers, share even with weak matches
            if paper.importance == "high" and colleague.topics:
                should_share = True
            
            if not should_share:
                all_colleague_actions.append(ColleagueAction(
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
                "monthly": "share_weekly",
                "never": "skip",
            }
            action_type = action_map.get(colleague.sharing_preference, "share_daily")
            
            # Generate relevance reason
            relevance_reason = _get_colleague_relevance_reason(
                paper, colleague, matching_topics, has_category_match
            )
            
            # Add marker if paper is also in owner's selection
            if also_for_owner:
                tagged_reason = f"[ALSO FOR OWNER] {relevance_reason}"
            else:
                tagged_reason = relevance_reason
            
            all_colleague_actions.append(ColleagueAction(
                action_type=action_type,
                colleague_id=colleague.id,
                colleague_name=colleague.name,
                colleague_email=colleague.email,
                paper_id=paper.arxiv_id,
                paper_title=paper.title,
                relevance_reason=tagged_reason,
                details={
                    "matching_topics": matching_topics,
                    "has_category_match": has_category_match,
                    "also_for_owner": also_for_owner,  # Track if sent to owner too
                }
            ))
            
            # Generate share file for immediate shares
            if (action_type == "share_immediate" and 
                colleague_sharing_settings.get("simulate_output", True)):
                share_content = _generate_share_content(
                    paper, colleague, relevance_reason, owner_name=researcher_name
                )
                colleague_slug = colleague.id.replace("_", "-")
                file_path = f"shares/colleague_share_{colleague_slug}_{paper.arxiv_id.replace('.', '_')}_{timestamp}.txt"
                all_files_to_write.append(FileToWrite(
                    file_type="share",
                    file_path=file_path,
                    content=share_content,
                    description=f"Paper share to {colleague.name} for {paper.arxiv_id}",
                ))
    
    # Generate summary
    share_counts = {
        "immediate": sum(1 for a in all_colleague_actions if a.action_type == "share_immediate"),
        "daily": sum(1 for a in all_colleague_actions if a.action_type == "share_daily"),
        "weekly": sum(1 for a in all_colleague_actions if a.action_type == "share_weekly"),
        "skipped": sum(1 for a in all_colleague_actions if a.action_type == "skip"),
    }
    
    total_shared = share_counts["immediate"] + share_counts["daily"] + share_counts["weekly"]
    
    return {
        "success": True,
        "message": f"Processed {len(papers_to_check)} papers for {len(auto_send_colleagues)} colleagues",
        "paper_count": len(papers_to_check),
        "owner_paper_count": len(owner_paper_ids),
        "share_counts": share_counts,
        "total_shares": total_shared,
        "colleague_actions": [a.model_dump() for a in all_colleague_actions],
        "files_to_write": [f.model_dump() for f in all_files_to_write],
    }


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
