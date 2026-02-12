"""
Shared email template system for ResearchPulse.

All emails use a unified design system:
- Dark theme (#0f172a background)
- Pink/purple gradient accents
- ResearchPulse logo (hosted URL)
- Mobile-friendly HTML
- Consistent header, footer, typography, and action buttons

This module provides reusable building blocks + complete templates
for both owner and colleague emails.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from html import escape
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# Logo Embedding
# =============================================================================

# Use hosted logo URL - more reliable for emails than base64 embedding
_LOGO_URL = "https://researchpulse-yp0b.onrender.com/static/public/logo.png"


def get_logo_html(width: int = 40, height: int = 40) -> str:
    """Return an <img> tag with the ResearchPulse logo URL."""
    return (
        f'<img src="{_LOGO_URL}" alt="ResearchPulse" '
        f'width="{width}" height="{height}" '
        f'style="display:inline-block;vertical-align:middle;border:0;outline:none;" />'
    )


# =============================================================================
# Shared HTML building blocks
# =============================================================================

def _email_preheader(text: str) -> str:
    """Hidden preheader text for email clients."""
    return f'<div style="display:none;max-height:0;overflow:hidden;">{escape(text)}</div>'


def email_header(subtitle: str = "") -> str:
    """Render the shared email header with logo and branding."""
    logo = get_logo_html(width=36, height=36)
    sub = ""
    if subtitle:
        sub = f'<p style="color:rgba(255,255,255,0.8);font-size:13px;margin:8px 0 0 0;">{escape(subtitle)}</p>'
    return f'''
    <!-- Header -->
    <tr>
        <td style="background:linear-gradient(135deg, #a84370 0%, #ec4899 50%, #f472b6 100%);padding:28px 30px;text-align:center;border-radius:16px 16px 0 0;">
            <div style="display:inline-block;vertical-align:middle;">
                {logo}
            </div>
            <span style="font-size:26px;font-weight:700;color:white;vertical-align:middle;margin-left:10px;letter-spacing:-0.5px;">ResearchPulse</span>
            {sub}
        </td>
    </tr>'''


def email_footer(extra_links: str = "") -> str:
    """Render the shared email footer."""
    return f'''
    <!-- Footer -->
    <tr>
        <td style="background:#0f172a;padding:25px 30px;border-top:1px solid #1e293b;text-align:center;border-radius:0 0 16px 16px;">
            {extra_links}
            <p style="color:#64748b;font-size:12px;margin:8px 0 4px 0;">
                Curated with ❤️ by <strong style="color:#94a3b8;">ResearchPulse AI</strong>
            </p>
            <p style="color:#475569;font-size:11px;margin:0;">
                Keeping you at the forefront of research
            </p>
        </td>
    </tr>'''


def email_wrapper_start() -> str:
    """Opening HTML/body/table tags for all ResearchPulse emails."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!--[if mso]><style>body,table,td{font-family:Arial,sans-serif !important;}</style><![endif]-->
</head>
<body style="margin:0;padding:0;background-color:#0f172a;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Oxygen,Ubuntu,sans-serif;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background-color:#0f172a;">
<tr><td align="center" style="padding:40px 20px;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="max-width:600px;background:#1e293b;border-radius:16px;overflow:hidden;box-shadow:0 25px 50px -12px rgba(0,0,0,0.5);">'''


def email_wrapper_end() -> str:
    """Closing tags for all ResearchPulse emails."""
    return '''
</table>
</td></tr></table>
</body>
</html>'''


def action_button(label: str, url: str, color: str = "linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%)") -> str:
    """Render a responsive action button."""
    return (
        f'<a href="{escape(url)}" target="_blank" '
        f'style="display:inline-block;background:{color};color:white;font-size:14px;'
        f'font-weight:bold;padding:12px 28px;border-radius:25px;text-decoration:none;'
        f'text-align:center;">{escape(label)}</a>'
    )


def section_title(title: str) -> str:
    """Inline section heading inside an email body cell."""
    return (
        f'<p style="color:#94a3b8;font-size:11px;text-transform:uppercase;'
        f'letter-spacing:1px;margin:0 0 10px 0;">{escape(title)}</p>'
    )


def info_box(content: str, border_color: str = "#8b5cf6") -> str:
    """Rounded info box for call-outs."""
    return (
        f'<table width="100%" cellpadding="0" cellspacing="0" border="0">'
        f'<tr><td style="background:#1e293b;border-radius:12px;padding:20px;'
        f'border-left:4px solid {border_color};">'
        f'{content}</td></tr></table>'
    )


# =============================================================================
# Colleague management links
# =============================================================================

def colleague_management_links(
    remove_url: str,
    update_url: str,
) -> str:
    """Render management info for colleague emails (removal via email reply)."""
    return ''


def colleague_management_footer_links(remove_url: str, update_url: str) -> str:
    """Inline footer text for emails (removal via email reply)."""
    return ''


# =============================================================================
# Complete email templates
# =============================================================================


def render_colleague_paper_email(
    paper: Dict[str, Any],
    colleague_name: str,
    relevance_reason: str,
    matched_interests: List[str],
    remove_url: str = "",
    update_url: str = "",
    owner_name: str = "a researcher",
) -> tuple[str, str, str]:
    """
    Render a paper-share email for a colleague.

    Returns (subject, plain_text, html).
    """
    title = paper.get("title", "Untitled Paper")
    arxiv_id = paper.get("arxiv_id", "")
    authors = paper.get("authors", [])
    categories = paper.get("categories", [])
    pub_date = paper.get("publication_date", "")
    abstract = (paper.get("abstract") or "")[:600]
    if len(paper.get("abstract", "") or "") > 600:
        abstract += "..."
    link = paper.get("link") or f"https://arxiv.org/abs/{arxiv_id}"

    first_name = colleague_name.split()[0] if colleague_name else "there"
    authors_str = ", ".join(authors[:3])
    if len(authors) > 3:
        authors_str += f" et al."
    cats_str = ", ".join(categories[:3]) if categories else ""
    interests_str = ", ".join(matched_interests) if matched_interests else "your research area"

    # Subject
    subject = f"Paper for you: {title[:80]}"

    # Plain text
    plain = (
        f"Hi {first_name},\n\n"
        f"ResearchPulse found a paper matching your interests ({interests_str}).\n\n"
        f"Title: {title}\n"
        f"Authors: {authors_str}\n"
        f"arXiv: {arxiv_id}\n"
        f"Link: {link}\n\n"
        f"Why: {relevance_reason}\n\n"
    )
    if abstract:
        plain += f"Abstract:\n{abstract}\n\n"
    if remove_url:
        plain += f"Remove me: {remove_url}\n"
    if update_url:
        plain += f"Update interests: {update_url}\n"
    plain += "\n-- ResearchPulse\n"

    # HTML
    mgmt = ""
    if remove_url or update_url:
        mgmt = colleague_management_links(remove_url or "#", update_url or "#")
    footer_links = ""
    if remove_url or update_url:
        footer_links = colleague_management_footer_links(remove_url or "#", update_url or "#")

    html = email_wrapper_start()
    html += email_header(subtitle=f"Paper Recommendation • {datetime.now().strftime('%B %d, %Y')}")

    # Greeting
    html += f'''
    <tr><td style="padding:25px 30px 15px 30px;">
        <p style="color:#94a3b8;font-size:15px;margin:0;">Hi <strong style="color:white;">{escape(first_name)}</strong>,</p>
        <p style="color:#64748b;font-size:14px;margin:8px 0 0 0;">We found a paper matching your interest in <strong style="color:#f472b6;">{escape(interests_str)}</strong>:</p>
    </td></tr>'''

    # Paper card
    html += f'''
    <tr><td style="padding:0 30px 20px 30px;">
        <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr><td style="background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%);border:1px solid #334155;border-left:4px solid #ec4899;border-radius:12px;padding:25px;">
            <h2 style="color:#f1f5f9;font-size:18px;line-height:1.4;margin:0 0 12px 0;">
                <a href="{escape(link)}" style="color:#f1f5f9;text-decoration:none;">{escape(title)}</a>
            </h2>
            <p style="color:#94a3b8;font-size:13px;margin:0 0 8px 0;">👤 {escape(authors_str)}</p>
            <p style="color:#64748b;font-size:12px;margin:0 0 8px 0;">📅 {escape(pub_date or "Recently published")} &nbsp;•&nbsp; 🏷️ {escape(arxiv_id)}</p>'''

    if cats_str:
        html += f'<p style="color:#64748b;font-size:12px;margin:0 0 8px 0;">📂 {escape(cats_str)}</p>'

    html += f'''
            <p style="color:#a5b4fc;font-size:12px;margin:10px 0 0 0;">💡 <em>{escape(relevance_reason)}</em></p>
        </td></tr></table>
    </td></tr>'''

    # Abstract
    if abstract:
        html += f'''
    <tr><td style="padding:0 30px 20px 30px;">
        {info_box(section_title("📄 Abstract") + f'<p style="color:#cbd5e1;font-size:14px;line-height:1.7;margin:0;">{escape(abstract)}</p>')}
    </td></tr>'''

    # CTA
    html += f'''
    <tr><td style="padding:10px 30px 20px 30px;text-align:center;">
        {action_button("Open on arXiv →", link)}
    </td></tr>'''

    html += mgmt
    html += email_footer(extra_links=footer_links)
    html += email_wrapper_end()

    return subject, plain, html


def render_onboarding_instruction_email(
    colleague_name: str,
    reason: str = "missing_code",
) -> tuple[str, str, str]:
    """
    Render an instruction email for a colleague who emailed without a valid code,
    or whose message could not be parsed.

    reason: 'missing_code' | 'invalid_code' | 'parse_error' | 'not_configured'

    Returns (subject, plain_text, html).
    """
    first_name = colleague_name if colleague_name else "there"

    if reason == "parse_error":
        headline = "We couldn't understand your message"
        intro = (
            "Thanks for reaching out! We received your invite code, but we couldn't "
            "parse your details. Please reply using the template below."
        )
    elif reason == "invalid_code":
        headline = "Invite code not recognised"
        intro = (
            "The invite code you provided wasn't recognised. Please double-check it "
            "and try again using the template below. If you don't have a code, "
            "please ask the ResearchPulse owner for one."
        )
    elif reason == "not_configured":
        headline = "Colleague sharing not ready yet"
        intro = (
            "Thank you for your interest! The ResearchPulse owner hasn't finished "
            "setting up colleague sharing yet. Please ask the owner for a valid invite code "
            "and try again later."
        )
    else:
        headline = "Invite code required"
        intro = (
            "Thank you for your interest in receiving research updates from ResearchPulse! "
            "To subscribe, you'll need an invite code from the ResearchPulse owner. "
            "Once you have it, reply using the template below."
        )

    template_block = (
        "Code: YOUR_INVITE_CODE\n"
        "Name: Your Full Name\n"
        "Research interests: topic1, topic2, topic3"
    )
    example_block = (
        "Code: ABC123\n"
        "Name: Jane Smith\n"
        "Research interests: machine learning, NLP, transformers"
    )

    subject = f"ResearchPulse: {headline}"

    # Plain text
    plain = (
        f"Hi {first_name},\n\n"
        f"{intro}\n\n"
        "--- Copy & paste template ---\n"
        f"{template_block}\n"
        "---\n\n"
        "Example:\n"
        f"{example_block}\n\n"
        "Accepted formats for the code field:\n"
        '  Code: XXXX\n'
        '  Invite code: XXXX\n'
        '  CODE=XXXX\n\n'
        "Interests can be comma-separated or on separate lines.\n\n"
        "Best regards,\nResearchPulse\n"
    )

    # HTML
    html = email_wrapper_start()
    html += email_header(subtitle=headline)
    html += f'''
    <tr><td style="padding:25px 30px;">
        <p style="color:#e2e8f0;font-size:15px;line-height:1.6;margin:0 0 16px 0;">Hi <strong style="color:white;">{escape(first_name)}</strong>,</p>
        <p style="color:#94a3b8;font-size:14px;line-height:1.6;margin:0 0 20px 0;">{escape(intro)}</p>
        {section_title("📋 Reply Template")}
        <pre style="background:#0f172a;border:1px solid #334155;border-radius:10px;padding:16px;color:#a5b4fc;font-size:13px;line-height:1.8;white-space:pre-wrap;margin:0 0 20px 0;">{escape(template_block)}</pre>
        {section_title("✏️ Example")}
        <pre style="background:#0f172a;border:1px solid #334155;border-radius:10px;padding:16px;color:#4ade80;font-size:13px;line-height:1.8;white-space:pre-wrap;margin:0 0 20px 0;">{escape(example_block)}</pre>
    </td></tr>'''
    html += email_footer()
    html += email_wrapper_end()

    return subject, plain, html


def render_colleague_confirmation_email(
    colleague_name: str,
    interests: List[str],
    remove_url: str = "",
    update_url: str = "",
) -> tuple[str, str, str]:
    """
    Render a confirmation email after a colleague successfully registers.

    Returns (subject, plain_text, html).
    """
    first_name = colleague_name.split()[0] if colleague_name else "there"
    interests_str = ", ".join(interests) if interests else "(not provided yet)"

    subject = "ResearchPulse: Welcome! You're now subscribed 🎉"

    # Plain text
    plain = (
        f"Hi {first_name},\n\n"
        "Great news – you've been successfully added to ResearchPulse!\n\n"
        f"Stored interests: {interests_str}\n\n"
        "What happens next?\n"
        "  • You'll receive paper recommendations matching your interests.\n"
        "  • Emails come as papers are discovered (frequency depends on owner settings).\n\n"
        "To remove yourself from the list, reply with:\n"
        "  Code: YOUR_INVITE_CODE\n"
        "  Instruction: Remove\n\n"
        "Best regards,\nResearchPulse\n"
    )

    # HTML
    html = email_wrapper_start()
    html += email_header(subtitle="Welcome aboard! 🎉")
    html += f'''
    <tr><td style="padding:25px 30px;">
        <p style="color:#e2e8f0;font-size:15px;line-height:1.6;margin:0 0 16px 0;">Hi <strong style="color:white;">{escape(first_name)}</strong>,</p>
        <p style="color:#94a3b8;font-size:14px;line-height:1.6;margin:0 0 16px 0;">
            Great news – you've been successfully added to <strong style="color:#f472b6;">ResearchPulse</strong>!
            You'll start receiving personalised paper recommendations matching your interests.
        </p>
        {section_title("📚 Your stored interests")}
        <p style="color:#e2e8f0;font-size:14px;margin:0 0 20px 0;">{escape(interests_str)}</p>

        {section_title("🔧 How to unsubscribe")}
        <p style="color:#94a3b8;font-size:14px;line-height:1.6;margin:0 0 20px 0;">To remove yourself from the list, reply with:</p>
        <div style="background:#0f172a;border-radius:8px;padding:12px 16px;font-family:monospace;font-size:13px;color:#e2e8f0;margin:0 0 20px 0;">
            Code: YOUR_INVITE_CODE<br>
            Instruction: Remove
        </div>
    </td></tr>'''
    html += email_footer(extra_links="")
    html += email_wrapper_end()

    return subject, plain, html


def render_colleague_update_confirmation_email(
    colleague_name: str,
    new_interests: List[str],
    remove_url: str = "",
    update_url: str = "",
) -> tuple[str, str, str]:
    """Email confirming that a colleague's interests were updated."""
    first_name = colleague_name.split()[0] if colleague_name else "there"
    interests_str = ", ".join(new_interests) if new_interests else "(none)"
    subject = "ResearchPulse: Your interests have been updated"

    plain = (
        f"Hi {first_name},\n\n"
        "Your research interests have been updated.\n\n"
        f"New interests: {interests_str}\n\n"
        "Future recommendations will be based on these topics.\n\n"
        "-- ResearchPulse\n"
    )

    html = email_wrapper_start()
    html += email_header(subtitle="Interests Updated ✏️")
    html += f'''
    <tr><td style="padding:25px 30px;">
        <p style="color:#e2e8f0;font-size:15px;line-height:1.6;margin:0 0 12px 0;">Hi <strong style="color:white;">{escape(first_name)}</strong>,</p>
        <p style="color:#94a3b8;font-size:14px;margin:0 0 20px 0;">Your research interests have been updated. Future recommendations will be based on:</p>
        <p style="color:#e2e8f0;font-size:14px;background:#0f172a;border-radius:10px;padding:16px;border-left:4px solid #3b82f6;margin:0 0 20px 0;">{escape(interests_str)}</p>
    </td></tr>'''
    footer_links = ""
    if remove_url or update_url:
        footer_links = colleague_management_footer_links(remove_url or "#", update_url or "#")
    html += email_footer(extra_links=footer_links)
    html += email_wrapper_end()
    return subject, plain, html


def render_colleague_removed_email(
    colleague_name: str,
) -> tuple[str, str, str]:
    """Email confirming that a colleague has been removed."""
    first_name = colleague_name.split()[0] if colleague_name else "there"
    subject = "ResearchPulse: You've been unsubscribed"
    plain = (
        f"Hi {first_name},\n\n"
        "You've been removed from ResearchPulse. You won't receive any more paper recommendations.\n\n"
        "If this was a mistake, contact the ResearchPulse owner to re-subscribe.\n\n"
        "-- ResearchPulse\n"
    )
    html = email_wrapper_start()
    html += email_header(subtitle="Unsubscribed")
    html += f'''
    <tr><td style="padding:25px 30px;">
        <p style="color:#e2e8f0;font-size:15px;line-height:1.6;margin:0 0 12px 0;">Hi <strong style="color:white;">{escape(first_name)}</strong>,</p>
        <p style="color:#94a3b8;font-size:14px;margin:0 0 16px 0;">You've been successfully removed from ResearchPulse. You won't receive any more paper recommendations.</p>
        <p style="color:#64748b;font-size:13px;margin:0;">If this was a mistake, please contact the ResearchPulse owner to re-subscribe.</p>
    </td></tr>'''
    html += email_footer()
    html += email_wrapper_end()
    return subject, plain, html


def render_token_error_email(
    colleague_name: str,
    reason: str = "expired",
) -> tuple[str, str, str]:
    """Email shown when a colleague token is invalid or expired."""
    first_name = colleague_name.split()[0] if colleague_name else "there"
    if reason == "expired":
        detail = "The link you used has expired."
    else:
        detail = "The link you used is invalid."

    subject = "ResearchPulse: Link expired or invalid"
    plain = (
        f"Hi {first_name},\n\n"
        f"{detail}\n\n"
        "Next steps:\n"
        "  • Reply to any ResearchPulse email to manage your subscription.\n"
        "  • Or contact the ResearchPulse owner directly.\n\n"
        "-- ResearchPulse\n"
    )
    html = email_wrapper_start()
    html += email_header(subtitle="Link Issue")
    html += f'''
    <tr><td style="padding:25px 30px;">
        <p style="color:#e2e8f0;font-size:15px;margin:0 0 12px 0;">Hi <strong style="color:white;">{escape(first_name)}</strong>,</p>
        <p style="color:#94a3b8;font-size:14px;margin:0 0 16px 0;">{escape(detail)}</p>
        <p style="color:#64748b;font-size:13px;margin:0;">Reply to any ResearchPulse email or contact the owner directly to manage your subscription.</p>
    </td></tr>'''
    html += email_footer()
    html += email_wrapper_end()
    return subject, plain, html


# =============================================================================
# Onboarding flow templates (HTML)
# =============================================================================


def render_clarify_intent_email(
    colleague_name: str,
    custom_message: str = "",
) -> tuple[str, str, str]:
    """
    Email sent when someone provides a valid code but unclear signup intent.
    Returns (subject, plain_text, html).
    """
    first_name = colleague_name.split()[0] if colleague_name else "there"
    intro = f"{custom_message}\n\n" if custom_message else ""

    subject = "ResearchPulse: Would you like research paper updates?"

    plain = (
        f"Hi {first_name},\n\n"
        f"{intro}"
        "I received your message with a valid join code! 🎉\n\n"
        "ResearchPulse sends personalised research paper recommendations based on "
        "your interests. To get started, I'd love to know:\n\n"
        "What research topics interest you?\n\n"
        "For example:\n"
        '  - "I\'m interested in machine learning and NLP"\n'
        '  - "Computer vision and autonomous driving"\n'
        '  - "Reinforcement learning and robotics"\n\n'
        "Just reply with your research interests and I'll start sending you "
        "relevant papers!\n\n"
        "Best regards,\nResearchPulse\n"
    )

    html = email_wrapper_start()
    html += email_header(subtitle="Welcome! 🎉")
    intro_html = f'<p style="color:#94a3b8;font-size:14px;line-height:1.6;margin:0 0 16px 0;">{escape(custom_message)}</p>' if custom_message else ""
    html += f'''
    <tr><td style="padding:25px 30px;">
        <p style="color:#e2e8f0;font-size:15px;line-height:1.6;margin:0 0 16px 0;">Hi <strong style="color:white;">{escape(first_name)}</strong>,</p>
        {intro_html}
        <p style="color:#94a3b8;font-size:14px;line-height:1.6;margin:0 0 16px 0;">
            I received your message with a valid join code! 🎉
        </p>
        <p style="color:#e2e8f0;font-size:14px;line-height:1.6;margin:0 0 16px 0;">
            <strong style="color:#f472b6;">ResearchPulse</strong> sends personalised research paper
            recommendations based on your interests. To get started, I'd love to know:
        </p>
        {section_title("📚 What research topics interest you?")}
        <ul style="color:#94a3b8;font-size:14px;padding-left:20px;margin:0 0 20px 0;">
            <li style="margin-bottom:6px;">"I'm interested in machine learning and NLP"</li>
            <li style="margin-bottom:6px;">"Computer vision and autonomous driving"</li>
            <li>"Reinforcement learning and robotics"</li>
        </ul>
        <p style="color:#e2e8f0;font-size:14px;margin:0;">
            Just reply with your research interests and I'll start sending you relevant papers!
        </p>
    </td></tr>'''
    html += email_footer()
    html += email_wrapper_end()
    return subject, plain, html


def render_onboarding_questions_email(
    colleague_name: str,
) -> tuple[str, str, str]:
    """
    Email asking a new colleague for both name and research interests.
    Returns (subject, plain_text, html).
    """
    first_name = colleague_name.split()[0] if colleague_name else "there"
    subject = "ResearchPulse: Help us personalise your updates!"

    plain = (
        f"Hi {first_name},\n\n"
        "Welcome to ResearchPulse! 🎉 You've been added to the system.\n\n"
        "To personalise your research paper recommendations, please reply with:\n\n"
        "1. Your Name: What should we call you?\n\n"
        "2. Research Topics: What topics interest you?\n"
        "   (e.g., NLP, computer vision, reinforcement learning, transformers, LLMs)\n\n"
        "3. Venues (optional): Any preferred venues?\n"
        "   (e.g., arXiv, NeurIPS, ACL, ICML, CVPR)\n\n"
        "Just reply to this email with your answers – no special format needed!\n\n"
        "Example reply:\n"
        '"My name is Alex. I\'m interested in large language models and NLP, '
        'especially transformers and prompt engineering."\n\n'
        "Best regards,\nResearchPulse\n"
    )

    html = email_wrapper_start()
    html += email_header(subtitle="Welcome aboard! 🎉")
    html += f'''
    <tr><td style="padding:25px 30px;">
        <p style="color:#e2e8f0;font-size:15px;line-height:1.6;margin:0 0 16px 0;">Hi <strong style="color:white;">{escape(first_name)}</strong>,</p>
        <p style="color:#94a3b8;font-size:14px;line-height:1.6;margin:0 0 20px 0;">
            Welcome to <strong style="color:#f472b6;">ResearchPulse</strong>! 🎉
            You've been added to the system. To personalise your recommendations, please reply with:
        </p>
        {section_title("1️⃣ Your name")}
        <p style="color:#e2e8f0;font-size:14px;margin:0 0 16px 0;">What should we call you?</p>
        {section_title("2️⃣ Research topics")}
        <p style="color:#e2e8f0;font-size:14px;margin:0 0 4px 0;">What topics interest you?</p>
        <p style="color:#94a3b8;font-size:13px;margin:0 0 16px 0;">e.g., NLP, computer vision, reinforcement learning, transformers, LLMs</p>
        {section_title("3️⃣ Venues (optional)")}
        <p style="color:#e2e8f0;font-size:14px;margin:0 0 20px 0;">Any preferred venues? (arXiv, NeurIPS, ACL, ICML, CVPR)</p>
        <pre style="background:#0f172a;border:1px solid #334155;border-radius:10px;padding:16px;color:#4ade80;font-size:13px;line-height:1.8;white-space:pre-wrap;margin:0 0 20px 0;">My name is Alex. I'm interested in large language models and NLP, especially transformers and prompt engineering.</pre>
        <p style="color:#94a3b8;font-size:13px;margin:0;">Just reply to this email – no special format needed!</p>
    </td></tr>'''
    html += email_footer()
    html += email_wrapper_end()
    return subject, plain, html


def render_onboarding_interests_email(
    colleague_name: str,
) -> tuple[str, str, str]:
    """
    Email asking a colleague to provide their research interests.
    Returns (subject, plain_text, html).
    """
    first_name = colleague_name.split()[0] if colleague_name else "there"
    subject = "ResearchPulse: What research topics interest you?"

    plain = (
        f"Hi {first_name},\n\n"
        "Welcome to ResearchPulse! 🎉 You've been successfully added.\n\n"
        "To start sending you relevant paper recommendations, we need to know "
        "your research interests.\n\n"
        "Please reply with the topics you're interested in:\n"
        '  - e.g., "machine learning, NLP, transformers"\n'
        '  - e.g., "computer vision, object detection, autonomous driving"\n'
        '  - e.g., "reinforcement learning, robotics, multi-agent systems"\n\n'
        "Just reply to this email with your interests – no special format needed!\n\n"
        "Best regards,\nResearchPulse\n"
    )

    html = email_wrapper_start()
    html += email_header(subtitle="Almost there! 📚")
    html += f'''
    <tr><td style="padding:25px 30px;">
        <p style="color:#e2e8f0;font-size:15px;line-height:1.6;margin:0 0 16px 0;">Hi <strong style="color:white;">{escape(first_name)}</strong>,</p>
        <p style="color:#94a3b8;font-size:14px;line-height:1.6;margin:0 0 20px 0;">
            Welcome to <strong style="color:#f472b6;">ResearchPulse</strong>! 🎉
            You've been successfully added. To start sending you relevant paper recommendations,
            we need to know your research interests.
        </p>
        {section_title("📋 Reply with your interests")}
        <ul style="color:#94a3b8;font-size:14px;padding-left:20px;margin:0 0 20px 0;">
            <li style="margin-bottom:6px;">"machine learning, NLP, transformers"</li>
            <li style="margin-bottom:6px;">"computer vision, object detection, autonomous driving"</li>
            <li>"reinforcement learning, robotics, multi-agent systems"</li>
        </ul>
        <p style="color:#94a3b8;font-size:13px;margin:0;">Just reply to this email – no special format needed!</p>
    </td></tr>'''
    html += email_footer()
    html += email_wrapper_end()
    return subject, plain, html


def render_onboarding_name_email(
    colleague_name: str,
) -> tuple[str, str, str]:
    """
    Email asking a colleague to provide their name.
    Returns (subject, plain_text, html).
    """
    first_name = colleague_name.split()[0] if colleague_name else "there"
    subject = "ResearchPulse: One quick question!"

    plain = (
        f"Hi {first_name},\n\n"
        "Welcome to ResearchPulse! 🎉 We've noted your research interests.\n\n"
        "One quick question: What's your name? This helps us personalise your updates.\n\n"
        "Just reply with your name and you'll be all set!\n\n"
        "Best regards,\nResearchPulse\n"
    )

    html = email_wrapper_start()
    html += email_header(subtitle="One quick question! ✏️")
    html += f'''
    <tr><td style="padding:25px 30px;">
        <p style="color:#e2e8f0;font-size:15px;line-height:1.6;margin:0 0 16px 0;">Hi <strong style="color:white;">{escape(first_name)}</strong>,</p>
        <p style="color:#94a3b8;font-size:14px;line-height:1.6;margin:0 0 20px 0;">
            Welcome to <strong style="color:#f472b6;">ResearchPulse</strong>! 🎉
            We've noted your research interests.
        </p>
        <p style="color:#e2e8f0;font-size:15px;line-height:1.6;margin:0 0 16px 0;">
            One quick question: <strong style="color:white;">What's your name?</strong>
        </p>
        <p style="color:#94a3b8;font-size:14px;margin:0;">This helps us personalise your updates. Just reply with your name and you'll be all set!</p>
    </td></tr>'''
    html += email_footer()
    html += email_wrapper_end()
    return subject, plain, html
