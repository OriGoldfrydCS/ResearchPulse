"""
Colleague self-service API routes for ResearchPulse.

Provides endpoints for colleague actions triggered via signed token links
embedded in emails:
- GET /colleague/remove?token=...  – remove (deactivate) a colleague
- GET /colleague/update?token=...  – show update form / confirm interests update
- POST /colleague/update           – process interest update submission
"""

from __future__ import annotations

import logging
import os
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Form
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["colleague-self-service"])


def _get_store():
    """Lazy import to avoid circular imports."""
    from ..db.store import get_default_store
    return get_default_store()


def _simple_html_page(title: str, body_html: str) -> str:
    """Wrap content in a minimal branded HTML page."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} – ResearchPulse</title>
<style>
body{{margin:0;padding:40px 20px;background:#0f172a;color:#e2e8f0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;}}
.card{{max-width:500px;margin:0 auto;background:#1e293b;border-radius:16px;padding:40px;box-shadow:0 25px 50px -12px rgba(0,0,0,.5);}}
h1{{color:#f472b6;font-size:24px;margin:0 0 16px;}}
p{{color:#94a3b8;line-height:1.6;}}
a{{color:#60a5fa;}}
.btn{{display:inline-block;background:linear-gradient(135deg,#ec4899,#8b5cf6);color:#fff;padding:12px 28px;border-radius:25px;text-decoration:none;font-weight:bold;border:none;cursor:pointer;font-size:14px;}}
textarea{{width:100%;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:8px;padding:12px;font-size:14px;min-height:100px;box-sizing:border-box;}}
label{{display:block;color:#94a3b8;margin-bottom:8px;font-size:13px;}}
</style></head>
<body><div class="card">{body_html}</div></body></html>"""


# =============================================================================
# Remove endpoint
# =============================================================================

@router.get("/colleague/remove")
async def colleague_remove(token: str = Query(..., description="Signed action token")):
    """One-click colleague removal via signed token link."""
    from ..tools.colleague_tokens import verify_token, TokenAction
    from ..tools.email_templates import render_colleague_removed_email
    from ..tools.outbound_email import send_outbound_email, EmailType

    payload = verify_token(token)
    if payload is None:
        return HTMLResponse(
            _simple_html_page("Link expired", "<h1>Link expired or invalid</h1>"
                              "<p>This link has expired or is no longer valid.</p>"
                              "<p>To manage your subscription, reply to any ResearchPulse email "
                              "or contact the owner directly.</p>"),
            status_code=200,
        )

    if payload.action != TokenAction.REMOVE:
        return HTMLResponse(_simple_html_page("Invalid link", "<h1>Invalid link</h1><p>This link cannot be used for removal.</p>"), status_code=400)

    store = _get_store()
    owner_id = UUID(payload.owner_id)
    colleagues = store.list_colleagues(owner_id)
    target = None
    for c in colleagues:
        if c.get("email", "").lower() == payload.colleague_email.lower():
            target = c
            break

    if target is None:
        return HTMLResponse(
            _simple_html_page("Not found", "<h1>Already removed</h1><p>You are not currently subscribed.</p>"),
            status_code=200,
        )

    # Deactivate (soft-delete)
    store.update_colleague(UUID(str(target["id"])), {"enabled": False, "auto_send_emails": False})
    logger.info("[COLLEAGUE_REMOVE] Removed colleague %s for owner %s", payload.colleague_email, payload.owner_id)

    # Send confirmation email
    try:
        subj, plain, html = render_colleague_removed_email(target.get("name", ""))
        send_outbound_email(to_email=payload.colleague_email, subject=subj, body=plain,
                            email_type=EmailType.COLLEAGUE_JOIN, html_body=html)
    except Exception as e:
        logger.warning("Failed to send removal confirmation: %s", e)

    return HTMLResponse(
        _simple_html_page("Removed", "<h1>You've been unsubscribed</h1>"
                          "<p>You will no longer receive paper recommendations from ResearchPulse.</p>"
                          "<p>If this was a mistake, contact the owner to re-subscribe.</p>"),
        status_code=200,
    )


# =============================================================================
# Update interests – GET (show form)
# =============================================================================

@router.get("/colleague/update")
async def colleague_update_form(token: str = Query(..., description="Signed action token")):
    """Show a form where colleague can update their interests."""
    from ..tools.colleague_tokens import verify_token, TokenAction

    payload = verify_token(token)
    if payload is None:
        return HTMLResponse(
            _simple_html_page("Link expired", "<h1>Link expired or invalid</h1>"
                              "<p>Reply to any ResearchPulse email or contact the owner.</p>"),
            status_code=200,
        )

    if payload.action != TokenAction.UPDATE:
        return HTMLResponse(_simple_html_page("Invalid link", "<h1>Invalid link</h1>"), status_code=400)

    store = _get_store()
    owner_id = UUID(payload.owner_id)
    colleagues = store.list_colleagues(owner_id)
    current_interests = ""
    for c in colleagues:
        if c.get("email", "").lower() == payload.colleague_email.lower():
            current_interests = c.get("interests") or c.get("research_interests") or ""
            break

    form_html = (
        f"<h1>Update your interests</h1>"
        f"<p>Enter your research interests below (comma-separated).</p>"
        f'<form method="POST" action="/colleague/update">'
        f'<input type="hidden" name="token" value="{token}">'
        f'<label for="interests">Research interests</label>'
        f'<textarea name="interests" id="interests" placeholder="machine learning, NLP, transformers">{current_interests}</textarea>'
        f'<br><br><button type="submit" class="btn">Save</button>'
        f'</form>'
    )
    return HTMLResponse(_simple_html_page("Update Interests", form_html), status_code=200)


# =============================================================================
# Update interests – POST (process)
# =============================================================================

@router.post("/colleague/update")
async def colleague_update_submit(
    token: str = Form(...),
    interests: str = Form(""),
):
    """Process interest update from the form."""
    from ..tools.colleague_tokens import verify_token, TokenAction
    from ..tools.email_templates import render_colleague_update_confirmation_email
    from ..tools.outbound_email import send_outbound_email, EmailType

    payload = verify_token(token)
    if payload is None:
        return HTMLResponse(
            _simple_html_page("Link expired", "<h1>Link expired</h1><p>Please request a new link.</p>"),
            status_code=200,
        )

    if payload.action != TokenAction.UPDATE:
        return HTMLResponse(_simple_html_page("Invalid", "<h1>Invalid token</h1>"), status_code=400)

    interests_clean = interests.strip()
    if not interests_clean:
        return HTMLResponse(
            _simple_html_page("Error", "<h1>No interests provided</h1><p>Please go back and enter your interests.</p>"),
            status_code=400,
        )

    store = _get_store()
    owner_id = UUID(payload.owner_id)
    colleagues = store.list_colleagues(owner_id)
    target = None
    for c in colleagues:
        if c.get("email", "").lower() == payload.colleague_email.lower():
            target = c
            break

    if target is None:
        return HTMLResponse(
            _simple_html_page("Not found", "<h1>Subscription not found</h1>"),
            status_code=200,
        )

    # Derive categories
    derived = {}
    try:
        from ..tools.arxiv_categories import derive_arxiv_categories_from_interests
        derived = derive_arxiv_categories_from_interests(interests_clean)
    except Exception:
        pass

    update_data = {
        "interests": interests_clean,
        "research_interests": interests_clean,
        "derived_arxiv_categories": derived,
        "categories": derived.get("primary", []) + derived.get("secondary", []),
    }
    store.update_colleague(UUID(str(target["id"])), update_data)
    logger.info("[COLLEAGUE_UPDATE] Updated interests for %s", payload.colleague_email)

    # Send confirmation
    try:
        interests_list = [i.strip() for i in interests_clean.split(",") if i.strip()]
        from ..tools.colleague_tokens import generate_remove_url, generate_update_url
        base_url = os.getenv("RESEARCHPULSE_BASE_URL", "https://researchpulse.app")
        remove_url = generate_remove_url(base_url, payload.owner_id, payload.colleague_email)
        update_url = generate_update_url(base_url, payload.owner_id, payload.colleague_email)
        subj, plain, html = render_colleague_update_confirmation_email(
            target.get("name", ""), interests_list, remove_url, update_url,
        )
        send_outbound_email(to_email=payload.colleague_email, subject=subj, body=plain,
                            email_type=EmailType.COLLEAGUE_JOIN, html_body=html)
    except Exception as e:
        logger.warning("Failed to send update confirmation: %s", e)

    return HTMLResponse(
        _simple_html_page("Updated", "<h1>Interests updated!</h1>"
                          f"<p>Your new interests: <strong>{interests_clean}</strong></p>"
                          "<p>Future recommendations will be based on these topics.</p>"),
        status_code=200,
    )
