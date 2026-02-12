"""
Tool: summarize_paper - Generate AI summaries of research papers.

Downloads the PDF from arXiv, extracts text, and uses OpenAI to generate
a structured summary. Summaries are persisted in the paper_views table.

Features:
- PDF download from arXiv via the paper's pdf_url
- Text extraction using pypdf
- LLM-powered summarization via OpenAI GPT
- Cached: summaries are stored in DB and reused
- Auto-summarize: called automatically for high-priority papers
"""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from uuid import UUID

import httpx
from pypdf import PdfReader

# Load .env file if present (must be done before any os.getenv calls)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_PDF_BYTES = 20 * 1024 * 1024  # 20 MB limit
MAX_TEXT_CHARS = 80_000  # Truncate text sent to LLM
SUMMARY_MODEL = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")  # Use project LLM config


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def _download_pdf(pdf_url: str) -> bytes:
    """Download a PDF from a URL. Returns raw bytes."""
    logger.info(f"[SUMMARIZE] Downloading PDF from {pdf_url}")
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        resp = client.get(pdf_url)
        resp.raise_for_status()
        if len(resp.content) > MAX_PDF_BYTES:
            raise ValueError(f"PDF too large ({len(resp.content)} bytes > {MAX_PDF_BYTES})")
        return resp.content


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    full_text = "\n\n".join(pages_text)
    logger.info(f"[SUMMARIZE] Extracted {len(full_text)} chars from {len(reader.pages)} pages")
    return full_text[:MAX_TEXT_CHARS]


# ---------------------------------------------------------------------------
# LLM summarisation
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = """You are a research paper summarisation assistant. 
Given the full text of an academic paper, produce a clear, well-structured summary.

Your summary MUST include the following sections (use markdown headers):

## Key Findings
A bullet list of the 3-5 most important results or contributions.

## Methodology
A concise description of the approach, models, or experimental setup used.

## Main Contributions
What is novel about this work compared to prior art.

## Limitations & Future Work
Any stated or apparent limitations and suggested future directions.

## TL;DR
A single paragraph (3-4 sentences) summarising the paper for a busy researcher.

Keep the entire summary under 800 words. Use clear, academic language.
Do NOT include the paper title or authors in the summary - those are shown separately.
"""


def _generate_summary_with_llm(paper_text: str, title: str = "") -> str:
    """Call OpenAI-compatible LLM to generate a structured summary of the paper text."""
    import openai

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
    model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

    if not api_key:
        raise RuntimeError("LLM_API_KEY is not set - cannot generate summary")

    client = openai.OpenAI(api_key=api_key, base_url=api_base)

    user_message = f"Summarise the following research paper:\n\nTitle: {title}\n\n{paper_text}"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=8000,  # Reasoning models need extra tokens for thinking + output
    )

    # Extract summary - handle reasoning models that may put output in different fields
    choice = response.choices[0] if response.choices else None
    if not choice:
        logger.error(f"[SUMMARIZE] LLM returned no choices. Full response: {response}")
        raise RuntimeError("LLM returned empty response (no choices)")

    summary = choice.message.content or ""

    # Some reasoning models return empty content - check for output_text or reasoning_content
    if not summary:
        # Try output_text (OpenAI reasoning models)
        summary = getattr(choice.message, 'output_text', '') or ''
    if not summary:
        # Try reasoning_content
        summary = getattr(choice.message, 'reasoning_content', '') or ''
    if not summary:
        # Try refusal field as debug info
        refusal = getattr(choice.message, 'refusal', None)
        if refusal:
            logger.error(f"[SUMMARIZE] LLM refused: {refusal}")
            raise RuntimeError(f"LLM refused to summarize: {refusal}")
        # Log the raw response for debugging
        logger.error(f"[SUMMARIZE] LLM returned empty content. Message: {choice.message}")
        raise RuntimeError("LLM returned empty summary content")

    logger.info(f"[SUMMARIZE] LLM summary generated ({len(summary)} chars)")
    return summary.strip()


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def summarize_paper(
    paper_view_id: UUID,
    pdf_url: str,
    title: str = "",
    force: bool = False,
) -> Dict[str, Any]:
    """
    Generate (or return cached) AI summary of a paper.

    Steps:
    1. Check if a summary already exists in the DB (unless force=True).
    2. Download the PDF from arXiv.
    3. Extract text using pypdf.
    4. Send to OpenAI for summarisation.
    5. Save the summary to the paper_views table.

    Args:
        paper_view_id: UUID of the paper_view record.
        pdf_url: URL to the paper PDF (typically arXiv).
        title: Paper title for context.
        force: If True, regenerate even if a summary already exists.

    Returns:
        Dict with 'success', 'summary', 'cached', 'error' fields.
    """
    from src.db.database import get_db_session
    from src.db.orm_models import PaperView

    # 1. Check cache
    with get_db_session() as db:
        view = db.query(PaperView).filter(PaperView.id == paper_view_id).first()
        if not view:
            return {"success": False, "summary": None, "cached": False, "error": "Paper view not found"}

        if view.summary and not force:
            logger.info(f"[SUMMARIZE] Returning cached summary for paper_view {paper_view_id}")
            return {
                "success": True,
                "summary": view.summary,
                "cached": True,
                "generated_at": view.summary_generated_at.isoformat() + "Z" if view.summary_generated_at else None,
                "error": None,
            }

    # 2-4. Download → extract → summarise
    try:
        pdf_bytes = _download_pdf(pdf_url)
        paper_text = _extract_text_from_pdf(pdf_bytes)
        if len(paper_text.strip()) < 100:
            return {"success": False, "summary": None, "cached": False, "error": "Could not extract enough text from PDF"}
        summary = _generate_summary_with_llm(paper_text, title=title)
    except httpx.HTTPStatusError as e:
        logger.error(f"[SUMMARIZE] PDF download failed: {e}")
        return {"success": False, "summary": None, "cached": False, "error": f"Failed to download PDF: {e.response.status_code}"}
    except Exception as e:
        logger.error(f"[SUMMARIZE] Summarisation failed: {e}", exc_info=True)
        return {"success": False, "summary": None, "cached": False, "error": str(e)}

    # 5. Persist summary
    try:
        with get_db_session() as db:
            view = db.query(PaperView).filter(PaperView.id == paper_view_id).first()
            if view:
                view.summary = summary
                view.summary_generated_at = datetime.utcnow()
                db.commit()
                logger.info(f"[SUMMARIZE] Summary saved for paper_view {paper_view_id}")
    except Exception as e:
        logger.error(f"[SUMMARIZE] Failed to persist summary: {e}", exc_info=True)
        # Still return the summary even if DB save failed
        return {"success": True, "summary": summary, "cached": False, "generated_at": None, "error": f"Summary generated but save failed: {e}"}

    return {
        "success": True,
        "summary": summary,
        "cached": False,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "error": None,
    }


def auto_summarize_high_priority(
    user_id: UUID,
) -> Dict[str, Any]:
    """
    Auto-summarise all high-priority papers that don't have a summary yet.

    Called after an agent run completes. Only processes papers with
    importance='high' and no existing summary.

    Returns:
        Dict with 'summarized_count', 'failed_count', 'skipped_count'.
    """
    from src.db.database import get_db_session
    from src.db.orm_models import PaperView, Paper

    results = {"summarized_count": 0, "failed_count": 0, "skipped_count": 0}

    try:
        with get_db_session() as db:
            views = (
                db.query(PaperView)
                .join(Paper)
                .filter(
                    PaperView.user_id == user_id,
                    PaperView.importance == "high",
                    PaperView.summary.is_(None),
                    PaperView.is_deleted != True,
                )
                .all()
            )

            logger.info(f"[AUTO_SUMMARIZE] Found {len(views)} high-priority papers without summaries")

            for view in views:
                paper = view.paper
                if not paper:
                    results["skipped_count"] += 1
                    continue

                external_id = paper.external_id or ""
                pdf_url = paper.pdf_url or (f"https://arxiv.org/pdf/{external_id}.pdf" if external_id else "")
                if not pdf_url:
                    results["skipped_count"] += 1
                    continue

                result = summarize_paper(
                    paper_view_id=view.id,
                    pdf_url=pdf_url,
                    title=paper.title or "",
                    force=False,
                )

                if result.get("success"):
                    results["summarized_count"] += 1
                else:
                    results["failed_count"] += 1
                    logger.warning(f"[AUTO_SUMMARIZE] Failed for paper {external_id}: {result.get('error')}")

    except Exception as e:
        logger.error(f"[AUTO_SUMMARIZE] Error: {e}", exc_info=True)

    logger.info(
        f"[AUTO_SUMMARIZE] Done: {results['summarized_count']} summarized, "
        f"{results['failed_count']} failed, {results['skipped_count']} skipped"
    )
    return results
