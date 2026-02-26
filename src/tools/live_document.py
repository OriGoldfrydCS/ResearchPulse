"""
Module: live_document - Living research briefing document.

This module generates and maintains a living markdown document that
summarizes the user's research landscape based on paper runs.

**Document structure:**
1. Header with title and generation time
2. Executive Summary (AI-generated)
3. Top Papers (sorted by relevance/novelty)
4. Trending Topics (emerging themes)
5. Category Breakdown (papers per category)
6. Recent History (rolling window)
7. Change Log (what changed since last update)

**Update modes:**
- Full refresh: Regenerate entire document
- Incremental: Append new papers, update rolling sections

Feature flag: LIVE_DOCUMENT_ENABLED (defaults to False)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Add parent to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class DocumentPaper(BaseModel):
    """Paper formatted for document display."""
    arxiv_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    abstract_snippet: str = ""
    relevance_score: float = 0.0
    novelty_score: float = 0.0
    llm_novelty_score: Optional[float] = None
    categories: List[str] = Field(default_factory=list)
    added_at: str = ""
    arxiv_url: str = ""


class TrendingTopic(BaseModel):
    """Detected trending topic."""
    topic: str
    paper_count: int
    avg_relevance: float
    emerging: bool = False


class DocumentSection(BaseModel):
    """A section of the live document."""
    name: str
    content: str
    updated_at: str


class LiveDocumentData(BaseModel):
    """Full live document data structure."""
    user_id: str
    title: str = "ResearchPulse - Live Document"
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    last_updated: str = ""
    
    # Content sections
    executive_summary: str = ""
    top_papers: List[DocumentPaper] = Field(default_factory=list)
    trending_topics: List[TrendingTopic] = Field(default_factory=list)
    category_breakdown: Dict[str, int] = Field(default_factory=dict)
    recent_papers: List[DocumentPaper] = Field(default_factory=list)
    
    # Metadata
    total_papers_tracked: int = 0
    runs_included: List[str] = Field(default_factory=list)
    change_log: List[str] = Field(default_factory=list)


# =============================================================================
# Prompt Templates
# =============================================================================

EXECUTIVE_SUMMARY_PROMPT = """You are a research analyst. Generate a concise executive summary (3-5 sentences) of the user's current research landscape based on their recent papers.

**USER PROFILE:**
Research Topics: {research_topics}

**TOP PAPERS THIS PERIOD:**
{papers_summary}

**TRENDING TOPICS:**
{trending_topics}

Write a brief, professional summary highlighting:
1. Overall research focus areas
2. Notable new papers or directions
3. Any emerging trends

Keep it concise and actionable."""


# =============================================================================
# Live Document Manager
# =============================================================================

class LiveDocumentManager:
    """
    Manages the live research briefing document.
    
    This generates and updates a living document that summarizes
    the user's research landscape across multiple runs.
    """
    
    def __init__(
        self,
        max_top_papers: int = 10,
        max_recent_papers: int = 20,
        rolling_window_days: int = 7,
        model: str = None,
    ):
        """
        Initialize the manager.
        
        Args:
            max_top_papers: Maximum papers in top papers section
            max_recent_papers: Maximum papers in recent section
            rolling_window_days: Days to include in rolling window
            model: OpenAI model for summary generation
        """
        self.max_top_papers = max_top_papers
        self.max_recent_papers = max_recent_papers
        self.rolling_window_days = rolling_window_days
        self.model = model or os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    
    def _fetch_existing_document(self, user_id: str) -> Optional[LiveDocumentData]:
        """Fetch existing document from database."""
        try:
            from db.database import is_database_configured, get_db_session
            from db.orm_models import LiveDocument
            
            if not is_database_configured():
                return None
            
            with get_db_session() as db:
                doc = db.query(LiveDocument).filter_by(
                    user_id=uuid.UUID(user_id)
                ).first()
                
                if doc and doc.document_data:
                    return LiveDocumentData(**doc.document_data)
                
            return None
            
        except Exception as e:
            logger.warning(f"Failed to fetch existing document: {e}")
            return None
    
    def _format_paper(self, paper: Dict) -> DocumentPaper:
        """Format a paper for document display."""
        arxiv_id = paper.get("arxiv_id", "")
        
        return DocumentPaper(
            arxiv_id=arxiv_id,
            title=paper.get("title", "Untitled"),
            authors=paper.get("authors", [])[:3],
            abstract_snippet=paper.get("abstract", ""),
            relevance_score=paper.get("relevance_score", 0),
            novelty_score=paper.get("novelty_score", 0),
            llm_novelty_score=paper.get("llm_novelty_score"),
            categories=paper.get("categories", []),
            added_at=datetime.utcnow().isoformat() + "Z",
            arxiv_url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
        )
    
    def _detect_trending_topics(
        self,
        papers: List[Dict],
        user_profile: Dict,
    ) -> List[TrendingTopic]:
        """Detect trending topics from papers."""
        topic_counts: Dict[str, List[float]] = {}
        user_topics = set(t.lower() for t in user_profile.get("research_topics", []))
        
        # Count papers per category and track relevance
        for paper in papers:
            cats = paper.get("categories", [])
            relevance = paper.get("relevance_score", 0)
            
            for cat in cats:
                if cat not in topic_counts:
                    topic_counts[cat] = []
                topic_counts[cat].append(relevance)
        
        # Build trending topics
        topics = []
        for topic, relevances in topic_counts.items():
            if len(relevances) >= 2:
                topics.append(TrendingTopic(
                    topic=topic,
                    paper_count=len(relevances),
                    avg_relevance=sum(relevances) / len(relevances),
                    emerging=topic.lower() not in user_topics,
                ))
        
        # Sort by paper count and relevance
        topics.sort(key=lambda t: (t.paper_count, t.avg_relevance), reverse=True)
        
        return topics[:10]
    
    def _generate_executive_summary(
        self,
        user_profile: Dict,
        top_papers: List[DocumentPaper],
        trending_topics: List[TrendingTopic],
    ) -> str:
        """Generate AI-powered executive summary."""
        if not top_papers:
            return "No papers analyzed yet. Run a research pulse to populate this document."
        
        try:
            from openai import OpenAI
            
            api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
            api_base = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
            if not api_key:
                logger.warning("Live document: neither LLM_API_KEY nor OPENAI_API_KEY is set")
                return f"This live document summarises the {len(top_papers)} most relevant papers based on your research interests."
            client = OpenAI(api_key=api_key, base_url=api_base)
            
            papers_text = "\n".join([
                f"- {p.title} (relevance: {p.relevance_score:.2f})"
                for p in top_papers[:5]
            ])
            
            topics_text = "\n".join([
                f"- {t.topic}: {t.paper_count} papers (avg relevance: {t.avg_relevance:.2f})"
                + (" [EMERGING]" if t.emerging else "")
                for t in trending_topics[:5]
            ])
            
            prompt = EXECUTIVE_SUMMARY_PROMPT.format(
                research_topics=", ".join(user_profile.get("research_topics", [])),
                papers_summary=papers_text or "No papers yet",
                trending_topics=topics_text or "No trends detected",
            )
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"Executive summary generation failed: {e}")
            return f"This live document summarises the {len(top_papers)} most relevant papers based on your research interests."
    
    def _build_category_breakdown(self, papers: List[Dict]) -> Dict[str, int]:
        """Build category breakdown from papers."""
        counts: Dict[str, int] = {}
        
        for paper in papers:
            for cat in paper.get("categories", []):
                counts[cat] = counts.get(cat, 0) + 1
        
        # Sort by count
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:15])
    
    def generate_document(
        self,
        run_id: str,
        user_id: str,
        user_profile: Dict[str, Any],
        scored_papers: List[Dict],
        mode: str = "incremental",
    ) -> LiveDocumentData:
        """
        Generate or update the live document.
        
        Args:
            run_id: Current run ID
            user_id: User UUID string
            user_profile: User's research profile
            scored_papers: Papers from this run
            mode: "full" or "incremental"
        
        Returns:
            Updated LiveDocumentData
        """
        now = datetime.utcnow().isoformat() + "Z"
        
        # Fetch or create document
        if mode == "incremental":
            existing = self._fetch_existing_document(user_id)
        else:
            existing = None
        
        if existing:
            doc = existing
            doc.change_log.insert(0, f"[{now}] Incremental update from run {run_id}")
        else:
            doc = LiveDocumentData(
                user_id=user_id,
                change_log=[f"[{now}] Document created from run {run_id}"],
            )
        
        doc.last_updated = now
        
        if run_id not in doc.runs_included:
            doc.runs_included.append(run_id)
        
        # Format new papers
        new_papers = [self._format_paper(p) for p in scored_papers]
        
        # Merge with existing papers
        all_paper_ids = {p.arxiv_id for p in doc.top_papers}
        all_paper_ids.update(p.arxiv_id for p in doc.recent_papers)
        
        for paper in new_papers:
            if paper.arxiv_id not in all_paper_ids:
                doc.recent_papers.insert(0, paper)
                all_paper_ids.add(paper.arxiv_id)
        
        # Trim recent papers to rolling window
        cutoff = (datetime.utcnow() - timedelta(days=self.rolling_window_days)).isoformat() + "Z"
        doc.recent_papers = [
            p for p in doc.recent_papers 
            if p.added_at >= cutoff
        ][:self.max_recent_papers]
        
        # Update top papers (by combined score)
        all_papers = doc.top_papers + new_papers
        all_papers.sort(
            key=lambda p: p.relevance_score + (p.novelty_score * 0.5),
            reverse=True
        )
        
        # Deduplicate
        seen_ids = set()
        deduped = []
        for p in all_papers:
            if p.arxiv_id not in seen_ids:
                deduped.append(p)
                seen_ids.add(p.arxiv_id)
        
        doc.top_papers = deduped[:self.max_top_papers]
        
        # Update trending topics
        all_raw_papers = scored_papers
        doc.trending_topics = self._detect_trending_topics(all_raw_papers, user_profile)
        
        # Update category breakdown
        doc.category_breakdown = self._build_category_breakdown(scored_papers)
        
        # Update total count
        doc.total_papers_tracked = len(seen_ids)
        
        # Generate executive summary
        doc.executive_summary = self._generate_executive_summary(
            user_profile,
            doc.top_papers,
            doc.trending_topics,
        )
        
        # Trim change log
        doc.change_log = doc.change_log[:20]
        
        return doc
    
    def render_markdown(self, doc: LiveDocumentData) -> str:
        """Render the document as markdown."""
        lines = []
        
        # Header
        lines.append(f"# {doc.title}")
        lines.append(f"")
        lines.append(f"*Last updated: {doc.last_updated}*")
        lines.append(f"")
        
        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(doc.executive_summary)
        lines.append("")
        
        # Top Papers
        lines.append(f"## Top Papers ({len(doc.top_papers)})")
        lines.append("")
        
        for i, paper in enumerate(doc.top_papers, 1):
            novelty_val = (paper.llm_novelty_score / 100.0) if paper.llm_novelty_score else paper.novelty_score
            novelty_str = f" | Novelty: {novelty_val:.2f}" if novelty_val else ""
            combined = paper.relevance_score + (novelty_val or 0)
            
            lines.append(f"### {i}. {paper.title}")
            lines.append(f"")
            lines.append(f"**Relevance:** {paper.relevance_score:.2f}{novelty_str} | **Score:** {combined:.2f}")
            lines.append(f"")
            if paper.authors:
                lines.append(f"**Authors:** {', '.join(paper.authors)}")
                lines.append("")
            if paper.categories:
                lines.append(f"**Categories:** {', '.join(paper.categories)}")
                lines.append("")
            if paper.abstract_snippet:
                lines.append(f"> {paper.abstract_snippet}")
                lines.append("")
            if paper.arxiv_url:
                lines.append(f"[View on arXiv]({paper.arxiv_url})")
                lines.append("")
        
        return "\n".join(lines)
    
    def render_text(self, doc: LiveDocumentData) -> str:
        """Render the document as plain text (no markdown formatting)."""
        lines = []

        lines.append(doc.title.upper())
        lines.append("=" * len(doc.title))
        lines.append(f"Last updated: {doc.last_updated}")
        lines.append("")

        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 17)
        lines.append(doc.executive_summary)
        lines.append("")

        lines.append(f"TOP PAPERS ({len(doc.top_papers)})")
        lines.append("-" * 20)
        for i, paper in enumerate(doc.top_papers, 1):
            novelty_val = (paper.llm_novelty_score / 100.0) if paper.llm_novelty_score else paper.novelty_score
            novelty_str = f" | Novelty: {novelty_val:.2f}" if novelty_val else ""
            combined = paper.relevance_score + (novelty_val or 0)
            lines.append(f"{i}. {paper.title}")
            lines.append(f"   Relevance: {paper.relevance_score:.2f}{novelty_str} | Score: {combined:.2f}")
            if paper.authors:
                lines.append(f"   Authors: {', '.join(paper.authors)}")
            if paper.categories:
                lines.append(f"   Categories: {', '.join(paper.categories)}")
            if paper.abstract_snippet:
                lines.append(f"   {paper.abstract_snippet}")
            if paper.arxiv_url:
                lines.append(f"   {paper.arxiv_url}")
            lines.append("")

        return "\n".join(lines)

    def _md_to_html(self, md: str) -> str:
        """Convert markdown text to HTML using regex-based parsing."""
        import re
        lines = md.split('\n')
        html_lines = []
        in_table = False
        in_blockquote = False
        in_ul = False

        for line in lines:
            stripped = line.strip()

            # Close blockquote if no longer in one
            if in_blockquote and not stripped.startswith('>'):
                html_lines.append('</blockquote>')
                in_blockquote = False

            # Close list if no longer in one
            if in_ul and not stripped.startswith('- ') and not stripped.startswith('* '):
                html_lines.append('</ul>')
                in_ul = False

            # Table rows
            if stripped.startswith('|') and stripped.endswith('|'):
                cells = [c.strip() for c in stripped.strip('|').split('|')]
                # Skip separator rows like |---|---|
                if all(re.match(r'^[-:]+$', c) for c in cells):
                    continue
                if not in_table:
                    in_table = True
                    html_lines.append('<table>')
                    # First row is header
                    html_lines.append('<tr>' + ''.join(f'<th>{self._inline_md(c)}</th>' for c in cells) + '</tr>')
                else:
                    html_lines.append('<tr>' + ''.join(f'<td>{self._inline_md(c)}</td>' for c in cells) + '</tr>')
                continue
            elif in_table:
                html_lines.append('</table>')
                in_table = False

            # Headers
            if stripped.startswith('### '):
                html_lines.append(f'<h3>{self._inline_md(stripped[4:])}</h3>')
            elif stripped.startswith('## '):
                html_lines.append(f'<h2>{self._inline_md(stripped[3:])}</h2>')
            elif stripped.startswith('# '):
                html_lines.append(f'<h1>{self._inline_md(stripped[2:])}</h1>')
            elif stripped.startswith('> '):
                if not in_blockquote:
                    html_lines.append('<blockquote>')
                    in_blockquote = True
                html_lines.append(f'<p>{self._inline_md(stripped[2:])}</p>')
            elif stripped.startswith('- ') or stripped.startswith('* '):
                if not in_ul:
                    html_lines.append('<ul>')
                    in_ul = True
                html_lines.append(f'<li>{self._inline_md(stripped[2:])}</li>')
            elif stripped.startswith('---') or stripped.startswith('***'):
                html_lines.append('<hr>')
            elif stripped == '':
                html_lines.append('')
            else:
                html_lines.append(f'<p>{self._inline_md(stripped)}</p>')

        # Close any open tags
        if in_table:
            html_lines.append('</table>')
        if in_blockquote:
            html_lines.append('</blockquote>')
        if in_ul:
            html_lines.append('</ul>')

        return '\n'.join(html_lines)

    def _inline_md(self, text: str) -> str:
        """Convert inline markdown (bold, italic, links, code) to HTML."""
        import re
        # Escape HTML
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        # Bold: **text** or __text__
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
        # Italic: *text* or _text_
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        text = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'<em>\1</em>', text)
        # Inline code: `text`
        text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
        # Links: [text](url)
        text = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2" target="_blank">\1</a>', text)
        return text

    def render_html(self, doc: LiveDocumentData) -> str:
        """Render the document as properly formatted HTML."""
        markdown = self.render_markdown(doc)
        body_html = self._md_to_html(markdown)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{doc.title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; line-height: 1.6; color: #1a1a2e; }}
        h1 {{ color: #2563eb; border-bottom: 2px solid #2563eb; padding-bottom: 10px; }}
        h2 {{ color: #1e40af; margin-top: 30px; }}
        h3 {{ color: #1e3a8a; }}
        p {{ margin: 6px 0; }}
        blockquote {{ background: #f3f4f6; padding: 15px; border-left: 4px solid #2563eb; margin: 10px 0; border-radius: 4px; }}
        blockquote p {{ margin: 4px 0; }}
        ul {{ padding-left: 24px; }}
        li {{ margin: 4px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #e5e7eb; padding: 10px; text-align: left; }}
        th {{ background: #f3f4f6; font-weight: 600; }}
        a {{ color: #2563eb; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
        hr {{ border: none; border-top: 1px solid #e5e7eb; margin: 20px 0; }}
        strong {{ color: #111827; }}
    </style>
</head>
<body>
{body_html}
</body>
</html>"""

        return html

    def render_pdf(self, doc: LiveDocumentData) -> bytes:
        """Render the document as a PDF binary using fpdf2."""
        from fpdf import FPDF

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Title
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, doc.title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 6, f"Last updated: {doc.last_updated}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # Executive Summary
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Executive Summary", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, doc.executive_summary)
        pdf.ln(4)

        # Papers
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, f"Latest Papers ({len(doc.top_papers)})", new_x="LMARGIN", new_y="NEXT")

        for i, paper in enumerate(doc.top_papers, 1):
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 6, f"{i}. {paper.title}")

            pdf.set_font("Helvetica", "", 9)
            novelty_val = (paper.llm_novelty_score / 100.0) if paper.llm_novelty_score else paper.novelty_score
            novelty_str = f" | Novelty: {novelty_val:.2f}" if novelty_val else ""
            combined = paper.relevance_score + (novelty_val or 0)
            pdf.cell(0, 5, f"Relevance: {paper.relevance_score:.2f}{novelty_str} | Score: {combined:.2f}", new_x="LMARGIN", new_y="NEXT")

            if paper.authors:
                pdf.cell(0, 5, f"Authors: {', '.join(paper.authors)}", new_x="LMARGIN", new_y="NEXT")
            if paper.categories:
                pdf.cell(0, 5, f"Categories: {', '.join(paper.categories)}", new_x="LMARGIN", new_y="NEXT")
            if paper.abstract_snippet:
                pdf.set_font("Helvetica", "I", 9)
                pdf.multi_cell(0, 5, paper.abstract_snippet)
            if paper.arxiv_url:
                pdf.set_font("Helvetica", "", 9)
                pdf.cell(0, 5, paper.arxiv_url, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)

        return pdf.output()


# =============================================================================
# Database Operations
# =============================================================================

def save_live_document(doc: LiveDocumentData) -> Dict[str, Any]:
    """
    Save live document to database (upsert).
    
    Args:
        doc: The document data to save
    
    Returns:
        Dict with success status
    """
    try:
        from db.database import is_database_configured, get_db_session
        from db.orm_models import LiveDocument, LiveDocumentHistory
        
        if not is_database_configured():
            logger.warning("Database not configured, skipping document save")
            return {"success": False, "error": "database_not_configured"}
        
        with get_db_session() as db:
            user_uuid = uuid.UUID(doc.user_id)
            
            # Find existing
            existing = db.query(LiveDocument).filter_by(
                user_id=user_uuid
            ).first()
            
            now = datetime.utcnow()
            doc_dict = doc.model_dump()
            
            if existing:
                # Save current version to history
                history = LiveDocumentHistory(
                    document_id=existing.id,
                    document_data=existing.document_data,
                    markdown_content=existing.markdown_content,
                    created_at=existing.updated_at or existing.created_at,
                )
                db.add(history)
                
                # Update existing
                existing.document_data = doc_dict
                existing.updated_at = now
                
                # Generate and save rendered content
                manager = LiveDocumentManager()
                existing.markdown_content = manager.render_markdown(doc)
                
                doc_id = str(existing.id)
            else:
                # Create new
                manager = LiveDocumentManager()
                
                new_doc = LiveDocument(
                    user_id=user_uuid,
                    title=doc.title,
                    document_data=doc_dict,
                    markdown_content=manager.render_markdown(doc),
                    created_at=now,
                )
                db.add(new_doc)
                db.flush()
                doc_id = str(new_doc.id)
            
            db.commit()
        
        logger.info(f"Saved live document for user {doc.user_id}")
        
        return {"success": True, "document_id": doc_id}
        
    except Exception as e:
        logger.error(f"Failed to save live document: {e}")
        return {"success": False, "error": str(e)}


def get_live_document(user_id: uuid.UUID) -> Optional[Dict[str, Any]]:
    """Get the live document for a user."""
    try:
        from db.database import is_database_configured, get_db_session
        from db.orm_models import LiveDocument, live_document_to_dict
        
        if not is_database_configured():
            return None
        
        with get_db_session() as db:
            doc = db.query(LiveDocument).filter_by(
                user_id=user_id
            ).first()
            
            if doc:
                return live_document_to_dict(doc)
            
            return None
            
    except Exception as e:
        logger.error(f"Failed to get live document: {e}")
        return None


def get_document_history(
    document_id: uuid.UUID,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Get version history for a document."""
    try:
        from db.database import is_database_configured, get_db_session
        from db.orm_models import LiveDocumentHistory, live_document_history_to_dict
        
        if not is_database_configured():
            return []
        
        with get_db_session() as db:
            history = db.query(LiveDocumentHistory).filter_by(
                document_id=document_id
            ).order_by(
                LiveDocumentHistory.created_at.desc()
            ).limit(limit).all()
            
            return [live_document_history_to_dict(h) for h in history]
            
    except Exception as e:
        logger.error(f"Failed to get document history: {e}")
        return []


# =============================================================================
# Integration Helper
# =============================================================================

def update_live_document_from_run(
    run_id: str,
    user_id: str,
    user_profile: Dict[str, Any],
    scored_papers: List[Dict],
) -> Dict[str, Any]:
    """
    Main integration point for live document updates.
    
    This function:
    1. Checks if feature is enabled
    2. Creates manager with config
    3. Generates/updates document
    4. Saves to database
    5. Returns document and rendered content
    
    Args:
        run_id: Current run ID
        user_id: User UUID string
        user_profile: User's research profile
        scored_papers: Papers scored in this run
    
    Returns:
        Dict with document data and rendered markdown
    """
    from config.feature_flags import is_feature_enabled, get_feature_config
    
    if not is_feature_enabled("LIVE_DOCUMENT", user_id):
        return {
            "enabled": False,
        }
    
    try:
        config = get_feature_config().live_document
        
        manager = LiveDocumentManager(
            max_top_papers=config.max_top_papers,
            max_recent_papers=config.max_recent_papers,
            rolling_window_days=config.rolling_window_days,
            model=config.model,
        )
        
        # Generate document
        doc = manager.generate_document(
            run_id=run_id,
            user_id=user_id,
            user_profile=user_profile,
            scored_papers=scored_papers,
            mode="incremental",
        )
        
        # Save to database
        save_result = save_live_document(doc)
        
        # Render
        markdown = manager.render_markdown(doc)
        
        return {
            "enabled": True,
            "document": doc.model_dump(),
            "markdown": markdown,
            "save_result": save_result,
            "top_papers_count": len(doc.top_papers),
            "trending_topics_count": len(doc.trending_topics),
        }
        
    except Exception as e:
        logger.error(f"Live document update failed: {e}")
        return {
            "enabled": True,
            "error": str(e),
        }
