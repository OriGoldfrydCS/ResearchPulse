"""
Recipient Resolution for ResearchPulse.

Given a paper's metadata, the owner's interests, and a list of colleagues,
this module returns a de-duplicated set of recipients, each annotated with
the matching rule that triggered inclusion.

Rules
-----
1. Owner match: paper categories/topics overlap with owner interests.
2. Colleague match: paper categories/topics overlap with colleague interests.
3. Both: paper matches owner AND one or more colleagues.

Guarantee: every recipient appears at most once per paper per run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class MatchRule(str, Enum):
    OWNER_ONLY = "owner_only"
    COLLEAGUE_ONLY = "colleague_only"
    BOTH = "both"


@dataclass
class RecipientEntry:
    """A single recipient for a paper delivery."""
    email: str
    name: str
    is_owner: bool
    match_rule: MatchRule
    matched_interests: List[str] = field(default_factory=list)
    colleague_id: Optional[str] = None


def _normalise(text: str) -> str:
    return text.lower().strip()


def _interests_overlap(
    paper_topics: List[str],
    paper_categories: List[str],
    interest_topics: List[str],
    interest_categories: List[str],
) -> Tuple[bool, List[str]]:
    """Check whether a paper matches a recipient's interests.

    Returns (matched: bool, matching_terms: list[str]).
    """
    matched_terms: List[str] = []

    # Direct topic match (case-insensitive, substring)
    norm_paper = {_normalise(t) for t in paper_topics if t}
    norm_interest = {_normalise(t) for t in interest_topics if t}

    for pt in norm_paper:
        for it in norm_interest:
            if pt == it or pt in it or it in pt:
                matched_terms.append(it)

    # Category match
    paper_cats = {c.lower() for c in paper_categories}
    interest_cats = {c.lower() for c in interest_categories}
    cat_overlap = paper_cats & interest_cats
    matched_terms.extend(sorted(cat_overlap))

    # de-dup
    seen: Set[str] = set()
    unique: List[str] = []
    for t in matched_terms:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return bool(unique), unique


def resolve_recipients(
    paper_metadata: Dict[str, Any],
    owner_email: str,
    owner_name: str,
    owner_topics: List[str],
    owner_categories: List[str],
    colleagues: List[Dict[str, Any]],
) -> List[RecipientEntry]:
    """Compute the de-duplicated recipient list for a single paper.

    Parameters
    ----------
    paper_metadata : dict
        Must contain ``title``, ``categories`` (list[str]).
        May contain ``abstract``, ``authors``, ``arxiv_id``, etc.
    owner_email, owner_name : str
        Owner's contact info.
    owner_topics, owner_categories : list[str]
        Owner's research interests (free-text topics and arXiv categories).
    colleagues : list[dict]
        Each dict must have ``email``, ``name``, ``id``.
        Optional: ``topics`` (list), ``categories`` (list),
        ``enabled`` (bool, default True), ``auto_send_emails`` (bool, default True).

    Returns
    -------
    list[RecipientEntry]
        De-duplicated recipients.  At most one entry per email address.
    """
    paper_title = paper_metadata.get("title", "")
    paper_abstract = paper_metadata.get("abstract", "")
    paper_categories = paper_metadata.get("categories", [])

    # Build coarse topic tokens from title + abstract
    combined_text = f"{paper_title} {paper_abstract}".lower()
    paper_topic_tokens = [
        w.strip(".,!?;:()[]{}\"'")
        for w in combined_text.split()
        if len(w.strip(".,!?;:()[]{}\"'")) > 4
    ]

    # --- owner match ---
    owner_matched, owner_terms = _interests_overlap(
        paper_topic_tokens, paper_categories,
        owner_topics, owner_categories,
    )

    # --- colleague matches ---
    colleague_matches: List[Tuple[Dict[str, Any], List[str]]] = []
    for col in colleagues:
        if not col.get("enabled", True):
            continue
        if not col.get("auto_send_emails", True):
            continue

        col_topics = col.get("topics", [])
        col_categories = col.get("categories", [])
        matched, terms = _interests_overlap(
            paper_topic_tokens, paper_categories,
            col_topics, col_categories,
        )
        if matched:
            colleague_matches.append((col, terms))

    # --- build de-duplicated recipient set ---
    recipients: Dict[str, RecipientEntry] = {}  # keyed by lowercase email

    if owner_matched:
        recipients[owner_email.lower()] = RecipientEntry(
            email=owner_email,
            name=owner_name,
            is_owner=True,
            match_rule=MatchRule.OWNER_ONLY,  # may upgrade to BOTH below
            matched_interests=owner_terms,
        )

    for col, terms in colleague_matches:
        col_email_lower = col["email"].lower()
        if col_email_lower in recipients:
            # Already present (e.g. owner email == colleague email edge case)
            existing = recipients[col_email_lower]
            if existing.is_owner:
                existing.match_rule = MatchRule.BOTH
                existing.matched_interests = list(
                    dict.fromkeys(existing.matched_interests + terms)
                )
            continue

        rule = MatchRule.BOTH if owner_matched else MatchRule.COLLEAGUE_ONLY
        recipients[col_email_lower] = RecipientEntry(
            email=col["email"],
            name=col.get("name", ""),
            is_owner=False,
            match_rule=rule,
            matched_interests=terms,
            colleague_id=str(col.get("id", "")),
        )

    # If owner matched AND at least one colleague matched, mark owner as BOTH
    if owner_matched and colleague_matches:
        owner_entry = recipients.get(owner_email.lower())
        if owner_entry and owner_entry.match_rule == MatchRule.OWNER_ONLY:
            # Owner match_rule stays OWNER_ONLY – "BOTH" means both owner and
            # colleague independently matched; owner still only appears once.
            pass  # owner rule is already correct

    result = list(recipients.values())

    # --- audit log ---
    for r in result:
        logger.info(
            "[RECIPIENT_RESOLVE] paper=%s recipient=%s rule=%s interests=%s",
            paper_metadata.get("arxiv_id", "?"),
            r.email,
            r.match_rule.value,
            r.matched_interests[:5],
        )

    return result
