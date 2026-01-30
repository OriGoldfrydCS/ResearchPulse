"""
JSON Store - Utilities for reading and writing JSON database files.

Provides atomic writes and helpful error messages for the demo JSON DBs.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Default data directory (relative to project root)
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"


class JSONStoreError(Exception):
    """Base exception for JSON store operations."""
    pass


class FileNotFoundError(JSONStoreError):
    """Raised when a JSON file is not found."""
    pass


class JSONParseError(JSONStoreError):
    """Raised when JSON parsing fails."""
    pass


class WriteError(JSONStoreError):
    """Raised when writing to a file fails."""
    pass


# =============================================================================
# Core Load/Save Functions
# =============================================================================

def load_json(path: str | Path) -> dict | list:
    """
    Load and parse a JSON file with helpful error messages.
    
    Args:
        path: Path to the JSON file.
        
    Returns:
        Parsed JSON data (dict or list).
        
    Raises:
        FileNotFoundError: If the file does not exist.
        JSONParseError: If the file contains invalid JSON.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}\n"
            f"Expected location: {path.absolute()}\n"
            f"Please ensure the data files are present in the data/ directory."
        )
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise JSONParseError(
            f"Invalid JSON in file: {path}\n"
            f"Error at line {e.lineno}, column {e.colno}: {e.msg}\n"
            f"Please check the file for syntax errors."
        ) from e
    except PermissionError as e:
        raise JSONStoreError(
            f"Permission denied reading file: {path}\n"
            f"Please check file permissions."
        ) from e


def save_json(path: str | Path, data: dict | list, indent: int = 2) -> None:
    """
    Save data to a JSON file with atomic write (temp file then replace).
    
    This ensures that the file is never left in a partially written state
    if the process is interrupted.
    
    Args:
        path: Path to the JSON file.
        data: Data to serialize and write.
        indent: JSON indentation level (default: 2).
        
    Raises:
        WriteError: If writing fails.
    """
    path = Path(path)
    
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Write to a temporary file in the same directory
        # This ensures atomic rename will work (same filesystem)
        fd, temp_path = tempfile.mkstemp(
            suffix=".json.tmp",
            prefix=path.stem + "_",
            dir=path.parent
        )
        
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
                f.write("\n")  # Trailing newline
            
            # Atomic replace (works on both Windows and Unix)
            # On Windows, we need to remove the target first
            if os.name == "nt" and path.exists():
                os.replace(temp_path, str(path))
            else:
                os.rename(temp_path, str(path))
                
        except Exception:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
            
    except PermissionError as e:
        raise WriteError(
            f"Permission denied writing to file: {path}\n"
            f"Please check file and directory permissions."
        ) from e
    except OSError as e:
        raise WriteError(
            f"Failed to write file: {path}\n"
            f"OS error: {e}"
        ) from e


# =============================================================================
# Research Profile Helpers
# =============================================================================

def get_research_profile(data_dir: Path | None = None) -> dict:
    """
    Load the research profile configuration.
    
    Args:
        data_dir: Optional data directory path. Uses default if not provided.
        
    Returns:
        Research profile dictionary containing:
        - researcher_name
        - affiliation
        - research_topics
        - my_papers
        - preferred_venues
        - avoid_topics
        - time_budget_per_week_minutes
        - arxiv_categories_include
        - arxiv_categories_exclude
        - stop_policy
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    return load_json(data_dir / "research_profile.json")


def get_research_topics(data_dir: Path | None = None) -> List[str]:
    """Get the list of research topics from the profile."""
    profile = get_research_profile(data_dir)
    return profile.get("research_topics", [])


def get_arxiv_categories(data_dir: Path | None = None) -> tuple[List[str], List[str]]:
    """
    Get the arXiv category include/exclude lists.
    
    Returns:
        Tuple of (include_categories, exclude_categories)
    """
    profile = get_research_profile(data_dir)
    return (
        profile.get("arxiv_categories_include", []),
        profile.get("arxiv_categories_exclude", [])
    )


def get_stop_policy_from_profile(data_dir: Path | None = None) -> dict:
    """Get the stop policy from the research profile."""
    profile = get_research_profile(data_dir)
    return profile.get("stop_policy", {})


# =============================================================================
# Colleagues Helpers
# =============================================================================

def get_colleagues(data_dir: Path | None = None) -> List[dict]:
    """
    Load the colleagues database.
    
    Args:
        data_dir: Optional data directory path.
        
    Returns:
        List of colleague dictionaries, each containing:
        - id
        - name
        - email
        - affiliation
        - topics
        - sharing_preference
        - arxiv_categories_interest (optional)
        - notes (optional)
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    data = load_json(data_dir / "colleagues.json")
    return data.get("colleagues", [])


def get_colleague_by_id(colleague_id: str, data_dir: Path | None = None) -> Optional[dict]:
    """Get a specific colleague by their ID."""
    colleagues = get_colleagues(data_dir)
    for colleague in colleagues:
        if colleague.get("id") == colleague_id:
            return colleague
    return None


def get_colleagues_for_topics(topics: List[str], data_dir: Path | None = None) -> List[dict]:
    """
    Find colleagues interested in any of the given topics.
    
    Args:
        topics: List of topics to match.
        data_dir: Optional data directory path.
        
    Returns:
        List of matching colleagues.
    """
    colleagues = get_colleagues(data_dir)
    topics_lower = {t.lower() for t in topics}
    
    matching = []
    for colleague in colleagues:
        colleague_topics = {t.lower() for t in colleague.get("topics", [])}
        if colleague_topics & topics_lower:  # Intersection
            matching.append(colleague)
    
    return matching


# =============================================================================
# Delivery Policy Helpers
# =============================================================================

def get_delivery_policy(data_dir: Path | None = None) -> dict:
    """
    Load the delivery policy configuration.
    
    Args:
        data_dir: Optional data directory path.
        
    Returns:
        Delivery policy dictionary containing:
        - importance_policies (high/medium/low/log_only)
        - email_settings
        - calendar_settings
        - reading_list_settings
        - colleague_sharing_settings
        - global_settings
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    return load_json(data_dir / "delivery_policy.json")


def get_policy_for_importance(
    importance: str,
    data_dir: Path | None = None
) -> dict:
    """
    Get the delivery policy for a specific importance level.
    
    Args:
        importance: One of "high", "medium", "low", "log_only"
        data_dir: Optional data directory path.
        
    Returns:
        Policy dictionary for the importance level.
    """
    policy = get_delivery_policy(data_dir)
    importance_policies = policy.get("importance_policies", {})
    return importance_policies.get(importance, importance_policies.get("log_only", {}))


# =============================================================================
# Papers State Helpers
# =============================================================================

def get_papers_state(data_dir: Path | None = None) -> dict:
    """
    Load the papers state database.
    
    Args:
        data_dir: Optional data directory path.
        
    Returns:
        Papers state dictionary containing:
        - papers: List of paper records
        - last_updated: Timestamp
        - total_papers_seen: Count
        - stats: Decision counts
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    return load_json(data_dir / "papers_state.json")


def get_seen_paper_ids(data_dir: Path | None = None) -> Set[str]:
    """
    Get the set of all paper IDs that have been seen.
    
    Args:
        data_dir: Optional data directory path.
        
    Returns:
        Set of paper_id strings.
    """
    state = get_papers_state(data_dir)
    papers = state.get("papers", [])
    return {p.get("paper_id") for p in papers if p.get("paper_id")}


def get_paper_by_id(paper_id: str, data_dir: Path | None = None) -> Optional[dict]:
    """
    Get a specific paper record by its ID.
    
    Args:
        paper_id: The arXiv paper ID.
        data_dir: Optional data directory path.
        
    Returns:
        Paper record dictionary or None if not found.
    """
    state = get_papers_state(data_dir)
    papers = state.get("papers", [])
    for paper in papers:
        if paper.get("paper_id") == paper_id:
            return paper
    return None


def upsert_paper(
    paper_record: dict,
    data_dir: Path | None = None
) -> dict:
    """
    Insert or update a paper record in the papers state.
    
    If a paper with the same paper_id exists, it will be updated.
    Otherwise, a new record will be inserted.
    
    Args:
        paper_record: Dictionary containing at minimum:
            - paper_id: arXiv paper ID
            - title: Paper title
            - decision: ignored | saved | shared | logged
            - importance: high | medium | low
        data_dir: Optional data directory path.
        
    Returns:
        The updated papers state dictionary.
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    state = get_papers_state(data_dir)
    papers = state.get("papers", [])
    
    paper_id = paper_record.get("paper_id")
    if not paper_id:
        raise ValueError("paper_record must contain a 'paper_id' field")
    
    # Ensure required fields have defaults
    paper_record.setdefault("date_seen", datetime.utcnow().isoformat() + "Z")
    paper_record.setdefault("decision", "logged")
    paper_record.setdefault("importance", "low")
    paper_record.setdefault("notes", "")
    paper_record.setdefault("embedded_in_pinecone", False)
    
    # Find existing paper or append new one
    found = False
    for i, paper in enumerate(papers):
        if paper.get("paper_id") == paper_id:
            # Update existing record (merge fields)
            papers[i] = {**paper, **paper_record}
            found = True
            break
    
    if not found:
        papers.append(paper_record)
    
    # Update metadata
    state["papers"] = papers
    state["last_updated"] = datetime.utcnow().isoformat() + "Z"
    state["total_papers_seen"] = len(papers)
    
    # Recalculate stats
    stats = {"saved": 0, "shared": 0, "ignored": 0, "logged": 0}
    for paper in papers:
        decision = paper.get("decision", "logged")
        if decision in stats:
            stats[decision] += 1
    state["stats"] = stats
    
    # Save atomically
    save_json(data_dir / "papers_state.json", state)
    
    return state


def upsert_papers(
    paper_records: List[dict],
    data_dir: Path | None = None
) -> dict:
    """
    Insert or update multiple paper records in batch.
    
    More efficient than calling upsert_paper multiple times
    as it only writes to disk once.
    
    Args:
        paper_records: List of paper record dictionaries.
        data_dir: Optional data directory path.
        
    Returns:
        The updated papers state dictionary.
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    state = get_papers_state(data_dir)
    papers = state.get("papers", [])
    
    # Build lookup for existing papers
    papers_by_id = {p.get("paper_id"): i for i, p in enumerate(papers)}
    
    for paper_record in paper_records:
        paper_id = paper_record.get("paper_id")
        if not paper_id:
            continue
        
        # Ensure required fields have defaults
        paper_record.setdefault("date_seen", datetime.utcnow().isoformat() + "Z")
        paper_record.setdefault("decision", "logged")
        paper_record.setdefault("importance", "low")
        paper_record.setdefault("notes", "")
        paper_record.setdefault("embedded_in_pinecone", False)
        
        if paper_id in papers_by_id:
            # Update existing
            idx = papers_by_id[paper_id]
            papers[idx] = {**papers[idx], **paper_record}
        else:
            # Insert new
            papers.append(paper_record)
            papers_by_id[paper_id] = len(papers) - 1
    
    # Update metadata
    state["papers"] = papers
    state["last_updated"] = datetime.utcnow().isoformat() + "Z"
    state["total_papers_seen"] = len(papers)
    
    # Recalculate stats
    stats = {"saved": 0, "shared": 0, "ignored": 0, "logged": 0}
    for paper in papers:
        decision = paper.get("decision", "logged")
        if decision in stats:
            stats[decision] += 1
    state["stats"] = stats
    
    # Save atomically
    save_json(data_dir / "papers_state.json", state)
    
    return state


# =============================================================================
# arXiv Categories Helpers
# =============================================================================

def get_arxiv_categories_db(data_dir: Path | None = None) -> List[dict]:
    """
    Load the arXiv categories database.
    
    Args:
        data_dir: Optional data directory path.
        
    Returns:
        List of category dictionaries with:
        - category_code
        - category_name
        - source
        - last_updated
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    data = load_json(data_dir / "arxiv_categories.json")
    return data.get("categories", [])


def get_category_name(category_code: str, data_dir: Path | None = None) -> Optional[str]:
    """Get the human-readable name for a category code."""
    categories = get_arxiv_categories_db(data_dir)
    for cat in categories:
        if cat.get("category_code") == category_code:
            return cat.get("category_name")
    return None


def validate_categories(
    category_codes: List[str],
    data_dir: Path | None = None
) -> tuple[List[str], List[str]]:
    """
    Validate a list of category codes against the categories DB.
    
    Args:
        category_codes: List of category codes to validate.
        data_dir: Optional data directory path.
        
    Returns:
        Tuple of (valid_codes, invalid_codes)
    """
    categories = get_arxiv_categories_db(data_dir)
    known_codes = {cat.get("category_code") for cat in categories}
    
    valid = [c for c in category_codes if c in known_codes]
    invalid = [c for c in category_codes if c not in known_codes]
    
    return valid, invalid


# =============================================================================
# Self-Check
# =============================================================================

def self_check() -> bool:
    """
    Run self-check tests for the JSON store utilities.
    
    Returns:
        True if all checks pass, False otherwise.
    """
    print("=" * 60)
    print("JSON Store Self-Check")
    print("=" * 60)
    
    all_passed = True
    
    def check(name: str, condition: bool) -> bool:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        return condition

    # Test 1: Load research profile
    print("\n1. Load Research Profile:")
    try:
        profile = get_research_profile()
        all_passed &= check("loads successfully", profile is not None)
        all_passed &= check("has researcher_name", "researcher_name" in profile)
        all_passed &= check("has stop_policy", "stop_policy" in profile)
    except Exception as e:
        all_passed &= check(f"load failed: {e}", False)

    # Test 2: Load colleagues
    print("\n2. Load Colleagues:")
    try:
        colleagues = get_colleagues()
        all_passed &= check("loads successfully", colleagues is not None)
        all_passed &= check("is a list", isinstance(colleagues, list))
        all_passed &= check("has entries", len(colleagues) > 0)
    except Exception as e:
        all_passed &= check(f"load failed: {e}", False)

    # Test 3: Load delivery policy
    print("\n3. Load Delivery Policy:")
    try:
        policy = get_delivery_policy()
        all_passed &= check("loads successfully", policy is not None)
        all_passed &= check("has importance_policies", "importance_policies" in policy)
    except Exception as e:
        all_passed &= check(f"load failed: {e}", False)

    # Test 4: Get seen paper IDs
    print("\n4. Get Seen Paper IDs:")
    try:
        seen_ids = get_seen_paper_ids()
        all_passed &= check("returns a set", isinstance(seen_ids, set))
        all_passed &= check("has entries", len(seen_ids) > 0)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 5: arXiv categories
    print("\n5. arXiv Categories:")
    try:
        categories = get_arxiv_categories_db()
        all_passed &= check("loads successfully", categories is not None)
        all_passed &= check("has cs.CL", any(c["category_code"] == "cs.CL" for c in categories))
        name = get_category_name("cs.CL")
        all_passed &= check("get_category_name works", name is not None)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 6: Validate categories
    print("\n6. Validate Categories:")
    try:
        valid, invalid = validate_categories(["cs.CL", "cs.LG", "fake.XX"])
        all_passed &= check("valid includes cs.CL", "cs.CL" in valid)
        all_passed &= check("invalid includes fake.XX", "fake.XX" in invalid)
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
