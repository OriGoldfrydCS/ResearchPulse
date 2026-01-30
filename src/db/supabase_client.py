"""
Supabase Client - Database integration for ResearchPulse.

This module provides Supabase integration as required by the Course Project specification.
It can be used alongside or as a replacement for the JSON-based storage.

Tables required in Supabase:
- papers: Store paper records with decisions and importance scores
- colleagues: Store colleague information for paper sharing
- research_profile: Store researcher profile and preferences

Environment variables required:
- SUPABASE_URL: Your Supabase project URL
- SUPABASE_KEY: Your Supabase anon/service key
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Track if Supabase is available
_supabase_client = None
_supabase_available = False


def get_supabase_client():
    """
    Get or create the Supabase client instance.
    
    Returns:
        Supabase client or None if not configured.
    """
    global _supabase_client, _supabase_available
    
    if _supabase_client is not None:
        return _supabase_client
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("[Supabase] Not configured - SUPABASE_URL and SUPABASE_KEY required")
        _supabase_available = False
        return None
    
    try:
        from supabase import create_client, Client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        _supabase_available = True
        print(f"[Supabase] Connected to {SUPABASE_URL}")
        return _supabase_client
    except ImportError:
        print("[Supabase] supabase-py not installed. Run: pip install supabase")
        _supabase_available = False
        return None
    except Exception as e:
        print(f"[Supabase] Connection failed: {e}")
        _supabase_available = False
        return None


def is_supabase_available() -> bool:
    """Check if Supabase is configured and available."""
    global _supabase_available
    if _supabase_client is None:
        get_supabase_client()
    return _supabase_available


# =============================================================================
# Papers Table Operations
# =============================================================================

def get_all_papers() -> List[Dict[str, Any]]:
    """
    Get all papers from Supabase.
    
    Returns:
        List of paper dictionaries.
    """
    client = get_supabase_client()
    if not client:
        return []
    
    try:
        response = client.table("papers").select("*").execute()
        return response.data or []
    except Exception as e:
        print(f"[Supabase] Error fetching papers: {e}")
        return []


def get_paper_by_id(paper_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific paper by its arXiv ID.
    
    Args:
        paper_id: The arXiv paper ID.
        
    Returns:
        Paper dictionary or None if not found.
    """
    client = get_supabase_client()
    if not client:
        return None
    
    try:
        response = client.table("papers").select("*").eq("paper_id", paper_id).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        print(f"[Supabase] Error fetching paper {paper_id}: {e}")
        return None


def get_seen_paper_ids() -> Set[str]:
    """
    Get the set of all paper IDs that have been seen.
    
    Returns:
        Set of paper_id strings.
    """
    client = get_supabase_client()
    if not client:
        return set()
    
    try:
        response = client.table("papers").select("paper_id").execute()
        return {p["paper_id"] for p in response.data} if response.data else set()
    except Exception as e:
        print(f"[Supabase] Error fetching seen papers: {e}")
        return set()


def upsert_paper(paper_record: Dict[str, Any]) -> bool:
    """
    Insert or update a paper record.
    
    Args:
        paper_record: Dictionary containing paper data.
        
    Returns:
        True if successful, False otherwise.
    """
    client = get_supabase_client()
    if not client:
        return False
    
    paper_id = paper_record.get("paper_id")
    if not paper_id:
        print("[Supabase] paper_record must contain 'paper_id'")
        return False
    
    # Ensure required fields
    paper_record.setdefault("date_seen", datetime.utcnow().isoformat() + "Z")
    paper_record.setdefault("decision", "logged")
    paper_record.setdefault("importance", "low")
    paper_record.setdefault("embedded_in_pinecone", False)
    
    try:
        response = client.table("papers").upsert(paper_record, on_conflict="paper_id").execute()
        return True
    except Exception as e:
        print(f"[Supabase] Error upserting paper {paper_id}: {e}")
        return False


def upsert_papers(paper_records: List[Dict[str, Any]]) -> int:
    """
    Insert or update multiple paper records.
    
    Args:
        paper_records: List of paper record dictionaries.
        
    Returns:
        Number of successfully upserted papers.
    """
    client = get_supabase_client()
    if not client:
        return 0
    
    # Prepare records with defaults
    now = datetime.utcnow().isoformat() + "Z"
    for record in paper_records:
        record.setdefault("date_seen", now)
        record.setdefault("decision", "logged")
        record.setdefault("importance", "low")
        record.setdefault("embedded_in_pinecone", False)
    
    try:
        response = client.table("papers").upsert(paper_records, on_conflict="paper_id").execute()
        return len(response.data) if response.data else 0
    except Exception as e:
        print(f"[Supabase] Error upserting papers: {e}")
        return 0


def get_papers_by_decision(decision: str) -> List[Dict[str, Any]]:
    """
    Get papers with a specific decision.
    
    Args:
        decision: One of 'saved', 'shared', 'ignored', 'logged'
        
    Returns:
        List of matching paper records.
    """
    client = get_supabase_client()
    if not client:
        return []
    
    try:
        response = client.table("papers").select("*").eq("decision", decision).execute()
        return response.data or []
    except Exception as e:
        print(f"[Supabase] Error fetching papers by decision: {e}")
        return []


def get_papers_stats() -> Dict[str, int]:
    """
    Get paper decision statistics.
    
    Returns:
        Dictionary with counts for each decision type.
    """
    papers = get_all_papers()
    stats = {"saved": 0, "shared": 0, "ignored": 0, "logged": 0}
    for paper in papers:
        decision = paper.get("decision", "logged")
        if decision in stats:
            stats[decision] += 1
    return stats


# =============================================================================
# Colleagues Table Operations
# =============================================================================

def get_all_colleagues() -> List[Dict[str, Any]]:
    """
    Get all colleagues from Supabase.
    
    Returns:
        List of colleague dictionaries.
    """
    client = get_supabase_client()
    if not client:
        return []
    
    try:
        response = client.table("colleagues").select("*").execute()
        return response.data or []
    except Exception as e:
        print(f"[Supabase] Error fetching colleagues: {e}")
        return []


def get_colleague_by_id(colleague_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific colleague by ID.
    
    Args:
        colleague_id: The colleague ID.
        
    Returns:
        Colleague dictionary or None if not found.
    """
    client = get_supabase_client()
    if not client:
        return None
    
    try:
        response = client.table("colleagues").select("*").eq("id", colleague_id).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        print(f"[Supabase] Error fetching colleague {colleague_id}: {e}")
        return None


def get_colleagues_for_topics(topics: List[str]) -> List[Dict[str, Any]]:
    """
    Find colleagues interested in any of the given topics.
    
    Args:
        topics: List of topics to match.
        
    Returns:
        List of matching colleagues.
    """
    all_colleagues = get_all_colleagues()
    topics_lower = {t.lower() for t in topics}
    
    matching = []
    for colleague in all_colleagues:
        colleague_topics = {t.lower() for t in colleague.get("topics", [])}
        if colleague_topics & topics_lower:
            matching.append(colleague)
    
    return matching


# =============================================================================
# Research Profile Operations
# =============================================================================

def get_research_profile() -> Optional[Dict[str, Any]]:
    """
    Get the research profile from Supabase.
    
    Returns:
        Research profile dictionary or None.
    """
    client = get_supabase_client()
    if not client:
        return None
    
    try:
        response = client.table("research_profile").select("*").limit(1).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        print(f"[Supabase] Error fetching research profile: {e}")
        return None


def update_research_profile(profile: Dict[str, Any]) -> bool:
    """
    Update the research profile in Supabase.
    
    Args:
        profile: The research profile data.
        
    Returns:
        True if successful, False otherwise.
    """
    client = get_supabase_client()
    if not client:
        return False
    
    try:
        response = client.table("research_profile").upsert(profile).execute()
        return True
    except Exception as e:
        print(f"[Supabase] Error updating research profile: {e}")
        return False


# =============================================================================
# Hybrid Store (JSON fallback + Supabase)
# =============================================================================

class HybridStore:
    """
    Hybrid data store that uses Supabase when available, 
    falling back to JSON files otherwise.
    
    This ensures the application works even without Supabase configured,
    making development and testing easier.
    """
    
    def __init__(self, prefer_supabase: bool = True):
        """
        Initialize the hybrid store.
        
        Args:
            prefer_supabase: If True, prefer Supabase when available.
        """
        self.prefer_supabase = prefer_supabase
        self._check_supabase()
    
    def _check_supabase(self) -> bool:
        """Check if Supabase is available."""
        return is_supabase_available() if self.prefer_supabase else False
    
    def get_seen_paper_ids(self) -> Set[str]:
        """Get seen paper IDs from the preferred store."""
        if self._check_supabase():
            return get_seen_paper_ids()
        else:
            from .json_store import get_seen_paper_ids as json_get_seen
            return json_get_seen()
    
    def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get a paper by ID from the preferred store."""
        if self._check_supabase():
            return get_paper_by_id(paper_id)
        else:
            from .json_store import get_paper_by_id as json_get_paper
            return json_get_paper(paper_id)
    
    def upsert_paper(self, paper_record: Dict[str, Any]) -> bool:
        """Upsert a paper to the preferred store."""
        if self._check_supabase():
            return upsert_paper(paper_record)
        else:
            from .json_store import upsert_paper as json_upsert
            json_upsert(paper_record)
            return True
    
    def upsert_papers(self, paper_records: List[Dict[str, Any]]) -> int:
        """Upsert multiple papers to the preferred store."""
        if self._check_supabase():
            return upsert_papers(paper_records)
        else:
            from .json_store import upsert_papers as json_upsert
            json_upsert(paper_records)
            return len(paper_records)
    
    def get_colleagues(self) -> List[Dict[str, Any]]:
        """Get colleagues from the preferred store."""
        if self._check_supabase():
            return get_all_colleagues()
        else:
            from .json_store import get_colleagues as json_get_colleagues
            return json_get_colleagues()
    
    def get_research_profile(self) -> Dict[str, Any]:
        """Get research profile from the preferred store."""
        if self._check_supabase():
            profile = get_research_profile()
            return profile if profile else {}
        else:
            from .json_store import get_research_profile as json_get_profile
            return json_get_profile()


# Create a default hybrid store instance
hybrid_store = HybridStore(prefer_supabase=True)


# =============================================================================
# Self-Check
# =============================================================================

def self_check() -> bool:
    """
    Run self-check for Supabase connectivity.
    
    Returns:
        True if Supabase is working, False otherwise.
    """
    print("=" * 60)
    print("Supabase Client Self-Check")
    print("=" * 60)
    
    print(f"\nSupabase URL: {SUPABASE_URL[:30]}..." if SUPABASE_URL else "\nSupabase URL: NOT SET")
    print(f"Supabase Key: {'*' * 10}..." if SUPABASE_KEY else "Supabase Key: NOT SET")
    
    client = get_supabase_client()
    
    if client is None:
        print("\n[WARN] Supabase not available - using JSON fallback")
        print("To enable Supabase, set SUPABASE_URL and SUPABASE_KEY in .env")
        return False
    
    print("\n[OK] Supabase client connected successfully!")
    
    # Test query
    try:
        response = client.table("papers").select("paper_id").limit(1).execute()
        print(f"[OK] Papers table accessible - {len(response.data)} sample records")
    except Exception as e:
        print(f"[WARN] Papers table not accessible: {e}")
        print("       You may need to create the 'papers' table in Supabase")
    
    return True


if __name__ == "__main__":
    self_check()
