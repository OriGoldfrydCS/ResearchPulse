"""
Tool: arxiv_categories - Fetch and manage arXiv category taxonomy.

This tool dynamically fetches arXiv categories from the official taxonomy,
caches them locally, and provides intelligent topic-to-category mapping.

**Features:**
- Fetches complete arXiv taxonomy from arxiv.org
- Caches categories locally for offline use
- Auto-updates when unknown categories are encountered
- Provides topic-to-category intelligent mapping
"""

from __future__ import annotations

import json
import re
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

ARXIV_TAXONOMY_URL = "https://arxiv.org/category_taxonomy"
CATEGORIES_FILE = Path("data/arxiv_categories.json")
CACHE_EXPIRY_DAYS = 30  # Refresh taxonomy after 30 days


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class ArxivCategory:
    """Represents an arXiv category."""
    code: str  # e.g., "cs.AI"
    name: str  # e.g., "Artificial Intelligence"
    group: str = ""  # e.g., "Computer Science"
    description: str = ""  # Full description if available
    source: str = "arxiv"
    last_updated: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category_code": self.code,
            "category_name": self.name,
            "group": self.group,
            "description": self.description,
            "source": self.source,
            "last_updated": self.last_updated or datetime.utcnow().isoformat() + "Z"
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArxivCategory":
        return cls(
            code=data.get("category_code", data.get("code", "")),
            name=data.get("category_name", data.get("name", "")),
            group=data.get("group", ""),
            description=data.get("description", ""),
            source=data.get("source", "arxiv"),
            last_updated=data.get("last_updated", "")
        )


@dataclass
class CategoryCache:
    """In-memory cache for arXiv categories."""
    categories: Dict[str, ArxivCategory] = field(default_factory=dict)
    groups: Dict[str, List[str]] = field(default_factory=dict)  # group -> [codes]
    last_sync: Optional[datetime] = None
    topic_mappings: Dict[str, List[str]] = field(default_factory=dict)  # topic -> [codes]


# Global cache
_cache: Optional[CategoryCache] = None


# =============================================================================
# Topic to Category Mapping (Dynamic)
# =============================================================================

# Base mappings for common topics - these will be enhanced by fetched data
BASE_TOPIC_MAPPINGS = {
    # Computer Science
    "machine learning": ["cs.LG", "stat.ML"],
    "deep learning": ["cs.LG", "cs.NE"],
    "artificial intelligence": ["cs.AI"],
    "natural language processing": ["cs.CL"],
    "nlp": ["cs.CL"],
    "computer vision": ["cs.CV"],
    "robotics": ["cs.RO"],
    "reinforcement learning": ["cs.LG", "cs.AI"],
    "neural networks": ["cs.NE", "cs.LG"],
    "information retrieval": ["cs.IR"],
    "databases": ["cs.DB"],
    "security": ["cs.CR"],
    "cryptography": ["cs.CR"],
    "software engineering": ["cs.SE"],
    "programming languages": ["cs.PL"],
    "distributed systems": ["cs.DC"],
    "networks": ["cs.NI"],
    "algorithms": ["cs.DS"],
    "data structures": ["cs.DS"],
    "computational complexity": ["cs.CC"],
    "graphics": ["cs.GR"],
    "human-computer interaction": ["cs.HC"],
    "hci": ["cs.HC"],
    "operating systems": ["cs.OS"],
    "compilers": ["cs.PL"],
    "formal methods": ["cs.LO"],
    "logic": ["cs.LO"],
    "multiagent systems": ["cs.MA"],
    "game theory": ["cs.GT"],
    "computational geometry": ["cs.CG"],
    "social networks": ["cs.SI"],
    "multimedia": ["cs.MM"],
    "performance": ["cs.PF"],
    "symbolic computation": ["cs.SC"],
    "sound": ["cs.SD"],
    "systems": ["cs.SY"],
    "emerging technologies": ["cs.ET"],
    "digital libraries": ["cs.DL"],
    "computers and society": ["cs.CY"],
    "cyberphysical systems": ["cs.SY", "eess.SY"],
    
    # Physics
    "physics": ["physics.gen-ph"],
    "quantum physics": ["quant-ph"],
    "quantum computing": ["quant-ph", "cs.ET"],
    "quantum mechanics": ["quant-ph"],
    "particle physics": ["hep-ph", "hep-th", "hep-ex", "hep-lat"],
    "high energy physics": ["hep-ph", "hep-th", "hep-ex"],
    "condensed matter": ["cond-mat.mes-hall", "cond-mat.mtrl-sci", "cond-mat.stat-mech"],
    "astrophysics": ["astro-ph.GA", "astro-ph.CO", "astro-ph.EP", "astro-ph.HE", "astro-ph.IM", "astro-ph.SR"],
    "cosmology": ["astro-ph.CO", "gr-qc"],
    "general relativity": ["gr-qc"],
    "nuclear physics": ["nucl-th", "nucl-ex"],
    "optics": ["physics.optics"],
    "atomic physics": ["physics.atom-ph"],
    "plasma physics": ["physics.plasm-ph"],
    "fluid dynamics": ["physics.flu-dyn"],
    "classical physics": ["physics.class-ph"],
    "computational physics": ["physics.comp-ph"],
    "data analysis": ["physics.data-an", "stat.ME"],
    "medical physics": ["physics.med-ph"],
    "accelerator physics": ["physics.acc-ph"],
    "atmospheric physics": ["physics.ao-ph"],
    "biological physics": ["physics.bio-ph"],
    "chemical physics": ["physics.chem-ph"],
    "geophysics": ["physics.geo-ph"],
    "space physics": ["physics.space-ph"],
    "superconductivity": ["cond-mat.supr-con"],
    "magnetism": ["cond-mat.mtrl-sci"],
    "semiconductors": ["cond-mat.mes-hall"],
    "statistical mechanics": ["cond-mat.stat-mech"],
    
    # Mathematics
    "mathematics": ["math.GM"],
    "algebra": ["math.RA", "math.AC", "math.GR"],
    "algebraic geometry": ["math.AG"],
    "analysis": ["math.CA", "math.FA", "math.CV"],
    "combinatorics": ["math.CO"],
    "differential geometry": ["math.DG"],
    "dynamical systems": ["math.DS"],
    "functional analysis": ["math.FA"],
    "geometry": ["math.MG", "math.DG"],
    "group theory": ["math.GR"],
    "logic": ["math.LO"],
    "number theory": ["math.NT"],
    "numerical analysis": ["math.NA"],
    "optimization": ["math.OC"],
    "probability": ["math.PR"],
    "representation theory": ["math.RT"],
    "statistics": ["stat.TH", "stat.ME", "stat.ML"],
    "topology": ["math.GT", "math.AT"],
    "category theory": ["math.CT"],
    
    # Biology / Life Sciences
    "biology": ["q-bio.GN", "q-bio.MN", "q-bio.NC", "q-bio.PE"],
    "bioinformatics": ["q-bio.GN", "q-bio.QM"],
    "genomics": ["q-bio.GN"],
    "molecular biology": ["q-bio.BM"],
    "neuroscience": ["q-bio.NC"],
    "computational biology": ["q-bio.QM"],
    "systems biology": ["q-bio.MN"],
    "cell biology": ["q-bio.CB"],
    "evolution": ["q-bio.PE"],
    "ecology": ["q-bio.PE"],
    "biophysics": ["q-bio.BM", "physics.bio-ph"],
    "genetics": ["q-bio.GN"],
    "tissues": ["q-bio.TO"],
    "subcellular": ["q-bio.SC"],
    
    # Economics / Finance
    "economics": ["econ.EM", "econ.GN", "econ.TH"],
    "econometrics": ["econ.EM"],
    "finance": ["q-fin.CP", "q-fin.EC", "q-fin.GN", "q-fin.MF", "q-fin.PM", "q-fin.PR", "q-fin.RM", "q-fin.ST", "q-fin.TR"],
    "financial economics": ["q-fin.EC"],
    "portfolio management": ["q-fin.PM"],
    "risk management": ["q-fin.RM"],
    "trading": ["q-fin.TR"],
    "pricing": ["q-fin.PR"],
    "computational finance": ["q-fin.CP"],
    "mathematical finance": ["q-fin.MF"],
    "statistical finance": ["q-fin.ST"],
    "microeconomics": ["econ.TH"],
    "macroeconomics": ["econ.GN"],
    
    # Engineering
    "electrical engineering": ["eess.SP", "eess.SY"],
    "signal processing": ["eess.SP"],
    "control systems": ["eess.SY", "cs.SY"],
    "audio processing": ["eess.AS"],
    "speech processing": ["eess.AS", "cs.CL"],
    "image processing": ["eess.IV"],
    "video processing": ["eess.IV"],
    
    # Statistics
    "statistical methods": ["stat.ME"],
    "bayesian": ["stat.ME", "stat.ML"],
    "time series": ["stat.ME", "stat.AP"],
    "causal inference": ["stat.ME"],
    "applications": ["stat.AP"],
    "computation": ["stat.CO"],
    "methodology": ["stat.ME"],
    "other statistics": ["stat.OT"],
    
    # Interdisciplinary
    "climate": ["physics.ao-ph", "astro-ph.EP"],
    "energy": ["physics.soc-ph", "cond-mat.mtrl-sci"],
    "social science": ["physics.soc-ph", "cs.SI", "cs.CY"],
    "network science": ["cs.SI", "physics.soc-ph"],
    "complex systems": ["nlin.CD", "physics.soc-ph"],
    "nonlinear dynamics": ["nlin.CD"],
    "pattern formation": ["nlin.PS"],
    "chaos": ["nlin.CD"],
}


# =============================================================================
# Core Functions
# =============================================================================

def get_cache() -> CategoryCache:
    """Get or initialize the category cache."""
    global _cache
    if _cache is None:
        _cache = CategoryCache()
        _load_cache_from_file()
    return _cache


def _load_cache_from_file() -> None:
    """
    Load categories into cache, prioritizing database over local file.
    
    Priority order:
    1. Database (Supabase/PostgreSQL) - for cross-instance sharing
    2. Local JSON file - for offline/development use
    3. Built-in fallback - comprehensive known categories
    """
    global _cache
    if _cache is None:
        _cache = CategoryCache()
    
    # 1. Try loading from database first (non-local persistence)
    db_cats = load_from_db()
    if db_cats:
        _cache.categories.update(db_cats)
        for cat in db_cats.values():
            if cat.group:
                if cat.group not in _cache.groups:
                    _cache.groups[cat.group] = []
                _cache.groups[cat.group].append(cat.code)
        logger.info(f"Loaded {len(db_cats)} categories from database")
        return
    
    # 2. Fall back to local JSON file
    if CATEGORIES_FILE.exists():
        try:
            with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for cat_data in data.get("categories", []):
                cat = ArxivCategory.from_dict(cat_data)
                _cache.categories[cat.code] = cat
                
                # Build group index
                if cat.group:
                    if cat.group not in _cache.groups:
                        _cache.groups[cat.group] = []
                    _cache.groups[cat.group].append(cat.code)
            
            # Parse last sync time
            metadata = data.get("metadata", {})
            last_sync_str = metadata.get("last_full_sync", "")
            if last_sync_str:
                try:
                    _cache.last_sync = datetime.fromisoformat(last_sync_str.replace("Z", "+00:00"))
                except:
                    pass
            
            logger.info(f"Loaded {len(_cache.categories)} categories from local file")
            
            # If no groups were loaded (old format), infer from known categories
            if not _cache.groups:
                known_cats = _get_known_categories()
                for code, known_cat in known_cats.items():
                    if code in _cache.categories and known_cat.group:
                        _cache.categories[code].group = known_cat.group
                        if known_cat.group not in _cache.groups:
                            _cache.groups[known_cat.group] = []
                        _cache.groups[known_cat.group].append(code)
            
            # Sync to database if available (populate DB from local file)
            if is_db_available() and _cache.categories:
                logger.info("Syncing local categories to database...")
                save_to_db(_cache.categories)
            
            return
            
        except Exception as e:
            logger.error(f"Error loading categories from file: {e}")
    
    # 3. Fall back to built-in known categories
    logger.info(f"Using built-in fallback categories")
    known_cats = _get_known_categories()
    _cache.categories.update(known_cats)
    for cat in known_cats.values():
        if cat.group:
            if cat.group not in _cache.groups:
                _cache.groups[cat.group] = []
            _cache.groups[cat.group].append(cat.code)


def save_cache_to_file() -> None:
    """Save current cache to local JSON file."""
    cache = get_cache()
    
    categories_list = [cat.to_dict() for cat in cache.categories.values()]
    
    # Sort by category code for consistency
    categories_list.sort(key=lambda x: x["category_code"])
    
    data = {
        "categories": categories_list,
        "metadata": {
            "total_categories": len(categories_list),
            "source_url": ARXIV_TAXONOMY_URL,
            "last_full_sync": datetime.utcnow().isoformat() + "Z",
            "notes": "Auto-fetched from arXiv taxonomy"
        }
    }
    
    # Ensure directory exists
    CATEGORIES_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(CATEGORIES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    
    logger.info(f"Saved {len(categories_list)} categories to {CATEGORIES_FILE}")


# =============================================================================
# Database Sync Functions
# =============================================================================

def is_db_available() -> bool:
    """Check if database is configured and available."""
    try:
        from ..db.database import is_database_configured
        return is_database_configured()
    except Exception:
        return False


def save_to_db(categories: Dict[str, ArxivCategory] = None) -> int:
    """
    Save categories to database (Supabase/PostgreSQL).
    
    Args:
        categories: Categories to save. If None, uses current cache.
        
    Returns:
        Number of categories saved/updated.
    """
    if not is_db_available():
        logger.debug("Database not available, skipping DB save")
        return 0
    
    cache = get_cache()
    cats_to_save = categories or cache.categories
    
    if not cats_to_save:
        return 0
    
    try:
        from ..db.database import get_db_session
        from ..db.orm_models import ArxivCategoryDB
        
        count = 0
        with get_db_session() as db:
            for code, cat in cats_to_save.items():
                # Upsert pattern - check if exists
                existing = db.query(ArxivCategoryDB).filter_by(code=code).first()
                if existing:
                    existing.name = cat.name
                    existing.group_name = cat.group
                    existing.description = cat.description
                    existing.source = cat.source or "arxiv"
                    existing.last_updated = datetime.utcnow()
                else:
                    new_cat = ArxivCategoryDB(
                        code=code,
                        name=cat.name,
                        group_name=cat.group,
                        description=cat.description,
                        source=cat.source or "arxiv",
                    )
                    db.add(new_cat)
                count += 1
        
        logger.info(f"Saved {count} categories to database")
        return count
        
    except Exception as e:
        logger.error(f"Error saving to database: {e}")
        return 0


def load_from_db() -> Dict[str, ArxivCategory]:
    """
    Load categories from database.
    
    Returns:
        Dictionary of category code -> ArxivCategory
    """
    if not is_db_available():
        return {}
    
    try:
        from ..db.database import get_db_session
        from ..db.orm_models import ArxivCategoryDB
        
        categories = {}
        with get_db_session() as db:
            db_cats = db.query(ArxivCategoryDB).all()
            for cat in db_cats:
                categories[cat.code] = ArxivCategory(
                    code=cat.code,
                    name=cat.name,
                    group=cat.group_name or "",
                    description=cat.description or "",
                    source=cat.source or "database",
                    last_updated=cat.last_updated.isoformat() + "Z" if cat.last_updated else ""
                )
        
        if categories:
            logger.info(f"Loaded {len(categories)} categories from database")
        return categories
        
    except Exception as e:
        logger.error(f"Error loading from database: {e}")
        return {}


def sync_to_db() -> int:
    """
    Sync current cache to database.
    
    This is called automatically after refresh_taxonomy() to persist
    categories to the database for cross-instance sharing.
    
    Returns:
        Number of categories synced.
    """
    return save_to_db()


async def fetch_arxiv_taxonomy() -> Dict[str, ArxivCategory]:
    """
    Fetch the complete arXiv taxonomy from the official website.
    
    Returns:
        Dictionary mapping category codes to ArxivCategory objects.
    """
    import aiohttp
    
    categories = {}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(ARXIV_TAXONOMY_URL, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch taxonomy: HTTP {response.status}")
                    return categories
                
                html = await response.text()
        
        # Parse the taxonomy HTML
        categories = _parse_taxonomy_html(html)
        logger.info(f"Fetched {len(categories)} categories from arXiv")
        
    except ImportError:
        logger.warning("aiohttp not installed, trying synchronous fetch")
        categories = _fetch_taxonomy_sync()
    except Exception as e:
        logger.error(f"Error fetching taxonomy: {e}")
        # Try synchronous fallback
        categories = _fetch_taxonomy_sync()
    
    return categories


def _fetch_taxonomy_sync() -> Dict[str, ArxivCategory]:
    """Synchronous fallback for fetching taxonomy."""
    import urllib.request
    
    categories = {}
    
    try:
        with urllib.request.urlopen(ARXIV_TAXONOMY_URL, timeout=30) as response:
            html = response.read().decode("utf-8")
        categories = _parse_taxonomy_html(html)
        logger.info(f"Fetched {len(categories)} categories (sync)")
    except Exception as e:
        logger.error(f"Sync fetch failed: {e}")
    
    return categories


def _parse_taxonomy_html(html: str) -> Dict[str, ArxivCategory]:
    """
    Parse arXiv taxonomy HTML page to extract categories.
    
    The taxonomy page has a specific structure with category codes and names.
    """
    categories = {}
    current_time = datetime.utcnow().isoformat() + "Z"
    
    # Pattern to match category entries
    # Looking for patterns like: <h4>cs.AI - Artificial Intelligence</h4>
    # or accordion entries with category codes
    
    # Try multiple patterns to handle different page formats
    patterns = [
        # Pattern 1: h4 with id attribute
        r'<h4[^>]*id="([^"]+)"[^>]*>\s*([^<]+)</h4>',
        # Pattern 2: accordion button format
        r'aria-controls="([^"]+)"[^>]*>\s*<span[^>]*>([^<]+)</span>',
        # Pattern 3: Simple category format
        r'<strong>([a-z-]+\.[A-Z][A-Za-z-]*)</strong>\s*[-–]\s*([^<\n]+)',
        # Pattern 4: Taxonomy list items
        r'>([a-z-]+\.[A-Z][A-Za-z-]*)\s*[-–—]\s*([^<]+)<',
    ]
    
    # Extract main groups
    group_pattern = r'<h2[^>]*>([^<]+)</h2>'
    current_group = "General"
    
    for match in re.finditer(group_pattern, html, re.IGNORECASE):
        group_name = match.group(1).strip()
        if group_name and not group_name.startswith("arXiv"):
            current_group = group_name
    
    # Try each pattern
    for pattern in patterns:
        for match in re.finditer(pattern, html):
            code = match.group(1).strip()
            name = match.group(2).strip()
            
            # Clean up the code
            code = code.replace("accordion-", "").strip()
            
            # Validate it looks like an arXiv category code
            if not re.match(r'^[a-z-]+\.[A-Z][A-Za-z-]*$', code):
                # Try to fix common issues
                if '.' in code and len(code) < 20:
                    code = code.lower().split('.')[0] + '.' + code.split('.')[-1]
                else:
                    continue
            
            # Clean up the name
            name = re.sub(r'\s+', ' ', name).strip()
            name = name.rstrip(' -–—')
            
            if code and name and code not in categories:
                categories[code] = ArxivCategory(
                    code=code,
                    name=name,
                    group=current_group,
                    source="arxiv",
                    last_updated=current_time
                )
    
    # If HTML parsing didn't work well, use known categories as fallback
    if len(categories) < 50:
        logger.warning("HTML parsing yielded few results, using fallback")
        categories = _get_known_categories()
    
    return categories


def _get_known_categories() -> Dict[str, ArxivCategory]:
    """
    Return a comprehensive list of known arXiv categories.
    Used as fallback when web fetching fails.
    """
    current_time = datetime.utcnow().isoformat() + "Z"
    
    # Complete list of arXiv categories as of 2024
    known = {
        # Computer Science (cs.*)
        "cs.AI": ("Artificial Intelligence", "Computer Science"),
        "cs.AR": ("Hardware Architecture", "Computer Science"),
        "cs.CC": ("Computational Complexity", "Computer Science"),
        "cs.CE": ("Computational Engineering, Finance, and Science", "Computer Science"),
        "cs.CG": ("Computational Geometry", "Computer Science"),
        "cs.CL": ("Computation and Language", "Computer Science"),
        "cs.CR": ("Cryptography and Security", "Computer Science"),
        "cs.CV": ("Computer Vision and Pattern Recognition", "Computer Science"),
        "cs.CY": ("Computers and Society", "Computer Science"),
        "cs.DB": ("Databases", "Computer Science"),
        "cs.DC": ("Distributed, Parallel, and Cluster Computing", "Computer Science"),
        "cs.DL": ("Digital Libraries", "Computer Science"),
        "cs.DM": ("Discrete Mathematics", "Computer Science"),
        "cs.DS": ("Data Structures and Algorithms", "Computer Science"),
        "cs.ET": ("Emerging Technologies", "Computer Science"),
        "cs.FL": ("Formal Languages and Automata Theory", "Computer Science"),
        "cs.GL": ("General Literature", "Computer Science"),
        "cs.GR": ("Graphics", "Computer Science"),
        "cs.GT": ("Computer Science and Game Theory", "Computer Science"),
        "cs.HC": ("Human-Computer Interaction", "Computer Science"),
        "cs.IR": ("Information Retrieval", "Computer Science"),
        "cs.IT": ("Information Theory", "Computer Science"),
        "cs.LG": ("Machine Learning", "Computer Science"),
        "cs.LO": ("Logic in Computer Science", "Computer Science"),
        "cs.MA": ("Multiagent Systems", "Computer Science"),
        "cs.MM": ("Multimedia", "Computer Science"),
        "cs.MS": ("Mathematical Software", "Computer Science"),
        "cs.NA": ("Numerical Analysis", "Computer Science"),
        "cs.NE": ("Neural and Evolutionary Computing", "Computer Science"),
        "cs.NI": ("Networking and Internet Architecture", "Computer Science"),
        "cs.OH": ("Other Computer Science", "Computer Science"),
        "cs.OS": ("Operating Systems", "Computer Science"),
        "cs.PF": ("Performance", "Computer Science"),
        "cs.PL": ("Programming Languages", "Computer Science"),
        "cs.RO": ("Robotics", "Computer Science"),
        "cs.SC": ("Symbolic Computation", "Computer Science"),
        "cs.SD": ("Sound", "Computer Science"),
        "cs.SE": ("Software Engineering", "Computer Science"),
        "cs.SI": ("Social and Information Networks", "Computer Science"),
        "cs.SY": ("Systems and Control", "Computer Science"),
        
        # Economics (econ.*)
        "econ.EM": ("Econometrics", "Economics"),
        "econ.GN": ("General Economics", "Economics"),
        "econ.TH": ("Theoretical Economics", "Economics"),
        
        # Electrical Engineering and Systems Science (eess.*)
        "eess.AS": ("Audio and Speech Processing", "Electrical Engineering and Systems Science"),
        "eess.IV": ("Image and Video Processing", "Electrical Engineering and Systems Science"),
        "eess.SP": ("Signal Processing", "Electrical Engineering and Systems Science"),
        "eess.SY": ("Systems and Control", "Electrical Engineering and Systems Science"),
        
        # Mathematics (math.*)
        "math.AC": ("Commutative Algebra", "Mathematics"),
        "math.AG": ("Algebraic Geometry", "Mathematics"),
        "math.AP": ("Analysis of PDEs", "Mathematics"),
        "math.AT": ("Algebraic Topology", "Mathematics"),
        "math.CA": ("Classical Analysis and ODEs", "Mathematics"),
        "math.CO": ("Combinatorics", "Mathematics"),
        "math.CT": ("Category Theory", "Mathematics"),
        "math.CV": ("Complex Variables", "Mathematics"),
        "math.DG": ("Differential Geometry", "Mathematics"),
        "math.DS": ("Dynamical Systems", "Mathematics"),
        "math.FA": ("Functional Analysis", "Mathematics"),
        "math.GM": ("General Mathematics", "Mathematics"),
        "math.GN": ("General Topology", "Mathematics"),
        "math.GR": ("Group Theory", "Mathematics"),
        "math.GT": ("Geometric Topology", "Mathematics"),
        "math.HO": ("History and Overview", "Mathematics"),
        "math.IT": ("Information Theory", "Mathematics"),
        "math.KT": ("K-Theory and Homology", "Mathematics"),
        "math.LO": ("Logic", "Mathematics"),
        "math.MG": ("Metric Geometry", "Mathematics"),
        "math.MP": ("Mathematical Physics", "Mathematics"),
        "math.NA": ("Numerical Analysis", "Mathematics"),
        "math.NT": ("Number Theory", "Mathematics"),
        "math.OA": ("Operator Algebras", "Mathematics"),
        "math.OC": ("Optimization and Control", "Mathematics"),
        "math.PR": ("Probability", "Mathematics"),
        "math.QA": ("Quantum Algebra", "Mathematics"),
        "math.RA": ("Rings and Algebras", "Mathematics"),
        "math.RT": ("Representation Theory", "Mathematics"),
        "math.SG": ("Symplectic Geometry", "Mathematics"),
        "math.SP": ("Spectral Theory", "Mathematics"),
        "math.ST": ("Statistics Theory", "Mathematics"),
        
        # Physics - Astrophysics (astro-ph.*)
        "astro-ph.CO": ("Cosmology and Nongalactic Astrophysics", "Physics"),
        "astro-ph.EP": ("Earth and Planetary Astrophysics", "Physics"),
        "astro-ph.GA": ("Astrophysics of Galaxies", "Physics"),
        "astro-ph.HE": ("High Energy Astrophysical Phenomena", "Physics"),
        "astro-ph.IM": ("Instrumentation and Methods for Astrophysics", "Physics"),
        "astro-ph.SR": ("Solar and Stellar Astrophysics", "Physics"),
        
        # Physics - Condensed Matter (cond-mat.*)
        "cond-mat.dis-nn": ("Disordered Systems and Neural Networks", "Physics"),
        "cond-mat.mes-hall": ("Mesoscale and Nanoscale Physics", "Physics"),
        "cond-mat.mtrl-sci": ("Materials Science", "Physics"),
        "cond-mat.other": ("Other Condensed Matter", "Physics"),
        "cond-mat.quant-gas": ("Quantum Gases", "Physics"),
        "cond-mat.soft": ("Soft Condensed Matter", "Physics"),
        "cond-mat.stat-mech": ("Statistical Mechanics", "Physics"),
        "cond-mat.str-el": ("Strongly Correlated Electrons", "Physics"),
        "cond-mat.supr-con": ("Superconductivity", "Physics"),
        
        # Physics - General Relativity (gr-qc)
        "gr-qc": ("General Relativity and Quantum Cosmology", "Physics"),
        
        # Physics - High Energy Physics (hep-*)
        "hep-ex": ("High Energy Physics - Experiment", "Physics"),
        "hep-lat": ("High Energy Physics - Lattice", "Physics"),
        "hep-ph": ("High Energy Physics - Phenomenology", "Physics"),
        "hep-th": ("High Energy Physics - Theory", "Physics"),
        
        # Physics - Mathematical Physics (math-ph)
        "math-ph": ("Mathematical Physics", "Physics"),
        
        # Physics - Nonlinear Sciences (nlin.*)
        "nlin.AO": ("Adaptation and Self-Organizing Systems", "Physics"),
        "nlin.CD": ("Chaotic Dynamics", "Physics"),
        "nlin.CG": ("Cellular Automata and Lattice Gases", "Physics"),
        "nlin.PS": ("Pattern Formation and Solitons", "Physics"),
        "nlin.SI": ("Exactly Solvable and Integrable Systems", "Physics"),
        
        # Physics - Nuclear (nucl-*)
        "nucl-ex": ("Nuclear Experiment", "Physics"),
        "nucl-th": ("Nuclear Theory", "Physics"),
        
        # Physics - General (physics.*)
        "physics.acc-ph": ("Accelerator Physics", "Physics"),
        "physics.ao-ph": ("Atmospheric and Oceanic Physics", "Physics"),
        "physics.app-ph": ("Applied Physics", "Physics"),
        "physics.atm-clus": ("Atomic and Molecular Clusters", "Physics"),
        "physics.atom-ph": ("Atomic Physics", "Physics"),
        "physics.bio-ph": ("Biological Physics", "Physics"),
        "physics.chem-ph": ("Chemical Physics", "Physics"),
        "physics.class-ph": ("Classical Physics", "Physics"),
        "physics.comp-ph": ("Computational Physics", "Physics"),
        "physics.data-an": ("Data Analysis, Statistics and Probability", "Physics"),
        "physics.ed-ph": ("Physics Education", "Physics"),
        "physics.flu-dyn": ("Fluid Dynamics", "Physics"),
        "physics.gen-ph": ("General Physics", "Physics"),
        "physics.geo-ph": ("Geophysics", "Physics"),
        "physics.hist-ph": ("History and Philosophy of Physics", "Physics"),
        "physics.ins-det": ("Instrumentation and Detectors", "Physics"),
        "physics.med-ph": ("Medical Physics", "Physics"),
        "physics.optics": ("Optics", "Physics"),
        "physics.plasm-ph": ("Plasma Physics", "Physics"),
        "physics.pop-ph": ("Popular Physics", "Physics"),
        "physics.soc-ph": ("Physics and Society", "Physics"),
        "physics.space-ph": ("Space Physics", "Physics"),
        
        # Quantitative Biology (q-bio.*)
        "q-bio.BM": ("Biomolecules", "Quantitative Biology"),
        "q-bio.CB": ("Cell Behavior", "Quantitative Biology"),
        "q-bio.GN": ("Genomics", "Quantitative Biology"),
        "q-bio.MN": ("Molecular Networks", "Quantitative Biology"),
        "q-bio.NC": ("Neurons and Cognition", "Quantitative Biology"),
        "q-bio.OT": ("Other Quantitative Biology", "Quantitative Biology"),
        "q-bio.PE": ("Populations and Evolution", "Quantitative Biology"),
        "q-bio.QM": ("Quantitative Methods", "Quantitative Biology"),
        "q-bio.SC": ("Subcellular Processes", "Quantitative Biology"),
        "q-bio.TO": ("Tissues and Organs", "Quantitative Biology"),
        
        # Quantitative Finance (q-fin.*)
        "q-fin.CP": ("Computational Finance", "Quantitative Finance"),
        "q-fin.EC": ("Economics", "Quantitative Finance"),
        "q-fin.GN": ("General Finance", "Quantitative Finance"),
        "q-fin.MF": ("Mathematical Finance", "Quantitative Finance"),
        "q-fin.PM": ("Portfolio Management", "Quantitative Finance"),
        "q-fin.PR": ("Pricing of Securities", "Quantitative Finance"),
        "q-fin.RM": ("Risk Management", "Quantitative Finance"),
        "q-fin.ST": ("Statistical Finance", "Quantitative Finance"),
        "q-fin.TR": ("Trading and Market Microstructure", "Quantitative Finance"),
        
        # Quantum Physics (quant-ph)
        "quant-ph": ("Quantum Physics", "Physics"),
        
        # Statistics (stat.*)
        "stat.AP": ("Applications", "Statistics"),
        "stat.CO": ("Computation", "Statistics"),
        "stat.ME": ("Methodology", "Statistics"),
        "stat.ML": ("Machine Learning", "Statistics"),
        "stat.OT": ("Other Statistics", "Statistics"),
        "stat.TH": ("Statistics Theory", "Statistics"),
    }
    
    categories = {}
    for code, (name, group) in known.items():
        categories[code] = ArxivCategory(
            code=code,
            name=name,
            group=group,
            source="fallback",
            last_updated=current_time
        )
    
    return categories


# =============================================================================
# Public API
# =============================================================================

async def refresh_taxonomy(force: bool = False) -> int:
    """
    Refresh the arXiv taxonomy from the official source.
    
    Args:
        force: If True, refresh even if cache is recent
        
    Returns:
        Number of categories fetched
    """
    cache = get_cache()
    
    # Check if refresh is needed
    if not force and cache.last_sync:
        age = datetime.utcnow() - cache.last_sync.replace(tzinfo=None)
        if age < timedelta(days=CACHE_EXPIRY_DAYS):
            logger.info(f"Cache is fresh (age: {age.days} days), skipping refresh")
            return len(cache.categories)
    
    # Fetch new taxonomy
    new_categories = await fetch_arxiv_taxonomy()
    
    if new_categories:
        # Update cache
        cache.categories.update(new_categories)
        cache.last_sync = datetime.utcnow()
        
        # Rebuild group index
        cache.groups.clear()
        for cat in cache.categories.values():
            if cat.group:
                if cat.group not in cache.groups:
                    cache.groups[cat.group] = []
                cache.groups[cat.group].append(cat.code)
        
        # Save to local file
        save_cache_to_file()
        
        # Sync to database for cross-instance persistence
        db_count = save_to_db()
        if db_count > 0:
            logger.info(f"Synced {db_count} categories to database")
    else:
        # Web fetch failed, use known categories as fallback
        logger.info("Web fetch failed, using known categories fallback")
        known_cats = _get_known_categories()
        cache.categories.update(known_cats)
        cache.last_sync = datetime.utcnow()
        
        # Rebuild group index
        cache.groups.clear()
        for cat in cache.categories.values():
            if cat.group:
                if cat.group not in cache.groups:
                    cache.groups[cat.group] = []
                cache.groups[cat.group].append(cat.code)
        
        # Save to local file
        save_cache_to_file()
        
        # Sync to database for cross-instance persistence
        db_count = save_to_db()
        if db_count > 0:
            logger.info(f"Synced {db_count} fallback categories to database")
    
    return len(cache.categories)


def refresh_taxonomy_sync(force: bool = False) -> int:
    """Synchronous version of refresh_taxonomy."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(refresh_taxonomy(force))


def get_category(code: str) -> Optional[ArxivCategory]:
    """
    Get a category by its code.
    
    If the category is not in cache, attempts to fetch it.
    
    Args:
        code: Category code (e.g., "cs.AI")
        
    Returns:
        ArxivCategory if found, None otherwise
    """
    cache = get_cache()
    
    if code in cache.categories:
        return cache.categories[code]
    
    # Try to fetch unknown category
    logger.info(f"Category {code} not in cache, attempting refresh")
    refresh_taxonomy_sync()
    
    return cache.categories.get(code)


def get_all_categories() -> Dict[str, ArxivCategory]:
    """Get all cached categories."""
    cache = get_cache()
    
    # Ensure we have categories
    if not cache.categories:
        refresh_taxonomy_sync()
    
    return cache.categories


def get_categories_by_group(group: str) -> List[ArxivCategory]:
    """Get all categories in a specific group (e.g., 'Computer Science')."""
    cache = get_cache()
    
    if not cache.categories:
        refresh_taxonomy_sync()
    
    codes = cache.groups.get(group, [])
    return [cache.categories[code] for code in codes if code in cache.categories]


def topic_to_categories(topic: str) -> List[str]:
    """
    Convert a research topic to relevant arXiv category codes.
    
    Uses intelligent matching and fetches new categories if needed.
    
    Args:
        topic: Research topic (e.g., "machine learning", "quantum physics")
        
    Returns:
        List of relevant arXiv category codes
    """
    cache = get_cache()
    topic_lower = topic.lower().strip()
    
    # Check base mappings first
    if topic_lower in BASE_TOPIC_MAPPINGS:
        return BASE_TOPIC_MAPPINGS[topic_lower]
    
    # Ensure we have categories
    if not cache.categories:
        refresh_taxonomy_sync()
    
    # Try fuzzy matching against category names
    matches = []
    topic_words = set(topic_lower.split())
    
    for code, cat in cache.categories.items():
        name_lower = cat.name.lower()
        
        # Exact substring match
        if topic_lower in name_lower or name_lower in topic_lower:
            matches.append((code, 2))  # High priority
            continue
        
        # Word overlap
        name_words = set(name_lower.split())
        overlap = topic_words & name_words
        if overlap:
            matches.append((code, len(overlap)))
    
    # Sort by match quality and return top matches
    matches.sort(key=lambda x: x[1], reverse=True)
    return [code for code, _ in matches[:5]]


def validate_category(code: str) -> Tuple[bool, Optional[str]]:
    """
    Validate if a category code exists.
    
    Args:
        code: Category code to validate
        
    Returns:
        Tuple of (is_valid, category_name or error message)
    """
    cat = get_category(code)
    if cat:
        return True, cat.name
    return False, f"Unknown category: {code}"


def search_categories(query: str, limit: int = 10) -> List[ArxivCategory]:
    """
    Search categories by name or description.
    
    Args:
        query: Search query
        limit: Maximum results to return
        
    Returns:
        List of matching ArxivCategory objects
    """
    cache = get_cache()
    
    if not cache.categories:
        refresh_taxonomy_sync()
    
    query_lower = query.lower()
    results = []
    
    for cat in cache.categories.values():
        score = 0
        
        # Check code
        if query_lower in cat.code.lower():
            score += 3
        
        # Check name
        if query_lower in cat.name.lower():
            score += 2
        
        # Check group
        if cat.group and query_lower in cat.group.lower():
            score += 1
        
        if score > 0:
            results.append((cat, score))
    
    # Sort by score and return top results
    results.sort(key=lambda x: x[1], reverse=True)
    return [cat for cat, _ in results[:limit]]


def get_category_groups() -> List[str]:
    """Get list of all category groups."""
    cache = get_cache()
    
    if not cache.categories:
        refresh_taxonomy_sync()
    
    return list(cache.groups.keys())


# =============================================================================
# Derive arXiv Categories from Research Interests
# =============================================================================

def derive_arxiv_categories_from_interests(interests_text: str) -> Dict[str, List[str]]:
    """
    Automatically derive arXiv categories from free-text research interests.
    
    This function maps natural language research interests to appropriate
    arXiv categories for filtering papers.
    
    Args:
        interests_text: Free-text description of research interests
                       e.g., "I'm interested in machine learning, NLP, and transformers"
    
    Returns:
        Dict with 'primary' and 'secondary' category lists:
        {
            "primary": ["cs.LG", "cs.CL"],     # High-confidence matches
            "secondary": ["stat.ML", "cs.AI"]  # Lower-confidence matches
        }
    
    Example:
        >>> derive_arxiv_categories_from_interests("deep learning and NLP for healthcare")
        {"primary": ["cs.LG", "cs.CL"], "secondary": ["cs.AI", "q-bio.QM"]}
    """
    if not interests_text or not interests_text.strip():
        logger.debug("[ARXIV_CATS] Empty interests text, returning empty categories")
        return {"primary": [], "secondary": []}
    
    interests_lower = interests_text.lower()
    primary_codes: Set[str] = set()
    secondary_codes: Set[str] = set()
    
    # Track matched keywords for logging
    matched_keywords = []
    
    # Phase 1: Direct keyword matching from BASE_TOPIC_MAPPINGS (high precision)
    for topic, codes in BASE_TOPIC_MAPPINGS.items():
        # Match full topic phrase
        if topic in interests_lower:
            primary_codes.update(codes)
            matched_keywords.append(topic)
            continue
        
        # Also match individual words from multi-word topics
        topic_words = topic.split()
        if len(topic_words) > 1:
            # If all words of a multi-word topic appear separately, it's a match
            if all(word in interests_lower for word in topic_words):
                primary_codes.update(codes)
                matched_keywords.append(topic)
    
    # Phase 2: Tokenize interests and match individual significant words
    # Extract words, filtering out common stopwords
    stopwords = {
        'i', 'me', 'my', 'we', 'our', 'you', 'your', 'the', 'a', 'an', 'and', 'or',
        'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'am',
        'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must',
        'interested', 'interest', 'research', 'work', 'working', 'field', 'fields',
        'area', 'areas', 'topic', 'topics', 'especially', 'particularly', 'like',
        'really', 'very', 'also', 'including', 'such', 'as', 'well', 'about', 'into',
    }
    
    # Extract meaningful words (letters only, length > 2)
    words = [w.strip('.,;:!?()[]{}"\'-') for w in interests_lower.split()]
    meaningful_words = [w for w in words if len(w) > 2 and w not in stopwords and w.isalpha()]
    
    # Check each word against topic mappings
    for word in meaningful_words:
        if word in BASE_TOPIC_MAPPINGS:
            codes = BASE_TOPIC_MAPPINGS[word]
            # Single-word matches go to secondary (lower confidence)
            secondary_codes.update(codes)
            if word not in matched_keywords:
                matched_keywords.append(word)
    
    # Phase 3: Semantic hints - detect domain indicators
    domain_hints = {
        # Healthcare / Biology indicators
        ('healthcare', 'medical', 'health', 'clinical', 'patient', 'disease', 'drug'):
            ['q-bio.QM', 'cs.CL'],  # Computational methods, NLP for medical
        # Finance indicators
        ('finance', 'financial', 'trading', 'stock', 'market', 'investment', 'portfolio'):
            ['q-fin.CP', 'q-fin.ST', 'stat.AP'],
        # Physics indicators
        ('physics', 'quantum', 'particles', 'energy', 'photon', 'laser'):
            ['quant-ph', 'physics.gen-ph'],
        # Robotics / Automation
        ('robot', 'robotics', 'autonomous', 'drone', 'vehicle', 'navigation'):
            ['cs.RO', 'cs.SY'],
        # Security
        ('security', 'privacy', 'encryption', 'attack', 'malware', 'intrusion'):
            ['cs.CR'],
        # Social / Network
        ('social', 'twitter', 'facebook', 'network', 'community', 'viral', 'misinformation'):
            ['cs.SI', 'cs.CY'],
    }
    
    for hint_words, hint_codes in domain_hints.items():
        if any(hw in interests_lower for hw in hint_words):
            secondary_codes.update(hint_codes)
    
    # Remove overlaps: if a code is in primary, don't also list in secondary
    secondary_codes = secondary_codes - primary_codes
    
    # Sort for consistency
    result = {
        "primary": sorted(list(primary_codes)),
        "secondary": sorted(list(secondary_codes)),
    }
    
    logger.info(f"[ARXIV_CATS] Derived categories from interests: "
                f"matched_keywords={matched_keywords}, "
                f"primary={result['primary']}, secondary={result['secondary']}")
    
    return result


def derive_categories_for_colleague(interests: str, existing_categories: List[str] = None) -> Dict[str, Any]:
    """
    Derive and merge arXiv categories for a colleague.
    
    This is a convenience wrapper that:
    1. Derives categories from interests
    2. Optionally merges with existing manually-set categories
    3. Returns a structure ready for DB storage
    
    Args:
        interests: Free-text research interests
        existing_categories: List of manually-set category codes (optional)
    
    Returns:
        Dict with derived categories and metadata:
        {
            "primary": [...],
            "secondary": [...],
            "manual": [...],          # Categories set manually by owner
            "all_categories": [...],  # Combined unique list
            "derived_from": "...",    # Source interests text (truncated)
        }
    """
    derived = derive_arxiv_categories_from_interests(interests)
    
    manual_cats = existing_categories or []
    
    # Combine all categories (unique)
    all_cats = set(derived["primary"]) | set(derived["secondary"]) | set(manual_cats)
    
    return {
        "primary": derived["primary"],
        "secondary": derived["secondary"],
        "manual": manual_cats,
        "all_categories": sorted(list(all_cats)),
        "derived_from": interests[:200] if interests else "",
    }


# =============================================================================
# JSON API (for tools)
# =============================================================================

def get_all_categories_json() -> Dict[str, Any]:
    """Get all categories as JSON-serializable dict."""
    categories = get_all_categories()
    return {
        "categories": [
            {"code": cat.code, "name": cat.name, "group": cat.group}
            for cat in categories.values()
        ],
        "total": len(categories),
        "groups": list(get_category_groups())
    }


def topic_to_categories_json(topic: str) -> Dict[str, Any]:
    """Get categories for a topic as JSON."""
    codes = topic_to_categories(topic)
    categories = get_all_categories()
    
    return {
        "topic": topic,
        "categories": [
            {"code": code, "name": categories[code].name if code in categories else "Unknown"}
            for code in codes
        ]
    }


# =============================================================================
# Self-Check
# =============================================================================

def self_check() -> bool:
    """Run self-check tests."""
    print("=" * 60)
    print("arxiv_categories Self-Check")
    print("=" * 60)
    
    all_passed = True

    def check(name: str, condition: bool) -> bool:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        return condition

    # Test 1: Cache initialization
    print("\n1. Cache Initialization:")
    try:
        cache = get_cache()
        all_passed &= check("cache initialized", cache is not None)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 2: Load from file or fallback
    print("\n2. Load Categories:")
    try:
        categories = get_all_categories()
        all_passed &= check("has categories", len(categories) > 0)
        all_passed &= check("has cs.AI", "cs.AI" in categories)
        all_passed &= check("has cs.LG", "cs.LG" in categories)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 3: Get single category
    print("\n3. Get Category:")
    try:
        cat = get_category("cs.AI")
        all_passed &= check("cs.AI exists", cat is not None)
        all_passed &= check("has correct name", "Artificial Intelligence" in cat.name if cat else False)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 4: Topic to categories
    print("\n4. Topic to Categories:")
    try:
        codes = topic_to_categories("machine learning")
        all_passed &= check("returns codes", len(codes) > 0)
        all_passed &= check("includes cs.LG", "cs.LG" in codes or "stat.ML" in codes)
        
        codes = topic_to_categories("quantum physics")
        all_passed &= check("quantum returns codes", len(codes) > 0)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 5: Search categories
    print("\n5. Search Categories:")
    try:
        results = search_categories("machine")
        all_passed &= check("search returns results", len(results) > 0)
        all_passed &= check("results have code", all(r.code for r in results))
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 6: Validate category
    print("\n6. Validate Category:")
    try:
        valid, msg = validate_category("cs.AI")
        all_passed &= check("cs.AI is valid", valid)
        
        valid, msg = validate_category("xx.FAKE")
        all_passed &= check("xx.FAKE is invalid", not valid)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 7: JSON output
    print("\n7. JSON Output:")
    try:
        result = get_all_categories_json()
        all_passed &= check("has categories key", "categories" in result)
        all_passed &= check("has total key", "total" in result)
        all_passed &= check("total > 0", result["total"] > 0)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 8: Category groups
    print("\n8. Category Groups:")
    try:
        groups = get_category_groups()
        all_passed &= check("returns groups", len(groups) > 0)
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
    
    # Run with --refresh to force refresh
    if len(sys.argv) > 1 and sys.argv[1] == "--refresh":
        print("Forcing taxonomy refresh...")
        count = refresh_taxonomy_sync(force=True)
        print(f"Refreshed {count} categories")
    else:
        success = self_check()
        sys.exit(0 if success else 1)
