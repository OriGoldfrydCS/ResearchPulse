"""
Database module - Unified data access layer.

Uses database (Supabase/PostgreSQL) as primary source, with local JSON fallback.
All data access should go through data_service for production use.
"""

# Primary data access layer - use these for all data operations
from .data_service import (
    # Database status
    is_db_available,
    # Research profile
    get_research_profile,
    update_research_profile,
    get_research_topics,
    get_arxiv_categories,
    get_stop_policy,
    # Colleagues
    get_colleagues,
    get_colleague_by_id,
    get_colleagues_for_topics,
    save_colleagues,
    # Delivery policy
    get_delivery_policy,
    get_policy_for_importance,
    save_delivery_policy,
    # Papers
    get_papers_state,
    get_seen_paper_ids,
    get_paper_by_id,
    upsert_paper,
    upsert_papers,
    # arXiv categories
    get_arxiv_categories_db,
    get_category_name,
    # Migration
    migrate_all_to_db,
    delete_local_data_files,
)

# Low-level JSON utilities (for development/fallback only)
from .json_store import (
    DEFAULT_DATA_DIR,
    JSONParseError,
    JSONStoreError,
    WriteError,
    load_json,
    save_json,
    validate_categories,
)

# Backward compatibility alias
get_stop_policy_from_profile = get_stop_policy

# Pydantic models for validation
from .models import (
    ArxivCategoriesDB,
    ArxivCategory,
    CalendarSettings,
    Colleague,
    ColleaguesDB,
    ColleagueSharingSettings,
    DeliveryPolicy as DeliveryPolicyModel,
    EmailSettings,
    GlobalSettings,
    ImportancePolicy,
    MyPaper,
    PaperRecord,
    PapersState,
    PapersStateStats,
    ReadingListSettings,
    ResearchProfile,
    StopPolicyConfig,
)

# ORM models for database operations
from .orm_models import (
    ArxivCategoryDB,
    User,
    Paper,
    PaperView,
    Colleague as ColleagueDB,
    Run,
    DeliveryPolicy as DeliveryPolicyDB,
)

__all__ = [
    # JSON Store
    "load_json",
    "save_json",
    "DEFAULT_DATA_DIR",
    "JSONStoreError",
    "JSONParseError",
    "WriteError",
    # Research Profile
    "get_research_profile",
    "get_research_topics",
    "get_arxiv_categories",
    "get_stop_policy_from_profile",
    # Colleagues
    "get_colleagues",
    "get_colleague_by_id",
    "get_colleagues_for_topics",
    # Delivery Policy
    "get_delivery_policy",
    "get_policy_for_importance",
    # Papers State
    "get_papers_state",
    "get_seen_paper_ids",
    "get_paper_by_id",
    "upsert_paper",
    "upsert_papers",
    # arXiv Categories
    "get_arxiv_categories_db",
    "get_category_name",
    "validate_categories",
    # Models
    "ResearchProfile",
    "MyPaper",
    "StopPolicyConfig",
    "PaperRecord",
    "PapersState",
    "PapersStateStats",
    "Colleague",
    "ColleaguesDB",
    "DeliveryPolicy",
    "ImportancePolicy",
    "EmailSettings",
    "CalendarSettings",
    "ReadingListSettings",
    "ColleagueSharingSettings",
    "GlobalSettings",
    "ArxivCategory",
    "ArxivCategoriesDB",
    # ORM Models (for database persistence)
    "ArxivCategoryDB",
]
