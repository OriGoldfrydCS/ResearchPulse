"""
Data Models - Pydantic models for type validation of JSON database records.

These models provide schema validation and documentation for the demo JSON DBs.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Research Profile Models
# =============================================================================

class MyPaper(BaseModel):
    """A paper authored by the researcher."""
    title: str = Field(..., description="Paper title")
    abstract: str = Field(..., description="Paper abstract")
    year: int = Field(..., description="Publication year")
    arxiv_id: Optional[str] = Field(None, description="arXiv paper ID")
    link: Optional[str] = Field(None, description="URL to the paper")


class StopPolicyConfig(BaseModel):
    """Stop policy configuration for bounded agent execution."""
    max_runtime_minutes: int = Field(6, description="Maximum runtime in minutes")
    max_papers_checked: int = Field(30, description="Maximum papers to evaluate")
    stop_if_no_new_papers: bool = Field(True, description="Stop if no unseen papers")
    max_rag_queries: int = Field(50, description="Maximum RAG retrieval queries")
    min_importance_to_act: Literal["high", "medium", "low"] = Field(
        "medium", description="Minimum importance to take action"
    )


class ResearchProfile(BaseModel):
    """Complete researcher profile configuration."""
    researcher_name: str = Field(..., description="Researcher's full name")
    affiliation: str = Field(..., description="Institution/organization")
    research_topics: List[str] = Field(default_factory=list, description="Research interests")
    my_papers: List[MyPaper] = Field(default_factory=list, description="Researcher's own papers")
    preferred_venues: List[str] = Field(default_factory=list, description="Preferred conferences/journals")
    avoid_topics: List[str] = Field(default_factory=list, description="Topics to exclude")
    time_budget_per_week_minutes: int = Field(120, description="Reading time budget")
    arxiv_categories_include: List[str] = Field(default_factory=list, description="arXiv categories to include")
    arxiv_categories_exclude: List[str] = Field(default_factory=list, description="arXiv categories to exclude")
    stop_policy: StopPolicyConfig = Field(default_factory=StopPolicyConfig, description="Agent stop policy")


# =============================================================================
# Papers State Models
# =============================================================================

class PaperRecord(BaseModel):
    """A record of a paper that has been processed by the agent."""
    paper_id: str = Field(..., description="arXiv paper ID (e.g., '2501.00123')")
    title: str = Field(..., description="Paper title")
    date_seen: str = Field(..., description="ISO timestamp when paper was first seen")
    decision: Literal["ignored", "saved", "shared", "logged"] = Field(
        "logged", description="Decision made for this paper"
    )
    importance: Literal["high", "medium", "low"] = Field(
        "low", description="Assessed importance level"
    )
    notes: str = Field("", description="Agent notes about the paper")
    embedded_in_pinecone: bool = Field(False, description="Whether embedded in vector store")


class PapersStateStats(BaseModel):
    """Statistics about paper decisions."""
    saved: int = Field(0, description="Papers saved for reading")
    shared: int = Field(0, description="Papers shared with colleagues")
    ignored: int = Field(0, description="Papers ignored as irrelevant")
    logged: int = Field(0, description="Papers logged without action")


class PapersState(BaseModel):
    """Complete papers state database."""
    papers: List[PaperRecord] = Field(default_factory=list, description="All paper records")
    last_updated: str = Field(..., description="ISO timestamp of last update")
    total_papers_seen: int = Field(0, description="Total papers processed")
    stats: PapersStateStats = Field(default_factory=PapersStateStats, description="Decision statistics")


# =============================================================================
# Colleagues Models
# =============================================================================

class Colleague(BaseModel):
    """A colleague for paper sharing."""
    id: str = Field(..., description="Unique colleague identifier")
    name: str = Field(..., description="Colleague's full name")
    email: str = Field(..., description="Email address")
    affiliation: Optional[str] = Field(None, description="Institution")
    topics: List[str] = Field(default_factory=list, description="Research interests")
    sharing_preference: Literal["immediate", "daily_digest", "weekly_digest", "on_request"] = Field(
        "daily_digest", description="Preferred sharing frequency"
    )
    arxiv_categories_interest: List[str] = Field(default_factory=list, description="Interested arXiv categories")
    notes: Optional[str] = Field(None, description="Additional notes")


class ColleaguesDB(BaseModel):
    """Colleagues database."""
    colleagues: List[Colleague] = Field(default_factory=list, description="All colleagues")
    sharing_preferences_legend: dict = Field(default_factory=dict, description="Preference descriptions")


# =============================================================================
# Delivery Policy Models
# =============================================================================

class ImportancePolicy(BaseModel):
    """Policy for a specific importance level."""
    notify_researcher: bool = Field(True, description="Whether to notify researcher")
    send_email: bool = Field(False, description="Send email summary")
    create_calendar_entry: bool = Field(False, description="Create calendar event")
    add_to_reading_list: bool = Field(True, description="Add to reading list")
    allow_colleague_sharing: bool = Field(True, description="Allow sharing with colleagues")
    priority_label: str = Field("normal", description="Priority label")


class EmailSettings(BaseModel):
    """Email delivery settings."""
    enabled: bool = Field(True, description="Whether email is enabled")
    simulate_output: bool = Field(True, description="Simulate by writing files")
    output_directory: str = Field("outputs/emails", description="Output directory")
    filename_template: str = Field("email_{timestamp}.txt", description="Filename template")
    include_link: bool = Field(True, description="Include paper link")
    include_abstract: bool = Field(True, description="Include paper abstract")
    include_relevance_explanation: bool = Field(True, description="Include relevance explanation")
    max_papers_per_email: int = Field(10, description="Max papers per email")
    subject_template: str = Field("[ResearchPulse] {priority}: {count} new papers found")
    from_address: str = Field("researchpulse@localhost", description="Sender address")


class CalendarSettings(BaseModel):
    """Calendar delivery settings."""
    enabled: bool = Field(True, description="Whether calendar is enabled")
    simulate_output: bool = Field(True, description="Simulate by writing .ics files")
    output_directory: str = Field("outputs/calendar", description="Output directory")
    filename_template: str = Field("event_{timestamp}.ics", description="Filename template")
    event_duration_minutes: int = Field(30, description="Event duration")
    default_reminder_minutes: int = Field(60, description="Reminder time before event")
    event_title_template: str = Field("Read: {paper_title}", description="Event title template")
    schedule_within_days: int = Field(7, description="Schedule within N days")


class ReadingListSettings(BaseModel):
    """Reading list settings."""
    enabled: bool = Field(True, description="Whether reading list is enabled")
    output_file: str = Field("outputs/reading_list.txt", description="Output file path")
    format: str = Field("markdown", description="Output format")
    include_link: bool = Field(True, description="Include paper link")
    include_date_added: bool = Field(True, description="Include date added")
    include_importance: bool = Field(True, description="Include importance level")
    max_entries: int = Field(100, description="Maximum entries")


class ColleagueSharingSettings(BaseModel):
    """Colleague sharing settings."""
    enabled: bool = Field(True, description="Whether sharing is enabled")
    simulate_output: bool = Field(True, description="Simulate by writing files")
    output_directory: str = Field("outputs/shares", description="Output directory")
    filename_template: str = Field("share_{colleague}_{timestamp}.txt", description="Filename template")
    include_personal_note: bool = Field(True, description="Include personal note")
    include_relevance_explanation: bool = Field(True, description="Include relevance explanation")
    respect_sharing_preferences: bool = Field(True, description="Respect colleague preferences")
    max_papers_per_share: int = Field(5, description="Max papers per share")


class GlobalSettings(BaseModel):
    """Global delivery settings."""
    quiet_hours_start: str = Field("22:00", description="Quiet hours start time")
    quiet_hours_end: str = Field("08:00", description="Quiet hours end time")
    timezone: str = Field("America/Los_Angeles", description="Timezone")
    batch_notifications: bool = Field(False, description="Batch notifications")
    log_all_decisions: bool = Field(True, description="Log all decisions")


class DeliveryPolicy(BaseModel):
    """Complete delivery policy configuration."""
    importance_policies: dict[str, ImportancePolicy] = Field(
        default_factory=dict, description="Policies per importance level"
    )
    email_settings: EmailSettings = Field(default_factory=EmailSettings)
    calendar_settings: CalendarSettings = Field(default_factory=CalendarSettings)
    reading_list_settings: ReadingListSettings = Field(default_factory=ReadingListSettings)
    colleague_sharing_settings: ColleagueSharingSettings = Field(default_factory=ColleagueSharingSettings)
    global_settings: GlobalSettings = Field(default_factory=GlobalSettings)


# =============================================================================
# arXiv Categories Model
# =============================================================================

class ArxivCategory(BaseModel):
    """An arXiv category entry."""
    category_code: str = Field(..., description="Category code (e.g., 'cs.CL')")
    category_name: str = Field(..., description="Human-readable name")
    source: str = Field("arxiv", description="Source")
    last_updated: str = Field(..., description="ISO timestamp")


class ArxivCategoriesDB(BaseModel):
    """arXiv categories database."""
    categories: List[ArxivCategory] = Field(default_factory=list, description="All categories")
    metadata: dict = Field(default_factory=dict, description="Database metadata")
