-- =============================================================================
-- ResearchPulse Supabase Schema
-- =============================================================================
-- Run this SQL in your Supabase SQL Editor to create the required tables
-- Go to: Supabase Dashboard → SQL Editor → New Query

-- =============================================================================
-- 1. Papers Table
-- Stores all processed paper records
-- =============================================================================
CREATE TABLE IF NOT EXISTS papers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT[],
    categories TEXT[],
    arxiv_url TEXT,
    pdf_url TEXT,
    published_date TIMESTAMPTZ,
    date_seen TIMESTAMPTZ DEFAULT NOW(),
    decision TEXT DEFAULT 'logged' CHECK (decision IN ('saved', 'shared', 'ignored', 'logged')),
    importance TEXT DEFAULT 'low' CHECK (importance IN ('high', 'medium', 'low')),
    relevance_score FLOAT,
    novelty_score FLOAT,
    heuristic_score FLOAT,
    embedded_in_pinecone BOOLEAN DEFAULT FALSE,
    notes TEXT,
    actions_taken JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_papers_paper_id ON papers(paper_id);
CREATE INDEX IF NOT EXISTS idx_papers_decision ON papers(decision);
CREATE INDEX IF NOT EXISTS idx_papers_importance ON papers(importance);
CREATE INDEX IF NOT EXISTS idx_papers_date_seen ON papers(date_seen DESC);

-- =============================================================================
-- 2. Colleagues Table  
-- Stores colleague information for paper sharing
-- =============================================================================
CREATE TABLE IF NOT EXISTS colleagues (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    affiliation TEXT,
    topics TEXT[],
    sharing_preference TEXT DEFAULT 'weekly' CHECK (sharing_preference IN ('immediate', 'daily', 'weekly', 'monthly', 'never')),
    arxiv_categories_interest TEXT[],
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for topic matching
CREATE INDEX IF NOT EXISTS idx_colleagues_topics ON colleagues USING GIN(topics);

-- =============================================================================
-- 3. Research Profile Table
-- Stores researcher profile and preferences
-- =============================================================================
CREATE TABLE IF NOT EXISTS research_profile (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    researcher_name TEXT NOT NULL,
    affiliation TEXT,
    research_topics TEXT[],
    my_papers TEXT[],
    preferred_venues TEXT[],
    avoid_topics TEXT[],
    time_budget_per_week_minutes INTEGER DEFAULT 120,
    arxiv_categories_include TEXT[],
    arxiv_categories_exclude TEXT[],
    stop_policy JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- 4. Delivery Policy Table
-- Stores delivery preferences for different importance levels
-- =============================================================================
CREATE TABLE IF NOT EXISTS delivery_policy (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    importance_level TEXT NOT NULL CHECK (importance_level IN ('high', 'medium', 'low', 'log_only')),
    send_email BOOLEAN DEFAULT FALSE,
    add_to_calendar BOOLEAN DEFAULT FALSE,
    add_to_reading_list BOOLEAN DEFAULT FALSE,
    share_with_colleagues BOOLEAN DEFAULT FALSE,
    calendar_reminder_minutes INTEGER DEFAULT 1440,
    email_format TEXT DEFAULT 'detailed',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert default delivery policies
INSERT INTO delivery_policy (importance_level, send_email, add_to_calendar, add_to_reading_list, share_with_colleagues)
VALUES 
    ('high', TRUE, TRUE, TRUE, TRUE),
    ('medium', FALSE, FALSE, TRUE, FALSE),
    ('low', FALSE, FALSE, FALSE, FALSE),
    ('log_only', FALSE, FALSE, FALSE, FALSE)
ON CONFLICT DO NOTHING;

-- =============================================================================
-- 5. Run History Table
-- Tracks agent execution history
-- =============================================================================
CREATE TABLE IF NOT EXISTS run_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id TEXT UNIQUE NOT NULL,
    status TEXT DEFAULT 'running' CHECK (status IN ('running', 'done', 'error')),
    user_prompt TEXT,
    start_time TIMESTAMPTZ DEFAULT NOW(),
    end_time TIMESTAMPTZ,
    papers_processed INTEGER DEFAULT 0,
    decisions_made INTEGER DEFAULT 0,
    artifacts_generated INTEGER DEFAULT 0,
    stop_reason TEXT,
    error_message TEXT,
    report JSONB,
    steps JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_run_history_run_id ON run_history(run_id);
CREATE INDEX IF NOT EXISTS idx_run_history_status ON run_history(status);

-- =============================================================================
-- Row Level Security (RLS) - Enable if using anon key
-- =============================================================================
-- For development, you can disable RLS:
-- ALTER TABLE papers DISABLE ROW LEVEL SECURITY;
-- ALTER TABLE colleagues DISABLE ROW LEVEL SECURITY;
-- ALTER TABLE research_profile DISABLE ROW LEVEL SECURITY;
-- ALTER TABLE delivery_policy DISABLE ROW LEVEL SECURITY;
-- ALTER TABLE run_history DISABLE ROW LEVEL SECURITY;

-- =============================================================================
-- Updated At Trigger
-- Automatically updates updated_at column
-- =============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_papers_updated_at BEFORE UPDATE ON papers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_colleagues_updated_at BEFORE UPDATE ON colleagues
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_research_profile_updated_at BEFORE UPDATE ON research_profile
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_delivery_policy_updated_at BEFORE UPDATE ON delivery_policy
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Done! Your Supabase tables are ready for ResearchPulse
-- =============================================================================

-- =============================================================================
-- 6. ArXiv Categories Table
-- Stores the arXiv taxonomy (fetched from arxiv.org and cached)
-- =============================================================================
CREATE TABLE IF NOT EXISTS arxiv_categories (
    code TEXT PRIMARY KEY,  -- e.g., "cs.AI"
    name TEXT NOT NULL,  -- e.g., "Artificial Intelligence"
    group_name TEXT,  -- e.g., "Computer Science"
    description TEXT,
    source TEXT DEFAULT 'arxiv',  -- "arxiv" or "fallback"
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for group filtering
CREATE INDEX IF NOT EXISTS idx_arxiv_categories_group ON arxiv_categories(group_name);

-- Trigger for updated_at
CREATE TRIGGER update_arxiv_categories_updated_at BEFORE UPDATE ON arxiv_categories
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
