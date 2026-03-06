"""
Diagnostic: Investigate why real runs produce papers_processed = 0.
Queries DB for recent runs, prompt requests, audit logs, and research profile.
Then runs a controlled real arXiv query to test the pipeline.
"""
import os
import sys
import json
import logging

# Setup
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from dotenv import load_dotenv
load_dotenv()

# Minimal logging
logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")
for name in ["tools.fetch_arxiv", "agent.react_agent", "db.data_service", "tools.decide_delivery"]:
    logging.getLogger(name).setLevel(logging.INFO)

from sqlalchemy import text
from db.database import get_db_session
from db.data_service import get_research_profile, get_colleagues, _get_default_user_id

def mask(s, show=4):
    if not s:
        return str(s)
    s = str(s)
    if len(s) <= show * 2:
        return s[:show] + "***"
    return s[:show] + "***" + s[-show:]

def main():
    print("=" * 70)
    print("UPSTREAM PAPER PIPELINE DIAGNOSTIC")
    print("=" * 70)

    user_id = _get_default_user_id()
    print(f"\nUser ID: {user_id}")

    # ================================================================
    # SECTION 1: Recent Runs from DB
    # ================================================================
    print("\n" + "=" * 70)
    print("[SECTION 1] RECENT RUNS")
    print("=" * 70)

    with get_db_session() as db:
        runs = db.execute(text("""
            SELECT run_id, user_prompt, started_at, ended_at, status,
                   stop_reason, papers_processed, decisions_made,
                   artifacts_generated, error_message,
                   report, metrics
            FROM runs
            WHERE user_id = :uid
            ORDER BY started_at DESC
            LIMIT 5
        """), {"uid": user_id}).fetchall()

    if not runs:
        print("  NO RUNS FOUND IN DATABASE!")
        return
    
    for i, r in enumerate(runs):
        print(f"\n  --- Run {i+1} ---")
        print(f"  run_id:             {r[0]}")
        print(f"  user_prompt:        {(r[1] or '')[:120]}")
        print(f"  started_at:         {r[2]}")
        print(f"  ended_at:           {r[3]}")
        print(f"  status:             {r[4]}")
        print(f"  stop_reason:        {r[5]}")
        print(f"  papers_processed:   {r[6]}")
        print(f"  decisions_made:     {r[7]}")
        print(f"  artifacts_generated:{r[8]}")
        print(f"  error_message:      {(r[9] or 'None')[:200]}")
        
        # Examine report JSON
        report = r[10] if r[10] else {}
        if isinstance(report, str):
            try:
                report = json.loads(report)
            except Exception:
                report = {}
        
        if report:
            summary = report.get("summary", "")
            stats = report.get("stats", {})
            print(f"  report.summary:     {str(summary)[:200]}")
            print(f"  report.stats:       {stats}")
        
        # Examine metrics JSON
        metrics = r[11] if r[11] else {}
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except Exception:
                metrics = {}
        if metrics:
            print(f"  metrics:            {json.dumps(metrics, default=str)[:300]}")

    # Focus on most recent run for deep dive
    most_recent = runs[0]
    recent_run_id = most_recent[0]
    recent_prompt = most_recent[1] or ""
    recent_stop_reason = most_recent[5]

    print(f"\n  >> DEEP DIVE on most recent run: {recent_run_id}")
    print(f"  >> User prompt: '{recent_prompt}'")
    print(f"  >> Stop reason: {recent_stop_reason}")

    # ================================================================
    # SECTION 2: Prompt Request Records
    # ================================================================
    print("\n" + "=" * 70)
    print("[SECTION 2] PROMPT REQUESTS (parsed prompts)")
    print("=" * 70)

    with get_db_session() as db:
        prompts = db.execute(text("""
            SELECT run_id, raw_prompt, template, topic, time_period, time_days,
                   requested_count, output_count, retrieval_count,
                   papers_retrieved, papers_returned,
                   output_enforced, output_insufficient,
                   compliance_status, compliance_message,
                   created_at
            FROM prompt_requests
            WHERE user_id = :uid
            ORDER BY created_at DESC
            LIMIT 5
        """), {"uid": user_id}).fetchall()

    if not prompts:
        print("  NO PROMPT REQUESTS FOUND - _parsed_prompt may never have been saved!")
    else:
        for i, p in enumerate(prompts):
            print(f"\n  --- Prompt {i+1} ---")
            print(f"  run_id:           {p[0]}")
            print(f"  raw_prompt:       {(p[1] or '')[:150]}")
            print(f"  template:         {p[2]}")
            print(f"  topic:            {p[3]}")
            print(f"  time_period:      {p[4]}")
            print(f"  time_days:        {p[5]}")
            print(f"  requested_count:  {p[6]}")
            print(f"  output_count:     {p[7]}")
            print(f"  retrieval_count:  {p[8]}")
            print(f"  papers_retrieved: {p[9]}")
            print(f"  papers_returned:  {p[10]}")
            print(f"  output_enforced:  {p[11]}")
            print(f"  output_insufficient: {p[12]}")
            print(f"  compliance:       {p[13]} - {(p[14] or '')[:100]}")
            print(f"  created_at:       {p[15]}")

    # ================================================================
    # SECTION 3: Run Audit Logs
    # ================================================================
    print("\n" + "=" * 70)
    print("[SECTION 3] RUN AUDIT LOGS")
    print("=" * 70)

    with get_db_session() as db:
        audits = db.execute(text("""
            SELECT run_id, papers_retrieved_count, papers_scored_count,
                   papers_shared_count, papers_discarded_count,
                   execution_time_ms, llm_calls_count, stop_reason,
                   full_log, created_at
            FROM run_audit_logs
            WHERE user_id = :uid
            ORDER BY created_at DESC
            LIMIT 3
        """), {"uid": user_id}).fetchall()

    if not audits:
        print("  NO AUDIT LOGS FOUND")
    else:
        for i, a in enumerate(audits):
            print(f"\n  --- Audit Log {i+1} ---")
            print(f"  run_id:               {a[0]}")
            print(f"  papers_retrieved:      {a[1]}")
            print(f"  papers_scored:         {a[2]}")
            print(f"  papers_shared:         {a[3]}")
            print(f"  papers_discarded:      {a[4]}")
            print(f"  execution_time_ms:     {a[5]}")
            print(f"  llm_calls:             {a[6]}")
            print(f"  stop_reason:           {a[7]}")
            print(f"  created_at:            {a[9]}")
            
            # Parse full_log for key events
            full_log = a[8] if a[8] else {}
            if isinstance(full_log, str):
                try:
                    full_log = json.loads(full_log)
                except:
                    full_log = {}
            
            if isinstance(full_log, dict):
                # Show first 15 log entries to trace the pipeline
                log_entries = full_log.get("log_entries", full_log.get("logs", []))
                if isinstance(log_entries, list) and log_entries:
                    print(f"  full_log entries: {len(log_entries)}")
                    for j, entry in enumerate(log_entries[:25]):
                        if isinstance(entry, dict):
                            msg = entry.get("message", entry.get("msg", str(entry)))[:150]
                            lvl = entry.get("level", "")
                            print(f"    [{j}] {lvl}: {msg}")
                        else:
                            print(f"    [{j}] {str(entry)[:150]}")
                elif isinstance(full_log, dict):
                    # Maybe the full_log IS the structured data
                    for k, v in list(full_log.items())[:10]:
                        print(f"    {k}: {str(v)[:150]}")

    # ================================================================
    # SECTION 4: Research Profile (from DB)
    # ================================================================
    print("\n" + "=" * 70)
    print("[SECTION 4] RESEARCH PROFILE (from DB)")
    print("=" * 70)

    profile = get_research_profile()
    print(f"  research_topics:          {profile.get('research_topics', [])}")
    print(f"  interests_include:        {profile.get('interests_include', '')[:100]}")
    print(f"  interests_exclude:        {profile.get('interests_exclude', '')[:100]}")
    print(f"  arxiv_categories_include: {profile.get('arxiv_categories_include', [])}")
    print(f"  arxiv_categories_exclude: {profile.get('arxiv_categories_exclude', [])}")
    print(f"  avoid_topics:             {profile.get('avoid_topics', [])}")
    print(f"  preferred_venues:         {profile.get('preferred_venues', [])}")

    # ================================================================
    # SECTION 5: Execution Settings (Papers per Run)
    # ================================================================
    print("\n" + "=" * 70)
    print("[SECTION 5] EXECUTION SETTINGS")
    print("=" * 70)

    try:
        with get_db_session() as db:
            settings = db.execute(text("""
                SELECT arxiv_fetch_count, autonomous_enabled, email_enabled,
                       calendar_enabled, time_period_default
                FROM execution_settings
                WHERE user_id = :uid
                LIMIT 1
            """), {"uid": user_id}).fetchone()

        if settings:
            print(f"  arxiv_fetch_count:     {settings[0]}")
            print(f"  autonomous_enabled:    {settings[1]}")
            print(f"  email_enabled:         {settings[2]}")
            print(f"  calendar_enabled:      {settings[3]}")
            print(f"  time_period_default:   {settings[4]}")
        else:
            print("  No execution_settings row found (defaults will be used)")
    except Exception as e:
        print(f"  execution_settings table error: {e}")
        print("  Table may not exist - defaults will be used")

    # Also check the function used by react_agent
    from agent.prompt_controller import get_arxiv_fetch_count
    actual_fetch_count = get_arxiv_fetch_count()
    print(f"  get_arxiv_fetch_count(): {actual_fetch_count}")

    # ================================================================
    # SECTION 6: Papers table - how many total?
    # ================================================================
    print("\n" + "=" * 70)
    print("[SECTION 6] PAPERS IN DATABASE")
    print("=" * 70)

    try:
        with get_db_session() as db:
            # Check actual column names in papers table
            cols = db.execute(text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'papers'
                ORDER BY ordinal_position
            """)).fetchall()
            col_names = [c[0] for c in cols]
            print(f"  papers columns: {col_names}")
            
            paper_count = db.execute(text("SELECT COUNT(*) FROM papers")).scalar()
            
            recent_papers = db.execute(text("""
                SELECT external_id, title, created_at, decision
                FROM papers
                ORDER BY created_at DESC
                LIMIT 5
            """)).fetchall()
    except Exception as e:
        print(f"  Error querying papers: {e}")
        paper_count = 0
        recent_papers = []

    print(f"  Total papers in DB: {paper_count}")
    if recent_papers:
        print(f"  Most recent papers:")
        for p in recent_papers:
            print(f"    {p[0]} | {(p[1] or '')[:60]} | {p[2]} | {p[3]}")
    else:
        print("  No papers stored")

    # ================================================================
    # SECTION 7: Controlled REAL arXiv query
    # ================================================================
    print("\n" + "=" * 70)
    print("[SECTION 7] CONTROLLED REAL ARXIV QUERY")
    print("=" * 70)

    from tools.fetch_arxiv import fetch_arxiv_papers

    # Test 1: Use the user's actual profile topics
    print("\n  --- Test A: Using actual user profile ---")
    research_topics = profile.get("research_topics", [])
    cats_include = profile.get("arxiv_categories_include", [])
    if not cats_include:
        cats_include = ["cs.LG", "stat.ML", "cs.AI"]
    
    topic_query = " OR ".join(research_topics) if research_topics else None
    print(f"  topic_query: {topic_query}")
    print(f"  categories:  {cats_include}")
    print(f"  days_back:   7")
    print(f"  max_results: {actual_fetch_count}")

    result_a = fetch_arxiv_papers(
        categories_include=cats_include,
        query=topic_query,
        max_results=actual_fetch_count,
        days_back=7,
        use_mock=False,
    )
    print(f"  success:     {result_a.success}")
    print(f"  total_found: {result_a.total_found}")
    print(f"  error:       {result_a.error}")
    if result_a.papers:
        for p in result_a.papers[:3]:
            print(f"    {p.arxiv_id}: {p.title[:70]}...")

    # Test 2: Known good topic "PCA" with broad categories
    print("\n  --- Test B: Known good query 'PCA' ---")
    result_b = fetch_arxiv_papers(
        categories_include=["cs.LG", "stat.ML"],
        query="PCA",
        max_results=5,
        days_back=14,
        use_mock=False,
    )
    print(f"  success:     {result_b.success}")
    print(f"  total_found: {result_b.total_found}")
    print(f"  error:       {result_b.error}")
    if result_b.papers:
        for p in result_b.papers[:3]:
            print(f"    {p.arxiv_id}: {p.title[:70]}...")

    # Test 3: User's likely real prompt topic "Vegetables" with broad search
    print("\n  --- Test C: 'Vegetables' keyword-only search ---")
    result_c = fetch_arxiv_papers(
        categories_include=[],  # no category filter
        query="Vegetables",
        max_results=5,
        days_back=30,
        use_mock=False,
    )
    print(f"  success:     {result_c.success}")
    print(f"  total_found: {result_c.total_found}")
    print(f"  error:       {result_c.error}")
    if result_c.papers:
        for p in result_c.papers[:3]:
            print(f"    {p.arxiv_id}: {p.title[:70]}...")

    # Test 4: Simulate exact same query the agent would build
    # from the most recent prompt
    print("\n  --- Test D: Simulated agent query from most recent prompt ---")
    if recent_prompt:
        from agent.prompt_controller import PromptController
        pc = PromptController()
        parsed, _ = pc.parse_and_save(recent_prompt, run_id="diag-test")
        print(f"  parsed.template:       {parsed.template.value}")
        print(f"  parsed.topic:          {parsed.topic}")
        print(f"  parsed.interests_only: {getattr(parsed, 'interests_only', None)}")
        print(f"  parsed.time_period:    {parsed.time_period}")
        print(f"  parsed.time_days:      {parsed.time_days}")
        print(f"  parsed.output_count:   {parsed.output_count}")
        print(f"  parsed.requested_count:{parsed.requested_count}")
        print(f"  parsed.retrieval_count:{parsed.retrieval_count}")
        print(f"  parsed.exclude_topics: {parsed.exclude_topics}")

        # Build query exactly as agent does
        prompt_interests = getattr(parsed, 'interests_only', '') or ''
        if prompt_interests:
            interest_terms = [t.strip() for t in prompt_interests.split(',') if t.strip()]
            sim_query = " OR ".join(interest_terms)
        elif research_topics:
            sim_query = " OR ".join(research_topics)
        else:
            sim_query = None
        
        # Category merging as agent does
        from agent.react_agent import map_interests_to_categories
        user_cats = map_interests_to_categories(prompt_interests) if prompt_interests else []
        print(f"  mapped prompt categories: {user_cats}")
        
        sim_cats = list(set(cats_include) | set(user_cats)) if user_cats else cats_include
        if len(sim_cats) > 10:
            sim_cats = sim_cats[:10]
        
        sim_days = parsed.time_days if parsed.time_days else 7
        sim_fetch = actual_fetch_count
        if parsed.requested_count:
            min_fetch = parsed.requested_count * 3
            if sim_fetch < min_fetch:
                sim_fetch = min_fetch

        print(f"  simulated query:    {sim_query}")
        print(f"  simulated cats:     {sim_cats}")
        print(f"  simulated days:     {sim_days}")
        print(f"  simulated fetch:    {sim_fetch}")

        result_d = fetch_arxiv_papers(
            categories_include=sim_cats,
            query=sim_query,
            max_results=sim_fetch,
            days_back=sim_days,
            use_mock=False,
        )
        print(f"  success:     {result_d.success}")
        print(f"  total_found: {result_d.total_found}")
        print(f"  error:       {result_d.error}")
        if result_d.papers:
            for p in result_d.papers[:3]:
                print(f"    {p.arxiv_id}: {p.title[:70]}...")
        
        # Test seen/unseen
        if result_d.papers and result_d.total_found > 0:
            print(f"\n  --- Seen/Unseen Check ---")
            from tools.check_seen import check_seen_papers_json
            papers_dicts = [p.model_dump() for p in result_d.papers]
            seen_result = check_seen_papers_json(papers=papers_dicts)
            unseen = seen_result.get("unseen_papers", [])
            seen_count = seen_result.get("summary", {}).get("seen", 0)
            print(f"  total checked:  {len(papers_dicts)}")
            print(f"  unseen:         {len(unseen)}")
            print(f"  already seen:   {seen_count}")
    else:
        print("  No recent prompt to simulate")

    # ================================================================
    # SECTION 8: Category Mapping Check
    # ================================================================
    print("\n" + "=" * 70)
    print("[SECTION 8] INTEREST-TO-CATEGORY MAPPING")
    print("=" * 70)

    from agent.react_agent import map_interests_to_categories as _map_cats
    for topic in (research_topics or []) + ["Vegetables", "PCA", "Multi Armed Bandits"]:
        cats = _map_cats(topic)
        print(f"  '{topic}' -> {cats}")

    # Also test taxonomy mapper
    try:
        from tools.arxiv_categories import topic_to_categories as taxonomy_map
        for topic in (research_topics or []) + ["Vegetables", "PCA", "Multi Armed Bandits"]:
            cats = taxonomy_map(topic)
            print(f"  taxonomy: '{topic}' -> {cats}")
    except ImportError:
        print("  taxonomy mapper not available")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
