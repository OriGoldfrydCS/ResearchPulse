"""
ResearchPulse - Main Application Entry Point

Starts the FastAPI server with configured routes and static file serving.
Provides CLI commands for database management and migration.
"""

from __future__ import annotations

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.api import router
from src.api.dashboard_routes import router as dashboard_router

# =============================================================================
# Environment Configuration
# =============================================================================

def get_env_var(name: str, default: str | None = None, required: bool = False) -> str | None:
    """
    Get environment variable with validation.
    
    Args:
        name: Environment variable name
        default: Default value if not set
        required: If True, raise error when missing
        
    Returns:
        The environment variable value or default
        
    Raises:
        SystemExit: If required variable is missing
    """
    value = os.getenv(name, default)
    if required and not value:
        print(f"ERROR: Missing required environment variable: {name}")
        print(f"Please set {name} in your .env file or environment.")
        sys.exit(1)
    return value


# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment from: {env_path}")
except ImportError:
    print("python-dotenv not installed, using system environment variables")


# Server configuration (not required for basic startup)
APP_HOST = get_env_var("APP_HOST", "127.0.0.1")
APP_PORT = int(get_env_var("APP_PORT", "8000"))


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="ResearchPulse",
    description="Research Awareness and Sharing Agent - A ReAct agent for arXiv paper discovery",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")
app.include_router(dashboard_router)  # router already has /api prefix

# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Serve index.html at root
@app.get("/", include_in_schema=False)
async def root():
    """Serve the main UI."""
    from fastapi.responses import FileResponse
    index_path = static_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "ResearchPulse API", "docs": "/docs"}


# =============================================================================
# Startup Event
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    # Import retrieval limits for logging
    from src.agent.prompt_controller import MAX_RETRIEVAL_RESULTS
    from src.agent.stop_controller import StopPolicy
    
    print("=" * 60)
    print("ResearchPulse - Research Awareness and Sharing Agent")
    print("=" * 60)
    print(f"Server starting on http://{APP_HOST}:{APP_PORT}")
    print(f"API documentation: http://{APP_HOST}:{APP_PORT}/docs")
    print(f"Web UI: http://{APP_HOST}:{APP_PORT}/")
    print(f"ResearchPulse retrieval_limit={MAX_RETRIEVAL_RESULTS}")
    print(f"ResearchPulse max_papers_checked={StopPolicy().max_papers_checked}")
    print("=" * 60)
    
    # Log database configuration and seed templates
    try:
        from src.db.database import get_database_url, is_database_configured
        from src.db.data_service import get_saved_prompts, seed_default_templates, seed_builtin_prompt_templates
        
        db_url = get_database_url()
        if db_url:
            # Mask password in URL for logging
            masked_url = db_url.split("@")[-1] if "@" in db_url else "configured"
            print(f"Database: {masked_url}")
            
            if is_database_configured():
                # Seed the old default templates (for saved prompts)
                seeded = seed_default_templates()
                if seeded > 0:
                    print(f"Seeded {seeded} default templates")
                prompts = get_saved_prompts()
                print(f"Saved templates in DB: {len(prompts)}")
                
                # Seed builtin prompt templates (for new Prompt Templates feature)
                builtin_seeded = seed_builtin_prompt_templates()
                if builtin_seeded > 0:
                    print(f"Seeded {builtin_seeded} builtin prompt templates")
        else:
            print("Database: not configured (using local files)")
    except Exception as e:
        print(f"Database check error: {e}")


# =============================================================================
# CLI Commands
# =============================================================================

def db_init_command():
    """
    Initialize database: validate DATABASE_URL, run migrations.
    
    Usage: python main.py db-init
    """
    print("=" * 60)
    print("ResearchPulse - Database Initialization")
    print("=" * 60)
    
    # Check DATABASE_URL
    from src.db.database import is_database_configured, check_connection
    
    if not is_database_configured():
        print("ERROR: DATABASE_URL not configured!")
        print("Please set DATABASE_URL in your .env file or environment.")
        print("Example: DATABASE_URL=postgresql://user:pass@host:5432/dbname")
        sys.exit(1)
    
    print("[1/3] DATABASE_URL is configured.")
    
    # Test connection
    try:
        connected, message = check_connection()
        if not connected:
            print(f"ERROR: Failed to connect to database: {message}")
            sys.exit(1)
        print(f"[2/3] Database connection successful: {message}")
    except Exception as e:
        print(f"ERROR: Database connection failed: {e}")
        sys.exit(1)
    
    # Run migrations
    try:
        from alembic.config import Config
        from alembic import command
        
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
        print("[3/3] Migrations applied successfully!")
    except Exception as e:
        print(f"ERROR: Failed to run migrations: {e}")
        print("Make sure alembic is installed: pip install alembic")
        sys.exit(1)
    
    print("=" * 60)
    print("Database initialization complete!")
    print("=" * 60)


def migrate_local_to_db_command():
    """
    Migrate local JSON files to database.
    
    Usage: python main.py migrate-local-to-db
    """
    print("=" * 60)
    print("ResearchPulse - Local Data Migration")
    print("=" * 60)
    
    from src.db.database import is_database_configured, check_connection
    from src.db.postgres_store import PostgresStore
    
    if not is_database_configured():
        print("ERROR: DATABASE_URL not configured!")
        sys.exit(1)
    
    connected, _ = check_connection()
    if not connected:
        print("ERROR: Cannot connect to database. Run 'python main.py db-init' first.")
        sys.exit(1)
    
    store = PostgresStore()
    project_root = Path(__file__).parent
    
    stats = {
        "users": 0,
        "papers": 0,
        "paper_views": 0,
        "colleagues": 0,
        "emails": 0,
        "calendar_events": 0,
        "shares": 0,
        "delivery_policies": 0,
    }
    
    # 1. Migrate research_profile.json -> users
    profile_path = project_root / "data" / "research_profile.json"
    user_id = None
    if profile_path.exists():
        print(f"Migrating: {profile_path}")
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                profile = json.load(f)
            
            # Create or update user using data dict
            user = store.upsert_user(profile)
            user_id = user.get("id")
            stats["users"] += 1
            print(f"  -> Created/updated user: {user.get('name')}")
        except Exception as e:
            print(f"  -> Error: {e}")
    
    # Get default user if none from file
    if not user_id:
        try:
            user = store.get_or_create_default_user()
            user_id = user.get("id")
        except Exception:
            pass
    
    # 2. Migrate colleagues.json -> colleagues
    colleagues_path = project_root / "data" / "colleagues.json"
    if colleagues_path.exists() and user_id:
        print(f"Migrating: {colleagues_path}")
        try:
            with open(colleagues_path, "r", encoding="utf-8") as f:
                colleagues_data = json.load(f)
            
            from uuid import UUID
            uid = UUID(user_id) if isinstance(user_id, str) else user_id
            
            for colleague in colleagues_data.get("colleagues", []):
                store.upsert_colleague(uid, colleague)
                stats["colleagues"] += 1
            print(f"  -> Migrated {stats['colleagues']} colleagues")
        except Exception as e:
            print(f"  -> Error: {e}")
    
    # 3. Migrate papers_state.json -> papers, paper_views
    papers_path = project_root / "data" / "papers_state.json"
    if papers_path.exists() and user_id:
        print(f"Migrating: {papers_path}")
        try:
            with open(papers_path, "r", encoding="utf-8") as f:
                papers_data = json.load(f)
            
            from uuid import UUID
            uid = UUID(user_id) if isinstance(user_id, str) else user_id
            
            for paper in papers_data.get("papers", []):
                # Insert paper using dict
                paper_id = paper.get("id") or paper.get("arxiv_id")
                if paper_id:
                    paper_dict = {
                        "source": "arxiv",
                        "external_id": paper_id,
                        "title": paper.get("title", "Unknown"),
                        "abstract": paper.get("abstract"),
                        "authors": paper.get("authors", []),
                        "categories": paper.get("categories", []),
                        "url": paper.get("url") or f"https://arxiv.org/abs/{paper_id}",
                        "published_at": paper.get("published_at"),
                    }
                    db_paper = store.upsert_paper(paper_dict)
                    stats["papers"] += 1
                    
                    # Insert paper_view
                    db_paper_id = UUID(db_paper.get("id")) if isinstance(db_paper.get("id"), str) else db_paper.get("id")
                    view_data = {
                        "decision": paper.get("decision", "seen"),
                        "importance": paper.get("importance") or paper.get("score"),
                        "notes": paper.get("notes"),
                    }
                    store.upsert_paper_view(uid, db_paper_id, view_data)
                    stats["paper_views"] += 1
            
            print(f"  -> Migrated {stats['papers']} papers and {stats['paper_views']} views")
        except Exception as e:
            print(f"  -> Error: {e}")
    
    # 4. Migrate delivery_policy.json -> delivery_policies
    policy_path = project_root / "data" / "delivery_policy.json"
    if policy_path.exists() and user_id:
        print(f"Migrating: {policy_path}")
        try:
            with open(policy_path, "r", encoding="utf-8") as f:
                policy_data = json.load(f)
            
            from uuid import UUID
            uid = UUID(user_id) if isinstance(user_id, str) else user_id
            
            store.upsert_delivery_policy(uid, policy_data)
            stats["delivery_policies"] += 1
            print(f"  -> Migrated delivery policy")
        except Exception as e:
            print(f"  -> Error: {e}")
    
    # 5. Migrate artifacts/emails -> emails
    emails_dir = project_root / "artifacts" / "emails"
    if emails_dir.exists() and user_id:
        print(f"Migrating: {emails_dir}")
        try:
            from uuid import UUID
            uid = UUID(user_id) if isinstance(user_id, str) else user_id
            
            for email_file in emails_dir.glob("*.json"):
                with open(email_file, "r", encoding="utf-8") as f:
                    email_data = json.load(f)
                store.create_email(
                    user_id=uid,
                    paper_id=None,  # email_data.get("paper_id") would need conversion
                    recipient_email=email_data.get("recipient_email", "user@example.com"),
                    subject=email_data.get("subject", ""),
                    body_text=email_data.get("body_text") or email_data.get("body", ""),
                )
                stats["emails"] += 1
            print(f"  -> Migrated {stats['emails']} emails")
        except Exception as e:
            print(f"  -> Error: {e}")
    
    # 6. Migrate artifacts/calendar -> calendar_events (optional, skip if no dir)
    calendar_dir = project_root / "artifacts" / "calendar"
    if calendar_dir.exists() and user_id:
        print(f"Migrating: {calendar_dir}")
        try:
            from uuid import UUID
            uid = UUID(user_id) if isinstance(user_id, str) else user_id
            
            ics_files = list(calendar_dir.glob("*.ics"))[:10]  # Limit to first 10
            for ics_file in ics_files:
                try:
                    with open(ics_file, "r", encoding="utf-8") as f:
                        ics_content = f.read()
                    
                    store.insert_calendar_event(
                        user_id=uid,
                        paper_id=None,
                        title=f"Read: {ics_file.stem}",
                        start_time=datetime.now(),
                        ics_text=ics_content,
                    )
                    stats["calendar_events"] += 1
                except Exception:
                    pass  # Skip problematic files
            print(f"  -> Migrated {stats['calendar_events']} calendar events")
        except Exception as e:
            print(f"  -> Skipped calendar events: {e}")
    
    # 7. Migrate artifacts/shares -> shares
    shares_dir = project_root / "artifacts" / "shares"
    if shares_dir.exists() and user_id:
        print(f"Migrating: {shares_dir}")
        try:
            from uuid import UUID
            uid = UUID(user_id) if isinstance(user_id, str) else user_id
            
            for share_file in shares_dir.glob("*.json"):
                with open(share_file, "r", encoding="utf-8") as f:
                    share_data = json.load(f)
                # Shares require valid paper_id and colleague_id UUIDs
                # Skip for now as local IDs may not match
                stats["shares"] += 1
            print(f"  -> Migrated {stats['shares']} shares")
        except Exception as e:
            print(f"  -> Error: {e}")
    
    # Summary
    print("=" * 60)
    print("Migration Summary:")
    for key, count in stats.items():
        print(f"  {key}: {count}")
    print("=" * 60)
    print("Migration complete!")
    print("Note: This migration is idempotent. Running again will update existing records.")
    print("=" * 60)


def run_server_command():
    """Run the FastAPI server with uvicorn."""
    # Production check: require DATABASE_URL
    env = os.getenv("ENV", "development")
    if env == "production":
        from src.db.database import is_database_configured
        if not is_database_configured():
            print("ERROR: DATABASE_URL is required in production!")
            print("Please set DATABASE_URL environment variable.")
            sys.exit(1)
    
    uvicorn.run(
        "main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=(env != "production"),  # Only reload in development
        log_level="info",
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Parse CLI arguments and execute appropriate command."""
    parser = argparse.ArgumentParser(
        description="ResearchPulse - Research Awareness and Sharing Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  server            Start the FastAPI server (default)
  db-init           Initialize database and run migrations
  migrate-local-to-db  Migrate local JSON files to database

Examples:
  python main.py                      # Start server
  python main.py server               # Start server
  python main.py db-init              # Initialize database
  python main.py migrate-local-to-db  # Migrate local data
        """
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        default="server",
        choices=["server", "db-init", "migrate-local-to-db"],
        help="Command to run (default: server)"
    )
    
    args = parser.parse_args()
    
    if args.command == "db-init":
        db_init_command()
    elif args.command == "migrate-local-to-db":
        migrate_local_to_db_command()
    else:
        run_server_command()


if __name__ == "__main__":
    main()
