"""
ResearchPulse - Main Application Entry Point

Starts the FastAPI server with configured routes and static file serving.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.api import router

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
    print("=" * 60)
    print("ResearchPulse - Research Awareness and Sharing Agent")
    print("=" * 60)
    print(f"Server starting on http://{APP_HOST}:{APP_PORT}")
    print(f"API documentation: http://{APP_HOST}:{APP_PORT}/docs")
    print(f"Web UI: http://{APP_HOST}:{APP_PORT}/")
    print("=" * 60)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the application with uvicorn."""
    uvicorn.run(
        "main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=True,  # Enable auto-reload during development
        log_level="info",
    )


if __name__ == "__main__":
    main()
