"""
ResearchPulse Test Suite.

This package contains comprehensive unit and integration tests for all agent functionalities.

Test Structure:
- unit/: Fast tests with mocked external providers
- integration/: Live tests with real email/calendar (requires RUN_LIVE_TESTS=1)

Run Tests:
- Unit tests only: pytest tests/unit -v
- Live integration tests: RUN_LIVE_TESTS=1 pytest tests/integration -v
- All tests: RUN_LIVE_TESTS=1 pytest tests -v
"""
