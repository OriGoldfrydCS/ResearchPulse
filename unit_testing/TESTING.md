# ResearchPulse Testing Guide

This document describes the testing strategy, structure, and instructions for the ResearchPulse test suite.

## Overview

The test suite is designed with two layers:

1. **Unit Tests** (Fast, mocked) - Test individual components with mocked external providers
2. **Integration Tests** (Optional live mode) - Test complete workflows, optionally with real email/calendar providers

## Test Structure

```
tests/
├── conftest.py                      # Shared fixtures and mocks
├── unit/
│   ├── test_reply_parser.py         # Reply parsing logic
│   ├── test_ics_generator.py        # ICS file generation
│   ├── test_email_sending.py        # Email payload and sending
│   ├── test_calendar_reminders.py   # Calendar event creation
│   └── test_paper_discovery.py      # Paper retrieval and scoring
└── integration/
    ├── test_inbox_reading.py        # CRITICAL: Email inbox reading
    ├── test_rescheduling.py         # Reschedule via email reply
    ├── test_colleague_workflows.py  # Colleague management
    └── test_end_to_end.py           # Full automation workflows
```

## Quick Start

### Running Unit Tests Only (Default)

```bash
# Run all unit tests (no external services required)
pytest tests/unit/

# Run with verbose output
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_reply_parser.py

# Run specific test
pytest tests/unit/test_reply_parser.py::TestParseReschedule::test_parse_tomorrow
```

### Running Integration Tests

```bash
# Run integration tests with mocks (no external services)
pytest tests/integration/

# Run specific integration test
pytest tests/integration/test_inbox_reading.py
```

### Running Live Integration Tests

Live tests use real email accounts to verify complete functionality.

```bash
# Enable live tests (requires configured credentials)
RUN_LIVE_TESTS=1 pytest tests/integration/

# Run only live inbox reading tests
RUN_LIVE_TESTS=1 pytest tests/integration/test_inbox_reading.py -k "Live"
```

### Running All Tests

```bash
# All tests (mocked)
pytest tests/

# All tests including live
RUN_LIVE_TESTS=1 pytest tests/
```

## Environment Variables

### Required for Live Tests

| Variable | Description | Default |
|----------|-------------|---------|
| `RUN_LIVE_TESTS` | Enable live tests (`1`, `true`, or `yes`) | `0` (disabled) |
| `SMTP_USER` | Email account username/address | - |
| `SMTP_PASSWORD` | Email account password or app password | - |
| `SMTP_HOST` | SMTP server hostname | `smtp.gmail.com` |
| `SMTP_PORT` | SMTP server port | `587` |
| `IMAP_HOST` | IMAP server hostname | `imap.gmail.com` |
| `IMAP_PORT` | IMAP server port | `993` |
| `DATABASE_URL` | PostgreSQL connection string | - |

### Example `.env` for Testing

```bash
# Email Configuration
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
IMAP_HOST=imap.gmail.com
IMAP_PORT=993

# Enable live tests
RUN_LIVE_TESTS=1
```

## Critical Test: Inbox Reading

### Background

A known issue was that ResearchPulse could NOT read incoming email (inbox) content. This must be explicitly tested.

### Test Coverage

The `tests/integration/test_inbox_reading.py` file specifically tests:

1. **IMAP Configuration** - Verify credentials and connection settings
2. **Email Body Extraction** - Parse various email encodings
3. **Subject Decoding** - Handle UTF-8 and encoded subjects
4. **Reply Detection** - Identify calendar invite replies
5. **Live Send/Receive** - Send email to self and verify receipt

### Running Inbox Tests

```bash
# Run mocked inbox tests
pytest tests/integration/test_inbox_reading.py

# Run LIVE inbox tests (actually sends/receives email)
RUN_LIVE_TESTS=1 pytest tests/integration/test_inbox_reading.py -k "Live"

# Run the diagnostic test to check IMAP connection
RUN_LIVE_TESTS=1 pytest tests/integration/test_inbox_reading.py::TestInboxReadingDiagnostics::test_imap_connection -v
```

### What the Live Inbox Test Does

1. Sends an email FROM ResearchPulse TO the same account (self-email)
2. Waits/polls the inbox (up to 60 seconds)
3. Verifies the email appears in the inbox
4. Verifies the agent can parse subject and body

## Test Coverage Summary

### 1. Paper Discovery Pipeline

| Test | File | Description |
|------|------|-------------|
| Paper retrieval | `test_paper_discovery.py` | Verify papers are fetched |
| Importance scoring | `test_paper_discovery.py` | Scores are computed and stored |
| Timestamps | `test_paper_discovery.py` | `added_at` and `published_at` stored |

### 2. Email Summary Sending

| Test | File | Description |
|------|------|-------------|
| One email per query | `test_email_sending.py` | Not one per paper |
| HTML grouping | `test_email_sending.py` | HIGH/MEDIUM/LOW sections |
| arXiv links | `test_email_sending.py` | Correct abs links |
| triggered_by | `test_email_sending.py` | "agent" vs "user" attribution |

### 3. Calendar Reminders

| Test | File | Description |
|------|------|-------------|
| One event per paper | `test_calendar_reminders.py` | Individual reminders |
| Duration estimation | `test_calendar_reminders.py` | Agent-estimated reading time |
| Event description | `test_calendar_reminders.py` | Paper title + link |
| triggered_by | `test_calendar_reminders.py` | Correct attribution |

### 4. Calendar Invite Emails (ICS)

| Test | File | Description |
|------|------|-------------|
| ICS validation | `test_ics_generator.py` | Valid ICS structure |
| ICS content | `test_ics_generator.py` | Required fields present |
| Email linking | `test_calendar_reminders.py` | Linked to calendar event_id |

### 5. Inbox Reading (CRITICAL)

| Test | File | Description |
|------|------|-------------|
| IMAP config | `test_inbox_reading.py` | Credentials loaded |
| Send to self | `test_inbox_reading.py` | Email reaches inbox |
| Parse inbox | `test_inbox_reading.py` | Agent can read messages |
| Reply detection | `test_inbox_reading.py` | Calendar replies identified |

### 6. Rescheduling via Email Reply

| Test | File | Description |
|------|------|-------------|
| Reply parsing | `test_rescheduling.py` | Intent extraction |
| Event update | `test_rescheduling.py` | DB reflects new time |
| New invite sent | `test_rescheduling.py` | Updated ICS |
| Audit trail | `test_rescheduling.py` | History recorded |

### 7. Colleague Workflows

| Test | File | Description |
|------|------|-------------|
| Send to colleagues | `test_colleague_workflows.py` | Summary emails |
| Recipient tracking | `test_colleague_workflows.py` | Colleague vs self |
| Signup detection | `test_colleague_workflows.py` | Email-based onboarding |
| DB persistence | `test_colleague_workflows.py` | Colleagues saved |

### 8. End-to-End Automation

| Test | File | Description |
|------|------|-------------|
| Full workflow | `test_end_to_end.py` | Papers → Email + Calendar |
| Auto labels | `test_end_to_end.py` | "sent automatically" |
| User triggers | `test_end_to_end.py` | "sent by your request" |

## Fixtures Available

The `conftest.py` provides these fixtures:

### Mock Providers

- `mock_email_provider` - MockEmailProvider with send/inbox methods
- `mock_calendar_provider` - MockCalendarProvider with create/update/delete

### Fake Data

- `fake_papers` - Sample papers with scores and metadata
- `fake_inbox_messages` - Sample email messages (reschedule, accept, signup, cancel)
- `fake_colleagues` - Sample colleagues with interests
- `test_settings` - Configuration settings

### Mock Database

- `mock_store` - Mock data store with common operations

### Live Test Support

- `live_email_config` - Real email credentials (skips if not configured)
- `live_test_subject_prefix` - Unique prefix for test emails `[RP TEST]`

## Writing New Tests

### Unit Test Example

```python
def test_parse_reschedule_request():
    """Test parsing a reschedule email reply."""
    from src.agent.reply_parser import parse_reply, ReplyIntent
    
    result = parse_reply(
        "Please move this to tomorrow at 3pm.",
        use_llm=False
    )
    
    assert result.intent == ReplyIntent.RESCHEDULE
    assert result.extracted_datetime_text is not None
```

### Integration Test Example

```python
@pytest.mark.skipif(
    not is_live_tests_enabled(),
    reason="Live tests disabled. Set RUN_LIVE_TESTS=1 to enable."
)
def test_live_send_receive():
    """Test actual email send/receive."""
    from src.tools.decide_delivery import _send_email_smtp
    
    email = os.getenv("SMTP_USER")
    success = _send_email_smtp(
        to_email=email,
        subject="[RP TEST] Integration Test",
        body="Test body",
    )
    
    assert success is True
```

## Safeguards for Live Tests

1. **Test Subject Prefix**: All live test emails use `[RP TEST] <timestamp>_<uuid>` prefix
2. **Self-Only**: Tests send to the same account (no external spam)
3. **Unique IDs**: Each test run has unique identifiers
4. **Skip by Default**: Live tests require explicit `RUN_LIVE_TESTS=1`
5. **Timeout**: Polling operations have 60-second timeout

## Troubleshooting

### Tests Skip with "Live tests disabled"

Set the environment variable:

```bash
export RUN_LIVE_TESTS=1  # Linux/Mac
set RUN_LIVE_TESTS=1     # Windows CMD
$env:RUN_LIVE_TESTS=1    # Windows PowerShell
```

### IMAP Connection Fails

1. Check `SMTP_USER` and `SMTP_PASSWORD` are set
2. For Gmail, use an App Password (not regular password)
3. Enable IMAP in Gmail settings
4. Check firewall/network allows port 993

### Email Not Found in Inbox

1. Email may take time to arrive (up to 60 seconds)
2. Check spam folder manually
3. Gmail may delay emails to self
4. Try running the diagnostic test:
   ```bash
   RUN_LIVE_TESTS=1 pytest tests/integration/test_inbox_reading.py::TestInboxReadingDiagnostics -v
   ```

### Module Import Errors

Ensure you're in the project root and the virtual environment is activated:

```bash
cd ResearchPulse
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
pytest tests/
```

## Known Issues / Regression Notes

### Inbox Reading Failure (Previously Known Issue)

**Problem**: ResearchPulse could not read incoming email (inbox) content.

**Root Cause**: [To be determined by investigation]

**Test Added**: `tests/integration/test_inbox_reading.py`

**How to Verify Fix**:
```bash
RUN_LIVE_TESTS=1 pytest tests/integration/test_inbox_reading.py::TestLiveInboxReading::test_send_and_receive_email -v
```

The test will:
1. Send an email to the configured account
2. Poll the inbox until the email appears
3. Verify the email can be parsed
4. Fail with detailed diagnostics if inbox reading doesn't work

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: pytest tests/unit/ -v

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    env:
      RUN_LIVE_TESTS: 1
      SMTP_USER: ${{ secrets.SMTP_USER }}
      SMTP_PASSWORD: ${{ secrets.SMTP_PASSWORD }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: pytest tests/integration/ -v
```
