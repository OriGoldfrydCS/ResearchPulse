"""
Unit tests for colleague_parser module.

Tests invite-code extraction, name extraction, interest extraction,
and full parse_signup_email orchestration.
"""

import pytest
from src.tools.colleague_parser import (
    extract_code,
    extract_name,
    extract_interests,
    parse_signup_email,
    ParsedSignup,
)


# =============================================================================
# Code Extraction
# =============================================================================

class TestExtractCode:
    """Test invite code extraction from text."""

    @pytest.mark.parametrize("text,expected", [
        ("Code: ABC123", "ABC123"),
        ("code: abc123", "abc123"),
        ("code :XY-99", "XY-99"),
        ("Invite code: my-secret-code", "my-secret-code"),
        ("invite Code: HELLO_WORLD", "HELLO_WORLD"),
        ("CODE=FOOBAR", "FOOBAR"),
        ("CODE = BAZ42", "BAZ42"),
        ("#JOIN1234", "JOIN1234"),
        ("my code is ABCDE", "ABCDE"),
        ("the code is XY123", "XY123"),
        ("use code MYCODE99", "MYCODE99"),
        ("use code: SECRET", "SECRET"),
    ])
    def test_valid_codes(self, text, expected):
        assert extract_code(text) == expected

    def test_no_code(self):
        assert extract_code("Hello, I want to join your group.") is None

    def test_false_positive_filtering(self):
        """Should filter out common words that happen to match patterns."""
        assert extract_code("#the") is None
        assert extract_code("#code") is None

    def test_code_in_subject_and_body(self):
        combined = "Join request\nCode: MYTOKEN123"
        assert extract_code(combined) == "MYTOKEN123"

    def test_code_too_short(self):
        """Codes shorter than minimum length should not match."""
        assert extract_code("Code: AB") is None  # pattern requires 3+

    def test_code_first_match_wins(self):
        text = "Code: FIRST\nuse code SECOND"
        assert extract_code(text) == "FIRST"


# =============================================================================
# Name Extraction
# =============================================================================

class TestExtractName:
    """Test colleague name extraction."""

    @pytest.mark.parametrize("text,from_name,expected", [
        ("Name: Alice Smith", "", "Alice Smith"),
        ("name = Bob Jones", "", "Bob Jones"),
        ("My name is Charlie Brown", "", "Charlie Brown"),
        ("I'm Diana Prince", "", "Diana Prince"),
        ("I am Edward Norton", "", "Edward Norton"),
    ])
    def test_name_patterns(self, text, from_name, expected):
        assert extract_name(text, from_name) == expected

    def test_fallback_to_from_name(self):
        assert extract_name("No name here.", "Jane Doe") == "Jane Doe"

    def test_no_name_found(self):
        assert extract_name("Hello, I want to join.", "") is None

    def test_from_name_stripped(self):
        assert extract_name("no name", "  Padded Name  ") == "Padded Name"

    def test_empty_from_name_no_fallback(self):
        assert extract_name("just text", "  ") is None


# =============================================================================
# Interest Extraction
# =============================================================================

class TestExtractInterests:
    """Test research interest extraction."""

    def test_comma_separated(self):
        text = "Interests: machine learning, NLP, computer vision"
        raw, parsed = extract_interests(text)
        assert "machine learning" in raw
        assert "machine learning" in parsed
        assert "NLP" in parsed
        assert len(parsed) == 3

    def test_newline_separated(self):
        text = "Research interests:\nmachine learning\ndeep learning\nrobots"
        raw, parsed = extract_interests(text)
        assert len(parsed) >= 2

    def test_topics_label(self):
        text = "Topics: quantum computing, cryptography"
        raw, parsed = extract_interests(text)
        assert "quantum computing" in parsed

    def test_areas_label(self):
        text = "Areas: biology, chemistry"
        raw, parsed = extract_interests(text)
        assert "biology" in parsed

    def test_no_interests(self):
        text = "Hello, I'd like to join."
        raw, parsed = extract_interests(text)
        assert raw == ""
        assert parsed == []


# =============================================================================
# Full Parse
# =============================================================================

class TestParseSignupEmail:
    """Test the full parse_signup_email orchestration."""

    def test_complete_email(self):
        result = parse_signup_email(
            subject="Join request",
            body="Code: JOIN123\nName: Alice Smith\nInterests: machine learning, NLP",
            from_name="Alice S",
        )
        assert result.code == "JOIN123"
        assert result.name == "Alice Smith"
        assert "machine learning" in result.interests
        assert result.parse_success is True
        assert result.parse_errors == []

    def test_code_in_subject(self):
        result = parse_signup_email(
            subject="Code: SUBJ42",
            body="Name: Bob\nInterests: physics",
            from_name="",
        )
        assert result.code == "SUBJ42"
        assert result.parse_success is True

    def test_missing_code(self):
        result = parse_signup_email(
            subject="Join request",
            body="Interests: AI, Deep Learning",
            from_name="Carol",
        )
        assert result.code is None
        assert result.parse_success is False
        assert "No invite code found" in result.parse_errors

    def test_missing_interests(self):
        result = parse_signup_email(
            subject="Join request",
            body="Code: ABC123\nHello, add me please.",
            from_name="Dave",
        )
        assert result.code == "ABC123"
        assert result.interests == []
        assert result.parse_success is False
        assert "No research interests found" in result.parse_errors

    def test_missing_both(self):
        result = parse_signup_email(
            subject="Hello",
            body="I'd like to join.",
            from_name="Eve",
        )
        assert result.parse_success is False
        assert len(result.parse_errors) == 2

    def test_name_from_header(self):
        """When body has no name, fall back to from_name header."""
        result = parse_signup_email(
            subject="Code: XY123",
            body="Interests: astrophysics",
            from_name="Header Person",
        )
        assert result.name == "Header Person"
        assert result.parse_success is True
