"""
Unit tests for email_templates module.

Tests logo embedding, HTML building blocks, and complete email template rendering.
"""

import pytest
from unittest.mock import patch, MagicMock
from html import escape

from src.tools.email_templates import (
    _get_logo_base64,
    email_header,
    email_footer,
    email_wrapper_start,
    email_wrapper_end,
    action_button,
    colleague_management_links,
    render_onboarding_instruction_email,
    render_colleague_confirmation_email,
    render_colleague_update_confirmation_email,
    render_colleague_removed_email,
    render_token_error_email,
)


# =============================================================================
# Logo Embedding
# =============================================================================

class TestLogoBase64:
    """Test logo loading and caching."""

    def test_returns_data_uri_or_fallback(self):
        # Reset cache
        import src.tools.email_templates as et
        et._LOGO_BASE64_CACHE = None
        uri = _get_logo_base64()
        assert uri.startswith("data:image/png;base64,")

    def test_caches_result(self):
        import src.tools.email_templates as et
        et._LOGO_BASE64_CACHE = None
        first = _get_logo_base64()
        second = _get_logo_base64()
        assert first == second
        assert et._LOGO_BASE64_CACHE is not None


# =============================================================================
# Building Blocks
# =============================================================================

class TestBuildingBlocks:
    """Test HTML building block functions."""

    def test_email_header_contains_logo(self):
        html = email_header()
        assert "<img" in html
        assert "ResearchPulse" in html

    def test_email_footer_contains_branding(self):
        html = email_footer()
        assert "ResearchPulse" in html

    def test_email_wrapper(self):
        start = email_wrapper_start()
        end = email_wrapper_end()
        assert "<html" in start
        assert "#0f172a" in start  # dark background
        assert "</html>" in end

    def test_action_button(self):
        html = action_button("https://example.com", "Click Me")
        assert "https://example.com" in html
        assert "Click Me" in html
        assert "<a " in html

    def test_colleague_management_links(self):
        html = colleague_management_links(
            remove_url="https://app.com/remove",
            update_url="https://app.com/update",
        )
        assert "https://app.com/remove" in html
        assert "https://app.com/update" in html
        assert "Unsubscribe" in html or "Remove" in html or "unsubscribe" in html


# =============================================================================
# Onboarding Instruction Email
# =============================================================================

class TestOnboardingInstructionEmail:
    """Test render_onboarding_instruction_email."""

    @pytest.mark.parametrize("reason", ["missing_code", "invalid_code", "parse_error"])
    def test_renders_for_all_reasons(self, reason):
        subject, plain, html = render_onboarding_instruction_email(colleague_name="Test User", reason=reason)
        assert isinstance(subject, str)
        assert len(subject) > 0
        assert isinstance(plain, str)
        assert len(plain) > 0
        assert isinstance(html, str)
        assert "<html" in html
        assert "ResearchPulse" in html

    def test_missing_code_content(self):
        subject, plain, html = render_onboarding_instruction_email(colleague_name="Test", reason="missing_code")
        assert "code" in plain.lower() or "code" in html.lower()

    def test_invalid_code_content(self):
        subject, plain, html = render_onboarding_instruction_email(colleague_name="Test", reason="invalid_code")
        assert "recognised" in plain.lower() or "recognised" in html.lower() or "invalid" in plain.lower() or "not recognized" in plain.lower()


# =============================================================================
# Confirmation Email
# =============================================================================

class TestConfirmationEmail:
    """Test render_colleague_confirmation_email."""

    def test_basic_render(self):
        subject, plain, html = render_colleague_confirmation_email(
            colleague_name="Alice",
            interests=[],
        )
        assert "Alice" in plain or "Alice" in html
        assert isinstance(html, str)
        assert "<html" in html

    def test_with_interests(self):
        subject, plain, html = render_colleague_confirmation_email(
            colleague_name="Alice",
            interests=["machine learning", "NLP"],
        )
        # Interests should appear in the email
        assert "machine learning" in html or "machine learning" in plain

    def test_with_management_links(self):
        subject, plain, html = render_colleague_confirmation_email(
            colleague_name="Alice",
            interests=["AI"],
            remove_url="https://app.com/remove?token=xyz",
            update_url="https://app.com/update?token=xyz",
        )
        assert "remove?token=xyz" in html or "remove?token=xyz" in plain


# =============================================================================
# Other Templates
# =============================================================================

class TestOtherTemplates:
    """Test remaining template functions."""

    def test_update_confirmation(self):
        subject, plain, html = render_colleague_update_confirmation_email(
            colleague_name="Carol",
            new_interests=["physics", "astronomy"],
        )
        assert "Carol" in html or "Carol" in plain
        assert "physics" in html or "physics" in plain
        assert "<html" in html

    def test_removed_email(self):
        subject, plain, html = render_colleague_removed_email(
            colleague_name="Dave",
        )
        assert "Dave" in html or "Dave" in plain
        assert "<html" in html

    def test_token_error(self):
        subject, plain, html = render_token_error_email(
            colleague_name="User",
            reason="expired",
        )
        assert "expired" in html.lower() or "expired" in plain.lower()
        assert "<html" in html
