"""
Unit tests for colleague_tokens module.

Tests HMAC-signed token generation, verification,
expiry enforcement, tamper detection, and URL builders.
"""

import os
import time
import pytest
from unittest.mock import patch

from src.tools.colleague_tokens import (
    TokenAction,
    TokenPayload,
    generate_token,
    verify_token,
    build_action_url,
    generate_remove_url,
    generate_update_url,
    DEFAULT_TOKEN_TTL_SECONDS,
)


@pytest.fixture(autouse=True)
def _stable_secret(monkeypatch):
    """Fix the signing key for deterministic tests."""
    monkeypatch.setenv("COLLEAGUE_TOKEN_SECRET", "test-secret-key-12345")


class TestGenerateToken:
    """Test token generation."""

    def test_returns_string(self):
        token = generate_token("owner-1", "bob@example.com", TokenAction.REMOVE)
        assert isinstance(token, str)
        assert len(token) > 10

    def test_different_actions_produce_different_tokens(self):
        t1 = generate_token("owner-1", "bob@example.com", TokenAction.REMOVE)
        t2 = generate_token("owner-1", "bob@example.com", TokenAction.UPDATE)
        assert t1 != t2


class TestVerifyToken:
    """Test token verification."""

    def test_roundtrip(self):
        token = generate_token("owner-1", "alice@example.com", TokenAction.REMOVE)
        payload = verify_token(token)
        assert payload is not None
        assert payload.owner_id == "owner-1"
        assert payload.colleague_email == "alice@example.com"
        assert payload.action == TokenAction.REMOVE

    def test_update_action(self):
        token = generate_token("owner-2", "c@d.com", TokenAction.UPDATE)
        payload = verify_token(token)
        assert payload.action == TokenAction.UPDATE

    def test_expired_token(self):
        token = generate_token("owner-1", "exp@test.com", TokenAction.REMOVE, ttl_seconds=-1)
        # Token is created with expiry in the past → already expired
        payload = verify_token(token)
        assert payload is None

    def test_tampered_token(self):
        token = generate_token("owner-1", "a@b.com", TokenAction.REMOVE)
        # Flip a character
        chars = list(token)
        idx = len(chars) // 2
        chars[idx] = "A" if chars[idx] != "A" else "B"
        tampered = "".join(chars)
        assert verify_token(tampered) is None

    def test_garbage_token(self):
        assert verify_token("not-a-real-token!!!") is None

    def test_empty_string(self):
        assert verify_token("") is None

    def test_wrong_secret_rejects(self, monkeypatch):
        token = generate_token("owner-1", "a@b.com", TokenAction.REMOVE)
        monkeypatch.setenv("COLLEAGUE_TOKEN_SECRET", "different-secret")
        # Need to invalidate cached key
        import importlib
        import src.tools.colleague_tokens as ct_mod
        importlib.reload(ct_mod)
        assert ct_mod.verify_token(token) is None
        # Restore
        monkeypatch.setenv("COLLEAGUE_TOKEN_SECRET", "test-secret-key-12345")
        importlib.reload(ct_mod)


class TestBuildActionUrl:
    """Test URL construction helpers."""

    def test_build_action_url_no_query(self):
        url = build_action_url("https://example.com/remove", "tok123")
        assert url == "https://example.com/remove?token=tok123"

    def test_build_action_url_existing_query(self):
        url = build_action_url("https://example.com/remove?foo=bar", "tok123")
        assert url == "https://example.com/remove?foo=bar&token=tok123"

    def test_generate_remove_url(self):
        url = generate_remove_url("https://app.com", "owner-1", "bob@test.com")
        assert "/colleague/remove" in url
        assert "token=" in url

    def test_generate_update_url(self):
        url = generate_update_url("https://app.com", "owner-1", "bob@test.com")
        assert "/colleague/update" in url
        assert "token=" in url

    def test_remove_url_token_verifies(self):
        url = generate_remove_url("https://app.com", "owner-1", "bob@test.com")
        token = url.split("token=")[1]
        payload = verify_token(token)
        assert payload is not None
        assert payload.action == TokenAction.REMOVE
        assert payload.colleague_email == "bob@test.com"

    def test_update_url_token_verifies(self):
        url = generate_update_url("https://app.com", "owner-1", "carol@test.com")
        token = url.split("token=")[1]
        payload = verify_token(token)
        assert payload is not None
        assert payload.action == TokenAction.UPDATE
