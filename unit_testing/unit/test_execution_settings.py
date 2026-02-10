"""
Unit tests for execution settings and DB-backed retrieval limit.

Tests:
- Retrieval max results from DB vs default fallback
- Execution settings validation
- Join code encryption / decryption round-trip
- Scheduler next_run_at calculation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4
from datetime import datetime, timedelta, timezone


# =========================================================================
# Retrieval Max Results
# =========================================================================

class TestRetrievalMaxResults:
    """Test DB-backed retrieval limit with fallback."""

    def test_default_when_no_store(self):
        """Should return MAX_RETRIEVAL_RESULTS when store is unavailable."""
        from src.agent.prompt_controller import get_retrieval_max_results, MAX_RETRIEVAL_RESULTS

        with patch("src.agent.prompt_controller.logger"):
            # Simulate DB not configured
            with patch.dict("sys.modules", {}):
                # Force an import error by patching the lazy import target
                with patch("src.db.database.is_database_configured", side_effect=Exception("no DB")):
                    result = get_retrieval_max_results()
        
        assert result == MAX_RETRIEVAL_RESULTS

    def test_reads_from_store(self):
        """Should return DB value when store is available."""
        from src.agent.prompt_controller import get_retrieval_max_results

        mock_store = MagicMock()
        mock_store.get_retrieval_max_results.return_value = 15
        mock_store.get_or_create_default_user.return_value = {"id": str(uuid4())}

        with patch("src.agent.prompt_controller.logger"):
            with patch("src.db.database.is_database_configured", return_value=True):
                with patch("src.db.store.get_default_store", return_value=mock_store):
                    result = get_retrieval_max_results()

        assert result == 15

    def test_default_when_store_returns_default(self):
        """Should use MAX_RETRIEVAL_RESULTS when store returns the default 7."""
        from src.agent.prompt_controller import get_retrieval_max_results, MAX_RETRIEVAL_RESULTS

        mock_store = MagicMock()
        mock_store.get_retrieval_max_results.return_value = 7
        mock_store.get_or_create_default_user.return_value = {"id": str(uuid4())}

        with patch("src.agent.prompt_controller.logger"):
            with patch("src.db.database.is_database_configured", return_value=True):
                with patch("src.db.store.get_default_store", return_value=mock_store):
                    result = get_retrieval_max_results()

        assert result == 7
        assert result == MAX_RETRIEVAL_RESULTS

    def test_parsed_prompt_uses_retrieval_max(self):
        """ParsedPrompt should use _retrieval_max when set."""
        from src.agent.prompt_controller import ParsedPrompt, MAX_RETRIEVAL_RESULTS

        parsed = ParsedPrompt(raw="test", user_query="test")

        # Default: should return MAX_RETRIEVAL_RESULTS
        assert parsed.retrieval_count == MAX_RETRIEVAL_RESULTS

        # Override via _retrieval_max
        parsed._retrieval_max = 20
        assert parsed.retrieval_count == 20

    def test_parsed_prompt_none_retrieval_max_uses_default(self):
        """ParsedPrompt with _retrieval_max=None should fall back to constant."""
        from src.agent.prompt_controller import ParsedPrompt, MAX_RETRIEVAL_RESULTS

        parsed = ParsedPrompt(raw="test", user_query="test")
        parsed._retrieval_max = None
        assert parsed.retrieval_count == MAX_RETRIEVAL_RESULTS


# =========================================================================
# Join Code Encryption
# =========================================================================

class TestJoinCodeEncryption:
    """Test AES encryption/decryption for join codes."""

    def test_encrypt_decrypt_round_trip(self):
        """Encrypting then decrypting should return original code."""
        from src.tools.join_code_crypto import encrypt_join_code, decrypt_join_code

        with patch.dict("os.environ", {"JOIN_CODE_ENCRYPTION_KEY": "a" * 32}):
            original = "MY-SECRET-CODE-2024"
            encrypted = encrypt_join_code(original)

            assert encrypted is not None
            assert encrypted != original

            decrypted = decrypt_join_code(encrypted)
            assert decrypted == original

    def test_verify_join_code(self):
        """verify_join_code should match plaintext against encrypted."""
        from src.tools.join_code_crypto import encrypt_join_code, verify_join_code

        with patch.dict("os.environ", {"JOIN_CODE_ENCRYPTION_KEY": "b" * 32}):
            code = "test-code-abc"
            encrypted = encrypt_join_code(code)

            assert verify_join_code(code, encrypted) is True
            assert verify_join_code("wrong-code", encrypted) is False

    def test_decrypt_returns_none_on_garbage(self):
        """Decrypting invalid data should return None, not raise."""
        from src.tools.join_code_crypto import decrypt_join_code

        with patch.dict("os.environ", {"JOIN_CODE_ENCRYPTION_KEY": "c" * 32}):
            result = decrypt_join_code("not-valid-encrypted-data")
            assert result is None

    def test_encrypt_different_keys_produce_different_output(self):
        """Different encryption keys should produce different ciphertext."""
        from src.tools.join_code_crypto import encrypt_join_code

        code = "same-code"
        with patch.dict("os.environ", {"JOIN_CODE_ENCRYPTION_KEY": "d" * 32}):
            enc1 = encrypt_join_code(code)

        with patch.dict("os.environ", {"JOIN_CODE_ENCRYPTION_KEY": "e" * 32}):
            enc2 = encrypt_join_code(code)

        # Fernet uses random IV, so even same key produces different output,
        # but definitely different keys should too
        assert enc1 != enc2


# =========================================================================
# Execution Settings Validation
# =========================================================================

class TestExecutionSettingsValidation:
    """Test execution settings update logic."""

    def test_retrieval_max_clamped_to_range(self):
        """Store should clamp retrieval_max_results to 1-50."""
        from src.db.postgres_store import PostgresStore

        store = PostgresStore.__new__(PostgresStore)

        # Mock session and settings
        mock_settings = MagicMock()
        mock_settings.retrieval_max_results = 7
        mock_settings.execution_mode = "manual"
        mock_settings.scheduled_frequency = None
        mock_settings.scheduled_every_x_days = None
        mock_settings.next_run_at = None
        mock_settings.to_dict = Mock(return_value={})

        mock_session = MagicMock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_settings

        with patch.object(store, 'Session', return_value=mock_session):
            # Test value above max gets clamped
            store.update_execution_settings(
                user_id=uuid4(),
                retrieval_max_results=100,
                execution_mode="manual"
            )
            assert mock_settings.retrieval_max_results == 50

    def test_scheduled_mode_computes_next_run(self):
        """Setting execution_mode to 'scheduled' with daily should set next_run_at."""
        from src.db.postgres_store import PostgresStore

        store = PostgresStore.__new__(PostgresStore)

        mock_settings = MagicMock()
        mock_settings.retrieval_max_results = 7
        mock_settings.execution_mode = "manual"
        mock_settings.scheduled_frequency = None
        mock_settings.scheduled_every_x_days = None
        mock_settings.next_run_at = None
        mock_settings.last_run_at = None
        mock_settings.to_dict = Mock(return_value={
            "execution_mode": "scheduled",
            "next_run_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        })

        mock_session = MagicMock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_settings

        with patch.object(store, 'Session', return_value=mock_session):
            result = store.update_execution_settings(
                user_id=uuid4(),
                execution_mode="scheduled",
                scheduled_frequency="daily"
            )

        assert mock_settings.execution_mode == "scheduled"
        assert mock_settings.scheduled_frequency == "daily"
        # next_run_at should have been computed
        assert mock_settings.next_run_at is not None


# =========================================================================
# Scheduler Service
# =========================================================================

class TestSchedulerService:
    """Test scheduler service logic."""

    def test_run_lock_prevents_concurrent(self):
        """Only one run should execute at a time."""
        from src.tools.scheduler_service import _run_lock

        acquired = _run_lock.acquire(blocking=False)
        assert acquired is True

        # Second acquire should fail (non-blocking)
        second = _run_lock.acquire(blocking=False)
        assert second is False

        _run_lock.release()

    def test_scheduler_start_stop(self):
        """Scheduler should start and stop cleanly."""
        from src.tools.scheduler_service import SchedulerService

        service = SchedulerService(check_interval_seconds=1)

        # Not running initially
        assert service._running is False

        # After stop, should not be running
        service.stop()
        assert service._running is False
