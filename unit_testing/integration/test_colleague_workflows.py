"""
Integration tests for colleague workflows.

Tests:
1. Sending summary emails to colleagues
2. Colleague email persistence with recipient info
3. Distinguishing colleague vs self recipients
4. Colleague onboarding via email signup
5. Detecting and processing colleague signup emails
"""

import pytest
import os
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from src.tools.decide_delivery import (
    ColleagueInfo,
    ColleagueAction,
    process_colleague_surplus,
    ScoredPaper,
)
from src.tools.email_poller import (
    is_colleague_signup_email,
    fetch_colleague_signup_emails,
)
from unit_testing.conftest import get_test_email, get_colleague_test_email


# Check if we should run live tests
def is_live_tests_enabled():
    """Check if live integration tests are enabled."""
    return os.getenv("RUN_LIVE_TESTS", "").lower() in ("1", "true", "yes")


class TestColleagueInfoModel:
    """Test the ColleagueInfo model."""
    
    def test_create_colleague(self):
        """Test creating a ColleagueInfo instance."""
        colleague = ColleagueInfo(
            id="c-001",
            name="Jane Smith",
            email="jane@university.edu",
            affiliation="MIT",
            topics=["machine learning", "NLP", "transformers"],
            sharing_preference="daily_digest",
            arxiv_categories_interest=["cs.LG", "cs.CL"],
            added_by="manual",
            auto_send_emails=True,
        )
        
        assert colleague.id == "c-001"
        assert colleague.name == "Jane Smith"
        assert colleague.sharing_preference == "daily_digest"
        assert colleague.auto_send_emails is True
    
    def test_colleague_default_values(self):
        """Test default values for ColleagueInfo."""
        colleague = ColleagueInfo(
            id="c-002",
            name="John Doe",
            email=get_colleague_test_email("john"),
        )
        
        assert colleague.affiliation is None
        assert colleague.topics == []
        assert colleague.sharing_preference == "daily_digest"
        assert colleague.added_by == "manual"
        assert colleague.auto_send_emails is True
    
    def test_colleague_email_added_by(self):
        """Test colleague added by email signup."""
        colleague = ColleagueInfo(
            id="c-003",
            name="Alice",
            email=get_colleague_test_email("alice"),
            added_by="email",
        )
        
        assert colleague.added_by == "email"


class TestColleagueSharingWorkflow:
    """Test colleague sharing workflow."""
    
    @pytest.fixture
    def sample_papers(self):
        """Create sample papers."""
        return [
            {
                "arxiv_id": "2401.00001",
                "title": "Deep Learning for NLP",
                "abstract": "We present a novel approach to natural language processing using transformers.",
                "relevance_score": 0.9,
                "novelty_score": 0.8,
                "importance": "high",
                "categories": ["cs.CL", "cs.LG"],
                "authors": ["Alice", "Bob"],
            },
            {
                "arxiv_id": "2401.00002",
                "title": "Computer Vision Advances",
                "abstract": "New techniques in image recognition and object detection.",
                "relevance_score": 0.7,
                "novelty_score": 0.6,
                "importance": "medium",
                "categories": ["cs.CV"],
                "authors": ["Charlie"],
            },
        ]
    
    @pytest.fixture
    def sample_colleagues(self):
        """Create sample colleagues."""
        return [
            {
                "id": "c-001",
                "name": "Jane Smith",
                "email": "jane@university.edu",
                "topics": ["NLP", "transformers", "language models"],
                "sharing_preference": "immediate",
                "arxiv_categories_interest": ["cs.CL"],
                "added_by": "manual",
                "auto_send_emails": True,
            },
            {
                "id": "c-002",
                "name": "Bob Johnson",
                "email": "bob@tech.com",
                "topics": ["computer vision", "image recognition"],
                "sharing_preference": "daily_digest",
                "arxiv_categories_interest": ["cs.CV"],
                "added_by": "manual",
                "auto_send_emails": True,
            },
            {
                "id": "c-003",
                "name": "Carol Williams",
                "email": "carol@research.org",
                "topics": ["robotics"],
                "sharing_preference": "on_request",
                "arxiv_categories_interest": ["cs.RO"],
                "added_by": "manual",
                "auto_send_emails": True,
            },
        ]
    
    @pytest.fixture
    def delivery_policy(self):
        """Create delivery policy."""
        return {
            "colleague_sharing_settings": {
                "enabled": True,
                "respect_sharing_preferences": True,
                "simulate_output": True,
            }
        }
    
    def test_process_colleague_surplus_basic(self, sample_papers, sample_colleagues, delivery_policy):
        """Test basic colleague surplus processing."""
        result = process_colleague_surplus(
            all_scored_papers=sample_papers,
            owner_paper_ids=["2401.00001"],  # Owner selected first paper
            colleagues=sample_colleagues,
            delivery_policy=delivery_policy,
            researcher_name="Dr. Researcher",
        )
        
        assert result["success"] is True
        assert result["paper_count"] == 2
        assert "colleague_actions" in result
    
    def test_colleague_topic_matching(self, sample_papers, sample_colleagues, delivery_policy):
        """Test that papers are matched to colleagues by topic."""
        result = process_colleague_surplus(
            all_scored_papers=sample_papers,
            owner_paper_ids=[],
            colleagues=sample_colleagues,
            delivery_policy=delivery_policy,
            researcher_name="Dr. Researcher",
        )
        
        actions = result["colleague_actions"]
        
        # Jane (NLP interested) should get NLP paper
        jane_actions = [a for a in actions if a["colleague_id"] == "c-001"]
        nlp_paper_action = [a for a in jane_actions if "2401.00001" in a["paper_id"]]
        
        assert len(nlp_paper_action) > 0
        # Should be immediate share for Jane
        if nlp_paper_action[0]["action_type"] != "skip":
            assert nlp_paper_action[0]["action_type"] == "share_immediate"
    
    def test_colleague_category_matching(self, sample_papers, sample_colleagues, delivery_policy):
        """Test that papers are matched by arXiv category."""
        result = process_colleague_surplus(
            all_scored_papers=sample_papers,
            owner_paper_ids=[],
            colleagues=sample_colleagues,
            delivery_policy=delivery_policy,
            researcher_name="Dr. Researcher",
        )
        
        actions = result["colleague_actions"]
        
        # Bob (CV interested) should get CV paper
        bob_actions = [a for a in actions if a["colleague_id"] == "c-002"]
        cv_paper_action = [a for a in bob_actions if "2401.00002" in a["paper_id"]]
        
        assert len(cv_paper_action) > 0
        # Bob wants daily digest
        if cv_paper_action[0]["action_type"] != "skip":
            assert cv_paper_action[0]["action_type"] == "share_daily"
    
    def test_on_request_colleague_skipped(self, sample_papers, sample_colleagues, delivery_policy):
        """Test that on_request colleagues are skipped."""
        result = process_colleague_surplus(
            all_scored_papers=sample_papers,
            owner_paper_ids=[],
            colleagues=sample_colleagues,
            delivery_policy=delivery_policy,
            researcher_name="Dr. Researcher",
        )
        
        actions = result["colleague_actions"]
        
        # Carol (on_request) should be skipped
        carol_actions = [a for a in actions if a["colleague_id"] == "c-003"]
        
        for action in carol_actions:
            assert action["action_type"] == "skip"
    
    def test_papers_sent_to_both_owner_and_colleagues(self, sample_papers, sample_colleagues, delivery_policy):
        """Test that papers can go to both owner AND colleagues."""
        # Owner selected the NLP paper
        result = process_colleague_surplus(
            all_scored_papers=sample_papers,
            owner_paper_ids=["2401.00001"],  # Owner gets NLP paper
            colleagues=sample_colleagues,
            delivery_policy=delivery_policy,
            researcher_name="Dr. Researcher",
        )
        
        actions = result["colleague_actions"]
        
        # Jane should still get NLP paper even though owner has it
        jane_nlp = [
            a for a in actions 
            if a["colleague_id"] == "c-001" and "2401.00001" in a["paper_id"]
        ]
        
        assert len(jane_nlp) > 0
        # Should mark it as also for owner
        if jane_nlp[0]["action_type"] != "skip":
            assert "also_for_owner" in jane_nlp[0].get("details", {}) or \
                   "ALSO FOR OWNER" in jane_nlp[0].get("relevance_reason", "")
    
    def test_no_colleagues_enabled(self, sample_papers):
        """Test handling no auto-send colleagues."""
        disabled_colleagues = [
            {
                "id": "c-001",
                "name": "Jane",
                "email": get_colleague_test_email("jane"),
                "auto_send_emails": False,  # Disabled
            }
        ]
        
        result = process_colleague_surplus(
            all_scored_papers=sample_papers,
            owner_paper_ids=[],
            colleagues=disabled_colleagues,
            delivery_policy={"colleague_sharing_settings": {"enabled": True}},
            researcher_name="Dr. Researcher",
        )
        
        assert result["message"] == "No colleagues have auto-send enabled"
    
    def test_sharing_disabled_in_policy(self, sample_papers, sample_colleagues):
        """Test that sharing can be disabled in policy."""
        result = process_colleague_surplus(
            all_scored_papers=sample_papers,
            owner_paper_ids=[],
            colleagues=sample_colleagues,
            delivery_policy={"colleague_sharing_settings": {"enabled": False}},
            researcher_name="Dr. Researcher",
        )
        
        assert result["message"] == "Colleague sharing is disabled in policy"
        assert result["colleague_actions"] == []


class TestColleagueSignupDetection:
    """Test detection of colleague signup emails."""
    
    def test_detect_subscribe_keyword(self):
        """Test detection using 'subscribe' keyword."""
        subject = "ResearchPulse subscription"
        body = "Hi, I'd like to subscribe to receive research updates."
        
        result = is_colleague_signup_email(subject, body)
        
        assert result is True
    
    def test_detect_add_me_keyword(self):
        """Test detection using 'add me' keyword."""
        subject = "Research papers"
        body = "Please add me to your ResearchPulse colleague list."
        
        result = is_colleague_signup_email(subject, body)
        
        assert result is True
    
    def test_detect_sign_up_keyword(self):
        """Test detection using 'sign up' keyword."""
        subject = "ResearchPulse"
        body = "I want to sign up for your paper updates."
        
        result = is_colleague_signup_email(subject, body)
        
        assert result is True
    
    def test_detect_colleague_list_mention(self):
        """Test detection mentioning colleague list."""
        test_email = get_colleague_test_email("newcolleague")
        subject = "Request"
        body = f"Can you add {test_email} to your ResearchPulse colleague list?"
        
        result = is_colleague_signup_email(subject, body)
        
        assert result is True
    
    def test_not_detected_unrelated_email(self):
        """Non-automated personal emails are now treated as potential interactions.
        
        The processing layer handles format checking and sends instructions once.
        Only clearly automated/system emails are filtered out.
        """
        subject = "Meeting tomorrow"
        body = "Let's discuss the project at 2pm."
        
        result = is_colleague_signup_email(subject, body)
        
        # Personal emails are now accepted for processing
        assert result is True
    
    def test_not_detected_without_researchpulse(self):
        """Emails without ResearchPulse mention are still accepted for processing.
        
        The broadened detection treats any non-automated email as a potential
        interaction. Format validation happens in the processing layer.
        """
        subject = "Newsletter subscription"
        body = "I want to subscribe to your mailing list."
        
        result = is_colleague_signup_email(subject, body)
        
        assert result is True


class TestColleagueEmailExtraction:
    """Test extraction of colleague email from signup requests."""
    
    def test_extract_email_from_body(self):
        """Test extracting email address from body."""
        body = "I want to be added as a colleague: jane.doe@university.edu"
        
        # Simple regex extraction
        import re
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        matches = re.findall(email_pattern, body)
        
        assert len(matches) > 0
        assert "jane.doe@university.edu" in matches
    
    def test_extract_email_from_from_header(self):
        """Test extracting email from From header."""
        from_header = "Jane Doe <jane.doe@university.edu>"
        
        import re
        match = re.search(r'<([^>]+)>', from_header)
        if match:
            email = match.group(1)
        else:
            email = from_header
        
        assert "jane.doe@university.edu" in email


class TestColleagueRecipientDistinction:
    """Test distinguishing colleague vs self recipients."""
    
    def test_self_recipient_detection(self):
        """Test that self-emails are identified."""
        user_email = get_test_email()
        recipient_email = get_test_email()
        
        is_self = user_email.lower() == recipient_email.lower()
        
        assert is_self is True
    
    def test_colleague_recipient_detection(self):
        """Test that colleague emails are identified."""
        user_email = get_test_email()
        recipient_email = get_colleague_test_email("colleague")
        
        is_self = user_email.lower() == recipient_email.lower()
        
        assert is_self is False
    
    def test_case_insensitive_comparison(self):
        """Test case-insensitive email comparison."""
        user_email = get_test_email().replace("@", "@").title()  # Change case
        recipient_email = get_test_email().lower()
        
        is_self = user_email.lower() == recipient_email.lower()
        
        assert is_self is True


class TestColleagueAddedByTracking:
    """Test tracking how colleagues were added."""
    
    def test_manual_add_tracking(self):
        """Test that manually added colleagues are tracked."""
        colleague = ColleagueInfo(
            id="c-001",
            name="Jane",
            email=get_colleague_test_email("jane"),
            added_by="manual",
        )
        
        assert colleague.added_by == "manual"
    
    def test_email_add_tracking(self):
        """Test that email-added colleagues are tracked."""
        colleague = ColleagueInfo(
            id="c-002",
            name="Bob",
            email=get_colleague_test_email("bob"),
            added_by="email",
        )
        
        assert colleague.added_by == "email"


class TestOwnerAndColleagueIdenticalInterests:
    """
    Regression test: when owner and colleague share the exact same interests,
    a matching paper must be delivered to BOTH and a share record must be
    created for the colleague.
    """

    @pytest.fixture
    def shared_interests_paper(self):
        """A paper that matches the shared interests."""
        return {
            "arxiv_id": "2401.99999",
            "title": "Advances in Machine Learning and Transformer Models",
            "abstract": "We propose a novel transformer architecture for machine learning tasks.",
            "relevance_score": 0.92,
            "novelty_score": 0.85,
            "importance": "high",
            "categories": ["cs.LG", "cs.CL"],
            "authors": ["Alice Researcher", "Bob Scientist"],
            "link": "https://arxiv.org/abs/2401.99999",
            "explanation": "Directly relevant to machine learning and transformers.",
        }

    @pytest.fixture
    def identical_colleague(self):
        """A colleague with the same interests as the owner."""
        return {
            "id": "c-identical",
            "name": "Identical Colleague",
            "email": get_colleague_test_email("identical"),
            "topics": ["machine learning", "transformers", "NLP"],
            "sharing_preference": "immediate",
            "arxiv_categories_interest": ["cs.LG", "cs.CL"],
            "added_by": "manual",
            "auto_send_emails": True,
        }

    @pytest.fixture
    def sharing_policy(self):
        return {
            "importance_policies": {
                "high": {
                    "send_email": True,
                    "create_calendar_entry": False,
                    "add_to_reading_list": True,
                    "allow_colleague_sharing": True,
                    "priority_label": "urgent",
                },
            },
            "email_settings": {"enabled": True, "simulate_output": True,
                               "include_abstract": True, "digest_mode": False},
            "reading_list_settings": {"enabled": True},
            "colleague_sharing_settings": {
                "enabled": True,
                "respect_sharing_preferences": True,
                "simulate_output": True,
            },
        }

    # ------------------------------------------------------------------
    # 1. process_colleague_surplus: colleague gets a share action
    # ------------------------------------------------------------------
    def test_surplus_creates_share_action_for_identical_colleague(
        self, shared_interests_paper, identical_colleague, sharing_policy
    ):
        """When owner and colleague have identical interests, the colleague
        must receive a non-skip share action for the matching paper."""
        result = process_colleague_surplus(
            all_scored_papers=[shared_interests_paper],
            owner_paper_ids=[shared_interests_paper["arxiv_id"]],
            colleagues=[identical_colleague],
            delivery_policy=sharing_policy,
            researcher_name="Dr. Owner",
        )

        assert result["success"] is True

        actions = result["colleague_actions"]
        colleague_action = [
            a for a in actions
            if a["colleague_id"] == "c-identical"
            and a["paper_id"] == "2401.99999"
        ]
        assert len(colleague_action) == 1, "Colleague must have exactly one action for the paper"
        assert colleague_action[0]["action_type"] == "share_immediate", (
            f"Expected share_immediate, got {colleague_action[0]['action_type']}"
        )

    # ------------------------------------------------------------------
    # 2. Share file artifact carries paper_id and colleague_id
    # ------------------------------------------------------------------
    def test_surplus_share_file_has_paper_and_colleague_ids(
        self, shared_interests_paper, identical_colleague, sharing_policy
    ):
        """The generated share FileToWrite must contain paper_id and
        colleague_id so the DB handler can create a Share record."""
        result = process_colleague_surplus(
            all_scored_papers=[shared_interests_paper],
            owner_paper_ids=[shared_interests_paper["arxiv_id"]],
            colleagues=[identical_colleague],
            delivery_policy=sharing_policy,
            researcher_name="Dr. Owner",
        )

        share_files = [
            f for f in result["files_to_write"]
            if f["file_type"] == "share"
        ]
        assert len(share_files) >= 1, "At least one share file must be generated"

        sf = share_files[0]
        assert sf["paper_id"] == "2401.99999", "Share file must carry paper_id"
        assert sf["colleague_id"] == "c-identical", "Share file must carry colleague_id"
        assert sf["colleague_email"] == identical_colleague["email"], (
            "Share file must carry colleague_email"
        )

    # ------------------------------------------------------------------
    # 3. Owner email is generated via decide_delivery_action
    # ------------------------------------------------------------------
    def test_owner_email_triggered(
        self, shared_interests_paper, identical_colleague, sharing_policy
    ):
        """decide_delivery_action must produce an email action for the owner
        (the owner's paper delivery is handled per-paper, not via surplus)."""
        from src.tools.decide_delivery import decide_delivery_action

        result = decide_delivery_action(
            scored_paper=shared_interests_paper,
            delivery_policy=sharing_policy,
            colleagues=[identical_colleague],
            researcher_name="Dr. Owner",
            researcher_email="owner@example.com",
            skip_colleague_sharing=True,  # Normal agent flow
        )

        email_actions = [
            a for a in result.researcher_actions if a.action_type == "email"
        ]
        assert len(email_actions) >= 1, "Owner must receive an email action"

    # ------------------------------------------------------------------
    # 4. Colleague email is triggered (share artifact written)
    # ------------------------------------------------------------------
    def test_colleague_email_triggered_via_share_artifact(
        self, shared_interests_paper, identical_colleague, sharing_policy
    ):
        """After surplus processing, a share artifact must exist that will
        trigger an email to the colleague when saved to the DB."""
        result = process_colleague_surplus(
            all_scored_papers=[shared_interests_paper],
            owner_paper_ids=[shared_interests_paper["arxiv_id"]],
            colleagues=[identical_colleague],
            delivery_policy=sharing_policy,
            researcher_name="Dr. Owner",
        )

        share_files = [
            f for f in result["files_to_write"]
            if f["file_type"] == "share"
            and f.get("colleague_email") == identical_colleague["email"]
        ]
        assert len(share_files) >= 1, (
            "A share artifact targeting the colleague email must be generated"
        )

    # ------------------------------------------------------------------
    # 5. save_artifact_to_db creates Share record and sends email
    # ------------------------------------------------------------------
    def test_save_share_artifact_records_share_and_sends_email(
        self, shared_interests_paper, identical_colleague, sharing_policy
    ):
        """Mock the DB and SMTP layers and verify that save_artifact_to_db
        creates a Share record and attempts to send an email."""
        from src.db.data_service import save_artifact_to_db

        # Build a share artifact the same way process_colleague_surplus does
        result = process_colleague_surplus(
            all_scored_papers=[shared_interests_paper],
            owner_paper_ids=[shared_interests_paper["arxiv_id"]],
            colleagues=[identical_colleague],
            delivery_policy=sharing_policy,
            researcher_name="Dr. Owner",
        )
        share_files = [f for f in result["files_to_write"] if f["file_type"] == "share"]
        assert share_files, "Pre-condition: share file must exist"
        artifact = share_files[0]

        # Patch DB availability and SMTP to avoid real side effects
        with patch("src.db.data_service.is_db_available", return_value=True), \
             patch("src.db.data_service._get_default_user_id", return_value="00000000-0000-0000-0000-000000000001"), \
             patch("src.db.data_service.get_or_create_default_user", return_value={"email": "owner@test.com", "name": "Owner"}), \
             patch("src.db.data_service.get_db_session") as mock_session_ctx, \
             patch("src.db.postgres_store.PostgresStore") as MockStore:

            # Setup mock DB session and store
            mock_db = MagicMock()
            mock_session_ctx.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session_ctx.return_value.__exit__ = MagicMock(return_value=False)

            # Mock paper lookup — the first query().filter_by().first() returns the paper,
            # subsequent ones for Colleague lookup should return a colleague mock
            mock_paper = MagicMock()
            mock_paper.id = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
            mock_paper.title = shared_interests_paper["title"]
            mock_paper.url = shared_interests_paper.get("link", "")

            mock_colleague_row = MagicMock()
            mock_colleague_row.id = uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
            mock_colleague_row.email = identical_colleague["email"]
            mock_colleague_row.name = identical_colleague["name"]

            # Make filter_by return different mocks based on call order
            mock_db.query.return_value.filter_by.return_value.first.side_effect = [
                mock_paper,        # Paper lookup
                MagicMock(importance="high"),  # PaperView lookup
                mock_colleague_row,  # Colleague lookup
            ]

            mock_store = MockStore.return_value
            mock_store.create_share.return_value = {"id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"}

            # Patch _send_email_smtp at the import location used by data_service
            with patch("src.tools.decide_delivery._send_email_smtp", return_value=True) as mock_send:
                save_result = save_artifact_to_db(
                    file_type=artifact["file_type"],
                    file_path=artifact["file_path"],
                    content=artifact["content"],
                    paper_id=artifact.get("paper_id"),
                    colleague_id=artifact.get("colleague_id"),
                    description=artifact.get("description"),
                    triggered_by="agent",
                    colleague_email=artifact.get("colleague_email"),
                    colleague_name=artifact.get("colleague_name"),
                )

                assert save_result["success"] is True, f"save_artifact_to_db failed: {save_result}"
                assert save_result["type"] == "share"

                # Verify email was sent to the colleague
                mock_send.assert_called_once()
                call_kwargs = mock_send.call_args
                assert call_kwargs[1]["to_email"] == identical_colleague["email"], (
                    "Email must be sent to colleague's address"
                )

    # ------------------------------------------------------------------
    # 6. Full end-to-end: both owner and colleague receive deliveries
    # ------------------------------------------------------------------
    def test_end_to_end_both_receive_paper(
        self, shared_interests_paper, identical_colleague, sharing_policy
    ):
        """Simulate the full agent flow:
        1. decide_delivery_action for owner (per-paper call)
        2. process_colleague_surplus after owner selection
        Assert that owner has an email action AND colleague has a share action.
        """
        from src.tools.decide_delivery import decide_delivery_action

        # Step 1: Owner delivery (skip_colleague_sharing=True, as in agent)
        owner_result = decide_delivery_action(
            scored_paper=shared_interests_paper,
            delivery_policy=sharing_policy,
            colleagues=[identical_colleague],
            researcher_name="Dr. Owner",
            researcher_email="owner@example.com",
            skip_colleague_sharing=True,
        )
        owner_email_actions = [
            a for a in owner_result.researcher_actions if a.action_type == "email"
        ]
        assert len(owner_email_actions) >= 1, "Owner must get an email action"

        # Step 2: Colleague surplus
        surplus_result = process_colleague_surplus(
            all_scored_papers=[shared_interests_paper],
            owner_paper_ids=[shared_interests_paper["arxiv_id"]],
            colleagues=[identical_colleague],
            delivery_policy=sharing_policy,
            researcher_name="Dr. Owner",
        )
        colleague_shares = [
            a for a in surplus_result["colleague_actions"]
            if a["colleague_id"] == "c-identical"
            and a["action_type"] != "skip"
        ]
        assert len(colleague_shares) >= 1, "Colleague must get a share action"

        share_files = [
            f for f in surplus_result["files_to_write"]
            if f["file_type"] == "share"
            and f.get("colleague_id") == "c-identical"
        ]
        assert len(share_files) >= 1, "A share artifact must exist for the colleague"
        assert share_files[0]["paper_id"] == shared_interests_paper["arxiv_id"]


class TestColleagueWorkflowNegativeCases:
    """Test error handling in colleague workflows."""
    
    def test_empty_colleagues_list(self):
        """Test handling empty colleagues list."""
        papers = [{
            "arxiv_id": "2401.00001",
            "title": "Test Paper",
            "relevance_score": 0.9,
            "novelty_score": 0.8,
            "importance": "high",
        }]
        
        result = process_colleague_surplus(
            all_scored_papers=papers,
            owner_paper_ids=[],
            colleagues=[],
            delivery_policy={"colleague_sharing_settings": {"enabled": True}},
            researcher_name="Dr. Researcher",
        )
        
        assert result["success"] is True
        assert result["message"] == "No colleagues configured"
    
    def test_empty_papers_list(self):
        """Test handling empty papers list."""
        colleagues = [{
            "id": "c-001",
            "name": "Jane",
            "email": get_colleague_test_email("jane"),
            "auto_send_emails": True,
        }]
        
        result = process_colleague_surplus(
            all_scored_papers=[],
            owner_paper_ids=[],
            colleagues=colleagues,
            delivery_policy={"colleague_sharing_settings": {"enabled": True}},
            researcher_name="Dr. Researcher",
        )
        
        assert result["success"] is True
        assert result["paper_count"] == 0
    
    def test_invalid_colleague_data(self):
        """Test handling invalid colleague data."""
        with pytest.raises((ValueError, TypeError)):
            ColleagueInfo(
                id="c-001",
                name="Jane",
                # Missing required email field
            )


@pytest.mark.skipif(
    not is_live_tests_enabled(),
    reason="Live tests disabled. Set RUN_LIVE_TESTS=1 to enable."
)
class TestLiveColleagueWorkflow:
    """Live integration tests for colleague workflows."""
    
    @pytest.fixture
    def test_prefix(self):
        """Generate unique test prefix."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"[RP TEST] {timestamp}_{unique_id}"
    
    def test_self_email_colleague_signup(self, test_prefix):
        """Test colleague signup via self-email.
        
        ResearchPulse sends an email to itself simulating a colleague signup request.
        """
        from src.tools.decide_delivery import _send_email_smtp
        
        email_account = os.getenv("SMTP_USER", "")
        if not email_account:
            pytest.skip("No email account configured")
        
        # Use email+alias so it still arrives in the same inbox
        test_colleague_email = get_colleague_test_email("testcolleague")
        
        subject = f"{test_prefix} ResearchPulse colleague signup"
        body = f"""
        Hello,
        
        I want to be added as a colleague: {test_colleague_email}
        
        Please add me to your ResearchPulse colleague list so I can receive 
        research paper updates about machine learning and NLP.
        
        Thanks!
        """
        
        success = _send_email_smtp(
            to_email=email_account,
            subject=subject,
            body=body,
        )
        
        assert success is True
        
        # Verify it's detected as signup
        is_signup = is_colleague_signup_email(subject, body)
        assert is_signup is True
        
        # Verify email can be extracted
        import re
        emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', body)
        assert test_colleague_email in emails
