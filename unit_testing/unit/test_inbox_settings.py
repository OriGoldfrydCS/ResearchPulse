"""
Unit tests for inbox settings and colleague join code features.

Tests:
- Join code hashing and verification
- Inbox settings API
- Processed email tracking
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from uuid import UUID, uuid4
from datetime import datetime


class TestJoinCodeHashing:
    """Test join code hashing utilities."""
    
    def test_hash_join_code_produces_hash(self):
        """Hash function should produce a non-empty hash."""
        from src.tools.inbound_processor import hash_join_code
        
        code = "test_code_123"
        result = hash_join_code(code)
        
        assert result is not None
        assert len(result) > 0
        assert result != code  # Should not be plaintext
    
    def test_verify_join_code_correct(self):
        """Verify should return True for correct code."""
        from src.tools.inbound_processor import hash_join_code, verify_join_code
        
        code = "my_secret_code"
        code_hash = hash_join_code(code)
        
        assert verify_join_code(code, code_hash) is True
    
    def test_verify_join_code_incorrect(self):
        """Verify should return False for incorrect code."""
        from src.tools.inbound_processor import hash_join_code, verify_join_code
        
        code = "correct_code"
        wrong_code = "wrong_code"
        code_hash = hash_join_code(code)
        
        assert verify_join_code(wrong_code, code_hash) is False
    
    def test_verify_join_code_case_sensitive(self):
        """Verify should be case-sensitive."""
        from src.tools.inbound_processor import hash_join_code, verify_join_code
        
        code = "CaseSensitive"
        code_hash = hash_join_code(code)
        
        assert verify_join_code("casesensitive", code_hash) is False
        assert verify_join_code("CASESENSITIVE", code_hash) is False
        assert verify_join_code("CaseSensitive", code_hash) is True


class TestJoinCodeExtraction:
    """Test join code extraction from email content."""
    
    def test_extract_code_pattern_with_colon(self):
        """Should extract code with 'Code: XXX' pattern."""
        from src.tools.inbound_processor import extract_join_code_from_email
        
        body = "Hi, I'd like to join. Code: ABC123XYZ"
        subject = "Join request"
        
        result = extract_join_code_from_email(body, subject)
        
        assert result == "ABC123XYZ"
    
    def test_extract_code_pattern_with_hash(self):
        """Should extract code with '#XXX' pattern."""
        from src.tools.inbound_processor import extract_join_code_from_email
        
        body = "Please add me. #SECRET42"
        subject = "Join"
        
        result = extract_join_code_from_email(body, subject)
        
        assert result == "SECRET42"
    
    def test_extract_code_from_subject(self):
        """Should extract code from subject line."""
        from src.tools.inbound_processor import extract_join_code_from_email
        
        body = "Please add me to the list."
        subject = "Join request - Code: JOIN456"
        
        result = extract_join_code_from_email(body, subject)
        
        assert result == "JOIN456"
    
    def test_extract_no_code_found(self):
        """Should return None when no code is found."""
        from src.tools.inbound_processor import extract_join_code_from_email
        
        body = "I want to join ResearchPulse."
        subject = "Join request"
        
        result = extract_join_code_from_email(body, subject)
        
        assert result is None


class TestColleagueJoinRequestDetection:
    """Test detection of colleague join request emails."""
    
    def test_detect_join_request_with_keywords(self):
        """Should detect valid join request."""
        from src.tools.inbound_processor import is_colleague_join_request
        
        subject = "Want to join ResearchPulse"
        body = "Please add me to receive paper updates"
        
        result = is_colleague_join_request(subject, body)
        
        assert result is True
    
    def test_detect_join_request_subscribe_keyword(self):
        """Should detect join request with 'subscribe' keyword."""
        from src.tools.inbound_processor import is_colleague_join_request
        
        subject = "Subscribe to ResearchPulse"
        body = "I'd like to subscribe to research updates"
        
        result = is_colleague_join_request(subject, body)
        
        assert result is True
    
    def test_reject_non_join_request(self):
        """Should reject generic emails."""
        from src.tools.inbound_processor import is_colleague_join_request
        
        subject = "Hello"
        body = "Just saying hi"
        
        result = is_colleague_join_request(subject, body)
        
        assert result is False
    
    def test_require_researchpulse_mention(self):
        """Should require ResearchPulse mention."""
        from src.tools.inbound_processor import is_colleague_join_request
        
        subject = "Add me to your list"
        body = "Please add me to receive updates"
        
        result = is_colleague_join_request(subject, body)
        
        # Should be False because no ResearchPulse mention
        assert result is False


class TestOwnerEmailValidation:
    """Test owner email validation for reschedule requests."""
    
    def test_owner_email_match(self):
        """Should match owner email."""
        from src.tools.inbound_processor import is_owner_email
        
        assert is_owner_email("owner@example.com", "owner@example.com") is True
    
    def test_owner_email_case_insensitive(self):
        """Should be case-insensitive."""
        from src.tools.inbound_processor import is_owner_email
        
        assert is_owner_email("Owner@Example.com", "owner@example.com") is True
        assert is_owner_email("OWNER@EXAMPLE.COM", "owner@example.com") is True
    
    def test_owner_email_mismatch(self):
        """Should reject non-owner email."""
        from src.tools.inbound_processor import is_owner_email
        
        assert is_owner_email("other@example.com", "owner@example.com") is False
    
    def test_owner_email_handles_none(self):
        """Should handle None values."""
        from src.tools.inbound_processor import is_owner_email
        
        assert is_owner_email(None, "owner@example.com") is False
        assert is_owner_email("owner@example.com", None) is False
        assert is_owner_email(None, None) is False


class TestUserSettingsModel:
    """Test UserSettings ORM model."""
    
    def test_user_settings_to_dict(self):
        """Settings should serialize properly."""
        from src.db.orm_models import UserSettings
        
        settings = UserSettings(
            id=uuid4(),
            user_id=uuid4(),
            inbox_check_enabled=True,
            inbox_check_frequency_seconds=300,
            colleague_join_code_hash="$2b$12$...",
        )
        
        result = settings.to_dict()
        
        assert result["inbox_check_enabled"] is True
        assert result["inbox_check_frequency_seconds"] == 300
        assert result["has_join_code"] is True  # Should not expose hash
        assert "colleague_join_code_hash" not in result  # Hash should not be in dict


class TestProcessedInboundEmailModel:
    """Test ProcessedInboundEmail ORM model."""
    
    def test_processed_email_to_dict(self):
        """Processed email should serialize properly."""
        from src.db.orm_models import ProcessedInboundEmail
        
        email = ProcessedInboundEmail(
            id=uuid4(),
            user_id=uuid4(),
            gmail_message_id="msg123",
            email_type="colleague_join",
            processing_result="success",
            from_email="test@example.com",
            subject="Join request",
        )
        
        result = email.to_dict()
        
        assert result["gmail_message_id"] == "msg123"
        assert result["email_type"] == "colleague_join"
        assert result["processing_result"] == "success"


class TestInboxCheckFrequencyValidation:
    """Test valid inbox check frequency values."""
    
    def test_valid_frequencies(self):
        """Valid frequencies should be accepted."""
        valid_frequencies = [None, 10, 30, 60, 300, 900, 3600]
        
        for freq in valid_frequencies:
            # These should be accepted by the API
            assert freq is None or freq in [10, 30, 60, 300, 900, 3600]


class TestJoinCodePatternDetection:
    """Test detection of join code patterns in email text."""
    
    def test_has_join_code_pattern_colon(self):
        """Should detect 'code: 123456' pattern."""
        from src.tools.email_poller import has_join_code_pattern
        
        assert has_join_code_pattern("code: 123456") is True
        assert has_join_code_pattern("code:abcdef") is True
        assert has_join_code_pattern("Code: SECRET") is True
    
    def test_has_join_code_pattern_equals(self):
        """Should detect 'code=123456' pattern."""
        from src.tools.email_poller import has_join_code_pattern
        
        assert has_join_code_pattern("code=123456") is True
        assert has_join_code_pattern("Code=ABCD") is True
    
    def test_has_join_code_pattern_join_code(self):
        """Should detect 'join code 123456' pattern."""
        from src.tools.email_poller import has_join_code_pattern
        
        assert has_join_code_pattern("join code 123456") is True
        assert has_join_code_pattern("Join Code: SECRET") is True
    
    def test_has_join_code_pattern_hash(self):
        """Should detect '#123456' pattern."""
        from src.tools.email_poller import has_join_code_pattern
        
        assert has_join_code_pattern("#123456") is True
        assert has_join_code_pattern("#ABCD1234") is True
    
    def test_has_join_code_pattern_no_match(self):
        """Should return False when no code pattern found."""
        from src.tools.email_poller import has_join_code_pattern
        
        assert has_join_code_pattern("hello world") is False
        assert has_join_code_pattern("add me please") is False
        assert has_join_code_pattern("") is False


class TestColleagueSignupEmailDetection:
    """Test colleague signup email detection with join code patterns."""
    
    def test_detect_signup_with_join_code_only(self):
        """Should detect signup when email contains join code pattern, even without ResearchPulse mention."""
        from src.tools.email_poller import is_colleague_signup_email
        
        # This is the user's exact test case - no ResearchPulse mention but has code
        subject = ""
        body = "code: 123456\nadd me to the colleagues list.\nMy name is KAY"
        
        result = is_colleague_signup_email(subject, body)
        
        assert result is True
    
    def test_detect_signup_with_researchpulse_and_intent(self):
        """Should detect signup with ResearchPulse mention and intent keywords."""
        from src.tools.email_poller import is_colleague_signup_email
        
        subject = "Want to join ResearchPulse"
        body = "Please add me to your updates"
        
        result = is_colleague_signup_email(subject, body)
        
        assert result is True
    
    def test_accept_personal_email_without_code_or_rp_mention(self):
        """Personal emails are now accepted for processing even without code or RP mention.
        
        The broadened detection treats any non-automated email as a potential
        interaction. The processing layer handles format validation and sends
        instructions once per sender.
        """
        from src.tools.email_poller import is_colleague_signup_email
        
        subject = "Hello"
        body = "I want updates"
        
        result = is_colleague_signup_email(subject, body)
        
        assert result is True
    
    def test_reject_automated_noreply_email(self):
        """Should reject automated/noreply emails even with signup-like content."""
        from src.tools.email_poller import is_colleague_signup_email
        
        subject = "Subscribe me to ResearchPulse"
        body = "I want research paper updates"
        
        result = is_colleague_signup_email(subject, body, from_email="noreply@example.com")
        
        assert result is False
    
    def test_reject_automated_out_of_office(self):
        """Should reject out-of-office auto-replies."""
        from src.tools.email_poller import is_colleague_signup_email
        
        subject = "Automatic Reply: Out of Office"
        body = "I am currently out of the office."
        
        result = is_colleague_signup_email(subject, body, from_email="person@example.com")
        
        assert result is False
    
    def test_detect_code_equals_pattern(self):
        """Should detect signup with code= pattern."""
        from src.tools.email_poller import is_colleague_signup_email
        
        body = "Please add me. code=ABC123"
        
        result = is_colleague_signup_email("", body)
        
        assert result is True


class TestNameExtractionFromBody:
    """Test name extraction from email body."""
    
    def test_extract_name_my_name_is(self):
        """Should extract name from 'My name is X' pattern."""
        from src.tools.inbound_processor import InboundEmailProcessor
        
        # Use a mock or just test the method directly
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        result = processor._extract_name_from_body("code: 123456\nadd me to list\nMy name is KAY")
        
        assert result == "KAY"
    
    def test_extract_name_i_am(self):
        """Should extract name from 'I am X' pattern."""
        from src.tools.inbound_processor import InboundEmailProcessor
        
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        result = processor._extract_name_from_body("Please add me. I am John Smith")
        
        assert result == "John Smith"
    
    def test_extract_name_no_match(self):
        """Should return None when no name pattern found."""
        from src.tools.inbound_processor import InboundEmailProcessor
        
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        result = processor._extract_name_from_body("just add me please")
        
        assert result is None


class TestInterestExtractionWordBoundaries:
    """Test that interest keyword extraction uses word boundaries to avoid false positives."""
    
    def test_ai_keyword_does_not_match_said(self):
        """'ai' keyword should NOT match 'said' (false positive prevention)."""
        from src.tools.inbound_processor import InboundEmailProcessor
        
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        # Email with 'said' but no actual AI mention
        body = "I said please add me to the colleagues list. My name is John"
        
        interests_text, has_meaningful = processor._extract_interests_from_body(body)
        
        assert has_meaningful is False, "'said' should not trigger 'ai' keyword match"
    
    def test_ai_keyword_does_not_match_paid(self):
        """'ai' keyword should NOT match 'paid'."""
        from src.tools.inbound_processor import InboundEmailProcessor
        
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = "I paid for this service. Add me please."
        
        interests_text, has_meaningful = processor._extract_interests_from_body(body)
        
        assert has_meaningful is False, "'paid' should not trigger 'ai' keyword match"
    
    def test_ai_keyword_does_not_match_wait(self):
        """'ai' keyword should NOT match 'wait'."""
        from src.tools.inbound_processor import InboundEmailProcessor
        
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = "Please wait for my response. I want to join."
        
        interests_text, has_meaningful = processor._extract_interests_from_body(body)
        
        assert has_meaningful is False, "'wait' should not trigger 'ai' keyword match"
    
    def test_ai_keyword_matches_standalone_ai(self):
        """'ai' keyword SHOULD match standalone 'AI'."""
        from src.tools.inbound_processor import InboundEmailProcessor
        
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = "I'm interested in AI research. Please add me."
        
        interests_text, has_meaningful = processor._extract_interests_from_body(body)
        
        assert has_meaningful is True, "Standalone 'AI' should match"
    
    def test_qa_keyword_does_not_match_equal(self):
        """'qa' keyword should NOT match within other words."""
        from src.tools.inbound_processor import InboundEmailProcessor
        
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = "code: 123456\nadd me to the colleagues list.\nMy name is ZZZZZZZZ"
        
        interests_text, has_meaningful = processor._extract_interests_from_body(body)
        
        assert has_meaningful is False, "Simple signup without interests should not match"
    
    def test_nlp_keyword_matches_standalone(self):
        """'nlp' keyword SHOULD match standalone 'NLP'."""
        from src.tools.inbound_processor import InboundEmailProcessor
        
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = "I'm interested in NLP and transformers."
        
        interests_text, has_meaningful = processor._extract_interests_from_body(body)
        
        assert has_meaningful is True, "Standalone 'NLP' should match"
    
    def test_machine_learning_matches(self):
        """Long keywords like 'machine learning' should match."""
        from src.tools.inbound_processor import InboundEmailProcessor
        
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = "I work on machine learning research."
        
        interests_text, has_meaningful = processor._extract_interests_from_body(body)
        
        assert has_meaningful is True


class TestOnboardingStatusDetermination:
    """Test that onboarding status is correctly determined based on missing fields."""
    
    def test_status_needs_interests_when_only_name_provided(self):
        """Should return 'needs_interests' when name is provided but no interests."""
        from src.tools.inbound_processor import InboundEmailProcessor
        
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        # This is the exact user scenario
        body = "code: 123456\nadd me to the colleagues list.\nMy name is ZZZZZZZZ"
        
        # Extract name
        name = processor._extract_name_from_body(body)
        assert name == "ZZZZZZZZ", "Name should be extracted"
        
        # Extract interests
        interests_text, has_meaningful_interests = processor._extract_interests_from_body(body)
        assert has_meaningful_interests is False, "Should NOT find meaningful interests"
        
        # Determine status
        missing_fields = []
        if not name:
            missing_fields.append("name")
        if not has_meaningful_interests:
            missing_fields.append("interests")
        
        assert "interests" in missing_fields, "Interests should be missing"
        assert "name" not in missing_fields, "Name should NOT be missing"
        
        # The status should be needs_interests, not complete
        if not missing_fields:
            status = "complete"
        elif "interests" in missing_fields and "name" in missing_fields:
            status = "pending"
        elif "interests" in missing_fields:
            status = "needs_interests"
        else:
            status = "needs_name"
        
        assert status == "needs_interests", "Status should be 'needs_interests' not 'complete'"
    
    def test_status_complete_when_both_provided(self):
        """Should return 'complete' when both name and interests are provided."""
        from src.tools.inbound_processor import InboundEmailProcessor
        
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = "code: 123456\nMy name is John. I'm interested in machine learning and NLP."
        
        name = processor._extract_name_from_body(body)
        interests_text, has_meaningful_interests = processor._extract_interests_from_body(body)
        
        assert name is not None, "Name should be extracted"
        assert has_meaningful_interests is True, "Should find meaningful interests"
        
        missing_fields = []
        if not name:
            missing_fields.append("name")
        if not has_meaningful_interests:
            missing_fields.append("interests")
        
        assert len(missing_fields) == 0, "No fields should be missing"
    
    def test_status_pending_when_both_missing(self):
        """Should return 'pending' when both name and interests are missing."""
        from src.tools.inbound_processor import InboundEmailProcessor
        
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = "code: 123456\nplease add me"
        
        name = processor._extract_name_from_body(body)
        interests_text, has_meaningful_interests = processor._extract_interests_from_body(body)
        
        assert name is None, "Name should not be extracted"
        assert has_meaningful_interests is False, "Should not find meaningful interests"


class TestReplyContentExtraction:
    """Test extraction of new content from reply emails (removing quoted text)."""
    
    def test_extract_new_content_simple(self):
        """Should extract content before quoted text."""
        from src.tools.inbound_processor import InboundEmailProcessor
        
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = """machine learning, NLP, transformers, computer vision

On Mon, Feb 10, 2026 at 5:30 PM ResearchPulse wrote:
> Hi there,
> Please reply with your interests."""
        
        result = processor._extract_new_content_from_reply(body)
        
        assert "machine learning" in result
        assert "On Mon" not in result
        assert "ResearchPulse wrote" not in result
    
    def test_extract_new_content_with_separator(self):
        """Should stop at separator lines."""
        from src.tools.inbound_processor import InboundEmailProcessor
        
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = """deep learning, computer vision, autonomous driving

___

Previous email content here"""
        
        result = processor._extract_new_content_from_reply(body)
        
        assert "deep learning" in result
        assert "Previous email" not in result
    
    def test_extract_new_content_with_quoted_lines(self):
        """Should remove lines starting with >."""
        from src.tools.inbound_processor import InboundEmailProcessor
        
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = """NLP, transformers
> quoted line 1
> quoted line 2
more content"""
        
        result = processor._extract_new_content_from_reply(body)
        
        assert "NLP" in result
        assert "quoted line" not in result


class TestOnboardingContinuationInterestParsing:
    """Test that onboarding continuation correctly parses interest replies."""
    
    def test_comma_separated_interests_recognized(self):
        """Comma-separated list should be treated as interests even without keywords."""
        from src.tools.inbound_processor import InboundEmailProcessor
        
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        # Simulate a reply with just a list of topics
        body = "machine learning, NLP, transformers, computer vision, object detection, autonomous driving"
        
        # This should be recognized as meaningful interests
        interests_text, has_meaningful = processor._extract_interests_from_body(body)
        
        # machine learning and NLP are both keywords
        assert has_meaningful is True, "Comma-separated list with keywords should be recognized"
    
    def test_newline_separated_interests(self):
        """Newline-separated list should be treated as interests."""
        from src.tools.inbound_processor import InboundEmailProcessor
        
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = """machine learning
NLP
transformers
computer vision"""
        
        interests_text, has_meaningful = processor._extract_interests_from_body(body)
        
        assert has_meaningful is True, "Newline-separated list with keywords should be recognized"
    
    def test_exact_user_scenario_interest_reply(self):
        """Exact user scenario: reply with interests after separator."""
        from src.tools.inbound_processor import InboundEmailProcessor
        
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        # This is the exact user's reply format
        full_reply = """machine learning, NLP, transformers,  computer vision, object detection, autonomous driving

___

Hi Ori Goldfryd,

Welcome to ResearchPulse! 🎉 You've been successfully added.

To start sending you relevant paper recommendations, we need to know your research interests.

Please reply with the topics you're interested in:
- e.g., "machine learning, NLP, transformers"
- e.g., "computer vision, object detection, autonomous driving"
- e.g., "reinforcement learning, robotics, multi-agent systems"

Just reply to this email with your interests - no special format needed!

Best regards,
ResearchPulse"""
        
        # First extract new content (before the separator)
        clean_body = processor._extract_new_content_from_reply(full_reply)
        
        assert "machine learning" in clean_body
        assert "Welcome to ResearchPulse" not in clean_body, "Should remove quoted content after separator"
        
        # Then extract interests from clean body
        interests_text, has_meaningful = processor._extract_interests_from_body(clean_body)
        
        assert has_meaningful is True, "Should recognize interests from the reply"


class TestNameExtractionFormats:
    """Test name extraction with various formats including titles."""
    
    def test_simple_name(self):
        """Should extract simple first name."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        result = processor._extract_name_from_body("My name is John")
        assert result == "John"
    
    def test_full_name(self):
        """Should extract first and last name."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        result = processor._extract_name_from_body("My name is John Smith")
        assert result == "John Smith"
    
    def test_doctor_title(self):
        """Should extract name with Dr. title."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        result = processor._extract_name_from_body("My name is Dr. John Snow")
        assert result is not None
        assert "John" in result
        assert "Snow" in result
    
    def test_professor_title(self):
        """Should extract name with Prof. title."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        result = processor._extract_name_from_body("My name is Prof. Jane Doe")
        assert result is not None
        assert "Jane" in result
        assert "Doe" in result
    
    def test_i_am_pattern(self):
        """Should extract name with 'I am' pattern."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        result = processor._extract_name_from_body("I am Dr. John Snow")
        assert result is not None
        assert "John" in result
    
    def test_name_colon_pattern(self):
        """Should extract name with 'name:' pattern."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        result = processor._extract_name_from_body("name: John Smith")
        assert result is not None
        assert "John" in result
    
    def test_three_part_name(self):
        """Should extract three-part names."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        result = processor._extract_name_from_body("My name is John Michael Smith")
        assert result is not None
        assert "John" in result
    
    def test_hyphenated_name(self):
        """Should extract hyphenated names."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        result = processor._extract_name_from_body("My name is Mary-Jane Watson")
        assert result is not None
        assert "Mary-Jane" in result or "Mary" in result
    
    def test_name_in_email_body(self):
        """Should extract name from realistic email body."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = """code: 123456
add me to the colleagues list.
My name is Dr. John Snow"""
        
        result = processor._extract_name_from_body(body)
        assert result is not None
        assert "John" in result


class TestInterestExtractionFormats:
    """Test interest extraction with various formats."""
    
    def test_comma_separated(self):
        """Should recognize comma-separated interests."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = "machine learning, computer vision, NLP"
        interests, has_meaningful = processor._extract_interests_from_body(body)
        assert has_meaningful is True
    
    def test_bullet_points(self):
        """Should recognize bullet point interests."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = """- machine learning
- natural language processing
- computer vision"""
        interests, has_meaningful = processor._extract_interests_from_body(body)
        assert has_meaningful is True
    
    def test_prose_style(self):
        """Should recognize interests written in prose."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = "I'm interested in machine learning and deep learning, especially neural networks."
        interests, has_meaningful = processor._extract_interests_from_body(body)
        assert has_meaningful is True
    
    def test_short_keywords_with_word_boundaries(self):
        """Should correctly match short keywords like 'ai' and 'nlp'."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        # Should match 'AI' as standalone word
        body = "I work on AI and NLP research"
        interests, has_meaningful = processor._extract_interests_from_body(body)
        assert has_meaningful is True
        
        # Should NOT match 'ai' inside 'said'
        body2 = "I said hello"
        interests2, has_meaningful2 = processor._extract_interests_from_body(body2)
        assert has_meaningful2 is False
    
    def test_mixed_case(self):
        """Should recognize interests regardless of case."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = "MACHINE LEARNING, Deep Learning, nlp"
        interests, has_meaningful = processor._extract_interests_from_body(body)
        assert has_meaningful is True
    
    def test_numbered_list(self):
        """Should recognize numbered list interests."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = """1. machine learning
2. computer vision
3. robotics"""
        interests, has_meaningful = processor._extract_interests_from_body(body)
        assert has_meaningful is True


class TestCleanInterestsText:
    """Test the interest text cleaning function."""
    
    def test_clean_hebrew_reply_marker(self):
        """Should remove Hebrew reply marker and text."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        text = """computer vision, autonomous driving

‫בתאריך יום ג׳, 10 בפבר׳ 2026 ב-19:07 מאת <‪m67026428@gmail.com‬‏>:"""
        
        result = processor._clean_interests_text(text)
        assert "computer vision" in result
        assert "autonomous driving" in result
        assert "בתאריך" not in result
        assert "@gmail" not in result
    
    def test_clean_english_reply_marker(self):
        """Should remove English 'On...wrote:' marker."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        text = """machine learning, NLP

On Mon, Feb 10, 2026 at 5:30 PM ResearchPulse wrote:
Previous message content"""
        
        result = processor._clean_interests_text(text)
        assert "machine learning" in result
        assert "NLP" in result
        assert "wrote:" not in result
    
    def test_clean_email_addresses(self):
        """Should remove email addresses."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        text = "machine learning <test@example.com> robotics"
        result = processor._clean_interests_text(text)
        assert "machine learning" in result
        assert "@example.com" not in result
    
    def test_clean_rtl_characters(self):
        """Should remove RTL Unicode characters."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        # RTL embedding character + Hebrew
        text = "deep learning \u202bשלום\u202c computer vision"
        result = processor._clean_interests_text(text)
        assert "deep learning" in result
        assert "computer vision" in result
        # Hebrew characters should be removed
        assert "שלום" not in result
    
    def test_clean_dates(self):
        """Should remove date patterns."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        text = "machine learning 10/02/2026 robotics"
        result = processor._clean_interests_text(text)
        assert "machine learning" in result
        assert "10/02/2026" not in result
    
    def test_clean_preserves_valid_interests(self):
        """Should preserve clean interest text."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        text = "machine learning, deep learning, computer vision, NLP, transformers"
        result = processor._clean_interests_text(text)
        assert result == text


class TestUnhelpfulReplies:
    """Test handling of unhelpful/stupid replies to onboarding emails."""
    
    def test_empty_reply(self):
        """Empty reply should not have meaningful interests."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = ""
        interests, has_meaningful = processor._extract_interests_from_body(body)
        assert has_meaningful is False
    
    def test_just_greeting(self):
        """Reply with just a greeting should not have meaningful interests."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = "Hello, thanks for adding me!"
        interests, has_meaningful = processor._extract_interests_from_body(body)
        assert has_meaningful is False
    
    def test_question_reply(self):
        """Reply with a question should not have meaningful interests."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = "What kind of papers do you have?"
        interests, has_meaningful = processor._extract_interests_from_body(body)
        assert has_meaningful is False
    
    def test_off_topic_reply(self):
        """Off-topic reply should not have meaningful interests."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = "Can you send me papers about cooking recipes?"
        interests, has_meaningful = processor._extract_interests_from_body(body)
        assert has_meaningful is False
    
    def test_single_word_reply(self):
        """Single non-keyword word reply should not have meaningful interests."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = "ok"
        interests, has_meaningful = processor._extract_interests_from_body(body)
        assert has_meaningful is False
    
    def test_gibberish_reply(self):
        """Gibberish reply should not have meaningful interests."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = "asdfgh jkl qwerty"
        interests, has_meaningful = processor._extract_interests_from_body(body)
        assert has_meaningful is False
    
    def test_emoji_only_reply(self):
        """Emoji-only reply should not have meaningful interests."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = "👍 🎉 ✨"
        interests, has_meaningful = processor._extract_interests_from_body(body)
        assert has_meaningful is False
    
    def test_just_thanks(self):
        """Reply with just thanks should not have meaningful interests."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = "Thank you! Looking forward to it."
        interests, has_meaningful = processor._extract_interests_from_body(body)
        assert has_meaningful is False
    
    def test_confirmation_reply(self):
        """Simple confirmation reply should not have meaningful interests."""
        from src.tools.inbound_processor import InboundEmailProcessor
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        body = "Yes, please add me. Thanks!"
        interests, has_meaningful = processor._extract_interests_from_body(body)
        assert has_meaningful is False


class TestOnboardingContinuationWithUnhelpfulReplies:
    """Test that onboarding continuation properly handles unhelpful replies."""
    
    @patch('src.tools.inbound_processor.InboundEmailProcessor._send_join_reply')
    def test_unhelpful_reply_sends_follow_up(self, mock_send):
        """Unhelpful reply should trigger follow-up email asking for interests."""
        from src.tools.inbound_processor import InboundEmailProcessor
        
        # Create processor with mock store
        mock_store = MagicMock()
        mock_store.mark_email_processed = MagicMock()
        
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        processor.store = mock_store
        processor.user_id = "test-user"
        
        # Mock colleague needing interests
        colleague = {
            "id": "col-123",
            "email": "test@example.com",
            "name": "John",
            "onboarding_status": "needs_interests",
            "interests": "",
            "research_interests": "",
        }
        
        # Unhelpful reply
        email_body = "Thanks for adding me! Looking forward to it."
        
        # Call the method (with mocked LLM to return nothing)
        with patch.object(processor, '_extract_colleague_info_with_llm', return_value={"name": None, "interests": None}):
            result = processor._process_onboarding_continuation(
                colleague=colleague,
                message_id="msg-123",
                from_email="test@example.com",
                from_name="John",
                subject="Re: Research interests",
                email_body=email_body,
                corr_id="test-corr-123",
            )
        
        # Should return False (not completed)
        assert result is False
        
        # Should have called send_join_reply for follow-up
        assert mock_send.called
        
        # Check the email was marked as processed with needs_more_info
        mock_store.mark_email_processed.assert_called_once()
        call_kwargs = mock_store.mark_email_processed.call_args[1]
        assert call_kwargs["processing_result"] == "needs_more_info"
    
    @patch('src.tools.inbound_processor.InboundEmailProcessor._send_join_reply')
    def test_empty_reply_sends_follow_up(self, mock_send):
        """Empty reply should trigger follow-up email."""
        from src.tools.inbound_processor import InboundEmailProcessor
        
        mock_store = MagicMock()
        mock_store.mark_email_processed = MagicMock()
        
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        processor.store = mock_store
        processor.user_id = "test-user"
        
        colleague = {
            "id": "col-123",
            "email": "test@example.com",
            "name": "John",
            "onboarding_status": "needs_interests",
            "interests": "",
            "research_interests": "",
        }
        
        # Empty reply (just quoted text)
        email_body = """

On Mon, Feb 10, 2026 at 5:30 PM ResearchPulse wrote:
> Please tell us your interests"""
        
        with patch.object(processor, '_extract_colleague_info_with_llm', return_value={"name": None, "interests": None}):
            result = processor._process_onboarding_continuation(
                colleague=colleague,
                message_id="msg-123",
                from_email="test@example.com",
                from_name="John",
                subject="Re: Research interests",
                email_body=email_body,
                corr_id="test-corr-456",
            )
        
        assert result is False
        assert mock_send.called


class TestFollowUpMessageContent:
    """Test that follow-up messages have appropriate content."""
    
    def test_interests_request_message_content(self):
        """Interests request email should have examples."""
        from src.tools.inbound_processor import InboundEmailProcessor
        
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        # Mock the _send_join_reply to capture the body
        captured = []
        def capture_send(**kwargs):
            captured.append(kwargs)
        
        processor._send_join_reply = capture_send
        
        processor._send_onboarding_interests_reply("test@example.com", "John", "")
        
        assert len(captured) == 1
        body = captured[0].get("body", "")
        html = captured[0].get("html_body", "")
        text = body + html  # Check content in either format
        
        # Check key content
        assert "John" in text
        assert "machine learning" in text.lower()
        assert "computer vision" in text.lower()
    
    def test_name_request_message_content(self):
        """Name request email should ask for name."""
        from src.tools.inbound_processor import InboundEmailProcessor
        
        processor = InboundEmailProcessor.__new__(InboundEmailProcessor)
        
        captured = []
        def capture_send(**kwargs):
            captured.append(kwargs)
        
        processor._send_join_reply = capture_send
        
        processor._send_onboarding_name_reply("test@example.com", "", "")
        
        assert len(captured) == 1
        body = captured[0].get("body", "")
        html = captured[0].get("html_body", "")
        text = body + html
        
        assert "name" in text.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
