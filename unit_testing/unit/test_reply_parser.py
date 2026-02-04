"""
Unit tests for the email reply parser.

Tests the rule-based parsing of email replies to calendar invitations,
including detection of reschedule, accept, decline, and other intents.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from src.agent.reply_parser import (
    parse_reply,
    parse_reply_rules,
    ParsedReply,
    ReplyIntent,
    _extract_datetime_from_text,
    _convert_to_24h,
)


class TestReplyIntentEnum:
    """Test the ReplyIntent enumeration."""
    
    def test_all_intents_exist(self):
        """Verify all expected intents are defined."""
        expected = ["RESCHEDULE", "ACCEPT", "DECLINE", "CANCEL", "QUESTION", "OTHER", "UNKNOWN"]
        for intent in expected:
            assert hasattr(ReplyIntent, intent)
    
    def test_intent_values(self):
        """Verify intent values are lowercase strings."""
        assert ReplyIntent.RESCHEDULE.value == "reschedule"
        assert ReplyIntent.ACCEPT.value == "accept"
        assert ReplyIntent.DECLINE.value == "decline"
        assert ReplyIntent.CANCEL.value == "cancel"
        assert ReplyIntent.QUESTION.value == "question"
        assert ReplyIntent.OTHER.value == "other"
        assert ReplyIntent.UNKNOWN.value == "unknown"


class TestParsedReplyDataclass:
    """Test the ParsedReply dataclass."""
    
    def test_create_parsed_reply(self):
        """Test creating a ParsedReply instance."""
        reply = ParsedReply(
            intent=ReplyIntent.RESCHEDULE,
            extracted_datetime=datetime(2026, 2, 10, 14, 0),
            extracted_datetime_text="February 10 at 2pm",
            confidence_score=0.85,
            reason="Found datetime in text",
            original_text="Let's move it to February 10 at 2pm",
        )
        
        assert reply.intent == ReplyIntent.RESCHEDULE
        assert reply.extracted_datetime == datetime(2026, 2, 10, 14, 0)
        assert reply.extracted_datetime_text == "February 10 at 2pm"
        assert reply.confidence_score == 0.85
        assert "datetime" in reply.reason.lower()
        assert "February 10" in reply.original_text


class TestRescheduleIntentParsing:
    """Test parsing of reschedule requests."""
    
    def test_reschedule_with_tomorrow(self):
        """Test parsing 'tomorrow at Xpm' format."""
        result = parse_reply_rules("This time doesn't work. Can we do tomorrow at 2pm instead?")
        
        assert result.intent == ReplyIntent.RESCHEDULE
        assert result.extracted_datetime is not None
        assert result.extracted_datetime.hour == 14
        assert result.confidence_score >= 0.6
    
    def test_reschedule_with_next_weekday(self):
        """Test parsing 'next Monday at Xpm' format."""
        result = parse_reply_rules("Please reschedule to next Monday at 10am")
        
        assert result.intent == ReplyIntent.RESCHEDULE
        assert result.extracted_datetime is not None
        assert result.extracted_datetime.hour == 10
    
    def test_reschedule_with_month_day(self):
        """Test parsing 'February 10 at 3pm' format."""
        result = parse_reply_rules("Move it to February 10 at 3pm")
        
        assert result.intent == ReplyIntent.RESCHEDULE
        assert result.extracted_datetime is not None
        assert result.extracted_datetime.month == 2
        assert result.extracted_datetime.day == 10
        assert result.extracted_datetime.hour == 15
    
    def test_reschedule_with_numeric_date(self):
        """Test parsing '2/15 at 4pm' format."""
        result = parse_reply_rules("Can't make it, move to 2/15 at 4pm")
        
        assert result.intent == ReplyIntent.RESCHEDULE
        assert result.extracted_datetime is not None
        assert result.extracted_datetime.month == 2
        assert result.extracted_datetime.day == 15
        assert result.extracted_datetime.hour == 16
    
    def test_reschedule_keywords_without_datetime(self):
        """Test reschedule intent detected even without specific datetime."""
        result = parse_reply_rules("This time doesn't work for me. Can we change it?")
        
        assert result.intent == ReplyIntent.RESCHEDULE
        assert result.extracted_datetime is None
        assert result.confidence_score < 0.6  # Lower confidence without datetime
    
    def test_reschedule_how_about(self):
        """Test 'how about' phrasing for reschedule."""
        result = parse_reply_rules("How about next Tuesday at 11am?")
        
        assert result.intent == ReplyIntent.RESCHEDULE
        assert result.extracted_datetime is not None


class TestAcceptIntentParsing:
    """Test parsing of acceptance replies."""
    
    def test_accept_looks_good(self):
        """Test 'looks good' acceptance."""
        result = parse_reply_rules("Looks good, I'll be there!")
        
        assert result.intent == ReplyIntent.ACCEPT
        assert result.confidence_score >= 0.7
    
    def test_accept_works_for_me(self):
        """Test 'works for me' acceptance."""
        result = parse_reply_rules("This works for me!")
        
        assert result.intent == ReplyIntent.ACCEPT
    
    def test_accept_sounds_good(self):
        """Test 'sounds good' acceptance."""
        result = parse_reply_rules("Sounds good, thanks!")
        
        assert result.intent == ReplyIntent.ACCEPT
    
    def test_accept_perfect(self):
        """Test 'perfect' acceptance."""
        result = parse_reply_rules("Perfect!")
        
        assert result.intent == ReplyIntent.ACCEPT


class TestDeclineAndCancelIntentParsing:
    """Test parsing of decline and cancel requests."""
    
    def test_cancel_keyword(self):
        """Test explicit 'cancel' keyword."""
        result = parse_reply_rules("Please cancel this event")
        
        assert result.intent == ReplyIntent.CANCEL
        assert result.confidence_score >= 0.7
    
    def test_decline_keyword(self):
        """Test explicit 'decline' keyword."""
        result = parse_reply_rules("I decline this invitation")
        
        assert result.intent == ReplyIntent.CANCEL
    
    def test_no_longer_needed(self):
        """Test 'no longer needed' phrasing."""
        result = parse_reply_rules("No longer needed")
        
        assert result.intent == ReplyIntent.CANCEL
    
    def test_forget_it(self):
        """Test 'forget it' phrasing."""
        result = parse_reply_rules("Just forget it")
        
        assert result.intent == ReplyIntent.CANCEL


class TestQuestionIntentParsing:
    """Test parsing of questions."""
    
    def test_simple_question(self):
        """Test question with question mark."""
        result = parse_reply_rules("What papers are included in this session?")
        
        assert result.intent == ReplyIntent.QUESTION
        assert result.confidence_score >= 0.5
    
    def test_multiple_questions(self):
        """Test email with multiple questions."""
        result = parse_reply_rules("How long is this? Where can I find the papers?")
        
        assert result.intent == ReplyIntent.QUESTION


class TestUnknownIntent:
    """Test handling of unrecognized intents."""
    
    def test_empty_text(self):
        """Test parsing empty or very short text."""
        result = parse_reply_rules("")
        assert result.intent == ReplyIntent.UNKNOWN
        
        result = parse_reply_rules("ok")
        assert result.intent == ReplyIntent.UNKNOWN
    
    def test_unrelated_text(self):
        """Test parsing unrelated content."""
        result = parse_reply_rules("The weather is nice today")
        
        assert result.intent == ReplyIntent.UNKNOWN
        assert result.confidence_score <= 0.3


class TestDateTimeExtraction:
    """Test the datetime extraction helper function."""
    
    def test_extract_tomorrow_at_time(self):
        """Test extracting 'tomorrow at 2pm'."""
        dt, text = _extract_datetime_from_text("tomorrow at 2pm")
        
        assert dt is not None
        tomorrow = datetime.now() + timedelta(days=1)
        assert dt.day == tomorrow.day
        assert dt.hour == 14
        assert "tomorrow at 2pm" in text.lower()
    
    def test_extract_at_time_only(self):
        """Test extracting just 'at 3pm'."""
        dt, text = _extract_datetime_from_text("Let's meet at 3pm")
        
        assert dt is not None
        assert dt.hour == 15
    
    def test_extract_returns_none_for_no_time(self):
        """Test that no match returns None."""
        dt, text = _extract_datetime_from_text("Let's meet sometime")
        
        assert dt is None
        assert text is None


class TestTimeConversion:
    """Test 12h to 24h time conversion."""
    
    def test_pm_conversion(self):
        """Test PM time conversion."""
        assert _convert_to_24h(2, "pm") == 14
        assert _convert_to_24h(12, "pm") == 12  # Noon stays 12
    
    def test_am_conversion(self):
        """Test AM time conversion."""
        assert _convert_to_24h(9, "am") == 9
        assert _convert_to_24h(12, "am") == 0  # Midnight
    
    def test_no_period(self):
        """Test when no AM/PM specified."""
        assert _convert_to_24h(15, None) == 15


class TestParseReplyWrapper:
    """Test the synchronous parse_reply wrapper."""
    
    def test_parse_reply_without_llm(self):
        """Test parse_reply with use_llm=False (default)."""
        result = parse_reply("Move this to tomorrow at 4pm", use_llm=False)
        
        assert isinstance(result, ParsedReply)
        assert result.intent == ReplyIntent.RESCHEDULE
        assert result.extracted_datetime is not None
    
    def test_parse_reply_preserves_original_text(self):
        """Test that original text is preserved in result."""
        original = "Thanks, looks good!"
        result = parse_reply(original)
        
        assert result.original_text == original


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_mixed_signals(self):
        """Test email with conflicting signals."""
        # Has both reschedule and accept - reschedule with date takes precedence
        result = parse_reply_rules("Looks good, but how about tomorrow at 5pm instead?")
        
        assert result.intent == ReplyIntent.RESCHEDULE
        assert result.extracted_datetime is not None
    
    def test_unicode_text(self):
        """Test parsing text with unicode characters."""
        result = parse_reply_rules("Let's move it to tomorrow at 2pm 📅")
        
        assert result.intent == ReplyIntent.RESCHEDULE
    
    def test_multiline_text(self):
        """Test parsing multi-line email text."""
        text = """Hi,

This time doesn't work for me.
Can we do next Friday at 3pm instead?

Thanks,
User"""
        result = parse_reply_rules(text)
        
        assert result.intent == ReplyIntent.RESCHEDULE
    
    def test_case_insensitivity(self):
        """Test that parsing is case-insensitive."""
        result1 = parse_reply_rules("CANCEL THIS")
        result2 = parse_reply_rules("cancel this")
        result3 = parse_reply_rules("Cancel This")
        
        assert result1.intent == result2.intent == result3.intent == ReplyIntent.CANCEL


# Negative tests
class TestNegativeCases:
    """Test expected failures and error handling."""
    
    def test_invalid_date_format(self):
        """Test handling of invalid date formats."""
        result = parse_reply_rules("Move to 13/45 at 2pm")  # Invalid month/day
        
        # Should not crash, but may have low confidence
        assert isinstance(result, ParsedReply)
    
    def test_very_long_text(self):
        """Test handling of very long email text."""
        long_text = "This is a test. " * 1000 + "Can we reschedule to tomorrow at 2pm?"
        result = parse_reply_rules(long_text)
        
        # Should still detect the reschedule request
        assert result.intent == ReplyIntent.RESCHEDULE
    
    def test_special_characters(self):
        """Test handling of special characters."""
        result = parse_reply_rules("Cancel!!! @#$%^&*()")
        
        assert result.intent == ReplyIntent.CANCEL
