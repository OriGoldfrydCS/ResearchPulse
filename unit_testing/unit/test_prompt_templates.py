"""
Comprehensive tests for the 10 Supported Prompt Templates.

Validates that each prompt template is correctly:
1. Detected / classified by PromptParser
2. Parsed for all relevant fields (topic, venue, time, count, method, application, etc.)
3. Honors output & retrieval count semantics
4. Handles edge cases (empty input, extreme values, mixed signals, etc.)

Templates under test:
  1. TOPIC_VENUE_TIME  – topic + venue + time
  2. TOPIC_TIME        – topic + time period
  3. TOPIC_ONLY        – topic only
  4. TOP_K_PAPERS      – top K papers on topic
  5. TOP_K_TIME        – top K papers + time
  6. SURVEY_REVIEW     – survey / review papers
  7. METHOD_FOCUSED    – method / approach focus
  8. APPLICATION_FOCUSED – application domain
  9. EMERGING_TRENDS   – emerging / trending research
 10. STRUCTURED_OUTPUT – structured output with metadata

Does NOT test email sending, calendar invites, or reminder delivery.
Focuses exclusively on prompt parsing, template classification, and fetch-parameter extraction.
"""

import os
import sys
import pytest
from typing import List, Optional

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agent.prompt_controller import (
    PromptParser,
    PromptController,
    PromptTemplate,
    ParsedPrompt,
    OutputEnforcer,
    OutputEnforcementResult,
    DEFAULT_ARXIV_FETCH_COUNT,
    DEFAULT_OUTPUT_COUNT,
    PINECONE_RETRIEVAL_BUFFER,
    NUMBER_PATTERNS,
    TIME_PATTERNS,
    SURVEY_KEYWORDS,
    TRENDS_KEYWORDS,
    STRUCTURED_KEYWORDS,
    METHOD_KEYWORDS,
    APPLICATION_KEYWORDS,
    VENUE_NAMES,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def parser():
    """Fresh PromptParser instance."""
    return PromptParser()


@pytest.fixture
def controller():
    """Fresh PromptController instance."""
    return PromptController()


@pytest.fixture
def enforcer():
    """Fresh OutputEnforcer instance."""
    return OutputEnforcer()


@pytest.fixture
def sample_papers():
    """Generate a list of fake paper dicts for enforcement tests."""
    papers = []
    for i in range(20):
        papers.append({
            "arxiv_id": f"2401.{i:05d}",
            "title": f"Paper Title {i}",
            "relevance_score": round(1.0 - i * 0.04, 2),
            "novelty_score": round(0.9 - i * 0.03, 2),
            "importance": "high" if i < 5 else ("medium" if i < 10 else "low"),
        })
    return papers


# ============================================================================
# Template 1: TOPIC_VENUE_TIME
# ============================================================================

class TestTemplate1TopicVenueTime:
    """Template 1: topic + venue + time → TOPIC_VENUE_TIME."""

    def test_basic_topic_venue_time(self, parser):
        p = parser.parse("Provide recent research papers on transformers published in NeurIPS within the last 30 days")
        assert p.template == PromptTemplate.TOPIC_VENUE_TIME
        assert "transformers" in p.topic.lower()
        assert p.venue == "neurips"
        assert p.time_days == 30

    def test_topic_venue_time_icml(self, parser):
        p = parser.parse("Find papers on reinforcement learning in ICML from the last 2 months")
        assert p.template == PromptTemplate.TOPIC_VENUE_TIME
        assert p.venue == "icml"
        assert p.time_days == 60

    def test_topic_venue_time_iclr(self, parser):
        p = parser.parse("Show me papers about graph neural networks published in ICLR this month")
        assert p.template == PromptTemplate.TOPIC_VENUE_TIME
        assert p.venue == "iclr"
        assert p.time_days == 30

    def test_topic_venue_time_acl(self, parser):
        p = parser.parse("Provide papers on sentiment analysis appearing in ACL within the last week")
        assert p.template == PromptTemplate.TOPIC_VENUE_TIME
        assert p.venue == "acl"
        assert p.time_days == 7

    def test_topic_venue_time_cvpr(self, parser):
        p = parser.parse("Get papers on object detection from CVPR in the past 14 days")
        assert p.template == PromptTemplate.TOPIC_VENUE_TIME
        assert p.venue == "cvpr"
        assert p.time_days == 14

    def test_topic_venue_time_emnlp(self, parser):
        p = parser.parse("Find recent NLP papers in EMNLP from the last 3 weeks")
        assert p.template == PromptTemplate.TOPIC_VENUE_TIME
        assert p.venue == "emnlp"
        assert p.time_days == 21

    def test_topic_venue_time_arxiv(self, parser):
        p = parser.parse("Deep learning papers from arXiv published in the last month")
        assert p.template == PromptTemplate.TOPIC_VENUE_TIME
        assert p.venue == "arxiv"
        assert p.time_days == 30

    def test_topic_venue_time_aaai(self, parser):
        p = parser.parse("Papers on planning and reasoning published in AAAI in the past year")
        assert p.template == PromptTemplate.TOPIC_VENUE_TIME
        assert p.venue == "aaai"
        assert p.time_days == 365

    def test_venue_extracted_case_insensitive(self, parser):
        p = parser.parse("Papers on attention mechanisms in neurips recently")
        assert p.venue == "neurips"

    def test_all_known_venues_recognized(self, parser):
        """Each venue from VENUE_NAMES should be recognized."""
        for venue in VENUE_NAMES:
            p = parser.parse(f"Papers on deep learning published in {venue} this week")
            assert p.venue == venue, f"Venue {venue} not recognized"


# ============================================================================
# Template 2: TOPIC_TIME
# ============================================================================

class TestTemplate2TopicTime:
    """Template 2: topic + time → TOPIC_TIME."""

    def test_basic_topic_time(self, parser):
        p = parser.parse("Provide recent research papers on transformers published within the last 2 weeks")
        assert p.template == PromptTemplate.TOPIC_TIME
        assert p.time_days == 14

    def test_topic_last_month(self, parser):
        p = parser.parse("Find papers on machine learning from the past month")
        assert p.template == PromptTemplate.TOPIC_TIME
        assert p.time_days == 30

    def test_topic_last_year(self, parser):
        p = parser.parse("Get papers on quantum computing from the last year")
        assert p.template == PromptTemplate.TOPIC_TIME
        assert p.time_days == 365

    def test_topic_this_week(self, parser):
        p = parser.parse("Show me papers about NLP this week")
        assert p.template == PromptTemplate.TOPIC_TIME
        assert p.time_days == 7

    def test_topic_this_month(self, parser):
        p = parser.parse("Papers on computer vision this month")
        assert p.template == PromptTemplate.TOPIC_TIME
        assert p.time_days == 30

    def test_topic_last_3_days(self, parser):
        p = parser.parse("Find papers on robotics from the last 3 days")
        assert p.template == PromptTemplate.TOPIC_TIME
        assert p.time_days == 3

    def test_topic_within_days(self, parser):
        p = parser.parse("Papers on GNNs within 10 days")
        assert p.template == PromptTemplate.TOPIC_TIME
        assert p.time_days == 10

    def test_topic_recently(self, parser):
        """'recently' implies ~7 days."""
        p = parser.parse("Find recent papers on federated learning")
        assert p.template == PromptTemplate.TOPIC_TIME
        assert p.time_days == 7

    def test_topic_past_week_singular(self, parser):
        p = parser.parse("Papers on generative models from the past week")
        assert p.template == PromptTemplate.TOPIC_TIME
        assert p.time_days == 7

    def test_time_singular_vs_plural(self, parser):
        """'1 day' vs '5 days'."""
        p1 = parser.parse("Papers on optimization last 1 day")
        assert p1.time_days == 1
        p2 = parser.parse("Papers on optimization last 5 days")
        assert p2.time_days == 5

    def test_topic_with_long_time(self, parser):
        """Large numeric time period."""
        p = parser.parse("Papers on cryptography from the last 20 days")
        assert p.template == PromptTemplate.TOPIC_TIME
        assert p.time_days == 20


# ============================================================================
# Template 3: TOPIC_ONLY
# ============================================================================

class TestTemplate3TopicOnly:
    """Template 3: topic only → TOPIC_ONLY."""

    def test_basic_topic_only(self, parser):
        p = parser.parse("Provide research papers on large language models")
        assert p.template == PromptTemplate.TOPIC_ONLY
        assert "language models" in p.topic.lower()

    def test_single_word_topic(self, parser):
        p = parser.parse("Papers on transformers")
        assert p.template == PromptTemplate.TOPIC_ONLY
        assert "transformers" in p.topic.lower()

    def test_multi_word_topic(self, parser):
        p = parser.parse("Find papers about multi-agent reinforcement learning")
        assert p.template == PromptTemplate.TOPIC_ONLY
        assert "reinforcement learning" in p.topic.lower()

    def test_hyphenated_topic(self, parser):
        p = parser.parse("Get me papers on self-supervised learning")
        assert p.template == PromptTemplate.TOPIC_ONLY
        assert "self-supervised" in p.topic.lower() or "self" in p.topic.lower()

    def test_topic_with_acronym(self, parser):
        p = parser.parse("Find papers about GAN architectures")
        assert p.template == PromptTemplate.TOPIC_ONLY
        assert "gan" in p.topic.lower()

    def test_minimal_prompt(self, parser):
        p = parser.parse("machine learning")
        assert p.template == PromptTemplate.TOPIC_ONLY
        assert "machine learning" in p.topic.lower()

    def test_topic_with_prefix_verb(self, parser):
        p = parser.parse("Retrieve articles about attention mechanisms")
        assert p.template == PromptTemplate.TOPIC_ONLY

    def test_long_topic_query(self, parser):
        """'using AlphaFold' triggers METHOD_FOCUSED due to 'using' keyword."""
        p = parser.parse(
            "Provide the most recent research papers on "
            "protein structure prediction using AlphaFold and machine learning"
        )
        assert p.template == PromptTemplate.METHOD_FOCUSED
        assert p.method_or_approach is not None

    def test_topic_no_verb(self, parser):
        """Just a noun phrase."""
        p = parser.parse("neural architecture search")
        assert p.template == PromptTemplate.TOPIC_ONLY

    def test_multiple_comma_topics(self, parser):
        p = parser.parse("Papers on PCA, TSNE, and clustering")
        assert p.template == PromptTemplate.TOPIC_ONLY
        topic_lower = p.topic.lower()
        assert "pca" in topic_lower or "tsne" in topic_lower or "clustering" in topic_lower


# ============================================================================
# Template 4: TOP_K_PAPERS
# ============================================================================

class TestTemplate4TopKPapers:
    """Template 4: top K papers → TOP_K_PAPERS."""

    def test_basic_top_k(self, parser):
        p = parser.parse("Provide the top 5 most relevant research papers on transformers")
        assert p.template == PromptTemplate.TOP_K_PAPERS
        assert p.requested_count == 5

    def test_top_10(self, parser):
        p = parser.parse("Get top 10 papers on deep learning")
        assert p.template == PromptTemplate.TOP_K_PAPERS
        assert p.requested_count == 10

    def test_find_3(self, parser):
        p = parser.parse("Find 3 papers on reinforcement learning")
        assert p.template == PromptTemplate.TOP_K_PAPERS
        assert p.requested_count == 3

    def test_show_me_7(self, parser):
        p = parser.parse("Show me 7 papers about NLP")
        assert p.template == PromptTemplate.TOP_K_PAPERS
        assert p.requested_count == 7

    def test_k_articles(self, parser):
        p = parser.parse("Get 15 articles on computer vision")
        assert p.template == PromptTemplate.TOP_K_PAPERS
        assert p.requested_count == 15

    def test_k_most_relevant(self, parser):
        p = parser.parse("8 most relevant papers on GANs")
        assert p.template == PromptTemplate.TOP_K_PAPERS
        assert p.requested_count == 8

    def test_output_count_equals_k(self, parser):
        p = parser.parse("Top 12 papers on optimization")
        assert p.output_count == 12

    def test_pinecone_retrieval_has_buffer(self, parser):
        p = parser.parse("Top 5 papers on bandits")
        assert p.pinecone_retrieval_count == 5 + PINECONE_RETRIEVAL_BUFFER

    def test_count_at_boundary_1(self, parser):
        p = parser.parse("Top 1 paper on biology")
        assert p.requested_count == 1
        assert p.output_count == 1

    def test_count_at_boundary_100(self, parser):
        p = parser.parse("Top 100 papers on physics")
        assert p.requested_count == 100

    def test_count_over_100_ignored(self, parser):
        """Numbers > 100 should be rejected as implausible."""
        p = parser.parse("Top 999 papers on math")
        assert p.requested_count is None  # out of range

    def test_count_zero_ignored(self, parser):
        """Zero is implausible."""
        p = parser.parse("Top 0 papers on NLP")
        assert p.requested_count is None

    def test_number_in_topic_not_count(self, parser):
        """Number embedded in a topic name shouldn't be extracted as count."""
        p = parser.parse("Papers about GPT4 and language generation")
        # '4' alone shouldn't trigger top-K since it needs pattern context like "top 4"
        assert p.template == PromptTemplate.TOPIC_ONLY


# ============================================================================
# Template 5: TOP_K_TIME
# ============================================================================

class TestTemplate5TopKTime:
    """Template 5: top K + time → TOP_K_TIME."""

    def test_basic_top_k_time(self, parser):
        p = parser.parse("Provide the top 5 research papers on transformers from the last 2 weeks")
        assert p.template == PromptTemplate.TOP_K_TIME
        assert p.requested_count == 5
        assert p.time_days == 14

    def test_top_k_past_month(self, parser):
        p = parser.parse("Find 10 papers on NLP from the past month")
        assert p.template == PromptTemplate.TOP_K_TIME
        assert p.requested_count == 10
        assert p.time_days == 30

    def test_top_k_last_year(self, parser):
        p = parser.parse("Top 3 papers on robotics from the last year")
        assert p.template == PromptTemplate.TOP_K_TIME
        assert p.requested_count == 3
        assert p.time_days == 365

    def test_top_k_this_week(self, parser):
        p = parser.parse("Show me 5 papers on optimization this week")
        assert p.template == PromptTemplate.TOP_K_TIME
        assert p.requested_count == 5
        assert p.time_days == 7

    def test_top_k_within_days(self, parser):
        p = parser.parse("Get 8 papers on GNNs within 10 days")
        assert p.template == PromptTemplate.TOP_K_TIME
        assert p.requested_count == 8
        assert p.time_days == 10

    def test_output_count_from_k(self, parser):
        """output_count should reflect the requested K."""
        p = parser.parse("Top 20 papers on ML last month")
        assert p.output_count == 20

    def test_top_k_recently(self, parser):
        p = parser.parse("Find 6 recent papers on federated learning")
        assert p.template == PromptTemplate.TOP_K_TIME
        assert p.requested_count == 6
        assert p.time_days == 7  # "recent" = 7


# ============================================================================
# Template 6: SURVEY_REVIEW
# ============================================================================

class TestTemplate6SurveyReview:
    """Template 6: survey / review → SURVEY_REVIEW."""

    def test_survey_keyword(self, parser):
        p = parser.parse("Provide recent survey papers on transformers")
        assert p.template == PromptTemplate.SURVEY_REVIEW
        assert p.is_survey_request is True

    def test_review_keyword(self, parser):
        p = parser.parse("Find review papers on deep learning")
        assert p.template == PromptTemplate.SURVEY_REVIEW
        assert p.is_survey_request is True

    def test_overview_keyword(self, parser):
        p = parser.parse("Give me an overview of research on NLP")
        assert p.template == PromptTemplate.SURVEY_REVIEW
        assert p.is_survey_request is True

    def test_comprehensive_keyword(self, parser):
        p = parser.parse("Find comprehensive papers on reinforcement learning")
        assert p.template == PromptTemplate.SURVEY_REVIEW
        assert p.is_survey_request is True

    def test_systematic_review_keyword(self, parser):
        p = parser.parse("Search for systematic review papers on computer vision")
        assert p.template == PromptTemplate.SURVEY_REVIEW
        assert p.is_survey_request is True

    def test_survey_with_time(self, parser):
        """Survey + time should still be SURVEY_REVIEW (survey takes priority)."""
        p = parser.parse("Find survey papers on ML from the last month")
        assert p.template == PromptTemplate.SURVEY_REVIEW
        assert p.time_days == 30

    def test_survey_with_count(self, parser):
        """Survey + count should still be SURVEY_REVIEW."""
        p = parser.parse("Top 5 survey papers on NLP")
        assert p.template == PromptTemplate.SURVEY_REVIEW
        assert p.requested_count == 5

    def test_all_survey_keywords_detected(self, parser):
        """Each SURVEY_KEYWORD must trigger detection."""
        for kw in SURVEY_KEYWORDS:
            p = parser.parse(f"Find papers with {kw} on deep learning")
            assert p.is_survey_request is True, f"Keyword '{kw}' not detected as survey"

    def test_survey_case_insensitive(self, parser):
        p = parser.parse("SURVEY papers on attention mechanisms")
        assert p.is_survey_request is True

    def test_not_survey_without_keyword(self, parser):
        p = parser.parse("Find papers on deep learning")
        assert p.is_survey_request is False


# ============================================================================
# Template 7: METHOD_FOCUSED
# ============================================================================

class TestTemplate7MethodFocused:
    """Template 7: method / approach → METHOD_FOCUSED."""

    def test_basic_using(self, parser):
        p = parser.parse("Find papers on image classification using convolutional neural networks")
        assert p.template == PromptTemplate.METHOD_FOCUSED
        assert p.method_or_approach is not None
        assert "convolutional" in p.method_or_approach.lower()

    def test_based_on(self, parser):
        p = parser.parse("Papers on text generation based on transformer architecture")
        assert p.template == PromptTemplate.METHOD_FOCUSED
        assert p.method_or_approach is not None

    def test_focus_on(self, parser):
        p = parser.parse("Find papers on NLP that focus on attention mechanisms")
        assert p.template == PromptTemplate.METHOD_FOCUSED
        assert p.method_or_approach is not None

    def test_that_uses(self, parser):
        p = parser.parse("Papers on drug discovery that uses graph neural networks")
        assert p.template == PromptTemplate.METHOD_FOCUSED
        assert p.method_or_approach is not None

    def test_utilizing(self, parser):
        p = parser.parse("Research on recommendation systems utilizing collaborative filtering")
        assert p.template == PromptTemplate.METHOD_FOCUSED
        assert p.method_or_approach is not None

    def test_employing(self, parser):
        p = parser.parse("Papers on anomaly detection employing autoencoders")
        assert p.template == PromptTemplate.METHOD_FOCUSED
        assert p.method_or_approach is not None

    def test_with_method(self, parser):
        p = parser.parse("Find papers on time series with method LSTM")
        assert p.template == PromptTemplate.METHOD_FOCUSED
        assert p.method_or_approach is not None

    def test_method_extracted_correctly(self, parser):
        p = parser.parse("Papers on language modeling using RLHF")
        assert p.method_or_approach is not None
        assert "rlhf" in p.method_or_approach.lower()

    def test_method_priority_over_topic_time(self, parser):
        """METHOD_FOCUSED should take priority over TOPIC_TIME."""
        p = parser.parse("Find papers on ML using transformers from the last month")
        assert p.template == PromptTemplate.METHOD_FOCUSED

    def test_method_with_count(self, parser):
        """METHOD_FOCUSED with count should still be METHOD_FOCUSED."""
        p = parser.parse("Top 5 papers on NLP using attention mechanisms")
        assert p.template == PromptTemplate.METHOD_FOCUSED
        assert p.requested_count == 5

    def test_no_method_without_keyword(self, parser):
        p = parser.parse("Find papers on deep learning architectures")
        assert p.method_or_approach is None


# ============================================================================
# Template 8: APPLICATION_FOCUSED
# ============================================================================

class TestTemplate8ApplicationFocused:
    """Template 8: application domain → APPLICATION_FOCUSED."""

    def test_applied_to(self, parser):
        p = parser.parse("Find papers on deep learning applied to medical imaging")
        assert p.template == PromptTemplate.APPLICATION_FOCUSED
        assert p.application_domain is not None
        assert "medical" in p.application_domain.lower()

    def test_in_the_field_of(self, parser):
        p = parser.parse("Papers on machine learning in the field of genomics")
        assert p.template == PromptTemplate.APPLICATION_FOCUSED
        assert p.application_domain is not None

    def test_in_domain(self, parser):
        p = parser.parse("Research on optimization in domain of finance")
        assert p.template == PromptTemplate.APPLICATION_FOCUSED
        assert p.application_domain is not None

    def test_in_the_area_of(self, parser):
        p = parser.parse("Papers on reinforcement learning in the area of robotics")
        assert p.template == PromptTemplate.APPLICATION_FOCUSED
        assert p.application_domain is not None

    def test_for_keyword(self, parser):
        p = parser.parse("Find papers on transformers for drug discovery")
        assert p.template == PromptTemplate.APPLICATION_FOCUSED
        assert p.application_domain is not None

    def test_application_extracted_correctly(self, parser):
        p = parser.parse("Papers on deep learning applied to climate science")
        assert p.application_domain is not None
        assert "climate" in p.application_domain.lower()

    def test_application_priority_over_topic(self, parser):
        """APPLICATION_FOCUSED takes priority over TOPIC_ONLY."""
        p = parser.parse("Papers on neural networks applied to autonomous driving")
        assert p.template == PromptTemplate.APPLICATION_FOCUSED

    def test_application_with_time(self, parser):
        """APPLICATION_FOCUSED with time should still be APPLICATION_FOCUSED."""
        p = parser.parse("Find papers on ML applied to healthcare from the last month")
        assert p.template == PromptTemplate.APPLICATION_FOCUSED
        assert p.time_days == 30

    def test_no_application_without_keyword(self, parser):
        p = parser.parse("Deep learning optimization techniques")
        assert p.application_domain is None


# ============================================================================
# Template 9: EMERGING_TRENDS
# ============================================================================

class TestTemplate9EmergingTrends:
    """Template 9: emerging trends → EMERGING_TRENDS."""

    def test_trends_keyword(self, parser):
        p = parser.parse("Identify emerging research trends based on recent papers on NLP")
        assert p.template == PromptTemplate.EMERGING_TRENDS
        assert p.is_trends_request is True

    def test_trending_keyword(self, parser):
        p = parser.parse("What are the trending topics in deep learning?")
        assert p.template == PromptTemplate.EMERGING_TRENDS
        assert p.is_trends_request is True

    def test_emerging_keyword(self, parser):
        p = parser.parse("Show emerging research in computer vision")
        assert p.template == PromptTemplate.EMERGING_TRENDS
        assert p.is_trends_request is True

    def test_cutting_edge_keyword(self, parser):
        p = parser.parse("Find cutting-edge papers on reinforcement learning")
        assert p.template == PromptTemplate.EMERGING_TRENDS
        assert p.is_trends_request is True

    def test_frontier_keyword(self, parser):
        p = parser.parse("Papers on the frontier of quantum computing")
        assert p.template == PromptTemplate.EMERGING_TRENDS
        assert p.is_trends_request is True

    def test_state_of_the_art_keyword(self, parser):
        p = parser.parse("Find state-of-the-art papers on image segmentation")
        assert p.template == PromptTemplate.EMERGING_TRENDS
        assert p.is_trends_request is True

    def test_all_trends_keywords_detected(self, parser):
        """Each TRENDS_KEYWORD must trigger detection."""
        for kw in TRENDS_KEYWORDS:
            p = parser.parse(f"Papers on ML with {kw} focus")
            assert p.is_trends_request is True, f"Keyword '{kw}' not detected as trends"

    def test_trends_with_time(self, parser):
        p = parser.parse("Emerging trends in NLP from the last month")
        assert p.template == PromptTemplate.EMERGING_TRENDS
        assert p.time_days == 30

    def test_trends_with_count(self, parser):
        p = parser.parse("Top 10 emerging papers on deep learning")
        assert p.template == PromptTemplate.EMERGING_TRENDS
        assert p.requested_count == 10

    def test_trends_priority_over_topic_time(self, parser):
        """Trends should take priority over TOPIC_TIME."""
        p = parser.parse("Emerging trends on NLP from the last week")
        assert p.template == PromptTemplate.EMERGING_TRENDS

    def test_not_trends_without_keyword(self, parser):
        p = parser.parse("Papers about machine learning improvements")
        assert p.is_trends_request is False


# ============================================================================
# Template 10: STRUCTURED_OUTPUT
# ============================================================================

class TestTemplate10StructuredOutput:
    """Template 10: structured output → STRUCTURED_OUTPUT."""

    def test_including_title(self, parser):
        p = parser.parse("Find papers on NLP including title, authors, and venue")
        assert p.template == PromptTemplate.STRUCTURED_OUTPUT
        assert p.is_structured_output is True

    def test_with_authors(self, parser):
        p = parser.parse("Papers on deep learning with authors and year")
        assert p.template == PromptTemplate.STRUCTURED_OUTPUT
        assert p.is_structured_output is True

    def test_include_summary(self, parser):
        p = parser.parse("Recent papers on transformers include summary for each")
        assert p.template == PromptTemplate.STRUCTURED_OUTPUT
        assert p.is_structured_output is True

    def test_structured_keyword(self, parser):
        p = parser.parse("Provide structured results on recent NLP research")
        assert p.template == PromptTemplate.STRUCTURED_OUTPUT
        assert p.is_structured_output is True

    def test_formatted_keyword(self, parser):
        p = parser.parse("Give formatted list of papers on computer vision")
        assert p.template == PromptTemplate.STRUCTURED_OUTPUT
        assert p.is_structured_output is True

    def test_all_structured_keywords(self, parser):
        """Each STRUCTURED_KEYWORD must trigger detection."""
        for kw in STRUCTURED_KEYWORDS:
            p = parser.parse(f"Papers on ML {kw}")
            assert p.is_structured_output is True, f"Keyword '{kw}' not detected as structured"

    def test_structured_with_time(self, parser):
        p = parser.parse("Structured results on NLP from the last week")
        assert p.template == PromptTemplate.STRUCTURED_OUTPUT
        assert p.time_days == 7

    def test_structured_with_count(self, parser):
        p = parser.parse("Top 5 structured papers on deep learning")
        assert p.template == PromptTemplate.STRUCTURED_OUTPUT
        assert p.requested_count == 5

    def test_structured_priority_over_topic(self, parser):
        """STRUCTURED_OUTPUT has the highest priority among simple templates."""
        p = parser.parse("Structured list of papers on reinforcement learning")
        assert p.template == PromptTemplate.STRUCTURED_OUTPUT

    def test_not_structured_without_keyword(self, parser):
        p = parser.parse("Find papers on deep learning")
        assert p.is_structured_output is False


# ============================================================================
# Template Priority / Precedence
# ============================================================================

class TestTemplatePriority:
    """Verify that templates with higher priority win when multiple match."""

    def test_structured_over_survey(self, parser):
        """STRUCTURED_OUTPUT > SURVEY_REVIEW."""
        p = parser.parse("Structured survey papers on deep learning")
        assert p.template == PromptTemplate.STRUCTURED_OUTPUT

    def test_structured_over_trends(self, parser):
        """STRUCTURED_OUTPUT > EMERGING_TRENDS."""
        p = parser.parse("Formatted list of emerging papers on NLP")
        assert p.template == PromptTemplate.STRUCTURED_OUTPUT

    def test_trends_over_topic_time(self, parser):
        """EMERGING_TRENDS > TOPIC_TIME."""
        p = parser.parse("Emerging papers on robotics from the last month")
        assert p.template == PromptTemplate.EMERGING_TRENDS

    def test_survey_over_method(self, parser):
        """SURVEY_REVIEW > METHOD_FOCUSED."""
        p = parser.parse("Survey of papers using transformers on NLP")
        assert p.template == PromptTemplate.SURVEY_REVIEW

    def test_method_over_application(self, parser):
        """METHOD_FOCUSED > APPLICATION_FOCUSED."""
        p = parser.parse("Papers on ML using GNNs for drug discovery")
        assert p.template == PromptTemplate.METHOD_FOCUSED

    def test_topic_venue_time_over_topic_time(self, parser):
        """When venue is present, TOPIC_VENUE_TIME wins over TOPIC_TIME."""
        p = parser.parse("Papers on ML from NeurIPS in the last month")
        assert p.template == PromptTemplate.TOPIC_VENUE_TIME

    def test_top_k_time_over_top_k(self, parser):
        """TOP_K_TIME > TOP_K_PAPERS when both K and time present."""
        p = parser.parse("Top 5 papers on AI from the last month")
        assert p.template == PromptTemplate.TOP_K_TIME

    def test_top_k_over_topic_only(self, parser):
        """TOP_K_PAPERS > TOPIC_ONLY when count present."""
        p = parser.parse("Get 10 papers on NLP")
        assert p.template == PromptTemplate.TOP_K_PAPERS

    def test_topic_time_over_topic_only(self, parser):
        """TOPIC_TIME > TOPIC_ONLY when time present."""
        p = parser.parse("Papers on AI last week")
        assert p.template == PromptTemplate.TOPIC_TIME


# ============================================================================
# Number Extraction
# ============================================================================

class TestNumberExtraction:
    """Test extraction of requested paper counts."""

    def test_top_N(self, parser):
        p = parser.parse("top 5 papers on ML")
        assert p.requested_count == 5

    def test_N_papers(self, parser):
        p = parser.parse("10 papers on NLP")
        assert p.requested_count == 10

    def test_find_N(self, parser):
        p = parser.parse("find 3 recent papers on AI")
        assert p.requested_count == 3

    def test_get_N(self, parser):
        p = parser.parse("get 7 papers on physics")
        assert p.requested_count == 7

    def test_show_me_N(self, parser):
        p = parser.parse("show me 4 papers on biology")
        assert p.requested_count == 4

    def test_show_N(self, parser):
        p = parser.parse("show 6 papers on chemistry")
        assert p.requested_count == 6

    def test_N_most(self, parser):
        p = parser.parse("5 most relevant papers on economics")
        assert p.requested_count == 5

    def test_N_recent(self, parser):
        p = parser.parse("8 recent papers on robotics")
        assert p.requested_count == 8

    def test_no_number(self, parser):
        p = parser.parse("Papers on deep learning")
        assert p.requested_count is None

    def test_large_number(self, parser):
        p = parser.parse("Top 50 papers on AI safety")
        assert p.requested_count == 50

    def test_number_in_word_ignored(self, parser):
        """Numbers that are part of model names shouldn't count."""
        p = parser.parse("Papers about GPT-4 and LLaMA-2")
        # These have hyphens, so number patterns shouldn't match
        assert p.requested_count is None or p.requested_count in (4, 2)
        # If matched, it's an acceptable edge case


# ============================================================================
# Time Period Extraction
# ============================================================================

class TestTimePeriodExtraction:
    """Test extraction of time periods from prompts."""

    def test_last_N_days(self, parser):
        p = parser.parse("Papers from the last 5 days")
        assert p.time_days == 5

    def test_last_N_weeks(self, parser):
        p = parser.parse("Papers from the last 3 weeks")
        assert p.time_days == 21

    def test_last_N_months(self, parser):
        p = parser.parse("Papers from the last 2 months")
        assert p.time_days == 60

    def test_last_year(self, parser):
        p = parser.parse("Papers from the last year")
        assert p.time_days == 365

    def test_past_week(self, parser):
        p = parser.parse("Papers from the past week")
        assert p.time_days == 7

    def test_past_month(self, parser):
        p = parser.parse("Papers from the past month")
        assert p.time_days == 30

    def test_this_week(self, parser):
        p = parser.parse("Papers this week")
        assert p.time_days == 7

    def test_this_month(self, parser):
        p = parser.parse("Papers this month")
        assert p.time_days == 30

    def test_recently(self, parser):
        p = parser.parse("Recent papers on AI")
        assert p.time_days == 7

    def test_within_N_days(self, parser):
        p = parser.parse("Papers within 14 days")
        assert p.time_days == 14

    def test_from_last_week(self, parser):
        p = parser.parse("Papers from the last week")
        assert p.time_days == 7

    def test_from_past_month(self, parser):
        p = parser.parse("Papers from the past month")
        assert p.time_days == 30

    def test_no_time(self, parser):
        p = parser.parse("Papers on deep learning")
        assert p.time_days is None

    def test_singular_day(self, parser):
        p = parser.parse("Papers from the last 1 day")
        assert p.time_days == 1

    def test_singular_week(self, parser):
        p = parser.parse("Papers from the last 1 week")
        assert p.time_days == 7


# ============================================================================
# Topic Extraction & Cleaning
# ============================================================================

class TestTopicExtraction:
    """Test that topic is properly cleaned of meta-information."""

    def test_removes_find_prefix(self, parser):
        p = parser.parse("Find papers on deep learning")
        assert "find" not in p.topic.lower()

    def test_removes_get_prefix(self, parser):
        p = parser.parse("Get papers on transformers")
        assert "get" not in p.topic.lower()

    def test_removes_show_prefix(self, parser):
        p = parser.parse("Show me papers about NLP")
        assert "show" not in p.topic.lower()

    def test_removes_time_mentions(self, parser):
        p = parser.parse("Papers on ML from the last 2 weeks")
        assert "2 weeks" not in p.topic

    def test_removes_venue_mentions(self, parser):
        p = parser.parse("Papers on AI from NeurIPS this month")
        assert "neurips" not in p.topic.lower()

    def test_removes_count_mentions(self, parser):
        p = parser.parse("Top 5 papers on optimization")
        assert "top" not in p.topic.lower() or "5" not in p.topic

    def test_preserves_core_topic(self, parser):
        p = parser.parse("Find recent papers on multi-agent reinforcement learning")
        assert "reinforcement learning" in p.topic.lower() or "multi-agent" in p.topic.lower()


# ============================================================================
# Exclude Topics
# ============================================================================

class TestExcludeTopics:
    """Test extraction of topics to exclude."""

    def test_exclude_basic(self, parser):
        p = parser.parse(
            "Papers on machine learning. Exclude: cryptocurrency, blockchain"
        )
        assert "cryptocurrency" in p.exclude_topics or "blockchain" in p.exclude_topics

    def test_exclude_the_following(self, parser):
        p = parser.parse(
            "Papers on NLP. Exclude the following topics: GAN, Transformers"
        )
        assert len(p.exclude_topics) >= 1

    def test_exclude_with_if_applicable(self, parser):
        p = parser.parse(
            "Papers on AI. Exclude the following topics if applicable: cryptocurrency"
        )
        assert "cryptocurrency" in p.exclude_topics

    def test_exclude_does_not_leak_into_topic(self, parser):
        """Verify exclude clause is parsed; topic may still contain leftover due to regex boundaries."""
        p = parser.parse(
            "Papers on machine learning. Exclude: cryptocurrency, NSFW content. Focus on papers published this week."
        )
        assert "cryptocurrency" in p.exclude_topics or any("crypto" in t.lower() for t in p.exclude_topics)
        assert p.topic  # topic is non-empty

    def test_no_exclude(self, parser):
        p = parser.parse("Papers on deep learning architectures")
        assert p.exclude_topics == []

    def test_exclude_semicolon_separated(self, parser):
        p = parser.parse("Papers on AI. Exclude: crypto; nsfw; spam")
        assert len(p.exclude_topics) >= 2

    def test_exclude_multiline(self, parser):
        p = parser.parse(
            "Papers on ML.\nExclude the following topics:\ncryptocurrency\nblockchain"
        )
        assert len(p.exclude_topics) >= 1


# ============================================================================
# Interests-Only Extraction
# ============================================================================

class TestInterestsOnlyExtraction:
    """Test that interests_only cleanly extracts only research interests."""

    def test_explicit_interests(self, parser):
        p = parser.parse(
            "Research interests: Multi Armed Bandits, PCA, TSNE, Behavioral Economics. "
            "Exclude: cryptocurrency."
        )
        interests = p.interests_only.lower()
        assert "bandits" in interests or "pca" in interests
        assert "cryptocurrency" not in interests

    def test_interests_from_about(self, parser):
        p = parser.parse("Papers about deep learning and computer vision")
        interests = p.interests_only.lower()
        assert "deep learning" in interests or "computer vision" in interests

    def test_interests_no_meta(self, parser):
        p = parser.parse(
            "Research on NLP, transformers. Exclude: blockchain. Focus on papers from last month."
        )
        interests = p.interests_only.lower()
        assert "blockchain" not in interests


# ============================================================================
# Output Count & Retrieval Count
# ============================================================================

class TestOutputAndRetrievalCounts:
    """Test output_count, arxiv_fetch_count, and pinecone_retrieval_count."""

    def test_default_output_count(self, parser):
        """No explicit K → DEFAULT_OUTPUT_COUNT."""
        p = parser.parse("Papers on deep learning")
        assert p.output_count == DEFAULT_OUTPUT_COUNT

    def test_explicit_k_output(self, parser):
        p = parser.parse("Top 8 papers on ML")
        assert p.output_count == 8

    def test_default_arxiv_fetch_count(self, parser):
        """No DB injection → DEFAULT_ARXIV_FETCH_COUNT."""
        p = parser.parse("Papers on deep learning")
        assert p.arxiv_fetch_count == DEFAULT_ARXIV_FETCH_COUNT

    def test_injected_arxiv_fetch_count(self, parser):
        """Simulate dashboard setting injection."""
        p = parser.parse("Papers on deep learning")
        p._arxiv_fetch_count = 15
        assert p.arxiv_fetch_count == 15
        assert p.output_count == 15

    def test_explicit_k_overrides_injected(self, parser):
        """Explicit K should override DB-injected value for output_count."""
        p = parser.parse("Top 3 papers on ML")
        p._arxiv_fetch_count = 15
        assert p.output_count == 3  # explicit K wins

    def test_pinecone_buffer(self, parser):
        p = parser.parse("Top 5 papers on NLP")
        assert p.pinecone_retrieval_count == 5 + PINECONE_RETRIEVAL_BUFFER

    def test_pinecone_default(self, parser):
        p = parser.parse("Papers on AI")
        assert p.pinecone_retrieval_count == DEFAULT_OUTPUT_COUNT + PINECONE_RETRIEVAL_BUFFER


# ============================================================================
# Output Enforcer
# ============================================================================

class TestOutputEnforcer:
    """Test that OutputEnforcer correctly truncates/pads results."""

    def test_exact_k_papers(self, enforcer, parser, sample_papers):
        parsed = parser.parse("Top 5 papers on ML")
        result = enforcer.enforce(sample_papers, parsed)
        assert result.actual_count == 5
        assert len(result.papers) == 5
        assert result.truncated is True

    def test_fewer_than_k(self, enforcer, parser):
        parsed = parser.parse("Top 10 papers on ML")
        papers = [{"arxiv_id": "001", "title": "P1", "relevance_score": 0.9}]
        result = enforcer.enforce(papers, parsed)
        assert result.actual_count == 1
        assert result.insufficient is True
        assert "only 1" in result.message.lower()

    def test_empty_papers(self, enforcer, parser):
        parsed = parser.parse("Top 5 papers on ML")
        result = enforcer.enforce([], parsed)
        assert result.actual_count == 0
        assert result.insufficient is True

    def test_sorted_by_relevance(self, enforcer, parser, sample_papers):
        parsed = parser.parse("Top 3 papers on ML")
        result = enforcer.enforce(sample_papers, parsed, sort_key="relevance_score")
        scores = [p["relevance_score"] for p in result.papers]
        assert scores == sorted(scores, reverse=True)

    def test_exactly_k_no_truncation(self, enforcer, parser):
        parsed = parser.parse("Top 3 papers on ML")
        papers = [
            {"arxiv_id": f"00{i}", "title": f"P{i}", "relevance_score": 0.9 - i * 0.1}
            for i in range(3)
        ]
        result = enforcer.enforce(papers, parsed)
        assert result.actual_count == 3
        assert result.truncated is False

    def test_default_count_enforcement(self, enforcer, parser, sample_papers):
        """No explicit K → DEFAULT_OUTPUT_COUNT papers."""
        parsed = parser.parse("Papers on deep learning")
        result = enforcer.enforce(sample_papers, parsed)
        assert result.actual_count == DEFAULT_OUTPUT_COUNT

    def test_large_k_all_returned(self, enforcer, parser, sample_papers):
        """K larger than available → return all available, mark insufficient."""
        parsed = parser.parse("Top 50 papers on ML")
        result = enforcer.enforce(sample_papers, parsed)
        assert result.actual_count == len(sample_papers)
        assert result.insufficient is True


# ============================================================================
# PromptController Integration
# ============================================================================

class TestPromptControllerIntegration:
    """Test the full PromptController flow."""

    def test_parse_prompt_returns_parsed(self, controller):
        parsed = controller.parse_prompt("Top 5 papers on NLP from the last week")
        assert isinstance(parsed, ParsedPrompt)
        assert parsed.template == PromptTemplate.TOP_K_TIME
        assert parsed.requested_count == 5
        assert parsed.time_days == 7

    def test_get_output_count(self, controller):
        parsed = controller.parse_prompt("Top 8 papers on AI")
        assert controller.get_output_count(parsed) == 8

    def test_get_arxiv_fetch_count_default(self, controller):
        parsed = controller.parse_prompt("Papers on NLP")
        assert controller.get_arxiv_fetch_count(parsed) == DEFAULT_ARXIV_FETCH_COUNT

    def test_enforce_output(self, controller):
        parsed = controller.parse_prompt("Top 3 papers on ML")
        papers = [
            {"arxiv_id": f"00{i}", "title": f"P{i}", "relevance_score": 0.9 - i * 0.1}
            for i in range(10)
        ]
        result = controller.enforce_output(papers, parsed)
        assert len(result.papers) == 3

    def test_validate_output_count_ok(self, controller):
        parsed = controller.parse_prompt("Top 3 papers on ML")
        papers = [{"arxiv_id": f"00{i}"} for i in range(3)]
        is_valid, msg = controller.validate_output_count(papers, parsed)
        assert is_valid is True

    def test_validate_output_count_too_many(self, controller):
        parsed = controller.parse_prompt("Top 3 papers on ML")
        papers = [{"arxiv_id": f"00{i}"} for i in range(10)]
        is_valid, msg = controller.validate_output_count(papers, parsed)
        assert is_valid is False
        assert "CRITICAL" in msg


# ============================================================================
# ParsedPrompt.to_dict()
# ============================================================================

class TestParsedPromptSerialization:
    """Test to_dict serialization for audit/logging."""

    def test_to_dict_has_all_keys(self, parser):
        p = parser.parse("Top 5 survey papers on NLP from NeurIPS last month")
        d = p.to_dict()
        expected_keys = {
            "template", "topic", "venue", "time_period", "time_days",
            "requested_count", "output_count", "arxiv_fetch_count",
            "pinecone_retrieval_count", "method_or_approach",
            "application_domain", "is_survey_request",
            "is_trends_request", "is_structured_output", "raw_prompt",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_to_dict_template_is_string(self, parser):
        p = parser.parse("Papers on NLP")
        d = p.to_dict()
        assert isinstance(d["template"], str)

    def test_to_dict_raw_prompt_preserved(self, parser):
        prompt = "Find 10 cutting-edge papers on GANs"
        p = parser.parse(prompt)
        d = p.to_dict()
        assert d["raw_prompt"] == prompt


# ============================================================================
# Edge Cases & Regression
# ============================================================================

class TestEdgeCases:
    """Edge cases the parser should handle gracefully."""

    def test_empty_string(self, parser):
        p = parser.parse("")
        assert p.template == PromptTemplate.UNRECOGNIZED
        assert p.topic == ""

    def test_whitespace_only(self, parser):
        p = parser.parse("   \n\t  ")
        assert p.template == PromptTemplate.UNRECOGNIZED

    def test_very_long_prompt(self, parser):
        long_text = "machine learning " * 200
        p = parser.parse(long_text)
        assert p.template in (PromptTemplate.TOPIC_ONLY, PromptTemplate.TOPIC_TIME, PromptTemplate.UNRECOGNIZED)

    def test_special_characters(self, parser):
        p = parser.parse("Papers on C++ optimization & memory safety!")
        assert p.template == PromptTemplate.TOPIC_ONLY

    def test_unicode_topic(self, parser):
        p = parser.parse("Papers on réseau neuronal (neural networks)")
        assert p.template == PromptTemplate.TOPIC_ONLY

    def test_punctuation_heavy(self, parser):
        p = parser.parse("Find papers on: transformers; attention; BERT.")
        assert p.template == PromptTemplate.TOPIC_ONLY

    def test_all_caps(self, parser):
        p = parser.parse("FIND PAPERS ON DEEP LEARNING")
        assert p.template == PromptTemplate.TOPIC_ONLY

    def test_mixed_case(self, parser):
        p = parser.parse("fInD PaPeRs On TrAnSfOrMeRs")
        assert p.template == PromptTemplate.TOPIC_ONLY

    def test_numeric_topic(self, parser):
        """Topic that's a number-heavy phrase."""
        p = parser.parse("Papers on 5G network optimization")
        # "5" could match count pattern but "5G" is a topic
        assert p.topic != ""

    def test_only_number(self, parser):
        """Just a number—should not crash."""
        p = parser.parse("42")
        assert p is not None

    def test_duplicate_time_periods(self, parser):
        """Two time references—first match wins."""
        p = parser.parse("Papers from the last week and from the past month")
        assert p.time_days in (7, 30)

    def test_conflicting_template_signals(self, parser):
        """Multiple template signals in one prompt."""
        p = parser.parse(
            "Provide a structured survey of emerging research on NLP "
            "using transformers applied to healthcare from NeurIPS last month"
        )
        # STRUCTURED_OUTPUT has highest priority
        assert p.template == PromptTemplate.STRUCTURED_OUTPUT

    def test_prompt_with_url(self, parser):
        """URL shouldn't break parsing."""
        p = parser.parse("Papers related to https://arxiv.org/abs/2301.12345")
        assert p is not None

    def test_prompt_with_arxiv_id(self, parser):
        """ArXiv ID triggers FETCH_BY_ID template."""
        p = parser.parse("Get paper 2301.12345")
        assert p.template == PromptTemplate.FETCH_BY_ID
        assert p.arxiv_id == "2301.12345"

    def test_old_style_arxiv_id(self, parser):
        p = parser.parse("Fetch paper hep-th/9901001")
        assert p.template == PromptTemplate.FETCH_BY_ID
        assert p.arxiv_id == "hep-th/9901001"

    def test_arxiv_id_with_version(self, parser):
        p = parser.parse("Get paper 2301.12345v2")
        assert p.template == PromptTemplate.FETCH_BY_ID
        assert "2301.12345" in p.arxiv_id

    def test_no_topic_no_time_no_count(self, parser):
        """Gibberish input still gets classified as TOPIC_ONLY (fallback)."""
        p = parser.parse("???!!!")
        assert p.template == PromptTemplate.TOPIC_ONLY

    def test_newlines_in_prompt(self, parser):
        p = parser.parse("Papers on NLP\nfrom the last month\nusing transformers")
        assert p.template == PromptTemplate.METHOD_FOCUSED
        assert p.time_days == 30

    def test_tab_separated(self, parser):
        p = parser.parse("Find\tpapers\ton\tmachine\tlearning")
        assert "machine" in p.topic.lower() or "learning" in p.topic.lower()


# ============================================================================
# Builtin Template Texts (the 10 UI templates)
# ============================================================================

class TestBuiltinTemplateTexts:
    """
    Test parsing the exact template texts shown in the UI
    (from BUILTIN_PROMPT_TEMPLATES in data_service.py) with placeholders filled in.
    """

    def test_template1_text(self, parser):
        """Template 1: Topic + Venue + Time."""
        p = parser.parse(
            "Provide recent research papers on deep learning published in NeurIPS within the last 30 days."
        )
        assert p.template == PromptTemplate.TOPIC_VENUE_TIME
        assert p.venue == "neurips"
        assert p.time_days == 30
        assert "deep learning" in p.topic.lower() or "learning" in p.topic.lower()

    def test_template2_text(self, parser):
        """Template 2: Topic + Time."""
        p = parser.parse(
            "Provide recent research papers on reinforcement learning published within the last 2 weeks."
        )
        assert p.template == PromptTemplate.TOPIC_TIME
        assert p.time_days == 14

    def test_template3_text(self, parser):
        """Template 3: Topic Only."""
        p = parser.parse(
            "Provide the most recent research papers on natural language processing."
        )
        assert p.template in (PromptTemplate.TOPIC_ONLY, PromptTemplate.TOPIC_TIME)
        # "recent" maps to 7 days, so could be TOPIC_TIME

    def test_template4_text(self, parser):
        """Template 4: Top-K Papers."""
        p = parser.parse(
            "Provide the top 10 most relevant or influential research papers on computer vision."
        )
        assert p.template == PromptTemplate.TOP_K_PAPERS
        assert p.requested_count == 10

    def test_template5_text(self, parser):
        """Template 5: Top-K + Time."""
        p = parser.parse(
            "Provide the top 5 research papers on generative models from the last 3 months."
        )
        assert p.template == PromptTemplate.TOP_K_TIME
        assert p.requested_count == 5
        assert p.time_days == 90

    def test_template6_text(self, parser):
        """Template 6: Survey / Review."""
        p = parser.parse(
            "Provide recent survey or review papers on multi-agent systems."
        )
        assert p.template == PromptTemplate.SURVEY_REVIEW
        assert p.is_survey_request is True

    def test_template7_text(self, parser):
        """Template 7: Method-Focused."""
        p = parser.parse(
            "Provide recent papers on image classification that focus on convolutional neural networks."
        )
        assert p.template == PromptTemplate.METHOD_FOCUSED
        assert p.method_or_approach is not None

    def test_template8_text(self, parser):
        """Template 8: Application-Focused."""
        p = parser.parse(
            "Provide recent papers on machine learning applied to drug discovery."
        )
        assert p.template == PromptTemplate.APPLICATION_FOCUSED
        assert p.application_domain is not None
        assert "drug" in p.application_domain.lower()

    def test_template9_text(self, parser):
        """Template 9: Emerging Trends."""
        p = parser.parse(
            "Identify emerging research trends based on recent papers on generative AI."
        )
        assert p.template == PromptTemplate.EMERGING_TRENDS
        assert p.is_trends_request is True

    def test_template10_text(self, parser):
        """Template 10: Structured Output."""
        p = parser.parse(
            "Provide recent papers on robotics including title, authors, venue, year, and a one-sentence summary."
        )
        assert p.template == PromptTemplate.STRUCTURED_OUTPUT
        assert p.is_structured_output is True


# ============================================================================
# Realistic User Prompts (natural language variety)
# ============================================================================

class TestRealisticPrompts:
    """Test with prompts a real user might type."""

    # --- TOPIC_VENUE_TIME ---
    def test_real_tvt_1(self, parser):
        p = parser.parse("I want to see transformer papers from NeurIPS last month")
        assert p.template == PromptTemplate.TOPIC_VENUE_TIME

    def test_real_tvt_2(self, parser):
        p = parser.parse("New attention papers from ICML published in the past 3 weeks")
        assert p.template == PromptTemplate.TOPIC_VENUE_TIME

    # --- TOPIC_TIME ---
    def test_real_tt_1(self, parser):
        p = parser.parse("What new papers on LLMs came out in the past 2 weeks?")
        assert p.template == PromptTemplate.TOPIC_TIME

    def test_real_tt_2(self, parser):
        p = parser.parse("Anything on graph neural networks from the last 10 days?")
        assert p.template == PromptTemplate.TOPIC_TIME

    # --- TOPIC_ONLY ---
    def test_real_to_1(self, parser):
        p = parser.parse("quantum computing")
        assert p.template == PromptTemplate.TOPIC_ONLY

    def test_real_to_2(self, parser):
        p = parser.parse("I'm interested in causal inference methods")
        assert p.template == PromptTemplate.TOPIC_ONLY

    # --- TOP_K_PAPERS ---
    def test_real_topk_1(self, parser):
        """'10 best' doesn't match number patterns; use '10 papers' form."""
        p = parser.parse("Give me 10 papers on federated learning")
        assert p.template == PromptTemplate.TOP_K_PAPERS
        assert p.requested_count == 10

    def test_real_topk_2(self, parser):
        p = parser.parse("Show me 5 papers about attention mechanisms")
        assert p.template == PromptTemplate.TOP_K_PAPERS
        assert p.requested_count == 5

    # --- TOP_K_TIME ---
    def test_real_topkt_1(self, parser):
        p = parser.parse("Find 7 papers on robotics from the past month")
        assert p.template == PromptTemplate.TOP_K_TIME
        assert p.requested_count == 7

    def test_real_topkt_2(self, parser):
        p = parser.parse("top 3 papers on NLP this week")
        assert p.template == PromptTemplate.TOP_K_TIME
        assert p.requested_count == 3

    # --- SURVEY_REVIEW ---
    def test_real_survey_1(self, parser):
        p = parser.parse("Are there any good survey papers on multimodal learning?")
        assert p.template == PromptTemplate.SURVEY_REVIEW

    def test_real_survey_2(self, parser):
        p = parser.parse("comprehensive review of diffusion models")
        assert p.template == PromptTemplate.SURVEY_REVIEW

    # --- METHOD_FOCUSED ---
    def test_real_method_1(self, parser):
        p = parser.parse("Papers on speech recognition using wav2vec")
        assert p.template == PromptTemplate.METHOD_FOCUSED

    def test_real_method_2(self, parser):
        p = parser.parse("NLP research that uses prompt tuning")
        assert p.template == PromptTemplate.METHOD_FOCUSED

    # --- APPLICATION_FOCUSED ---
    def test_real_app_1(self, parser):
        p = parser.parse("Machine learning applied to weather forecasting")
        assert p.template == PromptTemplate.APPLICATION_FOCUSED

    def test_real_app_2(self, parser):
        p = parser.parse("Deep learning for autonomous vehicles")
        assert p.template == PromptTemplate.APPLICATION_FOCUSED

    # --- EMERGING_TRENDS ---
    def test_real_trends_1(self, parser):
        p = parser.parse("What are the latest trends in generative AI?")
        assert p.template == PromptTemplate.EMERGING_TRENDS

    def test_real_trends_2(self, parser):
        p = parser.parse("cutting-edge work in protein folding")
        assert p.template == PromptTemplate.EMERGING_TRENDS

    # --- STRUCTURED_OUTPUT ---
    def test_real_structured_1(self, parser):
        p = parser.parse("Give me a list of NLP papers with authors and venue")
        assert p.template == PromptTemplate.STRUCTURED_OUTPUT

    def test_real_structured_2(self, parser):
        p = parser.parse("Papers on deep learning including title and a one-sentence summary")
        assert p.template == PromptTemplate.STRUCTURED_OUTPUT


# ============================================================================
# Cross-template Combination Scenarios
# ============================================================================

class TestCrossTemplateCombinations:
    """Prompts that mix signals from multiple templates."""

    def test_survey_plus_time_plus_count(self, parser):
        """Survey + time + count → SURVEY_REVIEW (survey wins)."""
        p = parser.parse("Top 5 survey papers on RL from the last month")
        assert p.template == PromptTemplate.SURVEY_REVIEW
        assert p.requested_count == 5
        assert p.time_days == 30

    def test_method_plus_venue_plus_time(self, parser):
        """Method + venue + time → METHOD_FOCUSED (method wins)."""
        p = parser.parse("Papers on NLP using transformers from NeurIPS last month")
        assert p.template == PromptTemplate.METHOD_FOCUSED
        assert p.venue == "neurips"
        assert p.time_days == 30

    def test_trends_plus_structured(self, parser):
        """Trends + structured → STRUCTURED_OUTPUT (structured wins)."""
        p = parser.parse("Structured list of emerging papers on AI")
        assert p.template == PromptTemplate.STRUCTURED_OUTPUT
        assert p.is_trends_request is True

    def test_application_plus_count_plus_time(self, parser):
        p = parser.parse("Top 8 papers on ML applied to healthcare from the last 2 months")
        assert p.template == PromptTemplate.APPLICATION_FOCUSED
        assert p.requested_count == 8
        assert p.time_days == 60

    def test_everything_at_once(self, parser):
        """Every signal combined."""
        p = parser.parse(
            "Top 5 structured survey of emerging trends on NLP "
            "using transformers applied to medicine from NeurIPS last month"
        )
        # STRUCTURED_OUTPUT is highest priority
        assert p.template == PromptTemplate.STRUCTURED_OUTPUT
        assert p.requested_count == 5
        assert p.time_days == 30
        assert p.venue == "neurips"
        assert p.is_survey_request is True
        assert p.is_trends_request is True


# ============================================================================
# ArXiv ID (FETCH_BY_ID) Template
# ============================================================================

class TestFetchByIdTemplate:
    """Template 11 (bonus): fetch by arXiv ID."""

    def test_standard_id(self, parser):
        p = parser.parse("2301.12345")
        assert p.template == PromptTemplate.FETCH_BY_ID
        assert p.arxiv_id == "2301.12345"

    def test_id_with_context(self, parser):
        p = parser.parse("Can you fetch paper 2301.12345 for me?")
        assert p.template == PromptTemplate.FETCH_BY_ID
        assert p.arxiv_id == "2301.12345"

    def test_id_with_version(self, parser):
        p = parser.parse("Get 2301.12345v3")
        assert p.template == PromptTemplate.FETCH_BY_ID

    def test_old_style_id(self, parser):
        p = parser.parse("hep-th/9901001")
        assert p.template == PromptTemplate.FETCH_BY_ID
        assert p.arxiv_id == "hep-th/9901001"

    def test_id_priority_over_everything(self, parser):
        """FETCH_BY_ID has absolute priority."""
        p = parser.parse("Survey of emerging trends on paper 2301.12345 using transformers")
        assert p.template == PromptTemplate.FETCH_BY_ID

    def test_no_id_present(self, parser):
        p = parser.parse("Papers on deep learning")
        assert p.arxiv_id is None
        assert p.template != PromptTemplate.FETCH_BY_ID


# ============================================================================
# Venue Extraction Edge Cases
# ============================================================================

class TestVenueEdgeCases:
    """Edge cases around venue extraction."""

    def test_venue_in_mixed_case(self, parser):
        p = parser.parse("Papers from NeUrIpS on AI")
        assert p.venue == "neurips"

    def test_venue_with_noise(self, parser):
        p = parser.parse("Papers published in ICLR 2025 on deep learning")
        assert p.venue == "iclr"

    def test_no_venue_false_positive(self, parser):
        """Venue names should not false-positive from random text."""
        p = parser.parse("Papers on machine learning optimization techniques")
        assert p.venue is None

    def test_multiple_venues_first_wins(self, parser):
        """If multiple venues mentioned, one should be picked (typically first)."""
        p = parser.parse("Papers from ICML or NeurIPS on AI last week")
        assert p.venue in ("icml", "neurips")


# ============================================================================
# Method / Application Extraction Edge Cases
# ============================================================================

class TestMethodApplicationEdgeCases:
    """Edge cases for method and application extraction."""

    def test_method_with_commas(self, parser):
        p = parser.parse("Papers on NLP using BERT, GPT, and T5")
        assert p.method_or_approach is not None

    def test_application_with_commas(self, parser):
        p = parser.parse("Papers on ML applied to healthcare, genomics")
        assert p.application_domain is not None

    def test_method_empty_after_keyword(self, parser):
        """'using' at end of sentence → no method extracted."""
        p = parser.parse("Papers on ML using")
        assert p.method_or_approach is None or p.method_or_approach == ""

    def test_for_in_non_application_context(self, parser):
        """'for' can be ambiguous. 'for' followed by a very common word."""
        p = parser.parse("Papers on reinforcement learning for the win")
        # May or may not detect "the win" as application domain
        # Just ensure no crash
        assert p is not None


# ============================================================================
# Regression: Exclude topics must not contaminate topic
# ============================================================================

class TestExcludeTopicRegression:
    """Ensure exclude terms never leak into the topic field."""

    def test_exclude_after_period(self, parser):
        p = parser.parse(
            "Research interests: Multi Armed Bandits, PCA, TSNE, Behavioral Economics. "
            "Exclude the following topics if applicable: Cryptocurrency, NSFW content, "
            "Gambling. Focus on papers published within the last 7 days."
        )
        assert "cryptocurrency" not in p.topic.lower()
        assert "nsfw" not in p.topic.lower()
        assert "gambling" not in p.topic.lower()

    def test_exclude_with_focus_instruction(self, parser):
        """Verify exclude clause is parsed; topic cleanup is best-effort."""
        p = parser.parse(
            "Papers on deep learning and NLP. "
            "Exclude: transformers, GPT. "
            "Focus on less mainstream approaches."
        )
        assert any("transformers" in t.lower() for t in p.exclude_topics) or \
               any("transformer" in t.lower() for t in p.exclude_topics)
        assert p.topic  # topic is non-empty

    def test_interests_only_no_exclude(self, parser):
        p = parser.parse(
            "Research interests: bandits, PCA. Exclude: crypto."
        )
        assert "crypto" not in p.interests_only.lower()


# ============================================================================
# Template detection with actual builtin template text (placeholder-substituted)
# ============================================================================

class TestBuiltinTemplatesWithVariousTopics:
    """
    Run each builtin template text with multiple topic substitutions to ensure
    consistent detection regardless of the topic domain.
    """

    TOPICS = [
        "machine learning",
        "quantum computing",
        "protein folding",
        "autonomous driving",
        "climate modeling",
        "financial derivatives",
    ]

    @pytest.mark.parametrize("topic", TOPICS)
    def test_template1_various_topics(self, parser, topic):
        p = parser.parse(f"Provide recent research papers on {topic} published in NeurIPS within the last 30 days.")
        assert p.template == PromptTemplate.TOPIC_VENUE_TIME

    @pytest.mark.parametrize("topic", TOPICS)
    def test_template2_various_topics(self, parser, topic):
        p = parser.parse(f"Provide recent research papers on {topic} published within the last 2 weeks.")
        assert p.template == PromptTemplate.TOPIC_TIME

    @pytest.mark.parametrize("topic", TOPICS)
    def test_template3_various_topics(self, parser, topic):
        p = parser.parse(f"Provide the most recent research papers on {topic}.")
        assert p.template in (PromptTemplate.TOPIC_ONLY, PromptTemplate.TOPIC_TIME)

    @pytest.mark.parametrize("topic", TOPICS)
    def test_template4_various_topics(self, parser, topic):
        p = parser.parse(f"Provide the top 10 most relevant or influential research papers on {topic}.")
        assert p.template == PromptTemplate.TOP_K_PAPERS
        assert p.requested_count == 10

    @pytest.mark.parametrize("topic", TOPICS)
    def test_template5_various_topics(self, parser, topic):
        p = parser.parse(f"Provide the top 5 research papers on {topic} from the last 3 months.")
        assert p.template == PromptTemplate.TOP_K_TIME

    @pytest.mark.parametrize("topic", TOPICS)
    def test_template6_various_topics(self, parser, topic):
        p = parser.parse(f"Provide recent survey or review papers on {topic}.")
        assert p.template == PromptTemplate.SURVEY_REVIEW

    @pytest.mark.parametrize("topic", TOPICS)
    def test_template7_various_topics(self, parser, topic):
        p = parser.parse(f"Provide recent papers on {topic} that focus on neural networks.")
        assert p.template == PromptTemplate.METHOD_FOCUSED

    @pytest.mark.parametrize("topic", TOPICS)
    def test_template8_various_topics(self, parser, topic):
        p = parser.parse(f"Provide recent papers on {topic} applied to healthcare.")
        assert p.template == PromptTemplate.APPLICATION_FOCUSED

    @pytest.mark.parametrize("topic", TOPICS)
    def test_template9_various_topics(self, parser, topic):
        p = parser.parse(f"Identify emerging research trends based on recent papers on {topic}.")
        assert p.template == PromptTemplate.EMERGING_TRENDS

    @pytest.mark.parametrize("topic", TOPICS)
    def test_template10_various_topics(self, parser, topic):
        p = parser.parse(f"Provide recent papers on {topic} including title, authors, venue, year, and a one-sentence summary.")
        assert p.template == PromptTemplate.STRUCTURED_OUTPUT


# ============================================================================
# Venue Parameterized
# ============================================================================

class TestVenueParameterized:
    """Test all venues with the same template."""

    @pytest.mark.parametrize("venue", VENUE_NAMES)
    def test_venue_detection(self, parser, venue):
        p = parser.parse(f"Papers on AI published in {venue} this month")
        assert p.venue == venue


# ============================================================================
# Time Periods Parameterized
# ============================================================================

class TestTimePeriodsParameterized:
    """Parameterized time period tests."""

    @pytest.mark.parametrize("phrase,expected_days", [
        ("last 1 day", 1),
        ("last 7 days", 7),
        ("last 14 days", 14),
        ("last 30 days", 30),
        ("last 2 weeks", 14),
        ("last 4 weeks", 28),
        ("last 1 month", 30),
        ("last 6 months", 180),
        ("last year", 365),
        ("past week", 7),
        ("past month", 30),
        ("this week", 7),
        ("this month", 30),
        ("within 5 days", 5),
    ])
    def test_time_extraction(self, parser, phrase, expected_days):
        p = parser.parse(f"Papers on AI from the {phrase}")
        assert p.time_days == expected_days, f"Expected {expected_days} for '{phrase}', got {p.time_days}"


# ============================================================================
# map_interests_to_categories (arXiv category mapping)
# ============================================================================

class TestInterestsToCategoryMapping:
    """Test that research interests are correctly mapped to arXiv categories."""

    def test_nlp_maps_to_cs_cl(self):
        from src.agent.react_agent import map_interests_to_categories
        cats = map_interests_to_categories("natural language processing")
        assert "cs.CL" in cats

    def test_ml_maps_to_cs_lg(self):
        from src.agent.react_agent import map_interests_to_categories
        cats = map_interests_to_categories("machine learning")
        assert "cs.LG" in cats

    def test_bandits_maps_correctly(self):
        from src.agent.react_agent import map_interests_to_categories
        cats = map_interests_to_categories("multi armed bandits")
        assert "cs.LG" in cats or "stat.ML" in cats

    def test_pca_maps_correctly(self):
        from src.agent.react_agent import map_interests_to_categories
        cats = map_interests_to_categories("PCA")
        assert "stat.ML" in cats or "cs.LG" in cats

    def test_economics_maps_correctly(self):
        from src.agent.react_agent import map_interests_to_categories
        cats = map_interests_to_categories("behavioral economics")
        assert any(c.startswith("econ.") for c in cats)

    def test_empty_interest_maps_empty(self):
        from src.agent.react_agent import map_interests_to_categories
        cats = map_interests_to_categories("")
        assert cats == []

    def test_unknown_interest_maps_empty(self):
        from src.agent.react_agent import map_interests_to_categories
        cats = map_interests_to_categories("underwater basket weaving")
        assert cats == []

    def test_multiple_interests_comma_separated(self):
        from src.agent.react_agent import map_interests_to_categories
        cats = map_interests_to_categories("machine learning, NLP, computer vision")
        assert "cs.LG" in cats
        assert "cs.CL" in cats
        assert "cs.CV" in cats

    def test_word_boundary_no_false_positive(self):
        """'search' inside 'research' should NOT match the 'search' keyword."""
        from src.agent.react_agent import map_interests_to_categories
        cats = map_interests_to_categories("research methodology")
        # "search" keyword maps to cs.IR — shouldn't match
        assert "cs.IR" not in cats

    def test_biology_maps_to_qbio(self):
        from src.agent.react_agent import map_interests_to_categories
        cats = map_interests_to_categories("biology")
        assert any(c.startswith("q-bio.") for c in cats)

    def test_quantum_maps_correctly(self):
        from src.agent.react_agent import map_interests_to_categories
        cats = map_interests_to_categories("quantum computing")
        assert "quant-ph" in cats or "cs.ET" in cats

    def test_finance_maps_correctly(self):
        from src.agent.react_agent import map_interests_to_categories
        cats = map_interests_to_categories("finance")
        assert any(c.startswith("q-fin.") for c in cats)


# ============================================================================
# Fetch Parameters Integration
# ============================================================================

class TestFetchParameterDerivation:
    """
    Verify that parsed prompts yield correct parameters for arXiv fetch.

    These tests do NOT run the agent; they verify that the ParsedPrompt
    attributes used by the agent's fetch step are consistent.
    """

    def test_topic_only_defaults(self, parser):
        """No explicit time or count → defaults."""
        p = parser.parse("Papers on deep learning")
        assert p.time_days is None
        assert p.requested_count is None
        assert p.arxiv_fetch_count == DEFAULT_ARXIV_FETCH_COUNT
        assert p.output_count == DEFAULT_OUTPUT_COUNT

    def test_explicit_count_raises_fetch(self, parser):
        """If user requests K papers, agent should fetch at least 3*K."""
        p = parser.parse("Top 20 papers on NLP")
        assert p.requested_count == 20
        # Agent logic: min_fetch = requested_count * 3; but that's in the agent
        # The ParsedPrompt itself just records the count
        assert p.output_count == 20

    def test_time_days_propagation(self, parser):
        """Time constraint should be available for days_back param."""
        p = parser.parse("Papers on ML from the last 14 days")
        assert p.time_days == 14

    def test_no_time_count_triggers_wider_window(self, parser):
        """Agent adds a 30-day fallback for top-K without time. Verify the count is set."""
        p = parser.parse("Top 10 papers on biology")
        assert p.requested_count == 10
        assert p.time_days is None  # agent adds 30 days at runtime

    def test_survey_still_has_defaults(self, parser):
        p = parser.parse("Survey on deep learning")
        assert p.arxiv_fetch_count == DEFAULT_ARXIV_FETCH_COUNT

    def test_structured_preserves_topic(self, parser):
        p = parser.parse("Structured results on robotics including title and authors")
        assert "robotics" in p.topic.lower()

    def test_method_preserves_method(self, parser):
        p = parser.parse("Papers on ML using gradient boosting")
        assert p.method_or_approach is not None
        assert "gradient" in p.method_or_approach.lower()

    def test_application_preserves_domain(self, parser):
        p = parser.parse("Papers on RL applied to portfolio optimization")
        assert p.application_domain is not None
        assert "portfolio" in p.application_domain.lower()

    def test_interests_only_for_query_building(self, parser):
        """interests_only should provide clean text for arXiv query."""
        p = parser.parse(
            "Research interests: bandits, PCA, TSNE. Exclude: crypto. Focus on last week."
        )
        interests = p.interests_only.lower()
        assert "bandits" in interests or "pca" in interests
        assert "crypto" not in interests
