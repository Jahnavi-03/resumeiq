import json
import pytest
from unittest.mock import MagicMock, patch

from app.core.config import Settings
from app.models.schemas import CandidateAnalysis, CandidateRanking
from app.services.scorer import ScoringService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_settings() -> Settings:
    return Settings(
        groq_api_key="test-key",
        chat_model="llama-3.1-8b-instant",
        embedding_model="all-MiniLM-L6-v2",
        _env_file=None,
    )


# Minimal valid CandidateAnalysis JSON the LLM would return
VALID_CANDIDATE_JSON = {
    "ats_score": 72,
    "keyword_match": 65,
    "format_score": 7,
    "missing_required_skills": ["Kubernetes", "Terraform"],
    "missing_preferred_skills": ["GraphQL"],
    "formatting_issues": ["Two-column layout detected"],
    "bullet_analyses": [
        {
            "original_bullet": "Led a team",
            "original_score": 3,
            "rewritten_bullet": "Led cross-functional team of 8 engineers to deliver payment system",
            "rewritten_score": 9,
            "issue": "No metric, weak verb",
            "improvement_notes": "Added team size and quantified outcome",
            "priority": "High",
        }
    ],
    "suggestions": ["Add metrics to bullets", "Include Kubernetes experience"],
    "projected_score": 88,
}

# Minimal valid CandidateRanking JSON the LLM would return
VALID_RANKING_JSON = {
    "name": "Jane Doe",
    "match_score": 82,
    "strengths": ["Strong Python skills", "Relevant experience"],
    "skill_gaps": ["Missing Kubernetes"],
    "recommendation": "Strong Yes",
}

RESUME = "Jane Doe\nSoftware Engineer\nPython, FastAPI, SQL\nLed a team of 5."
JD = "We are hiring a Senior Backend Engineer. Required: Python, Kubernetes."
CONTEXT = "ATS tip: use standard section headers. FAANG tip: quantify achievements."


# ---------------------------------------------------------------------------
# Fixture — ScoringService with ChatGroq mocked out
# ---------------------------------------------------------------------------

@pytest.fixture
def scorer() -> ScoringService:
    """
    ScoringService with ChatGroq patched at import time so no real API
    calls are made and no real API key is needed.
    """
    with patch("app.services.scorer.ChatGroq"):
        service = ScoringService(settings=_fake_settings())
    return service


# ---------------------------------------------------------------------------
# score_candidate — happy path
# ---------------------------------------------------------------------------

class TestScoreCandidateValid:
    def test_returns_candidate_analysis_object(self, scorer: ScoringService):
        scorer._llm.invoke.return_value.content = json.dumps(VALID_CANDIDATE_JSON)
        result = scorer.score_candidate(RESUME, JD, CONTEXT)
        assert isinstance(result, CandidateAnalysis)

    def test_ats_score_is_correct(self, scorer: ScoringService):
        scorer._llm.invoke.return_value.content = json.dumps(VALID_CANDIDATE_JSON)
        result = scorer.score_candidate(RESUME, JD, CONTEXT)
        assert result.ats_score == 72

    def test_keyword_match_is_correct(self, scorer: ScoringService):
        scorer._llm.invoke.return_value.content = json.dumps(VALID_CANDIDATE_JSON)
        result = scorer.score_candidate(RESUME, JD, CONTEXT)
        assert result.keyword_match == 65

    def test_missing_skills_are_parsed(self, scorer: ScoringService):
        scorer._llm.invoke.return_value.content = json.dumps(VALID_CANDIDATE_JSON)
        result = scorer.score_candidate(RESUME, JD, CONTEXT)
        assert "Kubernetes" in result.missing_required_skills

    def test_bullet_analysis_improvement_is_computed(self, scorer: ScoringService):
        scorer._llm.invoke.return_value.content = json.dumps(VALID_CANDIDATE_JSON)
        result = scorer.score_candidate(RESUME, JD, CONTEXT)
        bullet = result.bullet_analyses[0]
        # improvement = rewritten_score - original_score = 9 - 3 = 6
        assert bullet.improvement == 6

    def test_llm_is_called_once_on_success(self, scorer: ScoringService):
        scorer._llm.invoke.return_value.content = json.dumps(VALID_CANDIDATE_JSON)
        scorer.score_candidate(RESUME, JD, CONTEXT)
        assert scorer._llm.invoke.call_count == 1


# ---------------------------------------------------------------------------
# score_candidate — JSON with markdown wrapper (safety net test)
# ---------------------------------------------------------------------------

class TestScoreCandidateMarkdown:
    def test_strips_markdown_code_block(self, scorer: ScoringService):
        # LLM wraps JSON in ```json ... ``` despite our instructions
        wrapped = f"```json\n{json.dumps(VALID_CANDIDATE_JSON)}\n```"
        scorer._llm.invoke.return_value.content = wrapped
        result = scorer.score_candidate(RESUME, JD, CONTEXT)
        assert isinstance(result, CandidateAnalysis)

    def test_strips_plain_code_block(self, scorer: ScoringService):
        wrapped = f"```\n{json.dumps(VALID_CANDIDATE_JSON)}\n```"
        scorer._llm.invoke.return_value.content = wrapped
        result = scorer.score_candidate(RESUME, JD, CONTEXT)
        assert isinstance(result, CandidateAnalysis)


# ---------------------------------------------------------------------------
# score_candidate — retry logic
# ---------------------------------------------------------------------------

class TestScoreCandidateRetry:
    def test_invalid_json_triggers_retry(self, scorer: ScoringService):
        # First call returns garbage, second call returns valid JSON
        scorer._llm.invoke.side_effect = [
            MagicMock(content="this is not json at all"),
            MagicMock(content=json.dumps(VALID_CANDIDATE_JSON)),
        ]
        result = scorer.score_candidate(RESUME, JD, CONTEXT)
        assert isinstance(result, CandidateAnalysis)
        assert scorer._llm.invoke.call_count == 2

    def test_two_invalid_json_responses_raise_value_error(self, scorer: ScoringService):
        scorer._llm.invoke.side_effect = [
            MagicMock(content="not json"),
            MagicMock(content="still not json"),
        ]
        with pytest.raises(ValueError, match="invalid JSON after retry"):
            scorer.score_candidate(RESUME, JD, CONTEXT)

    def test_error_message_includes_raw_response(self, scorer: ScoringService):
        scorer._llm.invoke.side_effect = [
            MagicMock(content="bad response"),
            MagicMock(content="also bad"),
        ]
        with pytest.raises(ValueError, match="also bad"):
            scorer.score_candidate(RESUME, JD, CONTEXT)


# ---------------------------------------------------------------------------
# score_candidate — Pydantic validation errors
# ---------------------------------------------------------------------------

class TestScoreCandidateValidation:
    def test_ats_score_above_100_raises(self, scorer: ScoringService):
        bad = {**VALID_CANDIDATE_JSON, "ats_score": 150}
        scorer._llm.invoke.return_value.content = json.dumps(bad)
        with pytest.raises(ValueError, match="schema"):
            scorer.score_candidate(RESUME, JD, CONTEXT)

    def test_invalid_priority_raises(self, scorer: ScoringService):
        bad_bullet = {**VALID_CANDIDATE_JSON["bullet_analyses"][0], "priority": "Critical"}
        bad = {**VALID_CANDIDATE_JSON, "bullet_analyses": [bad_bullet]}
        scorer._llm.invoke.return_value.content = json.dumps(bad)
        with pytest.raises(ValueError, match="schema"):
            scorer.score_candidate(RESUME, JD, CONTEXT)

    def test_missing_required_field_raises(self, scorer: ScoringService):
        # Remove ats_score — a required field
        bad = {k: v for k, v in VALID_CANDIDATE_JSON.items() if k != "ats_score"}
        scorer._llm.invoke.return_value.content = json.dumps(bad)
        with pytest.raises(ValueError, match="schema"):
            scorer.score_candidate(RESUME, JD, CONTEXT)

    def test_wrong_type_raises(self, scorer: ScoringService):
        # ats_score should be a number, not a string
        bad = {**VALID_CANDIDATE_JSON, "ats_score": "high"}
        scorer._llm.invoke.return_value.content = json.dumps(bad)
        with pytest.raises(ValueError, match="schema"):
            scorer.score_candidate(RESUME, JD, CONTEXT)


# ---------------------------------------------------------------------------
# score_recruiter — happy path
# ---------------------------------------------------------------------------

class TestScoreRecruiterValid:
    def test_returns_candidate_ranking_object(self, scorer: ScoringService):
        scorer._llm.invoke.return_value.content = json.dumps(VALID_RANKING_JSON)
        result = scorer.score_recruiter(RESUME, JD, CONTEXT, "Jane Doe")
        assert isinstance(result, CandidateRanking)

    def test_name_is_correct(self, scorer: ScoringService):
        scorer._llm.invoke.return_value.content = json.dumps(VALID_RANKING_JSON)
        result = scorer.score_recruiter(RESUME, JD, CONTEXT, "Jane Doe")
        assert result.name == "Jane Doe"

    def test_match_score_is_correct(self, scorer: ScoringService):
        scorer._llm.invoke.return_value.content = json.dumps(VALID_RANKING_JSON)
        result = scorer.score_recruiter(RESUME, JD, CONTEXT, "Jane Doe")
        assert result.match_score == 82

    def test_recommendation_is_correct(self, scorer: ScoringService):
        scorer._llm.invoke.return_value.content = json.dumps(VALID_RANKING_JSON)
        result = scorer.score_recruiter(RESUME, JD, CONTEXT, "Jane Doe")
        assert result.recommendation == "Strong Yes"

    def test_llm_called_once_on_success(self, scorer: ScoringService):
        scorer._llm.invoke.return_value.content = json.dumps(VALID_RANKING_JSON)
        scorer.score_recruiter(RESUME, JD, CONTEXT, "Jane Doe")
        assert scorer._llm.invoke.call_count == 1


# ---------------------------------------------------------------------------
# score_recruiter — retry and validation
# ---------------------------------------------------------------------------

class TestScoreRecruiterRetry:
    def test_invalid_json_triggers_retry(self, scorer: ScoringService):
        scorer._llm.invoke.side_effect = [
            MagicMock(content="not json"),
            MagicMock(content=json.dumps(VALID_RANKING_JSON)),
        ]
        result = scorer.score_recruiter(RESUME, JD, CONTEXT, "Jane Doe")
        assert isinstance(result, CandidateRanking)
        assert scorer._llm.invoke.call_count == 2

    def test_two_failures_raise_value_error(self, scorer: ScoringService):
        scorer._llm.invoke.side_effect = [
            MagicMock(content="bad"),
            MagicMock(content="also bad"),
        ]
        with pytest.raises(ValueError, match="invalid JSON after retry"):
            scorer.score_recruiter(RESUME, JD, CONTEXT, "Jane Doe")


class TestScoreRecruiterValidation:
    def test_invalid_recommendation_raises(self, scorer: ScoringService):
        bad = {**VALID_RANKING_JSON, "recommendation": "Definitely"}
        scorer._llm.invoke.return_value.content = json.dumps(bad)
        with pytest.raises(ValueError, match="schema"):
            scorer.score_recruiter(RESUME, JD, CONTEXT, "Jane Doe")

    def test_match_score_above_100_raises(self, scorer: ScoringService):
        bad = {**VALID_RANKING_JSON, "match_score": 110}
        scorer._llm.invoke.return_value.content = json.dumps(bad)
        with pytest.raises(ValueError, match="schema"):
            scorer.score_recruiter(RESUME, JD, CONTEXT, "Jane Doe")

    def test_missing_name_raises(self, scorer: ScoringService):
        bad = {k: v for k, v in VALID_RANKING_JSON.items() if k != "name"}
        scorer._llm.invoke.return_value.content = json.dumps(bad)
        with pytest.raises(ValueError, match="schema"):
            scorer.score_recruiter(RESUME, JD, CONTEXT, "Jane Doe")
