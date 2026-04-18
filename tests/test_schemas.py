import pytest
from pydantic import ValidationError
from app.models.schemas import (
    BulletAnalysis,
    CandidateAnalysis,
    CandidateRanking,
    RecruiterAnalysis,
    HealthCheck,
)


# ---------------------------------------------------------------------------
# Helpers — valid base data reused across tests
# ---------------------------------------------------------------------------

def make_bullet(**overrides) -> dict:
    base = {
        "original_bullet": "Worked on backend systems",
        "original_score": 3,
        "rewritten_bullet": "Reduced API latency by 40% by refactoring 3 backend services",
        "rewritten_score": 9,
        "issue": "No metric, weak verb",
        "improvement_notes": "Added quantified impact and XYZ format",
        "priority": "High",
    }
    return {**base, **overrides}


def make_candidate_analysis(**overrides) -> dict:
    base = {
        "ats_score": 72.0,
        "keyword_match": 65.0,
        "format_score": 7.5,
        "missing_required_skills": ["Kubernetes", "Terraform"],
        "missing_preferred_skills": ["GraphQL"],
        "formatting_issues": ["Two-column layout detected"],
        "bullet_analyses": [BulletAnalysis(**make_bullet())],
        "suggestions": ["Add metrics to bullets", "Remove tables"],
        "projected_score": 88.0,
    }
    return {**base, **overrides}


def make_candidate_ranking(**overrides) -> dict:
    base = {
        "name": "Alice Johnson",
        "match_score": 85.0,
        "strengths": ["Strong Python skills", "AWS experience", "Led teams"],
        "skill_gaps": ["No Kubernetes", "Missing CI/CD", "No Terraform"],
        "recommendation": "Strong Yes",
    }
    return {**base, **overrides}


def make_recruiter_analysis(**overrides) -> dict:
    base = {
        "rankings": [
            CandidateRanking(**make_candidate_ranking()),
            CandidateRanking(**make_candidate_ranking(
                name="Bob Smith", match_score=70.0, recommendation="Yes"
            )),
        ],
        "total_candidates": 2,
        "best_match": "Alice Johnson",
    }
    return {**base, **overrides}


# ---------------------------------------------------------------------------
# BulletAnalysis
# ---------------------------------------------------------------------------

class TestBulletAnalysis:
    def test_valid_bullet(self):
        b = BulletAnalysis(**make_bullet())
        assert b.original_score == 3
        assert b.rewritten_score == 9
        assert b.priority == "High"

    def test_improvement_is_computed_automatically(self):
        b = BulletAnalysis(**make_bullet(original_score=4, rewritten_score=9))
        # improvement must equal rewritten_score - original_score, never set manually
        assert b.improvement == 5

    def test_improvement_can_be_zero(self):
        b = BulletAnalysis(**make_bullet(original_score=7, rewritten_score=7))
        assert b.improvement == 0

    def test_original_score_below_minimum_raises(self):
        with pytest.raises(ValidationError):
            BulletAnalysis(**make_bullet(original_score=0))  # min is 1

    def test_original_score_above_maximum_raises(self):
        with pytest.raises(ValidationError):
            BulletAnalysis(**make_bullet(original_score=11))  # max is 10

    def test_rewritten_score_below_minimum_raises(self):
        with pytest.raises(ValidationError):
            BulletAnalysis(**make_bullet(rewritten_score=0))

    def test_rewritten_score_above_maximum_raises(self):
        with pytest.raises(ValidationError):
            BulletAnalysis(**make_bullet(rewritten_score=11))

    def test_invalid_priority_raises(self):
        with pytest.raises(ValidationError):
            BulletAnalysis(**make_bullet(priority="Critical"))  # not in Literal

    def test_all_valid_priorities(self):
        for priority in ("High", "Medium", "Low"):
            b = BulletAnalysis(**make_bullet(priority=priority))
            assert b.priority == priority

    def test_missing_required_field_raises(self):
        data = make_bullet()
        del data["original_bullet"]
        with pytest.raises(ValidationError):
            BulletAnalysis(**data)


# ---------------------------------------------------------------------------
# CandidateAnalysis
# ---------------------------------------------------------------------------

class TestCandidateAnalysis:
    def test_valid_candidate_analysis(self):
        c = CandidateAnalysis(**make_candidate_analysis())
        assert c.ats_score == 72.0
        assert c.keyword_match == 65.0
        assert c.format_score == 7.5
        assert len(c.missing_required_skills) == 2
        assert len(c.bullet_analyses) == 1

    def test_ats_score_above_100_raises(self):
        with pytest.raises(ValidationError):
            CandidateAnalysis(**make_candidate_analysis(ats_score=101))

    def test_ats_score_below_0_raises(self):
        with pytest.raises(ValidationError):
            CandidateAnalysis(**make_candidate_analysis(ats_score=-1))

    def test_keyword_match_boundary_values(self):
        c_min = CandidateAnalysis(**make_candidate_analysis(keyword_match=0))
        c_max = CandidateAnalysis(**make_candidate_analysis(keyword_match=100))
        assert c_min.keyword_match == 0
        assert c_max.keyword_match == 100

    def test_format_score_above_10_raises(self):
        with pytest.raises(ValidationError):
            CandidateAnalysis(**make_candidate_analysis(format_score=10.1))

    def test_projected_score_above_100_raises(self):
        with pytest.raises(ValidationError):
            CandidateAnalysis(**make_candidate_analysis(projected_score=100.1))

    def test_empty_lists_are_valid(self):
        # A perfect resume may have no missing skills or issues
        c = CandidateAnalysis(**make_candidate_analysis(
            missing_required_skills=[],
            missing_preferred_skills=[],
            formatting_issues=[],
        ))
        assert c.missing_required_skills == []

    def test_missing_ats_score_raises(self):
        data = make_candidate_analysis()
        del data["ats_score"]
        with pytest.raises(ValidationError):
            CandidateAnalysis(**data)


# ---------------------------------------------------------------------------
# CandidateRanking
# ---------------------------------------------------------------------------

class TestCandidateRanking:
    def test_valid_ranking(self):
        r = CandidateRanking(**make_candidate_ranking())
        assert r.name == "Alice Johnson"
        assert r.match_score == 85.0
        assert r.recommendation == "Strong Yes"

    def test_all_valid_recommendations(self):
        for rec in ("Strong Yes", "Yes", "Maybe", "No"):
            r = CandidateRanking(**make_candidate_ranking(recommendation=rec))
            assert r.recommendation == rec

    def test_invalid_recommendation_raises(self):
        with pytest.raises(ValidationError):
            CandidateRanking(**make_candidate_ranking(recommendation="Never"))

    def test_match_score_above_100_raises(self):
        with pytest.raises(ValidationError):
            CandidateRanking(**make_candidate_ranking(match_score=101))

    def test_match_score_below_0_raises(self):
        with pytest.raises(ValidationError):
            CandidateRanking(**make_candidate_ranking(match_score=-5))

    def test_strengths_accepts_more_than_3_items(self):
        # No cap — the LLM prompt controls how many items are returned
        r = CandidateRanking(**make_candidate_ranking(
            strengths=["A", "B", "C", "D"]
        ))
        assert len(r.strengths) == 4

    def test_skill_gaps_accepts_more_than_3_items(self):
        r = CandidateRanking(**make_candidate_ranking(
            skill_gaps=["X", "Y", "Z", "W"]
        ))
        assert len(r.skill_gaps) == 4

    def test_missing_name_raises(self):
        data = make_candidate_ranking()
        del data["name"]
        with pytest.raises(ValidationError):
            CandidateRanking(**data)


# ---------------------------------------------------------------------------
# RecruiterAnalysis
# ---------------------------------------------------------------------------

class TestRecruiterAnalysis:
    def test_valid_recruiter_analysis(self):
        r = RecruiterAnalysis(**make_recruiter_analysis())
        assert r.total_candidates == 2
        assert r.best_match == "Alice Johnson"
        assert len(r.rankings) == 2

    def test_best_match_not_in_rankings_raises(self):
        with pytest.raises(ValidationError):
            RecruiterAnalysis(**make_recruiter_analysis(best_match="Ghost Candidate"))

    def test_missing_total_candidates_raises(self):
        data = make_recruiter_analysis()
        del data["total_candidates"]
        with pytest.raises(ValidationError):
            RecruiterAnalysis(**data)

    def test_empty_rankings_with_best_match_raises(self):
        with pytest.raises(ValidationError):
            RecruiterAnalysis(rankings=[], total_candidates=0, best_match="Alice Johnson")


# ---------------------------------------------------------------------------
# HealthCheck
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_default_status_is_ok(self):
        h = HealthCheck()
        assert h.status == "ok"

    def test_custom_status(self):
        h = HealthCheck(status="degraded")
        assert h.status == "degraded"
