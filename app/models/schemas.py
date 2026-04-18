from typing import Literal
from pydantic import BaseModel, Field, computed_field, model_validator


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

# Scores that go up to 10 (bullet quality, format)
Score10 = Field(ge=1, le=10)

# Scores that go up to 100 (ATS, keyword match, projected)
Score100 = Field(ge=0, le=100)


# ---------------------------------------------------------------------------
# Bullet point analysis
# ---------------------------------------------------------------------------

class BulletAnalysis(BaseModel):
    # The exact bullet text copied from the resume
    original_bullet: str

    # Quality of the original bullet: 1-3 weak, 4-6 average, 7-9 strong, 10 perfect
    original_score: int = Score10

    # The improved version written by the AI with metrics and action verbs
    rewritten_bullet: str

    # Quality of the rewritten bullet (should always be >= original_score)
    rewritten_score: int = Score10

    # What was wrong with the original (e.g. "No metric, weak verb")
    issue: str

    # What the rewrite added (e.g. "Added quantified impact and XYZ format")
    improvement_notes: str

    # How urgently this bullet needs fixing
    priority: Literal["High", "Medium", "Low"]

    @computed_field  # auto-calculated, never set manually
    @property
    def improvement(self) -> int:
        """Points gained: rewritten_score minus original_score."""
        return self.rewritten_score - self.original_score


# ---------------------------------------------------------------------------
# Candidate mode — single resume analysed against one job description
# ---------------------------------------------------------------------------

class CandidateAnalysis(BaseModel):
    # Overall ATS compatibility score (0-100)
    ats_score: float = Score100

    # Percentage of job description keywords found in the resume (0-100)
    keyword_match: float = Score100

    # How ATS-friendly the resume format is (0-10)
    format_score: float = Field(ge=0, le=10)

    # Required skills listed in the JD that are absent from the resume
    missing_required_skills: list[str]

    # Nice-to-have skills listed in the JD that are absent from the resume
    missing_preferred_skills: list[str]

    # Formatting problems that confuse ATS parsers (e.g. "Two-column layout detected")
    formatting_issues: list[str]

    # Per-bullet breakdown with scores and rewrites
    bullet_analyses: list[BulletAnalysis]

    # High-level action items the candidate should take
    suggestions: list[str]

    # Estimated score the candidate could reach after acting on all suggestions
    projected_score: float = Score100


# ---------------------------------------------------------------------------
# Recruiter mode — one candidate entry inside a ranked list
# ---------------------------------------------------------------------------

class CandidateRanking(BaseModel):
    # Candidate's full name extracted from the resume
    name: str

    # How well this candidate matches the job description (0-100)
    match_score: float = Score100

    # Top strengths this candidate has relative to the JD
    strengths: list[str]

    # Skills or experiences the candidate is missing
    skill_gaps: list[str]

    # Hiring recommendation based on match score and analysis
    recommendation: Literal["Strong Yes", "Yes", "Maybe", "No"]


# ---------------------------------------------------------------------------
# Recruiter mode — full response returned after uploading many resumes
# ---------------------------------------------------------------------------

class RecruiterAnalysis(BaseModel):
    # All candidates sorted highest match_score first
    rankings: list[CandidateRanking]

    # Total number of resumes that were processed
    total_candidates: int

    # Name of the candidate with the highest match score
    best_match: str

    @model_validator(mode="after")
    def best_match_must_exist_in_rankings(self) -> "RecruiterAnalysis":
        """Ensure best_match is actually one of the ranked candidates."""
        names = [c.name for c in self.rankings]
        if self.best_match not in names:
            raise ValueError(
                f"best_match '{self.best_match}' is not in the rankings list"
            )
        return self


# ---------------------------------------------------------------------------
# Health check — confirms the API is alive
# ---------------------------------------------------------------------------

class HealthCheck(BaseModel):
    # Always "ok" when the service is running normally
    status: str = "ok"
