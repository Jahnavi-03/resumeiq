import pytest
from unittest.mock import MagicMock, patch

from app.core.config import Settings
from app.models.schemas import CandidateRanking, RecruiterAnalysis
from app.recruiter_mode import RecruiterPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_settings() -> Settings:
    return Settings(
        groq_api_key="test-key",
        embedding_model="all-MiniLM-L6-v2",
        chroma_db_path="/tmp/test_chroma",
        _env_file=None,
    )


def _fake_ranking(name: str, match_score: float) -> CandidateRanking:
    """Build a valid CandidateRanking with the given name and score."""
    return CandidateRanking(
        name=name,
        match_score=match_score,
        strengths=["Python", "FastAPI"],
        skill_gaps=["Kubernetes"],
        recommendation="Yes",
    )


FAKE_PDF = b"%PDF-1.4 fake"
JD = "Senior Backend Engineer. Required: Python, Kubernetes."


# ---------------------------------------------------------------------------
# Fixture — RecruiterPipeline with all services mocked
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline() -> RecruiterPipeline:
    with patch("app.recruiter_mode.ParserFactory") as mock_factory, \
         patch("app.recruiter_mode.KnowledgeBaseIngestor") as mock_ingestor_cls, \
         patch("app.recruiter_mode.ScoringService") as mock_scorer_cls:

        # Parser returns different resume text per filename
        def get_parser_side_effect(filename):
            mock_parser = MagicMock()
            mock_parser.extract_text.return_value = f"Resume text for {filename}"
            return mock_parser

        mock_factory.get_parser.side_effect = get_parser_side_effect

        # Ingestor — ingest skips, retriever returns one fake doc
        mock_ingestor = MagicMock()
        mock_ingestor.ingest.return_value = 0
        fake_doc = MagicMock()
        fake_doc.page_content = "ATS tip: use standard headers."
        mock_ingestor.get_retriever.return_value.invoke.return_value = [fake_doc]
        mock_ingestor_cls.return_value = mock_ingestor

        # Scorer returns different rankings per candidate name
        def score_recruiter_side_effect(**kwargs):
            name = kwargs["candidate_name"]
            scores = {
                "Jane Doe": 85,
                "John Smith": 62,
                "Alice Lee": 91,
            }
            return _fake_ranking(name, scores.get(name, 70))

        mock_scorer = MagicMock()
        mock_scorer.score_recruiter.side_effect = score_recruiter_side_effect
        mock_scorer_cls.return_value = mock_scorer

        pl = RecruiterPipeline(settings=_fake_settings())

    return pl


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_ingest_called_on_startup(self):
        with patch("app.recruiter_mode.ParserFactory"), \
             patch("app.recruiter_mode.KnowledgeBaseIngestor") as mock_ingestor_cls, \
             patch("app.recruiter_mode.ScoringService"):

            mock_ingestor = MagicMock()
            mock_ingestor_cls.return_value = mock_ingestor
            RecruiterPipeline(settings=_fake_settings())
            mock_ingestor.ingest.assert_called_once()


# ---------------------------------------------------------------------------
# Name extraction
# ---------------------------------------------------------------------------

class TestExtractName:
    @pytest.fixture
    def bare_pipeline(self) -> RecruiterPipeline:
        """Pipeline with no side effects set — just for testing _extract_name."""
        with patch("app.recruiter_mode.ParserFactory"), \
             patch("app.recruiter_mode.KnowledgeBaseIngestor"), \
             patch("app.recruiter_mode.ScoringService"):
            return RecruiterPipeline(settings=_fake_settings())

    def test_extracts_name_from_first_line(self, bare_pipeline):
        text = "Jane Doe\nSoftware Engineer\njane@email.com"
        assert bare_pipeline._extract_name(text) == "Jane Doe"

    def test_skips_email_line(self, bare_pipeline):
        text = "jane@email.com\nJane Doe\nSoftware Engineer"
        assert bare_pipeline._extract_name(text) == "Jane Doe"

    def test_skips_line_with_digits(self, bare_pipeline):
        text = "+1 555 123 4567\nJane Doe\nEngineer"
        assert bare_pipeline._extract_name(text) == "Jane Doe"

    def test_skips_url(self, bare_pipeline):
        text = "https://linkedin.com/in/jane\nJane Doe\nEngineer"
        assert bare_pipeline._extract_name(text) == "Jane Doe"

    def test_skips_long_lines(self, bare_pipeline):
        text = "Experienced Software Engineer with ten years in backend systems\nJane Doe"
        assert bare_pipeline._extract_name(text) == "Jane Doe"

    def test_returns_unknown_when_no_name_found(self, bare_pipeline):
        text = "jane@email.com\nhttps://github.com/jane\n+1 555 000 0000"
        assert bare_pipeline._extract_name(text) == "Unknown Candidate"

    def test_handles_empty_lines_in_resume(self, bare_pipeline):
        # Leading blank lines should be skipped before looking for name
        text = "\n\nJane Doe\nSoftware Engineer"
        assert bare_pipeline._extract_name(text) == "Jane Doe"


# ---------------------------------------------------------------------------
# analyze() — happy path
# ---------------------------------------------------------------------------

class TestAnalyzeSuccess:
    def test_returns_recruiter_analysis(self, pipeline: RecruiterPipeline):
        files = [(FAKE_PDF, "jane.pdf"), (FAKE_PDF, "john.pdf")]
        result = pipeline.analyze(files, JD)
        assert isinstance(result, RecruiterAnalysis)

    def test_total_candidates_is_correct(self, pipeline: RecruiterPipeline):
        files = [(FAKE_PDF, "jane.pdf"), (FAKE_PDF, "john.pdf")]
        result = pipeline.analyze(files, JD)
        assert result.total_candidates == 2

    def test_rankings_sorted_highest_first(self, pipeline: RecruiterPipeline):
        # jane=85, john=62 → jane should come first
        files = [(FAKE_PDF, "jane.pdf"), (FAKE_PDF, "john.pdf")]

        # Override name extraction so we get predictable names
        pipeline._extract_name = lambda text: (
            "Jane Doe" if "jane" in text else "John Smith"
        )

        result = pipeline.analyze(files, JD)
        scores = [r.match_score for r in result.rankings]
        assert scores == sorted(scores, reverse=True)

    def test_best_match_is_highest_scorer(self, pipeline: RecruiterPipeline):
        files = [(FAKE_PDF, "jane.pdf"), (FAKE_PDF, "john.pdf"), (FAKE_PDF, "alice.pdf")]
        pipeline._extract_name = lambda text: (
            "Jane Doe" if "jane" in text
            else "John Smith" if "john" in text
            else "Alice Lee"
        )
        result = pipeline.analyze(files, JD)
        # Alice scores 91 — highest
        assert result.best_match == result.rankings[0].name

    def test_single_file_works(self, pipeline: RecruiterPipeline):
        result = pipeline.analyze([(FAKE_PDF, "jane.pdf")], JD)
        assert result.total_candidates == 1

    def test_scorer_called_once_per_resume(self, pipeline: RecruiterPipeline):
        files = [(FAKE_PDF, "jane.pdf"), (FAKE_PDF, "john.pdf")]
        pipeline.analyze(files, JD)
        assert pipeline._scorer.score_recruiter.call_count == 2


# ---------------------------------------------------------------------------
# analyze() — edge cases and error handling
# ---------------------------------------------------------------------------

class TestAnalyzeEdgeCases:
    def test_empty_files_list_raises(self, pipeline: RecruiterPipeline):
        with pytest.raises(ValueError, match="No files provided"):
            pipeline.analyze([], JD)

    def test_failed_parse_is_skipped(self, pipeline: RecruiterPipeline):
        # First file fails, second succeeds
        def get_parser_side_effect(filename):
            mock_parser = MagicMock()
            if "bad" in filename:
                mock_parser.extract_text.side_effect = ValueError("not a PDF")
            else:
                mock_parser.extract_text.return_value = "Jane Doe\nEngineer"
            return mock_parser

        pipeline._parser_factory.get_parser.side_effect = get_parser_side_effect
        files = [(FAKE_PDF, "bad.pdf"), (FAKE_PDF, "jane.pdf")]
        result = pipeline.analyze(files, JD)
        # Only one resume processed successfully
        assert result.total_candidates == 1

    def test_all_files_fail_raises(self, pipeline: RecruiterPipeline):
        # Replace the side_effect so every get_parser() returns a parser that fails
        def always_fail(filename):
            mock_parser = MagicMock()
            mock_parser.extract_text.side_effect = ValueError("not a PDF")
            return mock_parser

        pipeline._parser_factory.get_parser.side_effect = always_fail
        files = [(FAKE_PDF, "a.pdf"), (FAKE_PDF, "b.pdf")]
        with pytest.raises(ValueError, match="All uploaded resumes failed"):
            pipeline.analyze(files, JD)

    def test_failed_file_does_not_appear_in_rankings(self, pipeline: RecruiterPipeline):
        def get_parser_side_effect(filename):
            mock_parser = MagicMock()
            if "bad" in filename:
                mock_parser.extract_text.side_effect = ValueError("corrupted")
            else:
                mock_parser.extract_text.return_value = "Jane Doe\nEngineer"
            return mock_parser

        pipeline._parser_factory.get_parser.side_effect = get_parser_side_effect
        files = [(FAKE_PDF, "bad.pdf"), (FAKE_PDF, "jane.pdf")]
        result = pipeline.analyze(files, JD)
        names = [r.name for r in result.rankings]
        # bad.pdf should not appear at all
        assert not any("bad" in name.lower() for name in names)

    def test_scoring_failure_skips_that_candidate(self, pipeline: RecruiterPipeline):
        call_count = {"n": 0}

        def score_side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ValueError("LLM failed")
            return _fake_ranking(kwargs["candidate_name"], 75)

        pipeline._scorer.score_recruiter.side_effect = score_side_effect
        files = [(FAKE_PDF, "jane.pdf"), (FAKE_PDF, "john.pdf")]
        result = pipeline.analyze(files, JD)
        # First candidate failed scoring, only second should appear
        assert result.total_candidates == 1
