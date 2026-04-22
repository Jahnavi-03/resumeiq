import pytest
from unittest.mock import MagicMock, patch

from app.candidate_mode import CandidatePipeline
from app.core.config import Settings
from app.models.schemas import BulletAnalysis, CandidateAnalysis


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


def _fake_candidate_analysis() -> CandidateAnalysis:
    """A valid CandidateAnalysis object the mock scorer returns."""
    return CandidateAnalysis(
        ats_score=75,
        keyword_match=68,
        format_score=8,
        missing_required_skills=["Kubernetes"],
        missing_preferred_skills=["GraphQL"],
        formatting_issues=["Two-column layout detected"],
        bullet_analyses=[
            BulletAnalysis(
                original_bullet="Led a team",
                original_score=3,
                rewritten_bullet="Led cross-functional team of 8 engineers",
                rewritten_score=9,
                issue="No metric",
                improvement_notes="Added team size",
                priority="High",
            )
        ],
        suggestions=["Add metrics", "Add Kubernetes"],
        projected_score=88,
    )


FAKE_PDF_BYTES = b"%PDF-1.4 fake content"
FAKE_RESUME_TEXT = "Jane Doe\nSoftware Engineer\nPython, FastAPI"
FAKE_JD = "Senior Backend Engineer. Required: Python, Kubernetes."


# ---------------------------------------------------------------------------
# Fixture — CandidatePipeline with all three services mocked
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline() -> CandidatePipeline:
    """
    CandidatePipeline with ParserFactory, KnowledgeBaseIngestor, and
    ScoringService all replaced with mocks so no real I/O happens.
    """
    with patch("app.candidate_mode.ParserFactory") as mock_factory, \
         patch("app.candidate_mode.KnowledgeBaseIngestor") as mock_ingestor_cls, \
         patch("app.candidate_mode.ScoringService") as mock_scorer_cls:

        # Set up parser mock — get_parser() returns a parser whose
        # extract_text() returns our fake resume text
        mock_parser = MagicMock()
        mock_parser.extract_text.return_value = FAKE_RESUME_TEXT
        mock_factory.get_parser.return_value = mock_parser

        # Set up ingestor mock — ingest() does nothing, get_retriever()
        # returns a retriever whose invoke() returns one fake document
        mock_ingestor = MagicMock()
        mock_ingestor.ingest.return_value = 0
        fake_doc = MagicMock()
        fake_doc.page_content = "ATS tip: use standard headers."
        mock_ingestor.get_retriever.return_value.invoke.return_value = [fake_doc]
        mock_ingestor_cls.return_value = mock_ingestor

        # Set up scorer mock — score_candidate() returns a valid analysis
        mock_scorer = MagicMock()
        mock_scorer.score_candidate.return_value = _fake_candidate_analysis()
        mock_scorer_cls.return_value = mock_scorer

        pl = CandidatePipeline(settings=_fake_settings())

    return pl


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_ingest_is_called_on_startup(self):
        """ingest() must be called in __init__ to ensure ChromaDB is loaded."""
        with patch("app.candidate_mode.ParserFactory"), \
             patch("app.candidate_mode.KnowledgeBaseIngestor") as mock_ingestor_cls, \
             patch("app.candidate_mode.ScoringService"):

            mock_ingestor = MagicMock()
            mock_ingestor_cls.return_value = mock_ingestor

            CandidatePipeline(settings=_fake_settings())

            mock_ingestor.ingest.assert_called_once()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestAnalyzeSuccess:
    def test_returns_candidate_analysis(self, pipeline: CandidatePipeline):
        result = pipeline.analyze(FAKE_PDF_BYTES, "resume.pdf", FAKE_JD)
        assert isinstance(result, CandidateAnalysis)

    def test_ats_score_is_correct(self, pipeline: CandidatePipeline):
        result = pipeline.analyze(FAKE_PDF_BYTES, "resume.pdf", FAKE_JD)
        assert result.ats_score == 75

    def test_parser_receives_correct_filename(self, pipeline: CandidatePipeline):
        pipeline.analyze(FAKE_PDF_BYTES, "resume.pdf", FAKE_JD)
        pipeline._parser_factory.get_parser.assert_called_once_with("resume.pdf")

    def test_parser_receives_correct_bytes(self, pipeline: CandidatePipeline):
        pipeline.analyze(FAKE_PDF_BYTES, "resume.pdf", FAKE_JD)
        pipeline._ingestor.get_retriever.return_value.invoke  # retriever was called
        parser = pipeline._parser_factory.get_parser.return_value
        parser.extract_text.assert_called_once_with(FAKE_PDF_BYTES)

    def test_scorer_receives_resume_text(self, pipeline: CandidatePipeline):
        pipeline.analyze(FAKE_PDF_BYTES, "resume.pdf", FAKE_JD)
        call_kwargs = pipeline._scorer.score_candidate.call_args.kwargs
        assert call_kwargs["resume_text"] == FAKE_RESUME_TEXT

    def test_scorer_receives_job_description(self, pipeline: CandidatePipeline):
        pipeline.analyze(FAKE_PDF_BYTES, "resume.pdf", FAKE_JD)
        call_kwargs = pipeline._scorer.score_candidate.call_args.kwargs
        assert call_kwargs["job_description"] == FAKE_JD

    def test_scorer_receives_context_string(self, pipeline: CandidatePipeline):
        pipeline.analyze(FAKE_PDF_BYTES, "resume.pdf", FAKE_JD)
        call_kwargs = pipeline._scorer.score_candidate.call_args.kwargs
        # context must be a non-empty string (joined from retrieved docs)
        assert isinstance(call_kwargs["context"], str)
        assert len(call_kwargs["context"]) > 0

    def test_retriever_is_called_with_combined_query(self, pipeline: CandidatePipeline):
        pipeline.analyze(FAKE_PDF_BYTES, "resume.pdf", FAKE_JD)
        retriever = pipeline._ingestor.get_retriever.return_value
        call_args = retriever.invoke.call_args[0][0]
        # query must contain content from both resume and JD
        assert FAKE_RESUME_TEXT[:10] in call_args
        assert FAKE_JD[:10] in call_args


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestAnalyzeErrors:
    def test_pdf_parsing_error_raises_value_error(self, pipeline: CandidatePipeline):
        pipeline._parser_factory.get_parser.return_value.extract_text.side_effect = \
            ValueError("not a valid PDF")
        with pytest.raises(ValueError, match="Failed to parse resume"):
            pipeline.analyze(FAKE_PDF_BYTES, "resume.pdf", FAKE_JD)

    def test_unsupported_file_type_raises(self, pipeline: CandidatePipeline):
        pipeline._parser_factory.get_parser.side_effect = \
            ValueError("Unsupported file type '.docx'")
        with pytest.raises(ValueError, match="Failed to parse resume"):
            pipeline.analyze(b"some bytes", "resume.docx", FAKE_JD)

    def test_retrieval_error_raises_value_error(self, pipeline: CandidatePipeline):
        pipeline._ingestor.get_retriever.return_value.invoke.side_effect = \
            Exception("ChromaDB connection lost")
        with pytest.raises(ValueError, match="Failed to retrieve knowledge context"):
            pipeline.analyze(FAKE_PDF_BYTES, "resume.pdf", FAKE_JD)

    def test_scoring_error_raises_value_error(self, pipeline: CandidatePipeline):
        pipeline._scorer.score_candidate.side_effect = \
            ValueError("LLM returned invalid JSON")
        with pytest.raises(ValueError, match="Failed to score resume"):
            pipeline.analyze(FAKE_PDF_BYTES, "resume.pdf", FAKE_JD)

    def test_parse_error_does_not_call_scorer(self, pipeline: CandidatePipeline):
        pipeline._parser_factory.get_parser.return_value.extract_text.side_effect = \
            ValueError("empty file")
        with pytest.raises(ValueError):
            pipeline.analyze(FAKE_PDF_BYTES, "resume.pdf", FAKE_JD)
        pipeline._scorer.score_candidate.assert_not_called()
