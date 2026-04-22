import io
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import app
from app.models.schemas import (
    BulletAnalysis,
    CandidateAnalysis,
    CandidateRanking,
    RecruiterAnalysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_settings() -> Settings:
    return Settings(
        groq_api_key="test-key",
        embedding_model="all-MiniLM-L6-v2",
        chroma_db_path="/tmp/test_chroma",
        max_upload_mb=5,
        _env_file=None,
    )


def _fake_candidate_analysis() -> CandidateAnalysis:
    return CandidateAnalysis(
        ats_score=75,
        keyword_match=68,
        format_score=8,
        missing_required_skills=["Kubernetes"],
        missing_preferred_skills=["GraphQL"],
        formatting_issues=["Two-column layout"],
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
        suggestions=["Add metrics"],
        projected_score=88,
    )


def _fake_recruiter_analysis() -> RecruiterAnalysis:
    return RecruiterAnalysis(
        rankings=[
            CandidateRanking(
                name="Jane Doe",
                match_score=91,
                strengths=["Python"],
                skill_gaps=["Kubernetes"],
                recommendation="Strong Yes",
            ),
            CandidateRanking(
                name="John Smith",
                match_score=72,
                strengths=["SQL"],
                skill_gaps=["FastAPI"],
                recommendation="Maybe",
            ),
        ],
        total_candidates=2,
        best_match="Jane Doe",
    )


FAKE_PDF = b"%PDF-1.4 fake content"
JD = "Senior Backend Engineer. Required: Python, Kubernetes."


# ---------------------------------------------------------------------------
# Fixture — TestClient with get_settings overridden
# ---------------------------------------------------------------------------

@pytest.fixture
def client() -> TestClient:
    """
    FastAPI TestClient with get_settings dependency overridden so no real
    .env file is needed.
    """
    app.dependency_overrides[__import__("app.core.config", fromlist=["get_settings"]).get_settings] = _fake_settings
    yield TestClient(app)
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_ok_status(self, client: TestClient):
        response = client.get("/health")
        assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# POST /analyze/candidate — happy path
# ---------------------------------------------------------------------------

class TestCandidateEndpointSuccess:
    @pytest.fixture(autouse=True)
    def mock_pipeline(self):
        """Patch CandidatePipeline so no real services are instantiated."""
        with patch("app.main.CandidatePipeline") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.analyze.return_value = _fake_candidate_analysis()
            mock_cls.return_value = mock_instance
            self.mock_cls = mock_cls
            self.mock_instance = mock_instance
            yield

    def _post(self, client: TestClient, pdf: bytes = FAKE_PDF, filename: str = "resume.pdf"):
        return client.post(
            "/analyze/candidate",
            data={"job_description": JD},
            files={"resume": (filename, io.BytesIO(pdf), "application/pdf")},
        )

    def test_returns_200(self, client: TestClient):
        assert self._post(client).status_code == 200

    def test_returns_candidate_analysis(self, client: TestClient):
        body = self._post(client).json()
        assert "ats_score" in body
        assert body["ats_score"] == 75

    def test_pipeline_analyze_called_once(self, client: TestClient):
        self._post(client)
        self.mock_instance.analyze.assert_called_once()

    def test_pipeline_receives_job_description(self, client: TestClient):
        self._post(client)
        call_kwargs = self.mock_instance.analyze.call_args
        # analyze(file_bytes, filename, job_description) — positional args
        assert JD in call_kwargs.args or JD in call_kwargs.kwargs.values()

    def test_pipeline_receives_filename(self, client: TestClient):
        self._post(client, filename="my_cv.pdf")
        call_args = self.mock_instance.analyze.call_args.args
        assert "my_cv.pdf" in call_args


# ---------------------------------------------------------------------------
# POST /analyze/candidate — error handling
# ---------------------------------------------------------------------------

class TestCandidateEndpointErrors:
    def test_value_error_returns_422(self, client: TestClient):
        with patch("app.main.CandidatePipeline") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.analyze.side_effect = ValueError("bad PDF")
            mock_cls.return_value = mock_instance

            response = client.post(
                "/analyze/candidate",
                data={"job_description": JD},
                files={"resume": ("resume.pdf", io.BytesIO(FAKE_PDF), "application/pdf")},
            )
            assert response.status_code == 422

    def test_unexpected_error_returns_500(self, client: TestClient):
        with patch("app.main.CandidatePipeline") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.analyze.side_effect = RuntimeError("something exploded")
            mock_cls.return_value = mock_instance

            response = client.post(
                "/analyze/candidate",
                data={"job_description": JD},
                files={"resume": ("resume.pdf", io.BytesIO(FAKE_PDF), "application/pdf")},
            )
            assert response.status_code == 500

    def test_oversized_file_returns_413(self, client: TestClient):
        # max_upload_mb=5, so 6 MB should trigger 413
        big_file = b"x" * (6 * 1024 * 1024)
        with patch("app.main.CandidatePipeline"):
            response = client.post(
                "/analyze/candidate",
                data={"job_description": JD},
                files={"resume": ("big.pdf", io.BytesIO(big_file), "application/pdf")},
            )
        assert response.status_code == 413

    def test_413_detail_mentions_filename(self, client: TestClient):
        big_file = b"x" * (6 * 1024 * 1024)
        with patch("app.main.CandidatePipeline"):
            response = client.post(
                "/analyze/candidate",
                data={"job_description": JD},
                files={"resume": ("big.pdf", io.BytesIO(big_file), "application/pdf")},
            )
        assert "big.pdf" in response.json()["detail"]

    def test_missing_job_description_returns_422(self, client: TestClient):
        # job_description is required — omitting it should fail validation
        response = client.post(
            "/analyze/candidate",
            files={"resume": ("resume.pdf", io.BytesIO(FAKE_PDF), "application/pdf")},
        )
        assert response.status_code == 422

    def test_missing_resume_returns_422(self, client: TestClient):
        response = client.post(
            "/analyze/candidate",
            data={"job_description": JD},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /analyze/recruiter — happy path
# ---------------------------------------------------------------------------

class TestRecruiterEndpointSuccess:
    @pytest.fixture(autouse=True)
    def mock_pipeline(self):
        with patch("app.main.RecruiterPipeline") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.analyze.return_value = _fake_recruiter_analysis()
            mock_cls.return_value = mock_instance
            self.mock_instance = mock_instance
            yield

    def _post(self, client: TestClient, n_files: int = 2):
        files = [
            ("resumes", (f"candidate_{i}.pdf", io.BytesIO(FAKE_PDF), "application/pdf"))
            for i in range(n_files)
        ]
        return client.post(
            "/analyze/recruiter",
            data={"job_description": JD},
            files=files,
        )

    def test_returns_200(self, client: TestClient):
        assert self._post(client).status_code == 200

    def test_returns_recruiter_analysis(self, client: TestClient):
        body = self._post(client).json()
        assert "rankings" in body
        assert "best_match" in body
        assert "total_candidates" in body

    def test_total_candidates_is_correct(self, client: TestClient):
        body = self._post(client).json()
        assert body["total_candidates"] == 2

    def test_best_match_field_present(self, client: TestClient):
        body = self._post(client).json()
        assert body["best_match"] == "Jane Doe"

    def test_single_file_returns_200(self, client: TestClient):
        assert self._post(client, n_files=1).status_code == 200

    def test_pipeline_analyze_called_once(self, client: TestClient):
        self._post(client)
        self.mock_instance.analyze.assert_called_once()


# ---------------------------------------------------------------------------
# POST /analyze/recruiter — error handling
# ---------------------------------------------------------------------------

class TestRecruiterEndpointErrors:
    def test_value_error_returns_422(self, client: TestClient):
        with patch("app.main.RecruiterPipeline") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.analyze.side_effect = ValueError("All files failed")
            mock_cls.return_value = mock_instance

            response = client.post(
                "/analyze/recruiter",
                data={"job_description": JD},
                files=[("resumes", ("a.pdf", io.BytesIO(FAKE_PDF), "application/pdf"))],
            )
            assert response.status_code == 422

    def test_unexpected_error_returns_500(self, client: TestClient):
        with patch("app.main.RecruiterPipeline") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.analyze.side_effect = RuntimeError("crash")
            mock_cls.return_value = mock_instance

            response = client.post(
                "/analyze/recruiter",
                data={"job_description": JD},
                files=[("resumes", ("a.pdf", io.BytesIO(FAKE_PDF), "application/pdf"))],
            )
            assert response.status_code == 500

    def test_oversized_file_returns_413(self, client: TestClient):
        big_file = b"x" * (6 * 1024 * 1024)
        with patch("app.main.RecruiterPipeline"):
            response = client.post(
                "/analyze/recruiter",
                data={"job_description": JD},
                files=[("resumes", ("big.pdf", io.BytesIO(big_file), "application/pdf"))],
            )
        assert response.status_code == 413

    def test_missing_job_description_returns_422(self, client: TestClient):
        response = client.post(
            "/analyze/recruiter",
            files=[("resumes", ("a.pdf", io.BytesIO(FAKE_PDF), "application/pdf"))],
        )
        assert response.status_code == 422
