import logging

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.candidate_mode import CandidatePipeline
from app.core.config import Settings, get_settings
from app.models.schemas import CandidateAnalysis, HealthCheck, RecruiterAnalysis
from app.recruiter_mode import RecruiterPipeline

# ---------------------------------------------------------------------------
# What does this file do?
#
# This is the FastAPI entry point. It creates the app, registers CORS, and
# defines three endpoints:
#
#   GET  /health               → returns { status: "ok" }
#   POST /analyze/candidate    → one resume + JD → CandidateAnalysis
#   POST /analyze/recruiter    → many resumes + JD → RecruiterAnalysis
#
# Each endpoint:
#   1. Validates the uploaded file size before reading the full bytes.
#   2. Calls the appropriate pipeline (CandidatePipeline / RecruiterPipeline).
#   3. Returns a typed Pydantic response.
#   4. Maps known errors to specific HTTP status codes.
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ResumeIQ API",
    description="AI-powered dual mode resume analyzer",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS
#
# Why do we need CORS?
# The Streamlit frontend runs on http://localhost:8501.
# The FastAPI backend runs on http://localhost:8000.
# Browsers block cross-origin requests by default (different ports = different
# origins). CORSMiddleware tells the browser: "this server allows requests
# from any origin" so Streamlit can call our API without being blocked.
#
# allow_origins=["*"]      → accept requests from any domain
# allow_credentials=True   → allow cookies/auth headers if needed later
# allow_methods=["*"]      → allow GET, POST, etc.
# allow_headers=["*"]      → allow any request header
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthCheck)
def health() -> HealthCheck:
    """
    Simple health check endpoint.
    Returns { status: "ok" } so monitoring tools and the frontend can verify
    the API is running before submitting a resume.
    """
    return HealthCheck(status="ok")


@app.post("/analyze/candidate", response_model=CandidateAnalysis)
async def analyze_candidate(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    settings: Settings = Depends(get_settings),
) -> CandidateAnalysis:
    """
    Analyze a single resume against a job description (candidate mode).

    Why use UploadFile + Form instead of JSON?
    PDF files are binary data — they cannot be sent as JSON.
    Multipart form allows one request to carry both the binary file (resume)
    and text fields (job_description) at the same time.

    File size check happens here, before reading all bytes, because:
    - UploadFile gives us the file size via resume.size without loading the
      full file into memory first.
    - Rejecting early (HTTP 413) avoids wasting memory and time on huge files.
    """
    # -------------------------------------------------------------------
    # File size validation
    # -------------------------------------------------------------------
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if resume.size and resume.size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File '{resume.filename}' is too large. "
                f"Maximum allowed size is {settings.max_upload_mb} MB."
            ),
        )

    file_bytes = await resume.read()

    # Check size again after reading in case resume.size was not set
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File '{resume.filename}' is too large. "
                f"Maximum allowed size is {settings.max_upload_mb} MB."
            ),
        )

    # -------------------------------------------------------------------
    # Run pipeline
    # -------------------------------------------------------------------
    try:
        pipeline = CandidatePipeline(settings=settings)
        return pipeline.analyze(file_bytes, resume.filename or "resume.pdf", job_description)
    except ValueError as exc:
        # ValueError means something we expected went wrong:
        # bad PDF, empty resume, LLM returned invalid JSON, etc.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        # Unexpected errors — log the full traceback for debugging
        logger.exception("Unexpected error in /analyze/candidate")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again.",
        ) from exc


@app.post("/analyze/recruiter", response_model=RecruiterAnalysis)
async def analyze_recruiter(
    resumes: list[UploadFile] = File(...),
    job_description: str = Form(...),
    settings: Settings = Depends(get_settings),
) -> RecruiterAnalysis:
    """
    Analyze multiple resumes against a job description (recruiter mode).

    Accepts a list of uploaded files. Each file is size-checked, then all
    valid files are passed to RecruiterPipeline which returns a ranked list.
    """
    max_bytes = settings.max_upload_mb * 1024 * 1024

    # -------------------------------------------------------------------
    # Read and validate each file
    # -------------------------------------------------------------------
    files: list[tuple[bytes, str]] = []

    for upload in resumes:
        if upload.size and upload.size > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File '{upload.filename}' is too large. "
                    f"Maximum allowed size is {settings.max_upload_mb} MB."
                ),
            )

        file_bytes = await upload.read()

        if len(file_bytes) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File '{upload.filename}' is too large. "
                    f"Maximum allowed size is {settings.max_upload_mb} MB."
                ),
            )

        files.append((file_bytes, upload.filename or "resume.pdf"))

    # -------------------------------------------------------------------
    # Run pipeline
    # -------------------------------------------------------------------
    try:
        pipeline = RecruiterPipeline(settings=settings)
        return pipeline.analyze(files, job_description)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /analyze/recruiter")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again.",
        ) from exc
