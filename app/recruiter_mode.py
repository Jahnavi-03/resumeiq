import logging

from app.core.config import Settings, get_settings
from app.models.schemas import CandidateRanking, RecruiterAnalysis
from app.services.parser import ParserFactory
from app.services.scorer import ScoringService
from knowledge_base.ingestor import KnowledgeBaseIngestor

# ---------------------------------------------------------------------------
# What does this file do?
#
# This file orchestrates the recruiter pipeline. A recruiter uploads many
# resumes at once and wants a ranked list of candidates.
#
# For each resume:
#   PDF bytes → parser → resume text → name extraction
#                                                      \
#   ChromaDB  → retriever → context → scorer → CandidateRanking
#                                                      /
#   Job description ───────────────────────────────────
#
# After all resumes are processed:
#   Sort CandidateRankings by match_score (highest first)
#   Return RecruiterAnalysis with rankings + best match name
#
# Why process files one at a time instead of in batch?
# Each resume needs its own RAG retrieval query built from its own text.
# The context chunks should be tailored to each candidate's background.
# Batch processing would mix contexts across candidates, reducing accuracy.
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


class RecruiterPipeline:
    """
    Orchestrates the full recruiter analysis pipeline.
    Processes multiple resumes, ranks them by match score,
    and returns a RecruiterAnalysis with the full ranked list.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

        self._parser_factory = ParserFactory

        # Load knowledge base into ChromaDB on startup.
        # Safe to call every run — skips if already populated.
        self._ingestor = KnowledgeBaseIngestor(settings=self._settings)
        self._ingestor.ingest()

        self._scorer = ScoringService(settings=self._settings)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _extract_name(self, resume_text: str) -> str:
        """
        Extract the candidate's name from the top of the resume.

        How name extraction works:
        A resume almost always has the candidate's name on the very first
        line — it's the largest, most prominent piece of text. We look at
        the first 3 non-empty lines and pick the first one that:
          - Is not too long (names are rarely more than 6 words)
          - Does not look like a section header or contact detail
            (no "@", no "http", no digits like phone numbers)

        This is a heuristic — it works well for standard resumes but will
        fall back to "Unknown Candidate" for unusual formats. That is
        acceptable because the name is used only for display in rankings,
        not for any scoring logic.

        Args:
            resume_text: Full plain text extracted from the PDF.

        Returns:
            Candidate's name as a string, or "Unknown Candidate" if not found.
        """
        lines = [line.strip() for line in resume_text.splitlines() if line.strip()]

        for line in lines[:3]:  # only look at first 3 non-empty lines
            words = line.split()

            # Skip if too long — names are short, section headers can be long
            if len(words) > 6:
                continue

            # Skip if it looks like an email address
            if "@" in line:
                continue

            # Skip if it looks like a URL
            if "http" in line.lower():
                continue

            # Skip if it contains digits — could be a phone number or address
            if any(char.isdigit() for char in line):
                continue

            # First line that passes all checks is the name
            return line

        return "Unknown Candidate"

    # -----------------------------------------------------------------------
    # Main pipeline
    # -----------------------------------------------------------------------

    def analyze(
        self,
        files: list[tuple[bytes, str]],
        job_description: str,
    ) -> RecruiterAnalysis:
        """
        Process multiple resumes and return a ranked list of candidates.

        Args:
            files:           List of (file_bytes, filename) tuples. Each tuple
                             is one resume uploaded by the recruiter.
            job_description: The job posting all candidates are evaluated against.

        Returns:
            RecruiterAnalysis with candidates sorted by match_score descending.

        Raises:
            ValueError: If files list is empty, or if every resume fails to parse.
        """
        if not files:
            raise ValueError(
                "No files provided. Please upload at least one resume."
            )

        rankings: list[CandidateRanking] = []

        for file_bytes, filename in files:

            # -----------------------------------------------------------
            # Step 1 — Parse PDF → resume text
            # -----------------------------------------------------------
            # Why skip instead of crash on failure?
            # In recruiter mode the user uploads many resumes at once.
            # If one file is corrupted or accidentally a .png, we should
            # still process the other valid resumes and return results.
            # Crashing would discard all successfully processed resumes.
            # We log a warning so the recruiter knows which file failed.
            try:
                parser = self._parser_factory.get_parser(filename)
                resume_text = parser.extract_text(file_bytes)
            except ValueError as exc:
                logger.warning(
                    "Skipping '%s' — could not parse: %s", filename, exc
                )
                continue  # move on to the next file

            # -----------------------------------------------------------
            # Step 2 — Extract candidate name from top of resume
            # -----------------------------------------------------------
            candidate_name = self._extract_name(resume_text)

            # -----------------------------------------------------------
            # Step 3 — Retrieve relevant RAG context for this candidate
            # -----------------------------------------------------------
            try:
                retriever = self._ingestor.get_retriever()
                query = resume_text[:500] + " " + job_description[:200]
                docs = retriever.invoke(query)
                context = "\n\n".join(doc.page_content for doc in docs)
            except Exception as exc:
                logger.warning(
                    "Skipping '%s' — context retrieval failed: %s", filename, exc
                )
                continue

            # -----------------------------------------------------------
            # Step 4 — Score candidate against job description
            # -----------------------------------------------------------
            try:
                ranking = self._scorer.score_recruiter(
                    resume_text=resume_text,
                    job_description=job_description,
                    context=context,
                    candidate_name=candidate_name,
                )
            except ValueError as exc:
                logger.warning(
                    "Skipping '%s' — scoring failed: %s", filename, exc
                )
                continue

            rankings.append(ranking)

        # -----------------------------------------------------------------------
        # After all files processed — check we have at least one result
        # -----------------------------------------------------------------------
        if not rankings:
            raise ValueError(
                "All uploaded resumes failed to process. "
                "Please check that the files are valid text-based PDFs."
            )

        # -----------------------------------------------------------------------
        # Sort candidates by match_score highest first.
        #
        # sorted() returns a new list — it does not modify rankings in place.
        # key=lambda r: r.match_score extracts the score from each CandidateRanking.
        # reverse=True means highest score comes first (descending order).
        # -----------------------------------------------------------------------
        sorted_rankings = sorted(
            rankings,
            key=lambda r: r.match_score,
            reverse=True,
        )

        # The best match is always the first item after sorting
        best_match = sorted_rankings[0].name

        return RecruiterAnalysis(
            rankings=sorted_rankings,
            total_candidates=len(sorted_rankings),
            best_match=best_match,
        )
