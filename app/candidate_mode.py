from app.core.config import Settings, get_settings
from app.models.schemas import CandidateAnalysis
from app.services.parser import ParserFactory
from app.services.scorer import ScoringService
from knowledge_base.ingestor import KnowledgeBaseIngestor

# ---------------------------------------------------------------------------
# What does this file do?
#
# This file is the "orchestrator" for candidate mode. It connects three
# separate services together into one clean pipeline:
#
#   PDF bytes → parser → resume text
#                                   \
#   ChromaDB   → retriever → context → scorer → CandidateAnalysis
#                                   /
#   Job description ────────────────
#
# None of parser, ingestor, or scorer know about each other.
# CandidatePipeline is the only place that knows the full flow.
# ---------------------------------------------------------------------------


class CandidatePipeline:
    """
    Orchestrates the full candidate analysis pipeline:
      1. Parse PDF → extract resume text
      2. Retrieve relevant knowledge chunks from ChromaDB
      3. Score the resume against the job description using the LLM
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

        # ParserFactory is a class with class methods — no instance needed.
        # We store it so tests can mock it if needed.
        self._parser_factory = ParserFactory

        # Create the ingestor and immediately call ingest().
        #
        # Why ingest() in __init__?
        # The knowledge base must be loaded into ChromaDB before any analysis
        # can run. If ChromaDB is empty, the retriever returns nothing and the
        # LLM gets no expert context.
        #
        # ingest() is safe to call every startup — it checks internally if
        # ChromaDB already has data and skips loading if so. On first run it
        # loads all five files. Every subsequent run returns immediately (0).
        self._ingestor = KnowledgeBaseIngestor(settings=self._settings)
        self._ingestor.ingest()

        # Scoring service wraps the Groq LLM
        self._scorer = ScoringService(settings=self._settings)

    def analyze(
        self,
        file_bytes: bytes,
        filename: str,
        job_description: str,
    ) -> CandidateAnalysis:
        """
        Run the full candidate analysis pipeline.

        Args:
            file_bytes:      Raw bytes of the uploaded PDF file.
            filename:        Original filename e.g. "resume.pdf". Used to pick
                             the correct parser for the file type.
            job_description: The job posting the candidate is applying for.

        Returns:
            A fully validated CandidateAnalysis Pydantic object.

        Raises:
            ValueError: If PDF parsing, RAG retrieval, or scoring fails.
        """

        # -------------------------------------------------------------------
        # Step 1 — Parse the PDF into plain text
        # -------------------------------------------------------------------
        # ParserFactory.get_parser() looks at the file extension and returns
        # the right parser. For "resume.pdf" it returns a PdfParser instance.
        # parser.extract_text() reads the bytes and returns a plain string.
        try:
            parser = self._parser_factory.get_parser(filename)
            resume_text = parser.extract_text(file_bytes)
        except ValueError as exc:
            raise ValueError(f"Failed to parse resume: {exc}") from exc

        # -------------------------------------------------------------------
        # Step 2 — Retrieve relevant knowledge chunks from ChromaDB
        # -------------------------------------------------------------------
        # We build a query by combining the start of the resume and the start
        # of the job description. This gives ChromaDB enough context to find
        # the most relevant chunks — e.g. if the resume mentions Python and
        # the JD mentions FAANG, it retrieves both Python tips and FAANG guide
        # chunks.
        #
        # Why only first 500/200 characters?
        # The retrieval query doesn't need to be the full text — ChromaDB
        # searches by semantic meaning, not keywords. A short representative
        # slice is enough to pull the right chunks. Using the full text would
        # make the embedding slower with no benefit.
        #
        # Why join chunks into one string?
        # The LLM prompt expects one block of "expert knowledge" text.
        # The retriever returns a list of Document objects. We join their
        # page_content into one string separated by double newlines so the
        # LLM sees them as separate paragraphs of context.
        try:
            retriever = self._ingestor.get_retriever()
            query = resume_text[:500] + " " + job_description[:200]
            docs = retriever.invoke(query)
            context = "\n\n".join(doc.page_content for doc in docs)
        except Exception as exc:
            raise ValueError(f"Failed to retrieve knowledge context: {exc}") from exc

        # -------------------------------------------------------------------
        # Step 3 — Score the resume using the LLM
        # -------------------------------------------------------------------
        # scorer.score_candidate() builds the full prompt, calls Groq,
        # parses the JSON response, and returns a validated CandidateAnalysis.
        try:
            return self._scorer.score_candidate(
                resume_text=resume_text,
                job_description=job_description,
                context=context,
            )
        except ValueError as exc:
            raise ValueError(f"Failed to score resume: {exc}") from exc
