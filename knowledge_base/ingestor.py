from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app.core.config import Settings, get_settings
from app.services.embedder import EmbeddingService

# ---------------------------------------------------------------------------
# The five knowledge base files we load into ChromaDB.
# These are plain .txt files sitting in the knowledge_base/ folder.
# ---------------------------------------------------------------------------

_KB_DIR = Path(__file__).parent  # absolute path to the knowledge_base/ folder

_KB_FILES = [
    "ats_rules.txt",
    "faang_guide.txt",
    "bullet_examples.txt",
    "scoring_rubric.txt",
    "skill_taxonomy.txt",
]

_COLLECTION_NAME = "resume_knowledge"


# ---------------------------------------------------------------------------
# What is chunking?
#
# Our knowledge base files are long — hundreds of lines each. If we embedded
# an entire file as one vector, we'd lose fine-grained meaning. A query like
# "how to write bullet points?" would have to compete against everything else
# in the file.
#
# Chunking splits each file into small overlapping pieces (~500 characters).
# Each chunk gets its own vector. When a user asks a question, ChromaDB finds
# the 3 most relevant *chunks* — not the most relevant file. This gives the
# LLM precise, focused context.
#
# What is overlap?
#
# chunk 1: "...optimize your resume for ATS by using keywords from the job"
# chunk 2: "using keywords from the job description in your skills section..."
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#           overlap (repeated from end of chunk 1)
#
# Without overlap, a sentence that straddles two chunks would be split in
# half and lose meaning. The 50-character overlap ensures no sentence is
# cut off at a boundary.
# ---------------------------------------------------------------------------


class KnowledgeBaseIngestor:
    """
    Loads the five knowledge base text files, splits them into chunks,
    embeds each chunk, and stores them in a ChromaDB vector store.

    After ingestion, get_retriever() returns a LangChain retriever that
    fetches the top 3 most relevant chunks for any query.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embedder = EmbeddingService(settings=self._settings)

        # The splitter is configured once and reused for all files.
        # chunk_size=500  → each chunk is at most 500 characters
        # chunk_overlap=50 → last 50 characters of one chunk repeat at the
        #                    start of the next, preserving sentence context
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

        # Build the Chroma vector store object.
        # persist_directory → ChromaDB saves its data here so it survives restarts.
        # embedding_function → the LangChain wrapper around our HuggingFace model.
        self._db = Chroma(
            collection_name=_COLLECTION_NAME,
            embedding_function=self._embedder._model,
            persist_directory=self._settings.chroma_db_path,
        )

    # -----------------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------------

    def _load_files(self) -> list[Document]:
        """
        Read all five knowledge base files from disk.
        Returns a list of LangChain Document objects — one per file.
        Each Document carries the file's full text and its filename as metadata.
        """
        documents: list[Document] = []

        for filename in _KB_FILES:
            path = _KB_DIR / filename

            # Guard 1 — file must exist
            if not path.exists():
                raise FileNotFoundError(
                    f"Knowledge base file not found: '{filename}'. "
                    f"Expected it at: {path}"
                )

            text = path.read_text(encoding="utf-8").strip()

            # Guard 2 — file must not be empty
            if not text:
                raise ValueError(
                    f"Knowledge base file '{filename}' is empty. "
                    "Please add content before ingesting."
                )

            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": filename},  # stored alongside each chunk in ChromaDB
                )
            )

        return documents

    # -----------------------------------------------------------------------
    # Ingestion pipeline
    # -----------------------------------------------------------------------

    def ingest(self) -> int:
        """
        Run the full pipeline: load → chunk → embed → store.

        Why check if the collection is already populated?
        -----------------------------------------------------
        Every time the app starts, it would call ingest(). Without this
        check, we'd re-embed and re-insert all five files on every restart,
        creating thousands of duplicate chunks. ChromaDB would return the
        same result multiple times for every query.

        By checking first, we only pay the embedding cost once — the very
        first run. Every restart after that skips ingestion and returns
        immediately.

        Returns:
            The number of chunks stored. Returns 0 if ingestion was skipped.
        """
        # Check how many documents are already in the collection
        existing_count = self._db._collection.count()

        if existing_count > 0:
            # Already populated — skip ingestion entirely
            return 0

        # Step 1 — load all five files into Document objects
        documents = self._load_files()

        # Step 2 — split every document into chunks
        # split_documents returns a flat list of chunk Documents.
        # Each chunk inherits the metadata (source filename) from its parent.
        chunks = self._splitter.split_documents(documents)

        # Step 3 — embed every chunk and store it in ChromaDB.
        # add_documents handles embedding internally using the embedding_function
        # we passed to Chroma() in __init__. It then saves to persist_directory.
        self._db.add_documents(chunks)

        return len(chunks)

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def get_retriever(self):
        """
        Return a LangChain retriever that fetches the top 3 most relevant
        chunks from ChromaDB for any text query.

        How it works:
        1. The query string is embedded into a 384-float vector
        2. ChromaDB computes the cosine similarity between the query vector
           and every stored chunk vector
        3. The 3 chunks with the highest similarity are returned as Documents

        The retriever is a standard LangChain object — it can be plugged
        directly into any LangChain chain or used standalone with .invoke().
        """
        return self._db.as_retriever(
            search_kwargs={"k": 3}  # return top 3 most relevant chunks
        )
