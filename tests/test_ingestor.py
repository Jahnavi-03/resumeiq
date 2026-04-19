import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

from app.core.config import Settings
from knowledge_base.ingestor import KnowledgeBaseIngestor, _KB_FILES, _KB_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_settings(chroma_path: str) -> Settings:
    """Settings that point ChromaDB at a temp directory, no .env needed."""
    return Settings(
        groq_api_key="test-key",
        embedding_model="all-MiniLM-L6-v2",
        chroma_db_path=chroma_path,
        _env_file=None,
    )


# ---------------------------------------------------------------------------
# Fixture — one ingestor per test module, reuses the loaded model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ingestor(tmp_path_factory) -> KnowledgeBaseIngestor:
    """
    A real KnowledgeBaseIngestor pointing at a temporary ChromaDB directory.
    scope="module" so the embedding model loads once for all tests.
    tmp_path_factory is pytest's built-in fixture for module-scoped temp dirs.
    """
    chroma_dir = tmp_path_factory.mktemp("chroma")
    settings = _fake_settings(str(chroma_dir))
    return KnowledgeBaseIngestor(settings=settings)


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

class TestFileLoading:
    def test_all_five_files_exist_on_disk(self):
        # Verify the actual knowledge base files are present before anything else.
        for filename in _KB_FILES:
            path = _KB_DIR / filename
            assert path.exists(), f"Missing knowledge base file: {filename}"

    def test_all_five_files_are_non_empty(self):
        for filename in _KB_FILES:
            path = _KB_DIR / filename
            text = path.read_text(encoding="utf-8").strip()
            assert text, f"Knowledge base file is empty: {filename}"

    def test_load_files_returns_five_documents(self, ingestor: KnowledgeBaseIngestor):
        docs = ingestor._load_files()
        assert len(docs) == 5

    def test_each_document_has_source_metadata(self, ingestor: KnowledgeBaseIngestor):
        docs = ingestor._load_files()
        sources = [doc.metadata["source"] for doc in docs]
        for filename in _KB_FILES:
            assert filename in sources

    def test_missing_file_raises_file_not_found(self, ingestor: KnowledgeBaseIngestor):
        # Patch _KB_FILES to include a file that doesn't exist
        fake_files = ["ats_rules.txt", "this_file_does_not_exist.txt"]
        with patch("knowledge_base.ingestor._KB_FILES", fake_files):
            with pytest.raises(FileNotFoundError, match="not found"):
                ingestor._load_files()

    def test_empty_file_raises_value_error(self, ingestor: KnowledgeBaseIngestor, tmp_path):
        # Create a real empty file and patch _KB_DIR to point at it
        empty_file = tmp_path / "ats_rules.txt"
        empty_file.write_text("")
        with patch("knowledge_base.ingestor._KB_DIR", tmp_path):
            with patch("knowledge_base.ingestor._KB_FILES", ["ats_rules.txt"]):
                with pytest.raises(ValueError, match="empty"):
                    ingestor._load_files()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

class TestChunking:
    def test_chunking_produces_multiple_chunks(self, ingestor: KnowledgeBaseIngestor):
        # Each knowledge base file is much longer than 500 characters,
        # so every file should produce more than one chunk.
        docs = ingestor._load_files()
        chunks = ingestor._splitter.split_documents(docs)
        assert len(chunks) > len(_KB_FILES)

    def test_each_chunk_is_within_size_limit(self, ingestor: KnowledgeBaseIngestor):
        docs = ingestor._load_files()
        chunks = ingestor._splitter.split_documents(docs)
        for chunk in chunks:
            # Allow a small buffer — the splitter tries to respect 500 chars
            # but may slightly exceed it at natural split boundaries.
            assert len(chunk.page_content) <= 600, (
                f"Chunk too large: {len(chunk.page_content)} chars"
            )

    def test_chunks_inherit_source_metadata(self, ingestor: KnowledgeBaseIngestor):
        docs = ingestor._load_files()
        chunks = ingestor._splitter.split_documents(docs)
        for chunk in chunks:
            assert "source" in chunk.metadata


# ---------------------------------------------------------------------------
# Ingestion into ChromaDB
# ---------------------------------------------------------------------------

class TestIngestion:
    def test_ingest_stores_chunks_in_chromadb(self, ingestor: KnowledgeBaseIngestor):
        count = ingestor.ingest()
        # ingest() returns the number of chunks added (or 0 if skipped)
        stored = ingestor._db._collection.count()
        assert stored > 0

    def test_ingest_returns_chunk_count_on_first_run(self, tmp_path):
        # Use a fresh ChromaDB directory so this is definitely a first run
        settings = _fake_settings(str(tmp_path / "fresh_chroma"))
        fresh = KnowledgeBaseIngestor(settings=settings)
        count = fresh.ingest()
        assert count > 0

    def test_ingest_skipped_if_collection_already_populated(self, tmp_path):
        # First run populates the collection
        settings = _fake_settings(str(tmp_path / "chroma"))
        ing = KnowledgeBaseIngestor(settings=settings)
        first_count = ing.ingest()
        assert first_count > 0

        # Second run on the same collection should return 0 (skipped)
        second_count = ing.ingest()
        assert second_count == 0

    def test_double_ingest_does_not_duplicate_data(self, tmp_path):
        settings = _fake_settings(str(tmp_path / "chroma"))
        ing = KnowledgeBaseIngestor(settings=settings)
        ing.ingest()
        count_after_first = ing._db._collection.count()

        ing.ingest()  # should be skipped
        count_after_second = ing._db._collection.count()

        assert count_after_first == count_after_second


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class TestRetriever:
    def test_get_retriever_returns_documents(self, ingestor: KnowledgeBaseIngestor):
        # Make sure data is ingested before testing retrieval
        ingestor.ingest()
        retriever = ingestor.get_retriever()
        docs = retriever.invoke("how to write a resume bullet point")
        assert len(docs) > 0

    def test_retriever_returns_at_most_3_chunks(self, ingestor: KnowledgeBaseIngestor):
        ingestor.ingest()
        retriever = ingestor.get_retriever()
        docs = retriever.invoke("ATS optimization tips")
        assert len(docs) <= 3

    def test_retriever_returns_relevant_content(self, ingestor: KnowledgeBaseIngestor):
        ingestor.ingest()
        retriever = ingestor.get_retriever()
        docs = retriever.invoke("applicant tracking system keywords")
        # The combined text of returned chunks should mention ATS-related terms
        combined = " ".join(doc.page_content.lower() for doc in docs)
        assert any(word in combined for word in ["ats", "keyword", "applicant", "resume"])
