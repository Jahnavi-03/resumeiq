import math
import pytest
from unittest.mock import MagicMock, patch

from app.core.config import Settings
from app.services.embedder import EmbeddingService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_settings() -> Settings:
    """Return a Settings object without reading the .env file."""
    return Settings(
        groq_api_key="test-key",
        embedding_model="all-MiniLM-L6-v2",
        _env_file=None,
    )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Measure how similar two vectors are.
    Returns a value between -1 and 1:
      1.0  = identical direction (very similar meaning)
      0.0  = perpendicular (unrelated)
     -1.0  = opposite direction
    """
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def service() -> EmbeddingService:
    """
    One real EmbeddingService shared across all tests in this file.

    scope="module" means the model is loaded once for the whole test file —
    not once per test. This makes the test suite much faster since model
    loading takes a few seconds.
    """
    return EmbeddingService(settings=_fake_settings())


# ---------------------------------------------------------------------------
# get_embedding — return type and shape
# ---------------------------------------------------------------------------

class TestGetEmbeddingReturnType:
    def test_returns_a_list(self, service: EmbeddingService):
        result = service.get_embedding("software engineer with Python experience")
        assert isinstance(result, list)

    def test_returns_list_of_floats(self, service: EmbeddingService):
        result = service.get_embedding("data scientist")
        assert all(isinstance(x, float) for x in result)

    def test_returns_384_dimensions(self, service: EmbeddingService):
        # all-MiniLM-L6-v2 always produces exactly 384 numbers.
        # If this breaks, the model was swapped out.
        result = service.get_embedding("machine learning engineer")
        assert len(result) == 384


# ---------------------------------------------------------------------------
# get_embedding — input validation
# ---------------------------------------------------------------------------

class TestGetEmbeddingValidation:
    def test_empty_string_raises(self, service: EmbeddingService):
        with pytest.raises(ValueError, match="empty"):
            service.get_embedding("")

    def test_whitespace_only_raises(self, service: EmbeddingService):
        with pytest.raises(ValueError, match="empty"):
            service.get_embedding("    ")


# ---------------------------------------------------------------------------
# get_embeddings — batch
# ---------------------------------------------------------------------------

class TestGetEmbeddingsBatch:
    def test_returns_one_vector_per_text(self, service: EmbeddingService):
        texts = ["software engineer", "data analyst", "product manager"]
        results = service.get_embeddings(texts)
        assert len(results) == len(texts)

    def test_each_vector_has_384_dimensions(self, service: EmbeddingService):
        texts = ["backend developer", "frontend developer"]
        results = service.get_embeddings(texts)
        assert all(len(v) == 384 for v in results)

    def test_preserves_order(self, service: EmbeddingService):
        # Embedding the same text alone vs in a batch should give the same vector.
        text = "Python developer with FastAPI experience"
        single = service.get_embedding(text)
        batch = service.get_embeddings([text, "unrelated text"])
        assert _cosine_similarity(single, batch[0]) > 0.999

    def test_empty_list_raises(self, service: EmbeddingService):
        with pytest.raises(ValueError, match="empty list"):
            service.get_embeddings([])

    def test_empty_string_in_list_raises(self, service: EmbeddingService):
        with pytest.raises(ValueError, match="index 1"):
            service.get_embeddings(["valid text", ""])


# ---------------------------------------------------------------------------
# Semantic similarity — the core purpose of embeddings
# ---------------------------------------------------------------------------

class TestSemanticSimilarity:
    def test_similar_texts_have_high_cosine_similarity(self, service: EmbeddingService):
        # These two phrases mean almost the same thing — vectors should be close.
        vec_a = service.get_embedding("software engineer with Python skills")
        vec_b = service.get_embedding("Python developer and software engineer")
        similarity = _cosine_similarity(vec_a, vec_b)
        assert similarity > 0.8, f"Expected > 0.8, got {similarity:.3f}"

    def test_different_texts_have_low_cosine_similarity(self, service: EmbeddingService):
        # A resume skill and a food item should be far apart in meaning.
        vec_a = service.get_embedding("machine learning and neural networks")
        vec_b = service.get_embedding("chocolate cake with strawberry frosting")
        similarity = _cosine_similarity(vec_a, vec_b)
        assert similarity < 0.5, f"Expected < 0.5, got {similarity:.3f}"

    def test_same_text_has_perfect_similarity(self, service: EmbeddingService):
        text = "experienced backend engineer"
        vec_a = service.get_embedding(text)
        vec_b = service.get_embedding(text)
        similarity = _cosine_similarity(vec_a, vec_b)
        assert similarity > 0.999


# ---------------------------------------------------------------------------
# Model loading — failure handled cleanly
# ---------------------------------------------------------------------------

class TestModelLoading:
    def test_model_load_failure_raises_runtime_error(self):
        with patch(
            "app.services.embedder.HuggingFaceEmbeddings",
            side_effect=Exception("network timeout"),
        ):
            with pytest.raises(RuntimeError, match="Failed to load embedding model"):
                EmbeddingService(settings=_fake_settings())
