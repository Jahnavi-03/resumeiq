from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import Settings, get_settings

# ---------------------------------------------------------------------------
# What is an embedding?
#
# An embedding is a list of numbers (a vector) that represents the *meaning*
# of a piece of text. Texts with similar meanings end up with similar vectors.
#
# Example:
#   "software engineer"  →  [0.12, -0.45, 0.88, ...]   (384 numbers)
#   "developer"          →  [0.11, -0.43, 0.85, ...]   (very close!)
#   "chocolate cake"     →  [0.91,  0.23, -0.10, ...]  (very different)
#
# Why HuggingFace instead of OpenAI?
#   - Runs locally on your machine — no API call, no cost per request
#   - all-MiniLM-L6-v2 is fast and small (80 MB) but still high quality
#   - Output is always 384 dimensions, consistent and predictable
#
# What does the vector represent?
#   Each of the 384 numbers captures some aspect of meaning — things like
#   topic, tone, domain, relationships between words. No single number
#   means anything alone; the whole vector together encodes the meaning.
# ---------------------------------------------------------------------------


class EmbeddingService:
    """
    Converts text into vectors (embeddings) using a local HuggingFace model.

    The model is loaded once at construction time and reused for every call.
    Loading takes a few seconds the first time — after that, embedding is fast.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        # Use provided settings or fall back to the app-wide cached settings
        self._settings = settings or get_settings()

        # Load the model once here — NOT inside get_embedding / get_embeddings.
        # Loading a model takes 2–5 seconds. If we loaded it on every call,
        # each embedding request would be slow.
        try:
            self._model = HuggingFaceEmbeddings(
                model_name=self._settings.embedding_model
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model "
                f"'{self._settings.embedding_model}': {exc}"
            ) from exc

    def get_embedding(self, text: str) -> list[float]:
        """
        Embed a single string and return a vector of floats.

        Args:
            text: The text to embed (e.g. a resume or a job description).

        Returns:
            A list of 384 floats representing the meaning of the text.

        Raises:
            ValueError: If text is empty or whitespace only.
        """
        if not text or not text.strip():
            raise ValueError(
                "Cannot embed empty text. Please provide a non-empty string."
            )

        # embed_query is LangChain's method for embedding a single string.
        # It returns a plain list of floats.
        return self._model.embed_query(text)

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of strings in one batch and return a list of vectors.

        Batch embedding is faster than calling get_embedding() in a loop
        because the model processes all texts together in one forward pass.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of vectors — one vector per input string, same order.

        Raises:
            ValueError: If the list is empty or any individual text is empty.
        """
        if not texts:
            raise ValueError(
                "Cannot embed an empty list. Please provide at least one string."
            )

        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(
                    f"Text at index {i} is empty. All texts must be non-empty strings."
                )

        # embed_documents is LangChain's method for batch embedding.
        # Much faster than looping over embed_query one at a time.
        return self._model.embed_documents(texts)
