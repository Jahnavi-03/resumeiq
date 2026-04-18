import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


# ---------------------------------------------------------------------------
# Abstract base — defines the interface every parser must follow
# ---------------------------------------------------------------------------

class BaseParser(ABC):
    """
    All parsers must implement extract_text.
    This class cannot be instantiated directly — only subclasses can be used.
    """

    @abstractmethod
    def extract_text(self, file_bytes: bytes) -> str:
        """
        Receive raw file bytes, return extracted text as a single string.
        Raise ValueError for any problem with the file.
        """


# ---------------------------------------------------------------------------
# PDF parser — the concrete implementation for .pdf files
# ---------------------------------------------------------------------------

class PdfParser(BaseParser):
    # Every valid PDF file starts with these 4 bytes
    _PDF_MAGIC = b"%PDF"

    def extract_text(self, file_bytes: bytes) -> str:
        # Guard 1 — reject completely empty uploads
        if not file_bytes:
            raise ValueError(
                "The uploaded file is empty. Please upload a valid PDF."
            )

        # Guard 2 — reject files that are not PDFs by checking the header
        if not file_bytes.startswith(self._PDF_MAGIC):
            raise ValueError(
                "The uploaded file is not a valid PDF. "
                "Please upload a file that ends in .pdf."
            )

        # PyPDFLoader needs a file path, not bytes in memory.
        # Write to a temp file, read it, then delete it immediately.
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name  # save the path so we can delete it in finally

            loader = PyPDFLoader(tmp_path)
            documents = loader.load()  # returns a list of LangChain Document objects

        except ValueError:
            raise  # re-raise our own validation errors untouched

        except Exception as exc:
            raise ValueError(
                f"Could not parse the PDF file: {exc}"
            ) from exc

        finally:
            # Always delete the temp file — even if an exception was raised
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # Each Document holds the text of one page in .page_content
        # Join all pages with a newline separator
        text = "\n".join(doc.page_content for doc in documents).strip()

        # Guard 3 — empty text usually means a scanned/image-only PDF
        if not text:
            raise ValueError(
                "No text could be extracted from this PDF. "
                "It may be a scanned or image-based PDF. "
                "Please upload a text-based PDF."
            )

        return text


# ---------------------------------------------------------------------------
# Factory — maps file extensions to parser classes
# ---------------------------------------------------------------------------

class ParserFactory:
    """
    Registry that maps a file extension (e.g. '.pdf') to a parser class.
    Call get_parser('resume.pdf') to receive a ready-to-use parser instance.
    """

    # Class-level dict shared across all uses — extension → parser class
    _registry: dict[str, type[BaseParser]] = {}

    @classmethod
    def register(cls, extension: str, parser_class: type[BaseParser]) -> None:
        """Register a parser class for a given file extension."""
        cls._registry[extension.lower()] = parser_class

    @classmethod
    def get_parser(cls, filename: str) -> BaseParser:
        """
        Return an instance of the correct parser for the given filename.
        Raises ValueError if the file extension has no registered parser.
        """
        ext = Path(filename).suffix.lower()  # e.g. 'resume.PDF' → '.pdf'

        if ext not in cls._registry:
            supported = ", ".join(cls._registry.keys()) or "none"
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported types: {supported}"
            )

        return cls._registry[ext]()  # instantiate and return


# ---------------------------------------------------------------------------
# Registration — wire up all supported parsers
# ---------------------------------------------------------------------------

ParserFactory.register(".pdf", PdfParser)
