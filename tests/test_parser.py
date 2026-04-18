import os
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.services.parser import BaseParser, PdfParser, ParserFactory


# ---------------------------------------------------------------------------
# Minimal valid PDF header — enough to pass the magic-bytes check.
# We mock PyPDFLoader so the actual content does not matter.
# ---------------------------------------------------------------------------
VALID_PDF_BYTES = b"%PDF-1.4 fake content"


# ---------------------------------------------------------------------------
# ParserFactory
# ---------------------------------------------------------------------------

class TestParserFactory:
    def test_returns_pdf_parser_for_pdf_extension(self):
        parser = ParserFactory.get_parser("resume.pdf")
        assert isinstance(parser, PdfParser)

    def test_extension_matching_is_case_insensitive(self):
        # Users may upload "RESUME.PDF" — should still work
        parser = ParserFactory.get_parser("RESUME.PDF")
        assert isinstance(parser, PdfParser)

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            ParserFactory.get_parser("resume.docx")

    def test_no_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            ParserFactory.get_parser("resume")

    def test_error_message_lists_supported_types(self):
        with pytest.raises(ValueError, match=".pdf"):
            ParserFactory.get_parser("resume.txt")


# ---------------------------------------------------------------------------
# PdfParser — input validation (no real PDF needed)
# ---------------------------------------------------------------------------

class TestPdfParserValidation:
    def test_empty_bytes_raises(self):
        parser = PdfParser()
        with pytest.raises(ValueError, match="empty"):
            parser.extract_text(b"")

    def test_non_pdf_bytes_raises(self):
        parser = PdfParser()
        with pytest.raises(ValueError, match="not a valid PDF"):
            parser.extract_text(b"this is a plain text file, not a pdf")

    def test_image_file_passed_as_bytes_raises(self):
        # PNG magic bytes — definitely not a PDF
        png_bytes = b"\x89PNG\r\n\x1a\n"
        parser = PdfParser()
        with pytest.raises(ValueError, match="not a valid PDF"):
            parser.extract_text(png_bytes)


# ---------------------------------------------------------------------------
# PdfParser — extraction (PyPDFLoader mocked so no real file I/O needed)
# ---------------------------------------------------------------------------

class TestPdfParserExtraction:
    def _make_docs(self, *pages: str) -> list[Document]:
        """Helper: turn plain strings into LangChain Document objects."""
        return [Document(page_content=p, metadata={}) for p in pages]

    def test_single_page_text_is_returned(self):
        parser = PdfParser()
        with patch("app.services.parser.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = self._make_docs(
                "Jane Doe\nSoftware Engineer"
            )
            result = parser.extract_text(VALID_PDF_BYTES)
        assert "Jane Doe" in result
        assert "Software Engineer" in result

    def test_multiple_pages_are_joined(self):
        parser = PdfParser()
        with patch("app.services.parser.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = self._make_docs(
                "Page one content",
                "Page two content",
            )
            result = parser.extract_text(VALID_PDF_BYTES)
        assert "Page one content" in result
        assert "Page two content" in result

    def test_scanned_pdf_with_no_text_raises(self):
        # PyPDFLoader returns documents but every page is blank — scanned PDF
        parser = PdfParser()
        with patch("app.services.parser.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = self._make_docs("", "")
            with pytest.raises(ValueError, match="scanned"):
                parser.extract_text(VALID_PDF_BYTES)

    def test_temp_file_is_deleted_after_successful_extraction(self):
        parser = PdfParser()
        recorded_path: list[str] = []

        # Intercept NamedTemporaryFile to capture the temp file path
        real_ntf = __import__("tempfile").NamedTemporaryFile

        def capturing_ntf(**kwargs):
            tmp = real_ntf(**kwargs)
            recorded_path.append(tmp.name)
            return tmp

        with patch("app.services.parser.tempfile.NamedTemporaryFile", side_effect=capturing_ntf):
            with patch("app.services.parser.PyPDFLoader") as MockLoader:
                MockLoader.return_value.load.return_value = self._make_docs("Resume text")
                parser.extract_text(VALID_PDF_BYTES)

        assert recorded_path, "No temp file was created"
        assert not os.path.exists(recorded_path[0]), "Temp file was not deleted"

    def test_temp_file_is_deleted_even_when_loader_raises(self):
        parser = PdfParser()
        recorded_path: list[str] = []

        real_ntf = __import__("tempfile").NamedTemporaryFile

        def capturing_ntf(**kwargs):
            tmp = real_ntf(**kwargs)
            recorded_path.append(tmp.name)
            return tmp

        with patch("app.services.parser.tempfile.NamedTemporaryFile", side_effect=capturing_ntf):
            with patch("app.services.parser.PyPDFLoader") as MockLoader:
                MockLoader.return_value.load.side_effect = RuntimeError("disk error")
                with pytest.raises(ValueError, match="Could not parse"):
                    parser.extract_text(VALID_PDF_BYTES)

        assert recorded_path, "No temp file was created"
        assert not os.path.exists(recorded_path[0]), "Temp file leaked after error"


# ---------------------------------------------------------------------------
# BaseParser — cannot be used directly
# ---------------------------------------------------------------------------

class TestBaseParser:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            BaseParser()  # type: ignore

    def test_subclass_without_extract_text_cannot_instantiate(self):
        class IncompleteParser(BaseParser):
            pass  # forgot to implement extract_text

        with pytest.raises(TypeError):
            IncompleteParser()
