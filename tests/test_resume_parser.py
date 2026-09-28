import os
import unittest

from services.resume_parser import (
    extract_resume_text,
    extract_text_from_docx,
    extract_text_from_pdf,
)
from exceptions import CorruptedFileError, EmptyExtractedTextError, InvalidFileTypeError

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def read_fixture(name: str) -> bytes:
    with open(os.path.join(FIXTURES, name), "rb") as f:
        return f.read()


class TestExtractTextFromPdf(unittest.TestCase):
    def test_extracts_real_text_from_valid_pdf(self):
        data = read_fixture("valid_resume.pdf")
        text = extract_text_from_pdf(data)
        self.assertIn("John Doe", text)
        self.assertIn("Python", text)
        self.assertIn("Flask", text)

    def test_corrupted_pdf_raises(self):
        data = read_fixture("corrupted.pdf")
        with self.assertRaises(CorruptedFileError):
            extract_text_from_pdf(data)


class TestExtractTextFromDocx(unittest.TestCase):
    def test_extracts_real_text_from_valid_docx(self):
        data = read_fixture("valid_resume.docx")
        text = extract_text_from_docx(data)
        self.assertIn("Jane Smith", text)
        self.assertIn("Python", text)

    def test_extracts_text_from_tables(self):
        # valid_resume.docx has a table row: Tools | Docker basics, Postman, GitHub
        data = read_fixture("valid_resume.docx")
        text = extract_text_from_docx(data)
        self.assertIn("Docker basics", text)
        self.assertIn("Postman", text)

    def test_corrupted_docx_raises(self):
        data = read_fixture("corrupted.docx")
        with self.assertRaises(CorruptedFileError):
            extract_text_from_docx(data)


class TestExtractResumeText(unittest.TestCase):
    def test_dispatches_pdf_correctly(self):
        data = read_fixture("valid_resume.pdf")
        text = extract_resume_text("pdf", data)
        self.assertIn("John Doe", text)

    def test_dispatches_docx_correctly(self):
        data = read_fixture("valid_resume.docx")
        text = extract_resume_text("docx", data)
        self.assertIn("Jane Smith", text)

    def test_unsupported_extension_raises(self):
        with self.assertRaises(InvalidFileTypeError):
            extract_resume_text("txt", b"hello")

    def test_corrupted_pdf_raises_corrupted_error(self):
        data = read_fixture("corrupted.pdf")
        with self.assertRaises(CorruptedFileError):
            extract_resume_text("pdf", data)

    def test_near_empty_pdf_raises_empty_extracted_text_error(self):
        # near_empty.pdf has a real text layer ("Hi") but well under our
        # MIN_EXTRACTED_TEXT_CHARS threshold -- simulates a mostly-blank
        # or image-only resume.
        data = read_fixture("near_empty.pdf")
        with self.assertRaises(EmptyExtractedTextError):
            extract_resume_text("pdf", data)


if __name__ == "__main__":
    unittest.main()
