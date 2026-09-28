import os
import unittest

from services.file_validation import get_extension, validate_file
from exceptions import (
    EmptyFileError,
    FileTooLargeError,
    InvalidFileTypeError,
    MissingFilenameError,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def read_fixture(name: str) -> bytes:
    with open(os.path.join(FIXTURES, name), "rb") as f:
        return f.read()


class TestGetExtension(unittest.TestCase):
    def test_lowercases_extension(self):
        self.assertEqual(get_extension("Resume.PDF"), "pdf")

    def test_no_extension_returns_empty_string(self):
        self.assertEqual(get_extension("resume"), "")

    def test_multiple_dots_uses_last_segment(self):
        self.assertEqual(get_extension("my.resume.final.docx"), "docx")


class TestValidateFile(unittest.TestCase):
    def test_valid_pdf_passes_and_returns_extension(self):
        data = read_fixture("valid_resume.pdf")
        self.assertEqual(validate_file("resume.pdf", data), "pdf")

    def test_valid_docx_passes_and_returns_extension(self):
        data = read_fixture("valid_resume.docx")
        self.assertEqual(validate_file("resume.docx", data), "docx")

    def test_missing_filename_raises(self):
        with self.assertRaises(MissingFilenameError):
            validate_file("", b"some bytes")

    def test_blank_filename_raises(self):
        with self.assertRaises(MissingFilenameError):
            validate_file("   ", b"some bytes")

    def test_invalid_extension_raises(self):
        data = read_fixture("invalid_type.txt")
        with self.assertRaises(InvalidFileTypeError):
            validate_file("notes.txt", data)

    def test_empty_file_raises(self):
        data = read_fixture("empty.pdf")
        with self.assertRaises(EmptyFileError):
            validate_file("empty.pdf", data)

    def test_oversized_file_raises(self):
        data = read_fixture("oversized.pdf")
        with self.assertRaises(FileTooLargeError):
            validate_file("huge.pdf", data)

    def test_corrupted_pdf_still_passes_validation(self):
        # Validation only checks extension/size/emptiness, not whether the
        # content is actually parseable -- that's resume_parser's job.
        data = read_fixture("corrupted.pdf")
        self.assertEqual(validate_file("corrupted.pdf", data), "pdf")


if __name__ == "__main__":
    unittest.main()
