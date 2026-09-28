"""
Resume text extraction service.

Deterministic, no AI involved. Given already-validated file bytes and a
known extension, extracts plain text so downstream NLP/skill-matching
can work on it. Kept separate from file_validation.py: validation
decides *whether* we should try to read the file, extraction is about
*how* we actually read it.
"""
import io

from pypdf import PdfReader
from pypdf.errors import PdfReadError
from docx import Document
from docx.opc.exceptions import PackageNotFoundError

from config import MIN_EXTRACTED_TEXT_CHARS
from exceptions import CorruptedFileError, EmptyExtractedTextError, InvalidFileTypeError


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract plain text from PDF bytes using pypdf.

    Raises:
        CorruptedFileError: the bytes aren't a readable PDF at all.
    """
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        # An encrypted PDF with an empty owner password can sometimes be
        # opened this way; anything else we treat as unreadable rather
        # than guessing at a password.
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise CorruptedFileError(
                    "This PDF is password-protected and can't be read."
                ) from exc

        pages_text = []
        for page in reader.pages:
            pages_text.append(page.extract_text() or "")
        return "\n".join(pages_text)

    except CorruptedFileError:
        raise
    except (PdfReadError, ValueError, OSError) as exc:
        raise CorruptedFileError(
            "The PDF file appears to be corrupted or malformed and could not be read."
        ) from exc


def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extract plain text from DOCX bytes using python-docx.

    Raises:
        CorruptedFileError: the bytes aren't a readable DOCX at all.
    """
    try:
        document = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in document.paragraphs]

        # Resumes often put content in tables (skills tables, two-column
        # layouts) -- python-docx doesn't include table text in
        # `.paragraphs`, so we walk tables explicitly or we'd silently
        # lose real resume content.
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    paragraphs.append(cell.text)

        return "\n".join(paragraphs)

    except PackageNotFoundError as exc:
        raise CorruptedFileError(
            "The DOCX file appears to be corrupted or is not a valid Word document."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - last-resort guard for malformed zips/XML
        raise CorruptedFileError(
            "The DOCX file could not be read due to an unexpected format error."
        ) from exc


def extract_resume_text(extension: str, file_bytes: bytes) -> str:
    """
    Dispatch to the correct parser based on (already-validated) extension,
    then enforce that the result contains meaningful text.

    Args:
        extension: 'pdf' or 'docx', as returned by file_validation.validate_file.
        file_bytes: the raw file content.

    Returns:
        Extracted, stripped plain text.

    Raises:
        InvalidFileTypeError: extension isn't one this function supports
            (defensive check -- validate_file should already prevent this).
        CorruptedFileError: file couldn't be parsed.
        EmptyExtractedTextError: parsing succeeded but produced no
            meaningful text (e.g. a scanned/image-only PDF).
    """
    if extension == "pdf":
        raw_text = extract_text_from_pdf(file_bytes)
    elif extension == "docx":
        raw_text = extract_text_from_docx(file_bytes)
    else:
        raise InvalidFileTypeError(f"No parser available for '.{extension}' files.")

    text = raw_text.strip()
    if len(text) < MIN_EXTRACTED_TEXT_CHARS:
        raise EmptyExtractedTextError(
            "No readable text could be extracted from this file. "
            "It may be a scanned image, blank, or password-protected."
        )

    return text
