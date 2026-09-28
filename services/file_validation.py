"""
File validation service.

Deterministic, no AI involved. Given a filename and raw bytes, checks
that the file is an acceptable resume upload *before* we spend time
trying to parse it. Every check here maps to one of the ABSOLUTE RULES
around not trusting uploaded input.
"""
from config import ALLOWED_RESUME_EXTENSIONS, MAX_RESUME_SIZE_BYTES
from exceptions import (
    EmptyFileError,
    FileTooLargeError,
    InvalidFileTypeError,
    MissingFilenameError,
)


def get_extension(filename: str) -> str:
    """
    Return the lowercase extension of filename, without the dot.
    'Resume.PDF' -> 'pdf'. Returns '' if there is no extension at all.
    """
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()


def validate_file(filename: str, file_bytes: bytes) -> str:
    """
    Validate an uploaded resume file.

    Args:
        filename: original filename as provided by the client.
        file_bytes: the raw file content.

    Returns:
        The validated lowercase extension ("pdf" or "docx"), so the
        caller knows which parser to dispatch to without re-deriving it.

    Raises:
        MissingFilenameError: no filename was provided.
        InvalidFileTypeError: extension isn't in ALLOWED_RESUME_EXTENSIONS.
        EmptyFileError: file_bytes has zero length.
        FileTooLargeError: file_bytes exceeds MAX_RESUME_SIZE_BYTES.
    """
    if not filename or not filename.strip():
        raise MissingFilenameError("No file was selected.")

    extension = get_extension(filename)
    if extension not in ALLOWED_RESUME_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_RESUME_EXTENSIONS)).upper()
        raise InvalidFileTypeError(
            f"Unsupported file type '.{extension or '?'}'. Allowed types: {allowed}."
        )

    if len(file_bytes) == 0:
        raise EmptyFileError("The uploaded file is empty (0 bytes).")

    if len(file_bytes) > MAX_RESUME_SIZE_BYTES:
        max_mb = MAX_RESUME_SIZE_BYTES / (1024 * 1024)
        raise FileTooLargeError(
            f"File is too large. Maximum allowed size is {max_mb:.0f} MB."
        )

    return extension
