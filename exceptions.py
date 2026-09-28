"""
Custom exception hierarchy for the Resume Analyzer.

Using specific exception classes (instead of generic Exception or
ValueError everywhere) means calling code -- like the Flask route --
can catch precisely what it expects and return the right HTTP status
and message for each failure, rather than guessing from a string.
"""


class ResumeAnalyzerError(Exception):
    """Base class for all application-specific errors."""


# --- File validation errors ---------------------------------------------

class FileValidationError(ResumeAnalyzerError):
    """Base class for problems with an uploaded file, before parsing."""


class InvalidFileTypeError(FileValidationError):
    """Raised when the uploaded file extension isn't PDF or DOCX."""


class FileTooLargeError(FileValidationError):
    """Raised when the uploaded file exceeds MAX_RESUME_SIZE_BYTES."""


class EmptyFileError(FileValidationError):
    """Raised when the uploaded file has zero bytes."""


class MissingFilenameError(FileValidationError):
    """Raised when no file was actually selected/uploaded."""


# --- Text extraction errors ----------------------------------------------

class TextExtractionError(ResumeAnalyzerError):
    """Base class for problems while extracting text from a valid file."""


class CorruptedFileError(TextExtractionError):
    """Raised when the file can't be parsed at all (corrupt/malformed)."""


class EmptyExtractedTextError(TextExtractionError):
    """
    Raised when parsing succeeds but yields no usable text -- e.g. a
    scanned/image-only PDF with no embedded text layer.
    """


# --- Scoring errors --------------------------------------------------------

class ScoringError(ResumeAnalyzerError):
    """Base class for problems that prevent computing an ATS score."""


class EmptyResumeTextError(ScoringError):
    """Raised when resume text is empty/whitespace-only -- no score is computed."""


class EmptyJobDescriptionTextError(ScoringError):
    """Raised when job description text is empty/whitespace-only -- no score is computed."""


# --- LLM integration errors -------------------------------------------------

class LLMError(ResumeAnalyzerError):
    """Base class for problems calling or interpreting the LLM API."""


class MissingAPIKeyError(LLMError):
    """Raised when no LLM_API_KEY is configured."""


class InvalidAPIKeyError(LLMError):
    """Raised when the LLM API rejects the configured key (HTTP 401)."""


class LLMTimeoutError(LLMError):
    """Raised when the LLM API call exceeds LLM_TIMEOUT_SECONDS."""


class LLMRateLimitError(LLMError):
    """Raised when the LLM API returns HTTP 429."""


class LLMRequestError(LLMError):
    """Raised for network failures or other non-2xx LLM API responses."""


class MalformedLLMResponseError(LLMError):
    """Raised when the LLM response is empty, not valid JSON, or missing expected fields."""
