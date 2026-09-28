"""
Application configuration.

Loads environment variables via python-dotenv and defines constants
used across the app (file upload limits, allowed resume formats, etc).
Centralizing these here means every module reads limits from one place
instead of hardcoding "magic numbers" in multiple files.
"""
import os
from dotenv import load_dotenv

# Load variables from a .env file (if present) into the process environment.
# This must run before we read os.environ.get(...) below.
load_dotenv()

# --- File upload rules -------------------------------------------------

# Which resume file extensions we accept. Kept as a set for O(1) lookup.
ALLOWED_RESUME_EXTENSIONS = {"pdf", "docx"}

# Max upload size in bytes. Oversized resumes are rejected before we ever
# try to parse them, so a huge file can't tie up the server.
MAX_RESUME_SIZE_BYTES = int(os.environ.get("MAX_RESUME_SIZE_MB", "5")) * 1024 * 1024

# Minimum number of extracted characters for a resume to be considered
# non-empty. Some PDFs "extract" a handful of whitespace/control
# characters even when there's no real text (e.g. an image-only PDF) --
# this threshold guards against treating that as a valid extraction.
MIN_EXTRACTED_TEXT_CHARS = 30

# --- LLM configuration (services/llm_service.py) -----------------------

LLM_API_KEY = os.environ.get("LLM_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
LLM_API_URL = "https://api.anthropic.com/v1/messages"
LLM_TIMEOUT_SECONDS = int(os.environ.get("LLM_TIMEOUT_SECONDS", "30"))
