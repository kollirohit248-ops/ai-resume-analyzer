"""
Text preprocessing service.

Deterministic, no AI involved. Turns raw extracted resume/job-description
text into a form that skill_extraction.py (Feature 4) can reliably match
against a skill dictionary.

Pipeline (each stage is a separate, independently testable function):

    RAW TEXT
        |  normalize_unicode()      -- fix encoding oddities, smart punctuation
        v
    (unicode-normalized)
        |  clean_whitespace()       -- collapse/trim whitespace
        v
    (whitespace-clean)
        |  normalize_punctuation()  -- strip bullet markers, collapse repeats
        v
    CLEAN TEXT                      -- human-readable, original casing kept
        |  normalize_for_matching() -- lowercase for case-insensitive matching
        v
    NORMALIZED TEXT
        |  tokenize()                -- word-level tokens, tech terms preserved
        v
    TOKENS

`preprocess()` runs the whole pipeline and returns a PreprocessedText
holding every stage -- the original raw_text is always kept alongside
the derived stages, never overwritten, so a later feature (e.g. showing
the user their resume text, or sending readable text to the LLM) can
still use it.
"""
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List

# --- Data model ------------------------------------------------------------


@dataclass
class PreprocessedText:
    """Holds every stage of the pipeline for one piece of text."""
    raw_text: str
    clean_text: str
    normalized_text: str
    tokens: List[str] = field(default_factory=list)


# --- Stage 1: unicode normalization -----------------------------------------

# Characters commonly produced by Word/Google Docs exports that have a
# plain-ASCII equivalent we want for reliable string matching later.
# Left-hand side chars are NOT touched by NFKC normalization (NFKC does not
# fold "smart quotes" or dashes into ASCII), so we handle them explicitly.
_UNICODE_REPLACEMENTS = {
    "\u2018": "'", "\u2019": "'",   # ‘ ’  smart single quotes
    "\u201c": '"', "\u201d": '"',   # “ ”  smart double quotes
    "\u2013": "-", "\u2014": "-",   # – —  en/em dash
    "\u2212": "-",                    # −    minus sign
    "\u2026": "...",                 # …    ellipsis
    "\u00a0": " ",                    # non-breaking space
    "\u200b": "",                     # zero-width space
}
_UNICODE_TRANSLATION_TABLE = str.maketrans(_UNICODE_REPLACEMENTS)


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode text to a consistent, ASCII-friendly form.

    - NFKC normalization folds compatibility characters (e.g. full-width
      forms, ligatures) into their standard equivalents.
    - Explicit replacements handle smart quotes/dashes/ellipsis/NBSP that
      NFKC does not fold, since a resume pasted from Word commonly
      contains these and we don't want them breaking string matches
      against a plain-ASCII skill dictionary.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    return normalized.translate(_UNICODE_TRANSLATION_TABLE)


# --- Stage 2: whitespace cleanup --------------------------------------------

_WHITESPACE_RUN_RE = re.compile(r"[ \t\r\n\f\v]+")


def clean_whitespace(text: str) -> str:
    """
    Collapse any run of whitespace (spaces, tabs, newlines) into a single
    space, and trim leading/trailing whitespace.

    Note: this intentionally discards line-break structure. That's fine
    for skill/keyword matching (this feature's purpose), but it means
    this function is NOT suitable for a future "resume section detection"
    feature that needs to know where lines/paragraphs break -- that
    would need to operate on the raw text before this stage.
    """
    if not text:
        return ""
    return _WHITESPACE_RUN_RE.sub(" ", text).strip()


# --- Stage 3: punctuation normalization --------------------------------------

# Bullet/list-marker characters that add visual structure but no semantic
# content. Safe to strip outright -- none of these appear inside a
# technical term we care about (C++, C#, .NET, Node.js, etc).
_BULLET_CHARS_RE = re.compile(r"[•‣●▪◦○∙►▶]")

# Collapse 2+ repeated identical punctuation marks into one
# ("!!!" -> "!", "??" -> "?"). Requires at least 2 repeats, so a lone
# "." in ".NET" or "Node.js" is left completely alone.
_REPEATED_PUNCT_RE = re.compile(r"([!?.,;:])\1+")


def normalize_punctuation(text: str) -> str:
    """
    Light-touch punctuation cleanup that does NOT strip punctuation that
    carries meaning in technical terms (+, #, ., etc). Only removes
    decorative bullet characters and collapses repeated punctuation runs.
    """
    if not text:
        return ""
    text = _BULLET_CHARS_RE.sub("", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)
    # Bullet removal / punctuation collapsing can leave extra spaces
    # behind (e.g. "• Python" -> " Python"), so re-run whitespace cleanup.
    return clean_whitespace(text)


# --- Stage 4: matching normalization (case) ---------------------------------


def normalize_for_matching(clean_text: str) -> str:
    """
    Lowercase clean text for case-insensitive matching.

    Lowercasing is safe for our purposes because the skill dictionary
    (Feature 4) will also be stored/compared in lowercase -- casing is
    only meaningful for *display* (e.g. showing "React.js" back to the
    user), which is handled separately by mapping a matched normalized
    term back to its canonical display form, not by preprocessing.
    """
    if not clean_text:
        return ""
    return clean_text.lower()


# --- Stage 5: tokenization ---------------------------------------------------

# Special technical-term patterns get first shot at matching, in this
# order, before the generic alnum fallback. Order matters within a
# regex alternation: the first alternative that matches at a given
# position wins, so these must come before the generic pattern.
#
# These patterns run against already-lowercased text (see tokenize()),
# so they're written in lowercase.
_TOKEN_PATTERN = re.compile(
    r"c\+\+"                    # C++
    r"|[a-z]\+\+"                # generic X++ (e.g. g++)
    r"|[a-z]#"                   # C#, F#
    r"|\.net\b"                  # .NET
    r"|[a-z][a-z0-9]*\.js\b"    # Node.js, React.js, Next.js, Vue.js, ...
    r"|[a-z0-9]+"                 # generic alphanumeric word
)


def tokenize(normalized_text: str) -> List[str]:
    """
    Split normalized (lowercased) text into word-level tokens.

    Unlike a naive `text.split()` or a punctuation-stripping tokenizer,
    this preserves single-word technical terms that contain meaningful
    punctuation: "c++" stays "c++" (not "c"), "c#" stays "c#" (not "c"),
    ".net" stays ".net" (not "net"), "node.js" stays "node.js" (not
    "node" + "js").

    Multi-word technical terms ("machine learning", "generative ai",
    "large language models") are NOT merged into single tokens here --
    they remain as separate adjacent tokens. They are still fully
    preserved (not destroyed), and Feature 4's skill matcher will look
    for them as phrases in `normalized_text` (substring/n-gram matching)
    rather than as single tokens. This keeps tokenize() simple and
    general-purpose instead of hardcoding every multi-word phrase into
    the tokenizer itself.
    """
    if not normalized_text:
        return []
    return _TOKEN_PATTERN.findall(normalized_text)


# --- Full pipeline ------------------------------------------------------------


def preprocess(raw_text: str) -> PreprocessedText:
    """
    Run the full preprocessing pipeline on a piece of raw text (resume
    text or job description text) and return every stage.

    The input `raw_text` is stored unchanged in the result -- nothing in
    this module ever mutates or discards the original text.
    """
    raw_text = raw_text or ""

    unicode_clean = normalize_unicode(raw_text)
    whitespace_clean = clean_whitespace(unicode_clean)
    clean_text = normalize_punctuation(whitespace_clean)

    normalized_text = normalize_for_matching(clean_text)
    tokens = tokenize(normalized_text)

    return PreprocessedText(
        raw_text=raw_text,
        clean_text=clean_text,
        normalized_text=normalized_text,
        tokens=tokens,
    )
