"""
Skill extraction service.

Deterministic, no AI/LLM involved anywhere in this file.

FEATURE 3A -- skill detection + job description analysis:
    find_skills()            -- detect dictionary skills in any text
    analyze_job_description() -- structured analysis of a JD: detected
                                  skills, which are flagged "required" by
                                  context, education/experience mentions,
                                  responsibility terms

FEATURE 3B -- resume/JD comparison:
    compare_resume_to_job()  -- resume skills vs. JD skills -> matched /
                                  missing, required vs. general
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Pattern, Tuple

from services.skill_dictionary import MULTI_WORD_PATTERNS, SINGLE_WORD_LOOKUP
from services.text_preprocessing import preprocess

# ============================================================================
# FEATURE 3A -- Skill detection
# ============================================================================


@dataclass(frozen=True)
class SkillMatch:
    """One skill detected in a piece of text."""
    canonical_name: str
    category: str


def find_skills(tokens: List[str], normalized_text: str) -> List[SkillMatch]:
    """
    Detect every dictionary skill present in the given tokens/normalized
    text.

    Args:
        tokens: token list from text_preprocessing.tokenize() (or
            PreprocessedText.tokens).
        normalized_text: the lowercased, whitespace-clean text from
            PreprocessedText.normalized_text -- used for multi-word
            phrase matching.

    Returns:
        A list of SkillMatch, one per DISTINCT skill found (never two
        entries for the same skill even if several of its aliases, or
        the same alias multiple times, appear in the text) -- sorted
        alphabetically by canonical name for deterministic output.
    """
    token_set = set(tokens)
    matched: Dict[str, SkillMatch] = {}

    for alias, skill in SINGLE_WORD_LOOKUP.items():
        if skill.canonical_name in matched:
            continue
        if alias in token_set:
            matched[skill.canonical_name] = SkillMatch(skill.canonical_name, skill.category)

    for pattern, skill in MULTI_WORD_PATTERNS:
        if skill.canonical_name in matched:
            continue
        if pattern.search(normalized_text):
            matched[skill.canonical_name] = SkillMatch(skill.canonical_name, skill.category)

    return sorted(matched.values(), key=lambda s: s.canonical_name)


def find_skills_in_raw_text(raw_text: str) -> List[SkillMatch]:
    """Convenience wrapper: preprocess raw text, then find_skills() on it."""
    result = preprocess(raw_text)
    return find_skills(result.tokens, result.normalized_text)


# --- Required-skill context detection ---------------------------------------

# Phrases that signal "what follows is a hard requirement", based on
# common JD phrasing. This list is intentionally explicit and small --
# every phrase here was requested and can be pointed to directly when
# explaining why a skill was (or wasn't) flagged as required.
REQUIRED_CONTEXT_PHRASES: Tuple[str, ...] = (
    "required skills",
    "must have",
    "must know",
    "should have",
    "key skills",
    "proficiency in",
    "experience with",
    "requirements",
    "required",
)

# How far forward (characters) we scan after a cue phrase looking for
# skills, or up to the next sentence-ending punctuation, whichever
# comes first. This bounds the search so a cue phrase near the top of a
# JD doesn't sweep in unrelated skills mentioned paragraphs later.
_REQUIRED_CONTEXT_WINDOW_CHARS = 200

_SENTENCE_END_RE = re.compile(r"[.!?]")

# Phrases that signal a shift AWAY from hard requirements. Preprocessing
# collapses newlines/section breaks (see text_preprocessing.clean_whitespace),
# so without this, a forward window starting in a "Required Skills:"
# section can run straight into a following "Nice to have:" section and
# wrongly flag optional skills as required. Scanning also stops here,
# in addition to sentence-ending punctuation.
_BOUNDARY_PHRASES: Tuple[str, ...] = (
    "nice to have",
    "good to have",
    "preferred",
    "bonus",
    "optional",
    "desirable",
)


def _forward_window(text: str, start: int, max_chars: int = _REQUIRED_CONTEXT_WINDOW_CHARS) -> str:
    """
    Return the text from `start` up to whichever comes first: the next
    sentence-ending punctuation, the next boundary phrase (see
    _BOUNDARY_PHRASES), or max_chars.
    """
    segment = text[start:start + max_chars]
    cut_points = []

    end_match = _SENTENCE_END_RE.search(segment)
    if end_match:
        cut_points.append(end_match.start())

    for phrase in _BOUNDARY_PHRASES:
        idx = segment.find(phrase)
        if idx != -1:
            cut_points.append(idx)

    if cut_points:
        return segment[:min(cut_points)]
    return segment


def detect_required_skill_names(normalized_text: str) -> set:
    """
    Find which canonical skill names are flagged as "required" by
    context, using REQUIRED_CONTEXT_PHRASES.

    Method (deterministic, documented): for every occurrence of a cue
    phrase, take the text in a forward window immediately after it
    (bounded by _forward_window -- stopping at the next sentence end,
    the next boundary phrase like "nice to have"/"preferred", or a
    fixed character cap), run the normal skill detector on just that
    window, and union the results across all cue-phrase occurrences.

    This is intentionally a forward-looking heuristic: it catches
    "Required skills: Python, Java, SQL" and "Proficiency in Python is
    required", but will NOT catch a JD phrased skill-then-cue, e.g.
    "Python, Java -- required" with the cue after the skills. That
    limitation is documented in the Feature 3A write-up.

    A JD with none of these cue phrases at all will return an empty
    set -- we deliberately do not assume every skill mentioned is a
    hard requirement (see docstring of analyze_job_description).
    """
    required_names = set()
    for phrase in REQUIRED_CONTEXT_PHRASES:
        for match in re.finditer(re.escape(phrase), normalized_text):
            window_text = _forward_window(normalized_text, match.end())
            window_preprocessed = preprocess(window_text)
            for skill_match in find_skills(window_preprocessed.tokens, window_preprocessed.normalized_text):
                required_names.add(skill_match.canonical_name)
    return required_names


# --- Education requirement detection -----------------------------------------

_EDUCATION_PATTERNS: Tuple[Pattern, ...] = tuple(
    re.compile(p) for p in (
        r"\bb\.?\s?tech\b",
        r"\bm\.?\s?tech\b",
        r"\bb\.?\s?e\b",
        r"\bb\.?\s?sc\b",
        r"\bm\.?\s?sc\b",
        r"\bbca\b",
        r"\bmca\b",
        r"\bbachelor'?s?\s+degree\b",
        r"\bmaster'?s?\s+degree\b",
        r"\bbachelor'?s?\s+in\b",
        r"\bmaster'?s?\s+in\b",
        r"\bdegree\s+in\b",
        r"\bcomputer science\b",
        r"\bcomputer engineering\b",
        r"\binformation technology\b",
        r"\bengineering\s+degree\b",
        r"\bgraduate\b",
        r"\bgraduation\b",
    )
)


def detect_education_requirements(normalized_text: str) -> List[str]:
    """
    Return the distinct education-related phrases found in the text
    (e.g. "b.tech", "computer science", "bachelor's degree"), in the
    order their pattern is checked. Deterministic pattern matching, not
    an attempt at full semantic understanding of degree requirements.
    """
    found = []
    for pattern in _EDUCATION_PATTERNS:
        match = pattern.search(normalized_text)
        if match:
            phrase = match.group(0).strip()
            if phrase not in found:
                found.append(phrase)
    return found


# --- Experience requirement detection ----------------------------------------

_EXPERIENCE_PATTERNS: Tuple[Pattern, ...] = tuple(
    re.compile(p) for p in (
        r"\d+\+?\s*(?:-\s*\d+\s*)?\s*years?\s*(?:of\s+)?experience",
        r"\bfresher[s]?\b",
        r"\bentry[\s-]level\b",
        r"\bno\s+prior\s+experience\b",
        r"\bnew\s+graduate[s]?\b",
    )
)


def detect_experience_requirements(normalized_text: str) -> List[str]:
    """
    Return the distinct experience-related phrases found in the text
    (e.g. "2+ years of experience", "fresher"), in the order their
    pattern is checked.
    """
    found = []
    for pattern in _EXPERIENCE_PATTERNS:
        for match in pattern.finditer(normalized_text):
            phrase = match.group(0).strip()
            if phrase not in found:
                found.append(phrase)
    return found


# --- Responsibility term detection -------------------------------------------
#
# Not "skills" in the technical sense, but recurring JD phrases about
# what the role actually involves day-to-day. Matched the same
# boundary-safe way as skills (space/hyphen-flexible phrase regex),
# just against a separate, smaller vocabulary.

_RESPONSIBILITY_TERMS: Tuple[Tuple[str, str], ...] = (
    ("software development", "software development"),
    ("scalable software", "scalable software"),
    ("reusable software", "reusable software"),
    ("reliable software", "reliable software"),
    ("building features", "building features"),
    ("maintaining modules", "maintaining modules"),
    ("maintaining existing modules", "maintaining existing modules"),
    ("problem solving", "problem solving"),
    ("analytical thinking", "analytical thinking"),
    ("communication", "communication"),
    ("communication skills", "communication skills"),
    ("team collaboration", "team collaboration"),
    ("code review", "code review"),
    ("debugging", "debugging"),
    ("testing", "testing"),
)


def _build_responsibility_patterns() -> List[Tuple[Pattern, str]]:
    patterns = []
    for alias, canonical in _RESPONSIBILITY_TERMS:
        words = alias.split(" ")
        if len(words) == 1:
            pattern = re.compile(r"\b" + re.escape(words[0]) + r"\b")
        else:
            pattern = re.compile(r"\b" + r"[\s\-]+".join(re.escape(w) for w in words) + r"\b")
        patterns.append((pattern, canonical))
    return patterns


_RESPONSIBILITY_PATTERNS = _build_responsibility_patterns()


def detect_responsibility_terms(normalized_text: str) -> List[str]:
    """
    Return the distinct responsibility-related phrases found in the
    text, in dictionary order, deduplicated (e.g. "communication" and
    "communication skills" both present would only report once each,
    not merged -- they're treated as related but distinct phrases here
    since responsibility terms aren't canonicalized like skills are).
    """
    found = []
    for pattern, canonical in _RESPONSIBILITY_PATTERNS:
        if pattern.search(normalized_text) and canonical not in found:
            found.append(canonical)
    return found


# --- Full job description analysis -------------------------------------------


@dataclass
class JobDescriptionAnalysis:
    detected_skills: List[SkillMatch] = field(default_factory=list)
    required_skills: List[SkillMatch] = field(default_factory=list)
    general_skills: List[SkillMatch] = field(default_factory=list)
    education_requirements: List[str] = field(default_factory=list)
    experience_requirements: List[str] = field(default_factory=list)
    responsibility_terms: List[str] = field(default_factory=list)


def analyze_job_description(raw_text: str) -> JobDescriptionAnalysis:
    """
    Run the full, deterministic job-description analysis pipeline.

    NOT every skill mentioned in a JD is treated as a hard requirement
    -- only skills found within a forward window of one of
    REQUIRED_CONTEXT_PHRASES are flagged "required"; everything else
    detected goes into "general_skills". This is a deliberate,
    documented design choice (see detect_required_skill_names), not an
    oversight: a JD that casually mentions a technology in a "nice to
    have" or narrative context shouldn't be treated the same as one
    that explicitly requires it.
    """
    preprocessed = preprocess(raw_text)
    detected = find_skills(preprocessed.tokens, preprocessed.normalized_text)

    required_names = detect_required_skill_names(preprocessed.normalized_text)
    required = [s for s in detected if s.canonical_name in required_names]
    general = [s for s in detected if s.canonical_name not in required_names]

    return JobDescriptionAnalysis(
        detected_skills=detected,
        required_skills=required,
        general_skills=general,
        education_requirements=detect_education_requirements(preprocessed.normalized_text),
        experience_requirements=detect_experience_requirements(preprocessed.normalized_text),
        responsibility_terms=detect_responsibility_terms(preprocessed.normalized_text),
    )


# ============================================================================
# FEATURE 3B -- Resume / job description comparison
# ============================================================================


@dataclass
class SkillComparisonResult:
    resume_skills: List[SkillMatch] = field(default_factory=list)
    job_skills: List[SkillMatch] = field(default_factory=list)
    required_skills: List[SkillMatch] = field(default_factory=list)
    matched_required_skills: List[SkillMatch] = field(default_factory=list)
    missing_required_skills: List[SkillMatch] = field(default_factory=list)
    matched_general_skills: List[SkillMatch] = field(default_factory=list)
    missing_general_skills: List[SkillMatch] = field(default_factory=list)
    # Added for Feature 4 (scoring engine): the JD-side context that
    # education/experience scoring needs, already computed inside
    # compare_resume_to_job() via analyze_job_description() -- exposed
    # here so scoring.py doesn't need to re-run JD analysis itself.
    # Purely additive: every field above is unchanged, so Feature 3A/3B
    # tests are unaffected.
    education_requirements: List[str] = field(default_factory=list)
    experience_requirements: List[str] = field(default_factory=list)
    responsibility_terms: List[str] = field(default_factory=list)


def compare_resume_to_job(resume_raw_text: str, job_raw_text: str) -> SkillComparisonResult:
    """
    Compare a resume's detected skills against a job description's
    detected skills, splitting the JD's required and general skills
    each into matched/missing based on whether the resume has them.

    Every list here is already deduplicated by canonical skill name
    (find_skills() and analyze_job_description() guarantee that), so
    e.g. a resume mentioning "Python" and "Python programming" still
    only ever produces one "Python" entry -- never two.
    """
    resume_preprocessed = preprocess(resume_raw_text)
    resume_skills = find_skills(resume_preprocessed.tokens, resume_preprocessed.normalized_text)
    resume_names = {s.canonical_name for s in resume_skills}

    jd_analysis = analyze_job_description(job_raw_text)

    matched_required = [s for s in jd_analysis.required_skills if s.canonical_name in resume_names]
    missing_required = [s for s in jd_analysis.required_skills if s.canonical_name not in resume_names]
    matched_general = [s for s in jd_analysis.general_skills if s.canonical_name in resume_names]
    missing_general = [s for s in jd_analysis.general_skills if s.canonical_name not in resume_names]

    return SkillComparisonResult(
        resume_skills=resume_skills,
        job_skills=jd_analysis.detected_skills,
        required_skills=jd_analysis.required_skills,
        matched_required_skills=matched_required,
        missing_required_skills=missing_required,
        matched_general_skills=matched_general,
        missing_general_skills=missing_general,
        education_requirements=jd_analysis.education_requirements,
        experience_requirements=jd_analysis.experience_requirements,
        responsibility_terms=jd_analysis.responsibility_terms,
    )
