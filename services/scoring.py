"""
ATS-style compatibility scoring engine.

Deterministic, no AI/LLM involved anywhere in this file -- the LLM
(Feature 5) will consume this module's output for qualitative
commentary, but has no influence over the numeric score.

This is an APPLICATION-SPECIFIC DETERMINISTIC ATS-STYLE COMPATIBILITY
HEURISTIC, not an official/industry ATS algorithm. It is built from
four components:

    R -- Required Skill Match      (weight 0.50)
    K -- General Keyword Overlap   (weight 0.25)
    E -- Experience/Project Relevance (weight 0.15)
    D -- Education Relevance       (weight 0.10)

    score = 100 x (0.50R + 0.25K + 0.15E + 0.10D)
    score = round(clamp(score, 0, 100))

A component that is genuinely not applicable to the given resume/JD
pair (see each compute_* function's docstring for exactly when that
is) is excluded, and its weight is redistributed proportionally among
the remaining applicable components -- it is never silently treated as
zero. A component that IS applicable but happens to evaluate to zero
(e.g. the JD has required skills but the resume matches none of them)
correctly stays in the score as a real, weighted zero.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from exceptions import EmptyJobDescriptionTextError, EmptyResumeTextError
from services.section_detection import sectionize
from services.skill_extraction import (
    SkillMatch,
    compare_resume_to_job,
    detect_education_requirements,
    find_skills,
)
from services.text_preprocessing import preprocess

# Component weights, before any redistribution.
_WEIGHTS = {"R": 0.50, "K": 0.25, "E": 0.15, "D": 0.10}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ============================================================================
# Structured result
# ============================================================================


@dataclass
class ExperienceRelevanceDetail:
    """Extra detail behind the E component, for the UI/explanation."""
    matched_skills_considered: List[str] = field(default_factory=list)
    demonstrated_in_practice: List[str] = field(default_factory=list)
    not_demonstrated: List[str] = field(default_factory=list)
    practical_sections_found: List[str] = field(default_factory=list)


@dataclass
class ATSScoreResult:
    final_score: int

    required_skill_score: Optional[float]      # 0-100, or None if not applicable
    general_keyword_score: Optional[float]
    experience_project_score: Optional[float]
    education_score: Optional[float]

    matched_required_skills: List[str] = field(default_factory=list)
    missing_required_skills: List[str] = field(default_factory=list)
    matched_general_skills: List[str] = field(default_factory=list)
    missing_general_skills: List[str] = field(default_factory=list)

    education_match: Optional[bool] = None
    experience_relevance: ExperienceRelevanceDetail = field(default_factory=ExperienceRelevanceDetail)

    applicable_components: List[str] = field(default_factory=list)
    redistributed_weights: Dict[str, float] = field(default_factory=dict)
    low_confidence: bool = False

    explanation: str = ""


# ============================================================================
# Component R -- Required Skill Match (weight 0.50)
# ============================================================================


def _compute_r(matched_required: List[SkillMatch], total_required: int) -> Optional[float]:
    """
    R = matched required skills / total required skills.

    Not applicable (None) only when the JD has NO detectable required
    skills at all (total_required == 0) -- there is nothing to measure
    against. If the JD DOES have required skills but the resume matches
    none of them, R is a real, applicable 0.0 (not "not applicable") --
    that's a meaningful, weighted signal that the resume fails the
    JD's hard requirements.
    """
    if total_required == 0:
        return None
    return len(matched_required) / total_required


# ============================================================================
# Component K -- General Keyword / Context Overlap (weight 0.25)
# ============================================================================


def _compute_k(matched_general: List[SkillMatch], total_general: int) -> Optional[float]:
    """
    K = matched general (non-required) skills / total general skills.

    Not applicable (None) only when the JD has no detected general
    skills at all. Otherwise, same logic as R: zero matches is a real,
    applicable 0.0.
    """
    if total_general == 0:
        return None
    return len(matched_general) / total_general


# ============================================================================
# Component E -- Experience / Project Relevance (weight 0.15)
# ============================================================================


def _compute_e(
    matched_skill_names: set,
    total_jd_skills: int,
    resume_raw_text: str,
) -> (Optional[float], ExperienceRelevanceDetail):
    """
    E = (matched skills that also appear in a practical resume section)
        / (total matched skills)

    "Practical section" = Experience/Work Experience/Professional
    Experience/Projects/Internship/Employment, as detected by
    services.section_detection.sectionize() on the resume's RAW text
    (which preserves line breaks -- Feature 2's normalized text does
    not, by design, so this module reads the raw text separately for
    this one purpose). A skill only listed under "Technical Skills" or
    "Education" does NOT count as practically demonstrated, even if it
    matches the JD.

    This is a HEURISTIC, not semantic understanding: it only checks
    whether a skill's name/alias literally appears within the text of
    a recognized practical section -- it cannot tell whether the skill
    was meaningfully *used* there versus just mentioned in passing.

    Three distinct cases, each handled differently (documented,
    consistent):

      1. total_jd_skills == 0 (JD has no detectable skills at all)
         -> E is NOT APPLICABLE. There's nothing to be "relevant" to.

      2. total_jd_skills > 0 but matched_skill_names is empty (resume
         doesn't overlap with the JD's skills at all)
         -> E = 0.0 (applicable). A real signal: none of the JD's
         skills are demonstrated anywhere, practically or otherwise.

      3. matched_skill_names is non-empty, but the resume has NO
         detectable practical sections at all (no Experience/Projects/
         Internship/Employment heading found anywhere)
         -> E is NOT APPLICABLE, not 0. We deliberately do NOT treat
         "we couldn't detect a Projects/Experience section" the same
         as "this candidate has no practical experience" -- those are
         different claims, and the former is a limitation of our
         heading-based detector (e.g. an unconventional resume
         format), not evidence about the candidate. Scoring it as 0
         would unfairly penalize a resume our simple heading matcher
         just didn't parse. Its weight is redistributed instead.

      4. matched_skill_names is non-empty AND at least one practical
         section was detected
         -> E = (matched skills found in that section text) / (total
         matched skills). This can legitimately compute to 0.0 if
         sections exist but none of the matched skills appear in them
         -- that IS a fair, applicable signal.
    """
    if total_jd_skills == 0:
        return None, ExperienceRelevanceDetail()

    if not matched_skill_names:
        return 0.0, ExperienceRelevanceDetail()

    sectioned = sectionize(resume_raw_text)
    if not sectioned.has_practical_sections:
        return None, ExperienceRelevanceDetail(
            matched_skills_considered=sorted(matched_skill_names),
        )

    practical_preprocessed = preprocess(sectioned.practical_text)
    practical_skill_names = {
        s.canonical_name
        for s in find_skills(practical_preprocessed.tokens, practical_preprocessed.normalized_text)
    }

    demonstrated = matched_skill_names & practical_skill_names
    not_demonstrated = matched_skill_names - practical_skill_names
    e_value = len(demonstrated) / len(matched_skill_names)

    detail = ExperienceRelevanceDetail(
        matched_skills_considered=sorted(matched_skill_names),
        demonstrated_in_practice=sorted(demonstrated),
        not_demonstrated=sorted(not_demonstrated),
        practical_sections_found=[s.name for s in sectioned.sections if s.is_practical],
    )
    return e_value, detail


# ============================================================================
# Component D -- Education Relevance (weight 0.10)
# ============================================================================


def _compute_d(jd_education: List[str], resume_raw_text: str) -> (Optional[float], Optional[bool]):
    """
    D = 1.0 if the resume mentions at least one of the same
        education-related phrases the JD requires, else 0.0.

    Not applicable (None) when the JD itself states no detectable
    education requirement -- we never invent a requirement the JD
    didn't state. When the JD DOES state one, comparison is a simple
    overlap check between the JD's detected education phrases (see
    skill_extraction.detect_education_requirements, e.g. "b.tech",
    "computer science") and the same phrases detected in the resume.
    This is deliberately simple set-overlap, not a judgment about
    degree prestige or field closeness.
    """
    if not jd_education:
        return None, None

    resume_preprocessed = preprocess(resume_raw_text)
    resume_education = detect_education_requirements(resume_preprocessed.normalized_text)

    overlap = set(jd_education) & set(resume_education)
    matched = bool(overlap)
    return (1.0 if matched else 0.0), matched


# ============================================================================
# Explanation text
# ============================================================================


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{round(value * 100)}%"


def _build_explanation(
    final_score: int,
    r: Optional[float], k: Optional[float], e: Optional[float], d: Optional[float],
    matched_required: List[str], missing_required: List[str],
    matched_general: List[str], missing_general: List[str],
    low_confidence: bool,
) -> str:
    lines = [f"ATS-style Compatibility Score: {final_score}", ""]

    if low_confidence:
        lines.append(
            "LOW CONFIDENCE: little or no skill information could be detected "
            "in the job description, so this score may not be meaningful."
        )
        lines.append("")

    lines.append(f"Required Skills: {_pct(r)}")
    lines.append(f"General Keywords: {_pct(k)}")
    lines.append(f"Experience/Project Relevance: {_pct(e)}")
    lines.append(f"Education: {_pct(d)}")
    lines.append("")

    lines.append("Matched Required Skills:")
    lines.append("\n".join(matched_required) if matched_required else "None")
    lines.append("")

    lines.append("Missing Required Skills:")
    lines.append("\n".join(missing_required) if missing_required else "None")
    lines.append("")

    lines.append("Matched General Skills:")
    lines.append("\n".join(matched_general) if matched_general else "None")
    lines.append("")

    lines.append("Missing General Skills:")
    lines.append("\n".join(missing_general) if missing_general else "None")

    return "\n".join(lines)


# ============================================================================
# Main entry point
# ============================================================================


def compute_ats_score(resume_raw_text: str, job_raw_text: str) -> ATSScoreResult:
    """
    Compute the deterministic ATS-style compatibility score for a
    resume against a job description.

    Raises:
        EmptyResumeTextError: resume_raw_text is empty/whitespace-only.
        EmptyJobDescriptionTextError: job_raw_text is empty/whitespace-only.
            Neither case produces a score -- a score computed from no
            input would be meaningless, not just low-confidence.
    """
    if not resume_raw_text or not resume_raw_text.strip():
        raise EmptyResumeTextError("Resume text is empty; a compatibility score cannot be calculated.")
    if not job_raw_text or not job_raw_text.strip():
        raise EmptyJobDescriptionTextError("Job description text is empty; a compatibility score cannot be calculated.")

    comparison = compare_resume_to_job(resume_raw_text, job_raw_text)

    total_required = len(comparison.matched_required_skills) + len(comparison.missing_required_skills)
    total_general = len(comparison.matched_general_skills) + len(comparison.missing_general_skills)
    total_jd_skills = len(comparison.job_skills)

    matched_skill_names = {s.canonical_name for s in comparison.matched_required_skills} | {
        s.canonical_name for s in comparison.matched_general_skills
    }

    r = _compute_r(comparison.matched_required_skills, total_required)
    k = _compute_k(comparison.matched_general_skills, total_general)
    e, experience_detail = _compute_e(matched_skill_names, total_jd_skills, resume_raw_text)
    d, education_match = _compute_d(comparison.education_requirements, resume_raw_text)

    components = {"R": (r, _WEIGHTS["R"]), "K": (k, _WEIGHTS["K"]), "E": (e, _WEIGHTS["E"]), "D": (d, _WEIGHTS["D"])}
    applicable = {name: (val, weight) for name, (val, weight) in components.items() if val is not None}

    # "Nothing extractable from the JD" -- no skill signal of any kind
    # (neither required nor general skills were detected at all). A
    # score built from just D (or nothing) alone isn't a meaningful
    # compatibility signal, so we flag it rather than presenting a
    # confident-looking number.
    low_confidence = total_jd_skills == 0

    if not applicable:
        final_score = 0
        redistributed_weights: Dict[str, float] = {}
        low_confidence = True
    else:
        total_weight = sum(weight for _, weight in applicable.values())
        redistributed_weights = {name: weight / total_weight for name, (_, weight) in applicable.items()}
        score_fraction = sum(val * redistributed_weights[name] for name, (val, _) in applicable.items())
        final_score = round(_clamp(score_fraction * 100, 0, 100))

    matched_required_names = sorted(s.canonical_name for s in comparison.matched_required_skills)
    missing_required_names = sorted(s.canonical_name for s in comparison.missing_required_skills)
    matched_general_names = sorted(s.canonical_name for s in comparison.matched_general_skills)
    missing_general_names = sorted(s.canonical_name for s in comparison.missing_general_skills)

    explanation = _build_explanation(
        final_score, r, k, e, d,
        matched_required_names, missing_required_names,
        matched_general_names, missing_general_names,
        low_confidence,
    )

    return ATSScoreResult(
        final_score=final_score,
        required_skill_score=(round(r * 100) if r is not None else None),
        general_keyword_score=(round(k * 100) if k is not None else None),
        experience_project_score=(round(e * 100) if e is not None else None),
        education_score=(round(d * 100) if d is not None else None),
        matched_required_skills=matched_required_names,
        missing_required_skills=missing_required_names,
        matched_general_skills=matched_general_names,
        missing_general_skills=missing_general_names,
        education_match=education_match,
        experience_relevance=experience_detail,
        applicable_components=sorted(applicable.keys()),
        redistributed_weights=redistributed_weights,
        low_confidence=low_confidence,
        explanation=explanation,
    )
