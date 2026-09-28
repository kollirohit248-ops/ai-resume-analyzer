"""
Section detection utility.

Deterministic, no AI/NLP library involved. Operates on the RAW,
line-preserving resume text -- deliberately NOT the whitespace-collapsed
output of text_preprocessing.clean_whitespace(), because that collapse
(by design, see Feature 2) discards the line breaks section detection
needs. This module does not change, replace, or duplicate Feature 2's
pipeline; it is a separate, narrow pass over the original text used
only to support Feature 4's experience/project-relevance heuristic.

Heading detection is intentionally conservative: a line counts as a
section heading ONLY if, once stripped of surrounding whitespace and
punctuation, it matches a known heading phrase ON ITS OWN -- not if the
word merely appears somewhere in a longer line. This is what correctly
tells "PROJECTS" (a heading) apart from "Developed projects using
Python and SQL." (an ordinary sentence that happens to contain the
word "projects").
"""
import re
from dataclasses import dataclass, field
from typing import List

# Headings that count as "practical" evidence -- work actually done,
# not just claimed. A skill mentioned under one of these headings is
# treated as demonstrated in practice.
PRACTICAL_SECTION_HEADINGS = {
    "experience",
    "work experience",
    "professional experience",
    "relevant experience",
    "projects",
    "project experience",
    "personal projects",
    "academic projects",
    "internship",
    "internships",
    "internship experience",
    "employment",
    "employment history",
    "work history",
}

# Headings recognized but NOT treated as practical evidence -- a skill
# listed only here (and nowhere practical) is "claimed" but not
# demonstrated.
OTHER_SECTION_HEADINGS = {
    "skills",
    "technical skills",
    "core skills",
    "key skills",
    "education",
    "educational qualification",
    "educational background",
    "academic background",
    "certifications",
    "certificates",
    "achievements",
    "awards",
    "summary",
    "objective",
    "profile",
    "contact",
    "about",
    "hobbies",
    "interests",
    "languages",
    "references",
}

ALL_KNOWN_HEADINGS = PRACTICAL_SECTION_HEADINGS | OTHER_SECTION_HEADINGS

# A heading line may end with a colon, or use light decoration like
# "== Projects ==" or "* Projects *" -- stripped before comparison.
_HEADING_STRIP_CHARS = " \t:-=*_#•"


def _normalize_heading_line(line: str) -> str:
    """Strip decoration/whitespace and lowercase, for heading comparison."""
    return line.strip().strip(_HEADING_STRIP_CHARS).strip().lower()


def is_heading_line(line: str) -> bool:
    """
    True only if the WHOLE line (after stripping decoration/whitespace)
    exactly matches a known heading phrase. Deliberately strict -- a
    line containing other words ("Developed projects using Python")
    never qualifies, regardless of which keyword it contains.
    """
    return _normalize_heading_line(line) in ALL_KNOWN_HEADINGS


@dataclass
class ResumeSection:
    """One detected section of the resume."""
    name: str          # normalized heading text, e.g. "projects"
    heading: str        # the original heading line as it appeared
    text: str            # the section's body text (lines under the heading)
    is_practical: bool  # True for Experience/Projects/Internship/Employment


@dataclass
class SectionizedResume:
    sections: List[ResumeSection] = field(default_factory=list)
    preamble_text: str = ""  # text before the first recognized heading

    @property
    def practical_text(self) -> str:
        """Concatenated body text of every practical section."""
        return "\n".join(s.text for s in self.sections if s.is_practical)

    @property
    def has_practical_sections(self) -> bool:
        return any(s.is_practical for s in self.sections)


def sectionize(raw_text: str) -> SectionizedResume:
    """
    Split raw resume text into sections by scanning line by line for
    known heading phrases (see is_heading_line).

    Text before the first recognized heading is kept separately as
    `preamble_text` (typically a name/contact/summary block) and is
    never treated as practical, since it has no heading telling us
    what it is.

    If no headings are recognized at all, `sections` is empty and the
    entire input ends up in `preamble_text` -- we do not guess section
    boundaries beyond the exact-heading-line rule above.
    """
    if not raw_text:
        return SectionizedResume(sections=[], preamble_text="")

    lines = raw_text.splitlines()
    sections: List[ResumeSection] = []
    preamble_lines: List[str] = []
    current_lines: List[str] = []
    current_name = None
    current_heading = None
    current_is_practical = False

    def _flush_current():
        if current_name is not None:
            sections.append(
                ResumeSection(
                    name=current_name,
                    heading=current_heading,
                    text="\n".join(current_lines).strip(),
                    is_practical=current_is_practical,
                )
            )

    for line in lines:
        if is_heading_line(line):
            _flush_current()
            current_name = _normalize_heading_line(line)
            current_heading = line.strip()
            current_is_practical = current_name in PRACTICAL_SECTION_HEADINGS
            current_lines = []
            continue

        if current_name is None:
            preamble_lines.append(line)
        else:
            current_lines.append(line)

    _flush_current()

    return SectionizedResume(
        sections=sections,
        preamble_text="\n".join(preamble_lines).strip(),
    )
