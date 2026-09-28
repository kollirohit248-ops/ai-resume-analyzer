"""
Skill dictionary.

A data-driven, maintainable list of known software-engineering / AI
skills, each with a canonical display name, a category, and the
aliases (alternate spellings/abbreviations) that should count as the
same skill.

This module holds DATA and small structural helpers only -- the actual
matching logic (how we search resume/JD text for these skills) lives
in skill_extraction.py. Keeping them separate means extending the
dictionary (adding a new skill or alias) never requires touching
matching logic, and vice versa.

--- How matching works, at a glance -----------------------------------

Each alias is classified by whether it contains a space:

  * Single-word aliases ("python", "c++", "c#", ".net", "sql") are
    matched by exact membership in the token list produced by
    text_preprocessing.tokenize(). Because that tokenizer already keeps
    "c++"/"c#"/".net"/"node.js" as single, undestroyed tokens, an exact
    token match is inherently boundary-safe: "c" as a token can never
    equal "c++" or "javascript" as a token, so there's no risk of a
    substring false positive. This is the main defense against false
    positives like "C" matching inside "JavaScript" or "SQL" matching
    inside "SQLite".

  * Multi-word aliases ("machine learning", "object oriented
    programming") are matched with a word-boundary regex against the
    normalized (lowercased, whitespace-collapsed) text, built so that a
    hyphen or a space between words both count as a valid separator
    (so "Object-Oriented Programming" and "object oriented programming"
    both match the same alias without needing two dictionary entries).
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Pattern, Tuple


@dataclass(frozen=True)
class Skill:
    """One entry in the skill dictionary."""
    canonical_name: str   # display name, e.g. "Object-Oriented Programming"
    category: str         # e.g. "Programming", "AI/ML", "Core CS"
    aliases: Tuple[str, ...]  # lowercase alias strings that mean this skill


# --- The dictionary itself ---------------------------------------------
#
# Deliberately NOT exhaustive -- covers the categories requested for
# this project (Programming, Web/Backend, Database, AI/ML, Data, Core
# CS, Tools). Extend by adding a Skill(...) entry; nothing else needs
# to change for a new skill to start being detected.

SKILLS: List[Skill] = [
    # --- Programming ---
    Skill("Python", "Programming", ("python",)),
    Skill("Java", "Programming", ("java",)),
    Skill("C", "Programming", ("c",)),
    Skill("C++", "Programming", ("c++",)),
    Skill("C#", "Programming", ("c#",)),
    Skill("JavaScript", "Programming", ("javascript", "js")),
    Skill("TypeScript", "Programming", ("typescript", "ts")),
    Skill(".NET", "Programming", (".net",)),

    # --- Web / Backend ---
    Skill("Flask", "Web/Backend", ("flask",)),
    Skill("FastAPI", "Web/Backend", ("fastapi",)),
    Skill("Django", "Web/Backend", ("django",)),
    Skill("REST API", "Web/Backend", ("rest api", "restful api", "rest apis")),
    Skill("HTTP", "Web/Backend", ("http",)),
    Skill("Node.js", "Web/Backend", ("node.js", "nodejs", "node js")),
    Skill("React", "Web/Backend", ("react", "react.js", "reactjs")),
    Skill("HTML", "Web/Backend", ("html",)),
    Skill("CSS", "Web/Backend", ("css",)),

    # --- Database ---
    Skill("SQL", "Database", ("sql",)),
    Skill("SQLite", "Database", ("sqlite",)),
    Skill("MySQL", "Database", ("mysql",)),
    Skill("PostgreSQL", "Database", ("postgresql", "postgres")),
    Skill("MongoDB", "Database", ("mongodb", "mongo")),
    Skill("Redis", "Database", ("redis",)),

    # --- AI / ML ---
    Skill("Artificial Intelligence", "AI/ML", ("artificial intelligence", "ai")),
    Skill("Machine Learning", "AI/ML", ("machine learning", "ml")),
    Skill("Deep Learning", "AI/ML", ("deep learning",)),
    Skill(
        "Natural Language Processing",
        "AI/ML",
        ("natural language processing", "nlp"),
    ),
    Skill("Generative AI", "AI/ML", ("generative ai", "genai")),
    Skill(
        "Large Language Models",
        "AI/ML",
        ("large language models", "large language model", "llm", "llms"),
    ),
    Skill("Transformers", "AI/ML", ("transformers", "transformer")),
    Skill("Embeddings", "AI/ML", ("embeddings", "embedding")),
    # Deliberately NOT aliased to "cv" -- resumes routinely use "CV" to
    # mean "curriculum vitae", so that abbreviation would be a
    # near-guaranteed false positive. Full phrase only.
    Skill("Computer Vision", "AI/ML", ("computer vision",)),

    # --- Data ---
    Skill("Pandas", "Data", ("pandas",)),
    Skill("NumPy", "Data", ("numpy",)),
    Skill("Data Analysis", "Data", ("data analysis",)),
    Skill("Data Science", "Data", ("data science",)),

    # --- Core CS ---
    Skill(
        "Data Structures and Algorithms",
        "Core CS",
        ("data structures and algorithms", "data structures & algorithms", "dsa"),
    ),
    Skill(
        "Object-Oriented Programming",
        "Core CS",
        ("object oriented programming", "oop"),
    ),
    Skill(
        "DBMS",
        "Core CS",
        ("dbms", "database management systems", "database management system"),
    ),
    # Deliberately NOT aliased to "os" -- too short/ambiguous as a
    # standalone token (could realistically appear in other contexts).
    Skill("Operating Systems", "Core CS", ("operating systems", "operating system")),
    Skill("Computer Networks", "Core CS", ("computer networks", "computer networking")),

    # --- Tools ---
    Skill("Git", "Tools", ("git",)),
    Skill("GitHub", "Tools", ("github",)),
    Skill("Docker", "Tools", ("docker",)),
    Skill("Linux", "Tools", ("linux",)),
    Skill("VS Code", "Tools", ("vs code", "visual studio code", "vscode")),
]


# --- Build-time lookup structures ---------------------------------------
#
# Built once at import time so extraction doesn't rebuild these on every
# call. Two structures, matching the two matching strategies described
# in the module docstring.

def _build_lookup_structures(
    skills: List[Skill],
) -> Tuple[Dict[str, Skill], List[Tuple[Pattern, Skill]]]:
    """
    Split every skill's aliases into:
      - single_word_lookup: alias -> Skill, for exact token matching
      - multi_word_patterns: [(compiled regex, Skill), ...], for phrase
        matching against normalized text

    Raises ValueError at import time if two different skills claim the
    same alias -- a dictionary-authoring bug we want to catch
    immediately rather than silently mismatching skills later.
    """
    single_word_lookup: Dict[str, Skill] = {}
    multi_word_patterns: List[Tuple[Pattern, Skill]] = []

    for skill in skills:
        for alias in skill.aliases:
            if " " in alias:
                # Build a regex where any run of spaces/hyphens in the
                # alias can match a space OR a hyphen in the text, so
                # "object oriented programming" also matches text that
                # actually reads "object-oriented programming".
                words = alias.split(" ")
                pattern_body = r"[\s\-]+".join(re.escape(w) for w in words)
                pattern = re.compile(r"\b" + pattern_body + r"\b")
                multi_word_patterns.append((pattern, skill))
            else:
                if alias in single_word_lookup and single_word_lookup[alias] is not skill:
                    raise ValueError(
                        f"Alias '{alias}' is claimed by both "
                        f"'{single_word_lookup[alias].canonical_name}' and "
                        f"'{skill.canonical_name}' -- aliases must be unique "
                        f"across the whole dictionary."
                    )
                single_word_lookup[alias] = skill

    return single_word_lookup, multi_word_patterns


SINGLE_WORD_LOOKUP, MULTI_WORD_PATTERNS = _build_lookup_structures(SKILLS)
