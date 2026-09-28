"""
LLM integration service.

Calls the Anthropic Messages API directly via `requests` (no SDK) to
produce QUALITATIVE commentary on a resume/JD pair -- strengths,
weaknesses, missing-skill commentary, improvement suggestions, and a
short compatibility summary.

This module has NO influence over the numeric ATS score. It receives
the already-computed services.scoring.ATSScoreResult as read-only
context for the prompt and never recomputes or overrides any of its
fields -- the deterministic engine remains the sole source of truth
for the score, matched/missing skills, and component breakdown.

The HTTP call is made through an injectable `http_post` parameter
(defaulting to `requests.post`) specifically so tests can supply a
mock/fake instead of making real, paid network calls. This is the
"modularity" seam: swapping providers later means changing
_call_anthropic_api (or passing a different http_post), not touching
scoring.py or anything upstream of this module.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Callable, List

import requests

import config
from exceptions import (
    InvalidAPIKeyError,
    LLMRateLimitError,
    LLMRequestError,
    LLMTimeoutError,
    MalformedLLMResponseError,
    MissingAPIKeyError,
)
from services.scoring import ATSScoreResult

_REQUIRED_RESPONSE_KEYS = {
    "strengths",
    "weaknesses",
    "missing_skills_commentary",
    "improvement_suggestions",
    "compatibility_summary",
}


@dataclass
class LLMAnalysis:
    """Structured qualitative analysis, ready for the UI to render."""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    missing_skills_commentary: str = ""
    improvement_suggestions: List[str] = field(default_factory=list)
    compatibility_summary: str = ""


def _build_prompt(resume_text: str, job_text: str, ats_result: ATSScoreResult) -> str:
    """
    Build the analysis prompt. The prompt hands the LLM the deterministic
    results (score, matched/missing skills) as fixed CONTEXT and
    explicitly instructs it not to invent or restate a different score
    -- its job is qualitative commentary only.
    """
    return f"""You are assisting with resume feedback. You are given a resume, a job
description, and a deterministic compatibility analysis that has ALREADY
been calculated by a separate, non-negotiable scoring system. Do not
recalculate, restate as a different number, or contradict the score
below -- treat it as fixed ground truth. Your job is qualitative
commentary only.

DETERMINISTIC ANALYSIS (already computed, do not change):
- Compatibility score: {ats_result.final_score}/100
- Matched required skills: {", ".join(ats_result.matched_required_skills) or "None"}
- Missing required skills: {", ".join(ats_result.missing_required_skills) or "None"}
- Matched general skills: {", ".join(ats_result.matched_general_skills) or "None"}
- Missing general skills: {", ".join(ats_result.missing_general_skills) or "None"}

RESUME TEXT:
{resume_text}

JOB DESCRIPTION TEXT:
{job_text}

Respond with ONLY a single JSON object (no markdown fences, no prose
before or after) with exactly these keys:
{{
  "strengths": [list of short strings -- resume strengths relevant to this job],
  "weaknesses": [list of short strings -- resume weaknesses relevant to this job],
  "missing_skills_commentary": "one short paragraph discussing the missing skills listed above and how much they matter",
  "improvement_suggestions": [list of short, concrete, actionable suggestions],
  "compatibility_summary": "one short paragraph summarizing overall fit, consistent with the {ats_result.final_score}/100 score above"
}}"""


def _call_anthropic_api(prompt: str, http_post: Callable = requests.post) -> str:
    """
    Make the actual HTTP call and return the raw text content of the
    LLM's reply. Raises a specific LLMError subclass for every failure
    mode we handle -- callers never see a raw requests exception or an
    unguarded KeyError/JSONDecodeError.
    """
    if not config.LLM_API_KEY:
        raise MissingAPIKeyError(
            "No LLM API key is configured. Set LLM_API_KEY in your .env file."
        )

    headers = {
        "x-api-key": config.LLM_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": config.LLM_MODEL,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = http_post(
            config.LLM_API_URL, headers=headers, json=body, timeout=config.LLM_TIMEOUT_SECONDS
        )
    except requests.exceptions.Timeout as exc:
        raise LLMTimeoutError(
            f"LLM API call timed out after {config.LLM_TIMEOUT_SECONDS} seconds."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise LLMRequestError(f"LLM API request failed: {exc}") from exc

    if response.status_code == 401:
        raise InvalidAPIKeyError("LLM API rejected the configured API key (401 Unauthorized).")
    if response.status_code == 429:
        raise LLMRateLimitError("LLM API rate limit exceeded (429 Too Many Requests).")
    if response.status_code >= 400:
        raise LLMRequestError(
            f"LLM API returned an error status {response.status_code}: {response.text[:300]}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise MalformedLLMResponseError("LLM API response was not valid JSON.") from exc

    try:
        content_blocks = data["content"]
        text = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        )
    except (KeyError, TypeError) as exc:
        raise MalformedLLMResponseError(
            "LLM API response did not have the expected 'content' structure."
        ) from exc

    if not text.strip():
        raise MalformedLLMResponseError("LLM API returned an empty response.")

    return text


def _parse_llm_response(text: str) -> LLMAnalysis:
    """
    Parse the LLM's raw text reply into an LLMAnalysis. Tolerates the
    model wrapping its JSON in markdown code fences (common LLM habit)
    but otherwise requires valid JSON with every expected key.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise MalformedLLMResponseError(f"LLM response was not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise MalformedLLMResponseError("LLM response JSON was not an object.")

    missing_keys = _REQUIRED_RESPONSE_KEYS - parsed.keys()
    if missing_keys:
        raise MalformedLLMResponseError(
            f"LLM response JSON is missing expected field(s): {', '.join(sorted(missing_keys))}"
        )

    try:
        return LLMAnalysis(
            strengths=[str(s) for s in parsed["strengths"]],
            weaknesses=[str(s) for s in parsed["weaknesses"]],
            missing_skills_commentary=str(parsed["missing_skills_commentary"]),
            improvement_suggestions=[str(s) for s in parsed["improvement_suggestions"]],
            compatibility_summary=str(parsed["compatibility_summary"]),
        )
    except TypeError as exc:
        # e.g. "strengths" was a string/int instead of a list
        raise MalformedLLMResponseError(f"LLM response JSON had an unexpected field type: {exc}") from exc


def generate_resume_analysis(
    resume_text: str,
    job_text: str,
    ats_result: ATSScoreResult,
    http_post: Callable = requests.post,
) -> LLMAnalysis:
    """
    Produce qualitative LLM analysis for a resume/JD pair, given the
    already-computed deterministic ATSScoreResult. Never modifies
    ats_result and never returns a score of its own -- callers combine
    ats_result (numeric, deterministic) with this return value
    (qualitative, LLM-generated) when building the final report.

    Raises services.llm_service's LLMError subclasses (see exceptions.py)
    on any failure -- callers should catch these and degrade gracefully
    (e.g. show the deterministic score with a note that AI commentary
    is unavailable) rather than crash.
    """
    prompt = _build_prompt(resume_text, job_text, ats_result)
    raw_text = _call_anthropic_api(prompt, http_post=http_post)
    return _parse_llm_response(raw_text)
