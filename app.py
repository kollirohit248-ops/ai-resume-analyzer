"""
Flask application.

Wires together the existing, already-tested services into one HTTP
workflow: upload -> validate -> extract -> deterministic score (Feature
4) -> LLM qualitative analysis (Feature 5) -> JSON response.

This module contains NO scoring or analysis logic of its own -- it only
calls into services/*.py and shapes their results into JSON. The LLM
can never replace or modify the deterministic score: score_result is
computed first, independently, and llm analysis is generated
afterwards purely for qualitative commentary (see services/llm_service.py).
"""
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

import config
from exceptions import (
    FileValidationError,
    LLMError,
    ScoringError,
    TextExtractionError,
)
from services.file_validation import validate_file
from services.llm_service import generate_resume_analysis
from services.resume_parser import extract_resume_text
from services.scoring import ATSScoreResult, compute_ats_score

app = Flask(__name__)

# Hard cap on the whole request body, enforced by Werkzeug BEFORE any
# bytes are read into memory -- this is what actually stops a huge
# upload from being buffered at all. It's intentionally larger than
# config.MAX_RESUME_SIZE_BYTES (the real, user-facing resume size
# limit that validate_file() enforces with a friendly message) so a
# resume that's merely over the resume-specific limit still reaches
# validate_file() and gets a clear 400, rather than being cut off here
# with a generic 413. This is a coarse safety net against truly
# oversized payloads (hundreds of MB+), not the primary size policy.
app.config["MAX_CONTENT_LENGTH"] = max(20 * 1024 * 1024, config.MAX_RESUME_SIZE_BYTES * 4)


@app.errorhandler(RequestEntityTooLarge)
def handle_request_too_large(_exc):
    return jsonify({"error": "The uploaded file or request is too large."}), 413


def _score_to_dict(result: ATSScoreResult) -> dict:
    return {
        "final_score": result.final_score,
        "required_skill_score": result.required_skill_score,
        "general_keyword_score": result.general_keyword_score,
        "experience_project_score": result.experience_project_score,
        "education_score": result.education_score,
        "matched_required_skills": result.matched_required_skills,
        "missing_required_skills": result.missing_required_skills,
        "matched_general_skills": result.matched_general_skills,
        "missing_general_skills": result.missing_general_skills,
        "education_match": result.education_match,
        "applicable_components": result.applicable_components,
        "low_confidence": result.low_confidence,
        "explanation": result.explanation,
    }


def _analysis_to_dict(analysis) -> dict:
    return {
        "strengths": analysis.strengths,
        "weaknesses": analysis.weaknesses,
        "missing_skills_commentary": analysis.missing_skills_commentary,
        "improvement_suggestions": analysis.improvement_suggestions,
        "compatibility_summary": analysis.compatibility_summary,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    resume_file = request.files.get("resume")
    job_description = (request.form.get("job_description") or "").strip()

    if resume_file is None or resume_file.filename == "":
        return jsonify({"error": "No resume file was uploaded."}), 400
    if not job_description:
        return jsonify({"error": "Job description text is required."}), 400

    file_bytes = resume_file.read()

    try:
        extension = validate_file(resume_file.filename, file_bytes)
        resume_text = extract_resume_text(extension, file_bytes)
    except (FileValidationError, TextExtractionError) as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        score_result = compute_ats_score(resume_text, job_description)
    except ScoringError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Unexpected scoring failure")
        return jsonify({"error": "Something went wrong while analyzing the resume."}), 500

    response = {
        "score": _score_to_dict(score_result),
        "analysis": None,
        "analysis_error": None,
    }

    try:
        llm_analysis = generate_resume_analysis(resume_text, job_description, score_result)
        response["analysis"] = _analysis_to_dict(llm_analysis)
    except LLMError as exc:
        response["analysis_error"] = f"AI analysis is currently unavailable: {exc}"
    except Exception:
        app.logger.exception("Unexpected LLM failure")
        response["analysis_error"] = "AI analysis is currently unavailable."

    return jsonify(response), 200


if __name__ == "__main__":
    app.run(debug=True)
