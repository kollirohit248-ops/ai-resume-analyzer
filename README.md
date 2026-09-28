# Resume Analyzer

A resume-analysis web app that compares an uploaded resume (PDF or DOCX)
against a pasted job description, produces a **deterministic,
rule-based compatibility score**, and adds **LLM-generated qualitative
feedback** (strengths, weaknesses, improvement suggestions) on top of
it — without ever letting the LLM touch the numeric score.

## Features

- Upload a resume as PDF or DOCX; paste any job description as text
- Deterministic skill extraction with alias handling (e.g. "DSA" ↔
  "Data Structures and Algorithms") and false-positive-safe matching
  (e.g. "C" never matches inside "C++" or "JavaScript")
- Job description analysis: detects required vs. general/nice-to-have
  skills (via context phrases like "Required Skills:", "must have"),
  education requirements, experience requirements
- A deterministic, weighted ATS-style compatibility score (0–100) with
  a full component breakdown and human-readable explanation
- LLM-generated qualitative analysis (strengths, weaknesses,
  improvement suggestions, compatibility summary) layered on top of
  the deterministic result
- Graceful degradation: if the LLM call fails for any reason (missing
  key, network error, timeout, malformed response), the app still
  returns the full deterministic score with a clear notice that AI
  commentary is unavailable
- A simple one-page web UI: upload, paste, analyze, see results

## What this is *not*

This is an **application-specific deterministic compatibility
heuristic**, not an official or industry-standard ATS algorithm, and
not a substitute for human judgment in hiring. See **Limitations**
below.

## Architecture / pipeline

```
Resume upload (PDF/DOCX)              Job description (text)
        |                                       |
services/file_validation.py                     |
        |                                       |
services/resume_parser.py                       |
        |                                       |
        +-------------------+-------------------+
                             |
              services/text_preprocessing.py
       (unicode/whitespace normalization, tokenization --
        preserves technical terms like C++, C#, .NET, Node.js)
                             |
              services/skill_extraction.py
    (skill dictionary matching, required-vs-general classification,
     education/experience/responsibility detection, resume<->JD comparison)
                             |
              services/section_detection.py
   (line-preserving pass over the RAW resume text, used only to check
    whether a matched skill appears in an Experience/Projects section)
                             |
                 services/scoring.py
     (deterministic ATS-style score: R/K/E/D components, weighted,
      with redistribution when a component doesn't apply)
                             |
                 services/llm_service.py
   (qualitative-only commentary; receives the deterministic result as
    fixed context; cannot change the numeric score)
                             |
                          app.py
              (Flask route, JSON response, error handling)
                             |
                 templates/ + static/
                    (single-page web UI)
```

Every module above the Flask layer is independently testable and has
no dependency on Flask or on the LLM (`scoring.py` never imports
`llm_service.py`).

## Technology stack

| Layer | Choice | Why |
|---|---|---|
| Backend | Flask | Minimal, no framework magic to explain away in an interview |
| Frontend | HTML/CSS/vanilla JS | No build step or npm dependency chain |
| Resume parsing | `pypdf`, `python-docx` | Standard libraries for PDF/DOCX text extraction |
| Skill/NLP logic | Regex + a hand-built skill dictionary (stdlib only) | Transparent and fully explainable; no NLP library needed for keyword/phrase matching at this scope |
| LLM integration | Direct HTTP call via `requests` to the Anthropic Messages API | No SDK dependency; the HTTP call is injectable so tests never hit the real API |
| Secrets | `.env` via `python-dotenv` | Standard practice, keeps keys out of source control |
| Testing | `unittest` (stdlib) | No extra install needed |

No database, authentication, background workers, or deployment
tooling are part of this project.

## Setup

```bash
git clone <repository-url>
cd resume-analyzer
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set your own values:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `LLM_API_KEY` | Yes, for AI commentary | *(none)* | Anthropic API key. Without it, the app still works and returns the deterministic score, with a notice that AI analysis is unavailable. |
| `LLM_MODEL` | No | `claude-sonnet-4-6` | Model name passed to the Messages API |
| `LLM_TIMEOUT_SECONDS` | No | `30` | HTTP timeout for the LLM call |
| `MAX_RESUME_SIZE_MB` | No | `5` | Max accepted resume upload size |

## Running the app

```bash
python app.py
```

Then open `http://127.0.0.1:5000/` in a browser, upload a resume
(PDF/DOCX), paste a job description, and click Analyze.

## API

### `POST /api/analyze`

`multipart/form-data` with fields:
- `resume` — the PDF/DOCX file
- `job_description` — plain text

Returns JSON:

```json
{
  "score": {
    "final_score": 82,
    "required_skill_score": 90,
    "general_keyword_score": 76,
    "experience_project_score": 80,
    "education_score": 100,
    "matched_required_skills": ["Python", "Java"],
    "missing_required_skills": [],
    "matched_general_skills": ["SQL", "Git"],
    "missing_general_skills": ["Docker"],
    "education_match": true,
    "applicable_components": ["R", "K", "E", "D"],
    "low_confidence": false,
    "explanation": "ATS-style Compatibility Score: 82\n..."
  },
  "analysis": {
    "strengths": ["..."],
    "weaknesses": ["..."],
    "missing_skills_commentary": "...",
    "improvement_suggestions": ["..."],
    "compatibility_summary": "..."
  },
  "analysis_error": null
}
```

`analysis` is `null` (with `analysis_error` explaining why) if the LLM
call failed for any reason — `score` is always present and complete
regardless.

Error responses (`400`/`413`/`500`) return `{"error": "..."}`.

## Deterministic score vs. LLM analysis

These are deliberately separate systems:

- **`score` (deterministic)** is computed entirely by
  `services/scoring.py` from structured data produced by earlier
  modules — skill matching, section detection, regex-based education
  detection. Same input always produces the same score. No AI model is
  involved anywhere in this calculation.
- **`analysis` (LLM-generated)** is qualitative commentary only. The
  deterministic score and matched/missing skills are handed to the LLM
  as fixed context with an explicit instruction not to recalculate or
  contradict them — the LLM has no code path by which it could change
  `final_score` or any component score.

The ATS score formula:

```
score = 100 x (0.50*R + 0.25*K + 0.15*E + 0.10*D)
```

- **R** — required-skill match ratio
- **K** — general/non-required keyword overlap ratio
- **E** — fraction of matched skills that appear in a detected
  Experience/Projects/Internship section of the resume, rather than
  only in a Skills list
- **D** — whether the resume's detected education overlaps with the
  JD's detected education requirement

If a component doesn't apply to a given resume/JD pair (e.g. the JD
states no education requirement), its weight is redistributed
proportionally among the applicable components rather than counted as
zero. If the JD has no detectable skill information at all, the result
is flagged `low_confidence` rather than presented as a confident score.

## Testing

```bash
python3 -m unittest discover -s tests -v
```

261 tests across file validation/parsing, text preprocessing, the
skill dictionary and matcher, section detection, the scoring engine,
the LLM service (HTTP calls fully mocked — no real API calls made
during tests), and the Flask API.

## Limitations

- **Not an official ATS**: this is this application's own heuristic,
  not a real-world ATS vendor's algorithm.
- **Required-skill detection** is a forward-window heuristic around
  cue phrases ("Required Skills:", "must have", etc.) — a JD phrased
  skill-then-cue ("Python, Java — required") won't be caught, and a JD
  with none of the recognized cue phrases will have zero skills
  flagged as required.
- **Experience/project relevance (E)** only checks whether a skill's
  name literally appears inside a recognized resume section
  (Experience/Projects/Internship/Employment, detected by exact
  heading-line matching) — it cannot judge whether the skill was
  meaningfully used there.
- **Education matching (D)** is simple phrase overlap between detected
  JD and resume education mentions, not a judgment of degree
  prestige or field closeness.
- **Skill dictionary** covers common software-engineering/AI/data
  terms but is not exhaustive; extending it means adding entries to
  `services/skill_dictionary.py`.
- **LLM commentary quality** depends on the underlying model and
  isn't independently fact-checked beyond basic response-shape
  validation.
- **Synchronous API**: a slow LLM call delays the HTTP response rather
  than running in the background.
- No authentication, database, or persistence — each analysis is
  stateless.
