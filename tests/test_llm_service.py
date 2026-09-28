import unittest
from unittest.mock import patch

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
from services.llm_service import generate_resume_analysis
from services.scoring import ATSScoreResult

VALID_JSON_BODY = """{
  "strengths": ["Strong Python background", "Relevant project experience"],
  "weaknesses": ["No SQL experience mentioned"],
  "missing_skills_commentary": "The resume is missing SQL, which is listed as required.",
  "improvement_suggestions": ["Add a project demonstrating SQL usage"],
  "compatibility_summary": "This is a reasonably strong match for the role."
}"""


def make_ats_result(**overrides) -> ATSScoreResult:
    defaults = dict(
        final_score=75,
        required_skill_score=80,
        general_keyword_score=70,
        experience_project_score=90,
        education_score=100,
        matched_required_skills=["Python"],
        missing_required_skills=["SQL"],
        matched_general_skills=["Git"],
        missing_general_skills=["Docker"],
    )
    defaults.update(overrides)
    return ATSScoreResult(**defaults)


class FakeResponse:
    """Minimal stand-in for requests.Response, used only in tests."""
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text or (str(json_data) if json_data is not None else "")

    def json(self):
        if self._json_data is None:
            raise ValueError("No JSON body")
        return self._json_data


def anthropic_success_response(reply_text: str, status_code: int = 200) -> FakeResponse:
    return FakeResponse(
        status_code=status_code,
        json_data={"content": [{"type": "text", "text": reply_text}]},
    )


class TestSuccessfulResponse(unittest.TestCase):
    @patch.object(config, "LLM_API_KEY", "fake-test-key")
    def test_valid_response_parses_correctly(self):
        calls = []

        def fake_post(url, headers, json, timeout):
            calls.append((url, headers, json, timeout))
            return anthropic_success_response(VALID_JSON_BODY)

        result = generate_resume_analysis(
            "Python developer resume.", "Job requires Python and SQL.",
            make_ats_result(), http_post=fake_post,
        )

        self.assertEqual(result.strengths, ["Strong Python background", "Relevant project experience"])
        self.assertEqual(result.weaknesses, ["No SQL experience mentioned"])
        self.assertIn("SQL", result.missing_skills_commentary)
        self.assertEqual(result.improvement_suggestions, ["Add a project demonstrating SQL usage"])
        self.assertIn("strong match", result.compatibility_summary)
        self.assertEqual(len(calls), 1)

    @patch.object(config, "LLM_API_KEY", "fake-test-key")
    def test_markdown_fenced_json_is_parsed(self):
        fenced = "```json\n" + VALID_JSON_BODY + "\n```"

        def fake_post(url, headers, json, timeout):
            return anthropic_success_response(fenced)

        result = generate_resume_analysis(
            "resume text", "jd text", make_ats_result(), http_post=fake_post
        )
        self.assertEqual(result.strengths, ["Strong Python background", "Relevant project experience"])

    @patch.object(config, "LLM_API_KEY", "fake-test-key")
    def test_prompt_includes_deterministic_score_as_fixed_context(self):
        captured = {}

        def fake_post(url, headers, json, timeout):
            captured["prompt"] = json["messages"][0]["content"]
            return anthropic_success_response(VALID_JSON_BODY)

        ats = make_ats_result(final_score=42)
        generate_resume_analysis("resume", "jd", ats, http_post=fake_post)
        self.assertIn("42/100", captured["prompt"])
        normalized = " ".join(captured["prompt"].lower().split())
        self.assertIn("do not recalculate", normalized)


class TestMissingApiKey(unittest.TestCase):
    @patch.object(config, "LLM_API_KEY", None)
    def test_missing_api_key_raises_without_calling_http(self):
        calls = []

        def fake_post(*args, **kwargs):
            calls.append(1)
            return anthropic_success_response(VALID_JSON_BODY)

        with self.assertRaises(MissingAPIKeyError):
            generate_resume_analysis("resume", "jd", make_ats_result(), http_post=fake_post)

        self.assertEqual(calls, [], "HTTP call must not be attempted when the API key is missing")

    @patch.object(config, "LLM_API_KEY", "")
    def test_empty_string_api_key_also_raises(self):
        with self.assertRaises(MissingAPIKeyError):
            generate_resume_analysis(
                "resume", "jd", make_ats_result(),
                http_post=lambda *a, **k: anthropic_success_response(VALID_JSON_BODY),
            )


class TestProviderFailures(unittest.TestCase):
    @patch.object(config, "LLM_API_KEY", "fake-test-key")
    def test_network_failure_raises_llm_request_error(self):
        def fake_post(*args, **kwargs):
            raise requests.exceptions.ConnectionError("simulated network failure")

        with self.assertRaises(LLMRequestError):
            generate_resume_analysis("resume", "jd", make_ats_result(), http_post=fake_post)

    @patch.object(config, "LLM_API_KEY", "fake-test-key")
    def test_timeout_raises_llm_timeout_error(self):
        def fake_post(*args, **kwargs):
            raise requests.exceptions.Timeout("simulated timeout")

        with self.assertRaises(LLMTimeoutError):
            generate_resume_analysis("resume", "jd", make_ats_result(), http_post=fake_post)

    @patch.object(config, "LLM_API_KEY", "fake-test-key")
    def test_invalid_api_key_401_raises(self):
        def fake_post(*args, **kwargs):
            return FakeResponse(status_code=401, text="Unauthorized")

        with self.assertRaises(InvalidAPIKeyError):
            generate_resume_analysis("resume", "jd", make_ats_result(), http_post=fake_post)

    @patch.object(config, "LLM_API_KEY", "fake-test-key")
    def test_rate_limit_429_raises(self):
        def fake_post(*args, **kwargs):
            return FakeResponse(status_code=429, text="Too Many Requests")

        with self.assertRaises(LLMRateLimitError):
            generate_resume_analysis("resume", "jd", make_ats_result(), http_post=fake_post)

    @patch.object(config, "LLM_API_KEY", "fake-test-key")
    def test_generic_server_error_raises_llm_request_error(self):
        def fake_post(*args, **kwargs):
            return FakeResponse(status_code=500, text="Internal Server Error")

        with self.assertRaises(LLMRequestError):
            generate_resume_analysis("resume", "jd", make_ats_result(), http_post=fake_post)


class TestMalformedResponses(unittest.TestCase):
    @patch.object(config, "LLM_API_KEY", "fake-test-key")
    def test_non_json_http_body_raises(self):
        def fake_post(*args, **kwargs):
            return FakeResponse(status_code=200, json_data=None, text="not json at all")

        with self.assertRaises(MalformedLLMResponseError):
            generate_resume_analysis("resume", "jd", make_ats_result(), http_post=fake_post)

    @patch.object(config, "LLM_API_KEY", "fake-test-key")
    def test_missing_content_key_raises(self):
        def fake_post(*args, **kwargs):
            return FakeResponse(status_code=200, json_data={"unexpected": "shape"})

        with self.assertRaises(MalformedLLMResponseError):
            generate_resume_analysis("resume", "jd", make_ats_result(), http_post=fake_post)

    @patch.object(config, "LLM_API_KEY", "fake-test-key")
    def test_empty_text_content_raises(self):
        def fake_post(*args, **kwargs):
            return anthropic_success_response("")

        with self.assertRaises(MalformedLLMResponseError):
            generate_resume_analysis("resume", "jd", make_ats_result(), http_post=fake_post)

    @patch.object(config, "LLM_API_KEY", "fake-test-key")
    def test_reply_text_not_valid_json_raises(self):
        def fake_post(*args, **kwargs):
            return anthropic_success_response("I think this resume looks pretty good overall!")

        with self.assertRaises(MalformedLLMResponseError):
            generate_resume_analysis("resume", "jd", make_ats_result(), http_post=fake_post)

    @patch.object(config, "LLM_API_KEY", "fake-test-key")
    def test_reply_json_missing_required_keys_raises(self):
        def fake_post(*args, **kwargs):
            return anthropic_success_response('{"strengths": ["ok"]}')

        with self.assertRaises(MalformedLLMResponseError):
            generate_resume_analysis("resume", "jd", make_ats_result(), http_post=fake_post)

    @patch.object(config, "LLM_API_KEY", "fake-test-key")
    def test_reply_json_wrong_field_type_raises(self):
        bad = VALID_JSON_BODY.replace('"strengths": ["Strong Python background", "Relevant project experience"]', '"strengths": 12345')

        def fake_post(*args, **kwargs):
            return anthropic_success_response(bad)

        with self.assertRaises(MalformedLLMResponseError):
            generate_resume_analysis("resume", "jd", make_ats_result(), http_post=fake_post)


class TestScoreIndependence(unittest.TestCase):
    """The LLM must never be able to influence the numeric ATS score."""

    @patch.object(config, "LLM_API_KEY", "fake-test-key")
    def test_ats_result_object_is_not_mutated(self):
        ats = make_ats_result(final_score=55)
        original_score = ats.final_score

        def fake_post(*args, **kwargs):
            # Even an adversarial-looking reply that tries to state a
            # different score must not change ats.final_score.
            return anthropic_success_response(VALID_JSON_BODY.replace(
                "This is a reasonably strong match", "The real score should be 99, not what you think"
            ))

        generate_resume_analysis("resume", "jd", ats, http_post=fake_post)
        self.assertEqual(ats.final_score, original_score)

    def test_llm_analysis_has_no_numeric_score_field(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(
            __import__("services.llm_service", fromlist=["LLMAnalysis"]).LLMAnalysis
        )}
        self.assertNotIn("final_score", field_names)
        self.assertNotIn("score", field_names)
        self.assertNotIn("ats_score", field_names)


if __name__ == "__main__":
    unittest.main()
