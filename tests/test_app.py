import io
import os
import unittest
from unittest.mock import patch

import app as app_module
from exceptions import LLMRequestError, MissingAPIKeyError
from services.llm_service import LLMAnalysis

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def read_fixture(name: str) -> bytes:
    with open(os.path.join(FIXTURES, name), "rb") as f:
        return f.read()


FAKE_ANALYSIS = LLMAnalysis(
    strengths=["Strong Python background"],
    weaknesses=["Limited SQL exposure"],
    missing_skills_commentary="SQL is missing but the rest aligns well.",
    improvement_suggestions=["Add a project using SQL"],
    compatibility_summary="A solid match overall.",
)


class ResumeAnalyzerAppTestCase(unittest.TestCase):
    def setUp(self):
        app_module.app.testing = True
        self.client = app_module.app.test_client()


class TestIndexPage(ResumeAnalyzerAppTestCase):
    def test_index_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Resume Analyzer", response.data)


class TestAnalyzeSuccess(ResumeAnalyzerAppTestCase):
    @patch("app.generate_resume_analysis", return_value=FAKE_ANALYSIS)
    def test_full_workflow_with_valid_pdf(self, mock_llm):
        data = {
            "resume": (io.BytesIO(read_fixture("valid_resume.pdf")), "resume.pdf"),
            "job_description": "Required Skills: Python, Flask, SQL.",
        }
        response = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()

        self.assertIn("score", body)
        self.assertIn("final_score", body["score"])
        self.assertIsInstance(body["score"]["final_score"], int)
        self.assertIn("Python", body["score"]["matched_required_skills"])

        self.assertIsNotNone(body["analysis"])
        self.assertEqual(body["analysis"]["strengths"], ["Strong Python background"])
        self.assertIsNone(body["analysis_error"])
        mock_llm.assert_called_once()

    @patch("app.generate_resume_analysis", return_value=FAKE_ANALYSIS)
    def test_full_workflow_with_valid_docx(self, mock_llm):
        data = {
            "resume": (io.BytesIO(read_fixture("valid_resume.docx")), "resume.docx"),
            "job_description": "Required Skills: Python, Flask, SQL.",
        }
        response = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)


class TestAnalyzeValidationFailures(ResumeAnalyzerAppTestCase):
    def test_missing_resume_file(self):
        response = self.client.post(
            "/api/analyze",
            data={"job_description": "Required Skills: Python."},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("resume", response.get_json()["error"].lower())

    def test_missing_job_description(self):
        data = {"resume": (io.BytesIO(read_fixture("valid_resume.pdf")), "resume.pdf")}
        response = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertIn("job description", response.get_json()["error"].lower())

    def test_blank_job_description(self):
        data = {
            "resume": (io.BytesIO(read_fixture("valid_resume.pdf")), "resume.pdf"),
            "job_description": "   ",
        }
        response = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)

    def test_unsupported_file_type(self):
        data = {
            "resume": (io.BytesIO(read_fixture("invalid_type.txt")), "notes.txt"),
            "job_description": "Required Skills: Python.",
        }
        response = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)

    def test_corrupted_pdf(self):
        data = {
            "resume": (io.BytesIO(read_fixture("corrupted.pdf")), "resume.pdf"),
            "job_description": "Required Skills: Python.",
        }
        response = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)

    def test_empty_file(self):
        data = {
            "resume": (io.BytesIO(read_fixture("empty.pdf")), "resume.pdf"),
            "job_description": "Required Skills: Python.",
        }
        response = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)

    def test_near_empty_text_pdf(self):
        data = {
            "resume": (io.BytesIO(read_fixture("near_empty.pdf")), "resume.pdf"),
            "job_description": "Required Skills: Python.",
        }
        response = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)

    def test_oversized_file(self):
        data = {
            "resume": (io.BytesIO(read_fixture("oversized.pdf")), "resume.pdf"),
            "job_description": "Required Skills: Python.",
        }
        response = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)

    def test_request_larger_than_hard_cap_returns_413_json(self):
        # Bigger than app.config["MAX_CONTENT_LENGTH"] itself (not just
        # the resume-specific limit) -- Werkzeug must reject this before
        # our own validate_file() ever runs, and the response must still
        # be JSON the frontend can parse, not Werkzeug's default HTML.
        huge = b"0" * (app_module.app.config["MAX_CONTENT_LENGTH"] + 1024)
        data = {
            "resume": (io.BytesIO(huge), "resume.pdf"),
            "job_description": "Required Skills: Python.",
        }
        response = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 413)
        body = response.get_json()
        self.assertIsNotNone(body)
        self.assertIn("too large", body["error"].lower())


class TestAnalyzeLlmFailureIsGraceful(ResumeAnalyzerAppTestCase):
    @patch("app.generate_resume_analysis", side_effect=MissingAPIKeyError("no key configured"))
    def test_missing_api_key_does_not_fail_the_request(self, mock_llm):
        data = {
            "resume": (io.BytesIO(read_fixture("valid_resume.pdf")), "resume.pdf"),
            "job_description": "Required Skills: Python, Flask, SQL.",
        }
        response = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIsNotNone(body["score"])
        self.assertIsNone(body["analysis"])
        self.assertIn("unavailable", body["analysis_error"].lower())

    @patch("app.generate_resume_analysis", side_effect=LLMRequestError("network exploded"))
    def test_llm_request_failure_does_not_fail_the_request(self, mock_llm):
        data = {
            "resume": (io.BytesIO(read_fixture("valid_resume.pdf")), "resume.pdf"),
            "job_description": "Required Skills: Python, Flask, SQL.",
        }
        response = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIsNotNone(body["score"]["final_score"])
        self.assertIsNone(body["analysis"])
        self.assertIsNotNone(body["analysis_error"])

    def test_score_is_identical_regardless_of_llm_outcome(self):
        data_common = {
            "resume": (io.BytesIO(read_fixture("valid_resume.pdf")), "resume.pdf"),
            "job_description": "Required Skills: Python, Flask, SQL.",
        }
        with patch("app.generate_resume_analysis", return_value=FAKE_ANALYSIS):
            resp_ok = self.client.post("/api/analyze", data=dict(data_common), content_type="multipart/form-data")
        with patch("app.generate_resume_analysis", side_effect=LLMRequestError("boom")):
            resp_fail = self.client.post(
                "/api/analyze",
                data={
                    "resume": (io.BytesIO(read_fixture("valid_resume.pdf")), "resume.pdf"),
                    "job_description": "Required Skills: Python, Flask, SQL.",
                },
                content_type="multipart/form-data",
            )
        self.assertEqual(resp_ok.get_json()["score"], resp_fail.get_json()["score"])


if __name__ == "__main__":
    unittest.main()
