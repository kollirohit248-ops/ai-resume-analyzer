import math
import unittest

from exceptions import EmptyJobDescriptionTextError, EmptyResumeTextError
from services.scoring import compute_ats_score

STRONG_RESUME = """John Doe

TECHNICAL SKILLS
Python, Java, SQL, AI/ML

PROJECTS
Built a Python AI Resume Analyzer project using Machine Learning and SQL.
Built a Java Student Management System.

EDUCATION
B.Tech in Computer Science Engineering, 2027
"""

STRONG_JD = """Trainee Software Engineer

Required Skills:
Python, Java, SQL, AI/ML

Education:
Bachelor's degree in Computer Science required.
"""


class TestPerfectMatch(unittest.TestCase):
    def setUp(self):
        self.result = compute_ats_score(STRONG_RESUME, STRONG_JD)

    def test_final_score_is_100(self):
        self.assertEqual(self.result.final_score, 100)

    def test_required_skill_score_is_100(self):
        self.assertEqual(self.result.required_skill_score, 100)

    def test_no_missing_required_skills(self):
        self.assertEqual(self.result.missing_required_skills, [])

    def test_education_matches(self):
        self.assertTrue(self.result.education_match)

    def test_experience_score_is_100_due_to_projects_section(self):
        self.assertEqual(self.result.experience_project_score, 100)

    def test_not_low_confidence(self):
        self.assertFalse(self.result.low_confidence)


class TestPartialMatch(unittest.TestCase):
    def setUp(self):
        resume = "Skills: Python, SQL.\n\nPROJECTS\nBuilt a Python data pipeline using SQL."
        jd = "Required Skills: Python, Java, SQL, AI/ML. Nice to have: Docker, Git."
        self.result = compute_ats_score(resume, jd)

    def test_some_required_matched_some_missing(self):
        self.assertEqual(self.result.matched_required_skills, ["Python", "SQL"])
        self.assertIn("Java", self.result.missing_required_skills)

    def test_required_skill_score_reflects_partial_ratio(self):
        # 2 matched out of 5 detected required skills (AI/ML expands to
        # two distinct dictionary skills: Artificial Intelligence, ML)
        self.assertEqual(self.result.required_skill_score, 40)

    def test_general_keyword_score_reflects_zero_general_matches(self):
        # JD's general skills are Docker and Git; resume has neither.
        self.assertEqual(self.result.general_keyword_score, 0)

    def test_score_is_between_0_and_100(self):
        self.assertTrue(0 <= self.result.final_score <= 100)

    def test_education_not_applicable_no_requirement_in_jd(self):
        self.assertIsNone(self.result.education_score)
        self.assertNotIn("D", self.result.applicable_components)


class TestZeroSkillMatch(unittest.TestCase):
    def setUp(self):
        resume = "I enjoy painting and hiking."
        jd = "Required Skills: Python, Java. Nice to have: Docker."
        self.result = compute_ats_score(resume, jd)

    def test_final_score_is_zero(self):
        self.assertEqual(self.result.final_score, 0)

    def test_required_score_is_real_zero_not_none(self):
        # Applicable (JD has required skills) but resume matches none.
        self.assertEqual(self.result.required_skill_score, 0)
        self.assertIn("R", self.result.applicable_components)

    def test_general_score_is_real_zero_not_none(self):
        self.assertEqual(self.result.general_keyword_score, 0)
        self.assertIn("K", self.result.applicable_components)

    def test_experience_score_is_zero_no_matched_skills(self):
        self.assertEqual(self.result.experience_project_score, 0)


class TestRequiredSkillWeighting(unittest.TestCase):
    def test_no_required_skills_in_jd_redistributes_r_weight(self):
        resume = "I know Docker."
        jd = "Nice to have: Docker, Git."
        result = compute_ats_score(resume, jd)
        self.assertIsNone(result.required_skill_score)
        self.assertNotIn("R", result.applicable_components)
        # Only K applicable -> gets 100% of the weight.
        self.assertAlmostEqual(result.redistributed_weights["K"], 1.0)

    def test_all_required_skills_matched(self):
        resume = "Python, Java, SQL expert."
        jd = "Required Skills: Python, Java, SQL."
        result = compute_ats_score(resume, jd)
        self.assertEqual(result.required_skill_score, 100)
        self.assertEqual(result.missing_required_skills, [])

    def test_zero_required_skills_matched_but_applicable(self):
        resume = "I know nothing relevant."
        jd = "Required Skills: Python, Java."
        result = compute_ats_score(resume, jd)
        self.assertEqual(result.required_skill_score, 0)
        self.assertIn("R", result.applicable_components)


class TestGeneralKeywordWeighting(unittest.TestCase):
    def test_no_general_skills_in_jd_redistributes_k_weight(self):
        resume = "Python expert."
        jd = "Required Skills: Python."
        result = compute_ats_score(resume, jd)
        self.assertIsNone(result.general_keyword_score)
        self.assertNotIn("K", result.applicable_components)

    def test_common_words_do_not_count_as_general_skills(self):
        # "the", "and", "software", "company", "work", "using" are not
        # in the skill dictionary at all, so they can never contribute
        # to K -- only dictionary technology/domain terms can.
        resume = "I work using the software and do great work for the company."
        jd = "Required Skills: Python. The software company does great work using modern tools."
        result = compute_ats_score(resume, jd)
        # No dictionary skills at all beyond Python (which is required,
        # not general) -- K must be not-applicable, not inflated by
        # the shared common words.
        self.assertIsNone(result.general_keyword_score)


class TestExperienceProjectRelevance(unittest.TestCase):
    def test_fresher_projects_count_as_practical_evidence(self):
        resume = (
            "TECHNICAL SKILLS\nPython, SQL\n\n"
            "PROJECTS\nBuilt a Python AI Resume Analyzer project using SQL."
        )
        jd = "Required Skills: Python, SQL."
        result = compute_ats_score(resume, jd)
        self.assertEqual(result.experience_project_score, 100)
        self.assertEqual(
            sorted(result.experience_relevance.demonstrated_in_practice), ["Python", "SQL"]
        )

    def test_skills_only_in_technical_skills_section_not_demonstrated(self):
        resume = "TECHNICAL SKILLS\nPython, SQL"
        jd = "Required Skills: Python, SQL."
        result = compute_ats_score(resume, jd)
        # Matched skills exist, but no practical section was detected
        # at all -- treated as NOT APPLICABLE, not as a 0, since we
        # can't tell "no experience" apart from "unparsed format".
        self.assertIsNone(result.experience_project_score)
        self.assertNotIn("E", result.applicable_components)

    def test_partial_practical_demonstration(self):
        resume = (
            "TECHNICAL SKILLS\nPython, Java, SQL\n\n"
            "PROJECTS\nBuilt a Python application using SQL."
        )
        jd = "Required Skills: Python, Java, SQL."
        result = compute_ats_score(resume, jd)
        # Python and SQL appear in Projects; Java only in the skills list.
        self.assertIn("Python", result.experience_relevance.demonstrated_in_practice)
        self.assertIn("SQL", result.experience_relevance.demonstrated_in_practice)
        self.assertIn("Java", result.experience_relevance.not_demonstrated)
        self.assertAlmostEqual(result.experience_project_score, round(2 / 3 * 100))

    def test_example_b_from_spec_java_only_demonstrated(self):
        resume = (
            "TECHNICAL SKILLS\nPython\nJava\nSQL\n\n"
            "PROJECTS\nBuilt a Java application.\n"
        )
        jd = "Required Skills: Python, Java, SQL."
        result = compute_ats_score(resume, jd)
        self.assertEqual(result.experience_relevance.demonstrated_in_practice, ["Java"])
        self.assertEqual(
            sorted(result.experience_relevance.not_demonstrated), ["Python", "SQL"]
        )
        self.assertAlmostEqual(result.experience_project_score, round(1 / 3 * 100))

    def test_no_matched_skills_gives_real_zero_not_none(self):
        resume = "TECHNICAL SKILLS\nPainting\n\nPROJECTS\nPainted a mural."
        jd = "Required Skills: Python."
        result = compute_ats_score(resume, jd)
        self.assertEqual(result.experience_project_score, 0)
        self.assertIn("E", result.applicable_components)


class TestEducationRelevance(unittest.TestCase):
    def test_education_match(self):
        resume = "EDUCATION\nB.Tech in Computer Science"
        jd = "Required Skills: Python. Education: Bachelor's degree in Computer Science required."
        result = compute_ats_score(resume, jd)
        self.assertEqual(result.education_score, 100)
        self.assertTrue(result.education_match)

    def test_education_mismatch_resume_has_none(self):
        resume = "TECHNICAL SKILLS\nPython"
        jd = "Required Skills: Python. Education: Bachelor's degree in Computer Science required."
        result = compute_ats_score(resume, jd)
        self.assertEqual(result.education_score, 0)
        self.assertFalse(result.education_match)

    def test_no_education_requirement_in_jd_is_not_applicable(self):
        resume = "EDUCATION\nB.Tech in Computer Science"
        jd = "Required Skills: Python."
        result = compute_ats_score(resume, jd)
        self.assertIsNone(result.education_score)
        self.assertIsNone(result.education_match)
        self.assertNotIn("D", result.applicable_components)

    def test_random_education_words_do_not_fabricate_a_match(self):
        # Resume mentions "graduate" in an unrelated sense; JD requires
        # a specific degree phrase the resume never states.
        resume = "TECHNICAL SKILLS\nPython\n\nSUMMARY\nGraduate teaching assistant."
        jd = "Required Skills: Python. Education: BCA or MCA required."
        result = compute_ats_score(resume, jd)
        self.assertFalse(result.education_match)


class TestWeightRedistribution(unittest.TestCase):
    def test_weights_sum_to_one_when_all_applicable(self):
        result = compute_ats_score(STRONG_RESUME, STRONG_JD)
        total = sum(result.redistributed_weights.values())
        self.assertAlmostEqual(total, 1.0)

    def test_weights_sum_to_one_with_only_two_applicable(self):
        resume = "Python expert."
        jd = "Required Skills: Python."
        result = compute_ats_score(resume, jd)
        total = sum(result.redistributed_weights.values())
        self.assertAlmostEqual(total, 1.0)

    def test_missing_component_weight_is_never_silently_zero(self):
        # D is not applicable (no education requirement) -- its 0.10
        # weight must be redistributed among R/K/E, not dropped.
        resume = (
            "TECHNICAL SKILLS\nPython, Docker\n\nPROJECTS\nBuilt a Python tool using Docker."
        )
        jd = "Required Skills: Python. Nice to have: Docker."
        result = compute_ats_score(resume, jd)
        self.assertNotIn("D", result.redistributed_weights)
        self.assertAlmostEqual(sum(result.redistributed_weights.values()), 1.0)


class TestScoreBoundariesAndRounding(unittest.TestCase):
    def test_score_never_below_zero(self):
        resume = "I enjoy hiking."
        jd = "Required Skills: Python, Java, SQL."
        result = compute_ats_score(resume, jd)
        self.assertGreaterEqual(result.final_score, 0)

    def test_score_never_above_100(self):
        result = compute_ats_score(STRONG_RESUME, STRONG_JD)
        self.assertLessEqual(result.final_score, 100)

    def test_score_is_integer(self):
        result = compute_ats_score(STRONG_RESUME, STRONG_JD)
        self.assertIsInstance(result.final_score, int)

    def test_rounding_two_thirds(self):
        resume = "I know Python and Java."
        jd = "Required Skills: Python, Java, SQL."
        result = compute_ats_score(resume, jd)
        self.assertEqual(result.required_skill_score, round(2 / 3 * 100))

    def test_no_nan_or_infinite_values(self):
        result = compute_ats_score(STRONG_RESUME, STRONG_JD)
        self.assertFalse(math.isnan(result.final_score))
        self.assertFalse(math.isinf(result.final_score))


class TestDuplicatesAliasesCaseAndPunctuation(unittest.TestCase):
    def test_duplicate_skills_in_resume_do_not_inflate_score(self):
        resume = "Python, python, PYTHON, Python programming -- Java."
        jd = "Required Skills: Python, Java."
        result = compute_ats_score(resume, jd)
        self.assertEqual(result.matched_required_skills, ["Java", "Python"])
        self.assertEqual(result.final_score, 100)

    def test_alias_match_dsa_oop(self):
        resume = "Strong DSA and OOP fundamentals."
        jd = "Required Skills: Data Structures and Algorithms, Object-Oriented Programming."
        result = compute_ats_score(resume, jd)
        self.assertEqual(result.final_score, 100)

    def test_case_insensitivity(self):
        resume = "PYTHON developer."
        jd = "Required Skills: python."
        result = compute_ats_score(resume, jd)
        self.assertEqual(result.final_score, 100)

    def test_punctuation_differences_do_not_break_matching(self):
        resume = "Skills: C++, C#, .NET, Node.js!!!"
        jd = "Required Skills: C++, C#, .NET, Node.js."
        result = compute_ats_score(resume, jd)
        self.assertEqual(result.missing_required_skills, [])


class TestEdgeCases(unittest.TestCase):
    def test_empty_resume_raises(self):
        with self.assertRaises(EmptyResumeTextError):
            compute_ats_score("", "Required Skills: Python.")

    def test_whitespace_only_resume_raises(self):
        with self.assertRaises(EmptyResumeTextError):
            compute_ats_score("   \n\t  ", "Required Skills: Python.")

    def test_empty_jd_raises(self):
        with self.assertRaises(EmptyJobDescriptionTextError):
            compute_ats_score("I know Python.", "")

    def test_whitespace_only_jd_raises(self):
        with self.assertRaises(EmptyJobDescriptionTextError):
            compute_ats_score("I know Python.", "   \n\t  ")

    def test_resume_with_no_extractable_skills(self):
        result = compute_ats_score("I like long walks on the beach.", "Required Skills: Python.")
        self.assertEqual(result.matched_required_skills, [])
        self.assertEqual(result.required_skill_score, 0)

    def test_jd_with_no_extractable_skills_is_low_confidence(self):
        result = compute_ats_score(
            "I know Python.", "We want a kind, punctual, enthusiastic team member."
        )
        self.assertTrue(result.low_confidence)
        self.assertEqual(result.final_score, 0)
        self.assertEqual(result.applicable_components, [])

    def test_resume_with_no_projects_section(self):
        resume = "TECHNICAL SKILLS\nPython"
        jd = "Required Skills: Python."
        result = compute_ats_score(resume, jd)
        self.assertIsNone(result.experience_project_score)

    def test_no_division_by_zero_anywhere(self):
        # Both sides essentially empty of dictionary content -- must not
        # raise ZeroDivisionError anywhere in the pipeline.
        try:
            result = compute_ats_score("asdkj alskdj", "qwe rty uio")
        except ZeroDivisionError:
            self.fail("compute_ats_score raised ZeroDivisionError")
        self.assertEqual(result.final_score, 0)

    def test_very_short_text_does_not_crash(self):
        result = compute_ats_score("a", "b")
        self.assertEqual(result.final_score, 0)
        self.assertTrue(result.low_confidence)


class TestLowConfidence(unittest.TestCase):
    def test_low_confidence_flag_set_when_jd_has_no_skills(self):
        result = compute_ats_score(
            "Experienced Python developer.",
            "We are a friendly, fast-growing team looking for a great teammate.",
        )
        self.assertTrue(result.low_confidence)
        self.assertIn("LOW CONFIDENCE", result.explanation)

    def test_low_confidence_not_set_for_normal_jd(self):
        result = compute_ats_score(STRONG_RESUME, STRONG_JD)
        self.assertFalse(result.low_confidence)


class TestExplanationOutput(unittest.TestCase):
    def test_explanation_contains_final_score(self):
        result = compute_ats_score(STRONG_RESUME, STRONG_JD)
        self.assertIn(f"ATS-style Compatibility Score: {result.final_score}", result.explanation)

    def test_explanation_lists_matched_and_missing_required_skills(self):
        resume = "Skills: Python, SQL."
        jd = "Required Skills: Python, Java, SQL."
        result = compute_ats_score(resume, jd)
        self.assertIn("Python", result.explanation)
        self.assertIn("Java", result.explanation)

    def test_explanation_shows_na_for_inapplicable_component(self):
        resume = "Python expert."
        jd = "Required Skills: Python."
        result = compute_ats_score(resume, jd)
        self.assertIn("N/A", result.explanation)

    def test_explanation_is_not_hardcoded_and_reflects_actual_data(self):
        result_a = compute_ats_score("Python expert.", "Required Skills: Python.")
        result_b = compute_ats_score("I enjoy hiking.", "Required Skills: Python, Java.")
        self.assertNotEqual(result_a.explanation, result_b.explanation)


if __name__ == "__main__":
    unittest.main()
