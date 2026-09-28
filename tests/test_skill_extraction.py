import os
import unittest

from services.skill_extraction import (
    SkillMatch,
    analyze_job_description,
    compare_resume_to_job,
    detect_education_requirements,
    detect_experience_requirements,
    detect_required_skill_names,
    detect_responsibility_terms,
    find_skills_in_raw_text,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def read_fixture_text(name: str) -> str:
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return f.read()


def names(skill_matches):
    return sorted(s.canonical_name for s in skill_matches)


# ============================================================================
# FEATURE 3A -- skill detection
# ============================================================================


class TestExactSkillMatching(unittest.TestCase):
    def test_single_skill(self):
        self.assertEqual(names(find_skills_in_raw_text("I know Python.")), ["Python"])

    def test_several_skills(self):
        result = names(find_skills_in_raw_text("I use Python, Flask, and SQL daily."))
        self.assertEqual(result, ["Flask", "Python", "SQL"])

    def test_case_insensitive_matching(self):
        self.assertEqual(names(find_skills_in_raw_text("PYTHON, Python, python, PyThOn")), ["Python"])

    def test_no_skills_found_returns_empty_list(self):
        self.assertEqual(find_skills_in_raw_text("I enjoy hiking and painting."), [])

    def test_empty_text_returns_empty_list(self):
        self.assertEqual(find_skills_in_raw_text(""), [])

    def test_whitespace_only_text_returns_empty_list(self):
        self.assertEqual(find_skills_in_raw_text("   \n\t  "), [])


class TestMultiWordSkillMatching(unittest.TestCase):
    def test_machine_learning(self):
        self.assertIn("Machine Learning", names(find_skills_in_raw_text("Experience with machine learning.")))

    def test_generative_ai(self):
        self.assertIn("Generative AI", names(find_skills_in_raw_text("Built a generative AI chatbot.")))

    def test_object_oriented_programming_with_space(self):
        self.assertIn(
            "Object-Oriented Programming",
            names(find_skills_in_raw_text("Strong object oriented programming skills.")),
        )

    def test_object_oriented_programming_with_hyphen(self):
        self.assertIn(
            "Object-Oriented Programming",
            names(find_skills_in_raw_text("Strong object-oriented programming skills.")),
        )

    def test_rest_api(self):
        self.assertIn("REST API", names(find_skills_in_raw_text("Built several REST APIs.")))

    def test_natural_language_processing_full_phrase(self):
        self.assertIn(
            "Natural Language Processing",
            names(find_skills_in_raw_text("Focus on natural language processing.")),
        )


class TestAliases(unittest.TestCase):
    def test_dsa_alias(self):
        self.assertIn("Data Structures and Algorithms", names(find_skills_in_raw_text("Strong DSA skills.")))

    def test_dsa_full_phrase(self):
        self.assertIn(
            "Data Structures and Algorithms",
            names(find_skills_in_raw_text("Studied data structures and algorithms.")),
        )

    def test_oop_alias(self):
        self.assertIn("Object-Oriented Programming", names(find_skills_in_raw_text("Good OOP fundamentals.")))

    def test_llm_alias(self):
        self.assertIn("Large Language Models", names(find_skills_in_raw_text("Worked with LLMs.")))

    def test_llm_full_phrase(self):
        self.assertIn(
            "Large Language Models",
            names(find_skills_in_raw_text("Experience with large language models.")),
        )

    def test_nlp_alias(self):
        self.assertIn("Natural Language Processing", names(find_skills_in_raw_text("NLP is my focus area.")))

    def test_postgres_alias_maps_to_postgresql(self):
        self.assertIn("PostgreSQL", names(find_skills_in_raw_text("I use Postgres for everything.")))

    def test_postgres_and_postgresql_dont_duplicate(self):
        result = names(find_skills_in_raw_text("Worked with Postgres and PostgreSQL interchangeably."))
        self.assertEqual(result.count("PostgreSQL"), 1)


class TestDuplicateSkillHandling(unittest.TestCase):
    def test_python_mentioned_twice_reported_once(self):
        result = find_skills_in_raw_text("Python and Python programming are my strengths.")
        self.assertEqual(names(result), ["Python"])

    def test_full_example_from_spec(self):
        # From the feature spec: resume mentions "Python programming",
        # "object oriented programming", and "SQL" -- must NOT produce
        # both "OOP" and "Object-Oriented Programming" as separate
        # entries, nor "Python" and "Python Programming" separately.
        result = names(
            find_skills_in_raw_text(
                "Python programming, object oriented programming, SQL"
            )
        )
        self.assertEqual(result, ["Object-Oriented Programming", "Python", "SQL"])


class TestFalsePositivePrevention(unittest.TestCase):
    def test_c_does_not_match_inside_other_words(self):
        result = find_skills_in_raw_text("I love cats and coffee in the morning.")
        self.assertNotIn("C", names(result))

    def test_java_does_not_match_inside_javascript(self):
        result = names(find_skills_in_raw_text("I write JavaScript daily."))
        self.assertIn("JavaScript", result)
        self.assertNotIn("Java", result)

    def test_sql_does_not_match_inside_sqlite(self):
        result = names(find_skills_in_raw_text("We use SQLite for local storage."))
        self.assertIn("SQLite", result)
        self.assertNotIn("SQL", result)

    def test_rest_api_does_not_match_inside_arrest(self):
        result = find_skills_in_raw_text("The suspect will be placed under arrest.")
        self.assertNotIn("REST API", names(result))

    def test_cv_abbreviation_does_not_match_computer_vision(self):
        # Resumes routinely say "CV" to mean "curriculum vitae" -- must
        # not be treated as "Computer Vision".
        result = find_skills_in_raw_text("Please find my CV attached.")
        self.assertNotIn("Computer Vision", names(result))

    def test_computer_vision_full_phrase_still_matches(self):
        result = find_skills_in_raw_text("Worked on computer vision projects.")
        self.assertIn("Computer Vision", names(result))


class TestCLanguageFamilyDistinction(unittest.TestCase):
    def test_c_cpp_csharp_all_distinct(self):
        result = names(find_skills_in_raw_text("Skilled in C, C++, and C#."))
        self.assertEqual(result, ["C", "C#", "C++"])

    def test_only_cpp_mentioned(self):
        result = names(find_skills_in_raw_text("5 years of C++ experience."))
        self.assertEqual(result, ["C++"])
        self.assertNotIn("C", result)

    def test_only_csharp_mentioned(self):
        result = names(find_skills_in_raw_text("Built apps in C#."))
        self.assertEqual(result, ["C#"])
        self.assertNotIn("C", result)


class TestSpecificTechTerms(unittest.TestCase):
    def test_dotnet(self):
        self.assertIn(".NET", names(find_skills_in_raw_text("Experience with .NET.")))

    def test_nodejs(self):
        self.assertIn("Node.js", names(find_skills_in_raw_text("Backend built with Node.js.")))

    def test_nodejs_no_dot_variant(self):
        self.assertIn("Node.js", names(find_skills_in_raw_text("Backend built with NodeJS.")))

    def test_sql(self):
        self.assertIn("SQL", names(find_skills_in_raw_text("Strong SQL skills.")))

    def test_dotnet_nodejs_reactjs_together(self):
        result = names(find_skills_in_raw_text("Experience with .NET, Node.js, and React.js."))
        self.assertEqual(result, [".NET", "Node.js", "React"])


class TestRequiredSkillDetection(unittest.TestCase):
    def test_required_skills_heading(self):
        jd = "Required Skills: Python, Java, SQL. Nice to have: Docker."
        required = names(analyze_job_description(jd).required_skills)
        self.assertIn("Python", required)
        self.assertIn("Java", required)
        self.assertIn("SQL", required)

    def test_nice_to_have_skills_not_required(self):
        jd = "Required Skills: Python. Nice to have: Docker and Git."
        analysis = analyze_job_description(jd)
        self.assertNotIn("Docker", names(analysis.required_skills))
        self.assertNotIn("Git", names(analysis.required_skills))
        self.assertIn("Docker", names(analysis.general_skills))
        self.assertIn("Git", names(analysis.general_skills))

    def test_proficiency_in_phrase(self):
        jd = "Proficiency in Python is required for this role."
        required = names(analyze_job_description(jd).required_skills)
        self.assertIn("Python", required)

    def test_must_have_phrase(self):
        jd = "Must have experience with SQL."
        required = names(analyze_job_description(jd).required_skills)
        self.assertIn("SQL", required)

    def test_skill_with_no_context_cue_is_not_required(self):
        # A JD that just lists a technology in passing, with none of the
        # recognized required-context phrases anywhere, should NOT mark
        # it required -- we don't assume every mention is a hard
        # requirement (see detect_required_skill_names docstring).
        jd = "Our team occasionally uses Docker for local development."
        analysis = analyze_job_description(jd)
        self.assertNotIn("Docker", names(analysis.required_skills))
        self.assertIn("Docker", names(analysis.general_skills))

    def test_detect_required_skill_names_directly(self):
        from services.text_preprocessing import preprocess
        normalized = preprocess("Required Skills: Python and Java.").normalized_text
        required_names = detect_required_skill_names(normalized)
        self.assertIn("Python", required_names)
        self.assertIn("Java", required_names)


class TestEducationDetection(unittest.TestCase):
    def test_btech(self):
        self.assertIn("b.tech", detect_education_requirements("candidates must have a b.tech degree"))

    def test_bachelors_degree(self):
        result = detect_education_requirements("a bachelor's degree in computer science is required")
        self.assertIn("bachelor's degree", result)
        self.assertIn("computer science", result)

    def test_no_education_mentioned(self):
        self.assertEqual(detect_education_requirements("we are hiring a software engineer"), [])


class TestExperienceDetection(unittest.TestCase):
    def test_years_of_experience(self):
        result = detect_experience_requirements("looking for 2+ years of experience")
        self.assertTrue(any("years" in r and "experience" in r for r in result))

    def test_fresher(self):
        result = detect_experience_requirements("freshers are encouraged to apply")
        self.assertTrue(any("fresher" in r for r in result))

    def test_no_experience_mentioned(self):
        self.assertEqual(detect_experience_requirements("we value clean code and teamwork"), [])


class TestResponsibilityTermDetection(unittest.TestCase):
    def test_problem_solving(self):
        self.assertIn("problem solving", detect_responsibility_terms("strong problem solving skills required"))

    def test_communication(self):
        self.assertIn("communication", detect_responsibility_terms("excellent communication with the team"))

    def test_building_features(self):
        self.assertIn(
            "building features", detect_responsibility_terms("you will focus on building features from scratch")
        )


class TestJobDescriptionAnalysisEdgeCases(unittest.TestCase):
    def test_empty_jd(self):
        analysis = analyze_job_description("")
        self.assertEqual(analysis.detected_skills, [])
        self.assertEqual(analysis.required_skills, [])
        self.assertEqual(analysis.general_skills, [])
        self.assertEqual(analysis.education_requirements, [])
        self.assertEqual(analysis.experience_requirements, [])
        self.assertEqual(analysis.responsibility_terms, [])

    def test_jd_with_no_recognized_skills(self):
        analysis = analyze_job_description(
            "We are looking for a kind, punctual, and enthusiastic team member."
        )
        self.assertEqual(analysis.detected_skills, [])
        self.assertEqual(analysis.required_skills, [])


class TestRealisticRecruitCrmJdFixture(unittest.TestCase):
    """
    Uses a realistic Trainee-Software-Engineer-style JD (modeled on the
    Recruit CRM posting) as a test fixture only -- nothing about this
    specific JD is hardcoded into the application logic being tested.
    """

    def setUp(self):
        self.jd_text = read_fixture_text("recruit_crm_jd.txt")
        self.analysis = analyze_job_description(self.jd_text)

    def test_java_and_python_detected_as_required(self):
        required = names(self.analysis.required_skills)
        self.assertIn("Java", required)
        self.assertIn("Python", required)

    def test_dsa_and_oop_detected_as_required(self):
        required = names(self.analysis.required_skills)
        self.assertIn("Data Structures and Algorithms", required)
        self.assertIn("Object-Oriented Programming", required)

    def test_sql_detected_as_required(self):
        self.assertIn("SQL", names(self.analysis.required_skills))

    def test_nice_to_have_skills_are_general_not_required(self):
        required = names(self.analysis.required_skills)
        general = names(self.analysis.general_skills)
        for skill in ["Git", "GitHub", "REST API"]:
            self.assertIn(skill, general, f"{skill} should be general, not required")
            self.assertNotIn(skill, required, f"{skill} should not be required")

    def test_education_requirement_detected(self):
        self.assertIn("b.tech", self.analysis.education_requirements)

    def test_experience_requirement_detected(self):
        self.assertTrue(any("fresher" in e for e in self.analysis.experience_requirements))

    def test_responsibility_terms_detected(self):
        terms = self.analysis.responsibility_terms
        self.assertIn("problem solving", terms)
        self.assertIn("analytical thinking", terms)
        self.assertIn("communication", terms)
        self.assertIn("building features", terms)


# ============================================================================
# FEATURE 3B -- resume / job description comparison
# ============================================================================


class TestCompareResumeToJobBasics(unittest.TestCase):
    def test_strong_match(self):
        resume = "Experienced in Python, Java, SQL, and Object-Oriented Programming."
        jd = "Required Skills: Python, Java, SQL, OOP."
        result = compare_resume_to_job(resume, jd)
        self.assertEqual(names(result.missing_required_skills), [])
        self.assertEqual(
            names(result.matched_required_skills),
            ["Java", "Object-Oriented Programming", "Python", "SQL"],
        )

    def test_partial_match(self):
        resume = "Experienced in Python and SQL."
        jd = "Required Skills: Python, Java, SQL."
        result = compare_resume_to_job(resume, jd)
        self.assertEqual(names(result.matched_required_skills), ["Python", "SQL"])
        self.assertEqual(names(result.missing_required_skills), ["Java"])

    def test_no_match(self):
        resume = "Experienced in painting and hiking."
        jd = "Required Skills: Python, Java, SQL."
        result = compare_resume_to_job(resume, jd)
        self.assertEqual(names(result.matched_required_skills), [])
        self.assertEqual(names(result.missing_required_skills), ["Java", "Python", "SQL"])

    def test_alias_match(self):
        resume = "Strong DSA and OOP background, familiar with LLMs."
        jd = "Required Skills: Data Structures and Algorithms, Object-Oriented Programming, Large Language Models."
        result = compare_resume_to_job(resume, jd)
        self.assertEqual(
            names(result.matched_required_skills),
            ["Data Structures and Algorithms", "Large Language Models", "Object-Oriented Programming"],
        )
        self.assertEqual(names(result.missing_required_skills), [])

    def test_case_insensitive_match(self):
        resume = "PYTHON developer."
        jd = "Required Skills: python."
        result = compare_resume_to_job(resume, jd)
        self.assertEqual(names(result.matched_required_skills), ["Python"])

    def test_duplicate_skills_in_resume_not_duplicated_in_result(self):
        resume = "Python, python programming, PYTHON development -- all Python, all the time."
        jd = "Required Skills: Python."
        result = compare_resume_to_job(resume, jd)
        self.assertEqual(result.resume_skills.count(SkillMatch("Python", "Programming")), 1)
        self.assertEqual(names(result.matched_required_skills), ["Python"])

    def test_c_cpp_csharp_distinction_in_comparison(self):
        resume = "Skilled in C++ only."
        jd = "Required Skills: C, C++, C#."
        result = compare_resume_to_job(resume, jd)
        self.assertEqual(names(result.matched_required_skills), ["C++"])
        self.assertEqual(names(result.missing_required_skills), ["C", "C#"])

    def test_multiword_skills_in_comparison(self):
        resume = "Background in machine learning and natural language processing."
        jd = "Required Skills: Machine Learning, Natural Language Processing, Computer Vision."
        result = compare_resume_to_job(resume, jd)
        self.assertEqual(
            names(result.matched_required_skills),
            ["Machine Learning", "Natural Language Processing"],
        )
        self.assertEqual(names(result.missing_required_skills), ["Computer Vision"])

    def test_missing_required_skills_reported_correctly(self):
        resume = "I know HTML and CSS."
        jd = "Required Skills: Python, HTML, CSS, SQL."
        result = compare_resume_to_job(resume, jd)
        self.assertEqual(names(result.matched_required_skills), ["CSS", "HTML"])
        self.assertEqual(names(result.missing_required_skills), ["Python", "SQL"])

    def test_general_skills_matched_and_missing(self):
        resume = "I know Docker."
        jd = "Required Skills: Python. Nice to have: Docker, Git."
        result = compare_resume_to_job(resume, jd)
        self.assertEqual(names(result.matched_general_skills), ["Docker"])
        self.assertEqual(names(result.missing_general_skills), ["Git"])


class TestCompareResumeToJobEdgeCases(unittest.TestCase):
    def test_empty_resume(self):
        result = compare_resume_to_job("", "Required Skills: Python, Java.")
        self.assertEqual(result.resume_skills, [])
        self.assertEqual(names(result.matched_required_skills), [])
        self.assertEqual(names(result.missing_required_skills), ["Java", "Python"])

    def test_empty_jd(self):
        result = compare_resume_to_job("Experienced Python developer.", "")
        self.assertEqual(result.job_skills, [])
        self.assertEqual(result.required_skills, [])
        self.assertEqual(result.matched_required_skills, [])
        self.assertEqual(result.missing_required_skills, [])

    def test_jd_with_no_recognized_skills(self):
        result = compare_resume_to_job(
            "Experienced Python developer.",
            "We want a kind, punctual, and enthusiastic team member.",
        )
        self.assertEqual(result.job_skills, [])
        self.assertEqual(result.matched_required_skills, [])
        self.assertEqual(result.missing_required_skills, [])

    def test_both_empty(self):
        result = compare_resume_to_job("", "")
        self.assertEqual(result.resume_skills, [])
        self.assertEqual(result.job_skills, [])


class TestCompareResumeToJobWithRecruitCrmFixture(unittest.TestCase):
    def setUp(self):
        self.jd_text = read_fixture_text("recruit_crm_jd.txt")

    def test_strong_candidate(self):
        resume = (
            "Final-year B.Tech Computer Science student. Skilled in Python, Java, "
            "SQL, Data Structures and Algorithms, and Object-Oriented Programming. "
            "Built REST APIs and used Git/GitHub for version control. "
            "Some exposure to Machine Learning and Generative AI."
        )
        result = compare_resume_to_job(resume, self.jd_text)
        self.assertEqual(names(result.missing_required_skills), [])

    def test_weak_candidate_missing_most_required_skills(self):
        resume = "I know HTML and CSS and enjoy graphic design."
        result = compare_resume_to_job(resume, self.jd_text)
        required_names = names(result.required_skills)
        self.assertEqual(names(result.matched_required_skills), [])
        self.assertEqual(names(result.missing_required_skills), required_names)


if __name__ == "__main__":
    unittest.main()
