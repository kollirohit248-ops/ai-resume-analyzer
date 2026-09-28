import unittest

from services.section_detection import is_heading_line, sectionize


class TestIsHeadingLine(unittest.TestCase):
    def test_detects_all_caps_projects(self):
        self.assertTrue(is_heading_line("PROJECTS"))

    def test_detects_title_case_projects(self):
        self.assertTrue(is_heading_line("Projects"))

    def test_detects_work_experience(self):
        self.assertTrue(is_heading_line("WORK EXPERIENCE"))

    def test_detects_professional_experience(self):
        self.assertTrue(is_heading_line("Professional Experience"))

    def test_detects_internship_experience(self):
        self.assertTrue(is_heading_line("INTERNSHIP EXPERIENCE"))

    def test_detects_heading_with_trailing_colon(self):
        self.assertTrue(is_heading_line("Projects:"))

    def test_detects_heading_with_surrounding_whitespace(self):
        self.assertTrue(is_heading_line("   Projects   "))

    def test_detects_heading_with_decoration(self):
        self.assertTrue(is_heading_line("=== Projects ==="))
        self.assertTrue(is_heading_line("* Projects *"))

    def test_detects_mixed_case(self):
        self.assertTrue(is_heading_line("pRoJeCtS"))

    def test_does_not_detect_sentence_mentioning_projects(self):
        self.assertFalse(is_heading_line("I worked on several projects using Python."))

    def test_does_not_detect_sentence_starting_with_this_project(self):
        self.assertFalse(is_heading_line("This project uses SQL."))

    def test_does_not_detect_sentence_mentioning_experience(self):
        self.assertFalse(is_heading_line("Experience with Python and Java."))

    def test_does_not_detect_arbitrary_line(self):
        self.assertFalse(is_heading_line("John Doe -- Software Engineer"))

    def test_empty_line_is_not_a_heading(self):
        self.assertFalse(is_heading_line(""))
        self.assertFalse(is_heading_line("   "))

    def test_detects_technical_skills_as_non_practical_heading(self):
        # is_heading_line just detects headings in general -- whether
        # it's "practical" is a separate concern tested via sectionize.
        self.assertTrue(is_heading_line("TECHNICAL SKILLS"))


class TestSectionize(unittest.TestCase):
    def test_empty_text_returns_no_sections(self):
        result = sectionize("")
        self.assertEqual(result.sections, [])
        self.assertEqual(result.preamble_text, "")
        self.assertEqual(result.practical_text, "")
        self.assertFalse(result.has_practical_sections)

    def test_text_with_no_recognized_headings_is_all_preamble(self):
        text = "John Doe\nSoftware Engineer\nBuilt things with Python."
        result = sectionize(text)
        self.assertEqual(result.sections, [])
        self.assertIn("Software Engineer", result.preamble_text)
        self.assertFalse(result.has_practical_sections)

    def test_projects_section_is_practical(self):
        text = "PROJECTS\nBuilt a Python application using SQL."
        result = sectionize(text)
        self.assertEqual(len(result.sections), 1)
        self.assertTrue(result.sections[0].is_practical)
        self.assertIn("Built a Python application using SQL.", result.practical_text)

    def test_technical_skills_section_is_not_practical(self):
        text = "TECHNICAL SKILLS\nPython\nJava\nSQL"
        result = sectionize(text)
        self.assertEqual(len(result.sections), 1)
        self.assertFalse(result.sections[0].is_practical)
        self.assertEqual(result.practical_text, "")

    def test_multiple_sections_split_correctly(self):
        text = (
            "TECHNICAL SKILLS\n"
            "Python\nJava\nSQL\n"
            "\n"
            "PROJECTS\n"
            "Built a Python application using SQL.\n"
            "\n"
            "EDUCATION\n"
            "B.Tech in Computer Science\n"
        )
        result = sectionize(text)
        names = [s.name for s in result.sections]
        self.assertEqual(names, ["technical skills", "projects", "education"])

        skills_section = result.sections[0]
        projects_section = result.sections[1]
        education_section = result.sections[2]

        self.assertFalse(skills_section.is_practical)
        self.assertTrue(projects_section.is_practical)
        self.assertFalse(education_section.is_practical)

        self.assertIn("Built a Python application using SQL.", result.practical_text)
        self.assertNotIn("B.Tech", result.practical_text)
        self.assertNotIn("Python\nJava\nSQL", result.practical_text)

    def test_preamble_before_first_heading_is_not_practical(self):
        text = "John Doe\nSoftware Engineer\n\nPROJECTS\nBuilt a Python app."
        result = sectionize(text)
        self.assertIn("John Doe", result.preamble_text)
        self.assertNotIn("John Doe", result.practical_text)

    def test_case_insensitive_and_whitespace_tolerant_headings(self):
        text = "   projects   \nBuilt an app.\n\nWork Experience\nInterned somewhere."
        result = sectionize(text)
        names = [s.name for s in result.sections]
        self.assertEqual(names, ["projects", "work experience"])
        self.assertTrue(all(s.is_practical for s in result.sections))

    def test_sentence_mentioning_project_does_not_start_a_section(self):
        text = "I worked on several projects using Python.\nThis project uses SQL."
        result = sectionize(text)
        self.assertEqual(result.sections, [])
        self.assertIn("I worked on several projects using Python.", result.preamble_text)


if __name__ == "__main__":
    unittest.main()
