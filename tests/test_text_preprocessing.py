import unittest

from services.text_preprocessing import (
    PreprocessedText,
    clean_whitespace,
    normalize_for_matching,
    normalize_punctuation,
    normalize_unicode,
    preprocess,
    tokenize,
)


class TestNormalizeUnicode(unittest.TestCase):
    def test_smart_quotes_become_straight_quotes(self):
        text = "\u2018Python\u2019 and \u201cJava\u201d"
        self.assertEqual(normalize_unicode(text), "'Python' and \"Java\"")

    def test_em_and_en_dash_become_hyphen(self):
        text = "2023\u20132024 \u2014 full-time"
        self.assertEqual(normalize_unicode(text), "2023-2024 - full-time")

    def test_ellipsis_char_expands_to_three_dots(self):
        self.assertEqual(normalize_unicode("skills\u2026 many"), "skills... many")

    def test_non_breaking_space_becomes_regular_space(self):
        self.assertEqual(normalize_unicode("Python\u00a0Developer"), "Python Developer")

    def test_zero_width_space_is_removed(self):
        self.assertEqual(normalize_unicode("Java\u200bScript"), "JavaScript")

    def test_empty_input_returns_empty(self):
        self.assertEqual(normalize_unicode(""), "")


class TestCleanWhitespace(unittest.TestCase):
    def test_multiple_spaces_collapse(self):
        self.assertEqual(clean_whitespace("Python    Developer"), "Python Developer")

    def test_tabs_collapse(self):
        self.assertEqual(clean_whitespace("Python\t\tDeveloper"), "Python Developer")

    def test_newlines_collapse(self):
        self.assertEqual(clean_whitespace("Python\n\nDeveloper"), "Python Developer")

    def test_mixed_whitespace_collapses(self):
        self.assertEqual(clean_whitespace("Python \t\n Developer"), "Python Developer")

    def test_leading_trailing_whitespace_stripped(self):
        self.assertEqual(clean_whitespace("   Python Developer   "), "Python Developer")

    def test_empty_input_returns_empty(self):
        self.assertEqual(clean_whitespace(""), "")

    def test_whitespace_only_input_returns_empty(self):
        self.assertEqual(clean_whitespace("   \n\t  "), "")


class TestNormalizePunctuation(unittest.TestCase):
    def test_bullet_characters_removed(self):
        self.assertEqual(normalize_punctuation("• Python\n• Java"), "Python Java")

    def test_repeated_exclamation_collapses(self):
        self.assertEqual(normalize_punctuation("Expert!!!"), "Expert!")

    def test_repeated_periods_collapse_but_not_single_dot(self):
        self.assertEqual(normalize_punctuation("Great....job"), "Great.job")

    def test_single_dot_in_dotnet_is_untouched(self):
        self.assertEqual(normalize_punctuation(".NET"), ".NET")

    def test_single_dot_in_nodejs_is_untouched(self):
        self.assertEqual(normalize_punctuation("Node.js"), "Node.js")

    def test_plus_plus_is_untouched(self):
        self.assertEqual(normalize_punctuation("C++"), "C++")

    def test_hash_is_untouched(self):
        self.assertEqual(normalize_punctuation("C#"), "C#")

    def test_empty_input_returns_empty(self):
        self.assertEqual(normalize_punctuation(""), "")


class TestNormalizeForMatching(unittest.TestCase):
    def test_lowercases_text(self):
        self.assertEqual(normalize_for_matching("Python DEVELOPER"), "python developer")

    def test_preserves_special_characters_while_lowercasing(self):
        self.assertEqual(normalize_for_matching("C++ and C#"), "c++ and c#")

    def test_empty_input_returns_empty(self):
        self.assertEqual(normalize_for_matching(""), "")


class TestTokenize(unittest.TestCase):
    def test_plain_words(self):
        self.assertEqual(tokenize("python developer"), ["python", "developer"])

    def test_cplusplus_preserved_as_single_token(self):
        self.assertEqual(
            tokenize("skills: c++ and python"), ["skills", "c++", "and", "python"]
        )

    def test_csharp_preserved_as_single_token(self):
        self.assertEqual(
            tokenize("experience in c# programming"),
            ["experience", "in", "c#", "programming"],
        )

    def test_dotnet_preserved_as_single_token(self):
        self.assertEqual(tokenize("built apps using .net"), ["built", "apps", "using", ".net"])

    def test_nodejs_preserved_as_single_token(self):
        self.assertEqual(tokenize("backend in node.js"), ["backend", "in", "node.js"])

    def test_reactjs_preserved_as_single_token(self):
        self.assertEqual(tokenize("frontend with react.js"), ["frontend", "with", "react.js"])

    def test_sql_preserved(self):
        self.assertEqual(tokenize("strong sql skills"), ["strong", "sql", "skills"])

    def test_nlp_preserved(self):
        self.assertEqual(tokenize("worked on nlp projects"), ["worked", "on", "nlp", "projects"])

    def test_does_not_split_cplusplus_into_c(self):
        tokens = tokenize("c++")
        self.assertIn("c++", tokens)
        self.assertNotEqual(tokens, ["c"])

    def test_does_not_split_nodejs_into_node_and_js(self):
        tokens = tokenize("node.js")
        self.assertEqual(tokens, ["node.js"])
        self.assertNotIn("node", tokens)

    def test_multiword_phrase_preserved_as_adjacent_tokens(self):
        # Multi-word tech terms aren't merged into one token, but nothing
        # is lost -- "machine" and "learning" appear as adjacent tokens
        # in original order, which Feature 4's phrase matcher relies on.
        self.assertEqual(
            tokenize("experience in machine learning and generative ai"),
            ["experience", "in", "machine", "learning", "and", "generative", "ai"],
        )

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(tokenize(""), [])


class TestFullPreprocessPipeline(unittest.TestCase):
    def test_raw_text_is_preserved_unchanged(self):
        raw = "  C++   Developer\n\twith  \u2018Node.js\u2019 experience  "
        result = preprocess(raw)
        self.assertEqual(result.raw_text, raw)

    def test_returns_preprocessed_text_dataclass(self):
        result = preprocess("Python developer")
        self.assertIsInstance(result, PreprocessedText)

    def test_normal_text_flows_through_all_stages(self):
        result = preprocess("Python Developer with SQL skills")
        self.assertEqual(result.clean_text, "Python Developer with SQL skills")
        self.assertEqual(result.normalized_text, "python developer with sql skills")
        self.assertEqual(
            result.tokens, ["python", "developer", "with", "sql", "skills"]
        )

    def test_mixed_case_skills_normalized_consistently(self):
        result = preprocess("PYTHON, Python, python, PyThOn")
        self.assertEqual(result.tokens, ["python"] * 4)

    def test_technical_terms_preserved_end_to_end(self):
        raw = "Skilled in C++, C#, .NET, Node.js, React.js, SQL, and NLP."
        result = preprocess(raw)
        for term in ["c++", "c#", ".net", "node.js", "react.js", "sql", "nlp"]:
            self.assertIn(term, result.tokens, f"expected token {term!r} in {result.tokens}")

    def test_multiword_technical_phrases_survive_as_substrings(self):
        raw = "Experience with Machine Learning, Generative AI, and Large Language Models."
        result = preprocess(raw)
        self.assertIn("machine learning", result.normalized_text)
        self.assertIn("generative ai", result.normalized_text)
        self.assertIn("large language models", result.normalized_text)

    def test_empty_input_does_not_crash(self):
        result = preprocess("")
        self.assertEqual(result.raw_text, "")
        self.assertEqual(result.clean_text, "")
        self.assertEqual(result.normalized_text, "")
        self.assertEqual(result.tokens, [])

    def test_whitespace_only_input_does_not_crash(self):
        result = preprocess("   \n\t   ")
        self.assertEqual(result.clean_text, "")
        self.assertEqual(result.tokens, [])

    def test_unicode_heavy_resume_snippet(self):
        raw = "\u201cSoftware Engineer\u201d \u2013 5\u00a0years\u2026 C++ \u2018expert\u2019"
        result = preprocess(raw)
        self.assertIn("c++", result.tokens)
        self.assertNotIn("\u2018", result.clean_text)
        self.assertNotIn("\u00a0", result.clean_text)


if __name__ == "__main__":
    unittest.main()
