import unittest

from services.skill_dictionary import (
    MULTI_WORD_PATTERNS,
    SINGLE_WORD_LOOKUP,
    SKILLS,
    Skill,
    _build_lookup_structures,
)


class TestSkillDictionaryIntegrity(unittest.TestCase):
    def test_every_skill_has_at_least_one_alias(self):
        for skill in SKILLS:
            self.assertGreaterEqual(
                len(skill.aliases), 1, f"{skill.canonical_name} has no aliases"
            )

    def test_every_alias_is_lowercase(self):
        for skill in SKILLS:
            for alias in skill.aliases:
                self.assertEqual(
                    alias, alias.lower(),
                    f"alias '{alias}' for {skill.canonical_name} is not lowercase",
                )

    def test_no_duplicate_canonical_names(self):
        names = [s.canonical_name for s in SKILLS]
        self.assertEqual(len(names), len(set(names)), "duplicate canonical skill names found")

    def test_single_word_and_multi_word_aliases_are_correctly_split(self):
        for alias, skill in SINGLE_WORD_LOOKUP.items():
            self.assertNotIn(" ", alias, f"'{alias}' has a space but landed in SINGLE_WORD_LOOKUP")

        for pattern, skill in MULTI_WORD_PATTERNS:
            # every multi-word pattern's skill must have at least one alias with a space
            self.assertTrue(
                any(" " in a for a in skill.aliases),
                f"{skill.canonical_name} has a multi-word pattern but no multi-word alias",
            )

    def test_conflicting_alias_raises_value_error(self):
        conflicting_skills = [
            Skill("Skill A", "Test", ("shared-alias",)),
            Skill("Skill B", "Test", ("shared-alias",)),
        ]
        with self.assertRaises(ValueError):
            _build_lookup_structures(conflicting_skills)

    def test_dotnet_is_registered(self):
        # regression check -- this was missed in the first pass and caught
        # by an exploratory test before the automated suite was written.
        self.assertIn(".net", SINGLE_WORD_LOOKUP)
        self.assertEqual(SINGLE_WORD_LOOKUP[".net"].canonical_name, ".NET")

    def test_expected_categories_present(self):
        categories = {s.category for s in SKILLS}
        expected = {
            "Programming", "Web/Backend", "Database", "AI/ML",
            "Data", "Core CS", "Tools",
        }
        self.assertTrue(expected.issubset(categories))


if __name__ == "__main__":
    unittest.main()
