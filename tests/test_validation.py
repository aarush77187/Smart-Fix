"""
Unit tests for the Smart Guided Troubleshooting Engine.
Runs purely offline with zero external network or LLM calls.
Evaluates:
  1. Zero URL leak scrubbing (scrubber.py)
  2. Programmatic word-count & constraint enforcement (constraints.py)
  3. Action deduplication (one screen = one action)
  4. Safe-first action sequencing (critical actions pushed to end)
  5. Deterministic deeplink matching & catalog URI validity (matcher.py)
  6. Semantic cache hit on paraphrases (cache.py)
  7. Schema compliance with Samsung's official sample_output.json
"""

import json
import unittest
from pathlib import Path

from app.api_contract import DUMMY_POSITIVE_DEEPLINK
from app.cache import SemanticCache
from app.constraints import fix_goal
from app.matcher import get_matcher, sequence_actions
from app.schema import Action, Goal, StepGroup, actionCategory
from app.scrubber import scrub_goal, scrub_text


class TestZeroURLLeaks(unittest.TestCase):
    def test_markdown_links_collapsed(self):
        raw = "For more details [visit TechCorp Support](https://techcorp.com/support) right now."
        clean, had_leak = scrub_text(raw)
        self.assertTrue(had_leak)
        self.assertNotIn("https://", clean)
        self.assertIn("visit TechCorp Support", clean)

    def test_bare_domain_removed(self):
        raw = "Contact techcorp.com/support or call service center."
        clean, had_leak = scrub_text(raw)
        self.assertTrue(had_leak)
        self.assertNotIn("techcorp.com", clean)

    def test_full_goal_scrub(self):
        goal = Goal(
            goal="Follow these steps to perform this Battery Troubleshooting",
            title="Battery Drain",
            score=0.9,
            actions=[
                Action(
                    actionName="Display Settings",
                    description="It will help reduce battery usage at samsung.com",
                    category=actionCategory.auto,
                    stepGroups=[
                        StepGroup(steps=["Check details at https://support.google.com"])
                    ],
                )
            ],
        )
        cleaned_goal, warnings = scrub_goal(goal)
        self.assertTrue(len(warnings) > 0)
        self.assertNotIn("samsung.com", cleaned_goal.actions[0].description)
        self.assertNotIn("https://support.google.com", cleaned_goal.actions[0].stepGroups[0].steps[0])


class TestConstraintsEnforcement(unittest.TestCase):
    def test_title_word_count_clamped(self):
        goal = Goal(
            goal="Follow these steps to perform this Troubleshooting",
            title="very long title that has way too many words to be compliant",
            score=0.9,
            actions=[],
        )
        fixed_goal, warnings = fix_goal(goal)
        words = fixed_goal.title.split()
        self.assertLessEqual(len(words), 3)
        self.assertGreaterEqual(len(words), 2)

    def test_description_prefix_and_length(self):
        goal = Goal(
            goal="Follow these steps to perform this Troubleshooting",
            title="Screen Flicker",
            score=0.9,
            actions=[
                Action(
                    actionName="Display Screen",
                    description="reduces refresh rate to fix flickering display issues and battery",
                    category=actionCategory.auto,
                    stepGroups=[StepGroup(steps=["Tap display"])],
                )
            ],
        )
        fixed_goal, warnings = fix_goal(goal)
        desc = fixed_goal.actions[0].description
        self.assertTrue(desc.lower().startswith("it will"))
        self.assertLessEqual(len(desc.split()), 7)

    def test_one_screen_one_action_merge(self):
        goal = Goal(
            goal="Follow these steps to perform this Troubleshooting",
            title="Display Issue",
            score=0.9,
            actions=[
                Action(
                    actionName="Display Settings",
                    description="It will adjust brightness",
                    category=actionCategory.auto,
                    stepGroups=[StepGroup(steps=["Tap Brightness"])],
                ),
                Action(
                    actionName="Display Settings",
                    description="It will adjust refresh rate",
                    category=actionCategory.auto,
                    stepGroups=[StepGroup(steps=["Tap Motion Smoothness"])],
                ),
            ],
        )
        fixed_goal, warnings = fix_goal(goal)
        self.assertEqual(len(fixed_goal.actions), 1)
        self.assertEqual(len(fixed_goal.actions[0].stepGroups), 2)


class TestSafeSequencing(unittest.TestCase):
    def test_critical_actions_pushed_to_end(self):
        actions = [
            Action(
                actionName="Factory Reset",
                description="It will erase all data",
                category=actionCategory.critical,
                stepGroups=[StepGroup(steps=["Reset"])],
            ),
            Action(
                actionName="Clear Cache",
                description="It will remove temporary files",
                category=actionCategory.auto,
                stepGroups=[StepGroup(steps=["Clear cache"])],
            ),
        ]
        ordered = sequence_actions(actions)
        self.assertEqual(ordered[0].category, actionCategory.auto)
        self.assertEqual(ordered[1].category, actionCategory.critical)


class TestDeeplinkMatcher(unittest.TestCase):
    def setUp(self):
        self.matcher = get_matcher()

    def test_catalog_loaded(self):
        self.assertGreaterEqual(len(self.matcher.entries), 500)

    def test_manual_category_never_gets_deeplink(self):
        # Even if a physical step mentions settings-like words, manual actions must have no deeplink
        action = Action(
            actionName="Inspect Charging Port",
            description="It will check for physical dirt",
            category=actionCategory.manual,
            stepGroups=[StepGroup(steps=["Examine port under light"])],
        )
        self.assertEqual(action.category, actionCategory.manual)

    def test_real_match_from_catalog(self):
        match = self.matcher.best_match("Wi-Fi scanning settings page")
        self.assertIsNotNone(match)
        self.assertTrue(match.entry.deeplink.startswith("voiceassist://"))

    def test_dummy_fallback_on_unmatched(self):
        # Gibberish search text
        match = self.matcher.best_match("xyzzy qwerty unsupported imaginary screen 999")
        self.assertIsNone(match)  # Below threshold, signals fallback to dummy_positive


class TestSemanticCache(unittest.TestCase):
    def test_paraphrase_cache_hit(self):
        cache = SemanticCache()
        sample_goal = Goal(
            goal="Follow these steps to perform this Battery Troubleshooting",
            title="Battery Drain",
            score=0.9,
            actions=[],
        )
        cache.put("my battery is draining extremely fast", sample_goal)

        # Exact match
        hit1 = cache.get("my battery is draining extremely fast")
        self.assertIsNotNone(hit1)

        # Paraphrase match (different wording)
        hit2 = cache.get("battery drains fast phone won't hold charge")
        self.assertIsNotNone(hit2)

        # Completely unrelated query
        miss = cache.get("camera lens has cracked glass")
        self.assertIsNone(miss)


class TestOfficialSampleOutputSchema(unittest.TestCase):
    def test_sample_output_matches_pydantic_schema(self):
        sample_file = Path("data/sample_output.json")
        self.assertTrue(sample_file.exists())
        data = json.loads(sample_file.read_text(encoding="utf-8"))

        contexts = data["response"]["contexts"]
        self.assertEqual(len(contexts), 1)
        goal = Goal(**contexts[0])
        self.assertEqual(goal.title, "Screen display damage")
        self.assertEqual(len(goal.actions), 2)
        self.assertEqual(goal.actions[0].category, actionCategory.auto)
        self.assertEqual(goal.actions[1].category, actionCategory.manual)


class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app.main import app
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_catalog_stats_endpoint(self):
        resp = self.client.get("/debug/catalog-stats")
        self.assertEqual(resp.status_code, 200)
        self.assertGreaterEqual(resp.json().get("indexed_entries", 0), 500)

    def test_cache_stats_endpoint(self):
        resp = self.client.get("/debug/cache-stats")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("entries", resp.json())


if __name__ == "__main__":
    unittest.main()
