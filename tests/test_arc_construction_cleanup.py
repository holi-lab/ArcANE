"""Offline regression tests for source preparation and axis artifact saving."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from arc_construction.merge_volumes import strip_gutenberg
from arc_construction import phase3_literary_grounding as grounding


class GutenbergStripTests(unittest.TestCase):
    def test_wrapped_body_uses_offsets_after_header_removal(self):
        body = "Chapter 1\nA synthetic passage.\nChapter 2\nAnother passage."
        for article in ("THE", "THIS"):
            for lowercase in (False, True):
                with self.subTest(article=article, lowercase=lowercase):
                    start = f"*** START OF {article} PROJECT GUTENBERG EBOOK EXAMPLE ***"
                    end = f"*** END OF {article} PROJECT GUTENBERG EBOOK EXAMPLE ***"
                    if lowercase:
                        start, end = start.lower(), end.lower()
                    wrapped = f"Outside preface\n{start}\n{body}\n{end}\nOutside footer"
                    self.assertEqual(strip_gutenberg(wrapped), body)

    def test_partial_or_absent_markers(self):
        body = "Chapter 1\nA synthetic passage."
        for article in ("THE", "THIS"):
            with self.subTest(article=article):
                start = f"*** START OF {article} PROJECT GUTENBERG EBOOK EXAMPLE ***"
                end = f"*** END OF {article} PROJECT GUTENBERG EBOOK EXAMPLE ***"
                self.assertEqual(strip_gutenberg(f"{start}\n{body}"), body)
                self.assertEqual(strip_gutenberg(f"{body}\n{end}\nFooter"), body)
        self.assertEqual(strip_gutenberg(f"  {body}\n"), body)


class GroundingSaveTests(unittest.TestCase):
    def setUp(self):
        self.axes = {
            "Test Character": {
                "character": "Test Character",
                "arc_richness": "rich",
                "intrapersonal_axes": [
                    {"axis_id": "valid", "axis_name": "Validated axis"},
                    {"axis_id": "partial", "axis_name": "One-critic axis"},
                ],
                "relational_axes": [
                    {"axis_id": "missing", "axis_name": "Unevaluated axis"},
                ],
            },
        }
        citation = {"author": "Example", "title": "Synthetic test", "year": 1900}
        positive = {"verdict": True, "reasoning": "Synthetic finding", "citations": [citation]}
        self.results = {
            "Test Character": {
                "valid": {
                    "structuralist": copy.deepcopy(positive),
                    "psychological": copy.deepcopy(positive),
                },
                "partial": {"historical_cultural": copy.deepcopy(positive)},
            },
        }

    def _save(self, entrypoint, axes, results):
        with tempfile.TemporaryDirectory(prefix="arcane-save-test-") as task_dir:
            with patch.multiple(
                grounding.config,
                RESULTS_BASE=task_dir,
                CHARACTER_DIRS={"Test Character": "custom_slug"},
            ), patch.object(grounding, "confirm", return_value=True), redirect_stdout(io.StringIO()):
                returned = entrypoint(axes, results)
            snapshot = {
                path.relative_to(task_dir).as_posix(): path.read_bytes()
                for path in Path(task_dir).rglob("*.json")
            }
        return returned, snapshot

    def test_interactive_and_noninteractive_artifacts_match(self):
        original = copy.deepcopy((self.axes, self.results))
        returned, interactive = self._save(grounding.step5_save, self.axes, self.results)
        _, silent = self._save(grounding._save_no_confirm, self.axes, self.results)
        self.assertIs(returned, True)
        self.assertEqual(interactive, silent)
        self.assertEqual((self.axes, self.results), original)
        self.assertEqual(set(silent), {
            "final_grounded/custom_slug_grounded_axes.json",
            "final_grounded/all_characters_grounded.json",
            "final_auto/custom_slug_auto_axes.json",
            "final_auto/all_characters_auto.json",
        })

        full = json.loads(silent["final_grounded/custom_slug_grounded_axes.json"])
        filtered = json.loads(silent["final_auto/custom_slug_auto_axes.json"])
        self.assertEqual(full["character"], "Test Character")
        self.assertEqual(full["arc_richness"], "rich")
        axes = full["intrapersonal_axes"] + full["relational_axes"]
        self.assertEqual(
            [(axis["axis_id"], axis["literary_validation"]["validation_tag"]) for axis in axes],
            [("valid", "valid_axis"), ("partial", "partially_valid_axis"), ("missing", "none_axis")],
        )
        for axis in axes:
            self.assertEqual(set(axis["literary_validation"]["evaluators"]), {
                "structuralist", "psychological", "historical_cultural",
            })
        missing = axes[-1]["literary_validation"]["evaluators"]
        self.assertTrue(all(not critic["verdict"] for critic in missing.values()))
        self.assertTrue(all(critic["citations"] == [] for critic in missing.values()))
        self.assertEqual(filtered["intrapersonal_axes"], [full["intrapersonal_axes"][0]])
        self.assertEqual(filtered["relational_axes"], [])
        self.assertEqual(
            json.loads(silent["final_grounded/all_characters_grounded.json"]),
            {"Test Character": full},
        )
        self.assertEqual(
            json.loads(silent["final_auto/all_characters_auto.json"]),
            {"Test Character": filtered},
        )

    def test_empty_input_writes_empty_aggregates(self):
        _, artifacts = self._save(grounding._save_no_confirm, {}, {})
        self.assertEqual(set(artifacts), {
            "final_grounded/all_characters_grounded.json",
            "final_auto/all_characters_auto.json",
        })
        self.assertTrue(all(json.loads(value) == {} for value in artifacts.values()))

    def test_declined_confirmation_writes_nothing(self):
        with tempfile.TemporaryDirectory(prefix="arcane-save-decline-test-") as task_dir:
            with patch.object(grounding.config, "RESULTS_BASE", task_dir), \
                    patch.object(grounding, "confirm", return_value=False), \
                    redirect_stdout(io.StringIO()):
                self.assertIs(grounding.step5_save(self.axes, self.results), False)
            self.assertEqual(list(Path(task_dir).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
