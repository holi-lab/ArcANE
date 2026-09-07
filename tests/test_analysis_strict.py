"""Offline checks for complete table inputs and pre-write validation."""

import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from analysis import build_paper_table as table


class StrictTableTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(
            tempfile.TemporaryDirectory(prefix="arcane-strict-table-")
        ))
        self.public_results = table.RESULTS_ROOT
        self.results = self.root / "results"
        self.out = self.root / "output" / "table.tex"
        self.stack.enter_context(patch.object(table, "RESULTS_ROOT", self.results))
        self.models = [("test-model", "Test", "Small")]
        self.stack.enter_context(patch.object(table, "MODEL_PRESETS", {"main": self.models}))
        self._fixture("First Novel")
        self._fixture("Second Novel")

    def _fixture(self, novel, character="character", omit=None):
        directory = self.results / novel / character
        directory.mkdir(parents=True, exist_ok=True)
        tokens = {"in_text": "intext", "in_world": "inworld", "out_of_world": "outworld"}
        for kind, dims in (("eval", table.EVAL1_DIMS), ("traj", table.EVAL2_DIMS)):
            rows = []
            for mode, _ in table.MODES:
                for ptype, _ in table.SECTIONS:
                    if (kind, mode, ptype) == omit:
                        continue
                    rows.append({
                        "novel": novel, "character": character,
                        "model": "test-model", "mode": mode,
                        "probe_id": f"axis_{tokens[ptype]}_a0", "phase_idx": 0,
                        "scale": 100, "parse_ok": True, "error": None,
                        "scores": {dim: 50.0 for dim in dims}, "average": 50.0,
                    })
            path = directory / f"{kind}_records_deepseek.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return directory

    def _run(self, *extra, novels="First Novel,Second Novel", strict=True):
        stdout, stderr = io.StringIO(), io.StringIO()
        argv = ["--novels", novels, "--out", str(self.out), *extra]
        if strict:
            argv.append("--strict")
        with redirect_stdout(stdout), redirect_stderr(stderr):
            table.main(argv)
        return stdout.getvalue(), stderr.getvalue()

    def test_complete_inputs_succeed(self):
        _, stderr = self._run()
        self.assertEqual(stderr, "")
        data = json.loads(self.out.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(data["novels"], ["First Novel", "Second Novel"])
        self.assertEqual(table._missing_cells(data["per_novel"], data["novels"], self.models), [])

    def test_missing_judge_file_fails_even_when_other_character_fills_cells(self):
        directory = self.results / "First Novel" / "second_character"
        directory.mkdir()
        (directory / "eval_records_deepseek.jsonl").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "missing judge file.*traj_records_deepseek"):
            self._run()
        self.assertFalse(self.out.parent.exists())

    def test_missing_one_novel_cell_fails_even_when_other_novel_fills_overall(self):
        self._fixture("First Novel", omit=("traj", "arc", "in_world"))
        with self.assertRaisesRegex(SystemExit, "1 missing per-novel metric cell.*First Novel / test-model / arc / in_world / PTF"):
            self._run()
        self.assertFalse(self.out.parent.exists())

    def test_empty_character_files_fail_even_when_another_character_fills_cells(self):
        directory = self._fixture("First Novel", character="empty_character")
        for path in directory.glob("*.jsonl"):
            path.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "2 judge file.*no usable selected-model records"):
            self._run()
        self.assertFalse(self.out.parent.exists())

    def test_unusable_character_files_fail_even_when_another_character_fills_cells(self):
        directory = self._fixture("First Novel", character="unusable_character")
        for field, value in (("parse_ok", False), ("model", "unselected-model")):
            with self.subTest(field=field):
                for path in directory.glob("*.jsonl"):
                    path.write_text(json.dumps({
                        "model": "test-model", "mode": "arc", "parse_ok": True,
                        "probe_id": "axis_inworld_a0", "scores": {"apf": 50, "ptf_alignment": 50},
                        field: value,
                    }) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(SystemExit, "no usable selected-model records"):
                    self._run()
                self.assertFalse(self.out.parent.exists())

    def test_wrong_judge_tag_fails_without_creating_output(self):
        with self.assertRaisesRegex(SystemExit, "no usable judge records.*not-a-judge"):
            self._run("--judge-tag", "not-a-judge")
        self.assertFalse(self.out.parent.exists())

    def test_empty_selected_records_fail_without_strict(self):
        with self.assertRaisesRegex(SystemExit, "no usable judge records"):
            self._run("--judge-tag", "not-a-judge", strict=False)
        self.assertFalse(self.out.parent.exists())

    def test_failure_preserves_existing_tex_and_json(self):
        self.out.parent.mkdir()
        self.out.write_text("existing table", encoding="utf-8")
        json_path = self.out.with_suffix(".json")
        json_path.write_text("existing numeric dump", encoding="utf-8")
        self._fixture("First Novel", omit=("traj", "arc", "in_world"))
        with self.assertRaises(SystemExit):
            self._run()
        self.assertEqual(self.out.read_text(encoding="utf-8"), "existing table")
        self.assertEqual(json_path.read_text(encoding="utf-8"), "existing numeric dump")

    def test_non_strict_warns_about_missing_file_and_cell(self):
        self._fixture("First Novel", omit=("traj", "arc", "in_world"))
        (self.results / "First Novel" / "unscored_character").mkdir()
        _, stderr = self._run(strict=False)
        self.assertIn("2 missing judge file(s)", stderr)
        self.assertIn("1 missing per-novel metric cell(s)", stderr)
        self.assertTrue(self.out.is_file())

    def test_non_strict_excludes_existing_but_empty_novel_from_caption(self):
        (self.results / "Empty Novel").mkdir()
        _, stderr = self._run(novels="First Novel,Empty Novel", strict=False)
        self.assertIn("EXCLUDED", stderr)
        self.assertIn("Empty Novel", stderr)
        data = json.loads(self.out.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(data["novels"], ["First Novel"])
        self.assertNotIn("Empty Novel", self.out.read_text(encoding="utf-8"))

    def test_missing_novel_directory_fails_before_writing(self):
        with self.assertRaisesRegex(SystemExit, "EXCLUDED.*Missing Novel"):
            self._run(novels="First Novel,Missing Novel")
        self.assertFalse(self.out.parent.exists())

    def test_parse_failed_records_do_not_make_a_novel_usable(self):
        for path in (self.results / "First Novel").rglob("*.jsonl"):
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            for row in rows:
                row["parse_ok"] = False
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "no usable judge records"):
            self._run(novels="First Novel")
        self.assertFalse(self.out.parent.exists())

    def test_public_main_preset_still_excludes_unreleased_novel(self):
        if not (self.public_results / "don_quixote").is_dir():
            self.skipTest("bundled judge records are unavailable")
        with patch.object(table, "RESULTS_ROOT", self.public_results), \
                patch.object(table, "MODEL_PRESETS", {"main": table.MODELS_MAIN}):
            _, stderr = self._run(novels="main", strict=False)
        data = json.loads(self.out.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(data["novels"], [n for n in table.NOVELS_MAIN if n != "Harry_Potter"])
        self.assertIn("Harry_Potter", stderr)
        self.assertNotIn("Harry Potter", self.out.read_text(encoding="utf-8"))

    def test_public_four_novels_pass_strict(self):
        if not (self.public_results / "don_quixote").is_dir():
            self.skipTest("bundled judge records are unavailable")
        novels = [n for n in table.NOVELS_MAIN if n != "Harry_Potter"]
        with patch.object(table, "RESULTS_ROOT", self.public_results), \
                patch.object(table, "MODEL_PRESETS", {"main": table.MODELS_MAIN}):
            _, stderr = self._run(novels=",".join(novels))
        self.assertEqual(stderr, "")


if __name__ == "__main__":
    unittest.main()
