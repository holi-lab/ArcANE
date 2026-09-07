"""Offline regression tests for probe axis-source selection."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from probe_generation import cli, config


class AxisSourceTests(unittest.TestCase):
    def setUp(self):
        self.task_dir = tempfile.TemporaryDirectory(prefix="arcane-axis-source-")
        self.addCleanup(self.task_dir.cleanup)
        self.root = Path(self.task_dir.name)
        patcher = patch.object(config, "ARC_RESULTS", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.novel = "Example Novel"
        self.slug = "test_character"

    def _write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def _axes(self, source, axes=None):
        suffix = {"final": "final", "final_auto": "auto", "final_validated": "validated"}[source]
        return self._write(
            self.root / self.novel / source / f"{self.slug}_{suffix}_axes.json",
            {"intrapersonal_axes": axes or [], "relational_axes": []},
        )

    def _probe_stamp(self, source):
        return self._write(
            self.root / self.novel / "probes" / f"{self.slug}_probes.json",
            {"axis_source": source},
        )

    def _resolve(self, source="auto"):
        return config.axes_path(self.novel, self.slug, source)

    def test_human_validation_overrides_every_historical_source(self):
        validated = self._axes("final_validated", [{"axis_id": "kept"}])
        self._axes("final", [{"axis_id": "kept"}, {"axis_id": "rejected"}])
        self._axes("final_auto", [{"axis_id": "rejected"}])
        for stamped in ("final", "final_auto", "final_validated", None):
            with self.subTest(stamped=stamped):
                self._probe_stamp(stamped)
                self.assertEqual(self._resolve(), (validated, "final_validated"))

    def test_explicit_source_overrides_validation(self):
        sources = {source: self._axes(source)
                   for source in ("final", "final_auto", "final_validated")}
        self._probe_stamp("final_validated")
        for source, expected in sources.items():
            with self.subTest(source=source):
                self.assertEqual(self._resolve(source), (expected, source))

    def test_unvalidated_character_preserves_available_legacy_source(self):
        for source in ("final", "final_auto"):
            expected = self._axes(source)
            self._probe_stamp(source)
            with self.subTest(source=source):
                self.assertEqual(self._resolve(), (expected, source))

    def test_missing_validated_file_falls_back_to_legacy_source(self):
        (self.root / self.novel / "final_validated").mkdir(parents=True)
        expected = self._axes("final")
        self._probe_stamp("final")
        self.assertEqual(self._resolve(), (expected, "final"))

    def test_missing_stamped_file_falls_back_to_auto_axes(self):
        expected = self._axes("final_auto")
        for source in ("final", "final_validated"):
            with self.subTest(source=source):
                self._probe_stamp(source)
                self.assertEqual(self._resolve(), (expected, "final_auto"))

    def test_invalid_stamp_falls_back_to_auto_axes(self):
        expected = self._axes("final_auto")
        for source in ("unknown", None, {}, [], 1):
            with self.subTest(source=source):
                self._probe_stamp(source)
                self.assertEqual(self._resolve(), (expected, "final_auto"))

    def test_malformed_probe_metadata_falls_back_to_auto_axes(self):
        expected = self._axes("final_auto")
        path = self._probe_stamp("final")
        for payload in ("{", "[]", "null", "3"):
            with self.subTest(payload=payload):
                path.write_text(payload, encoding="utf-8")
                self.assertEqual(self._resolve(), (expected, "final_auto"))

    def test_no_existing_source_returns_expected_auto_path(self):
        path, source = self._resolve()
        self.assertEqual(source, "final_auto")
        self.assertEqual(path, self.root / self.novel / "final_auto" / f"{self.slug}_auto_axes.json")
        self.assertFalse(path.exists())

    def test_unknown_explicit_source_is_rejected(self):
        with self.assertRaises(ValueError):
            self._resolve("unknown")

    def test_empty_validated_axes_do_not_generate_from_rejected_axes(self):
        validated = self._axes("final_validated")
        self._axes("final", [{"axis_id": "rejected"}])
        self._probe_stamp("final")
        generate = AsyncMock()
        with patch.object(cli, "generate_arc_family", generate), patch.object(cli, "logger"):
            asyncio.run(cli.run_one_character(
                object(), asyncio.Semaphore(1), self.novel, "Test Character",
            ))
        self.assertEqual(self._resolve(), (validated, "final_validated"))
        generate.assert_not_awaited()

    def test_malformed_validated_file_fails_without_legacy_fallback(self):
        validated = self._axes("final_validated")
        validated.write_text("{", encoding="utf-8")
        self._axes("final", [{"axis_id": "rejected"}])
        self._probe_stamp("final")
        self.assertEqual(self._resolve(), (validated, "final_validated"))
        generate = AsyncMock()
        with patch.object(cli, "generate_arc_family", generate):
            with self.assertRaises(json.JSONDecodeError):
                asyncio.run(cli.run_one_character(
                    object(), asyncio.Semaphore(1), self.novel, "Test Character",
                ))
        generate.assert_not_awaited()

    def test_human_approved_critic_none_axis_is_retained(self):
        arc = {
            "axis_id": "human_approved",
            "annotation": {"valid_votes": 2, "n_annotators": 3},
            "literary_validation": {"validation_tag": "none_axis"},
        }
        self.assertTrue(cli._arc_passes_filter(arc, "final_validated"))
        with patch.object(cli, "logger"):
            for source in ("final", "final_auto"):
                with self.subTest(source=source):
                    self.assertFalse(cli._arc_passes_filter(arc, source))

    def test_other_critic_tags_keep_existing_behavior(self):
        for source in ("final", "final_auto", "final_validated"):
            for tag in ("valid_axis", "partially_valid_axis", ""):
                with self.subTest(source=source, tag=tag):
                    arc = {"literary_validation": {"validation_tag": tag}}
                    self.assertTrue(cli._arc_passes_filter(arc, source))

    def test_default_generation_keeps_only_human_approved_axes(self):
        kept = {
            "axis_id": "kept",
            "annotation": {"valid_votes": 2, "n_annotators": 3},
            "literary_validation": {"validation_tag": "none_axis"},
        }
        rejected = {
            "axis_id": "rejected",
            "literary_validation": {"validation_tag": "valid_axis"},
        }
        self._axes("final", [kept, rejected])
        self._axes("final_validated", [kept])
        path = self._probe_stamp("final")
        generate = AsyncMock(return_value={"axis_id": "kept"})
        with patch.object(cli, "generate_arc_family", generate):
            asyncio.run(cli.run_one_character(
                object(), asyncio.Semaphore(1), self.novel, "Test Character",
            ))
        generate.assert_awaited_once()
        self.assertEqual(generate.await_args.args[4], kept)
        output = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(output["axis_source"], "final_validated")
        self.assertEqual(output["families"], [{"axis_id": "kept"}])

    def test_explicit_unvalidated_generation_preserves_critic_filter(self):
        rejected_by_critic = {
            "axis_id": "critic_none",
            "literary_validation": {"validation_tag": "none_axis"},
        }
        kept_by_critic = {
            "axis_id": "critic_valid",
            "literary_validation": {"validation_tag": "valid_axis"},
        }
        self._axes("final_validated", [rejected_by_critic])
        for source in ("final", "final_auto"):
            with self.subTest(source=source):
                self._axes(source, [rejected_by_critic, kept_by_critic])
                generate = AsyncMock(return_value={"axis_id": "critic_valid"})
                with patch.object(cli, "generate_arc_family", generate), patch.object(cli, "logger"):
                    asyncio.run(cli.run_one_character(
                        object(), asyncio.Semaphore(1), self.novel, "Test Character",
                        axes_source=source,
                    ))
                generate.assert_awaited_once()
                self.assertEqual(generate.await_args.args[4], kept_by_critic)


if __name__ == "__main__":
    unittest.main()
