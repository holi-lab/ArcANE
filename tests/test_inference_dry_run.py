"""Offline regression tests for side-effect-free inference previews."""

import asyncio
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, patch

from evaluation.per_response import loader as response_loader
from evaluation.trajectory import loader as trajectory_loader
from inference import cli, config, context_builders, runner
from inference.loaders import Trial, trial_to_dict
from inference.prompts import render_system, render_user
from inference.records import LEGACY_DRY_RUN_RESPONSE, is_dry_run_record


class DryRunTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(
            tempfile.TemporaryDirectory(prefix="arcane-dry-run-test-")
        ))
        self.stack.enter_context(patch.multiple(
            config, RESULTS_ROOT=self.root / "results", CACHE_ROOT=self.root / "cache",
            ARC_RESULTS=self.root / "arcs",
        ))
        context_builders._arc_json.cache_clear()
        self.addCleanup(context_builders._arc_json.cache_clear)
        self.trial = Trial(
            novel="Synthetic Novel", character="Test Character", character_slug="test",
            axis_id="axis_1", axis_name="Trust", axis_type="intrapersonal",
            target_character=None, probe_id="probe_1", probe_type="in_world",
            era_label=None, phase_idx=0, phase_label="Early", query_chapter=1,
            scenario="A friend asks for help.", question="What do you do?",
        )
        self.model = "test/model"
        axes_path = config.axes_file(self.trial.novel, self.trial.character_slug)
        axes_path.parent.mkdir(parents=True)
        axes_path.write_text(json.dumps({
            "character": self.trial.character, "intrapersonal_axes": [], "relational_axes": [],
        }), encoding="utf-8")
        chapter_path = config.chapter_path(self.trial.novel, 1)
        chapter_path.parent.mkdir(parents=True)
        chapter_path.write_text("Synthetic chapter.", encoding="utf-8")
        summary_path = config.summary_path(self.trial.novel, 1)
        summary_path.parent.mkdir(parents=True)
        summary_path.write_text("A friend arrives.", encoding="utf-8")

    def _run(self, *, dry_run=True, modes=None, record_prompts=True):
        output = io.StringIO()
        with redirect_stdout(output):
            asyncio.run(runner.run(
                self.trial.novel, self.trial.character_slug, [self.trial],
                modes or ["vanilla"], [self.model],
                dry_run=dry_run, record_prompts=record_prompts,
            ))
        return [json.loads(line) for line in output.getvalue().splitlines()]

    def _snapshot(self):
        return {
            path.relative_to(self.root).as_posix(): path.read_bytes() if path.is_file() else None
            for path in self.root.rglob("*")
        }

    def _result_row(self, response="A real answer.", **overrides):
        return {
            **trial_to_dict(self.trial),
            "trial_id": runner._build_full_trial_id(self.trial, "vanilla", self.model),
            "mode": "vanilla", "model": self.model, "response": response, "error": None,
            **overrides,
        }

    def _write_rows(self, rows):
        path = config.results_path(self.trial.novel, self.trial.character_slug, "vanilla", self.model)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_all_modes_preview_without_clients_api_calls_or_writes(self):
        before = self._snapshot()
        with ExitStack() as stack:
            mocks = [
                stack.enter_context(patch.object(config, name, side_effect=AssertionError(name)))
                for name in ("make_chat_client", "make_embed_client")
            ]
            for name in ("build_rag_async", "build_lifechoice_async", "build_timechara_async"):
                mocks.append(stack.enter_context(patch.object(
                    context_builders, name, new_callable=AsyncMock, side_effect=AssertionError(name),
                )))
            mocks.append(stack.enter_context(patch.object(
                runner, "chat_text", new_callable=AsyncMock, side_effect=AssertionError("chat_text"),
            )))
            rows = self._run(modes=list(config.SUPPORTED_MODES))
            for mock in mocks:
                mock.assert_not_called()

        self.assertEqual(self._snapshot(), before)
        self.assertFalse(config.RESULTS_ROOT.exists())
        self.assertEqual(len(rows), 6)
        for row in rows:
            with self.subTest(mode=row["mode"]):
                self.assertIs(row["dry_run"], True)
                self.assertIsNone(row["response"])
                self.assertFalse(row["already_completed"])
                self.assertIn("prompt_system", row)
                if row["mode"] in ("rag", "lifechoice", "timechara"):
                    self.assertFalse(row["prompt_complete"])
                    self.assertTrue(row["deferred_context"])
                    self.assertIsNone(row["prompt_hash"])
                    self.assertIsNone(row["context_char_len"])
                else:
                    self.assertTrue(row["prompt_complete"])
                    self.assertIsNone(row["deferred_context"])
                    self.assertTrue(row["prompt_hash"])
        self.assertTrue(next(row for row in rows if row["mode"] == "rag")["precheck_error"])

    def test_complete_preview_matches_real_prompt_rendering(self):
        for row in self._run(modes=["vanilla", "arc", "summary"]):
            context = context_builders.build_context_sync(row["mode"], self.trial)
            system, user = render_system(self.trial, context), render_user(self.trial)
            self.assertEqual(row["prompt_system"], system)
            self.assertEqual(row["prompt_user"], user)
            self.assertEqual(row["prompt_hash"], runner._hash(system + "\n---\n" + user))

    def test_existing_results_unchanged_and_completion_reported(self):
        self._write_rows([self._result_row()])
        before = self._snapshot()
        [row] = self._run()
        self.assertTrue(row["already_completed"])
        self.assertEqual(self._snapshot(), before)

    def test_real_run_after_preview_is_not_skipped(self):
        self._run()
        with patch.object(config, "make_chat_client"), patch.object(config, "make_embed_client"), \
                patch.object(runner, "chat_text", new_callable=AsyncMock,
                             return_value=("A real answer.", {}, 1)) as chat:
            self._run(dry_run=False)
        chat.assert_awaited_once()
        path = config.results_path(self.trial.novel, self.trial.character_slug, "vanilla", self.model)
        [row] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(row["response"], "A real answer.")
        self.assertFalse(is_dry_run_record(row))
        self.assertEqual(runner._completed_trial_ids(path), {row["trial_id"]})

    def test_legacy_placeholder_is_retried_and_real_response_is_evaluated(self):
        path = self._write_rows([self._result_row(LEGACY_DRY_RUN_RESPONSE)])
        self.assertEqual(runner._completed_trial_ids(path), set())
        self.assertFalse(self._run()[0]["already_completed"])
        with patch.object(config, "make_chat_client"), patch.object(config, "make_embed_client"), \
                patch.object(runner, "chat_text", new_callable=AsyncMock,
                             return_value=("Real replacement.", {}, 1)) as chat:
            self._run(dry_run=False)
        chat.assert_awaited_once()
        self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)
        for loader in (response_loader, trajectory_loader):
            self.assertEqual(loader._load_responses(path), (
                self.model, {(self.trial.probe_id, self.trial.phase_idx): "Real replacement."},
            ))

    def test_real_context_failure_is_recorded_and_can_be_retried(self):
        with patch.object(config, "make_chat_client"), patch.object(config, "make_embed_client"), \
                patch.object(context_builders, "build_context_sync", side_effect=ValueError("bad input")), \
                patch.object(runner, "chat_text", new_callable=AsyncMock) as chat:
            self._run(dry_run=False)
        chat.assert_not_called()
        path = config.results_path(self.trial.novel, self.trial.character_slug, "vanilla", self.model)
        row = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(row["error"], "context_build: bad input")
        self.assertEqual(runner._completed_trial_ids(path), set())
        for loader in (response_loader, trajectory_loader):
            self.assertEqual(loader._load_responses(path)[1], {})

    def test_resume_separates_truncated_tail_and_preserves_its_bytes(self):
        path = self._write_rows([])
        for tail in (b'{"trial_id":"interrupted', b'{"response":"\xe2\x80'):
            with self.subTest(tail=tail):
                path.write_bytes(tail)
                with patch.object(config, "make_chat_client"), patch.object(config, "make_embed_client"), \
                        patch.object(runner, "chat_text", new_callable=AsyncMock,
                                     return_value=("New response with café and 茶.", {}, 1)) as chat:
                    self._run(dry_run=False)
                    self._run(dry_run=False)
                chat.assert_awaited_once()
                self.assertTrue(path.read_bytes().startswith(tail + b"\n"))
                self.assertEqual(len(path.read_bytes().splitlines()), 2)
                self.assertEqual(runner._completed_trial_ids(path), {self._result_row()["trial_id"]})
                for loader in (response_loader, trajectory_loader):
                    self.assertEqual(loader._load_responses(path), (self.model, {
                        (self.trial.probe_id, self.trial.phase_idx): "New response with café and 茶.",
                    }))

    def test_resume_preserves_valid_row_without_final_newline(self):
        old = self._result_row("Previous answer.", trial_id="previous-trial", phase_idx=1)
        for ending in (b"", b"\n", b"\r\n"):
            with self.subTest(ending=ending):
                path = self._write_rows([])
                before = json.dumps(old).encode("utf-8") + ending
                path.write_bytes(before)
                self.assertEqual(runner._completed_trial_ids(path), {"previous-trial"})
                with patch.object(config, "make_chat_client"), patch.object(config, "make_embed_client"), \
                        patch.object(runner, "chat_text", new_callable=AsyncMock,
                                     return_value=("New answer.", {}, 1)) as chat:
                    self._run(dry_run=False)
                chat.assert_awaited_once()
                self.assertTrue(path.read_bytes().startswith(before))
                self.assertEqual(len(path.read_bytes().splitlines()), 2)
                self.assertEqual(runner._completed_trial_ids(path), {
                    "previous-trial", self._result_row()["trial_id"],
                })
                for loader in (response_loader, trajectory_loader):
                    self.assertEqual(loader._load_responses(path), (self.model, {
                        (self.trial.probe_id, 1): "Previous answer.",
                        (self.trial.probe_id, 0): "New answer.",
                    }))

    def test_dry_run_does_not_repair_an_incomplete_tail(self):
        path = self._write_rows([])
        for tail in (b'{"trial_id":"interrupted', b'{"response":"\xe2\x80'):
            with self.subTest(tail=tail):
                path.write_bytes(tail)
                before = self._snapshot()
                with patch.object(config, "make_chat_client") as chat, \
                        patch.object(config, "make_embed_client") as embed:
                    [row] = self._run()
                chat.assert_not_called()
                embed.assert_not_called()
                self.assertFalse(row["already_completed"])
                self.assertEqual(self._snapshot(), before)

    def test_already_completed_run_does_not_add_a_newline(self):
        path = self._write_rows([])
        before = json.dumps(self._result_row()).encode("utf-8")
        path.write_bytes(before)
        with patch.object(config, "make_chat_client"), patch.object(config, "make_embed_client"), \
                patch.object(runner, "chat_text", new_callable=AsyncMock) as chat:
            self._run(dry_run=False)
        chat.assert_not_awaited()
        self.assertEqual(path.read_bytes(), before)

    def test_invalid_non_record_json_lines_are_ignored(self):
        path = self._write_rows([None, [], 5, self._result_row()])
        self.assertEqual(runner._completed_trial_ids(path), {self._result_row()["trial_id"]})
        for loader in (response_loader, trajectory_loader):
            self.assertEqual(loader._load_responses(path), (self.model, {
                (self.trial.probe_id, self.trial.phase_idx): "A real answer.",
            }))

    def test_both_evaluators_and_resume_ignore_old_and_new_previews(self):
        preview = self._run()[0]
        for rows in (
            [self._result_row(LEGACY_DRY_RUN_RESPONSE)],
            [preview],
            [self._result_row("Even nonempty preview text must not be scored.", dry_run=True)],
        ):
            with self.subTest(rows=rows):
                path = self._write_rows(rows)
                self.assertEqual(runner._completed_trial_ids(path), set())
                for loader in (response_loader, trajectory_loader):
                    self.assertEqual(loader._load_responses(path), (None, {}))

    def test_preview_after_real_row_does_not_replace_model_response(self):
        path = self._write_rows([
            self._result_row(), self._result_row(LEGACY_DRY_RUN_RESPONSE),
            self._result_row(None, dry_run=True, model="preview-only-name"),
        ])
        for loader in (response_loader, trajectory_loader):
            self.assertEqual(loader._load_responses(path), (
                self.model, {(self.trial.probe_id, self.trial.phase_idx): "A real answer."},
            ))
        self.assertEqual(runner._completed_trial_ids(path), {self._result_row()["trial_id"]})

    def test_real_empty_and_thinking_responses_keep_existing_evaluation_behavior(self):
        path = self._write_rows([
            self._result_row("", phase_idx=0, dry_run=False),
            self._result_row("<think>Hidden text.</think>Visible text.", phase_idx=1),
            self._result_row(None, phase_idx=2, error="chat_call: unavailable"),
        ])
        for loader in (response_loader, trajectory_loader):
            self.assertEqual(loader._load_responses(path), (self.model, {
                (self.trial.probe_id, 0): response_loader.EMPTY_RESPONSE_SENTINEL,
                (self.trial.probe_id, 1): "Visible text.",
            }))

    def test_missing_local_assets_reported_without_writes(self):
        with patch.object(config, "ARC_RESULTS", self.root / "missing-arcs"):
            before = self._snapshot()
            rows = self._run(modes=["arc", "timechara"])
            for row in rows:
                self.assertFalse(row["prompt_complete"])
                self.assertTrue(row["precheck_error"])
            self.assertIn("context_build:", rows[0]["error"])
            self.assertEqual(self._snapshot(), before)

    def test_no_record_prompts_omits_preview_prompt_fields(self):
        [row] = self._run(record_prompts=False)
        self.assertNotIn("prompt_system", row)
        self.assertNotIn("prompt_user", row)
        self.assertTrue(row["prompt_hash"])

    def test_cli_routes_dry_run_without_constructing_api_clients(self):
        with patch.object(cli, "iter_trials", return_value=iter([self.trial])), \
                patch.object(config, "make_chat_client") as chat, \
                patch.object(config, "make_embed_client") as embed, \
                redirect_stdout(io.StringIO()) as output:
            cli.main([
                "--novel", self.trial.novel, "--character", self.trial.character_slug,
                "--mode", "timechara", "--model", self.model, "--dry-run", "--limit", "1",
            ])
        chat.assert_not_called()
        embed.assert_not_called()
        self.assertTrue(json.loads(output.getvalue())["dry_run"])
        self.assertFalse(config.RESULTS_ROOT.exists())


if __name__ == "__main__":
    unittest.main()
