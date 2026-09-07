"""Offline regression checks for score validation and safe evaluation outputs."""

import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from evaluation.per_response import cli as response_cli, runner as response
from evaluation.per_response.loader import EvalUnit
from evaluation.per_response.prompt import EvalTrial
from evaluation.scoring import EvaluationRunError, validate_scale
from evaluation.trajectory import cli as trajectory_cli, runner as trajectory
from evaluation.trajectory.loader import TrajUnit
from evaluation.trajectory.prompt import PhaseRef


def verdict(module, value=50, **extra):
    return {"scores": dict.fromkeys(module.DIMENSIONS, value), **extra}


def unit_for(module, probe_id="axis_inworld_a0"):
    if module is response:
        return EvalUnit("novel", "character", probe_id, 0, "model",
                        EvalTrial("scenario", "question", "action", "speech", "thought", "phase"),
                        {"vanilla": "answer"})
    phases = [PhaseRef(i, "phase", "action", "speech", "thought", "answer") for i in (0, 1)]
    return TrajUnit("novel", "character", probe_id, "model", "scenario", "question",
                    {"vanilla": phases})


def record_for(module, unit, obj, error=None, scale=100):
    if module is response:
        return module._record(unit, "vanilla", obj, error, scale)
    return module._record(unit, "vanilla", unit.phases_by_mode["vanilla"], obj, error, scale)


def completion(obj):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(obj), refusal=None))],
        usage=None,
    )


class ScoreValidationTests(unittest.TestCase):
    def test_boundaries_and_explicit_scale(self):
        for module in (response, trajectory):
            for scale in (1, 5, 100):
                for value in (1, scale, (scale + 1) / 2):
                    with self.subTest(module=module.__name__, scale=scale, value=value):
                        self.assertEqual(module._scores_of(verdict(module, value), scale)["average"], value)
                self.assertIsNone(module._scores_of(verdict(module, scale + 1), scale))

    def test_invalid_dimension_values_are_rejected(self):
        for module in (response, trajectory):
            for value in (0, -1, 101, True, False, float("nan"), float("inf"),
                          -float("inf"), "50", None, 10 ** 1000):
                for dim in module.DIMENSIONS:
                    with self.subTest(module=module.__name__, dim=dim, value=str(value)[:30]):
                        obj = verdict(module)
                        obj["scores"][dim] = value
                        self.assertIsNone(module._scores_of(obj, 100))

    def test_invalid_shapes_and_missing_dimensions_are_rejected(self):
        for module in (response, trajectory):
            for obj in (None, [], 1, True, {}, {"scores": []}, {"scores": "invalid"},
                        {"scores": {module.DIMENSIONS[0]: 50}}):
                with self.subTest(module=module.__name__, obj=obj):
                    self.assertIsNone(module._scores_of(obj, 100))

    def test_average_is_always_recomputed(self):
        for module in (response, trajectory):
            for supplied in (75, True, float("nan"), "wrong", None):
                obj = {"scores": dict(zip(module.DIMENSIONS, (25, 20, 30))), "average": supplied}
                self.assertEqual(module._scores_of(obj, 100)["average"], 25)
                self.assertEqual(record_for(module, unit_for(module), obj)["average"], 25)

    def test_invalid_configuration_is_rejected(self):
        for scale in (0, -1, True, 1.5, "100", None):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                validate_scale(scale)

    def test_record_rejects_invalid_scores_and_preserves_errors(self):
        for module in (response, trajectory):
            unit = unit_for(module)
            for obj, error in ((verdict(module, 6), None), (verdict(module, 5), "call failed")):
                record = record_for(module, unit, obj, error, scale=5)
                self.assertFalse(record["parse_ok"])
                self.assertIsNone(record["scores"])
                self.assertIsNone(record["average"])
                self.assertTrue(record["error"])

    def test_invalid_replies_are_retried_at_requested_scale(self):
        for module in (response, trajectory):
            for bad in (verdict(module, 6), verdict(module, True), {"scores": []}, {}):
                with self.subTest(module=module.__name__, bad=bad):
                    create = Mock(side_effect=[completion(bad), completion(verdict(module, 5))])
                    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
                    with patch.object(module, "MAX_RETRIES", 2), patch.object(module.time, "sleep"):
                        text, _ = module._judge_chat(client, "test-judge", "system", "user", {}, scale=5)
                    self.assertEqual(create.call_count, 2)
                    self.assertEqual(json.loads(text), verdict(module, 5))

    def test_exhausted_invalid_replies_raise(self):
        for module in (response, trajectory):
            client = Mock()
            client.chat.completions.create.return_value = completion(verdict(module, 0))
            with patch.object(module, "MAX_RETRIES", 2), patch.object(module.time, "sleep"):
                with self.assertRaisesRegex(RuntimeError, "out-of-range"):
                    module._judge_chat(client, "test-judge", "system", "user", {}, scale=100)
            self.assertEqual(client.chat.completions.create.call_count, 2)


class EvaluationOutputTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="arcane-judge-")))
        self.stack.enter_context(patch.object(response.config, "RESULTS_ROOT", self.root))
        for module in (response, trajectory):
            self.stack.enter_context(patch.object(module.eval_config, "SCORE_SCALE", 100))
            self.stack.enter_context(patch.object(module.eval_config, "EVAL_CONCURRENCY", 1))
            self.stack.enter_context(patch.object(module.eval_config, "make_judge_client", return_value=Mock()))

    def paths(self, module):
        return (module.records_path("novel", "character", "test-judge"),
                module.summary_path("novel", "character", "test-judge"))

    def seed_outputs(self, module):
        paths = self.paths(module)
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"previous successful output\n")
        return paths

    def run_units(self, module, count=1):
        units = [unit_for(module, f"axis_inworld_a{i}") for i in range(count)]
        todo = units if module is response else [(u, "vanilla") for u in units]
        return module.run("novel", "character", todo, judge_model="test-judge")

    def partials(self, module):
        path, _ = self.paths(module)
        return sorted(path.parent.glob(path.name + ".*.partial"))

    def test_all_failures_preserve_records_and_summary(self):
        for module in (response, trajectory):
            with self.subTest(module=module.__name__):
                paths = self.seed_outputs(module)
                with patch.object(module, "_judge_chat", side_effect=RuntimeError("judge unavailable")):
                    with self.assertRaisesRegex(EvaluationRunError, "1/1 judge calls failed"):
                        self.run_units(module)
                for path in paths:
                    self.assertEqual(path.read_bytes(), b"previous successful output\n")
                rows = [json.loads(line) for line in self.partials(module)[0].read_text().splitlines()]
                self.assertEqual(len(rows), 1)
                self.assertFalse(rows[0]["parse_ok"])
                self.assertIn("judge unavailable", rows[0]["error"])

    def test_partial_failure_preserves_previous_outputs(self):
        for module in (response, trajectory):
            paths = self.seed_outputs(module)
            replies = [(json.dumps(verdict(module)), {}), RuntimeError("judge unavailable")]
            with patch.object(module, "_judge_chat", side_effect=replies):
                with self.assertRaisesRegex(EvaluationRunError, "1/2 judge calls failed"):
                    self.run_units(module, 2)
            for path in paths:
                self.assertEqual(path.read_bytes(), b"previous successful output\n")
            rows = [json.loads(line) for line in self.partials(module)[0].read_text().splitlines()]
            self.assertCountEqual([r["parse_ok"] for r in rows], [True, False])

    def test_fresh_failed_run_publishes_no_canonical_outputs(self):
        for module in (response, trajectory):
            with patch.object(module, "_judge_chat", return_value=(json.dumps(verdict(module, 0)), {})):
                with self.assertRaises(EvaluationRunError):
                    self.run_units(module)
            self.assertTrue(self.partials(module))
            for path in self.paths(module):
                self.assertFalse(path.exists())

    def test_failed_attempts_have_distinct_diagnostic_files(self):
        for module in (response, trajectory):
            with patch.object(module, "_judge_chat", side_effect=RuntimeError("judge unavailable")):
                for _ in range(2):
                    with self.assertRaises(EvaluationRunError):
                        self.run_units(module)
            self.assertEqual(len(self.partials(module)), 2)

    def test_success_replaces_outputs_with_validated_scores(self):
        for module in (response, trajectory):
            paths = self.seed_outputs(module)
            with patch.object(module, "_judge_chat", return_value=(json.dumps(verdict(module, 25, average=75)), {})):
                summary = self.run_units(module)
            self.assertEqual(summary["by_model"]["model"]["vanilla"]["average"], 25)
            self.assertEqual(json.loads(paths[0].read_text())["average"], 25)
            self.assertEqual(json.loads(paths[1].read_text()), summary)
            self.assertEqual(self.partials(module), [])

    def test_client_creation_failure_preserves_outputs(self):
        for module in (response, trajectory):
            paths = self.seed_outputs(module)
            with patch.object(module.eval_config, "make_judge_client", side_effect=RuntimeError("client failed")):
                with self.assertRaisesRegex(RuntimeError, "client failed"):
                    self.run_units(module)
            for path in paths:
                self.assertEqual(path.read_bytes(), b"previous successful output\n")

    def test_invalid_scale_fails_before_creating_outputs(self):
        for module in (response, trajectory):
            with patch.object(module.eval_config, "SCORE_SCALE", 0), self.assertRaises(ValueError):
                self.run_units(module)
            self.assertFalse((self.root / "novel").exists())

    def test_aggregate_validates_and_recomputes_without_editing_records(self):
        for module in (response, trajectory):
            path, _ = self.paths(module)
            path.parent.mkdir(parents=True, exist_ok=True)
            good = record_for(module, unit_for(module), verdict(module, 25))
            good["average"] = 75
            bad = record_for(module, unit_for(module, "axis_inworld_bad"), verdict(module))
            bad["scores"][module.DIMENSIONS[0]] = 0
            source = "".join(json.dumps(row) + "\n" for row in (good, bad))
            path.write_text(source, encoding="utf-8")
            with self.assertLogs(module.logger, level="WARNING") as logs:
                summary = module.aggregate(path, "novel", "character", "test-judge", 100)
            block = summary["by_model"]["model"]["vanilla"]
            self.assertEqual((block["average"], block["n"]), (25, 1))
            if module is trajectory:
                self.assertEqual(block["by_n_phases"]["2"]["average"], 25)
            self.assertIn("skipped 1 records", "\n".join(logs.output))
            self.assertEqual(path.read_text(encoding="utf-8"), source)


class EvaluationCliTests(unittest.TestCase):
    def test_nonpositive_limit_fails_before_loading_or_running(self):
        for cli, loader in ((response_cli, "build_eval_units"),
                            (trajectory_cli, "build_traj_units")):
            for limit in (0, -1):
                with patch.object(cli, loader) as load, patch.object(cli, "run") as run:
                    with self.assertRaisesRegex(SystemExit, "positive integer"):
                        cli.main(["--limit", str(limit)])
                    load.assert_not_called()
                    run.assert_not_called()

    def test_failed_run_exits_nonzero_without_success_summary(self):
        for cli, module, loader in ((response_cli, response, "build_eval_units"),
                                     (trajectory_cli, trajectory, "build_traj_units")):
            with patch.object(cli, loader, return_value=[unit_for(module)]), \
                    patch.object(cli, "run", side_effect=EvaluationRunError("judge calls failed")), \
                    patch.object(cli, "print_summary") as summary, redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, "judge calls failed") as raised:
                    cli.main(["--novel", "novel", "--character", "character"])
                self.assertNotEqual(raised.exception.code, 0)
                summary.assert_not_called()

    def test_estimate_only_never_runs_judge(self):
        for cli, module, loader in ((response_cli, response, "build_eval_units"),
                                     (trajectory_cli, trajectory, "build_traj_units")):
            with patch.object(cli, loader, return_value=[unit_for(module)]), \
                    patch.object(cli, "run") as run, redirect_stdout(io.StringIO()):
                cli.main(["--novel", "novel", "--character", "character", "--estimate-only"])
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
