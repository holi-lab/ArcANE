"""Offline argument-forwarding checks for the SFT and DPO shell launchers."""

import subprocess
import tempfile
import unittest
from pathlib import Path


SFT_ROOT = Path(__file__).resolve().parents[1] / "training" / "sft"
LAUNCHERS = {
    "train_sft_full.sh": ("sft", "full"),
    "train_sft_lora.sh": ("sft", "lora"),
    "train_dpo_full.sh": ("dpo", "full"),
    "train_dpo_lora.sh": ("dpo", "lora"),
}


class SFTLauncherTests(unittest.TestCase):
    def setUp(self):
        task_dir = tempfile.TemporaryDirectory(prefix="arcane-sft-launcher-")
        self.addCleanup(task_dir.cleanup)
        self.root = Path(task_dir.name)
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        self.args_path = self.root / "arguments"
        self.context_path = self.root / "context"
        mock_uv = bin_dir / "uv"
        mock_uv.write_text(
            "#!/usr/bin/env bash\n"
            "set -eu\n"
            "printf '%s\\0' \"$@\" > \"$ARCANE_TEST_ARGUMENTS\"\n"
            "printf '%s\\0' \"$CUDA_VISIBLE_DEVICES\" \"$PWD\" \"$WANDB_PROJECT\" "
            "> \"$ARCANE_TEST_CONTEXT\"\n"
            "exit \"${ARCANE_TEST_EXIT_CODE:-0}\"\n",
            encoding="utf-8",
        )
        mock_uv.chmod(0o755)
        self.environ = {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "ARCANE_TEST_ARGUMENTS": str(self.args_path),
            "ARCANE_TEST_CONTEXT": str(self.context_path),
        }

    @staticmethod
    def _fields(path):
        return path.read_bytes().decode("utf-8").rstrip("\0").split("\0")

    def _run(self, launcher, args=(), **overrides):
        # /bin/bash exercises the macOS Bash 3.2 regression on macOS and
        # the system Bash on Linux, without calling uv or loading a model.
        result = subprocess.run(
            ["/bin/bash", str(SFT_ROOT / "scripts" / launcher), *args],
            cwd=self.root,
            env={**self.environ, **overrides},
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result

    def test_all_launchers_support_the_default_single_process(self):
        for launcher, (task, tuning) in LAUNCHERS.items():
            with self.subTest(launcher=launcher):
                result = self._run(launcher)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                args = self._fields(self.args_path)
                self.assertEqual(args[:3], ["run", "accelerate", "launch"])
                self.assertNotIn("--multi_gpu", args)
                self.assertEqual(args[args.index("--num_processes") + 1], "1")
                self.assertEqual(args[args.index("--gpu_ids") + 1], "0")
                self.assertIn(f"arcane/{task}_train.py", args)
                self.assertEqual(
                    Path(args[args.index("--config") + 1]),
                    Path("configs") / tuning / f"{task}.yaml",
                )
                self.assertEqual(
                    Path(args[args.index("--config_file") + 1]),
                    Path("configs/accelerate.yaml"),
                )
                gpu_ids, workdir, project = self._fields(self.context_path)
                self.assertEqual((gpu_ids, project), ("0", "arcane"))
                self.assertEqual(Path(workdir).resolve(), SFT_ROOT.resolve())

    def test_all_launchers_enable_multi_gpu_for_multiple_processes(self):
        for launcher, (task, _) in LAUNCHERS.items():
            with self.subTest(launcher=launcher):
                result = self._run(launcher, GPU_IDS="1,3", NUM_PROCESSES="2")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                args = self._fields(self.args_path)
                self.assertEqual(args.count("--multi_gpu"), 1)
                self.assertLess(args.index("--multi_gpu"), args.index(f"arcane/{task}_train.py"))
                self.assertEqual(args[args.index("--num_processes") + 1], "2")
                self.assertEqual(args[args.index("--gpu_ids") + 1], "1,3")
                self.assertEqual(self._fields(self.context_path)[0], "1,3")

    def test_paths_and_forwarded_arguments_preserve_spaces_and_glob_characters(self):
        extra = [
            "--model-name-or-path", "/tmp/My Model [draft]",
            "--resume-from-checkpoint", "/tmp/run/checkpoint with spaces",
            "--report-to", "none",
        ]
        for launcher in LAUNCHERS:
            for processes in ("1", "2"):
                with self.subTest(launcher=launcher, processes=processes):
                    result = self._run(
                        launcher, extra,
                        CONFIG_PATH="configs/custom training.yaml",
                        ACCELERATE_CONFIG_PATH="configs/custom accelerate.yaml",
                        NUM_PROCESSES=processes,
                        WANDB_PROJECT="custom project",
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    args = self._fields(self.args_path)
                    self.assertEqual(args[-len(extra):], extra)
                    self.assertEqual(args[args.index("--config") + 1], "configs/custom training.yaml")
                    self.assertEqual(
                        args[args.index("--config_file") + 1], "configs/custom accelerate.yaml",
                    )
                    self.assertEqual(self._fields(self.context_path)[2], "custom project")

    def test_launcher_propagates_uv_failure(self):
        for launcher in LAUNCHERS:
            with self.subTest(launcher=launcher):
                result = self._run(launcher, ARCANE_TEST_EXIT_CODE="7")
                self.assertEqual(result.returncode, 7, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
