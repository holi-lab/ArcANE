"""Offline shell regressions for RL installation and launch defaults."""

import ast
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


RL_ROOT = Path(__file__).resolve().parents[1] / "training" / "rl"
INSTALLER = RL_ROOT / "scripts" / "install_env.sh"
LAUNCHER = RL_ROOT / "scripts" / "run_grpo.sh"
VERL_REVISION = "081df509ca9fe02d524f913a7a3f96eefe9ca568"


class RLShellTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="arcane-rl-setup-test-")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.env_prefix = self.root / "rl-env"
        self.log_path = self.root / "commands.log"
        self.verl_dir = self.root / "verl"
        (self.verl_dir / "scripts").mkdir(parents=True)
        (self.verl_dir / "pyproject.toml").write_text("# Synthetic pinned checkout\n")
        self.helper = self.verl_dir / "scripts" / "install_vllm_sglang_mcore.sh"
        self.helper.write_text(
            'printf "helper\\n" >> "$TEST_LOG"\n'
            'exit 99\n'
        )
        self._command("uname", '''
if [[ "$1" == "-s" ]]; then
  printf '%s\n' "${FAKE_OS:-Linux}"
else
  printf '%s\n' "${FAKE_ARCH:-x86_64}"
fi
''')
        self._command("nvidia-smi", '''
if [[ "${FAKE_GPU_FAILURE:-0}" == "1" ]]; then exit 1; fi
if [[ "${FAKE_NO_GPUS:-0}" != "1" ]]; then printf 'GPU 0: test GPU\n'; fi
''')
        self._command("git", '''
if [[ "$3" == "rev-parse" ]]; then
  printf '%s\n' "$FAKE_VERL_REVISION"
elif [[ "$3" == "diff" ]]; then
  exit "${FAKE_DIRTY_CHECKOUT:-0}"
else
  exit 99
fi
''')
        self._command("conda", '''
if [[ "$1 $2" == "info --base" ]]; then
  printf '%s\n' "$TEST_ROOT/conda"
else
  exit 99
fi
''')
        self.python_fixture = self._command("python", '''
if [[ "${1:-}" == "-c" && "${2:-}" == *sys.version_info.major* ]]; then
  printf '%s\n' "${FAKE_PYTHON_MINOR:-3.12}"
elif [[ "${1:-}" == "-c" && "${2:-}" == *_GLIBCXX_USE_CXX11_ABI* ]]; then
  printf '%s\n' "${FAKE_CXX11_ABI:-TRUE}"
elif [[ "${1:-}" == "-c" ]]; then
  [[ "${FAKE_PYTHON_MINOR:-3.12}" == "3.12" ]]
elif [[ "${1:-}" == "-" ]]; then
  printf 'runtime-validation\n' >> "$TEST_LOG"
  while IFS= read -r line; do :; done
  exit "${FAKE_RUNTIME_FAILURE:-0}"
elif [[ "$*" == "-m pip check" ]]; then
  printf 'pip-check\n' >> "$TEST_LOG"
  exit "${FAKE_PIP_CHECK_FAILURE:-0}"
else
  printf 'python\n' >> "$TEST_LOG"
  printf '%s\n' "$@" >> "$TEST_LOG"
  printf 'reward-mode=%s\n' "${ARCANE_REWARD_MODE:-}" >> "$TEST_LOG"
  if [[ "${FAKE_INSTALL_FAILURE:-0}" == "1" && "$*" == *"--editable"* ]]; then exit 1; fi
fi
''')
        profile = self.root / "conda" / "etc" / "profile.d" / "conda.sh"
        profile.parent.mkdir(parents=True)
        profile.write_text('''
conda() {
  if [[ "$1" == "create" ]]; then
    printf 'conda-create %s\n' "$*" >> "$TEST_LOG"
    mkdir -p "$ARCANE_RL_ENV/bin"
    cp "$TEST_ROOT/bin/python" "$ARCANE_RL_ENV/bin/python"
  elif [[ "$1" == "activate" ]]; then
    if [[ "${FAKE_ACTIVATION_FAILURE:-0}" != "1" ]]; then
      export PATH="$ARCANE_RL_ENV/bin:$PATH"
    fi
  else
    return 99
  fi
}
''')
        self.environ = {
            "PATH": f"{self.bin_dir}:/usr/bin:/bin",
            "HOME": os.environ.get("HOME", str(self.root)),
            "TEST_ROOT": str(self.root),
            "TEST_LOG": str(self.log_path),
            "ARCANE_RL_ENV": str(self.env_prefix),
            "VERL_DIR": str(self.verl_dir),
            "FAKE_VERL_REVISION": VERL_REVISION,
            "RAY_TMPDIR": str(self.root / "ray"),
        }

    def _command(self, name, body):
        path = self.bin_dir / name
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + body)
        path.chmod(0o755)
        return path

    def _run(self, script=INSTALLER, *, args=(), **overrides):
        return subprocess.run(
            ["/bin/bash", str(script), *args],
            env={**self.environ, **overrides},
            capture_output=True,
            text=True,
            timeout=10,
        )

    def _log(self):
        return self.log_path.read_text() if self.log_path.exists() else ""

    def _existing_environment(self):
        target = self.env_prefix / "bin" / "python"
        target.parent.mkdir(parents=True)
        target.write_bytes(self.python_fixture.read_bytes())
        target.chmod(0o755)
        return target

    def _local_launch(self, **overrides):
        model = self.root / "model"
        model.mkdir(exist_ok=True)
        train = self.root / "train.parquet"
        val = self.root / "val.parquet"
        train.touch()
        val.touch()
        return self._run(
            LAUNCHER,
            BASE_MODEL=str(model),
            TRAIN=str(train),
            VAL=str(val),
            ARCANE_REWARD_URL="http://127.0.0.1:8000/score",
            **overrides,
        )

    def test_unsupported_platforms_fail_before_environment_creation(self):
        for overrides in ({"FAKE_OS": "Darwin"}, {"FAKE_ARCH": "aarch64"}):
            with self.subTest(overrides=overrides):
                result = self._run(**overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("requires Linux x86_64", result.stdout)
                self.assertFalse(self.env_prefix.exists())

    def test_incompatible_requested_python_fails_before_creation(self):
        result = self._run(PYTHON_VERSION="3.11")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires Python 3.12", result.stdout)
        self.assertFalse(self.env_prefix.exists())

    def test_missing_gpu_fails_before_creation(self):
        for key in ("FAKE_GPU_FAILURE", "FAKE_NO_GPUS"):
            with self.subTest(key=key):
                result = self._run(**{key: "1"})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("no accessible NVIDIA GPU", result.stdout)
                self.assertFalse(self.env_prefix.exists())

    def test_wrong_verl_revision_is_rejected(self):
        result = self._run(FAKE_VERL_REVISION="different-revision")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(VERL_REVISION, result.stdout)
        self.assertFalse(self.env_prefix.exists())

    def test_modified_verl_checkout_is_rejected(self):
        result = self._run(FAKE_DIRTY_CHECKOUT="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("tracked changes", result.stdout)
        self.assertFalse(self.env_prefix.exists())

    def test_existing_python_311_environment_is_preserved(self):
        python_path = self._existing_environment()
        original = python_path.read_bytes()
        result = self._run(FAKE_PYTHON_MINOR="3.11")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("choose a new ARCANE_RL_ENV", result.stdout)
        self.assertEqual(python_path.read_bytes(), original)
        self.assertEqual(self._log(), "")

    def test_new_environment_uses_python_312_and_validates_runtime(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        log = self._log()
        self.assertIn("python=3.12", log)
        self.assertIn("--constraint", log)
        self.assertIn(str(RL_ROOT / "constraints.txt"), log)
        self.assertNotIn("helper", log)
        self.assertIn("pip-check", log)
        self.assertIn("runtime-validation", log)
        self.assertIn("Installed ArcANE RL environment", result.stdout)

    def test_existing_python_312_environment_is_reused(self):
        self._existing_environment()
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("conda-create", self._log())

    def test_failed_activation_does_not_install_into_another_environment(self):
        result = self._run(FAKE_ACTIVATION_FAILURE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("activation did not select", result.stdout)
        self.assertNotIn("helper", self._log())

    def test_dependency_install_failure_stops_installation(self):
        result = self._run(FAKE_INSTALL_FAILURE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("flash_attn-2.8.1", self._log())
        self.assertNotIn("pip-check", self._log())
        self.assertNotIn("Installed ArcANE RL environment", result.stdout)

    def test_flash_attention_wheel_matches_torch_abi_with_hash(self):
        hashes = {
            "TRUE": "55331d797171973c8babf2b5e5fce5f78859b1cd298d54a136e8994147fe9e95",
            "FALSE": "15db5bb6524dcbf292c3c116aa4c2fa823b80abff0eb5bc58454107bca1ba0c2",
        }
        for abi, digest in hashes.items():
            with self.subTest(abi=abi):
                result = self._run(FAKE_CXX11_ABI=abi)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                log = self._log()
                self.assertIn(
                    f"cxx11abi{abi}-cp312-cp312-linux_x86_64.whl#sha256={digest}", log,
                )
                self.assertIn("--no-deps", log.splitlines())

    def test_unknown_torch_abi_is_rejected(self):
        result = self._run(FAKE_CXX11_ABI="unknown")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported PyTorch CXX11 ABI", result.stdout)
        self.assertNotIn("flash_attn-2.8.1", self._log())

    def test_constraints_keep_the_compatible_major_versions(self):
        requirements = set((RL_ROOT / "constraints.txt").read_text().splitlines())
        self.assertTrue({
            "vllm==0.11.0", "torch==2.8.0", "transformers==4.57.6",
            "peft==0.17.1", "numpy==1.26.4", "opencv-python-headless==4.11.0.86",
            "huggingface-hub>=0.34.0,<1.0",
        } <= requirements)

    def test_dependency_conflict_prevents_success(self):
        result = self._run(FAKE_PIP_CHECK_FAILURE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("runtime-validation", self._log())
        self.assertNotIn("Installed ArcANE RL environment", result.stdout)

    def test_runtime_failure_prevents_success(self):
        result = self._run(FAKE_RUNTIME_FAILURE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("runtime-validation", self._log())
        self.assertNotIn("Installed ArcANE RL environment", result.stdout)

    def test_embedded_runtime_validation_has_valid_python_syntax(self):
        source = INSTALLER.read_text().split("python - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        ast.parse(source)

    def test_launcher_defaults_disable_thinking_and_load_bf16_without_shm(self):
        result = self._local_launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = self._log().splitlines()
        for expected in (
            "+data.apply_chat_template_kwargs.enable_thinking=False",
            "actor_rollout_ref.model.use_shm=False",
            "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16",
            "actor_rollout_ref.ref.fsdp_config.model_dtype=bf16",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
            "trainer.n_gpus_per_node=2",
            "reward-mode=remote",
        ):
            self.assertIn(expected, args)
        self.assertNotIn("scripts/download_hf_model.py", args)
        self.assertNotIn("scripts/download_hf_data.py", args)

    def test_launcher_keeps_hardware_and_hydra_overrides(self):
        result = self._local_launch(
            N_GPUS="4", ROLLOUT_TP="2", MODEL_USE_SHM="True",
            args=("actor_rollout_ref.actor.optim.lr=2e-5",),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = self._log().splitlines()
        self.assertIn("trainer.n_gpus_per_node=4", args)
        self.assertIn("actor_rollout_ref.rollout.tensor_model_parallel_size=2", args)
        self.assertIn("actor_rollout_ref.model.use_shm=True", args)
        self.assertGreater(
            args.index("actor_rollout_ref.actor.optim.lr=2e-5"),
            args.index("actor_rollout_ref.actor.optim.lr=1e-5"),
        )

    def test_smoke_retains_explicitly_non_reproductive_heuristic_mode(self):
        result = self._local_launch(SMOKE="1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("reward-mode=heuristic", self._log().splitlines())
        self.assertIn("trainer.total_training_steps=2", self._log().splitlines())


if __name__ == "__main__":
    unittest.main()
