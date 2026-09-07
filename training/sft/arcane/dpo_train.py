from __future__ import annotations

import argparse

from arcane.config import TrainConfig, apply_overrides, finalize_run_config, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a chat model with TRL DPO on preference data.")
    parser.add_argument("--config", default=None, help="Path to an OmegaConf YAML training config.")

    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--torch-dtype", choices=("auto", "bfloat16", "float16", "float32"), default=None)
    parser.add_argument("--attn-implementation", default=None)

    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default=None)
    parser.add_argument("--dataset-data-files", default=None)
    parser.add_argument("--dataset-num-proc", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-prompt-length", type=int, default=None)

    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--report-to", default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--mixed-precision",
        choices=("auto", "bf16", "fp16", "no"),
        default=None,
        help="Training precision. auto uses bf16 when supported, then fp16 on CUDA, else fp32.",
    )

    parser.add_argument("--dpo-beta", type=float, default=None)
    parser.add_argument("--dpo-loss-type", nargs="+", default=None,
                        help="One loss type, or several to combine (e.g. sigmoid sft) weighted by --dpo-loss-weights.")
    parser.add_argument("--dpo-loss-weights", type=float, nargs="+", default=None,
                        help="Weights matching --dpo-loss-type when combining multiple losses (e.g. 1.0 0.5).")
    parser.add_argument("--dpo-label-smoothing", type=float, default=None)
    parser.add_argument("--dpo-ld-alpha", type=float, default=None,
                        help="LD-DPO length-desensitization alpha in [0,1]; omit to disable.")
    parser.add_argument("--dpo-precompute-ref-log-probs", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dpo-precompute-ref-batch-size", type=int, default=None)
    parser.add_argument("--dpo-truncation-mode", choices=("keep_start", "keep_end"), default=None)

    parser.add_argument("--use-peft", "--peft", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)

    parser.add_argument("--eval-dataset-name", default=None)
    parser.add_argument("--eval-dataset-revision", default=None)
    parser.add_argument("--eval-dataset-config", default=None)
    parser.add_argument("--eval-dataset-split", default=None)
    parser.add_argument("--eval-dataset-data-files", default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--trainer-eval-strategy", choices=("no", "steps", "epoch"), default=None)
    parser.add_argument("--trainer-eval-steps", type=int, default=None)
    parser.add_argument("--trainer-eval-delay", type=float, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-accumulation-steps", type=int, default=None)
    parser.add_argument("--eval-on-start", action=argparse.BooleanOptionalAction, default=None)

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    config = load_config(args.config)
    overrides = vars(args).copy()
    overrides.pop("config")
    return apply_overrides(config, overrides)


def main() -> None:
    args = parse_args()
    config = finalize_run_config(build_config(args), task="dpo")
    from arcane.dpo_trainer import run_training

    run_training(config)


if __name__ == "__main__":
    main()
