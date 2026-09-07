from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig, PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from arcane.config import TrainConfig


def _patch_peft_tp_sharding_compat() -> None:
    """Work around a peft<->transformers version mismatch on LoRA adapter loading.

    peft 0.19.x calls ``_maybe_shard_state_dict_for_tp`` whenever a LoRA adapter
    state dict is loaded while ``torch.distributed`` is initialized (e.g. resuming
    from a checkpoint under DDP). That function unconditionally imports
    ``EmbeddingParallel`` from ``transformers.integrations.tensor_parallel``, which
    is absent in the installed transformers. The sharding is only relevant for
    tensor-parallel models; we run plain DDP, so neutralize it when the symbol is
    missing instead of crashing on resume.
    """
    try:
        from transformers.integrations.tensor_parallel import EmbeddingParallel  # noqa: F401
    except Exception:
        try:
            import peft.utils.save_and_load as peft_save_and_load

            peft_save_and_load._maybe_shard_state_dict_for_tp = lambda *args, **kwargs: None
        except Exception:
            pass


_patch_peft_tp_sharding_compat()


def resolve_precision(mixed_precision: str) -> tuple[bool, bool]:
    if mixed_precision == "bf16":
        return True, False
    if mixed_precision == "fp16":
        return False, True
    if mixed_precision == "no":
        return False, False
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return True, False
    if torch.cuda.is_available():
        return False, True
    return False, False


def resolve_torch_dtype(torch_dtype: str) -> str | torch.dtype:
    if torch_dtype == "auto":
        return "auto"
    if torch_dtype == "bfloat16":
        return torch.bfloat16
    if torch_dtype == "float16":
        return torch.float16
    if torch_dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {torch_dtype}")


def hub_revision_kwargs(name_or_path: str, revision: str | None) -> dict[str, str]:
    """Return a Hub revision only for non-local model sources."""
    if revision is None or Path(name_or_path).expanduser().exists():
        return {}
    return {"revision": revision}


def load_tokenizer(config: TrainConfig) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        **hub_revision_kwargs(config.model_name_or_path, config.model_revision),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if config.assistant_only_loss:
        from trl.chat_template_utils import get_training_chat_template

        training_chat_template = get_training_chat_template(tokenizer)
        if training_chat_template is not None:
            tokenizer.chat_template = training_chat_template
    return tokenizer


def load_peft_adapter_config(
    model_name_or_path: str,
    revision: str | None = None,
) -> PeftConfig | None:
    """Return the PEFT config if model_name_or_path is a LoRA-style adapter, else None."""
    try:
        return PeftConfig.from_pretrained(
            model_name_or_path,
            **hub_revision_kwargs(model_name_or_path, revision),
        )
    except Exception:
        return None


def load_model(config: TrainConfig) -> AutoModelForCausalLM:
    model_kwargs = {
        "torch_dtype": resolve_torch_dtype(config.torch_dtype),
        "trust_remote_code": True,
    }
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation

    source_revision_kwargs = hub_revision_kwargs(
        config.model_name_or_path,
        config.model_revision,
    )
    adapter_config = load_peft_adapter_config(
        config.model_name_or_path,
        config.model_revision,
    )
    if adapter_config is not None and adapter_config.base_model_name_or_path:
        # model_name_or_path is an adapter-only checkpoint (e.g. the LoRA SFT model).
        # Load its base model, apply the adapter, and merge it into a clean full model
        # so downstream training starts from the merged SFT weights. For DPO this also
        # makes the (adapter-disabled) reference the merged SFT model, and lets a single
        # fresh adapter be added on top instead of stacking a second one.
        base_model = AutoModelForCausalLM.from_pretrained(
            adapter_config.base_model_name_or_path,
            **model_kwargs,
            **hub_revision_kwargs(
                adapter_config.base_model_name_or_path,
                adapter_config.revision,
            ),
        )
        peft_model = PeftModel.from_pretrained(
            base_model,
            config.model_name_or_path,
            **source_revision_kwargs,
        )
        model = peft_model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            **model_kwargs,
            **source_revision_kwargs,
        )

    if config.gradient_checkpointing:
        model.config.use_cache = False
    return model


def build_peft_config(config: TrainConfig) -> LoraConfig | None:
    if not config.use_peft:
        return None

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        revision=(
            config.model_revision
            if hub_revision_kwargs(config.model_name_or_path, config.model_revision)
            else None
        ),
    )
