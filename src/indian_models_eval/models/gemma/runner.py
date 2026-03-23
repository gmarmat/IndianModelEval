"""
Gemma decoder-only runner. Uses prompt-based translation (no fine-tuning).
License: Gemma Terms of Use (permissive for research).
"""
from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...config import FLORES_LANG_CODES

logger = logging.getLogger(__name__)

_model_cache: dict[str, tuple] = {}

DEFAULT_PROMPT_TEMPLATE = (
    "Translate the following {src_lang_name} text to {tgt_lang_name}. "
    "Output only the translation, nothing else.\n\n"
    "Text: {source}\n\n"
    "Translation:"
)


def _load(model_path: str, precision: str, device: str) -> tuple:
    if model_path in _model_cache:
        return _model_cache[model_path]

    logger.info("Loading Gemma from %s", model_path)
    dtype = torch.float16 if precision == "fp16" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()

    _model_cache[model_path] = (model, tokenizer)
    return model, tokenizer


def translate(
    sentences: list[str],
    src_lang: str,
    tgt_lang: str,
    model_path: str,
    batch_size: int = 4,
    max_new_tokens: int = 512,
    precision: str = "fp16",
    device: str = "cuda",
    extra_config: dict | None = None,
) -> list[str]:
    """Translate using Gemma via prompt-based generation."""
    model, tokenizer = _load(model_path, precision, device)

    src_name = FLORES_LANG_CODES.get(src_lang, src_lang)
    tgt_name = FLORES_LANG_CODES.get(tgt_lang, tgt_lang)

    prompt_template = (extra_config or {}).get("prompt_template", DEFAULT_PROMPT_TEMPLATE)

    prompts = [
        prompt_template.format(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            src_lang_name=src_name,
            tgt_lang_name=tgt_name,
            source=sentence,
        )
        for sentence in sentences
    ]

    results: list[str] = []

    with torch.inference_mode():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_sources = sentences[i : i + batch_size]

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            input_len = inputs["input_ids"].shape[1]

            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Decode only the newly generated tokens
            for j, ids in enumerate(output_ids):
                new_tokens = ids[input_len:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                results.append(text)

    return results
