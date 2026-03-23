"""
NLLB (No Language Left Behind) runner.
License: CC-BY-NC-4.0 — NON-COMMERCIAL RESEARCH ONLY.
Never use in commercial products.
"""
from __future__ import annotations

import logging

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

_model_cache: dict[str, tuple] = {}


def _nllb_lang_code(flores_code: str) -> str:
    """
    Convert FLORES-200 code to NLLB language token.
    NLLB uses the same BCP-47 format so most codes work directly.
    A few require remapping.
    """
    # NLLB uses slightly different codes for some languages
    remaps = {
        "ory_Orya": "ory_Orya",   # same
        "gom_Deva": "gom_Deva",   # same
        "sat_Olck": "sat_Olck",   # same
    }
    return remaps.get(flores_code, flores_code)


def _load(model_path: str, precision: str, device: str) -> tuple:
    if model_path in _model_cache:
        return _model_cache[model_path]

    logger.info("Loading NLLB from %s", model_path)
    dtype = torch.float16 if precision == "fp16" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    _model_cache[model_path] = (model, tokenizer)
    return model, tokenizer


def translate(
    sentences: list[str],
    src_lang: str,
    tgt_lang: str,
    model_path: str,
    batch_size: int = 32,
    max_new_tokens: int = 256,
    precision: str = "fp16",
    device: str = "cuda",
    extra_config: dict | None = None,
) -> list[str]:
    """
    Translate using NLLB.
    CC-BY-NC-4.0 — non-commercial research only.
    """
    model, tokenizer = _load(model_path, precision, device)

    nllb_src = _nllb_lang_code(src_lang)
    nllb_tgt = _nllb_lang_code(tgt_lang)

    translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=nllb_src,
        tgt_lang=nllb_tgt,
        device=0 if device == "cuda" else -1,
        torch_dtype=torch.float16 if precision == "fp16" else torch.float32,
    )

    results: list[str] = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        outputs = translator(batch, max_length=max_new_tokens, batch_size=batch_size)
        results.extend(item["translation_text"] for item in outputs)

    return results
