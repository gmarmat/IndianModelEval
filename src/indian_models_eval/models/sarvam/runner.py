#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
#
# IndianModelsEval — Sarvam-Translate Runner (decoder-only, Gemma3 chat-template)
# Copyright (C) 2026 IndianModelsEval contributors
#
# This file is part of the Sarvam-Translate evaluation wrapper,
# licensed under GPL-3.0 due to Sarvam-Translate's GPL-3.0 license.
#
# This module is NEVER imported by the core harness (MIT).
# It is called exclusively via subprocess from eval_runner.py.
# See: docs/plans/2026-03-17-security-safety-audit.md → Guardrail L1
#
# Usage:
#   python runner.py <input.json> <output.json>
#
# Input JSON schema:
#   { "sentences": [...], "src_lang": "eng_Latn", "tgt_lang": "hin_Deva",
#     "model_path": "...", "batch_size": 8, "precision": "bf16" }
#
# Output JSON schema:
#   { "translations": [...], "model_revision": "..." }

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# FLORES-200 code -> English language name used in Sarvam's system prompt.
# Sarvam-Translate's HF card supports all 22 official Indian languages + English.
_SARVAM_LANG_NAMES = {
    "eng_Latn": "English",
    "asm_Beng": "Assamese",
    "ben_Beng": "Bengali",
    "brx_Deva": "Bodo",
    "doi_Deva": "Dogri",
    "gom_Deva": "Konkani",
    "guj_Gujr": "Gujarati",
    "hin_Deva": "Hindi",
    "kan_Knda": "Kannada",
    "kas_Arab": "Kashmiri",
    "mai_Deva": "Maithili",
    "mal_Mlym": "Malayalam",
    "mar_Deva": "Marathi",
    "mni_Beng": "Manipuri",
    "npi_Deva": "Nepali",
    "ory_Orya": "Odia",
    "pan_Guru": "Punjabi",
    "san_Deva": "Sanskrit",
    "sat_Olck": "Santali",
    "snd_Arab": "Sindhi",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "urd_Arab": "Urdu",
}


def translate_sarvam(
    sentences: list[str],
    src_lang: str,
    tgt_lang: str,
    model_path: str,
    batch_size: int = 8,
    precision: str = "bf16",
    max_new_tokens: int = 256,
) -> list[str]:
    tgt_name = _SARVAM_LANG_NAMES.get(tgt_lang)
    if tgt_name is None:
        raise ValueError(f"Sarvam does not support target language: {tgt_lang}")
    # Source language is implicit in the input text; Sarvam infers it.
    if src_lang not in _SARVAM_LANG_NAMES:
        raise ValueError(f"Sarvam does not support source language: {src_lang}")

    logger.info("Loading Sarvam-Translate from %s (precision=%s)", model_path, precision)
    if precision == "bf16":
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Decoder-only batched generation requires left-padding so generated tokens
    # continue from the actual end of each prompt, not from trailing pads.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).cuda().eval()

    system_prompt = f"Translate the text below to {tgt_name}."
    results: list[str] = []
    total_batches = (len(sentences) + batch_size - 1) // batch_size

    with torch.inference_mode():
        for batch_idx, i in enumerate(range(0, len(sentences), batch_size)):
            batch = sentences[i : i + batch_size]
            prompts: list[str] = []
            for sent in batch:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": sent},
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(text)

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to("cuda")
            input_token_len = inputs["input_ids"].shape[1]

            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            # With left-padding, generated tokens start at the same index for every row.
            generated = output_ids[:, input_token_len:]
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            results.extend(s.strip() for s in decoded)

            if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == total_batches:
                logger.info("batch %d/%d done", batch_idx + 1, total_batches)

    return results


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.json> <output.json>", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    with input_path.open() as f:
        data = json.load(f)

    translations = translate_sarvam(
        sentences=data["sentences"],
        src_lang=data["src_lang"],
        tgt_lang=data["tgt_lang"],
        model_path=data["model_path"],
        batch_size=data.get("batch_size", 8),
        precision=data.get("precision", "bf16"),
        max_new_tokens=data.get("max_new_tokens", 256),
    )

    revision = None
    try:
        refs = Path(data["model_path"]) / "refs" / "main"
        if refs.exists():
            revision = refs.read_text().strip()
    except Exception:
        pass

    # ensure_ascii=True: round-trips safely through harness's _run_subprocess
    # which opens output JSON with the default locale encoding (cp1252 on Windows).
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"translations": translations, "model_revision": revision}, f)


if __name__ == "__main__":
    main()
