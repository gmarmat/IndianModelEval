#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
#
# IndianModelsEval — Sarvam-Translate Runner
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
#     "model_path": "...", "batch_size": 8, "precision": "fp16" }
#
# Output JSON schema:
#   { "translations": [...], "model_revision": "..." }

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SARVAM_LANG_MAP = {
    "eng_Latn": "en",
    "ben_Beng": "bn",
    "guj_Gujr": "gu",
    "hin_Deva": "hi",
    "kan_Knda": "kn",
    "mal_Mlym": "ml",
    "mar_Deva": "mr",
    "ory_Orya": "or",
    "pan_Guru": "pa",
    "tam_Taml": "ta",
    "tel_Telu": "te",
}


def translate_sarvam(
    sentences: list[str],
    src_lang: str,
    tgt_lang: str,
    model_path: str,
    batch_size: int = 8,
    precision: str = "fp16",
) -> list[str]:
    src_code = _SARVAM_LANG_MAP.get(src_lang)
    tgt_code = _SARVAM_LANG_MAP.get(tgt_lang)

    if src_code is None:
        raise ValueError(f"Sarvam does not support source language: {src_lang}")
    if tgt_code is None:
        raise ValueError(f"Sarvam does not support target language: {tgt_lang}")

    logger.info("Loading Sarvam-Translate from %s", model_path)
    dtype = torch.float16 if precision == "fp16" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).cuda().eval()

    results: list[str] = []

    with torch.inference_mode():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]

            # Sarvam uses forced_bos_token_id for target language
            tgt_lang_id = tokenizer.convert_tokens_to_ids(f"<2{tgt_code}>")

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to("cuda")

            output_ids = model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                num_beams=5,
                max_length=256,
            )

            decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            results.extend(decoded)

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
        precision=data.get("precision", "fp16"),
    )

    # Get model revision hash if available
    revision = None
    try:
        refs = Path(data["model_path"]) / "refs" / "main"
        if refs.exists():
            revision = refs.read_text().strip()
    except Exception:
        pass

    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"translations": translations, "model_revision": revision}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
