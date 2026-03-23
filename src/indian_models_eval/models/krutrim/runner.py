"""
Krutrim Translate runner.
Source: https://github.com/ola-krutrim/KrutrimTranslate
License: Krutrim Community License Agreement v1.0
NON-COMMERCIAL use only — check license before publishing results commercially.

Uses CTranslate2 for inference (fast, quantized, no MSVC needed).
Language codes: same FLORES-200 format as the rest of this project.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Supported language pairs (FLORES-200 codes)
KRUTRIM_SUPPORTED_LANGS = {
    "ben_Beng", "hin_Deva", "kan_Knda", "mar_Deva",
    "mal_Mlym", "guj_Gujr", "pan_Guru", "tel_Telu", "tam_Taml",
}

_model_cache: dict[str, object] = {}


def _load(model_path: str, device: str) -> object:
    if model_path in _model_cache:
        return _model_cache[model_path]

    # Add engine dir to path so relative imports inside engine.py work
    engine_dir = str(Path(__file__).parent / "engine")
    if engine_dir not in sys.path:
        sys.path.insert(0, str(Path(__file__).parent))

    logger.info("Loading Krutrim Translate from %s", model_path)

    from .engine.engine import Model
    model = Model(model_path, device=device, model_type="ctranslate2")
    _model_cache[model_path] = model
    return model


def _get_checkpoint_dir(model_path: str, src_lang: str) -> str:
    """
    Krutrim ships two CT2 checkpoints inside the model folder:
      ct_model_english_indic/  — for eng_Latn → Indic
      ct_model_indic_english/  — for Indic → eng_Latn
    """
    base = Path(model_path)
    if src_lang == "eng_Latn":
        ckpt = base / "ct_model_english_indic"
    else:
        ckpt = base / "ct_model_indic_english"

    if not ckpt.exists():
        raise FileNotFoundError(
            f"Krutrim checkpoint not found: {ckpt}\n"
            f"Download with: snapshot_download('krutrim-ai-labs/KrutrimTranslate', local_dir='{base}')"
        )
    return str(ckpt)


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
    Translate using Krutrim Translate (CTranslate2).
    Krutrim Community License — non-commercial only.
    """
    if src_lang != "eng_Latn" and tgt_lang != "eng_Latn":
        raise ValueError(
            f"Krutrim Translate only supports English↔Indic pairs. "
            f"Got: {src_lang} → {tgt_lang}"
        )

    ckpt_dir = _get_checkpoint_dir(model_path, src_lang)
    cache_key = ckpt_dir
    model = _load(cache_key, device)

    beam_len = (extra_config or {}).get("beam_len", 3)

    results: list[str] = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i: i + batch_size]
        translated = model.batch_translate(batch, src_lang=src_lang, tgt_lang=tgt_lang, beam_len=beam_len)
        results.extend(translated)

    return results
