"""
Dataset loader for FLORES-200 and IN22-Gen.
Downloads and caches to data/ directory. Validates sentence count alignment.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import NamedTuple

from datasets import load_dataset

from .config import FLORES_TO_IN22

# Some IndicTrans2/internal codes differ from flores_plus dataset config names.
# Map internal code → flores_plus code before querying the dataset.
FLORES_PLUS_CODE_MAP: dict[str, str] = {
    "doi_Deva": "dgo_Deva",  # Dogri: IndicTrans2 uses doi_Deva, flores_plus uses dgo_Deva
}

logger = logging.getLogger(__name__)


class SentencePair(NamedTuple):
    source: str
    reference: str


def load_flores200(
    src_lang: str,
    tgt_lang: str,
    cache_dir: Path = Path("data"),
    split: str = "dev",
    n: int | None = None,
) -> list[SentencePair]:
    """
    Load FLORES-200 sentence pairs for (src_lang, tgt_lang).

    Args:
        src_lang: FLORES-200 code e.g. "eng_Latn"
        tgt_lang: FLORES-200 code e.g. "hin_Deva"
        cache_dir: local cache directory (gitignored)
        split: "dev" (997 sentences) — flores_plus only has dev split
        n: if set, return only first n pairs (for dry-run)

    Returns:
        List of (source, reference) sentence pairs.
    """
    cache_path = cache_dir / "flores200"
    cache_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading FLORES-200 %s → %s (%s)", src_lang, tgt_lang, split)

    # Apply code overrides: some internal codes differ from flores_plus config names
    flores_src = FLORES_PLUS_CODE_MAP.get(src_lang, src_lang)
    flores_tgt = FLORES_PLUS_CODE_MAP.get(tgt_lang, tgt_lang)
    if flores_src != src_lang:
        logger.info("flores_plus code override: %s → %s", src_lang, flores_src)
    if flores_tgt != tgt_lang:
        logger.info("flores_plus code override: %s → %s", tgt_lang, flores_tgt)

    # openlanguagedata/flores_plus is the modern Parquet-based FLORES-200
    # compatible with datasets v3+ (facebook/flores uses deprecated loading scripts)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    src_dataset = load_dataset(
        "openlanguagedata/flores_plus",
        flores_src,
        split=split,
        cache_dir=str(cache_path),
        token=hf_token or None,
    )
    tgt_dataset = load_dataset(
        "openlanguagedata/flores_plus",
        flores_tgt,
        split=split,
        cache_dir=str(cache_path),
        token=hf_token or None,
    )

    src_sentences: list[str] = src_dataset["text"]
    tgt_sentences: list[str] = tgt_dataset["text"]

    if len(src_sentences) != len(tgt_sentences):
        raise ValueError(
            f"Sentence count mismatch: {src_lang}={len(src_sentences)}, "
            f"{tgt_lang}={len(tgt_sentences)}"
        )

    pairs = [SentencePair(source=s, reference=t) for s, t in zip(src_sentences, tgt_sentences)]

    if n is not None:
        pairs = pairs[:n]

    logger.info("Loaded %d sentence pairs", len(pairs))
    return pairs


def load_in22_gen(
    src_lang: str,
    tgt_lang: str,
    cache_dir: Path = Path("data"),
    n: int | None = None,
) -> list[SentencePair]:
    """
    Load IN22-Gen sentence pairs for (src_lang, tgt_lang).

    IN22-Gen uses short language codes ("hin", "ben") unlike FLORES-200.
    Maps automatically from FLORES-200 codes via FLORES_TO_IN22.

    Args:
        src_lang: FLORES-200 code e.g. "eng_Latn"
        tgt_lang: FLORES-200 code e.g. "hin_Deva"
        cache_dir: local cache directory
        n: if set, return only first n pairs

    Returns:
        List of (source, reference) sentence pairs.
    """
    src_code = FLORES_TO_IN22.get(src_lang)
    tgt_code = FLORES_TO_IN22.get(tgt_lang)

    if src_code is None:
        raise ValueError(f"Language {src_lang!r} not available in IN22-Gen")
    if tgt_code is None:
        raise ValueError(f"Language {tgt_lang!r} not available in IN22-Gen")

    cache_path = cache_dir / "in22_gen"
    cache_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading IN22-Gen %s → %s", src_lang, tgt_lang)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    dataset = load_dataset(
        "ai4bharat/IN22-Gen",
        split="test",
        cache_dir=str(cache_path),
        token=hf_token or None,
    )

    # IN22-Gen has all languages as columns
    if src_code not in dataset.column_names:
        raise ValueError(f"Column {src_code!r} not found in IN22-Gen. Available: {dataset.column_names}")
    if tgt_code not in dataset.column_names:
        raise ValueError(f"Column {tgt_code!r} not found in IN22-Gen. Available: {dataset.column_names}")

    src_sentences: list[str] = dataset[src_code]
    tgt_sentences: list[str] = dataset[tgt_code]

    pairs = [SentencePair(source=s, reference=t) for s, t in zip(src_sentences, tgt_sentences)]

    if n is not None:
        pairs = pairs[:n]

    logger.info("Loaded %d IN22-Gen sentence pairs", len(pairs))
    return pairs


def load_pairs(
    src_lang: str,
    tgt_lang: str,
    dataset: str,
    cache_dir: Path = Path("data"),
    n: int | None = None,
) -> list[SentencePair]:
    """Unified loader — dispatches to the right dataset loader."""
    if dataset == "flores200":
        return load_flores200(src_lang, tgt_lang, cache_dir=cache_dir, n=n)
    elif dataset == "in22_gen":
        return load_in22_gen(src_lang, tgt_lang, cache_dir=cache_dir, n=n)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose from: flores200, in22_gen")
