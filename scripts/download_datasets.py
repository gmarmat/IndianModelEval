"""
Dataset downloader. Downloads FLORES-200 and IN22-Gen to data/.
Tracks what's already downloaded in data/manifest.json — skips on re-run.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

FLORES_LANGS = [
    "eng_Latn", "hin_Deva", "ben_Beng", "tam_Taml", "tel_Telu",
    "mar_Deva", "guj_Gujr", "kan_Knda", "mal_Mlym", "pan_Guru",
    "ory_Orya", "urd_Arab", "asm_Beng", "mai_Deva", "npi_Deva",
    # brx_Deva (Bodo) excluded — not available in openlanguagedata/flores_plus
    "mni_Beng", "sat_Olck",
]

MANIFEST_PATH = Path("data/manifest.json")


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"flores200": [], "in22_gen": False}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def download_all(data_dir: Path = Path("data"), force: bool = False) -> str:
    from datasets import load_dataset

    messages = []
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    # FLORES-200
    flores_dir = data_dir / "flores200"
    flores_dir.mkdir(exist_ok=True)
    messages.append("Downloading FLORES-200...")

    already_have = set(manifest.get("flores200", []))
    newly_downloaded = []

    for lang in FLORES_LANGS:
        if not force and lang in already_have:
            messages.append(f"  ✅ {lang} (cached)")
            continue
        try:
            load_dataset(
                "openlanguagedata/flores_plus",
                lang,
                split="devtest",
                cache_dir=str(flores_dir),
                token=hf_token or None,
            )
            messages.append(f"  ✅ {lang}")
            newly_downloaded.append(lang)
        except Exception as e:
            messages.append(f"  ❌ {lang}: {e}")

    manifest["flores200"] = sorted(already_have | set(newly_downloaded))
    _save_manifest(manifest)

    # IN22-Gen
    in22_dir = data_dir / "in22_gen"
    in22_dir.mkdir(exist_ok=True)
    messages.append("\nDownloading IN22-Gen...")

    if not force and manifest.get("in22_gen"):
        messages.append("  ✅ IN22-Gen (cached)")
    else:
        try:
            load_dataset(
                "ai4bharat/IN22-Gen",
                split="test",
                cache_dir=str(in22_dir),
                token=hf_token or None,
            )
            messages.append("  ✅ IN22-Gen")
            manifest["in22_gen"] = True
            _save_manifest(manifest)
        except Exception as e:
            messages.append(f"  ❌ IN22-Gen: {e}")

    messages.append("\nDone.")
    result = "\n".join(messages)
    print(result)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    force = "--force" in sys.argv
    download_all(force=force)
