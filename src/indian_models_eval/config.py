"""
Pydantic config models, app settings, and language code mappings.
All secret values come from environment variables — never from config files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Language code maps
# ---------------------------------------------------------------------------

# FLORES-200 BCP-47 codes → human-readable name
FLORES_LANG_CODES: dict[str, str] = {
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
    "sat_Olck": "Santhali",
    "snd_Arab": "Sindhi",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "urd_Arab": "Urdu",
}

# Languages where COMET is unreliable — attach reliability warning in results
COMET_UNRELIABLE_LANGS: frozenset[str] = frozenset(
    ["mni_Beng", "brx_Deva", "sat_Olck", "snd_Arab", "kas_Arab"]
)

# FLORES-200 code → IN22-Gen column name
FLORES_TO_IN22: dict[str, str] = {
    "eng_Latn": "eng",
    "asm_Beng": "asm",
    "ben_Beng": "ben",
    "brx_Deva": "brx",
    "doi_Deva": "doi",
    "gom_Deva": "kok",
    "guj_Gujr": "guj",
    "hin_Deva": "hin",
    "kan_Knda": "kan",
    "kas_Arab": "kas",
    "mai_Deva": "mai",
    "mal_Mlym": "mal",
    "mar_Deva": "mar",
    "mni_Beng": "mni",
    "npi_Deva": "npi",
    "ory_Orya": "ory",
    "pan_Guru": "pan",
    "san_Deva": "san",
    "sat_Olck": "sat",
    "snd_Arab": "snd",
    "tam_Taml": "tam",
    "tel_Telu": "tel",
    "urd_Arab": "urd",
}

# Viability threshold — below this chrF++ = "not viable"
VIABILITY_THRESHOLD: float = 40.0


# ---------------------------------------------------------------------------
# App settings (from .env)
# ---------------------------------------------------------------------------

class AppSettings(BaseSettings):
    """Reads secrets from .env — never hardcode values here."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    hf_token: str = Field(default="", description="HuggingFace API token")
    sarvam_api_key: str = Field(default="", description="Sarvam API key (only for --use-api)")


# ---------------------------------------------------------------------------
# Eval config (passed to eval runner)
# ---------------------------------------------------------------------------

DatasetName = Literal["flores200", "in22_gen"]


class EvalConfig(BaseModel):
    """Full configuration for a single evaluation run.
    Safe to serialize and commit — contains no secrets.
    """

    model_id: str
    datasets: list[DatasetName] = Field(default_factory=lambda: ["flores200"])
    language_pairs: list[tuple[str, str]] | None = Field(
        default=None,
        description="List of (src_lang, tgt_lang) FLORES-200 codes. None = all claimed pairs.",
    )

    # Execution
    dry_run: bool = False
    dry_run_sentences: int = 5
    batch_size: int = 8
    precision: Literal["fp16", "fp32", "int8"] = "fp16"
    seed: int = 42
    device: str = "cuda"
    max_new_tokens: int = 256
    use_api: bool = False

    # Cost control
    max_api_cost_usd: float = 20.0

    # Paths
    registry_path: Path = Path("model_registry.yaml")
    results_dir: Path = Path("results")
    data_cache_dir: Path = Path("data")

    @field_validator("language_pairs")
    @classmethod
    def validate_lang_codes(cls, v: list[tuple[str, str]] | None) -> list[tuple[str, str]] | None:
        if v is None:
            return v
        valid = set(FLORES_LANG_CODES.keys())
        for src, tgt in v:
            if src not in valid:
                raise ValueError(f"Unknown FLORES-200 lang code: {src!r}")
            if tgt not in valid:
                raise ValueError(f"Unknown FLORES-200 lang code: {tgt!r}")
        return v
