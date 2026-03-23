"""
Results writer — saves one JSON file per eval run to results/.
Includes full reproducibility metadata: model revision hash, seed, hardware, config.
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .claims_verifier import ClaimsVerdict
from .config import EvalConfig
from .eval_runner import TranslationResult
from .scoring_pipeline import ScoreResult
from .test_planner import ModelEntry

logger = logging.getLogger(__name__)


def _get_hf_revision(model_path: str | Path) -> str | None:
    """Read the HuggingFace model revision hash from the local snapshot."""
    try:
        refs_path = Path(model_path) / "refs" / "main"
        if refs_path.exists():
            return refs_path.read_text().strip()
        # Try reading from config.json commit_hash field
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with config_path.open() as f:
                data = json.load(f)
                return data.get("_commit_hash")
    except Exception:
        pass
    return None


def _get_git_hash() -> str | None:
    """Get current git commit hash for reproducibility."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


class ResultsWriter:
    def __init__(self, results_dir: Path = Path("results")) -> None:
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        model: ModelEntry,
        config: EvalConfig,
        translation_results: list[TranslationResult],
        scores: list[ScoreResult],
        verdict: ClaimsVerdict,
    ) -> Path:
        """
        Write a single results JSON file.

        Filename: results/{model_id}_{dataset}_{timestamp}.json

        Returns:
            Path to the written file.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        datasets_str = "+".join(sorted(set(s.dataset for s in scores)))
        filename = f"{model.id}_{datasets_str}_{timestamp}.json"
        output_path = self.results_dir / filename

        # Compute aggregate stats
        aggregate = self._aggregate(scores)

        # Build full results document
        doc = {
            "schema_version": "1.0",
            "harness_version": __version__,
            "timestamp": timestamp,

            "model": {
                "id": model.id,
                "display_name": model.display_name,
                "hf_id": model.hf_id,
                "license": model.license,
                "revision_hash": _get_hf_revision(model.local_path),
            },

            "eval_config": {
                "datasets": config.datasets,
                "seed": config.seed,
                "batch_size": config.batch_size,
                "precision": config.precision,
                "max_new_tokens": config.max_new_tokens,
                "dry_run": config.dry_run,
                "viability_threshold_chrf": 40.0,
            },

            "reproducibility": {
                "harness_git_hash": _get_git_hash(),
                "model_revision_hash": _get_hf_revision(model.local_path),
                "seed": config.seed,
            },

            "hardware": translation_results[0].hardware if translation_results else {},

            "aggregate": aggregate,

            "claims_verification": verdict.to_dict(),

            "per_language_pair": [s.to_dict() for s in scores],

            "latency": {
                "per_pair": [
                    {
                        "src_lang": r.src_lang,
                        "tgt_lang": r.tgt_lang,
                        "dataset": r.dataset,
                        "vram_peak_mb": r.vram_peak_mb,
                        "latency_seconds": r.latency_seconds,
                        "sentences_per_second": r.sentences_per_second,
                        "n_sentences": len(r.sources),
                    }
                    for r in translation_results
                ]
            },
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

        logger.info("Results written to %s", output_path)
        return output_path

    def _aggregate(self, scores: list[ScoreResult]) -> dict:
        if not scores:
            return {}

        chrf_values = [s.chrf for s in scores]
        bleu_values = [s.bleu for s in scores]
        comet_values = [s.comet for s in scores if s.comet is not None]
        reliable_comet = [s.comet for s in scores if s.comet is not None and s.comet_reliable]

        viable = [s for s in scores if s.viable]

        return {
            "n_pairs": len(scores),
            "viable_pairs": len(viable),
            "chrf_mean": round(sum(chrf_values) / len(chrf_values), 2),
            "chrf_min": round(min(chrf_values), 2),
            "chrf_max": round(max(chrf_values), 2),
            "bleu_mean": round(sum(bleu_values) / len(bleu_values), 2),
            "comet_mean": round(sum(comet_values) / len(comet_values), 4) if comet_values else None,
            "comet_mean_reliable_only": (
                round(sum(reliable_comet) / len(reliable_comet), 4) if reliable_comet else None
            ),
            "note": "Aggregate scores hide per-language variance. See per_language_pair for full breakdown.",
        }
