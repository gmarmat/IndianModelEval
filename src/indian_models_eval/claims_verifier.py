"""
Claims verifier — compares a model's claimed language support vs measured viable pairs.
A pair is "viable" if chrF++ >= VIABILITY_THRESHOLD (40.0).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import FLORES_LANG_CODES, VIABILITY_THRESHOLD
from .scoring_pipeline import ScoreResult
from .test_planner import ModelEntry


@dataclass
class ClaimsVerdict:
    model_id: str
    model_display_name: str
    claimed_pairs: int
    viable_pairs: int
    failed_pairs: list[dict]
    viability_threshold: float = VIABILITY_THRESHOLD

    @property
    def tested_pairs(self) -> int:
        """Pairs that were actually evaluated (viable + below threshold). Excludes not_evaluated."""
        not_evaluated = sum(1 for f in self.failed_pairs if f.get("reason") == "not_evaluated")
        return self.claimed_pairs - not_evaluated

    @property
    def viability_rate(self) -> float:
        if self.tested_pairs == 0:
            return 0.0
        return self.viable_pairs / self.tested_pairs

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_display_name": self.model_display_name,
            "claimed_pairs": self.claimed_pairs,
            "tested_pairs": self.tested_pairs,
            "viable_pairs": self.viable_pairs,
            "failed_pairs": self.failed_pairs,
            "viability_rate": round(self.viability_rate, 3),
            "viability_threshold_chrf": self.viability_threshold,
            "verdict": self._verdict_label(),
        }

    def _verdict_label(self) -> str:
        if self.tested_pairs == 0:
            return "not_evaluated"
        rate = self.viability_rate
        if rate >= 0.9:
            return "claims_verified"
        elif rate >= 0.7:
            return "mostly_verified"
        elif rate >= 0.5:
            return "partially_verified"
        else:
            return "claims_unverified"


class ClaimsVerifier:
    def __init__(self, model: ModelEntry) -> None:
        self._model = model

    def verify(self, scores: list[ScoreResult]) -> ClaimsVerdict:
        """
        Compare scored results against the model's claimed language pairs.

        Args:
            scores: list of ScoreResult from ScoringPipeline

        Returns:
            ClaimsVerdict with summary and failed pairs.
        """
        # Index scores by (src, tgt, dataset)
        score_index: dict[tuple[str, str, str], ScoreResult] = {
            (s.src_lang, s.tgt_lang, s.dataset): s for s in scores
        }

        claimed = self._model.claimed_pairs
        viable_count = 0
        failed: list[dict] = []

        # Deduplicate across datasets — a pair is viable if ANY dataset score passes
        pair_viability: dict[tuple[str, str], bool] = {}
        for score in scores:
            key = (score.src_lang, score.tgt_lang)
            if key not in pair_viability:
                pair_viability[key] = False
            if score.viable:
                pair_viability[key] = True

        for pair in claimed:
            key = (pair.src, pair.tgt)
            src_name = FLORES_LANG_CODES.get(pair.src, pair.src)
            tgt_name = FLORES_LANG_CODES.get(pair.tgt, pair.tgt)

            if key not in pair_viability:
                # Pair was claimed but not evaluated
                failed.append({
                    "src_lang": pair.src,
                    "tgt_lang": pair.tgt,
                    "src_lang_name": src_name,
                    "tgt_lang_name": tgt_name,
                    "reason": "not_evaluated",
                    "chrf": None,
                })
            elif pair_viability[key]:
                viable_count += 1
            else:
                # Find best score for this pair to report
                best_chrf = max(
                    (s.chrf for s in scores if s.src_lang == pair.src and s.tgt_lang == pair.tgt),
                    default=0.0,
                )
                failed.append({
                    "src_lang": pair.src,
                    "tgt_lang": pair.tgt,
                    "src_lang_name": src_name,
                    "tgt_lang_name": tgt_name,
                    "reason": "below_viability_threshold",
                    "chrf": round(best_chrf, 2),
                    "threshold": VIABILITY_THRESHOLD,
                })

        return ClaimsVerdict(
            model_id=self._model.id,
            model_display_name=self._model.display_name,
            claimed_pairs=len(claimed),
            viable_pairs=viable_count,
            failed_pairs=failed,
        )
