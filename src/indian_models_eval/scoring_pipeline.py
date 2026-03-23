"""
Scoring pipeline: chrF++, BLEU, COMET-22.
Attaches reliability flags for low-resource languages where COMET is unreliable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import sacrebleu
from comet import download_model, load_from_checkpoint

from .config import COMET_UNRELIABLE_LANGS, FLORES_LANG_CODES

logger = logging.getLogger(__name__)

COMET_MODEL_NAME = "Unbabel/wmt22-comet-da"


@dataclass
class ScoreResult:
    src_lang: str
    tgt_lang: str
    dataset: str
    chrf: float
    bleu: float
    comet: float | None
    viable: bool  # chrf >= VIABILITY_THRESHOLD (40.0)
    comet_reliable: bool
    n_sentences: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "src_lang": self.src_lang,
            "tgt_lang": self.tgt_lang,
            "src_lang_name": FLORES_LANG_CODES.get(self.src_lang, self.src_lang),
            "tgt_lang_name": FLORES_LANG_CODES.get(self.tgt_lang, self.tgt_lang),
            "dataset": self.dataset,
            "chrf": round(self.chrf, 2),
            "bleu": round(self.bleu, 2),
            "comet": round(self.comet, 4) if self.comet is not None else None,
            "viable": self.viable,
            "comet_reliable": self.comet_reliable,
            "comet_reliability_note": (
                None
                if self.comet_reliable
                else "COMET unreliable for this language — metric trained on high-resource pairs only. "
                     "See https://arxiv.org/abs/2406.03893"
            ),
            "n_sentences": self.n_sentences,
            **self.metadata,
        }


class ScoringPipeline:
    """Computes chrF++, BLEU, and COMET-22 for a batch of translations."""

    def __init__(self, scoring_model_dir: str | None = None, device: str = "cuda") -> None:
        """
        Args:
            scoring_model_dir: path to local COMET model weights. If None, downloads from HF.
            device: "cuda" or "cpu"
        """
        self._device = device
        self._comet_model = None
        self._comet_model_dir = scoring_model_dir

    def _ensure_comet_loaded(self) -> None:
        if self._comet_model is not None:
            return
        logger.info("Loading COMET-22 model...")
        if self._comet_model_dir:
            self._comet_model = load_from_checkpoint(self._comet_model_dir)
        else:
            model_path = download_model(COMET_MODEL_NAME)
            self._comet_model = load_from_checkpoint(model_path)
        logger.info("COMET-22 loaded.")

    def score(
        self,
        hypotheses: list[str],
        references: list[str],
        sources: list[str],
        src_lang: str,
        tgt_lang: str,
        dataset: str,
        run_comet: bool = True,
    ) -> ScoreResult:
        """
        Score a set of translations.

        Args:
            hypotheses: model outputs
            references: gold reference translations
            sources: original source sentences (for COMET)
            src_lang: FLORES-200 source language code
            tgt_lang: FLORES-200 target language code
            dataset: dataset name for metadata
            run_comet: set False to skip COMET (faster, for dry-run)

        Returns:
            ScoreResult with all metrics and reliability flags.
        """
        if len(hypotheses) != len(references):
            raise ValueError(
                f"Hypothesis/reference count mismatch: {len(hypotheses)} vs {len(references)}"
            )

        # chrF++ (character-level F-score, n=6, beta=2)
        chrf_score = sacrebleu.corpus_chrf(
            hypotheses,
            [references],
            word_order=2,  # chrF++
        ).score

        # BLEU
        bleu_score = sacrebleu.corpus_bleu(
            hypotheses,
            [references],
            tokenize="flores200",
        ).score

        # COMET-22
        comet_score: float | None = None
        if run_comet:
            try:
                self._ensure_comet_loaded()
                comet_data = [
                    {"src": s, "mt": h, "ref": r}
                    for s, h, r in zip(sources, hypotheses, references)
                ]
                comet_output = self._comet_model.predict(
                    comet_data, batch_size=8, gpus=1 if self._device == "cuda" else 0
                )
                comet_score = float(comet_output.system_score)
            except Exception as e:
                logger.warning("COMET scoring failed: %s", e)

        comet_reliable = tgt_lang not in COMET_UNRELIABLE_LANGS

        from .config import VIABILITY_THRESHOLD

        return ScoreResult(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            dataset=dataset,
            chrf=chrf_score,
            bleu=bleu_score,
            comet=comet_score,
            viable=chrf_score >= VIABILITY_THRESHOLD,
            comet_reliable=comet_reliable,
            n_sentences=len(hypotheses),
        )
