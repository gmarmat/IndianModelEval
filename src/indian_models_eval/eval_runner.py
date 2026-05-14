"""
Eval runner — dispatches to model-specific runners, manages GPU memory,
handles dry-run mode, logs VRAM and latency.
"""
from __future__ import annotations

import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .config import EvalConfig, FLORES_LANG_CODES
from .dataset_loader import SentencePair, load_pairs
from .test_planner import ModelEntry, TestCase

logger = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_vram_used_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def get_vram_total_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1024**2
    return 0.0


@dataclass
class TranslationResult:
    src_lang: str
    tgt_lang: str
    dataset: str
    sources: list[str]
    hypotheses: list[str]
    references: list[str]
    vram_peak_mb: float = 0.0
    latency_seconds: float = 0.0
    sentences_per_second: float = 0.0
    hardware: dict = field(default_factory=dict)


class EvalRunner:
    """
    Runs translation eval for a single model across all test cases.
    Manages GPU memory between language pairs.
    """

    def __init__(self, config: EvalConfig) -> None:
        self.config = config

    def run(
        self,
        model: ModelEntry,
        test_cases: list[TestCase],
        progress_callback: callable | None = None,
    ) -> Iterator[TranslationResult]:
        """
        Run evaluation for all test cases. Yields TranslationResult per (lang_pair, dataset).

        Args:
            model: ModelEntry from registry
            test_cases: list of (src_lang, tgt_lang, dataset) triples
            progress_callback: optional fn(current, total, msg) for UI updates
        """
        set_seeds(self.config.seed)

        n_sentences = self.config.dry_run_sentences if self.config.dry_run else None

        if self.config.dry_run:
            logger.info(
                "DRY RUN: running %d sentences per language pair",
                self.config.dry_run_sentences,
            )

        total = len(test_cases)
        for i, case in enumerate(test_cases):
            src_name = FLORES_LANG_CODES.get(case.src_lang, case.src_lang)
            tgt_name = FLORES_LANG_CODES.get(case.tgt_lang, case.tgt_lang)
            msg = f"[{i+1}/{total}] {src_name} → {tgt_name} ({case.dataset})"
            logger.info(msg)
            if progress_callback:
                progress_callback(i, total, msg)

            # Load sentence pairs
            pairs: list[SentencePair] = load_pairs(
                case.src_lang,
                case.tgt_lang,
                case.dataset,
                cache_dir=self.config.data_cache_dir,
                n=n_sentences,
            )

            sources = [p.source for p in pairs]
            references = [p.reference for p in pairs]

            # Translate
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()

            if model.requires_subprocess:
                hypotheses = self._run_subprocess(model, sources, case.src_lang, case.tgt_lang)
            else:
                hypotheses = self._run_direct(model, sources, case.src_lang, case.tgt_lang)

            elapsed = time.perf_counter() - t0
            vram_peak = (
                torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
            )

            # Aggressive cleanup between pairs to fight CUDA allocator fragmentation
            import gc as _gc
            _gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            hardware = {
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "vram_total_mb": round(get_vram_total_mb(), 1),
            }

            yield TranslationResult(
                src_lang=case.src_lang,
                tgt_lang=case.tgt_lang,
                dataset=case.dataset,
                sources=sources,
                hypotheses=hypotheses,
                references=references,
                vram_peak_mb=round(vram_peak, 1),
                latency_seconds=round(elapsed, 2),
                sentences_per_second=round(len(sources) / elapsed, 2) if elapsed > 0 else 0.0,
                hardware=hardware,
            )

            if self.config.dry_run:
                est_total = elapsed * (1012 / self.config.dry_run_sentences)
                logger.info(
                    "Dry run complete. VRAM peak: %.1f MB. "
                    "Estimated full run time: %.1f min",
                    vram_peak,
                    est_total / 60,
                )

    def _run_direct(
        self,
        model: ModelEntry,
        sources: list[str],
        src_lang: str,
        tgt_lang: str,
    ) -> list[str]:
        """Dispatch to the right model runner based on model type."""
        if model.model_type == "seq2seq":
            if "indictrans2" in model.id:
                from .models.indictrans2.runner import translate
            elif "nllb" in model.id:
                from .models.nllb.runner import translate
            elif "krutrim" in model.id:
                from .models.krutrim.runner import translate
            else:
                raise ValueError(f"No seq2seq runner for model: {model.id}")
        elif model.model_type == "decoder_only":
            if "gemma" in model.id:
                from .models.gemma.runner import translate
            else:
                raise ValueError(f"No decoder runner for model: {model.id}")
        else:
            raise ValueError(f"Unknown model type: {model.model_type!r}")

        return translate(
            sentences=sources,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            model_path=str(model.local_path),
            batch_size=self.config.batch_size,
            max_new_tokens=self.config.max_new_tokens,
            precision=self.config.precision,
            device=self.config.device,
            extra_config=model.eval_config,
        )

    def _run_subprocess(
        self,
        model: ModelEntry,
        sources: list[str],
        src_lang: str,
        tgt_lang: str,
    ) -> list[str]:
        """
        Run GPL-3.0 model (Sarvam) via subprocess to maintain license isolation.
        Core harness never imports Sarvam code directly.
        """
        import json
        import subprocess
        import tempfile

        sarvam_runner = Path(__file__).parent / "models" / "sarvam" / "runner.py"
        if not sarvam_runner.exists():
            raise FileNotFoundError(f"Sarvam runner not found: {sarvam_runner}")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f_in:
            json.dump(
                {
                    "sentences": sources,
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "model_path": str(model.local_path),
                    "batch_size": self.config.batch_size,
                    "precision": self.config.precision,
                },
                f_in,
            )
            input_path = Path(f_in.name)

        output_path = input_path.with_suffix(".out.json")

        # CREATE_NO_WINDOW (0x08000000) prevents the child from getting its own
        # console window. Without this, Intel MKL inside the subprocess traps
        # console-close events and aborts with forrtl error 200 when the
        # session disconnects (RDP detach, scheduled-task session change, etc.).
        _creationflags = 0x08000000 if sys.platform == "win32" else 0

        try:
            result = subprocess.run(
                [sys.executable, str(sarvam_runner), str(input_path), str(output_path)],
                check=True,
                capture_output=True,
                text=True,
                creationflags=_creationflags,
            )
            if result.stderr:
                logger.debug("Sarvam runner stderr: %s", result.stderr)

            # encoding="utf-8" required: subprocess JSON contains non-ASCII
            # if a runner writes with ensure_ascii=False, and on Windows the
            # default locale encoding (cp1252) silently mangles it.
            with output_path.open(encoding="utf-8") as f:
                data = json.load(f)

            return data["translations"]

        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
