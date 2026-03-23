"""
CLI runner for IndianModelsEval — same pipeline as the Gradio UI, no browser needed.

Usage:
    python scripts/run_eval_cli.py --model indictrans2-indic-en --batch-size 64
    python scripts/run_eval_cli.py --model nllb-200-distilled-600M --batch-size 32
    python scripts/run_eval_cli.py --model indictrans2-indic-en --dry-run
    python scripts/run_eval_cli.py --model indictrans2-indic-en --pairs hin_Deva-eng_Latn ben_Beng-eng_Latn
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

# ── Environment setup (mirrors app.py) ─────────────────────────────────────
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

_env_path = _root / ".env"
_hf_home = None
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        if _line.startswith("HF_HOME="):
            _hf_home = _line.split("=", 1)[1].strip()
        if _line.startswith("HF_TOKEN="):
            os.environ.setdefault("HF_TOKEN", _line.split("=", 1)[1].strip())
os.environ.setdefault("HF_HOME", _hf_home or str(_root / ".hf_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# ───────────────────────────────────────────────────────────────────────────

import torch

from src.indian_models_eval.config import EvalConfig, FLORES_LANG_CODES
from src.indian_models_eval.test_planner import ModelRegistry, TestPlanner
from src.indian_models_eval.eval_runner import EvalRunner, TranslationResult
from src.indian_models_eval.scoring_pipeline import ScoringPipeline
from src.indian_models_eval.claims_verifier import ClaimsVerifier
from src.indian_models_eval.results_writer import ResultsWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

RESULTS_DIR = _root / "results"
COMET_CHECKPOINT = _root / "models/scoring/wmt22-comet-da/checkpoints/model.ckpt"


def _clear_caches() -> None:
    from src.indian_models_eval.models.indictrans2 import runner as it2
    it2._model_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="IndianModelsEval CLI runner")
    parser.add_argument("--model", required=True, help="Model ID from model_registry.yaml")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true", help="Run 5 sentences per pair for timing")
    parser.add_argument("--dataset", choices=["flores200", "in22"], default="flores200")
    parser.add_argument("--pairs", nargs="+", metavar="SRC-TGT",
                        help="Specific pairs to test e.g. hin_Deva-eng_Latn (default: all claimed)")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--no-comet", action="store_true", help="Skip COMET scoring (faster)")
    args = parser.parse_args()

    # Load registry
    registry = ModelRegistry(_root / "model_registry.yaml")
    if args.model not in registry.all_ids:
        logger.error("Model '%s' not found. Available: %s", args.model, registry.all_ids)
        sys.exit(1)
    model = registry.get(args.model)

    # Build config and test cases
    datasets = [args.dataset]
    lang_pairs = [tuple(p.split("-", 1)) for p in args.pairs] if args.pairs else None
    config = EvalConfig(
        model_id=args.model,
        datasets=datasets,
        language_pairs=lang_pairs,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        precision=args.precision,
    )

    planner = TestPlanner(registry)
    test_cases = planner.plan(config)

    if not test_cases:
        logger.error("No test cases generated for model '%s'", args.model)
        sys.exit(1)

    logger.info("Model:      %s", model.display_name)
    logger.info("Test cases: %d pairs × %s", len(test_cases), datasets)
    logger.info("Batch size: %d | Precision: %s | Dry run: %s",
                args.batch_size, args.precision, args.dry_run)
    if torch.cuda.is_available():
        free_mb = torch.cuda.mem_get_info()[0] / 1024**2
        logger.info("VRAM free:  %.0f MB", free_mb)

    # Auto-resume support
    safe_id = args.model.replace("/", "_").replace("\\", "_")
    partial_path = RESULTS_DIR / f"_partial_{safe_id}.json"
    translation_results: list[TranslationResult] = []
    completed_keys: set[tuple[str, str, str]] = set()

    if partial_path.exists():
        try:
            partial_data = json.loads(partial_path.read_text(encoding="utf-8"))
            for d in partial_data:
                translation_results.append(TranslationResult(
                    src_lang=d["src_lang"], tgt_lang=d["tgt_lang"], dataset=d["dataset"],
                    sources=d["sources"], hypotheses=d["hypotheses"], references=d["references"],
                    vram_peak_mb=d.get("vram_peak_mb", 0.0),
                    latency_seconds=d.get("latency_seconds", 0.0),
                ))
                completed_keys.add((d["src_lang"], d["tgt_lang"], d["dataset"]))
            logger.info("Auto-resume: %d pairs already done, skipping.", len(translation_results))
        except Exception as e:
            logger.warning("Failed to load partial file, starting fresh: %s", e)
            translation_results, completed_keys = [], set()

    remaining = [tc for tc in test_cases
                 if (tc.src_lang, tc.tgt_lang, tc.dataset) not in completed_keys]

    # ── Phase 1: translations ──────────────────────────────────────────────
    runner = EvalRunner(config)
    total = len(test_cases)

    for result in runner.run(model, remaining):
        translation_results.append(result)
        src = FLORES_LANG_CODES.get(result.src_lang, result.src_lang)
        tgt = FLORES_LANG_CODES.get(result.tgt_lang, result.tgt_lang)
        done = len(translation_results)
        logger.info("[%d/%d] %s → %s — %.1fs, peak %.0f MB VRAM",
                    done, total, src, tgt, result.latency_seconds, result.vram_peak_mb)

        if not args.dry_run:
            data = [
                {"src_lang": r.src_lang, "tgt_lang": r.tgt_lang, "dataset": r.dataset,
                 "sources": r.sources, "hypotheses": r.hypotheses, "references": r.references,
                 "vram_peak_mb": r.vram_peak_mb, "latency_seconds": r.latency_seconds}
                for r in translation_results
            ]
            partial_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        _clear_caches()

    if args.dry_run:
        logger.info("Dry run complete.")
        for r in translation_results:
            src = FLORES_LANG_CODES.get(r.src_lang, r.src_lang)
            tgt = FLORES_LANG_CODES.get(r.tgt_lang, r.tgt_lang)
            est = r.latency_seconds * (1012 / config.dry_run_sentences)
            logger.info("  %s → %s: %.1fs/pair → est. full %.1f min",
                        src, tgt, r.latency_seconds, est / 60)
        return

    # ── Phase 2: COMET scoring ─────────────────────────────────────────────
    _clear_caches()
    comet_path = str(COMET_CHECKPOINT) if COMET_CHECKPOINT.exists() else None
    scorer = ScoringPipeline(scoring_model_dir=comet_path, device="cuda")

    all_scores = []
    n = len(translation_results)
    for i, result in enumerate(translation_results):
        src = FLORES_LANG_CODES.get(result.src_lang, result.src_lang)
        tgt = FLORES_LANG_CODES.get(result.tgt_lang, result.tgt_lang)
        logger.info("Scoring [%d/%d]: %s → %s", i + 1, n, src, tgt)
        score = scorer.score(
            hypotheses=result.hypotheses, references=result.references,
            sources=result.sources, src_lang=result.src_lang,
            tgt_lang=result.tgt_lang, dataset=result.dataset,
            run_comet=not args.no_comet,
        )
        all_scores.append(score)
        logger.info("  chrF++=%.1f BLEU=%.1f viable=%s", score.chrf, score.bleu, score.viable)

    # ── Phase 3: write results ─────────────────────────────────────────────
    partial_path.unlink(missing_ok=True)
    verifier = ClaimsVerifier(model)
    verdict = verifier.verify(all_scores)
    writer = ResultsWriter(RESULTS_DIR)
    output_path = writer.write(model, config, translation_results, all_scores, verdict)

    logger.info("=" * 60)
    logger.info("Results: %s", output_path)
    logger.info("Viable: %d/%d (chrF++ >= 40)", verdict.viable_pairs, verdict.tested_pairs)
    logger.info("Verdict: %s", verdict.verdict)
    logger.info("")
    for score in sorted(all_scores, key=lambda s: s.chrf, reverse=True):
        src = FLORES_LANG_CODES.get(score.src_lang, score.src_lang)
        tgt = FLORES_LANG_CODES.get(score.tgt_lang, score.tgt_lang)
        flag = "OK" if score.viable else "--"
        logger.info("  [%s] %s → %s: chrF++=%.1f BLEU=%.1f", flag, src, tgt, score.chrf, score.bleu)


if __name__ == "__main__":
    main()
