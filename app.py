"""
IndianModelsEval — Local Gradio UI
Run: python app.py

Tabs:
  1. Setup   — model paths, dataset download, HF token
  2. Run     — model selection, language pairs, dry-run / run
  3. Results — completed runs table, per-language breakdown
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import threading
from pathlib import Path

import torch

# Force UTF-8 output on Windows to handle emoji in logs
if sys.stdout and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Pin HF cache inside this project folder before any HuggingFace import
# Reads from .env first, then falls back to project-relative default
_HF_HOME_DEFAULT = str(Path(__file__).parent / ".hf_cache")
_hf_home_from_env = None
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        if _line.startswith("HF_HOME="):
            _hf_home_from_env = _line.split("=", 1)[1].strip()
            break
os.environ.setdefault("HF_HOME", _hf_home_from_env or _HF_HOME_DEFAULT)
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

# Use expandable CUDA memory segments to eliminate allocator fragmentation.
# Without this, 20+ pairs of varied-size allocations fragment the heap and cause
# OOM even when total free memory is sufficient.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gradio as gr

from src.indian_models_eval.config import AppSettings, EvalConfig, FLORES_LANG_CODES
from src.indian_models_eval.test_planner import ModelRegistry, TestPlanner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REGISTRY_PATH = Path("model_registry.yaml")
RESULTS_DIR = Path("results")
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
COMET_CHECKPOINT = Path("models/scoring/wmt22-comet-da/checkpoints/model.ckpt")

# ---------------------------------------------------------------------------
# Pause / Stop control — set by UI buttons, checked in run_eval between pairs
# ---------------------------------------------------------------------------
_pause_event = threading.Event()
_pause_event.set()   # set = not paused
_stop_event = threading.Event()


# ---------------------------------------------------------------------------
# Startup — load persisted token and authenticate
# ---------------------------------------------------------------------------

def _load_token_from_env() -> str:
    """Read HF_TOKEN from .env file. Returns empty string if not set."""
    env_path = Path(".env")
    if not env_path.exists():
        return ""
    for line in env_path.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip()
    return ""


def _apply_token(token: str) -> None:
    """Set token in environment and log into HuggingFace Hub."""
    if not token:
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        logger.info("HuggingFace Hub: authenticated")
    except Exception as e:
        logger.warning("HF Hub login failed: %s", e)


# Apply saved token immediately at import time
_SAVED_TOKEN = _load_token_from_env()
if _SAVED_TOKEN:
    _apply_token(_SAVED_TOKEN)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_registry() -> ModelRegistry | None:
    try:
        return ModelRegistry(REGISTRY_PATH)
    except Exception as e:
        logger.error("Failed to load registry: %s", e)
        return None


def _get_results_files() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("*.json"), reverse=True)


def _read_results(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tab 1: Setup
# ---------------------------------------------------------------------------

def _clear_runner_caches() -> list[str]:
    """Clear all in-process model caches. Returns list of cleared runner names."""
    import gc
    from src.indian_models_eval.models.indictrans2 import runner as it2_runner
    from src.indian_models_eval.models.nllb import runner as nllb_runner
    from src.indian_models_eval.models.krutrim import runner as krutrim_runner
    freed = []
    for r in [it2_runner, nllb_runner, krutrim_runner]:
        if hasattr(r, '_model_cache') and r._model_cache:
            r._model_cache.clear()
            freed.append(r.__name__.split('.')[-2])
    gc.collect()
    torch.cuda.empty_cache()
    return freed


def reset_gpu() -> str:
    """Clear GPU memory — translation models, COMET scorer, and CUDA cache."""
    import gc
    if not torch.cuda.is_available():
        return "No CUDA GPU detected."
    before = torch.cuda.memory_allocated() / 1024**2
    # Free COMET scorer if held from a finished/failed run
    s = _run_state.pop("_scorer", None)
    if s is not None:
        if hasattr(s, "_comet_model"):
            s._comet_model = None
        del s
    gc.collect()
    freed = _clear_runner_caches()
    after = torch.cuda.memory_allocated() / 1024**2
    return (f"GPU reset. Freed: {before - after:.0f} MB. "
            f"Allocated: {after:.0f} MB. Caches cleared: {freed or 'none'}")


def toggle_pause() -> str:
    """Pause or resume a running eval. Returns new button label."""
    if _pause_event.is_set():
        _pause_event.clear()   # pause — GPU freed by eval thread between pairs
        logger.info("Eval PAUSED by user.")
        return "▶ Resume"
    else:
        _pause_event.set()     # resume
        logger.info("Eval RESUMED by user.")
        return "⏸ Pause"


def stop_run() -> str:
    """Stop the running eval. GPU freed by eval thread after current pair."""
    _stop_event.set()
    _pause_event.set()  # unblock if currently paused
    logger.info("Eval STOP requested by user.")
    return "⏸ Pause"  # reset pause button label


def check_setup() -> str:
    lines = []

    # HF Token
    token = _load_token_from_env()
    if token:
        lines.append(f"✅ HF Token: saved ({token[:8]}...)")
    else:
        lines.append("⚠️  HF Token: not set — paste token below and click Save")

    # Registry
    if REGISTRY_PATH.exists():
        registry = _load_registry()
        if registry:
            lines.append(f"✅ Registry loaded — {len(registry.all_ids)} models")
        else:
            lines.append("❌ Registry found but failed to parse")
    else:
        lines.append(f"❌ Registry not found at {REGISTRY_PATH}")

    # Model weights
    if MODELS_DIR.exists():
        translation_dir = MODELS_DIR / "translation"
        if translation_dir.exists():
            downloaded = [d.name for d in translation_dir.iterdir() if d.is_dir()]
            if downloaded:
                lines.append(f"✅ Model weights: {', '.join(downloaded)}")
            else:
                lines.append("⚠️  models/translation/ exists but no model folders found")
        else:
            lines.append("⚠️  models/translation/ not created yet")
    else:
        lines.append("⚠️  models/ directory not found — create it and download model weights")

    # Data cache
    if DATA_DIR.exists():
        datasets = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
        if datasets:
            lines.append(f"✅ Datasets cached: {', '.join(datasets)}")
        else:
            lines.append("⚠️  data/ exists but no datasets downloaded yet")
    else:
        lines.append("⚠️  data/ not found — click Download Datasets")

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            lines.append(f"✅ GPU: {gpu} ({vram:.1f} GB VRAM)")
        else:
            lines.append("⚠️  No CUDA GPU detected — eval will run on CPU (very slow)")
    except ImportError:
        lines.append("❌ PyTorch not installed")

    return "\n".join(lines)


def save_hf_token(token: str) -> str:
    if not token.strip():
        return "No token provided."
    env_path = Path(".env")
    content = env_path.read_text() if env_path.exists() else ""
    if "HF_TOKEN=" in content:
        lines = [
            f"HF_TOKEN={token}" if line.startswith("HF_TOKEN=") else line
            for line in content.splitlines()
        ]
        env_path.write_text("\n".join(lines) + "\n")
    else:
        with env_path.open("a") as f:
            f.write(f"\nHF_TOKEN={token}\n")
    _apply_token(token.strip())
    return "✅ HF_TOKEN saved to .env and active for this session"


def download_datasets(hf_token: str) -> str:
    from scripts.download_datasets import download_all
    try:
        # Use field value if provided, else fall back to already-loaded env token
        token = hf_token.strip() or os.environ.get("HF_TOKEN", "")
        if token:
            _apply_token(token)
        result = download_all(data_dir=DATA_DIR)
        return result
    except Exception as e:
        return f"❌ Download failed: {e}"


# ---------------------------------------------------------------------------
# Tab 2: Run
# ---------------------------------------------------------------------------

def get_model_choices() -> list[str]:
    registry = _load_registry()
    if not registry:
        return []
    return [f"{m.id} — {m.display_name}" for m in registry.all_models]


def _lang_label(code: str) -> str:
    name = FLORES_LANG_CODES.get(code, code)
    return f"{name} ({code})"


def get_model_lang_options(model_choice: str):
    """Return src dropdown choices, tgt dropdown choices, and pair count text."""
    empty = gr.update(choices=[], value=None), gr.update(choices=[], value=None), ""
    if not model_choice:
        return empty
    model_id = model_choice.split(" — ")[0]
    registry = _load_registry()
    if not registry:
        return empty
    model = registry.get(model_id)
    pairs = model.claimed_pairs
    src_langs = sorted({p.src for p in pairs}, key=lambda c: FLORES_LANG_CODES.get(c, c))
    tgt_langs = sorted({p.tgt for p in pairs}, key=lambda c: FLORES_LANG_CODES.get(c, c))
    src_choices = [_lang_label(c) for c in src_langs]
    tgt_choices = [_lang_label(c) for c in tgt_langs]
    return (
        gr.update(choices=src_choices, value=None),
        gr.update(choices=tgt_choices, value=None),
        f"{len(pairs)} claimed language pairs for {model.display_name}",
    )


def _pairs_to_rows(pairs: list[list[str]]) -> list[list[str]]:
    return [
        [FLORES_LANG_CODES.get(s, s), FLORES_LANG_CODES.get(t, t)]
        for s, t in pairs
    ]


def _pair_count_str(pairs: list) -> str:
    return f"{len(pairs)} pair(s) selected" if pairs else ""


def add_pair(src_choice: str, tgt_choice: str, current_pairs: list) -> tuple:
    if not src_choice or not tgt_choice:
        return current_pairs, _pairs_to_rows(current_pairs), _pair_count_str(current_pairs)
    src_code = src_choice.split("(")[-1].rstrip(")")
    tgt_code = tgt_choice.split("(")[-1].rstrip(")")
    if src_code == tgt_code or [src_code, tgt_code] in current_pairs:
        return current_pairs, _pairs_to_rows(current_pairs), _pair_count_str(current_pairs)
    current_pairs = current_pairs + [[src_code, tgt_code]]
    return current_pairs, _pairs_to_rows(current_pairs), _pair_count_str(current_pairs)


def add_all_pairs(model_choice: str, current_pairs: list) -> tuple:
    if not model_choice:
        return current_pairs, _pairs_to_rows(current_pairs), _pair_count_str(current_pairs)
    model_id = model_choice.split(" — ")[0]
    registry = _load_registry()
    if not registry:
        return current_pairs, _pairs_to_rows(current_pairs), _pair_count_str(current_pairs)
    model = registry.get(model_id)
    all_p = [[p.src, p.tgt] for p in model.claimed_pairs]
    return all_p, _pairs_to_rows(all_p), _pair_count_str(all_p)


def clear_pairs() -> tuple:
    return [], [], ""


def get_pair_count(model_choice: str) -> str:
    if not model_choice:
        return ""
    model_id = model_choice.split(" — ")[0]
    registry = _load_registry()
    if not registry:
        return ""
    model = registry.get(model_id)
    return f"{len(model.claimed_pairs)} claimed pairs"


def get_results_file_choices() -> list[str]:
    return [f.name for f in _get_results_files()]


def refresh_results() -> tuple:
    return load_results_table(), get_results_file_choices()


# ---------------------------------------------------------------------------
# Background eval state — persists across browser sessions
# ---------------------------------------------------------------------------
_run_state: dict = {
    "running": False,
    "phase": "",        # "translating N/M", "scoring N/M", "done", "error", "stopped", "paused N/M"
    "log": [],          # last N status lines
    "result": "",       # final summary (set when done)
}


def _log(msg: str) -> None:
    logger.info(msg)
    _run_state["log"].append(msg)
    if len(_run_state["log"]) > 60:
        _run_state["log"] = _run_state["log"][-60:]


def get_status() -> str:
    """Polled by gr.Timer every few seconds to update the output textbox."""
    phase = _run_state["phase"]
    running = _run_state["running"]
    log_lines = _run_state["log"][-25:]

    if not phase and not running:
        return "No eval running."

    header = f"[{phase}]" if running else f"[{phase}]"
    return header + "\n\n" + "\n".join(log_lines) + (
        ("\n\n" + _run_state["result"]) if _run_state["result"] else ""
    )


def _eval_worker(
    model_id: str,
    config: "EvalConfig",
    model: object,
    test_cases: list,
) -> None:
    """Runs the full eval pipeline in a background thread — browser-independent."""
    from src.indian_models_eval.eval_runner import EvalRunner, TranslationResult
    from src.indian_models_eval.scoring_pipeline import ScoringPipeline
    from src.indian_models_eval.claims_verifier import ClaimsVerifier
    from src.indian_models_eval.results_writer import ResultsWriter

    _run_state["running"] = True
    _run_state["result"] = ""
    _run_state["log"] = []
    _run_state["_scorer"] = None  # track scorer for explicit cleanup

    dry_run = config.dry_run
    safe_model_id = model_id.replace('/', '_').replace('\\', '_')
    partial_path = RESULTS_DIR / f"_partial_{safe_model_id}.json"

    comet_path = str(COMET_CHECKPOINT) if COMET_CHECKPOINT.exists() else None
    scorer = ScoringPipeline(scoring_model_dir=comet_path, device="cuda")
    _run_state["_scorer"] = scorer  # keep reference for explicit cleanup
    verifier = ClaimsVerifier(model)
    writer = ResultsWriter(RESULTS_DIR)
    runner = EvalRunner(config)

    def _save_partial(results: list) -> None:
        data = [
            {
                "src_lang": r.src_lang, "tgt_lang": r.tgt_lang, "dataset": r.dataset,
                "sources": r.sources, "hypotheses": r.hypotheses, "references": r.references,
                "vram_peak_mb": r.vram_peak_mb, "latency_seconds": r.latency_seconds,
            }
            for r in results
        ]
        partial_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    try:
        # ── Auto-resume ──
        translation_results: list[TranslationResult] = []
        completed_keys: set[tuple[str, str, str]] = set()
        resume_note = ""
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
                resume_note = f"Resumed from partial save ({len(translation_results)} pairs already done).\n"
                _log(f"Auto-resume: {len(translation_results)} pairs loaded, skipping them.")
            except Exception as e:
                _log(f"Warning: failed to load partial file, starting fresh: {e}")
                translation_results = []
                completed_keys = set()

        remaining_cases = [tc for tc in test_cases
                           if (tc.src_lang, tc.tgt_lang, tc.dataset) not in completed_keys]
        total = len(test_cases)

        def _progress_cb(i: int, _total: int, msg: str) -> None:
            _run_state["phase"] = msg

        # ── Phase 1: translations ──
        for result in runner.run(model, remaining_cases, progress_callback=_progress_cb):
            translation_results.append(result)
            src_name = FLORES_LANG_CODES.get(result.src_lang, result.src_lang)
            tgt_name = FLORES_LANG_CODES.get(result.tgt_lang, result.tgt_lang)
            done = len(translation_results)
            _log(f"[{done}/{total}] Translated {src_name} -> {tgt_name} "
                 f"({len(result.hypotheses)} sentences, peak {result.vram_peak_mb:.0f} MB VRAM)")
            if not dry_run:
                _save_partial(translation_results)

            # Unload model between pairs — forces fresh CUDA allocation next pair,
            # defragmenting the heap. Adds ~3-5s reload time but prevents OOM.
            _clear_runner_caches()

            # Stop check
            if _stop_event.is_set():
                partial_path.unlink(missing_ok=True)
                _clear_runner_caches()
                _run_state["phase"] = "stopped"
                _run_state["result"] = (f"Stopped after {done}/{total} pairs. "
                                        f"Partial cleared — next run starts fresh.")
                _log("Eval stopped by user.")
                return

            # Pause check
            if not _pause_event.is_set():
                _clear_runner_caches()  # free GPU while paused
                _log(f"Paused after [{done}/{total}] — GPU memory freed. Click Resume to continue.")
                _run_state["phase"] = f"paused {done}/{total}"
                _pause_event.wait()
                if _stop_event.is_set():
                    partial_path.unlink(missing_ok=True)
                    _run_state["phase"] = "stopped"
                    _run_state["result"] = f"Stopped during pause after {done}/{total} pairs."
                    return
                _log("Resumed.")

        if dry_run:
            lines = ["DRY RUN COMPLETE\n"]
            for r in translation_results:
                src = FLORES_LANG_CODES.get(r.src_lang, r.src_lang)
                tgt = FLORES_LANG_CODES.get(r.tgt_lang, r.tgt_lang)
                lines.append(f"  {src} -> {tgt}: {r.latency_seconds:.1f}s, "
                              f"peak {r.vram_peak_mb:.0f} MB VRAM")
            _run_state["phase"] = "done"
            _run_state["result"] = "\n".join(lines)
            return

        # ── Phase 2: free GPU, then COMET ──
        _run_state["phase"] = "freeing GPU for COMET scoring..."
        _clear_runner_caches()
        _log("Translation model unloaded. Starting COMET scoring.")

        all_scores = []
        n = len(translation_results)
        for i, result in enumerate(translation_results):
            src_name = FLORES_LANG_CODES.get(result.src_lang, result.src_lang)
            tgt_name = FLORES_LANG_CODES.get(result.tgt_lang, result.tgt_lang)
            _run_state["phase"] = f"scoring {i+1}/{n}: {src_name} -> {tgt_name}"
            score = scorer.score(
                hypotheses=result.hypotheses, references=result.references,
                sources=result.sources, src_lang=result.src_lang,
                tgt_lang=result.tgt_lang, dataset=result.dataset,
                run_comet=True,
            )
            all_scores.append(score)
            _log(f"  Scored {src_name} -> {tgt_name}: chrF++={score.chrf:.1f} "
                 f"BLEU={score.bleu:.1f} viable={'yes' if score.viable else 'no'}")

        # ── Phase 3: write results ──
        partial_path.unlink(missing_ok=True)
        verdict = verifier.verify(all_scores)
        output_path = writer.write(model, config, translation_results, all_scores, verdict)

        ne = verdict.claimed_pairs - verdict.tested_pairs
        ne_note = f" ({ne} claimed pairs not tested)" if ne > 0 else ""
        summary_lines = [
            f"Eval complete — {output_path.name}\n",
            *(([resume_note]) if resume_note else []),
            f"Model: {model.display_name}",
            f"Tested: {verdict.tested_pairs} pair(s){ne_note}",
            f"Viable (chrF++ >= 40): {verdict.viable_pairs}/{verdict.tested_pairs}",
            f"Verdict: {verdict._verdict_label()}",
            "\nPer-language scores:",
        ]
        for score in sorted(all_scores, key=lambda s: s.chrf, reverse=True):
            src = FLORES_LANG_CODES.get(score.src_lang, score.src_lang)
            tgt = FLORES_LANG_CODES.get(score.tgt_lang, score.tgt_lang)
            viable = "OK" if score.viable else "--"
            summary_lines.append(f"  [{viable}] {src} -> {tgt}: chrF++={score.chrf:.1f}")

        _run_state["phase"] = "done"
        _run_state["result"] = "\n".join(summary_lines)
        _log("Eval complete.")

    except Exception as e:
        logger.exception("Eval worker failed")
        partial_note = f" Partial saved to {partial_path.name}." if partial_path.exists() else ""
        _run_state["phase"] = "error"
        _run_state["result"] = f"Eval failed: {e}{partial_note}"
        _log(f"ERROR: {e}")
    finally:
        import gc as _gc
        # Explicitly destroy COMET model so GPU tensors are released before empty_cache()
        s = _run_state.pop("_scorer", None)
        if s is not None:
            if hasattr(s, "_comet_model"):
                s._comet_model = None
            del s
        _gc.collect()
        _clear_runner_caches()  # always free GPU when thread exits (success, error, or stop)
        _run_state["running"] = False


def run_eval(
    model_choice: str,
    selected_pairs: list,
    dataset_flores: bool,
    dataset_in22: bool,
    dry_run: bool,
    batch_size: int,
) -> str:
    if _run_state["running"]:
        return "An eval is already running — check status above."
    if not model_choice:
        return "No model selected."

    model_id = model_choice.split(" — ")[0]
    datasets = []
    if dataset_flores:
        datasets.append("flores200")
    if dataset_in22:
        datasets.append("in22_gen")
    if not datasets:
        return "Select at least one dataset."

    lang_pairs = [(p[0], p[1]) for p in selected_pairs] if selected_pairs else None
    config = EvalConfig(model_id=model_id, datasets=datasets, language_pairs=lang_pairs,
                        dry_run=dry_run, batch_size=batch_size)

    registry = _load_registry()
    if not registry:
        return "Registry not loaded."
    model = registry.get(model_id)
    test_cases = TestPlanner(registry).plan(config)
    if not test_cases:
        return "No test cases — check language pair selection."

    # Reset control events
    _stop_event.clear()
    _pause_event.set()

    threading.Thread(
        target=_eval_worker,
        args=(model_id, config, model, test_cases),
        daemon=False,   # keeps running after browser closes
    ).start()

    return "Eval started in background — status updates every 5s. You can close this tab safely."


# ---------------------------------------------------------------------------
# Tab 3: Results
# ---------------------------------------------------------------------------

def _chrf_label(score) -> str:
    if score == "" or score is None:
        return ""
    try:
        v = float(score)
    except (TypeError, ValueError):
        return str(score)
    if v >= 55:
        label = "Excellent"
    elif v >= 45:
        label = "Good"
    elif v >= 40:
        label = "Marginal"
    else:
        label = "Not viable"
    return f"{v:.1f} ({label})"


def _bleu_label(score) -> str:
    if score == "" or score is None:
        return ""
    try:
        v = float(score)
    except (TypeError, ValueError):
        return str(score)
    if v >= 30:
        label = "Good"
    elif v >= 15:
        label = "Marginal"
    else:
        label = "Poor"
    return f"{v:.1f} ({label})"


def _comet_label(score) -> str:
    if score == "" or score is None:
        return "N/A"
    try:
        v = float(score)
    except (TypeError, ValueError):
        return str(score)
    if v >= 0.80:
        label = "Good"
    elif v >= 0.70:
        label = "Marginal"
    else:
        label = "Poor"
    return f"{v:.3f} ({label})"


def _viable_tested_label(claims: dict) -> str:
    viable = claims.get("viable_pairs", "?")
    tested = claims.get("tested_pairs")
    claimed = claims.get("claimed_pairs", "?")
    if tested is not None and tested != claimed:
        return f"{viable}/{tested} tested ({claimed} claimed)"
    return f"{viable}/{claimed}"


def load_results_table() -> list[list]:
    files = _get_results_files()
    rows = []
    for f in files:
        try:
            data = _read_results(f)
            model = data.get("model", {}).get("display_name", "?")
            datasets = ", ".join(data.get("eval_config", {}).get("datasets", []))
            agg = data.get("aggregate", {})
            claims = data.get("claims_verification", {})
            rows.append([
                f.name,
                model,
                datasets,
                data.get("timestamp", ""),
                _chrf_label(agg.get("chrf_mean", "")),
                _bleu_label(agg.get("bleu_mean", "")),
                _comet_label(agg.get("comet_mean", "")),
                _viable_tested_label(claims),
                claims.get("verdict", ""),
            ])
        except Exception:
            rows.append([f.name, "parse error", "", "", "", "", "", "", ""])
    return rows


def load_result_detail(filename: str) -> str:
    if not filename:
        return "Select a results file to see details."
    path = RESULTS_DIR / filename
    if not path.exists():
        return f"File not found: {filename}"
    data = _read_results(path)
    per_lang = data.get("per_language_pair", [])
    lines = [
        f"Model: {data['model']['display_name']}",
        f"Timestamp: {data['timestamp']}",
        f"Verdict: {data['claims_verification']['verdict']}\n",
        f"{'Language Pair':<30} {'chrF++':<8} {'BLEU':<8} {'COMET':<8} {'Viable':<8}",
        "-" * 65,
    ]
    for s in sorted(per_lang, key=lambda x: x.get("chrf", 0), reverse=True):
        pair = f"{s.get('src_lang_name', '?')} → {s.get('tgt_lang_name', '?')}"
        viable = "✅" if s.get("viable") else "❌"
        comet = f"{s.get('comet', ''):.4f}" if s.get("comet") is not None else "N/A"
        if not s.get("comet_reliable", True):
            comet += " ⚠️"
        lines.append(
            f"{pair:<30} {s.get('chrf', 0):<8.1f} {s.get('bleu', 0):<8.1f} {comet:<8} {viable}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    # Pre-compute initial dropdown state so From/To populate on first load
    _model_choices = get_model_choices()
    _default_model = _model_choices[0] if _model_choices else None
    _init_src, _init_tgt, _init_count = get_model_lang_options(_default_model)

    with gr.Blocks(title="IndianModelsEval") as demo:
        gr.Markdown("## IndianModelsEval — Indian Language Translation Benchmark")

        with gr.Tab("Run Eval"):
            pairs_state = gr.State([])

            with gr.Row():
                model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=_model_choices,
                    value=_default_model,
                    interactive=True,
                    scale=5,
                )
                pair_count_info = gr.Textbox(
                    label="", value=_init_count, interactive=False, max_lines=1, scale=1
                )

            with gr.Row():
                src_dropdown = gr.Dropdown(label="From", choices=_init_src["choices"], interactive=True, scale=3)
                tgt_dropdown = gr.Dropdown(label="To", choices=_init_tgt["choices"], interactive=True, scale=3)
                add_pair_btn = gr.Button("Add", scale=1)
                add_all_btn = gr.Button("Add All", scale=1)
                clear_btn = gr.Button("Clear", scale=1)

            pairs_table = gr.Dataframe(
                headers=["From", "To"],
                datatype=["str", "str"],
                interactive=False,
                label="Selected pairs (empty = run all claimed pairs)",
                column_count=2,
                row_count=(3, "dynamic"),
            )

            with gr.Row():
                dataset_flores = gr.Checkbox(label="FLORES-200", value=True)
                dataset_in22 = gr.Checkbox(label="IN22-Gen", value=False)
                dry_run_check = gr.Checkbox(label="Dry run (5 sentences)", value=False)
                batch_size_input = gr.Number(
                    label="Batch size", value=64, minimum=1, maximum=256, precision=0
                )

            with gr.Row():
                run_btn = gr.Button("Run Eval", variant="primary", scale=3)
                pause_btn = gr.Button("⏸ Pause", scale=1)
                stop_btn = gr.Button("⏹ Stop", variant="stop", scale=1)
            run_output = gr.Textbox(label="Status (auto-refreshes every 5s)", lines=16, interactive=False)
            eval_timer = gr.Timer(value=5, active=True)

            model_dropdown.change(
                get_model_lang_options,
                inputs=model_dropdown,
                outputs=[src_dropdown, tgt_dropdown, pair_count_info],
            )
            add_pair_btn.click(
                add_pair,
                inputs=[src_dropdown, tgt_dropdown, pairs_state],
                outputs=[pairs_state, pairs_table, pair_count_info],
            )
            add_all_btn.click(
                add_all_pairs,
                inputs=[model_dropdown, pairs_state],
                outputs=[pairs_state, pairs_table, pair_count_info],
            )
            clear_btn.click(clear_pairs, outputs=[pairs_state, pairs_table, pair_count_info])
            run_btn.click(
                run_eval,
                inputs=[model_dropdown, pairs_state, dataset_flores, dataset_in22,
                        dry_run_check, batch_size_input],
                outputs=run_output,
            )
            pause_btn.click(toggle_pause, outputs=pause_btn)
            stop_btn.click(stop_run, outputs=pause_btn)
            eval_timer.tick(get_status, outputs=run_output)

        with gr.Tab("Results"):
            with gr.Row():
                refresh_btn = gr.Button("Refresh", scale=1)
                file_dropdown = gr.Dropdown(
                    label="Select run",
                    choices=get_results_file_choices(),
                    interactive=True,
                    scale=4,
                )

            results_table = gr.Dataframe(
                headers=[
                    "File", "Model", "Datasets", "Timestamp",
                    "chrF++ mean", "BLEU mean", "COMET mean",
                    "Viable/Claimed", "Verdict",
                ],
                interactive=False,
            )

            with gr.Accordion("Score legend", open=False):
                gr.Markdown("""
| Metric | Good | Marginal | Not viable |
|--------|------|----------|------------|
| **chrF++** | > 50 | 40–50 | < 40 |
| **BLEU** | > 30 | 15–30 | < 15 |
| **COMET** | > 0.80 | 0.70–0.80 | < 0.70 |

**Viable/Claimed** — pairs scoring chrF++ ≥ 40 out of pairs tested (and total claimed).
COMET flagged ⚠️ for Manipuri, Bodo, Santhali — metric unreliable for these languages.
""")

            detail_output = gr.Textbox(label="Per-language breakdown", lines=20, interactive=False)

            refresh_btn.click(refresh_results, outputs=[results_table, file_dropdown])
            file_dropdown.change(load_result_detail, inputs=file_dropdown, outputs=detail_output)
            demo.load(load_results_table, outputs=results_table)

        with gr.Tab("Setup"):
            setup_status = gr.Textbox(label="Environment", lines=6, interactive=False)

            with gr.Row():
                check_btn = gr.Button("Refresh Check", scale=1)
                gr.HTML("<div></div>", elem_classes=[])

            gr.Markdown("**HuggingFace Token**")
            with gr.Row():
                hf_token_input = gr.Textbox(
                    label="HF Token",
                    type="password",
                    placeholder="hf_...",
                    value=_SAVED_TOKEN,
                    scale=4,
                )
                save_token_btn = gr.Button("Save", scale=1)

            token_status = gr.Textbox(label="", interactive=False, max_lines=1)

            gr.Markdown("**GPU Memory** — clears CUDA cache and unloads cached model weights. Use after a failed/OOM run.")
            with gr.Row():
                reset_gpu_btn = gr.Button("Reset GPU Memory", variant="stop", scale=1)
                reset_gpu_status = gr.Textbox(label="", interactive=False, max_lines=2, scale=3)

            gr.Markdown("**Datasets** — downloads FLORES-200 and IN22-Gen to `data/` (gitignored). Safe to re-run.")
            with gr.Row():
                download_btn = gr.Button("Download Datasets", scale=1)
                download_status = gr.Textbox(label="", interactive=False, lines=3, scale=3)

            check_btn.click(check_setup, outputs=setup_status)
            save_token_btn.click(save_hf_token, inputs=hf_token_input, outputs=token_status)
            reset_gpu_btn.click(reset_gpu, outputs=reset_gpu_status)
            download_btn.click(download_datasets, inputs=hf_token_input, outputs=download_status)

        with gr.Tab("About"):
            gr.Markdown("""
**IndianModelsEval** — independent benchmark verifying Indian language translation model claims.

Models are tested against their *own stated capabilities* using FLORES-200 and IN22-Gen.
A language pair is **viable** if chrF++ ≥ 40. Results include per-language breakdowns and claimed vs. verified counts.

| Model | Developer | Independent Eval? |
|-------|-----------|-------------------|
| IndicTrans2 (en→Indic) | AI4Bharat / IIT Madras | Self-reported only |
| IndicTrans2 (Indic→en) | AI4Bharat / IIT Madras | Self-reported only |
| NLLB-200 (600M) | Meta | Yes (well benchmarked) |
| Krutrim Translate | Krutrim / Ola | **None — first independent eval** |
| Sarvam-Translate | Sarvam AI | None |
| Gemma-3 | Google | Partial |

Running on RTX 3090 (24 GB VRAM) — no cloud costs. Results committed to git for reproducibility.
""")

        demo.load(check_setup, outputs=setup_status)

    return demo


if __name__ == "__main__":
    ui = build_ui()
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    ui.launch(server_name="127.0.0.1", server_port=port, share=False, theme=gr.themes.Default())
