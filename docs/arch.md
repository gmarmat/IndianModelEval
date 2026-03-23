# IndianModelsEval — Architecture

**Version:** 0.6 | **Updated:** 2026-03-23

> **LLM Instructions:** Token-optimized index. Read top-to-bottom. Tables over prose. Detailed docs in `docs/features/*.md`.

---

## Quick Reference

### What Is This?

Independent evaluation harness that verifies Indian language translation model claims. Runs on RTX 3090 (local, no cloud). Single operator. Publishes results to HF Space leaderboard.

### Tech Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| Language | Python 3.10+ | Type hints everywhere |
| ML framework | PyTorch + HuggingFace Transformers | CUDA on RTX 3090 |
| Metrics | sacrebleu + unbabel-comet | chrF++, BLEU, COMET-22 |
| UI | Gradio | 3-tab local app |
| Config | Pydantic v2 + pydantic-settings | Validates on load |
| Registry | YAML | `model_registry.yaml` at project root |
| Results | JSON | `results/{model}_{dataset}_{timestamp}.json` |
| Datasets | HuggingFace datasets | FLORES-200, IN22-Gen |
| Packaging | pyproject.toml + ruff | No setup.py |

### Project Structure

```
IndianModelsEval/
├── src/indian_models_eval/
│   ├── __init__.py             # version: YYYY.MM.DD.XX
│   ├── config.py               # Pydantic EvalConfig + AppSettings + lang maps
│   ├── dataset_loader.py       # FLORES-200 + IN22-Gen loaders
│   ├── test_planner.py         # ModelRegistry + TestPlanner (lang pair matrix)
│   ├── eval_runner.py          # GPU eval dispatcher, dry-run, VRAM logging
│   ├── scoring_pipeline.py     # chrF++ + BLEU + COMET-22, reliability flags
│   ├── claims_verifier.py      # Claimed vs viable (≥40 chrF++)
│   ├── results_writer.py       # JSON results writer
│   ├── models/
│   │   ├── indictrans2/runner.py   # MIT — seq2seq
│   │   ├── nllb/runner.py          # CC-BY-NC — seq2seq
│   │   ├── gemma/runner.py         # Gemma license — decoder-only
│   │   └── sarvam/runner.py        # GPL-3.0 — ISOLATED (subprocess only)
│   └── utils/
│       └── cost_guard.py           # API cost cap ($20 hard limit)
├── scripts/
│   ├── download_datasets.py    # One-time dataset download
│   └── run_eval_cli.py         # Headless CLI runner (no Gradio required)
├── results/                    # Committed JSON results
├── data/                       # gitignored — dataset cache
├── models/
│   ├── translation/            # gitignored — model weights
│   └── scoring/                # gitignored — COMET weights
├── model_registry.yaml         # 5 models, claimed languages, licenses
├── app.py                      # Gradio UI entry point
├── pyproject.toml
├── .env.example
└── docs/
    ├── arch.md                 # This file
    ├── PRD.md
    ├── features/               # Deep dives
    └── plans/                  # Research artifacts
```

### Environment Variables

```
# Required for dataset download / gated models
HF_TOKEN=

# Optional — API inference (local weights default)
SARVAM_API_KEY=

# CUDA allocator — set at app startup to prevent fragmentation on long runs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## Domain Concepts

| Concept | Description |
|---------|-------------|
| **Model Registry** | `model_registry.yaml` — each model's claimed languages, license, HF ID, local path |
| **Test Plan** | Per-model matrix of `(src_lang, tgt_lang, dataset)` triples derived from claimed capabilities |
| **Viability** | A language pair is "viable" if chrF++ ≥ 40. Below = effectively unusable. |
| **Claims Verification** | Comparing model's *claimed* language support vs *measured* viable pairs |
| **Reliability Flag** | Warning on COMET scores for low-resource langs (Manipuri, Bodo, Santhali) where metric is unreliable |
| **Sarvam Isolation** | Sarvam eval runs in subprocess — GPL-3.0 code never imported by core harness (MIT) |
| **FLORES-200 code** | `{lang}_{Script}` BCP-47 style e.g. `hin_Deva`, `eng_Latn`, `tam_Taml` |

---

## Data Flow

```
model_registry.yaml
        │
        ▼
TestPlanner          → (model, src_lang, tgt_lang, dataset) matrix
        │
        ▼
DatasetLoader        → [(src_sentence, ref_sentence), ...]
        │
        ▼
EvalRunner           → calls model runner (or Sarvam via subprocess)
        │              logs VRAM peak, latency, fertility
        ▼
ScoringPipeline      → {chrf: float, bleu: float, comet: float, reliable: bool}
        │
        ▼
ClaimsVerifier       → {claimed: N, viable: M, failed_pairs: [...]}
        │
        ▼
ResultsWriter        → results/{model}_{dataset}_{timestamp}.json
```

---

## Key Patterns

1. **Sarvam subprocess isolation** — `eval_runner.py` calls `src/indian_models_eval/models/sarvam/runner.py` via `subprocess.run([sys.executable, ...])`. Core harness never imports Sarvam code.
2. **Dry-run first** — every eval defaults to prompting for `--dry-run`. Runs 5 sentences, reports estimated time + VRAM.
3. **Pin everything** — results JSON records model revision hash, seed, dataset version, timestamp, precision, batch size.
4. **Pydantic config** — `EvalConfig` validates all eval parameters. No secret values in config files.
5. **VRAM guard** — `torch.cuda.empty_cache()` + `gc.collect()` between every pair. `expandable_segments:True` prevents CUDA allocator fragmentation across long runs.
6. **Two-phase eval** — Phase 1: all translations (translation model only on GPU); Phase 2: clear translation model, run COMET. Prevents dual-model VRAM pressure on 24GB.
7. **Partial save + auto-resume** — translations saved to `_partial_{model_id}.json` after each pair. On restart, completed pairs are loaded and skipped. OOM never loses more than one pair.
8. **Background eval thread** — `threading.Thread(daemon=False)` keeps running after browser close. `gr.Timer` polls status every 5s. Pause/Stop via `threading.Event` objects.
9. **IndicTrans2 script normalization** — model encodes ALL Indic scripts internally as Devanagari. En→Indic: post-process output with `UnicodeIndicTransliterator(hi → target)`. Indic→En: pre-process input with `UnicodeIndicTransliterator(source → hi)` before tokenization. Santhali (Ol Chiki), Sindhi (Arabic), Manipuri (Meitei Mayek) unsupported by IndicNLP — skip transliteration.
10. **FLORES code remapping** — `dataset_loader.py:FLORES_PLUS_CODE_MAP` remaps IndicTrans2 lang codes to flores_plus dataset codes (e.g. `doi_Deva → dgo_Deva` for Dogri).

---

## Feature Index

| # | Feature | Doc | Phase | Status |
|---|---------|-----|-------|--------|
| 1 | Model Registry | [model-registry.md](./features/model-registry.md) | 1 | done |
| 2 | Dataset Loader | [eval-pipeline.md](./features/eval-pipeline.md) | 1 | done |
| 3 | Test Planner | [eval-pipeline.md](./features/eval-pipeline.md) | 1 | done |
| 4 | Eval Runner | [eval-pipeline.md](./features/eval-pipeline.md) | 1 | done |
| 5 | Scoring Pipeline | [eval-pipeline.md](./features/eval-pipeline.md) | 1 | done |
| 6 | Claims Verifier | [eval-pipeline.md](./features/eval-pipeline.md) | 1 | done |
| 7 | Results Writer | [results-schema.md](./features/results-schema.md) | 1 | done |
| 8 | Sarvam Isolation | [eval-pipeline.md](./features/eval-pipeline.md) | 1 | done |
| 9 | Local Web UI (Gradio, Pause/Stop, bg thread) | — | 1 | done |
| 10 | Indic-COMET + xCOMET | [eval-pipeline.md](./features/eval-pipeline.md) | 2 | planned |
| 11 | Indic↔Indic pairs | [eval-pipeline.md](./features/eval-pipeline.md) | 2 | planned |
| 12 | Tokenization Fertility | [eval-pipeline.md](./features/eval-pipeline.md) | 2 | planned |
| 13 | HF Space Publish | [publishing.md](./features/publishing.md) | 2 | planned |

---

## Version History

| Ver | Date | Changes |
|-----|------|---------|
| 0.1 | 2026-03-18 | **Initial scaffold.** Architecture from PRD + plans. Phase 1 code written. |
| 0.2 | 2026-03-19 | **NLLB + Krutrim evals.** First results committed. FLORES `dev` split fix, Dogri code remap (`doi_Deva→dgo_Deva`). |
| 0.3 | 2026-03-20 | **IndicTrans2 En→Indic eval.** Devanagari-unified encoding fix, UnicodeIndicTransliterator post-processing, token skip (`[:, 2:]`), two-phase eval, partial save + auto-resume, background thread + Pause/Stop UI, `expandable_segments:True`. |
| 0.4 | 2026-03-23 | **IndicTrans2 Indic→En eval.** Completed; non-Devanagari input scripts still score low — needs pre-tokenization transliteration fix before re-run. |
| 0.5 | 2026-03-22 | **GitHub release.** Public repo at github.com/gmarmat/IndianModelEval. Private files excluded. New Key Patterns 6-10 documented. |
| 0.6 | 2026-03-23 | **Indic→En transliteration fix + CLI runner.** Pre-tokenization transliteration added to `indictrans2/runner.py`; re-run yields 19/22 viable (up from 11/22), verdict `mostly_verified`. `scripts/run_eval_cli.py` added for headless runs. |
