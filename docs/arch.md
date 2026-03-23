# IndianModelsEval — Architecture

**Version:** 0.1 | **Updated:** 2026-03-18

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
│   └── download_datasets.py    # One-time dataset download
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
# Required for dataset download
HF_TOKEN=

# Optional — API inference (local weights default)
SARVAM_API_KEY=
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
5. **VRAM guard** — `torch.cuda.empty_cache()` after every model eval. Batch size configured per model in registry.

---

## Feature Index

| # | Feature | Doc | Phase | Status |
|---|---------|-----|-------|--------|
| 1 | Model Registry | [model-registry.md](./features/model-registry.md) | 1 | planned |
| 2 | Dataset Loader | [eval-pipeline.md](./features/eval-pipeline.md) | 1 | planned |
| 3 | Test Planner | [eval-pipeline.md](./features/eval-pipeline.md) | 1 | planned |
| 4 | Eval Runner | [eval-pipeline.md](./features/eval-pipeline.md) | 1 | planned |
| 5 | Scoring Pipeline | [eval-pipeline.md](./features/eval-pipeline.md) | 1 | planned |
| 6 | Claims Verifier | [eval-pipeline.md](./features/eval-pipeline.md) | 1 | planned |
| 7 | Results Writer | [results-schema.md](./features/results-schema.md) | 1 | planned |
| 8 | Sarvam Isolation | [eval-pipeline.md](./features/eval-pipeline.md) | 1 | planned |
| 9 | Local Web UI | [ui.md](./features/ui.md) | 1 | planned |
| 10 | Indic-COMET + xCOMET | [eval-pipeline.md](./features/eval-pipeline.md) | 2 | planned |
| 11 | Indic↔Indic pairs | [eval-pipeline.md](./features/eval-pipeline.md) | 2 | planned |
| 12 | Tokenization Fertility | [eval-pipeline.md](./features/eval-pipeline.md) | 2 | planned |
| 13 | HF Space Publish | [publishing.md](./features/publishing.md) | 2 | planned |

---

## Version History

| Ver | Date | Changes |
|-----|------|---------|
| 0.1 | 2026-03-18 | **Initial scaffold.** Architecture from PRD + plans. Phase 1 code written. |
