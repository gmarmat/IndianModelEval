# Eval Pipeline

**Status:** done
**Last updated:** 2026-03-22

## Overview

The eval pipeline takes a model + dataset selection and produces scored translation results. Five stages: DatasetLoader → EvalRunner → (model runner) → ScoringPipeline → ResultsWriter.

## Requirements

- [ ] Load FLORES-200 and IN22-Gen sentence pairs
- [ ] Dry-run mode: 5 sentences/pair, reports estimated time + VRAM
- [ ] Log VRAM peak per language pair
- [ ] Compute chrF++, BLEU, COMET-22 per language pair
- [ ] Attach COMET reliability flags for low-resource langs
- [ ] Sarvam-Translate called via subprocess only (GPL-3.0 isolation)
- [ ] `torch.cuda.empty_cache()` between language pairs

## Design

### Data Flow

```
DatasetLoader.load_pairs(src, tgt, dataset)
    → [SentencePair(source, reference), ...]
    → EvalRunner.run(model, test_cases)
        → model runner (indictrans2 / nllb / gemma / sarvam subprocess)
        → [hypotheses]
    → ScoringPipeline.score(hyps, refs, srcs)
        → ScoreResult(chrf, bleu, comet, viable, comet_reliable)
    → ResultsWriter.write(...)
        → results/{model}_{dataset}_{timestamp}.json
```

### Model Runner Protocol

Every model runner implements the same signature:

```python
def translate(
    sentences: list[str],
    src_lang: str,       # FLORES-200 code
    tgt_lang: str,
    model_path: str,
    batch_size: int,
    max_new_tokens: int,
    precision: str,      # "fp16" | "fp32" | "int8"
    device: str,
    extra_config: dict | None,
) -> list[str]:
```

### Sarvam Isolation (L1 Guardrail)

```
eval_runner.py (MIT)
    └─ subprocess.run([sys.executable, sarvam/runner.py, input.json, output.json])
            └─ sarvam/runner.py (GPL-3.0)
                   └─ imports sarvamai/sarvam-translate
```

Core harness never `import`s Sarvam code. License boundary enforced at subprocess.

## Code Map

| File | Purpose |
|------|---------|
| `src/indian_models_eval/dataset_loader.py` | FLORES-200 + IN22-Gen loaders |
| `src/indian_models_eval/test_planner.py` | Registry + test matrix generation |
| `src/indian_models_eval/eval_runner.py` | GPU eval dispatcher, VRAM logging |
| `src/indian_models_eval/scoring_pipeline.py` | chrF++, BLEU, COMET-22 |
| `src/indian_models_eval/claims_verifier.py` | Claimed vs viable |
| `src/indian_models_eval/models/indictrans2/runner.py` | IndicTrans2 |
| `src/indian_models_eval/models/nllb/runner.py` | NLLB (CC-BY-NC) |
| `src/indian_models_eval/models/gemma/runner.py` | Gemma (decoder-only) |
| `src/indian_models_eval/models/sarvam/runner.py` | Sarvam (GPL-3.0, subprocess) |

## Notes

- IndicTrans2 requires `IndicTransTokenizer` — not on pip, install from GitHub
- FLORES-200 sentence order is aligned across languages — `zip()` is correct
- IN22-Gen has all languages as columns in a single split
- COMET unreliable langs: `mni_Beng`, `brx_Deva`, `sat_Olck`, `snd_Arab`, `kas_Arab`
- **FLORES split:** flores_plus only has `dev` split (not `devtest`). Default changed from `devtest` → `dev` (997 sentences).
- **FLORES code remap:** `FLORES_PLUS_CODE_MAP` in `dataset_loader.py` maps IndicTrans2 codes to flores_plus codes. Add new entries here for future mismatches.
- **IndicTrans2 Devanagari encoding:** Model encodes ALL Indic scripts into Devanagari internally. En→Indic fix: `UnicodeIndicTransliterator.transliterate(s, "hi", target_lang)` on output. Indic→En fix (pending): transliterate input from native script to Devanagari before tokenization.
- **Token skip:** IndicTrans2 output_ids = `[decoder_start, lang_tag, actual_tokens...]`. Slice `[:, 2:]` before `batch_decode` to avoid garbage leading characters.
- **CUDA fragmentation:** `expandable_segments:True` + `torch.cuda.empty_cache()` + `gc.collect()` after every pair. Model reloaded between pairs to fully defrag.
- **Two-phase eval:** Phase 1 = translate all pairs (COMET off). Phase 2 = clear translation model from GPU, run COMET. Prevents 24GB VRAM exhaustion.
- **Partial save:** `_partial_{model_id}.json` written after each pair. Auto-resume skips completed pairs on restart. Survives OOM and browser close.
- **Background thread:** `threading.Thread(daemon=False)` keeps eval running after browser close. `gr.Timer(value=5)` polls `get_status()` every 5s. `_pause_event`/`_stop_event` control flow.

## Decisions

**Decision:** Sarvam via subprocess, not direct import — **Why:** GPL-3.0 contamination would require core harness to be GPL-3.0. Subprocess keeps license boundary clean. — **Alternatives:** Separate venv (heavier, same effect).

**Decision:** Viability threshold = 40 chrF++ — **Why:** Below 40 is generally considered unusable for production MT. Open question in PRD — can be changed in `config.py:VIABILITY_THRESHOLD`. — **Alternatives:** 30 (more permissive), 50 (stricter).

**Decision:** FLORES-200 `dev` split (997 sentences) — **Why:** flores_plus dataset only has `dev` split. `devtest` doesn't exist in this HF dataset version. Standard MT benchmark split.
