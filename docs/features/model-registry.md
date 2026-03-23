# Model Registry

**Status:** planned
**Last updated:** 2026-03-18

## Overview

`model_registry.yaml` at project root defines all models under evaluation. The registry is the single source of truth for claimed language pairs, model paths, licenses, and eval config.

## Requirements

- [ ] 5 models in Phase 1 registry (IndicTrans2 en-indic, indic-en, Sarvam, NLLB, Gemma)
- [ ] Each model has: id, hf_id, local_path, type, license, claimed_languages, eval_config
- [ ] App validates registry on startup (fails fast if malformed)
- [ ] Adding a new model requires confirming local weights exist (PRD guardrail: ASK FIRST)

## Schema

```yaml
models:
  - id: string                  # unique identifier, used in filenames
    display_name: string        # human-readable
    hf_id: string               # HuggingFace model ID
    local_path: string          # path under models/translation/
    type: seq2seq | decoder_only
    license: string             # MIT | GPL-3.0 | CC-BY-NC-4.0 | Gemma
    direction: en_to_indic | indic_to_en | bidirectional | multilingual
    isolation: direct | subprocess   # subprocess = GPL-3.0 isolated
    claimed_languages:
      src: [flores_code, ...]
      tgt: [flores_code, ...]
    eval_config:
      batch_size: int
      max_new_tokens: int
      precision: fp16 | fp32 | int8
      prompt_template: string   # decoder_only only
      notes: string
```

## FLORES-200 Language Codes

| Code | Language | Script |
|------|----------|--------|
| eng_Latn | English | Latin |
| hin_Deva | Hindi | Devanagari |
| ben_Beng | Bengali | Bengali |
| tam_Taml | Tamil | Tamil |
| tel_Telu | Telugu | Telugu |
| mar_Deva | Marathi | Devanagari |
| guj_Gujr | Gujarati | Gujarati |
| kan_Knda | Kannada | Kannada |
| mal_Mlym | Malayalam | Malayalam |
| pan_Guru | Punjabi | Gurmukhi |
| ory_Orya | Odia | Odia |
| urd_Arab | Urdu | Arabic |
| asm_Beng | Assamese | Bengali |
| mai_Deva | Maithili | Devanagari |
| npi_Deva | Nepali | Devanagari |
| brx_Deva | Bodo | Devanagari |
| mni_Beng | Manipuri | Bengali |
| sat_Olck | Santhali | Ol Chiki |

## Adding a New Model

1. Check local weights are available in `models/translation/`
2. Add entry to `model_registry.yaml`
3. If GPL/AGPL: set `isolation: subprocess` and add runner in `models/{name}/runner.py`
4. Run `python -c "from src.indian_models_eval.test_planner import ModelRegistry; ModelRegistry('model_registry.yaml')"` to validate

## Code Map

| File | Purpose |
|------|---------|
| `model_registry.yaml` | Registry data |
| `src/indian_models_eval/test_planner.py` | `ModelRegistry` + `TestPlanner` |
| `src/indian_models_eval/config.py` | `FLORES_LANG_CODES` mapping |
