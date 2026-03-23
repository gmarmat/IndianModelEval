# Results Schema

**Status:** planned
**Last updated:** 2026-03-18

## Overview

One JSON file per eval run, saved to `results/` and committed to git. Contains full reproducibility metadata plus per-language scores and claims verdict.

## Filename

```
results/{model_id}_{dataset}_{timestamp}.json
e.g. results/indictrans2-en-indic_flores200_20260318T142300Z.json
```

## Schema (v1.0)

```json
{
  "schema_version": "1.0",
  "harness_version": "2026.03.18.01",
  "timestamp": "2026-03-18T14:23:00Z",

  "model": {
    "id": "indictrans2-en-indic",
    "display_name": "IndicTrans2 (En→Indic)",
    "hf_id": "ai4bharat/indictrans2-en-indic-1B",
    "license": "MIT",
    "revision_hash": "abc123..."
  },

  "eval_config": {
    "datasets": ["flores200"],
    "seed": 42,
    "batch_size": 16,
    "precision": "fp16",
    "max_new_tokens": 256,
    "dry_run": false,
    "viability_threshold_chrf": 40.0
  },

  "reproducibility": {
    "harness_git_hash": "...",
    "model_revision_hash": "...",
    "seed": 42
  },

  "hardware": {
    "gpu": "NVIDIA GeForce RTX 3090",
    "vram_total_mb": 24576.0
  },

  "aggregate": {
    "n_pairs": 22,
    "viable_pairs": 18,
    "chrf_mean": 52.3,
    "chrf_min": 12.1,
    "chrf_max": 68.4,
    "bleu_mean": 31.2,
    "comet_mean": 0.8234,
    "comet_mean_reliable_only": 0.8412,
    "note": "Aggregate scores hide per-language variance..."
  },

  "claims_verification": {
    "model_id": "...",
    "claimed_pairs": 22,
    "viable_pairs": 18,
    "failed_pairs": [...],
    "viability_rate": 0.818,
    "viability_threshold_chrf": 40.0,
    "verdict": "mostly_verified"
  },

  "per_language_pair": [
    {
      "src_lang": "eng_Latn",
      "tgt_lang": "hin_Deva",
      "src_lang_name": "English",
      "tgt_lang_name": "Hindi",
      "dataset": "flores200",
      "chrf": 52.3,
      "bleu": 31.2,
      "comet": 0.8412,
      "viable": true,
      "comet_reliable": true,
      "comet_reliability_note": null,
      "n_sentences": 1012
    }
  ],

  "latency": {
    "per_pair": [
      {
        "src_lang": "eng_Latn",
        "tgt_lang": "hin_Deva",
        "vram_peak_mb": 4200.0,
        "latency_seconds": 42.3,
        "sentences_per_second": 23.9,
        "n_sentences": 1012
      }
    ]
  }
}
```

## Verdict Labels

| Label | Condition |
|-------|-----------|
| `claims_verified` | ≥ 90% of claimed pairs viable |
| `mostly_verified` | 70–90% viable |
| `partially_verified` | 50–70% viable |
| `claims_unverified` | < 50% viable |

## Phase 2 Additions

Phase 2 will add:
- `tokenization.fertility_per_lang` — tokens/word per language
- Additional metric fields: `indic_comet`, `xcomet`
