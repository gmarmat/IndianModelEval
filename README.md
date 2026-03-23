# IndianModelEval

**An open evaluation harness for Indian language translation models.**

IndianModelEval is a reproducible, community-friendly benchmark framework for measuring how well AI models translate to and from Indian languages. Results are hardware-logged, dataset-pinned, and published as structured JSON — making them comparable, citable, and easy to contribute back to the community.

The goal is simple: independent, verifiable numbers that anyone with a capable GPU can reproduce and extend.

---

## Why This Exists

Indian language AI has seen remarkable progress, but published benchmarks are often self-reported by model developers, run on unspecified hardware, or evaluated on held-out datasets inaccessible to the public. This project aims to close that gap by building an open evaluation pipeline that:

- Runs on **publicly available datasets** (FLORES-200, IN22-Gen)
- Uses **standard, reproducible metrics** (chrF++, BLEU, COMET-22)
- Logs **hardware and software versions** so results can be compared across machines
- Publishes **raw results as JSON** — auditable, forkable, contributable

If you have a GPU and want to contribute evaluation results for a model that isn't yet covered, open a PR.

---

## Models Evaluated

| Model | Direction | Viable Pairs | chrF++ Mean | Verdict |
|-------|-----------|-------------|-------------|---------|
| [NLLB-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | En→Indic | 142/182 | 43.4 | Mostly verified |
| [IndicTrans2-1B](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B) | En→Indic | 13/22 | 40.8 | Partially verified |
| [IndicTrans2-1B](https://huggingface.co/ai4bharat/indictrans2-indic-en-1B) | Indic→En | 11/22 | 40.5 | Partially verified |
| [Krutrim Translate](https://huggingface.co/krutrim-ai-labs/KrutrimTranslate) | En→Indic | 3/3 tested | 54.6 | Unverified (90 claimed) |

> A pair is **viable** if chrF++ ≥ 40. COMET scores use `Unbabel/wmt22-comet-da`.

Full per-language breakdowns are in [`results/`](./results/).

---

## Languages Covered

All evaluations use **FLORES-200** language codes. Current coverage spans 22 Indian languages:

Assamese · Bengali · Bodo · Dogri · Gujarati · Hindi · Kannada · Kashmiri · Konkani · Maithili · Malayalam · Manipuri · Marathi · Nepali · Odia · Punjabi · Sanskrit · Santhali · Sindhi · Tamil · Telugu · Urdu

---

## Metrics

| Metric | Tool | Notes |
|--------|------|-------|
| **chrF++** | sacrebleu | Primary viability threshold: ≥ 40 |
| **BLEU** | sacrebleu | Secondary reference |
| **COMET-22** | unbabel-comet (`wmt22-comet-da`) | Marked unreliable for low-resource scripts |

COMET reliability warnings are attached to results for languages where the model has limited training coverage (Manipuri, Santhali, Bodo, Sindhi, Kashmiri).

---

## Datasets

| Dataset | Source | Split | Sentences/lang |
|---------|--------|-------|----------------|
| FLORES-200 | `openlanguagedata/flores_plus` | `dev` | ~997 |
| IN22-Gen | `ai4bharat/IN22-Gen` | `gen` | ~1024 |

Datasets are downloaded once and cached locally. No dataset files are committed to this repo.

---

## Project Structure

```
IndianModelEval/
├── app.py                          # Gradio UI (Run Eval / Results / Setup)
├── model_registry.yaml             # Model definitions and claimed language pairs
├── pyproject.toml                  # Dependencies (uv/pip)
├── src/indian_models_eval/
│   ├── config.py                   # EvalConfig, language code maps
│   ├── dataset_loader.py           # FLORES-200 + IN22-Gen loaders
│   ├── eval_runner.py              # GPU eval dispatcher, VRAM logging
│   ├── scoring_pipeline.py         # chrF++ + BLEU + COMET-22
│   ├── claims_verifier.py          # Claimed vs measured viability
│   ├── results_writer.py           # Structured JSON output
│   └── models/
│       ├── indictrans2/runner.py   # IndicTrans2 (MIT)
│       ├── nllb/runner.py          # NLLB-200 (CC-BY-NC)
│       ├── gemma/runner.py         # Gemma-3 (Gemma license)
│       └── sarvam/runner.py        # Sarvam-Translate (GPL-3.0, subprocess-isolated)
├── results/                        # JSON result files (committed)
└── scripts/
    └── download_datasets.py        # One-time dataset download
```

---

## Running It Yourself

### Requirements

- Python 3.10+
- NVIDIA GPU with ≥ 16 GB VRAM recommended (tested on RTX 3090 24 GB)
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

```bash
git clone https://github.com/gmarm/IndianModelEval
cd IndianModelEval

# Install dependencies
uv sync
# or: pip install -e .

# Copy and fill in your HuggingFace token (needed for gated models)
cp .env.example .env

# Download datasets (one-time, ~2 GB)
python scripts/download_datasets.py

# Download model weights (see model_registry.yaml for HF model IDs)
huggingface-cli download ai4bharat/indictrans2-en-indic-1B \
  --local-dir models/translation/indictrans2-en-indic-1B

# Launch the UI
python app.py
```

Open http://localhost:7860 — select a model, pick language pairs, click **Run Eval**.

---

## Contributing Results

The most valuable contribution is running an evaluation on hardware we don't have and submitting the results JSON.

1. Fork this repo
2. Run the eval for a model (existing or new) on your hardware
3. Add the result JSON to `results/`
4. Open a PR with a brief description of your hardware and model version

Result files are small (< 100 KB), human-readable JSON and include hardware metadata, dataset versions, random seeds, and per-language scores.

### Adding a New Model

1. Add an entry to `model_registry.yaml` with the model's HuggingFace ID, local path, and claimed language pairs
2. Add a runner in `src/indian_models_eval/models/<model_name>/runner.py` implementing `translate(sentences, src_lang, tgt_lang, model_path, **kwargs) -> list[str]`
3. Wire it up in `eval_runner.py`
4. Run a dry run (`Dry run (5 sentences)` checkbox) to verify, then a full eval

---

## Reproducibility

Every result file records:

```json
{
  "reproducibility": {
    "seed": 42,
    "model_revision": "abc123...",
    "dataset_version": "openlanguagedata/flores_plus@main",
    "harness_version": "2026.03.22.01"
  },
  "hardware": {
    "gpu": "NVIDIA GeForce RTX 3090",
    "vram_total_mb": 24576
  }
}
```

---

## License

Code: **MIT**

Model weights are subject to their respective licenses (see `model_registry.yaml`). Sarvam-Translate is GPL-3.0 and runs in a subprocess-isolated runner to maintain license boundary separation.

Dataset licenses: FLORES-200 is CC-BY-SA-4.0.
