# IndianModelsEval — Product Requirements Document

**Version:** 0.1 | **Updated:** 2026-03-17 | **Status:** Draft

---

## LLM Quick Start

- **What:** Local eval harness + minimal web UI that independently verifies whether Indian language models actually deliver on their claimed language support
- **Who:** Single operator (project owner) running evaluations on a local RTX 3090 workstation
- **Stack:** Python 3.10+, Gradio UI, HuggingFace Transformers, sacrebleu, unbabel-comet, PyTorch
- **Key constraint:** All inference runs locally on RTX 3090 (24GB VRAM) — no cloud inference, minimal external API calls
- **Data flow:** model_registry.yaml → test_planner → eval_runner (local GPU) → scoring pipeline → results JSON → HF Space leaderboard

---

## Executive Summary

Indian language translation models (IndicTrans2, Sarvam-Translate, NLLB, Gemma-3, Krutrim) each publish their own benchmark numbers on self-selected datasets. No independent third party has verified these claims using a consistent, reproducible methodology. IndianModelsEval is an independent evaluation harness that tests each model against its own stated capabilities, using standardized datasets and metrics, and publishes the results publicly. The output is a credible, reproducible benchmark that lets researchers and practitioners choose models based on independently verified performance — not marketing claims.

### Value Proposition

| Stakeholder | Value |
|-------------|-------|
| ML practitioners | Pick the right model for their language pair without running their own evals |
| Researchers | Reproducible, standardized baseline for Indian MT research |
| Model authors | Independent third-party validation (or correction) of their published numbers |
| Indian AI community | First public leaderboard specifically for Indian language translation claims verification |

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Models evaluated | ≥ 5 models in Phase 1 | Count in results/ |
| Language pairs covered | ≥ 16 En↔Indic + ≥ 6 Indic↔Indic | Pairs in results JSON |
| Results publicly posted | HF Space live + GitHub | Leaderboard URL accessible |
| Reproducibility | Any run with same config produces identical scores | Seed + model revision hash locked |
| Eval run time | Full FLORES-200 eval for 1 model in < 8 hours | Logged in results JSON |

---

## Problem Statement

### Current State

Model authors benchmark against their own baselines on self-selected datasets. Results are scattered across papers, blog posts, and model cards with no standardized methodology. Sarvam-Translate (released June 2025) has zero independent third-party benchmarks. IndicTrans2's numbers are self-reported. No single leaderboard covers Indian language translation with consistent metrics across all major models.

### Pain Points

| # | Pain Point | Who Feels It | Severity |
|---|-----------|-------------|----------|
| 1 | Can't compare models apples-to-apples — different datasets, metrics, conditions | Practitioners | High |
| 2 | Model claims are unverified — no independent replication exists | Researchers | High |
| 3 | No coverage of Indic↔Indic translation (e.g. Tamil→Hindi) | Practitioners | High |
| 4 | COMET scores unreliable for low-resource langs (Manipuri, Bodo) but never disclosed | Researchers | Medium |
| 5 | License implications (GPL-3.0 Sarvam, CC-BY-NC NLLB) not surfaced alongside results | Practitioners | Medium |
| 6 | No tokenization efficiency data — can't predict VRAM/speed for a given language | MLOps engineers | Medium |

### Desired State

A researcher or practitioner can visit the leaderboard, see independently verified scores for each model across claimed language pairs, understand viability (not just raw scores), and make an informed model choice in minutes. Results are fully reproducible from the published config + seeds.

---

## Target Users & Roles

| Role | Description | Access |
|------|-------------|--------|
| **Operator** (project owner) | Runs evaluations locally on RTX 3090, publishes results | Full — local app + publish |
| **Leaderboard reader** | Views published results on HF Space | Read-only — HF Space |
| **Community contributor** | (Phase 3) Submits their own model results via PR | PR to GitHub results/ |

### Access Model

Local app: single-user, no auth. HF Space leaderboard: public read-only. No login, no accounts.

---

## Boundaries

| Category | Rule |
|----------|------|
| **ALWAYS** | Run --dry-run before any eval. Log model revision hash + seed in every results JSON. Show per-language breakdown alongside aggregate scores. Flag COMET reliability warnings for low-resource langs. |
| **ASK FIRST** | Adding a new model to the registry. Changing the results JSON schema (breaks existing results). Enabling any API-based inference (cost implications). |
| **NEVER** | Commit model weights, datasets, or API keys to git. Run eval without pinning model revision hash. Publish aggregate-only scores without per-language breakdown. Use shell=True in subprocess calls. Hardcode API keys in config files. |

---

## Core Concepts

| Concept | Definition | Relates To |
|---------|-----------|------------|
| **Model Registry** | `model_registry.yaml` — each model's claimed languages, license, HF ID, local path | Test Planner |
| **Test Plan** | Per-model matrix of language pairs to test, derived from claimed capabilities | Eval Runner |
| **Translation Model** | Model being evaluated (IndicTrans2, Sarvam, etc.) — runs locally on GPU | Eval Runner |
| **Scoring Model** | Learned metric model (COMET-22, xCOMET, Indic-COMET) — runs after translation | Scoring Pipeline |
| **Claims Verification** | Comparison of model's claimed capabilities vs measured viability threshold (≥40 chrF++) | Results JSON |
| **Viability** | A language pair is "viable" if it scores ≥40 chrF++ — below this is effectively unusable in practice | Leaderboard |
| **Results JSON** | One file per run: `results/{model}_{dataset}_{timestamp}.json` — committed to git | Leaderboard, Publish |
| **Reliability Flag** | Warning attached to COMET scores for langs where the metric is unreliable (Manipuri, Bodo, Santhali) | Results JSON, UI |

---

## Core Features

### Phase 1 — MVP: Run evals, see results

**Goal:** Operator can evaluate any registered model against FLORES-200 + IN22-Gen on its claimed language pairs and view results locally.

| # | Feature | Acceptance Criteria | Priority |
|---|---------|-------------------|----------|
| 1.1 | **Model Registry** | `model_registry.yaml` defines 5 models with claimed languages, license, HF model ID, local path. App reads and validates on startup. | P0 |
| 1.2 | **Dataset Loader** | Downloads + caches FLORES-200 and IN22-Gen to `data/`. Validates checksums. Handles re-download gracefully. | P0 |
| 1.3 | **Test Planner** | Reads model registry → generates test matrix (language pairs × datasets) for the selected model only | P0 |
| 1.4 | **Eval Runner** | Runs translation on local GPU. Accepts `--dry-run` (5 sentences/lang, reports est. time + VRAM). Single model at a time. Logs VRAM peak per language pair. | P0 |
| 1.5 | **Scoring Pipeline** | Computes chrF++ + BLEU (sacrebleu) + COMET-22 per language pair. Attaches reliability flags for low-resource langs. | P0 |
| 1.6 | **Claims Verifier** | Compares results vs claimed languages. Outputs viable count. Flags pairs below viability threshold (< 40 chrF++). | P0 |
| 1.7 | **Results Writer** | Saves `results/{model}_{dataset}_{timestamp}.json` with full metadata: model revision hash, seed, hardware, batch size, precision, all scores, claimed vs viable. | P0 |
| 1.8 | **Sarvam Isolation** | Sarvam-Translate eval code in `src/models/sarvam/` with GPL-3.0 header, called via subprocess only — core harness stays MIT. | P0 |
| 1.9 | **Local Web UI — Setup Tab** | Point to `models/translation/` folder. Point to `models/scoring/` folder. Download datasets button. HF token entry (stored in .env, never committed). Validate paths on entry. | P1 |
| 1.10 | **Local Web UI — Run Tab** | Model dropdown (from registry). Dataset checkboxes. Language pair multi-select (filtered to model's claimed langs). Dry-run button → shows estimate. Run button → live progress (model, language, ETA, VRAM gauge). | P1 |
| 1.11 | **Local Web UI — Results Tab** | Table of all completed runs. Per-language breakdown. Claimed vs Viable summary. Sort/filter by model, dataset, language. | P1 |
| 1.12 | **Pre-commit hooks** | detect-secrets blocks API key commits. .gitignore covers weights, datasets, .env. | P1 |

### Phase 2 — Richer Metrics + Indic↔Indic

**Goal:** Upgrade metric suite to Indic-specific models; add Indic↔Indic pairs; add tokenization efficiency data.
**Depends on:** Phase 1 complete

| # | Feature | Acceptance Criteria | Priority |
|---|---------|-------------------|----------|
| 2.1 | **Indic-COMET + xCOMET** | Install via WSL2 if needed. Run alongside COMET-22. Surface Indic-COMET as headline neural metric. | P0 |
| 2.2 | **Indic↔Indic language pairs** | Add 6+ pairs (Hi↔Ta, Hi↔Bn, Hi↔Te, Hi↔Mr, Hi↔Ml, Ta↔Bn) to test planner. Only for models that claim these pairs. | P0 |
| 2.3 | **Tokenization Fertility Logger** | Reports tokens/word per model per language. Logged in results JSON under `tokenization.fertility_per_lang`. | P1 |
| 2.4 | **IndicGenBench dataset** | Add as third dataset option alongside FLORES-200 and IN22-Gen. | P1 |
| 2.5 | **Queue Mode** | UI option to queue multiple models for overnight sequential runs. Each model clears GPU memory before next. | P1 |
| 2.6 | **Publish Tab** | Push selected results JSON to GitHub. Triggers HF Space leaderboard update. Shows diff of what will be published before confirming. | P1 |

### Phase 3 — Community + Extended Coverage

**Goal:** Extend language coverage; enable community to submit results; post to secondary platforms.
**Depends on:** Phase 2 complete

| # | Feature | Acceptance Criteria | Priority |
|---|---------|-------------------|----------|
| 3.1 | **Extended languages** | Add Punjabi, Odia, Urdu, Assamese to test planner | P1 |
| 3.2 | **Low-resource spot check** | Add Manipuri, Bodo with explicit COMET reliability warnings | P2 |
| 3.3 | **Community submission harness** | Standardized inference script in repo. Others can run it and submit results JSON via GitHub PR. | P2 |
| 3.4 | **Papers With Code submission** | Export results in PWC-compatible format. Documented submission process. | P2 |

---

## Non-Functional Requirements

| Category | Requirement | Target |
|----------|-------------|--------|
| **Reproducibility** | Same config + seed → identical scores | Enforced by pinning model revision hash + all seeds |
| **VRAM safety** | Never OOM-crash mid-eval | Dry-run validates VRAM headroom before full run; `torch.cuda.empty_cache()` between models |
| **Cost** | No surprise API charges | --dry-run mandatory; API cost cap at $20 hard limit; local-first by default |
| **Security** | No secrets in git | detect-secrets pre-commit hook; Pydantic config schema (no secrets in config files) |
| **Portability** | Runs on Windows 11 + RTX 3090 | Primary target. WSL2 used for Linux-native metric libraries (Indic-COMET) if needed. |
| **Performance** | Full FLORES-200 eval (1 model) | < 8 hours on RTX 3090 |

---

## Data Sources & Integrations

| Source | Type | Purpose | Auth | Notes |
|--------|------|---------|------|-------|
| FLORES-200 | HuggingFace dataset | Primary MT benchmark | HF token | CC-BY-SA 4.0 — must attribute Meta |
| IN22-Gen | HuggingFace dataset | Indian language MT benchmark | HF token | CC0 — no restrictions |
| IndicGenBench | HuggingFace dataset | Generation benchmark (Phase 2) | HF token | MIT |
| Translation models | Local files in `models/translation/` | Models under evaluation | None (local) | User downloads manually |
| Scoring models | Local files in `models/scoring/` | COMET-22, xCOMET, Indic-COMET | None (local) | User downloads manually |
| HuggingFace Space | Git push | Public leaderboard | HF token | Free tier — static leaderboard from results JSON |

### Data Flow

```
model_registry.yaml
        │
        ▼
test_planner.py          ← reads claimed languages per model
        │
        ▼
eval_runner.py           ← loads translation model from models/translation/
        │                   translates FLORES-200 / IN22-Gen sentences
        │                   logs VRAM, latency, fertility
        ▼
scoring_pipeline.py      ← loads scoring models from models/scoring/
        │                   computes chrF++, BLEU, COMET-22 (+ xCOMET, Indic-COMET in Phase 2)
        │                   attaches reliability flags
        ▼
claims_verifier.py       ← claimed langs vs viable langs (≥40 chrF++)
        │
        ▼
results/{model}_{dataset}_{timestamp}.json   ← committed to git
        │
        ▼
HF Space leaderboard     ← reads results JSON on push, renders table
```

---

## Cost Model

| Category | Cost | Notes |
|----------|------|-------|
| **Compute** | ~$1/full eval run | Electricity only — RTX 3090 at 350W × ~6h |
| **Storage** | ~$0 | Models stored locally; results JSON tiny |
| **HF Space** | $0 | Free tier sufficient for static leaderboard |
| **Scoring models** | $0 ongoing | One-time download, run locally |
| **API calls** | $0 default | Local-first; API mode requires explicit flag + $20 hard cap |
| **Total monthly** | ~$5–10 | Electricity for regular eval runs |

### Free Tier Limits

HF Space free tier: 2 vCPU, 16GB RAM. Fine — leaderboard is a static table rendered from JSON, no GPU needed on the Space.

---

## Out of Scope

| Feature | Why Not Now | Target Version |
|---------|-----------|----------------|
| Gender / caste / dialect bias evaluation | Requires human annotation, different methodology | Never (separate project) |
| Speech / ASR evaluation | Different model type entirely | Never |
| Summarization / QA tasks | Out of MT scope | Never |
| Cloud inference / API-based model eval | Cost and reproducibility concerns | v3 only if explicitly requested |
| Real-time / streaming eval | Not needed for batch benchmark | Never |
| Multi-user app | Single operator only | Never |
| Domain-specific benchmarks (legal, medical) | No standardized datasets available yet | v3 |
| Sanskrit as primary language | Model support minimal; included as spot-check only | v1 (spot-check) |

---

## Open Questions

| # | Question | Impact | Status |
|---|----------|--------|--------|
| 1 | Is Krutrim Translate available as local weights or API only? | Blocks adding it to model registry | Open |
| 2 | Should HF Space leaderboard accept community PR submissions from day one or Phase 3 only? | Affects leaderboard UI design | Open |
| 3 | What design system will the UI use? | Affects all UI component decisions | Open — user will provide |
| 4 | Can Indic-COMET be installed natively on Windows or does it require WSL2? | Affects Phase 2 metric pipeline architecture | Open |
| 5 | Exact viability threshold — is 40 chrF++ the right cutoff for "usable"? | Affects claims verifier logic and leaderboard labels | Open |

---

## Dependencies

| Dependency | Risk | Mitigation |
|-----------|------|------------|
| Sarvam-Translate local weights availability on HuggingFace | Medium — GPL-3.0, may have gated access | Check HF model page; have API fallback |
| Indic-COMET pip installation on Windows | High — likely Linux-only library | WSL2 fallback; document setup |
| FLORES-200 + IN22-Gen HuggingFace availability | Low — stable datasets | Local cache after first download |
| RTX 3090 VRAM sufficient for all models | Low — largest model ~8GB in fp16 | Dry-run validates headroom; configurable batch size |

---

## Technical References

| Doc | Purpose | Status |
|-----|---------|--------|
| `docs/arch.md` | Architecture index — implementation source of truth | To be created |
| `docs/features/eval-pipeline.md` | Eval runner + scoring pipeline deep dive | To be created |
| `docs/features/ui.md` | UI tabs, components, design system integration | To be created |
| `docs/features/model-registry.md` | Registry schema, adding new models | To be created |
| `docs/features/results-schema.md` | Results JSON schema, all fields | To be created |
| `docs/features/publishing.md` | HF Space setup, Papers With Code process | To be created |
| `docs/plans/2026-03-17-project-improvements.md` | Research: scope improvements (Indic-COMET, Indic↔Indic) | Done |
| `docs/plans/2026-03-17-security-safety-audit.md` | Legal, cost, security, harm guardrails | Done |

---

## Revision History

| Ver | Date | Changes |
|-----|------|---------|
| 0.1 | 2026-03-17 | **Initial draft.** Full ideation captured: claims verifier framing, Phase 1-3 features, language scope, UI tabs, publishing plan, guardrails. |
