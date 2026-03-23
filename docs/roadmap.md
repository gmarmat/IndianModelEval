# IndianModelsEval — Roadmap

## In Progress
- Krutrim Translate eval (first independent benchmark)

## Up Next
- NLLB full eval — all 14 claimed language pairs (run overnight)
- IndicTrans2 weights — request access at huggingface.co/ai4bharat/indictrans2-en-indic-1B

## Backlog

### Models to Add
| Model | Source | Status | Notes |
|-------|--------|--------|-------|
| IndicTrans2 en→Indic | HuggingFace (gated) | Access needed | Best known Indian MT model |
| IndicTrans2 Indic→en | HuggingFace (gated) | Access needed | |
| Sarvam-Translate | HuggingFace (gated) | Access needed | GPL-3.0, subprocess isolation required |
| Gemma-3-4B | HuggingFace | Pending | Decoder-only, prompt-based |
| Krutrim-1 / Krutrim-2 | AIKosh | Investigate | LLMs, not MT-specific — prompt-based eval |

### Eval Coverage
- Run all 14 NLLB claimed pairs (not just 3)
- Add IN22-Gen dataset evals alongside FLORES-200
- Indic→English direction for all models

### Infrastructure
- Publish results as HuggingFace Dataset (public, citable)
- Publish leaderboard as HuggingFace Space
- Add git repo + commit history for reproducibility

### Future Models (AIKosh)
- Dhwani (speech LLM) — different task, future series
- Chitrarth (vision-language) — different task, future series
- A2TTS models — TTS eval requires different metrics (MOS), separate project

### LinkedIn Content Pipeline
- Post 1: "I built an independent benchmark for Indian AI translation models"
- Post 2: "Krutrim claims X — here's what I actually measured"
- Post 3: "IndicTrans2 vs Krutrim vs NLLB — the scorecard"
- Post 4: "Why self-reported AI benchmarks can't be trusted"
