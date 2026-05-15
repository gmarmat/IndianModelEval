# Contributing to IndianModelEval

Thanks for your interest. This project welcomes evaluation results for any Indian-language translation model, plus fixes, new metrics, and dataset additions.

## Add a new model

1. **Register it.** Add an entry to `model_registry.yaml` with `hf_id`, `local_path`, `license`, `claimed_languages`, and `eval_config`. Follow the schema of existing entries.
2. **Implement the runner.** Add `src/indian_models_eval/models/<model_name>/runner.py` exposing `translate(sentences, src_lang, tgt_lang, model_path, **kwargs) -> list[str]`.
3. **Wire it in.** Register the runner in `eval_runner.py`.
4. **Dry-run.** Use the Gradio UI's "Dry run (5 sentences)" checkbox or `python scripts/run_eval_cli.py --model <id> --dry-run` to smoke-test.
5. **Full run.** `python scripts/run_eval_cli.py --model <id>`. The result JSON lands in `results/<model>_<dataset>_<timestamp>.json`.
6. **Open a PR.** Include the result JSON, the HuggingFace commit hash of the model revision tested, and your hardware spec. Reviewers will reproduce against the registry config before merging.

## Re-run an existing model

If you re-run with a newer model revision, name the result with the new timestamp. Keep older runs in `results/` for diff-ability across revisions.

## Code changes

- Use `ruff` for lint (config in `pyproject.toml`). Pre-commit hooks enforce it.
- Type hints required (Python 3.10+).
- No new dependencies without a note in the PR description.

## Discussion first

Open an issue for any large change (new metric, new dataset family, new model class). Quick fixes can go straight to PR.

## License of contributions

By submitting a PR you agree to license your contribution under the same MIT license as the rest of the harness. Model weights remain under their respective upstream licenses.
