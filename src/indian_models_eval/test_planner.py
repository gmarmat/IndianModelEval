"""
Model registry loader and test planner.
Reads model_registry.yaml → generates the (src, tgt, dataset) test matrix.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import yaml

from .config import DatasetName, EvalConfig


class LangPair(NamedTuple):
    src: str
    tgt: str


class TestCase(NamedTuple):
    src_lang: str
    tgt_lang: str
    dataset: DatasetName


class ModelEntry:
    def __init__(self, data: dict) -> None:
        self.id: str = data["id"]
        self.display_name: str = data["display_name"]
        self.hf_id: str = data["hf_id"]
        self.local_path: Path = Path(data["local_path"])
        self.model_type: str = data["type"]  # seq2seq | decoder_only
        self.license: str = data["license"]
        self.direction: str = data["direction"]
        self.isolation: str = data.get("isolation", "direct")  # direct | subprocess
        self.claimed_languages: dict = data["claimed_languages"]
        self.eval_config: dict = data.get("eval_config", {})

    @property
    def claimed_pairs(self) -> list[LangPair]:
        """All (src, tgt) pairs this model claims to support.

        If the registry entry has an explicit `pairs` list (for models that don't
        support all src×tgt combinations), use that directly. Otherwise fall back
        to the cartesian product of src × tgt (excluding self-pairs).
        """
        explicit: list | None = self.claimed_languages.get("pairs")
        if explicit is not None:
            return [LangPair(src=p[0], tgt=p[1]) for p in explicit]

        src_langs: list[str] = self.claimed_languages.get("src", [])
        tgt_langs: list[str] = self.claimed_languages.get("tgt", [])
        pairs = []
        for src in src_langs:
            for tgt in tgt_langs:
                if src != tgt:
                    pairs.append(LangPair(src=src, tgt=tgt))
        return pairs

    @property
    def requires_subprocess(self) -> bool:
        return self.isolation == "subprocess"


class ModelRegistry:
    def __init__(self, registry_path: Path | str) -> None:
        self._path = Path(registry_path)
        self._models: dict[str, ModelEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Registry not found: {self._path}")
        with self._path.open() as f:
            data = yaml.safe_load(f)
        for entry in data["models"]:
            model = ModelEntry(entry)
            self._models[model.id] = model

    def get(self, model_id: str) -> ModelEntry:
        if model_id not in self._models:
            available = list(self._models.keys())
            raise KeyError(f"Model {model_id!r} not in registry. Available: {available}")
        return self._models[model_id]

    @property
    def all_ids(self) -> list[str]:
        return list(self._models.keys())

    @property
    def all_models(self) -> list[ModelEntry]:
        return list(self._models.values())


class TestPlanner:
    """Generates the test matrix for a model from its claimed language pairs."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def plan(self, config: EvalConfig) -> list[TestCase]:
        """Return list of TestCases to run based on config."""
        model = self._registry.get(config.model_id)

        if config.language_pairs is not None:
            pairs = [LangPair(src=s, tgt=t) for s, t in config.language_pairs]
        else:
            pairs = model.claimed_pairs

        test_cases: list[TestCase] = []
        for pair in pairs:
            for dataset in config.datasets:
                test_cases.append(
                    TestCase(src_lang=pair.src, tgt_lang=pair.tgt, dataset=dataset)
                )

        return test_cases

    def summary(self, config: EvalConfig) -> str:
        cases = self.plan(config)
        model = self._registry.get(config.model_id)
        n_pairs = len({(c.src_lang, c.tgt_lang) for c in cases})
        n_datasets = len(config.datasets)
        return (
            f"Model: {model.display_name}\n"
            f"Language pairs: {n_pairs}\n"
            f"Datasets: {', '.join(config.datasets)}\n"
            f"Total test cases: {len(cases)}"
        )
