"""HuggingFace 데이터셋 로더 래퍼.

v3 문서(워크로드_데이터셋_가이드_v3.md) §6 의 매핑을 재현:
- SAFIM, HumanEval+, MBPP+, LiveCodeBench, Aider Polyglot, CrossCodeEval,
  RepoBench, SWE-bench Verified, Multi-SWE-bench 등.

외부 `datasets>=2.18` 의존이 없는 경우 NotImplementedError 로 실패하며,
테스트에서는 `load_hf_dataset` 을 monkeypatch 하여 fixture 로 주입한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class DatasetSample:
    prompt: str
    response: str | None = None
    task_id: str | None = None
    extra: dict[str, Any] | None = None


KNOWN_DATASETS = {
    # slug → (hf id, default split, default subset)
    "safim": ("gonglinyuan/safim", "test", "block"),
    "humaneval_plus": ("evalplus/humanevalplus", "test", None),
    "mbpp_plus": ("evalplus/mbppplus", "test", None),
    "livecodebench": ("livecodebench/code_generation", "test", None),
    "aider_polyglot": ("Aider-AI/polyglot-benchmark", "test", None),
    "crosscodeeval": ("neulab/CrossCodeEval", "test", None),
    "repobench_c": ("tianyang/repobench-c-v2", "test", None),
    "swe_bench_verified": ("princeton-nlp/SWE-bench_Verified", "test", None),
    "multi_swe_bench": ("bytedance-research/Multi-SWE-bench", "test", None),
}


class DatasetLoader:
    def __init__(
        self,
        dataset_id: str,
        split: str = "test",
        subset: str | None = None,
        prompt_field: str = "prompt",
        response_field: str | None = None,
        task_id_field: str | None = "task_id",
    ):
        self.dataset_id = dataset_id
        self.split = split
        self.subset = subset
        self.prompt_field = prompt_field
        self.response_field = response_field
        self.task_id_field = task_id_field

    @classmethod
    def from_slug(cls, slug: str, **kwargs) -> "DatasetLoader":
        if slug not in KNOWN_DATASETS:
            raise KeyError(f"unknown dataset slug: {slug} (known: {sorted(KNOWN_DATASETS)})")
        hf_id, split, subset = KNOWN_DATASETS[slug]
        return cls(dataset_id=hf_id, split=split, subset=subset, **kwargs)

    def iter_samples(self, limit: int | None = None) -> Iterable[DatasetSample]:
        raw = load_hf_dataset(self.dataset_id, self.subset, self.split)
        count = 0
        for row in raw:
            prompt = row.get(self.prompt_field) or row.get("instruction") or row.get("problem") or ""
            if not isinstance(prompt, str):
                prompt = str(prompt)
            response = row.get(self.response_field) if self.response_field else None
            task_id = row.get(self.task_id_field) if self.task_id_field else None
            yield DatasetSample(
                prompt=prompt,
                response=response if isinstance(response, (str, type(None))) else str(response),
                task_id=str(task_id) if task_id is not None else None,
                extra={k: v for k, v in row.items() if k not in {self.prompt_field, self.response_field, self.task_id_field}},
            )
            count += 1
            if limit is not None and count >= limit:
                break


def load_hf_dataset(dataset_id: str, subset: str | None, split: str) -> Iterable[dict]:
    """HF datasets 패키지 런타임 lazy import — 미설치 시 안내 메시지."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise NotImplementedError(
            "HuggingFace `datasets` 미설치. `pip install datasets` 후 재시도."
        ) from e
    return load_dataset(dataset_id, subset, split=split)
