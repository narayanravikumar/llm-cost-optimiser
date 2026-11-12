"""Load RAGTruth and build the judge calibration set and the pipeline evaluation set."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "RAGTruth" / "dataset"
TASKS = ("QA", "Summary")


@dataclass(frozen=True)
class Source:
    source_id: str
    task_type: str
    prompt: str


@dataclass(frozen=True)
class LabelledResponse:
    id: str
    source: Source
    model: str
    response: str
    hallucinated: bool
    spans: tuple[str, ...]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def load_sources() -> dict[str, Source]:
    rows = _read_jsonl(DATA_DIR / "source_info.jsonl")
    return {
        r["source_id"]: Source(r["source_id"], r["task_type"], r["prompt"])
        for r in rows
        if r["task_type"] in TASKS
    }


def load_labelled(split: str = "test") -> list[LabelledResponse]:
    sources = load_sources()
    out = []
    for r in _read_jsonl(DATA_DIR / "response.jsonl"):
        src = sources.get(r["source_id"])
        if src is None or r["split"] != split:
            continue
        out.append(
            LabelledResponse(
                id=r["id"],
                source=src,
                model=r["model"],
                response=r["response"],
                hallucinated=bool(r["labels"]),
                spans=tuple(l["text"] for l in r["labels"]),
            )
        )
    return out


def judge_calibration_set(n: int = 200, seed: int = 7) -> list[LabelledResponse]:
    """Balanced set: half hallucinated, half clean, split evenly across tasks."""
    rng = random.Random(seed)
    rows = load_labelled("test")
    picked: list[LabelledResponse] = []
    per_cell = n // (2 * len(TASKS))
    for task in TASKS:
        for flag in (True, False):
            cell = [r for r in rows if r.source.task_type == task and r.hallucinated == flag]
            picked += rng.sample(cell, per_cell)
    rng.shuffle(picked)
    return picked


def pipeline_eval_set(n: int = 60, seed: int = 11) -> list[Source]:
    """Sources the optimiser runs each candidate configuration on."""
    rng = random.Random(seed)
    test_ids = {r.source.source_id for r in load_labelled("test")}
    sources = [s for s in load_sources().values() if s.source_id in test_ids]
    per_task = n // len(TASKS)
    picked = []
    for task in TASKS:
        picked += rng.sample([s for s in sources if s.task_type == task], per_task)
    return picked


if __name__ == "__main__":
    cal = judge_calibration_set()
    ev = pipeline_eval_set()
    print(f"judge calibration set: {len(cal)} responses, "
          f"{sum(r.hallucinated for r in cal)} hallucinated")
    print(f"pipeline eval set: {len(ev)} sources "
          f"({sum(s.task_type == 'QA' for s in ev)} QA, {sum(s.task_type == 'Summary' for s in ev)} Summary)")
