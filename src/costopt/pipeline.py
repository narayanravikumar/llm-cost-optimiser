"""Pipeline runner: run one configuration of the RAG answering pipeline over the evaluation set
and measure the three objectives (faithfulness, cost, latency).

Usage:
    uv run python -m src.costopt.pipeline --check          # 1 tiny call per model/effort (~$0.01)
    uv run python -m src.costopt.pipeline --dry-run        # estimate cost of the baseline
    uv run python -m src.costopt.pipeline --baseline       # evaluate the client's current setup
    uv run python -m src.costopt.pipeline --config model=haiku prompt=grounded context=35 max_tokens=256 effort=low
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import DATA_DIR, Source, pipeline_eval_set
from .judge import JUDGE_VERSION, judge
from .pricing import MODELS, cost_usd

ROOT = Path(__file__).resolve().parents[2]
EVALS = ROOT / "runs" / "evaluations_v2.jsonl"  # v2: quality = complete and faithful
JUDGE_MODEL = MODELS["haiku"]  # ADR 0003

# ---------------------------------------------------------------- configuration space

SPACE = {
    "model": ["haiku", "sonnet", "opus"],
    "prompt": ["plain", "grounded", "extract"],
    "context": ["full", "60", "35"],  # % of the reference material kept (ADR 0004)
    "max_tokens": [256, 512],
    "effort": ["low", "high"],
}


@dataclass(frozen=True)
class Config:
    model: str
    prompt: str
    context: str
    max_tokens: int
    effort: str

    @property
    def id(self) -> str:
        return f"{self.model}|{self.prompt}|{self.context}|{self.max_tokens}|{self.effort}"


def all_configs() -> list[Config]:
    return [Config(*v) for v in itertools.product(*SPACE.values())]


# The client's current setup: mid tier model, no special prompting, everything in context.
BASELINE = Config("sonnet", "plain", "full", 512, "high")

# ---------------------------------------------------------------- reference material


def _source_info() -> dict[str, object]:
    with (DATA_DIR / "source_info.jsonl").open() as f:
        return {r["source_id"]: r["source_info"] for r in map(json.loads, f)}


_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[a-z0-9]+")
STOP = set("the a an of to in and or for on at by with is are was were be as that this it from "
           "what which who how when where why do does did".split())


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)  # rough, good enough for a budget


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in STOP}


def trim(text: str, budget: str, query: str | None) -> str:
    """Keep the most relevant sentences up to `budget` % of the material's tokens, in original order.
    Relevance: overlap with the question (QA) or with the document's frequent words (Summary)."""
    if budget == "full":
        return text
    limit = _tokens(text) * int(budget) / 100
    sents = [s.strip() for s in _SENT.split(text) if s.strip()]
    if query:
        target = _words(query)
        score = lambda s: len(_words(s) & target)
    else:
        freq = Counter(w for s in sents for w in _words(s))
        score = lambda s: sum(freq[w] for w in _words(s)) / (1 + len(_words(s)) ** 0.5)
    ranked = sorted(range(len(sents)), key=lambda i: (-score(sents[i]), i))
    keep, used = set(), 0
    for i in ranked:
        t = _tokens(sents[i])
        if used + t > limit:
            continue
        keep.add(i)
        used += t
    return " ".join(sents[i] for i in sorted(keep))


# ---------------------------------------------------------------- prompts

GROUNDED_SYSTEM = ("Use only the information in the provided material. Do not add facts, dates, "
                   "numbers, names or causes that the material does not state. If the material does "
                   "not contain the answer, say so.")
EXTRACT_SYSTEM = (GROUNDED_SYSTEM + " First copy the sentences from the material you will rely on "
                  "inside <quotes></quotes>. Then write your final response inside <answer></answer>, "
                  "using only what the quotes support.")
UNANSWERABLE = "Unable to answer based on given passages."


def build(src: Source, info: object, cfg: Config) -> tuple[str, str]:
    """Return (system, user) for this source under this configuration."""
    if src.task_type == "QA":
        q, passages = info["question"], info["passages"]
        ctx = trim(passages, cfg.context, q)
        user = (f"Briefly answer the following question:\n{q}\n"
                f"Bear in mind that your response should be strictly based on the following passages:\n"
                f"{ctx}\n\nIn case the passages do not contain the necessary information to answer "
                f"the question, please reply with: \"{UNANSWERABLE}\"")
    else:
        m = re.search(r"within (\d+) words", src.prompt)
        words = m.group(1) if m else "100"
        user = f"Summarize the following news within {words} words:\n{trim(info, cfg.context, None)}"
    system = {"plain": "", "grounded": GROUNDED_SYSTEM, "extract": EXTRACT_SYSTEM}[cfg.prompt]
    return system, user


def final_answer(text: str, cfg: Config) -> str | None:
    """The answer the user would see, or None if there is no complete answer."""
    if cfg.prompt == "extract":
        m = re.search(r"<answer>(.*?)</answer>", text, re.S)
        return m.group(1).strip() if m and m.group(1).strip() else None
    return text.strip() or None


REFUSAL = re.compile(r"unable to answer|(do|does) not (contain|provide|mention|include)|"
                     r"no (information|mention)|cannot (answer|determine)", re.I)

# ---------------------------------------------------------------- evaluation


async def evaluate(cfg: Config, llm=None, sources: list[Source] | None = None) -> dict:
    from .llm import LLM

    llm = llm or LLM()
    sources = sources or pipeline_eval_set()
    info = _source_info()
    mid = MODELS[cfg.model]

    async def one(src: Source):
        system, user = build(src, info[src.source_id], cfg)
        gen = await llm.run(model=mid, system=system, user=user, max_tokens=cfg.max_tokens,
                            effort=cfg.effort, tag="gen-v1")
        truncated = gen.output_tokens >= cfg.max_tokens  # hit the cap, so cut off mid answer
        answer = None if truncated else final_answer(gen.output, cfg)
        if answer is None:  # unusable: no need to pay for a judgement
            return src, gen, "", None, truncated
        verdict = await judge(llm, JUDGE_MODEL, src.prompt, answer)  # judged against the FULL source
        return src, gen, answer, verdict, truncated

    t0 = time.perf_counter()
    results = await asyncio.gather(*(one(s) for s in sources))
    wall = time.perf_counter() - t0

    n = len(results)
    usable = [v is not None for _, _, _, v, _ in results]
    halluc = [bool(v.output["hallucinated"]) for _, _, _, v, _ in results if v is not None]
    good = [v is not None and not v.output["hallucinated"] for _, _, _, v, _ in results]
    refused = [bool(REFUSAL.search(a)) for _, _, a, v, _ in results if v is not None]
    lat = sorted(g.latency_s for _, g, *_ in results)
    gen_cost = sum(g.cost for _, g, *_ in results)
    judge_cost = sum(v.cost for _, _, _, v, _ in results if v is not None)

    def by_task(flags):
        return {t: sum(f for (s, *_), f in zip(results, flags) if s.task_type == t)
                / max(1, sum(s.task_type == t for s, *_ in results)) for t in ("QA", "Summary")}

    row = {
        "config": asdict(cfg),
        "config_id": cfg.id,
        "n": n,
        # Objective: share of requests that get a complete answer with no hallucination (ADR 0005)
        "quality": sum(good) / n,
        "quality_by_task": by_task(good),
        "complete_rate": sum(usable) / n,
        "truncated_rate": sum(t for *_, t in results) / n,
        "faithfulness": 1 - sum(halluc) / max(1, len(halluc)),  # among complete answers only
        "cost_per_1k": 1000 * gen_cost / n,
        "latency_p50": statistics.median(lat),
        "latency_p95": lat[int(0.95 * (len(lat) - 1))],
        "refusal_rate": sum(refused) / n,  # share of all requests answered with a refusal
        "avg_input_tokens": statistics.mean(g.input_tokens for _, g, *_ in results),
        "avg_output_tokens": statistics.mean(g.output_tokens for _, g, *_ in results),
        "eval_cost_usd": gen_cost + judge_cost,
        "judge": f"{JUDGE_MODEL}/{JUDGE_VERSION}",
        "wall_s": wall,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return row


def done_configs() -> dict[str, dict]:
    if not EVALS.exists():
        return {}
    return {r["config_id"]: r for r in map(json.loads, EVALS.read_text().splitlines())}


async def evaluate_and_log(cfg: Config, llm=None, force: bool = False) -> dict:
    """Evaluate once and append to runs/evaluations.jsonl. Re-runs return the saved row."""
    prev = done_configs().get(cfg.id)
    if prev and not force:
        return prev
    row = await evaluate(cfg, llm)
    EVALS.parent.mkdir(exist_ok=True)
    with EVALS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def show(row: dict) -> None:
    print(f"{row['config_id']:32s} quality {row['quality']:.2f} (complete {row['complete_rate']:.0%}, "
          f"faithful {row['faithfulness']:.2f})  "
          f"${row['cost_per_1k']:.2f}/1k  p95 {row['latency_p95']:.1f}s  "
          f"refusals {row['refusal_rate']:.0%}  (eval cost ${row['eval_cost_usd']:.2f})")


# ---------------------------------------------------------------- CLI


def estimate(cfg: Config, quiet: bool = False) -> float:
    sources, info = pipeline_eval_set(), _source_info()
    tin = sum(_tokens("".join(build(s, info[s.source_id], cfg))) for s in sources)
    tout = len(sources) * cfg.max_tokens * 0.6
    gen = cost_usd(MODELS[cfg.model], int(tin), int(tout))
    jud = cost_usd(JUDGE_MODEL, int(sum(_tokens(s.prompt) for s in sources) + tout + 300 * len(sources)),
                   150 * len(sources))
    if not quiet:
        print(f"{cfg.id}: ~{tin/1e3:.0f}k input tokens -> generation ~${gen:.2f} + judging ~${jud:.2f}")
    return gen + jud


async def check() -> None:
    """Confirm every model accepts every effort level before spending on full runs."""
    from .llm import LLM

    llm = LLM()
    for m, e in itertools.product(SPACE["model"], SPACE["effort"]):
        try:
            c = await llm.run(model=MODELS[m], system="", user="Reply with OK.", max_tokens=16,
                              effort=e, tag="check")
            print(f"{m:7s} effort={e:5s} ok   ({c.output!r})")
        except Exception as ex:  # noqa: BLE001
            print(f"{m:7s} effort={e:5s} FAIL {type(ex).__name__}: {str(ex)[:150]}")


def parse_config(pairs: list[str]) -> Config:
    d = asdict(BASELINE) | dict(p.split("=", 1) for p in pairs)
    d["max_tokens"] = int(d["max_tokens"])
    for k, v in d.items():
        assert v in SPACE[k], f"{k}={v} not in {SPACE[k]}"
    return Config(**d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--config", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.check:
        asyncio.run(check())
    else:
        cfg = parse_config(a.config) if a.config is not None else BASELINE
        if a.dry_run:
            estimate(cfg)
        else:
            show(asyncio.run(evaluate_and_log(cfg, force=a.force)))
