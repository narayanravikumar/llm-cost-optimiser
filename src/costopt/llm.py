"""Thin async wrapper over the Anthropic API that records tokens, cost and latency,
and caches results on disk so no request is paid for twice."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .pricing import cost_usd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "runs" / "llm_cache.jsonl"


def _load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


@dataclass
class Call:
    key: str
    model: str
    output: dict | str
    input_tokens: int
    output_tokens: int
    latency_s: float

    @property
    def cost(self) -> float:
        return cost_usd(self.model, self.input_tokens, self.output_tokens)


class LLM:
    def __init__(self, concurrency: int = 8):
        _load_env()
        from anthropic import AsyncAnthropic

        ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        headers = {"anthropic-workspace-id": ws} if ws else None
        self.client = AsyncAnthropic(default_headers=headers)
        self.sem = asyncio.Semaphore(concurrency)
        self.cache: dict[str, Call] = {}
        CACHE.parent.mkdir(exist_ok=True)
        if CACHE.exists():
            for line in CACHE.read_text().splitlines():
                d = json.loads(line)
                self.cache[d["key"]] = Call(**d)

    @staticmethod
    def make_key(**parts) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:24]

    async def run(self, *, model: str, system: str, user: str, max_tokens: int,
                  effort: str | None = None, tool: dict | None = None, tag: str = "") -> Call:
        key = self.make_key(model=model, system=system, user=user, max_tokens=max_tokens,
                            effort=effort, tool=tool, tag=tag)
        if key in self.cache:
            return self.cache[key]
        kwargs = dict(model=model, system=system, max_tokens=max_tokens,
                      messages=[{"role": "user", "content": user}])
        output_config: dict = {}
        if effort:
            output_config["effort"] = effort
        if tool:
            # Structured outputs: works on every model, unlike a forced tool_choice.
            schema = {**tool["input_schema"], "additionalProperties": False}
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if output_config:
            kwargs["output_config"] = output_config
        async with self.sem:
            t0 = time.perf_counter()
            msg = await self.client.messages.create(**kwargs)
            if tool and msg.stop_reason == "max_tokens":
                # Structured output was cut off: retry once with double the budget.
                kwargs["max_tokens"] = max_tokens * 2
                msg = await self.client.messages.create(**kwargs)
            latency = time.perf_counter() - t0
        text = "".join(b.text for b in msg.content if b.type == "text")
        if tool:
            try:
                output = json.loads(text)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON from {model} (stop_reason={msg.stop_reason}): {text[:200]}") from e
        else:
            output = text
        call = Call(key, model, output, msg.usage.input_tokens, msg.usage.output_tokens, latency)
        self.cache[key] = call
        with CACHE.open("a") as f:
            f.write(json.dumps(asdict(call)) + "\n")
        return call
