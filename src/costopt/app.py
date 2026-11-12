"""Web app: the Pareto front page plus a recommendation API that reuses recommend.py, so the
sliders and the command line always agree.

Usage (local):
    uv run uvicorn src.costopt.app:app --reload        # http://127.0.0.1:8000
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse

from .recommend import facts, recommend

WEB = Path(__file__).resolve().parents[2] / "web"
app = FastAPI(title="LLM pipeline cost optimiser")


@lru_cache
def results() -> dict:
    return json.loads((WEB / "results.json").read_text())


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/plotly.min.js")
def plotly_js() -> FileResponse:
    # Bundled (plotly.js-dist-min 2.35.2) so the page does not depend on a CDN.
    return FileResponse(WEB / "plotly.min.js", media_type="application/javascript",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/results.json")
def results_json() -> JSONResponse:
    return JSONResponse(results())


@app.get("/api/recommend")
def api_recommend(min_quality: float = Query(0.8, ge=0, le=1),
                  w_quality: float = Query(0.4, ge=0), w_cost: float = Query(0.4, ge=0),
                  w_latency: float = Query(0.2, ge=0), max_p95: float | None = None) -> dict:
    data = results()
    rows = data["rows"]
    base = next(r for r in rows if r["config_id"] == data["baseline_id"])
    weights = {"quality": w_quality, "cost": w_cost, "latency": w_latency}
    if sum(weights.values()) == 0:
        weights = {"quality": 1, "cost": 1, "latency": 1}
    rec = recommend(min_quality, weights, max_p95=max_p95, rows=rows, base=base)
    if rec["status"] != "ok":
        return {"status": rec["status"]}
    return {"status": "ok", "choice_id": rec["choice"]["config_id"],
            "runner_up_id": rec["runner_up"]["config_id"] if rec["runner_up"] else None,
            "facts": facts(rec, min_quality, weights)}


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}
