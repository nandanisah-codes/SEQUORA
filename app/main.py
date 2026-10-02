"""
SEQUORA — FastAPI Backend
============================
Exposes the RecommenderService as real HTTP endpoints.

Run locally with:
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/docs for an interactive test page.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.schemas import (
    DiagnosticsResponse,
    EventRequest,
    EventResponse,
    HealthResponse,
    NewItemRequest,
    NewItemResponse,
    RecommendRequest,
    RecommendResponse,
)
from app.services import RecommenderService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sequora.main")

app = FastAPI(title="SEQUORA", description="Hybrid recommendation engine API")

service: RecommenderService | None = None
_last_latency_ms: float | None = None


@app.on_event("startup")
def startup() -> None:
    global service
    logger.info("Loading RecommenderService...")
    service = RecommenderService(
        processed_dir=Path("processed_data"),
        artifacts_dir=Path("artifacts"),
    )
    logger.info("RecommenderService ready.")


def _require_service() -> RecommenderService:
    if service is None:
        raise HTTPException(status_code=503, detail="Service is still starting up.")
    return service


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    svc = _require_service()
    return HealthResponse(**svc.health())


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    global _last_latency_ms
    svc = _require_service()

    start = time.perf_counter()
    items = svc.recommend(
        user_id=req.user_id,
        session_item_ids=req.session_item_ids,
        top_k=req.top_k,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    _last_latency_ms = latency_ms

    return RecommendResponse(user_id=req.user_id, recommendations=items, latency_ms=latency_ms)


@app.post("/events", response_model=EventResponse)
def events(req: EventRequest, session_id: str = "default") -> EventResponse:
    """
    session_id would normally come from a cookie/auth token; for the demo
    it defaults to "default" — Streamlit will pass its own session id.
    """
    svc = _require_service()
    length = svc.record_event(session_id, req.item_id)
    return EventResponse(status="recorded", session_length=length)


@app.post("/items", response_model=NewItemResponse)
def add_item(req: NewItemRequest) -> NewItemResponse:
    """
    Registers a new item. NOTE: this endpoint accepts the item's metadata
    and confirms registration — actually extracting a fresh multimodal
    embedding and inserting it into the live FAISS index (POST /index/add)
    is wired in once models/multimodal.py's encoder is loaded into this
    service as well. For now this confirms the API contract works.
    """
    svc = _require_service()
    # Placeholder item_idx until this item is run through preprocessing /
    # assigned a real catalog index — demonstrates the endpoint shape.
    fake_idx = abs(hash(req.article_id)) % 1_000_000
    logger.info("Registered new item %s (placeholder idx=%d)", req.article_id, fake_idx)
    return NewItemResponse(article_id=req.article_id, item_idx=fake_idx, indexed=False)


@app.get("/diagnostics", response_model=DiagnosticsResponse)
def diagnostics() -> DiagnosticsResponse:
    svc = _require_service()
    return DiagnosticsResponse(
        faiss_index_size=svc.faiss_index.size if svc.faiss_index else 0,
        num_active_sessions=len(svc.sessions),
        model_versions={
            "sasrec": "loaded" if svc.sasrec else "unavailable",
            "two_tower": "loaded" if svc.two_tower else "unavailable",
        },
        last_recommend_latency_ms=_last_latency_ms,
    )