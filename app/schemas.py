"""
SEQUORA — API Schemas
========================
Pydantic models defining the shape of every request and response the
FastAPI backend handles. Keeping these separate from app/main.py makes
them easy to reuse from Streamlit and from tests.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    user_id: Optional[int] = Field(
        default=None,
        description="Existing user's internal id. Omit for a brand-new/anonymous session.",
    )
    session_item_ids: List[int] = Field(
        default_factory=list,
        description="Items clicked THIS session, most recent last. Used for cold-start "
                    "users (no user_id) or to blend with existing history.",
    )
    top_k: int = Field(default=10, ge=1, le=50)


class RecommendedItem(BaseModel):
    item_id: int
    score: float
    source: str  # "sasrec" | "two_tower" | "cold_start" | "popularity_fallback"


class RecommendResponse(BaseModel):
    user_id: Optional[int]
    recommendations: List[RecommendedItem]
    latency_ms: float


class EventRequest(BaseModel):
    user_id: Optional[int] = None
    item_id: int
    event_type: str = Field(default="click", description="click | purchase | view")


class EventResponse(BaseModel):
    status: str
    session_length: int


class NewItemRequest(BaseModel):
    article_id: str
    product_type_name: Optional[str] = None
    department_name: Optional[str] = None
    detail_desc: Optional[str] = None
    image_path: Optional[str] = Field(
        default=None, description="Path to the uploaded image, relative to the images folder."
    )


class NewItemResponse(BaseModel):
    article_id: str
    item_idx: int
    indexed: bool


class HealthResponse(BaseModel):
    status: str
    sasrec_loaded: bool
    two_tower_loaded: bool
    faiss_index_size: int
    ranker_loaded: bool


class DiagnosticsResponse(BaseModel):
    faiss_index_size: int
    num_active_sessions: int
    model_versions: dict
    last_recommend_latency_ms: Optional[float]