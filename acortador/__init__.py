"""Paquete principal para resumir videos con Gemini y Google Drive."""

from .models import (
    MAX_ALLOWED_DURATION_SEC,
    NormalizedSegment,
    SummaryResult,
    VideoSummaryPlan,
)
from .pipeline import summarize_video

__all__ = [
    "MAX_ALLOWED_DURATION_SEC",
    "NormalizedSegment",
    "SummaryResult",
    "VideoSummaryPlan",
    "summarize_video",
]
