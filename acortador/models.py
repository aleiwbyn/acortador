"""Modelos de datos, parseo de timestamps y presupuesto estricto de duración."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

MAX_ALLOWED_DURATION_SEC = 600.0  # Límite estricto e inviolable de 10 minutos.

_TIMESTAMP_RE = re.compile(
    r"^\s*(?:(?P<hours>\d+):)?(?P<minutes>\d{1,2}):(?P<seconds>\d{1,2}(?:\.\d+)?)\s*$"
)
_NUMERIC_RE = re.compile(r"^\s*\d+(?:\.\d+)?\s*$")


class RawSegment(BaseModel):
    """Segmento propuesto por Gemini en formato estructurado plano."""

    start_time: str
    end_time: str
    title: str
    key_idea: str
    importance: int


class VideoSummaryPlan(BaseModel):
    """Plan completo de resumen devuelto por Gemini."""

    video_title: str
    executive_summary: str
    main_ideas: list[str]
    segments: list[RawSegment]


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Metadatos técnicos de un archivo de video obtenidos con ffprobe."""

    path: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    size_bytes: int


@dataclass(frozen=True, slots=True)
class NormalizedSegment:
    """Segmento validado, fusionado y ajustado al presupuesto de duración."""

    start: float
    end: float
    title: str
    key_idea: str
    importance: int

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True, slots=True)
class SummaryResult:
    """Resultado completo de la generación del video resumido."""

    source_path: Path
    output_video_path: Path
    summary_md_path: Path
    subtitles_srt_path: Path
    source_duration: float
    summary_duration: float
    video_title: str
    executive_summary: str
    main_ideas: list[str]
    segments: list[NormalizedSegment]
    model_used: str
    used_proxy: bool


@dataclass(frozen=True, slots=True)
class BatchItemFailure:
    """Error registrado al procesar un video individual dentro de un lote."""

    source_path: Path
    error: str


@dataclass(frozen=True, slots=True)
class BatchSummaryResult:
    """Resultado consolidado del procesamiento por lote de múltiples videos."""

    source_dir: Path | None
    total_videos: int
    results: list[SummaryResult]
    skipped: list[Path]
    failures: list[BatchItemFailure]

    @property
    def succeeded_count(self) -> int:
        return len(self.results)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    @property
    def failed_count(self) -> int:
        return len(self.failures)


def parse_timestamp(value: str | float) -> float:
    """Convierte 'HH:MM:SS(.ms)', 'MM:SS(.ms)' o segundos numéricos a segundos."""
    if isinstance(value, bool):
        raise TypeError(f"Timestamp inválido: {value!r}")

    if isinstance(value, (int, float)):
        seconds = float(value)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"Timestamp numérico inválido o negativo: {value!r}")
        return seconds

    text = str(value).strip()
    if not text:
        raise ValueError("Timestamp vacío")

    if _NUMERIC_RE.match(text):
        seconds = float(text)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"Timestamp numérico inválido: {value!r}")
        return seconds

    match = _TIMESTAMP_RE.match(text)
    if not match:
        raise ValueError(f"Formato de timestamp no reconocido: {value!r}")

    hours_raw = match.group("hours")
    minutes = int(match.group("minutes"))
    seconds_part = float(match.group("seconds"))

    if minutes >= 60 or seconds_part >= 60.0:
        raise ValueError(f"Minutos o segundos fuera de rango en timestamp: {value!r}")

    hours = int(hours_raw) if hours_raw is not None else 0
    return hours * 3600.0 + minutes * 60.0 + seconds_part


def format_timestamp(seconds: float, include_hours: bool = False) -> str:
    """Formatea segundos como MM:SS o HH:MM:SS."""
    total = max(0, round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if include_hours or hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_srt_timestamp(seconds: float) -> str:
    """Formatea segundos al estándar SubRip (HH:MM:SS,mmm)."""
    millis_total = max(0, round(seconds * 1000.0))
    hours, rem = divmod(millis_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def compute_target_duration(
    source_duration: float,
    max_duration_sec: float = MAX_ALLOWED_DURATION_SEC,
) -> float:
    """Calcula una duración objetivo razonable sin superar max_duration_sec."""
    if not math.isfinite(source_duration) or source_duration <= 0:
        raise ValueError(f"source_duration debe ser > 0 (recibido: {source_duration})")
    if not math.isfinite(max_duration_sec) or not (1.0 <= max_duration_sec <= MAX_ALLOWED_DURATION_SEC):
        raise ValueError(
            f"max_duration_sec debe estar entre 1 y {MAX_ALLOWED_DURATION_SEC} s (recibido: {max_duration_sec})"
        )

    if source_duration > max_duration_sec:
        target = min(
            max_duration_sec * 0.90,
            max(source_duration * 0.35, max_duration_sec * 0.45),
        )
    else:
        floor_sec = min(source_duration * 0.85, 10.0)
        target = min(source_duration * 0.85, max(floor_sec, source_duration * 0.45))

    return round(max(min(source_duration, 1.0), min(target, max_duration_sec, source_duration)), 2)


def _merge_text(first: str, second: str) -> str:
    a = first.strip()
    b = second.strip()
    if not b or b.lower() in a.lower():
        return a
    if not a or a.lower() in b.lower():
        return b
    return f"{a} | {b}"


def _merge_overlapping_segments(
    segments: list[NormalizedSegment],
    merge_gap_sec: float,
) -> list[NormalizedSegment]:
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: (s.start, s.end))
    merged: list[NormalizedSegment] = [ordered[0]]

    for seg in ordered[1:]:
        prev = merged[-1]
        if seg.start <= prev.end + merge_gap_sec:
            merged[-1] = NormalizedSegment(
                start=prev.start,
                end=max(prev.end, seg.end),
                title=_merge_text(prev.title, seg.title),
                key_idea=_merge_text(prev.key_idea, seg.key_idea),
                importance=max(prev.importance, seg.importance),
            )
        else:
            merged.append(seg)

    return merged


def normalize_and_enforce_budget(
    raw_segments: list[RawSegment],
    source_duration: float,
    max_duration_sec: float = MAX_ALLOWED_DURATION_SEC,
    min_segment_sec: float = 2.0,
    merge_gap_sec: float = 1.5,
) -> list[NormalizedSegment]:
    """Valida timestamps, fusiona solapamientos y garantiza duración total <= 600 s."""
    if not math.isfinite(source_duration) or source_duration <= 0:
        raise ValueError(f"source_duration debe ser > 0 (recibido: {source_duration})")
    if not math.isfinite(max_duration_sec) or max_duration_sec <= 0:
        raise ValueError(f"max_duration_sec debe ser > 0 (recibido: {max_duration_sec})")

    effective_max = min(max_duration_sec, MAX_ALLOWED_DURATION_SEC, source_duration)
    effective_min = min(min_segment_sec, max(0.5, source_duration * 0.1))

    cleaned: list[NormalizedSegment] = []
    for raw in raw_segments:
        try:
            start = parse_timestamp(raw.start_time)
            end = parse_timestamp(raw.end_time)
        except ValueError:
            continue

        start = max(0.0, min(start, source_duration))
        end = max(0.0, min(end, source_duration))
        if end - start < effective_min:
            continue

        importance = max(1, min(5, int(raw.importance)))
        title = raw.title.strip() or "Idea clave"
        key_idea = raw.key_idea.strip() or title
        cleaned.append(
            NormalizedSegment(
                start=round(start, 3),
                end=round(end, 3),
                title=title,
                key_idea=key_idea,
                importance=importance,
            )
        )

    merged = _merge_overlapping_segments(cleaned, merge_gap_sec=merge_gap_sec)
    if not merged:
        raise ValueError("No se encontraron segmentos válidos dentro de la duración del video.")

    total_duration = sum(seg.duration for seg in merged)
    if total_duration > effective_max:
        prioritized = sorted(
            enumerate(merged),
            key=lambda item: (-item[1].importance, item[0]),
        )
        selected: list[NormalizedSegment] = []
        remaining = effective_max
        min_trim_sec = max(effective_min, 8.0)

        for _, seg in prioritized:
            if seg.duration <= remaining + 1e-6:
                selected.append(seg)
                remaining = max(0.0, remaining - seg.duration)
            elif (not selected and remaining >= effective_min) or remaining >= min_trim_sec:
                clipped_end = round(seg.start + remaining, 3)
                if clipped_end - seg.start >= effective_min:
                    selected.append(
                        NormalizedSegment(
                            start=seg.start,
                            end=clipped_end,
                            title=seg.title,
                            key_idea=seg.key_idea,
                            importance=seg.importance,
                        )
                    )
                    remaining = 0.0
                    break

        merged = sorted(selected, key=lambda s: s.start)

    # Recorte de seguridad por precisión flotante en el último segmento.
    overflow = sum(seg.duration for seg in merged) - effective_max
    if overflow > 1e-6 and merged:
        last = merged[-1]
        trimmed_end = round(max(last.start, last.end - overflow), 3)
        if trimmed_end - last.start >= effective_min:
            merged[-1] = NormalizedSegment(
                start=last.start,
                end=trimmed_end,
                title=last.title,
                key_idea=last.key_idea,
                importance=last.importance,
            )
        else:
            merged.pop()

    if not merged:
        raise ValueError("No quedó ningún segmento válido tras aplicar el presupuesto de duración.")

    return merged
