"""Orquestador principal: Drive -> proxy (si aplica) -> Gemini -> ffmpeg -> entregables."""

from __future__ import annotations

import math
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .ai_analyzer import DEFAULT_MODEL, analyze_video_with_gemini
from .drive_utils import (
    COLAB_DRIVE_ROOT,
    default_output_path,
    is_drive_mounted,
    resolve_video_source,
)
from .local_analyzer import build_local_summary_plan, transcribe_video_locally
from .models import (
    MAX_ALLOWED_DURATION_SEC,
    SummaryResult,
    normalize_and_enforce_budget,
)
from .video_editor import (
    create_analysis_proxy,
    ensure_ffmpeg_available,
    hard_trim_video,
    needs_analysis_proxy,
    probe_video,
    render_summary_video,
    write_markdown_summary,
    write_srt_subtitles,
)


def _notify(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _resolve_final_output_path(
    source_path: Path,
    downloaded_from_url: bool,
    output_path: str | Path | None,
    output_dir: str | Path | None,
) -> Path:
    if output_path is not None and str(output_path).strip():
        candidate = Path(str(output_path).strip()).expanduser().resolve()
        if candidate == source_path.resolve():
            raise ValueError("La ruta de salida no puede ser idéntica al video fuente.")
        return candidate

    if output_dir is not None and str(output_dir).strip():
        return default_output_path(source_path, output_dir=output_dir)

    if downloaded_from_url:
        fallback_dir = (
            COLAB_DRIVE_ROOT / "Resumenes_IA"
            if is_drive_mounted()
            else Path.cwd() / "salidas"
        )
        return default_output_path(source_path, output_dir=fallback_dir)

    return default_output_path(source_path)


def summarize_video(
    source: str,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_duration_minutes: float = 10.0,
    engine: str = "local",
    model: str = DEFAULT_MODEL,
    language: str = "es",
    focus_instructions: str = "",
    api_key: str | None = None,
    client: Any | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> SummaryResult:
    """Genera un video resumido (<= 10 minutos) con las ideas principales."""
    if not math.isfinite(max_duration_minutes) or not (0.5 <= max_duration_minutes <= 10.0):
        raise ValueError(
            f"max_duration_minutes debe estar entre 0.5 y 10.0 minutos (recibido: {max_duration_minutes})."
        )

    max_duration_sec = min(float(max_duration_minutes) * 60.0, MAX_ALLOWED_DURATION_SEC)
    ensure_ffmpeg_available()

    with tempfile.TemporaryDirectory(prefix="acortador_work_") as work_dir_str:
        work_dir = Path(work_dir_str)
        _notify(progress_callback, "Resolviendo video fuente...")
        source_path, downloaded_from_url = resolve_video_source(
            source,
            download_dir=work_dir / "drive_download",
        )
        final_output = _resolve_final_output_path(
            source_path=source_path,
            downloaded_from_url=downloaded_from_url,
            output_path=output_path,
            output_dir=output_dir,
        )

        _notify(progress_callback, f"Inspeccionando metadatos de {source_path.name}...")
        source_meta = probe_video(source_path)

        use_gemini = engine.lower().strip() == "gemini"

        if use_gemini:
            if needs_analysis_proxy(source_meta):
                _notify(progress_callback, "Creando proxy ligero para análisis rápido en Gemini...")
                analysis_path = work_dir / f"{source_path.stem}_proxy.mp4"
                create_analysis_proxy(
                    input_path=source_path,
                    output_path=analysis_path,
                    has_audio=source_meta.has_audio,
                )
                used_proxy = True
            else:
                analysis_path = source_path
                used_proxy = False

            plan = analyze_video_with_gemini(
                video_path=analysis_path,
                source_duration=source_meta.duration,
                max_duration_sec=max_duration_sec,
                model=model,
                language=language,
                focus_instructions=focus_instructions,
                api_key=api_key,
                client=client,
                progress_callback=progress_callback,
            )
            model_used_label = model
        else:
            used_proxy = False
            _notify(progress_callback, "Analizando con IA local (Whisper + TextRank, sin API keys)...")
            speech_segments = transcribe_video_locally(
                video_path=source_path,
                whisper_model_size="base",
                language=language if language != "auto" else None,
                progress_callback=progress_callback,
            )
            plan = build_local_summary_plan(
                speech_segments=speech_segments,
                source_duration=source_meta.duration,
                max_duration_sec=max_duration_sec,
                video_title=source_path.stem,
            )
            model_used_label = "whisper-local-nlp"

        segments = normalize_and_enforce_budget(
            plan.segments,
            source_duration=source_meta.duration,
            max_duration_sec=max_duration_sec,
        )

        render_summary_video(
            input_path=source_path,
            segments=segments,
            output_path=final_output,
            has_audio=source_meta.has_audio,
            progress_callback=progress_callback,
        )

        out_meta = probe_video(final_output)
        if out_meta.duration > max_duration_sec + 1.0:
            _notify(
                progress_callback,
                f"Aplicando recorte duro de seguridad a {max_duration_sec:.0f} s...",
            )
            hard_trim_video(final_output, final_output, max_duration_sec)
            out_meta = probe_video(final_output)

        srt_path = final_output.with_suffix(".srt")
        md_path = final_output.with_suffix(".md")
        write_srt_subtitles(segments, srt_path)

        result = SummaryResult(
            source_path=source_path,
            output_video_path=final_output,
            summary_md_path=md_path,
            subtitles_srt_path=srt_path,
            source_duration=source_meta.duration,
            summary_duration=out_meta.duration,
            video_title=plan.video_title.strip() or source_path.stem,
            executive_summary=plan.executive_summary.strip(),
            main_ideas=[idea.strip() for idea in plan.main_ideas if idea.strip()],
            segments=segments,
            model_used=model_used_label,
            used_proxy=used_proxy,
        )
        write_markdown_summary(result, md_path)
        _notify(progress_callback, f"¡Listo! Video resumido guardado en {final_output}")
        return result
