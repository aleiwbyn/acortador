"""Orquestador principal: Drive -> proxy (si aplica) -> Gemini -> ffmpeg -> entregables."""

from __future__ import annotations

import contextlib
import math
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .ai_analyzer import DEFAULT_MODEL, analyze_video_with_gemini
from .drive_utils import (
    COLAB_DRIVE_ROOT,
    default_output_path,
    extract_drive_file_id,
    is_drive_mounted,
    list_drive_videos,
    resolve_directory_source,
    resolve_video_source,
)
from .local_analyzer import build_local_summary_plan, transcribe_video_locally
from .models import (
    MAX_ALLOWED_DURATION_SEC,
    BatchItemFailure,
    BatchSummaryResult,
    SummaryResult,
    format_timestamp,
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


def format_batch_markdown_report(batch: BatchSummaryResult) -> str:
    """Genera un reporte Markdown consolidado para un lote de videos resumidos."""
    lines = [
        f"# Resumen por lote ({batch.total_videos} video(s))",
        "",
    ]
    if batch.source_dir is not None:
        lines.append(f"- **Directorio origen:** `{batch.source_dir}`")
    lines.append(
        f"- **Procesados:** {batch.succeeded_count} | "
        f"**Omitidos (ya existían):** {batch.skipped_count} | "
        f"**Con error:** {batch.failed_count}"
    )
    lines.extend(
        [
            "",
            "| Video | Estado | Original | Resumen | Salida |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )

    for res in batch.results:
        lines.append(
            f"| {res.source_path.name} | Completado | "
            f"{format_timestamp(res.source_duration)} | "
            f"{format_timestamp(res.summary_duration)} | "
            f"`{res.output_video_path.name}` |"
        )
    for skipped_path in batch.skipped:
        lines.append(f"| {skipped_path.name} | Omitido (ya existía) | - | - | - |")
    for failure in batch.failures:
        clean_err = failure.error.replace("|", " ").replace("\n", " ").strip()
        lines.append(f"| {failure.source_path.name} | Error | - | - | {clean_err} |")

    for res in batch.results:
        lines.extend(
            [
                "",
                f"## {res.video_title} (`{res.source_path.name}`)",
                "",
                res.executive_summary,
                "",
                "**Ideas principales:**",
            ]
        )
        for idea in res.main_ideas:
            lines.append(f"- {idea}")

    if batch.failures:
        lines.extend(["", "## Errores detectados", ""])
        for failure in batch.failures:
            lines.append(f"- **{failure.source_path.name}:** {failure.error}")

    return "\n".join(lines).strip() + "\n"


def _compute_item_output_dir(
    source_item: str | Path,
    output_dir: str | Path | None,
    base_source_dir: Path | None,
) -> Path | None:
    if output_dir is None or not str(output_dir).strip():
        return None

    raw_dir = Path(str(output_dir).strip()).expanduser()
    if not raw_dir.is_absolute() and not raw_dir.exists() and is_drive_mounted():
        target_base = COLAB_DRIVE_ROOT / str(output_dir).strip().lstrip("/\\")
    else:
        target_base = raw_dir

    if base_source_dir is not None and extract_drive_file_id(str(source_item)) is None:
        with contextlib.suppress(ValueError, OSError):
            rel_parent = (
                Path(source_item)
                .expanduser()
                .resolve()
                .parent.relative_to(base_source_dir.resolve())
            )
            if str(rel_parent) != ".":
                return target_base / rel_parent

    return target_base


def summarize_videos(
    sources: Sequence[str | Path],
    output_dir: str | Path | None = None,
    source_dir: str | Path | None = None,
    max_duration_minutes: float = 10.0,
    engine: str = "local",
    model: str = DEFAULT_MODEL,
    language: str = "es",
    focus_instructions: str = "",
    api_key: str | None = None,
    client: Any | None = None,
    skip_existing: bool = False,
    continue_on_error: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> BatchSummaryResult:
    """Resume una lista de videos en secuencia respetando <= 10 min por video."""
    if not math.isfinite(max_duration_minutes) or not (0.5 <= max_duration_minutes <= 10.0):
        raise ValueError(
            f"max_duration_minutes debe estar entre 0.5 y 10.0 minutos (recibido: {max_duration_minutes})."
        )
    ensure_ffmpeg_available()

    cleaned_sources: list[str] = []
    seen_sources: set[str] = set()
    for raw_item in sources:
        item_text = str(raw_item).strip()
        if item_text and item_text not in seen_sources:
            seen_sources.add(item_text)
            cleaned_sources.append(item_text)

    if not cleaned_sources:
        raise ValueError("Debes proporcionar al menos un video para procesar.")

    base_source_dir: Path | None = None
    if source_dir is not None and str(source_dir).strip():
        with contextlib.suppress(ValueError, FileNotFoundError):
            base_source_dir = resolve_directory_source(source_dir)

    results: list[SummaryResult] = []
    skipped: list[Path] = []
    failures: list[BatchItemFailure] = []
    total = len(cleaned_sources)

    for idx, item_str in enumerate(cleaned_sources, start=1):
        is_url = extract_drive_file_id(item_str) is not None
        label = item_str if is_url else Path(item_str).name
        item_out_dir = _compute_item_output_dir(item_str, output_dir, base_source_dir)

        if skip_existing and not is_url:
            expected_out = default_output_path(
                Path(item_str).expanduser(),
                output_dir=item_out_dir,
            )
            if expected_out.is_file() and expected_out.stat().st_size > 0:
                skipped.append(Path(item_str))
                _notify(
                    progress_callback,
                    f"[{idx}/{total}] {label} — Omitido (ya existe {expected_out.name})",
                )
                continue

        def _item_cb(msg: str, i: int = idx, name: str = label) -> None:
            _notify(progress_callback, f"[{i}/{total}] {name} — {msg}")

        try:
            res = summarize_video(
                source=item_str,
                output_dir=item_out_dir,
                max_duration_minutes=max_duration_minutes,
                engine=engine,
                model=model,
                language=language,
                focus_instructions=focus_instructions,
                api_key=api_key,
                client=client,
                progress_callback=_item_cb,
            )
            results.append(res)
        except Exception as exc:
            if not continue_on_error:
                raise
            failures.append(BatchItemFailure(source_path=Path(item_str), error=str(exc)))
            _notify(progress_callback, f"[{idx}/{total}] {label} — Error en {label}: {exc}")

    _notify(
        progress_callback,
        f"Lote finalizado: {len(results)} procesado(s), {len(skipped)} omitido(s), {len(failures)} error(es).",
    )
    return BatchSummaryResult(
        source_dir=base_source_dir,
        total_videos=total,
        results=results,
        skipped=skipped,
        failures=failures,
    )


def summarize_directory(
    directory: str | Path,
    output_dir: str | Path | None = None,
    recursive: bool = True,
    max_videos: int = 200,
    exclude_summaries: bool = True,
    max_duration_minutes: float = 10.0,
    engine: str = "local",
    model: str = DEFAULT_MODEL,
    language: str = "es",
    focus_instructions: str = "",
    api_key: str | None = None,
    client: Any | None = None,
    skip_existing: bool = True,
    continue_on_error: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> BatchSummaryResult:
    """Descubre todos los videos de un directorio y los resume en lote."""
    resolved_dir = resolve_directory_source(directory)
    videos = list_drive_videos(
        resolved_dir,
        recursive=recursive,
        max_results=max_videos,
        exclude_summaries=exclude_summaries,
    )
    if not videos:
        raise FileNotFoundError(
            f"No se encontraron videos soportados en el directorio: {resolved_dir}"
        )

    _notify(
        progress_callback,
        f"Se encontraron {len(videos)} video(s) en {resolved_dir}. Iniciando lote...",
    )
    return summarize_videos(
        sources=videos,
        output_dir=output_dir,
        source_dir=resolved_dir,
        max_duration_minutes=max_duration_minutes,
        engine=engine,
        model=model,
        language=language,
        focus_instructions=focus_instructions,
        api_key=api_key,
        client=client,
        skip_existing=skip_existing,
        continue_on_error=continue_on_error,
        progress_callback=progress_callback,
    )
