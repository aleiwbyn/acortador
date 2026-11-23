"""Inspección de video, proxy de análisis y recorte/concatenación con ffmpeg."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

from .models import (
    NormalizedSegment,
    SummaryResult,
    VideoMetadata,
    format_srt_timestamp,
    format_timestamp,
)

PROXY_THRESHOLD_BYTES = 1800 * 1024 * 1024  # 1.8 GB (Gemini Files API acepta nativamente hasta 2 GB)
GEMINI_NATIVE_EXTENSIONS = {
    ".mp4",
    ".mpeg",
    ".mpg",
    ".mov",
    ".avi",
    ".flv",
    ".webm",
    ".wmv",
    ".3gp",
}


def _notify(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _run_subprocess(cmd: list[str], error_context: str) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise RuntimeError(f"{error_context} (código {proc.returncode}): {stderr_tail}")
    return proc


def _parse_fps(rate_str: str | None) -> float:
    if not rate_str or rate_str == "0/0":
        return 30.0
    try:
        value = float(Fraction(rate_str))
        return value if value > 0 else 30.0
    except (ValueError, ZeroDivisionError):
        return 30.0


def ensure_ffmpeg_available() -> None:
    """Verifica que ffmpeg y ffprobe estén disponibles en PATH."""
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            f"Faltan herramientas requeridas en PATH: {', '.join(missing)}. "
            "Instala ffmpeg antes de ejecutar el resumen."
        )


def probe_video(video_path: Path) -> VideoMetadata:
    """Extrae duración, resolución, fps y pista de audio usando ffprobe."""
    if not video_path.is_file():
        raise FileNotFoundError(f"No existe el archivo de video: {video_path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    proc = _run_subprocess(cmd, f"No se pudo inspeccionar el video {video_path.name}")
    payload = json.loads(proc.stdout or "{}")

    streams = payload.get("streams") or []
    fmt = payload.get("format") or {}

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ValueError(f"El archivo no contiene ninguna pista de video válida: {video_path}")

    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    raw_duration = fmt.get("duration") or video_stream.get("duration") or 0.0
    duration = float(raw_duration)
    if duration <= 0:
        raise ValueError(f"No se pudo determinar una duración positiva para {video_path}")

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    fps = _parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))
    size_bytes = video_path.stat().st_size

    return VideoMetadata(
        path=video_path,
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        has_audio=has_audio,
        size_bytes=size_bytes,
    )


def needs_analysis_proxy(meta: VideoMetadata) -> bool:
    """Determina si conviene crear un proxy ligero antes de subir a Gemini."""
    return (
        meta.size_bytes > PROXY_THRESHOLD_BYTES
        or meta.path.suffix.lower() not in GEMINI_NATIVE_EXTENSIONS
    )


def create_analysis_proxy(input_path: Path, output_path: Path, has_audio: bool) -> Path:
    """Crea un proxy MP4 ultraligero a 2 fps preservando la línea de tiempo original."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-threads",
        "0",
        "-i",
        str(input_path),
        "-vf",
        "scale='min(640,iw)':-2,fps=2,pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "30",
        "-pix_fmt",
        "yuv420p",
    ]
    if has_audio:
        cmd.extend(["-c:a", "aac", "-ac", "1", "-b:a", "64k"])
    else:
        cmd.append("-an")
    cmd.extend(["-movflags", "+faststart", str(output_path)])

    _run_subprocess(cmd, f"Error generando proxy de análisis para {input_path.name}")
    return output_path


def build_segment_ffmpeg_cmd(
    input_path: Path,
    segment: NormalizedSegment,
    clip_path: Path,
    has_audio: bool,
    fade_sec: float = 0.15,
) -> list[str]:
    """Construye el comando ffmpeg para recortar un segmento con precisión de cuadros."""
    duration = segment.duration
    cmd = [
        "ffmpeg",
        "-y",
        "-threads",
        "0",
        "-ss",
        f"{segment.start:.3f}",
        "-i",
        str(input_path),
        "-t",
        f"{duration:.3f}",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
    ]

    if has_audio:
        effective_fade = min(fade_sec, duration * 0.2)
        if effective_fade > 0.02 and duration > 2.0 * effective_fade:
            fade_out_start = max(0.0, duration - effective_fade)
            audio_filter = (
                f"afade=t=in:st=0:d={effective_fade:.3f},"
                f"afade=t=out:st={fade_out_start:.3f}:d={effective_fade:.3f}"
            )
            cmd.extend(["-af", audio_filter])
        cmd.extend(["-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2"])
    else:
        cmd.append("-an")

    cmd.extend(["-avoid_negative_ts", "make_zero", str(clip_path)])
    return cmd


def render_summary_video(
    input_path: Path,
    segments: list[NormalizedSegment],
    output_path: Path,
    has_audio: bool,
    fade_sec: float = 0.15,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Recorta los segmentos seleccionados y los une en el video resumido final."""
    if not segments:
        raise ValueError("Se requiere al menos un segmento para renderizar el video resumido.")
    if output_path.resolve() == input_path.resolve():
        raise ValueError("La ruta de salida no puede sobrescribir el video fuente.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="acortador_render_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        clip_paths: list[Path] = []

        for idx, segment in enumerate(segments):
            _notify(
                progress_callback,
                f"Renderizando clip {idx + 1}/{len(segments)}: {segment.title}",
            )
            clip_path = tmp_dir / f"seg_{idx:03d}.mp4"
            cmd = build_segment_ffmpeg_cmd(
                input_path=input_path,
                segment=segment,
                clip_path=clip_path,
                has_audio=has_audio,
                fade_sec=fade_sec,
            )
            _run_subprocess(cmd, f"Error recortando segmento {idx + 1} ({segment.title})")
            clip_paths.append(clip_path)

        _notify(progress_callback, "Ensamblando video resumido final...")
        if len(clip_paths) == 1:
            concat_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(clip_paths[0]),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        else:
            list_file = tmp_dir / "concat_list.txt"
            lines = [
                "file '{}'".format(clip.resolve().as_posix().replace("'", "'\\''"))
                for clip in clip_paths
            ]
            list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
            concat_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output_path),
            ]

        _run_subprocess(concat_cmd, "Error concatenando segmentos del resumen")

    return output_path


def hard_trim_video(input_path: Path, output_path: Path, max_duration_sec: float) -> Path:
    """Recorta exactamente a max_duration_sec como salvaguarda final de duración."""
    tmp_output = output_path.with_suffix(".trimmed.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-t",
        f"{max_duration_sec:.3f}",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(tmp_output),
    ]
    _run_subprocess(cmd, "Error aplicando recorte máximo de duración")
    tmp_output.replace(output_path)
    return output_path


def write_srt_subtitles(segments: list[NormalizedSegment], output_srt_path: Path) -> Path:
    """Genera un archivo .srt alineado a la línea de tiempo del video resumido."""
    output_srt_path.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []
    cursor = 0.0

    for idx, seg in enumerate(segments, start=1):
        start_ts = format_srt_timestamp(cursor)
        end_ts = format_srt_timestamp(cursor + seg.duration)
        blocks.append(f"{idx}\n{start_ts} --> {end_ts}\n{seg.title}\n{seg.key_idea}\n")
        cursor += seg.duration

    output_srt_path.write_text("\n".join(blocks), encoding="utf-8")
    return output_srt_path


def write_markdown_summary(result: SummaryResult, output_md_path: Path) -> Path:
    """Guarda el reporte Markdown con ideas principales y tabla de timestamps."""
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    reduction_pct = (
        max(0.0, (1.0 - (result.summary_duration / result.source_duration)) * 100.0)
        if result.source_duration > 0
        else 0.0
    )

    lines: list[str] = [
        f"# {result.video_title}",
        "",
        f"- **Video fuente:** `{result.source_path}`",
        f"- **Video resumido:** `{result.output_video_path}`",
        f"- **Duración original:** {format_timestamp(result.source_duration, result.source_duration >= 3600)} ({result.source_duration:.1f} s)",
        f"- **Duración del resumen:** {format_timestamp(result.summary_duration, result.summary_duration >= 3600)} ({result.summary_duration:.1f} s)",
        f"- **Reducción de tiempo:** {reduction_pct:.1f}%",
        f"- **Modelo de IA:** `{result.model_used}`",
        "",
        "## Resumen ejecutivo",
        "",
        result.executive_summary.strip(),
        "",
        "## Ideas principales",
        "",
    ]

    for idx, idea in enumerate(result.main_ideas, start=1):
        lines.append(f"{idx}. {idea.strip()}")

    lines.extend(
        [
            "",
            "## Segmentos incluidos",
            "",
            "| # | Sección | Original | En resumen | Importancia | Idea clave |",
            "|---|---|---|---|---|---|",
        ]
    )

    cursor = 0.0
    use_hours = result.source_duration >= 3600
    for idx, seg in enumerate(result.segments, start=1):
        orig_range = f"{format_timestamp(seg.start, use_hours)}–{format_timestamp(seg.end, use_hours)}"
        sum_range = f"{format_timestamp(cursor)}–{format_timestamp(cursor + seg.duration)}"
        safe_title = seg.title.replace("|", "\\|")
        safe_idea = seg.key_idea.replace("|", "\\|")
        lines.append(
            f"| {idx} | {safe_title} | {orig_range} | {sum_range} | {seg.importance}/5 | {safe_idea} |"
        )
        cursor += seg.duration

    output_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_md_path
