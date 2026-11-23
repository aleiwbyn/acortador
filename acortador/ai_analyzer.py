"""Análisis multimodal de video con Gemini (SDK google-genai >= 3.2.0)."""

from __future__ import annotations

import logging
import mimetypes
import os
import re
import shutil
import tempfile
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .models import (
    MAX_ALLOWED_DURATION_SEC,
    VideoSummaryPlan,
    compute_target_duration,
    format_timestamp,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash-lite"
SUPPORTED_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-4-argon",
    "gemini-4-hydrogen",
    "gemini-4-neon",
)
AGENTIC_VIDEO_MODELS = {"gemini-4-argon", "gemini-3.5-flash-lite"}
AGENTIC_MIN_DURATION_SEC = 300.0

_SYSTEM_INSTRUCTION = (
    "Eres un editor y documentalista audiovisual experto. "
    "Analizas videos completos (audio, habla, texto en pantalla e imágenes) para extraer "
    "únicamente los momentos que contienen las ideas principales, tesis, demostraciones clave "
    "y conclusiones, eliminando rellenos, pausas largas, saludos y digresiones."
)

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

INLINE_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "video_title": {"type": "string"},
        "executive_summary": {"type": "string"},
        "main_ideas": {"type": "array", "items": {"type": "string"}},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                    "title": {"type": "string"},
                    "key_idea": {"type": "string"},
                    "importance": {"type": "integer"},
                },
                "required": ["start_time", "end_time", "title", "key_idea", "importance"],
            },
        },
    },
    "required": ["video_title", "executive_summary", "main_ideas", "segments"],
}


def to_safe_ascii_name(name: str) -> str:
    """Convierte un nombre con tildes/caracteres especiales a un nombre ASCII seguro."""
    normalized = unicodedata.normalize("NFKD", name)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    clean = re.sub(r"[^a-zA-Z0-9._-]", "_", ascii_only).strip("_")
    return clean or "video"


def _notify(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _file_state_name(file_obj: Any) -> str:
    state = getattr(file_obj, "state", None)
    if state is None:
        return ""
    name = getattr(state, "name", None)
    if isinstance(name, str):
        return name.upper()
    return str(state).upper()


def _strip_json_fences(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = _CODE_FENCE_RE.sub("", cleaned).strip()
    return cleaned


def build_summary_prompt(
    source_duration: float,
    target_duration: float,
    max_duration: float,
    language: str = "es",
    focus_instructions: str = "",
) -> str:
    """Construye el prompt estructurado para obtener los segmentos clave del video."""
    total_fmt = format_timestamp(source_duration, include_hours=source_duration >= 3600)
    target_fmt = format_timestamp(target_duration, include_hours=target_duration >= 3600)
    max_fmt = format_timestamp(max_duration, include_hours=max_duration >= 3600)

    focus_block = (
        f"\n- Enfoque adicional solicitado por el usuario: {focus_instructions.strip()}"
        if focus_instructions.strip()
        else ""
    )

    return (
        f"Analiza el video completo (duración total exacta: {total_fmt}, es decir {source_duration:.1f} s) "
        f"y diseña un plan de edición para crear un video resumido con las ideas principales.\n\n"
        f"Reglas obligatorias:\n"
        f"- Idioma de salida para títulos, resumen ejecutivo, ideas y descripciones: {language}.\n"
        f"- La suma total de duraciones de todos los segmentos seleccionados debe acercarse a "
        f"{target_fmt} ({target_duration:.0f} s) y NUNCA exceder {max_fmt} ({max_duration:.0f} s).\n"
        f"- Usa timestamps exactos en formato MM:SS o HH:MM:SS dentro del rango [00:00, {total_fmt}].\n"
        f"- Inicia y termina cada segmento respetando frases completas y cambios de escena naturales.\n"
        f"- Ordena los segmentos cronológicamente y evita solapamientos.\n"
        f"- Asigna a cada segmento un campo importance entero del 1 (contexto útil) al 5 (idea central indispensable).{focus_block}\n\n"
        f"Devuelve únicamente el JSON que cumple con el esquema solicitado."
    )


def analyze_video_with_gemini(
    video_path: Path,
    source_duration: float,
    max_duration_sec: float = MAX_ALLOWED_DURATION_SEC,
    model: str = DEFAULT_MODEL,
    language: str = "es",
    focus_instructions: str = "",
    api_key: str | None = None,
    client: Any | None = None,
    poll_interval: float = 4.0,
    timeout_sec: float = 1800.0,
    progress_callback: Callable[[str], None] | None = None,
) -> VideoSummaryPlan:
    """Sube el video a Gemini Files API y genera el plan estructurado de resumen."""
    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Modelo no soportado: {model!r}. Usa uno de: {', '.join(SUPPORTED_MODELS)}"
        )

    effective_max = min(float(max_duration_sec), MAX_ALLOWED_DURATION_SEC, source_duration)
    target_duration = compute_target_duration(source_duration, max_duration_sec=effective_max)

    if client is None:
        resolved_key = (api_key or os.environ.get("GEMINI_API_KEY") or "").strip()
        if not resolved_key:
            raise ValueError(
                "Falta GEMINI_API_KEY. Configúrala en los Secretos de Google Colab o en el entorno."
            )
        from google import genai  # type: ignore[import-not-found]

        active_client: Any = genai.Client(api_key=resolved_key)
    else:
        active_client = client

    prompt = build_summary_prompt(
        source_duration=source_duration,
        target_duration=target_duration,
        max_duration=effective_max,
        language=language,
        focus_instructions=focus_instructions,
    )

    uploaded = None
    deadline = time.monotonic() + timeout_sec

    # Crear symlink o copia temporal con nombre ASCII para evitar errores HTTP con tildes en headers
    with tempfile.TemporaryDirectory(prefix="gemini_up_") as upload_tmp_str:
        upload_tmp = Path(upload_tmp_str)
        safe_stem = to_safe_ascii_name(video_path.stem)
        safe_ext = video_path.suffix.lower()
        safe_upload_path = upload_tmp / f"{safe_stem}{safe_ext}"

        try:
            os.symlink(video_path.resolve(), safe_upload_path)
            upload_target = safe_upload_path
        except (OSError, NotImplementedError):
            try:
                str(video_path).encode("ascii")
                upload_target = video_path
            except UnicodeEncodeError:
                shutil.copyfile(video_path, safe_upload_path)
                upload_target = safe_upload_path

        try:
            _notify(progress_callback, f"Subiendo video a Gemini Files API ({video_path.name})...")
            uploaded = active_client.files.upload(file=str(upload_target))

            while _file_state_name(uploaded) != "ACTIVE":
                state = _file_state_name(uploaded)
                if state == "FAILED":
                    raise RuntimeError(f"Gemini Files API falló al procesar el archivo {video_path.name}.")
                if time.monotonic() > deadline:
                    raise TimeoutError("Tiempo de espera agotado mientras Gemini procesaba el video.")
                _notify(progress_callback, f"Procesando archivo en Gemini ({state or 'PROCESSING'})...")
                time.sleep(poll_interval)
                uploaded = active_client.files.get(name=uploaded.name)

            mime_type = (
                getattr(uploaded, "mime_type", None)
                or mimetypes.guess_type(video_path.name)[0]
                or "video/mp4"
            )

            video_part: dict[str, Any] = {
                "type": "video",
                "uri": uploaded.uri,
                "mime_type": mime_type,
            }

            _notify(progress_callback, f"Analizando contenido e ideas principales con {model}...")
            # Llamada directa estándar síncrona (evita problemas de colas de fondo y herramientas)
            try:
                interaction = active_client.interactions.create(
                    model=model,
                    input=[video_part, {"type": "text", "text": prompt}],
                    system_instruction=_SYSTEM_INSTRUCTION,
                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": INLINE_SUMMARY_SCHEMA,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.info("Intento síncrono no completado de inmediato (%s), intentando con sondeo...", exc)
                interaction = active_client.interactions.create(
                    model=model,
                    input=[video_part, {"type": "text", "text": prompt}],
                    system_instruction=_SYSTEM_INSTRUCTION,
                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": INLINE_SUMMARY_SCHEMA,
                    },
                    background=True,
                )

            while getattr(interaction, "status", None) in {"in_progress", "queued"}:
                if time.monotonic() > deadline:
                    raise TimeoutError("Tiempo de espera agotado esperando el análisis de Gemini.")
                _notify(progress_callback, f"Esperando respuesta de Gemini ({interaction.status})...")
                time.sleep(poll_interval)
                interaction = active_client.interactions.get(interaction.id)

            status = getattr(interaction, "status", None)
            if status not in {"completed", None}:
                errors_list = getattr(interaction, "errors", None) or []
                error_msgs = [getattr(e, "message", str(e)) for e in errors_list]
                error_detail = "; ".join(error_msgs) if error_msgs else getattr(interaction, "error", None)
                raise RuntimeError(
                    f"La interacción con Gemini terminó con estado {status!r}: {error_detail or 'Error interno del servidor de Gemini'}"
                )

            output_text = getattr(interaction, "output_text", None)
            if not output_text and getattr(interaction, "steps", None):
                for step in getattr(interaction, "steps", []):
                    if getattr(step, "type", None) == "model_output":
                        for part in getattr(step, "content", []) or []:
                            if getattr(part, "type", None) == "text":
                                output_text = (output_text or "") + getattr(part, "text", "")

            if not output_text or not str(output_text).strip():
                raise RuntimeError("Gemini completó la solicitud pero no devolvió texto JSON.")

            return VideoSummaryPlan.model_validate_json(_strip_json_fences(str(output_text)))
        finally:
            if uploaded is not None and getattr(uploaded, "name", None):
                try:
                    active_client.files.delete(name=uploaded.name)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("No se pudo eliminar el archivo temporal %s: %s", uploaded.name, exc)
