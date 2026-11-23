"""Interfaz web interactiva con Gradio para Google Colab y ejecución local."""

from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import Any

from .ai_analyzer import DEFAULT_MODEL, SUPPORTED_MODELS
from .drive_utils import (
    COLAB_DRIVE_ROOT,
    list_drive_subfolders,
    list_drive_videos,
    resolve_directory_source,
)
from .pipeline import format_batch_markdown_report, summarize_videos

_BATCH_PREFIX_RE = re.compile(r"^\[(\d+)/(\d+)\]\s*(.*)$")


def _build_folder_choices(root_folder: str) -> list[str]:
    folders = list_drive_subfolders(root_folder)
    choices = [str(p) for p in folders]
    clean_root = (root_folder or "").strip()
    if clean_root and clean_root not in choices:
        choices.insert(0, clean_root)
    return choices or [str(COLAB_DRIVE_ROOT)]


def _build_video_choices(
    folder: str,
    recursive: bool = True,
) -> tuple[list[tuple[str, str]], list[str], str]:
    clean_folder = (folder or "").strip()
    if not clean_folder:
        return (
            [],
            [],
            "Indica un directorio de Google Drive para cargar su lista de videos.",
        )

    videos = list_drive_videos(clean_folder, recursive=recursive, exclude_summaries=True)
    if not videos:
        return (
            [],
            [],
            f"No se encontraron videos pendientes en {clean_folder}. Puedes elegir otra carpeta o pegar rutas/enlaces abajo.",
        )

    base_path = Path(clean_folder).expanduser()
    with contextlib.suppress(ValueError, FileNotFoundError):
        base_path = resolve_directory_source(clean_folder)

    choices: list[tuple[str, str]] = []
    for video_path in videos:
        try:
            label = video_path.resolve().relative_to(base_path.resolve()).as_posix()
        except (ValueError, OSError):
            label = video_path.name
        choices.append((label, str(video_path)))

    selected = [value for _, value in choices]
    status = f"Se encontraron {len(choices)} video(s) en {clean_folder} (todos seleccionados)."
    return choices, selected, status


def _estimate_batch_progress(msg: str) -> float:
    idx = 1
    total = 1
    stage_msg = msg.strip()
    match = _BATCH_PREFIX_RE.match(stage_msg)
    if match:
        idx = max(1, int(match.group(1)))
        total = max(idx, int(match.group(2)))
        stage_msg = match.group(3)

    stage_p = 0.5
    if "Resolviendo" in stage_msg or "Inspeccionando" in stage_msg or "Extrayendo" in stage_msg:
        stage_p = 0.15
    elif "proxy" in stage_msg or "Whisper" in stage_msg:
        stage_p = 0.30
    elif "Subiendo" in stage_msg or "Transcribiendo" in stage_msg:
        stage_p = 0.45
    elif "Analizando" in stage_msg or "Gemini" in stage_msg or "TextRank" in stage_msg:
        stage_p = 0.65
    elif "Renderizando" in stage_msg or "clip" in stage_msg:
        stage_p = 0.85
    elif (
        "Listo" in stage_msg
        or "Ensamblando" in stage_msg
        or "Omitido" in stage_msg
        or "Error en" in stage_msg
    ):
        stage_p = 0.98

    return min(1.0, max(0.0, ((idx - 1) + stage_p) / total))


def _on_directory_change(folder: str, recursive: bool) -> tuple[Any, str]:
    import gradio as gr  # type: ignore[import-not-found]

    choices, selected, status = _build_video_choices(folder, recursive=recursive)
    return gr.update(choices=choices, value=selected), status


def _select_all_videos(folder: str, recursive: bool) -> tuple[Any, str]:
    import gradio as gr  # type: ignore[import-not-found]

    choices, selected, status = _build_video_choices(folder, recursive=recursive)
    return gr.update(choices=choices, value=selected), status


def build_ui(default_drive_dir: str = str(COLAB_DRIVE_ROOT)) -> Any:
    """Construye la aplicación Gradio lista para resumir listas de videos en Colab o local."""
    import gradio as gr  # type: ignore[import-not-found]

    initial_folders = _build_folder_choices(default_drive_dir)
    initial_choices, initial_selected, _ = _build_video_choices(default_drive_dir, recursive=True)
    progress_tracker = gr.Progress()

    def _refresh_all(folder: str, recursive: bool) -> tuple[Any, Any, str]:
        target_folder = (folder or "").strip() or default_drive_dir
        folders = _build_folder_choices(default_drive_dir)
        if target_folder not in folders:
            folders.insert(0, target_folder)
        choices, selected, status = _build_video_choices(target_folder, recursive=recursive)
        return (
            gr.update(choices=folders, value=target_folder),
            gr.update(choices=choices, value=selected),
            status,
        )

    def _run_batch_summary(
        selected_folder: str,
        selected_videos: list[str] | None,
        manual_or_url: str,
        engine_choice: str,
        max_minutes: float,
        model: str,
        language: str,
        focus: str,
        output_dir: str,
        skip_existing: bool,
        progress: Any = progress_tracker,
    ) -> tuple[Any, str | None, str, list[str], str]:
        manual_lines = [line.strip() for line in (manual_or_url or "").splitlines() if line.strip()]
        using_directory_selection = not manual_lines
        sources: list[str] = manual_lines if manual_lines else [v for v in (selected_videos or []) if v]

        if not sources:
            raise gr.Error(
                "Selecciona al menos un video del directorio o pega rutas/enlaces de Drive."
            )

        def _on_progress(msg: str) -> None:
            progress(_estimate_batch_progress(msg), desc=msg)

        engine_mode = "gemini" if "Gemini" in (engine_choice or "") else "local"

        batch = summarize_videos(
            sources=sources,
            output_dir=output_dir.strip() or None,
            source_dir=(selected_folder or "").strip() if using_directory_selection else None,
            max_duration_minutes=float(max_minutes),
            engine=engine_mode,
            model=model,
            language=language.strip() or "es",
            focus_instructions=focus,
            skip_existing=bool(skip_existing),
            continue_on_error=True,
            progress_callback=_on_progress,
        )

        files: list[str] = []
        preview_choices: list[tuple[str, str]] = []
        for res in batch.results:
            files.extend(
                [
                    str(res.output_video_path),
                    str(res.summary_md_path),
                    str(res.subtitles_srt_path),
                ]
            )
            preview_choices.append((res.output_video_path.name, str(res.output_video_path)))

        first_preview = preview_choices[0][1] if preview_choices else None
        if len(batch.results) == 1 and batch.skipped_count == 0 and batch.failed_count == 0:
            md_text = Path(batch.results[0].summary_md_path).read_text(encoding="utf-8")
        else:
            md_text = format_batch_markdown_report(batch)

        status_text = (
            f"Lote completado: {batch.succeeded_count} procesado(s), "
            f"{batch.skipped_count} omitido(s), {batch.failed_count} error(es)."
        )
        return (
            gr.update(choices=preview_choices, value=first_preview),
            first_preview,
            md_text,
            files,
            status_text,
        )

    with gr.Blocks(title="Resumidor de Video con IA (≤ 10 min)") as demo:
        gr.Markdown(
            "# Resumidor de Video con IA (máximo 10 minutos por video)\n"
            "Selecciona un **directorio de Google Drive** para cargar su lista de videos y resumirlos en lote "
            "(≤ 10 min cada uno), con subtítulos e ideas principales. Funciona **100% gratis en local sin API keys**, o con Gemini."
        )

        with gr.Row():
            with gr.Column(scale=1):
                engine_choice = gr.Radio(
                    label="Modo de Inteligencia Artificial",
                    choices=[
                        "Local (100% Gratis, sin API keys, Whisper + NLP)",
                        "Gemini Cloud (Requiere API Key)",
                    ],
                    value="Local (100% Gratis, sin API keys, Whisper + NLP)",
                )
                drive_folder = gr.Dropdown(
                    label="Directorio en Google Drive",
                    choices=initial_folders,
                    value=default_drive_dir,
                    allow_custom_value=True,
                    interactive=True,
                )
                with gr.Row():
                    recursive_chk = gr.Checkbox(label="Incluir subcarpetas", value=True)
                    refresh_btn = gr.Button("Actualizar carpetas y videos", variant="secondary")

                video_selector = gr.CheckboxGroup(
                    label="Videos del directorio seleccionado",
                    choices=initial_choices,
                    value=initial_selected,
                    interactive=True,
                )
                with gr.Row():
                    select_all_btn = gr.Button("Seleccionar todos", size="sm")
                    clear_sel_btn = gr.Button("Deseleccionar todos", size="sm")

                manual_source = gr.Textbox(
                    label="O pega rutas/enlaces de Drive manuales (uno por línea; si escribes aquí, se procesan solo estos)",
                    lines=2,
                    placeholder="https://drive.google.com/file/d/.../view",
                )
                max_minutes = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=10.0,
                    step=0.5,
                    label="Duración máxima por resumen (minutos)",
                )
                with gr.Row():
                    model = gr.Dropdown(
                        label="Modelo Gemini",
                        choices=list(SUPPORTED_MODELS),
                        value=DEFAULT_MODEL,
                    )
                    language = gr.Dropdown(
                        label="Idioma del reporte",
                        choices=["es", "en", "pt", "fr"],
                        value="es",
                        allow_custom_value=True,
                    )
                focus = gr.Textbox(
                    label="Enfoque opcional",
                    lines=2,
                    placeholder="Ej.: Prioriza conclusiones prácticas, ejemplos y métricas clave.",
                )
                output_dir = gr.Textbox(
                    label="Carpeta de salida en Drive (opcional)",
                    placeholder="Vacío = guardar junto a cada video origen",
                )
                skip_existing_chk = gr.Checkbox(
                    label="Omitir videos que ya tienen resumen generado",
                    value=True,
                )
                run_btn = gr.Button("Generar resúmenes", variant="primary")
                status_box = gr.Textbox(label="Estado", interactive=False)

            with gr.Column(scale=1):
                preview_selector = gr.Dropdown(
                    label="Video en vista previa",
                    choices=[],
                    value=None,
                    interactive=True,
                )
                video_preview = gr.Video(label="Vista previa del video resumido")
                download_files = gr.File(label="Archivos generados (.mp4, .md, .srt)")
                summary_markdown = gr.Markdown(label="Resumen e ideas principales")

        drive_folder.change(
            fn=_on_directory_change,
            inputs=[drive_folder, recursive_chk],
            outputs=[video_selector, status_box],
        )
        recursive_chk.change(
            fn=_on_directory_change,
            inputs=[drive_folder, recursive_chk],
            outputs=[video_selector, status_box],
        )
        refresh_btn.click(
            fn=_refresh_all,
            inputs=[drive_folder, recursive_chk],
            outputs=[drive_folder, video_selector, status_box],
        )
        select_all_btn.click(
            fn=_select_all_videos,
            inputs=[drive_folder, recursive_chk],
            outputs=[video_selector, status_box],
        )
        clear_sel_btn.click(
            fn=lambda: (gr.update(value=[]), "Selección limpiada."),
            inputs=[],
            outputs=[video_selector, status_box],
        )
        preview_selector.change(
            fn=lambda path: path,
            inputs=[preview_selector],
            outputs=[video_preview],
        )
        run_btn.click(
            fn=_run_batch_summary,
            inputs=[
                drive_folder,
                video_selector,
                manual_source,
                engine_choice,
                max_minutes,
                model,
                language,
                focus,
                output_dir,
                skip_existing_chk,
            ],
            outputs=[
                preview_selector,
                video_preview,
                summary_markdown,
                download_files,
                status_box,
            ],
        )

    return demo


def launch_app(
    share: bool = True,
    default_drive_dir: str = str(COLAB_DRIVE_ROOT),
    server_port: int | None = None,
) -> Any:
    """Lanza la interfaz web de Gradio."""
    demo = build_ui(default_drive_dir=default_drive_dir)
    return demo.launch(share=share, server_port=server_port, show_error=True)
