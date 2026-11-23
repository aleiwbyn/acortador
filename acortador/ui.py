"""Interfaz web interactiva con Gradio para Google Colab y ejecución local."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .ai_analyzer import DEFAULT_MODEL, SUPPORTED_MODELS
from .drive_utils import COLAB_DRIVE_ROOT, list_drive_videos
from .pipeline import summarize_video


def _refresh_drive_dropdown(folder: str) -> tuple[Any, str]:
    import gradio as gr  # type: ignore[import-not-found]

    videos = list_drive_videos(folder)
    choices = [str(p) for p in videos]
    if choices:
        status = f"Se encontraron {len(choices)} video(s) en {folder}."
        return gr.update(choices=choices, value=choices[0]), status
    return (
        gr.update(choices=[], value=None),
        f"No se encontraron videos en {folder}. Puedes pegar un enlace compartido o ruta manual abajo.",
    )


def build_ui(default_drive_dir: str = str(COLAB_DRIVE_ROOT)) -> Any:
    """Construye la aplicación Gradio lista para usarse en Colab o local."""
    import gradio as gr  # type: ignore[import-not-found]

    initial_videos = [str(p) for p in list_drive_videos(default_drive_dir)]
    progress_tracker = gr.Progress()

    def _run_summary(
        selected_video: str | None,
        manual_or_url: str,
        engine_choice: str,
        max_minutes: float,
        model: str,
        language: str,
        focus: str,
        output_dir: str,
        progress: Any = progress_tracker,
    ) -> tuple[str | None, str, list[str], str]:
        source = (manual_or_url or "").strip() or (selected_video or "").strip()
        if not source:
            raise gr.Error(
                "Selecciona un video de la lista de Google Drive o pega una ruta/enlace de Drive."
            )

        logs: list[str] = []

        def _on_progress(msg: str) -> None:
            logs.append(msg)
            p = 0.5
            if "Resolviendo" in msg or "Inspeccionando" in msg or "Extrayendo" in msg:
                p = 0.15
            elif "proxy" in msg or "Whisper" in msg:
                p = 0.3
            elif "Subiendo" in msg or "Transcribiendo" in msg:
                p = 0.45
            elif "Analizando" in msg or "Gemini" in msg or "TextRank" in msg:
                p = 0.65
            elif "Renderizando" in msg or "clip" in msg:
                p = 0.85
            elif "Listo" in msg or "Ensamblando" in msg:
                p = 0.95
            progress(p, desc=msg)

        engine_mode = "gemini" if "Gemini" in (engine_choice or "") else "local"

        result = summarize_video(
            source=source,
            output_dir=output_dir.strip() or None,
            max_duration_minutes=float(max_minutes),
            engine=engine_mode,
            model=model,
            language=language.strip() or "es",
            focus_instructions=focus,
            progress_callback=_on_progress,
        )

        md_text = Path(result.summary_md_path).read_text(encoding="utf-8")
        files = [
            str(result.output_video_path),
            str(result.summary_md_path),
            str(result.subtitles_srt_path),
        ]
        status_text = (
            f"Completado: {result.summary_duration:.1f} s "
            f"(original: {result.source_duration:.1f} s) | Guardado en {result.output_video_path}"
        )
        return str(result.output_video_path), md_text, files, status_text

    with gr.Blocks(title="Resumidor de Video con IA (≤ 10 min)") as demo:
        gr.Markdown(
            "# Resumidor de Video con IA (máximo 10 minutos)\n"
            "Extrae las mejores partes de un video de Google Drive y ensambla un video editado (≤ 10 min) "
            "con subtítulos e ideas principales. Funciona **100% gratis en local sin API keys**, o con Gemini."
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
                drive_folder = gr.Textbox(
                    label="Carpeta en Google Drive",
                    value=default_drive_dir,
                    placeholder="/content/drive/MyDrive/Videos",
                )
                refresh_btn = gr.Button("Actualizar lista de videos", variant="secondary")
                video_dropdown = gr.Dropdown(
                    label="Videos encontrados en Google Drive",
                    choices=initial_videos,
                    value=initial_videos[0] if initial_videos else None,
                    interactive=True,
                )
                manual_source = gr.Textbox(
                    label="O pega una ruta manual o enlace compartido de Google Drive",
                    placeholder="https://drive.google.com/file/d/.../view",
                )
                max_minutes = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=10.0,
                    step=0.5,
                    label="Duración máxima del resumen (minutos)",
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
                    placeholder="Vacío = guardar en la misma carpeta del video origen",
                )
                run_btn = gr.Button("Generar video resumido", variant="primary")
                status_box = gr.Textbox(label="Estado", interactive=False)

            with gr.Column(scale=1):
                video_preview = gr.Video(label="Vista previa del video resumido")
                download_files = gr.File(label="Archivos generados (.mp4, .md, .srt)")
                summary_markdown = gr.Markdown(label="Ideas principales y líneas de tiempo")

        refresh_btn.click(
            fn=_refresh_drive_dropdown,
            inputs=[drive_folder],
            outputs=[video_dropdown, status_box],
        )
        run_btn.click(
            fn=_run_summary,
            inputs=[
                video_dropdown,
                manual_source,
                engine_choice,
                max_minutes,
                model,
                language,
                focus,
                output_dir,
            ],
            outputs=[video_preview, summary_markdown, download_files, status_box],
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
