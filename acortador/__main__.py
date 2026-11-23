"""Interfaz de línea de comandos para resumir videos desde Drive o archivo local."""

from __future__ import annotations

import argparse
import sys

from .ai_analyzer import DEFAULT_MODEL, SUPPORTED_MODELS
from .drive_utils import COLAB_DRIVE_ROOT
from .pipeline import summarize_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acortador",
        description="Resume un video de Google Drive en un clip <= 10 minutos con sus ideas principales.",
    )
    parser.add_argument(
        "--source",
        help="Ruta del video en Drive/local o enlace compartido de Google Drive.",
    )
    parser.add_argument(
        "--output",
        help="Ruta exacta del archivo .mp4 resumido.",
    )
    parser.add_argument(
        "--output-dir",
        help="Carpeta destino para guardar <nombre>_resumen.mp4, .md y .srt.",
    )
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=10.0,
        help="Duración máxima permitida en minutos (0.5 a 10.0; por defecto 10.0).",
    )
    parser.add_argument(
        "--engine",
        choices=["local", "gemini"],
        default="local",
        help="Motor de análisis: 'local' (100% gratis, sin API keys, Whisper+TextRank) o 'gemini' (nube).",
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default=DEFAULT_MODEL,
        help=f"Modelo Gemini a utilizar si --engine=gemini (por defecto {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--language",
        default="es",
        help="Idioma para el resumen y subtítulos (por defecto 'es').",
    )
    parser.add_argument(
        "--focus",
        default="",
        help="Instrucciones opcionales de enfoque.",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Inicia la interfaz web interactiva con Gradio.",
    )
    parser.add_argument(
        "--drive-dir",
        default=str(COLAB_DRIVE_ROOT),
        help="Carpeta inicial de Google Drive para el modo --ui.",
    )
    parser.add_argument(
        "--no-share",
        action="store_true",
        help="Desactiva el túnel público share=True de Gradio en modo --ui.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.ui:
        from .ui import launch_app

        launch_app(share=not args.no_share, default_drive_dir=args.drive_dir)
        return 0

    if not args.source:
        parser.error("Debes especificar --source o usar --ui.")

    result = summarize_video(
        source=args.source,
        output_path=args.output,
        output_dir=args.output_dir,
        max_duration_minutes=args.max_minutes,
        engine=args.engine,
        model=args.model,
        language=args.language,
        focus_instructions=args.focus,
        progress_callback=lambda msg: print(f"[acortador] {msg}"),
    )
    print(f"Video resumido: {result.output_video_path}")
    print(f"Resumen Markdown: {result.summary_md_path}")
    print(f"Subtítulos SRT: {result.subtitles_srt_path}")
    print(
        f"Duración: {result.summary_duration:.1f} s "
        f"(original: {result.source_duration:.1f} s, segmentos: {len(result.segments)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
