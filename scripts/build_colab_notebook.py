"""Genera el notebook autocontenido de Google Colab (colab/resumidor_video_ia.ipynb)."""

from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = PROJECT_ROOT / "acortador"
OUTPUT_NOTEBOOK = PROJECT_ROOT / "colab" / "resumidor_video_ia.ipynb"


def _to_source_lines(text: str) -> list[str]:
    lines = text.strip("\n").splitlines(keepends=True)
    return lines if lines else [""]


def _markdown_cell(cell_id: str, text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": _to_source_lines(text),
    }


def _code_cell(cell_id: str, code: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _to_source_lines(code),
    }


def encode_package_payload(package_dir: Path = PACKAGE_DIR) -> str:
    """Empaqueta los módulos Python de acortador/ en un blob zlib+base64."""
    modules = sorted(package_dir.glob("*.py"), key=lambda p: p.name)
    if not modules:
        raise FileNotFoundError(f"No se encontraron archivos .py en {package_dir}")

    raw_files = {
        module.name: base64.b64encode(module.read_bytes()).decode("ascii")
        for module in modules
    }
    compressed = zlib.compress(json.dumps(raw_files, sort_keys=True).encode("utf-8"), level=9)
    return base64.b64encode(compressed).decode("ascii")


def build_notebook_dict(package_dir: Path = PACKAGE_DIR) -> dict[str, Any]:
    """Construye la estructura nbformat 4.5 completa para Google Colab."""
    payload = encode_package_payload(package_dir)

    cell_1_intro = """
# Resumidor de Video con IA (máximo 10 minutos) — Google Drive + Google Colab

Esta aplicación toma un **video fuente alojado en tu Google Drive**, extrae sus **ideas principales** respetando un presupuesto estricto de **≤ 10 minutos (600 s)** y genera en tu Drive:

- `<nombre>_resumen.mp4`: video resumido ensamblado con `ffmpeg`.
- `<nombre>_resumen.md`: resumen ejecutivo, ideas principales y tabla de timestamps.
- `<nombre>_resumen.srt`: subtítulos alineados a la nueva línea de tiempo.

### Motores disponibles
- **Motor Local (por defecto, 100% gratis, sin API keys):** usa `faster-whisper` + `TextRank` directamente en Colab.
- **Motor Gemini Cloud (opcional):** análisis audiovisual multimodal con Gemini si configuras tu `GEMINI_API_KEY`.

### Flujo de uso y sincronización automática
1. Ejecuta la **Celda 1** para montar Google Drive e instalar dependencias.
2. Ejecuta la **Celda 2** (opcional: solo si usarás Gemini en la nube).
3. Ejecuta la **Celda 3** para inicializar y **sincronizar automáticamente el código más reciente desde GitHub** (tarda 1 segundo, sin tener que resubir el notebook).
4. Elige **Modo A (Interfaz Web Gradio)** en la Celda 4A o **Modo B (Formulario Colab)** en la Celda 4B.
"""

    cell_2_setup = """
#@title 1. Montar Google Drive e instalar dependencias
import shutil
import subprocess
from google.colab import drive

drive.mount("/content/drive")

subprocess.check_call(
    [
        "pip",
        "install",
        "-q",
        "-U",
        "faster-whisper>=1.0.0",
        "av<18.0.0",
        "google-genai>=3.2.0",
        "pydantic>=2.0",
        "gradio>=4.0",
        "gdown>=5.0",
    ]
)

if shutil.which("nvidia-smi") is not None:
    try:
        subprocess.check_call(
            [
                "pip",
                "install",
                "-q",
                "nvidia-cublas-cu12",
                "nvidia-cudnn-cu12==9.*",
            ]
        )
    except subprocess.CalledProcessError as exc:
        print(f"Aviso: no se pudieron instalar librerías CUDA 12 ({exc}); se usará fallback automático a CPU si hace falta.")

if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
    print("Instalando ffmpeg...")
    subprocess.check_call(["apt-get", "update", "-qq"])
    subprocess.check_call(["apt-get", "install", "-y", "-qq", "ffmpeg"])

print("Google Drive montado y dependencias listas.")
"""

    cell_3_apikey = """
#@title 2. (Opcional) Configurar GEMINI_API_KEY solo si usas el modo Gemini
import os

api_key = ""
try:
    from google.colab import userdata

    api_key = (userdata.get("GEMINI_API_KEY") or "").strip()
except Exception:
    api_key = ""

if not api_key:
    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()

if api_key:
    os.environ["GEMINI_API_KEY"] = api_key
    print("GEMINI_API_KEY configurada. Podras usar tanto el modo Local como Gemini.")
else:
    print("Sin API Key: Se usara el motor Local (100% Gratis con Whisper + TextRank).")
"""

    cell_4_unpack = f'''
#@title 3. Inicializar y sincronizar paquete acortador/
import base64
import json
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

sincronizar_desde_github = True  #@param {{type:"boolean"}}
_REPO_URL = "https://github.com/aleiwbyn/acortador.git"
_REPO_DIR = Path("/content/acortador_repo")
_TARGET_DIR = Path("/content/acortador")
_PACKAGE_PAYLOAD = "{payload}"

_TARGET_DIR.mkdir(parents=True, exist_ok=True)
_synced_from_github = False

if sincronizar_desde_github:
    try:
        if (_REPO_DIR / ".git").is_dir():
            subprocess.run(
                ["git", "-C", str(_REPO_DIR), "pull", "--ff-only"],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            if _REPO_DIR.exists():
                shutil.rmtree(_REPO_DIR)
            subprocess.run(
                ["git", "clone", "--depth", "1", _REPO_URL, str(_REPO_DIR)],
                check=True,
                capture_output=True,
                text=True,
            )
        _src_pkg = _REPO_DIR / "acortador"
        _src_files = sorted(_src_pkg.glob("*.py"))
        if _src_files:
            for _old in _TARGET_DIR.glob("*.py"):
                _old.unlink()
            for _py_file in _src_files:
                shutil.copy2(_py_file, _TARGET_DIR / _py_file.name)
            _synced_from_github = True
            print(f"Código actualizado desde GitHub en {{_TARGET_DIR}} ({{len(_src_files)}} módulos).")
    except Exception as _exc:
        print(f"No se pudo sincronizar desde GitHub ({{_exc}}); usando copia embebida...")

if not _synced_from_github:
    _decoded = json.loads(zlib.decompress(base64.b64decode(_PACKAGE_PAYLOAD)).decode("utf-8"))
    for _old in _TARGET_DIR.glob("*.py"):
        _old.unlink()
    for _filename, _b64_content in _decoded.items():
        (_TARGET_DIR / _filename).write_bytes(base64.b64decode(_b64_content))
    print(f"Motor acortador cargado desde copia embebida en {{_TARGET_DIR}} ({{len(_decoded)}} módulos).")

for _mod in [k for k in list(sys.modules) if k == "acortador" or k.startswith("acortador.")]:
    del sys.modules[_mod]

if "/content" not in sys.path:
    sys.path.insert(0, "/content")
'''

    cell_5_gradio = """
#@title 4A. Modo A — Aplicación Web Interactiva (Gradio)
from acortador.ui import launch_app

launch_app(share=True, default_drive_dir="/content/drive/MyDrive")
"""

    cell_6_form = """
#@title 4B. Modo B — Ejecución directa con Formulario Colab
from pathlib import Path
from IPython.display import Markdown, Video, display
from acortador import summarize_video

video_fuente = "/content/drive/MyDrive/mi_video.mp4"  #@param {type:"string"}
motor_ia = "local"  #@param ["local", "gemini"]
carpeta_salida = ""  #@param {type:"string"}
duracion_maxima_min = 10.0  #@param {type:"slider", min:1, max:10, step:0.5}
modelo_gemini = "gemini-3.5-flash-lite"  #@param ["gemini-3.5-flash-lite", "gemini-4-argon", "gemini-4-hydrogen", "gemini-4-neon"]
idioma = "es"  #@param ["es", "en", "pt", "fr"]
enfoque_opcional = ""  #@param {type:"string"}

resultado = summarize_video(
    source=video_fuente,
    output_dir=carpeta_salida.strip() or None,
    max_duration_minutes=float(duracion_maxima_min),
    engine=motor_ia,
    model=modelo_gemini,
    language=idioma,
    focus_instructions=enfoque_opcional,
    progress_callback=lambda msg: print(f"[acortador] {msg}"),
)

md_report = Path(resultado.summary_md_path).read_text(encoding="utf-8")
display(Markdown(md_report))

embed_preview = resultado.output_video_path.stat().st_size <= 50 * 1024 * 1024
display(Video(str(resultado.output_video_path), embed=embed_preview))
"""

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {
                "name": OUTPUT_NOTEBOOK.name,
                "provenance": [],
                "gpuType": "T4",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
            "accelerator": "GPU",
        },
        "cells": [
            _markdown_cell("intro", cell_1_intro),
            _code_cell("setup_drive_deps", cell_2_setup),
            _code_cell("setup_api_key", cell_3_apikey),
            _code_cell("unpack_package", cell_4_unpack),
            _code_cell("mode_a_gradio", cell_5_gradio),
            _code_cell("mode_b_colab_form", cell_6_form),
        ],
    }


def write_notebook(
    output_path: Path = OUTPUT_NOTEBOOK,
    package_dir: Path = PACKAGE_DIR,
) -> Path:
    """Genera y guarda el archivo .ipynb en disco."""
    notebook = build_notebook_dict(package_dir=package_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


if __name__ == "__main__":
    generated = write_notebook()
    print(f"Notebook generado en: {generated}")
