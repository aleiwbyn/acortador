# Resumidor de Video con IA (≤ 10 minutos) para Google Drive + Google Colab

[![Abrir en Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aleiwbyn/acortador/blob/main/colab/resumidor_video_ia.ipynb)

Aplicación modular en Python que analiza un video fuente alojado en Google Drive, extrae sus ideas principales y construye con `ffmpeg` un video resumido que **nunca excede los 10 minutos (600 s)**, junto con un reporte Markdown y subtítulos `.srt`.

Funciona con dos motores de IA:
1. **Motor Local (Por defecto, 100% Gratis, Sin API Keys):** Utiliza `faster-whisper` para transcribir el discurso con marcas de tiempo exactas y un algoritmo `TextRank` (PageRank sobre grafos semánticos) en Python puro que puntúa las ideas clave y fusiona bloques naturales de discurso. Se ejecuta directamente en la CPU/GPU de Google Colab sin costo alguno ni cuotas.
2. **Motor Gemini Cloud:** Utiliza la API de Gemini (`google-genai >= 3.2.0`) para análisis audiovisual multimodal completo si dispones de una API key.

## Entregables por cada video procesado

- `<nombre>_resumen.mp4`: video resumido con las secciones clave en orden lógico.
- `<nombre>_resumen.md`: resumen ejecutivo, lista numerada de ideas principales y tabla de timestamps (original vs. resumen).
- `<nombre>_resumen.srt`: subtítulos sincronizados con la línea de tiempo del resumen.

## Uso rápido en Google Colab (Recomendado - Sin API Keys y con sincronización automática)

1. Haz clic en el enlace directo para abrir siempre la versión más reciente desde GitHub:
   https://colab.research.google.com/github/aleiwbyn/acortador/blob/main/colab/resumidor_video_ia.ipynb
2. Ejecuta la celda 1 para montar tu **Google Drive** (`/content/drive/MyDrive`) e instalar dependencias (`faster-whisper`, `av<18.0.0`, `ffmpeg`, `gradio`, `gdown`, etc.).
3. Ejecuta la celda 2: La API key es **opcional**. Puedes dejarla vacía para usar el motor local gratuito.
4. Ejecuta la celda 3 para descargar y sincronizar automáticamente en 1 segundo los últimos cambios desde GitHub hacia `/content/acortador` (si ya tienes una sesión activa de Colab, basta con re-ejecutar esta celda para actualizar sin recargar ni resubir archivos).
5. Elige cómo trabajar:
   - **Modo A (Celda 4A):** interfaz web **Gradio** con explorador de carpetas de Drive, selector de motor (`Local (100% Gratis, sin API keys)` vs `Gemini Cloud`), deslizador de duración (1–10 min) y descarga/previsualización directa.
   - **Modo B (Celda 4B):** formulario nativo `#@param` con ruta en Drive o enlace compartido de Google Drive, seleccionando `motor_ia = "local"`.

## Uso local (CLI o interfaz web)

Requiere Python 3.10+ y `ffmpeg` / `ffprobe` en `PATH`:

```bash
pip install -r requirements.txt

# Resumen 100% gratis con motor local (Whisper + TextRank):
python -m acortador --source "https://drive.google.com/file/d/TU_FILE_ID/view" --engine local --max-minutes 10

# O usando Gemini si tienes API key configurada:
python -m acortador --source "mi_video.mp4" --engine gemini --max-minutes 10

# Interfaz web Gradio:
python -m acortador --ui --drive-dir "./videos" --no-share
```

## Estructura del proyecto

- `acortador/local_analyzer.py`: motor local sin API keys con `faster-whisper` y grafo `TextRank` para selección de segmentos clave con audio/discurso y fallback visual.
- `acortador/ai_analyzer.py`: cliente para análisis multimodal con Gemini Cloud.
- `acortador/models.py`: modelos de datos, parseo de timestamps y motor estricto de presupuesto de duración (`MAX_ALLOWED_DURATION_SEC = 600.0`).
- `acortador/drive_utils.py`: detección de Drive montado, listado de videos y descarga de enlaces compartidos con `gdown`.
- `acortador/video_editor.py`: inspección con `ffprobe`, recorte frame-accurate, concatenación `ffmpeg` con `-threads 0 -preset ultrafast`, y generación de `.srt` y `.md`.
- `acortador/pipeline.py`: orquestación de extremo a extremo y salvaguarda final de duración ≤ 10 min.
- `acortador/ui.py` y `acortador/__main__.py`: interfaz web Gradio y CLI.
- `scripts/build_colab_notebook.py`: empaquetador que regenera `colab/resumidor_video_ia.ipynb`.

## Regenerar el notebook y ejecutar pruebas

```bash
python scripts/build_colab_notebook.py
python -m pytest -v
```
