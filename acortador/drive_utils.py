"""Utilidades para detectar, listar y descargar videos desde Google Drive."""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from urllib.parse import parse_qs, urlparse

COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive")
SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".mpeg",
    ".mpg",
    ".wmv",
    ".flv",
    ".3gp",
    ".m4v",
}

_DRIVE_FILE_PATH_RE = re.compile(r"/file/d/([a-zA-Z0-9_-]{10,})")
_DRIVE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{10,}$")
_GENERATED_SUMMARY_SUFFIXES = ("_resumen", "_resumen_ia")
_EXCLUDED_SUBDIRS = {"__pycache__", "node_modules"}


def is_drive_mounted(drive_root: Path = COLAB_DRIVE_ROOT) -> bool:
    """Indica si la carpeta raíz de Google Drive está disponible localmente."""
    return drive_root.exists() and drive_root.is_dir()


def is_generated_summary_file(path: Path) -> bool:
    """Indica si un archivo corresponde a un resumen generado previamente."""
    return path.stem.lower().endswith(_GENERATED_SUMMARY_SUFFIXES)


def _unicode_variants(raw_path: str) -> list[str]:
    variants = [raw_path]
    nfc = unicodedata.normalize("NFC", raw_path)
    nfd = unicodedata.normalize("NFD", raw_path)
    if nfc not in variants:
        variants.append(nfc)
    if nfd not in variants:
        variants.append(nfd)
    return variants


def resolve_directory_source(
    folder: str | Path,
    drive_root: Path = COLAB_DRIVE_ROOT,
) -> Path:
    """Resuelve una carpeta local o relativa a Google Drive con soporte Unicode NFC/NFD."""
    cleaned = str(folder).strip().strip('"').strip("'")
    if not cleaned:
        raise ValueError("Debes proporcionar una ruta de carpeta válida.")

    for variant in _unicode_variants(cleaned):
        candidate = Path(variant).expanduser()
        if candidate.is_dir():
            return candidate.resolve()

        if not candidate.is_absolute():
            drive_candidate = drive_root / variant.lstrip("/\\")
            if drive_candidate.is_dir():
                return drive_candidate.resolve()

    mounted_hint = (
        f"Drive montado en {drive_root}."
        if is_drive_mounted(drive_root)
        else f"Drive no está montado en {drive_root}."
    )
    raise FileNotFoundError(f"No se encontró el directorio: {cleaned}. {mounted_hint}")


def list_drive_subfolders(
    root_folder: str | Path,
    max_depth: int = 3,
    max_results: int = 200,
) -> list[Path]:
    """Lista la carpeta raíz y sus subcarpetas visibles hasta max_depth niveles."""
    direct = Path(root_folder).expanduser()
    if direct.is_dir():
        root = direct
    else:
        try:
            root = resolve_directory_source(root_folder)
        except (ValueError, FileNotFoundError):
            return []

    if max_results <= 0:
        return []

    subdirs: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        current = Path(dirpath)
        depth = len(current.relative_to(root).parts)
        if depth >= max_depth:
            dirnames[:] = []
            continue

        dirnames[:] = sorted(
            d for d in dirnames if not d.startswith(".") and d not in _EXCLUDED_SUBDIRS
        )
        for dirname in dirnames:
            subdirs.append(current / dirname)
            if len(subdirs) >= max_results - 1:
                return [root, *sorted(subdirs, key=lambda p: p.as_posix().lower())]

    return [root, *sorted(subdirs, key=lambda p: p.as_posix().lower())]


def extract_drive_file_id(source: str) -> str | None:
    """Extrae el file_id de un enlace compartido de Google Drive o devuelve None."""
    text = source.strip().strip('"').strip("'")
    if not text.lower().startswith(("http://", "https://")):
        return None

    parsed = urlparse(text)
    host = parsed.netloc.lower()
    if host not in {"drive.google.com", "docs.google.com"} and not host.endswith(".drive.google.com"):
        return None

    match = _DRIVE_FILE_PATH_RE.search(parsed.path)
    if match:
        return match.group(1)

    query_ids = parse_qs(parsed.query).get("id")
    if query_ids and _DRIVE_ID_RE.match(query_ids[0]):
        return query_ids[0]

    return None


def list_drive_videos(
    folder: str | Path,
    recursive: bool = True,
    max_results: int = 200,
    exclude_summaries: bool = True,
) -> list[Path]:
    """Lista archivos de video soportados dentro de una carpeta de Drive."""
    direct = Path(folder).expanduser()
    if direct.exists() and direct.is_dir():
        root = direct
    else:
        try:
            root = resolve_directory_source(folder)
        except (ValueError, FileNotFoundError):
            return []

    found: list[Path] = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for filename in sorted(filenames):
                if filename.startswith("."):
                    continue
                candidate = Path(dirpath) / filename
                if candidate.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
                    continue
                if exclude_summaries and is_generated_summary_file(candidate):
                    continue
                found.append(candidate)
                if len(found) >= max_results:
                    return sorted(found, key=lambda p: p.as_posix().lower())
    else:
        for candidate in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if candidate.name.startswith(".") or not candidate.is_file():
                continue
            if candidate.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
                continue
            if exclude_summaries and is_generated_summary_file(candidate):
                continue
            found.append(candidate)
            if len(found) >= max_results:
                break

    return sorted(found, key=lambda p: p.as_posix().lower())


def resolve_video_source(
    source: str,
    download_dir: Path,
    drive_root: Path = COLAB_DRIVE_ROOT,
) -> tuple[Path, bool]:
    """Resuelve una ruta local/Drive montada o descarga un enlace de Google Drive."""
    cleaned = source.strip().strip('"').strip("'")
    if not cleaned:
        raise ValueError("Debes proporcionar una ruta de video en Drive o un enlace de Google Drive.")

    file_id = extract_drive_file_id(cleaned)
    if file_id:
        import gdown  # type: ignore[import-not-found]

        download_dir.mkdir(parents=True, exist_ok=True)
        output_hint = f"{download_dir}{os.sep}"
        downloaded = gdown.download(id=file_id, output=output_hint, quiet=False, fuzzy=True)
        if not downloaded:
            raise RuntimeError(
                f"No se pudo descargar el archivo de Google Drive (ID: {file_id}). "
                "Verifica que el enlace tenga permiso 'Cualquier persona con el enlace'."
            )
        resolved = Path(downloaded).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"La descarga de Drive no generó el archivo esperado: {resolved}")
        return resolved, True

    # Intentar resolver tanto con la cadena directa como con variantes NFC y NFD
    variants = [cleaned]
    nfc = unicodedata.normalize("NFC", cleaned)
    nfd = unicodedata.normalize("NFD", cleaned)
    if nfc not in variants:
        variants.append(nfc)
    if nfd not in variants:
        variants.append(nfd)

    for variant in variants:
        candidate = Path(variant).expanduser()
        if candidate.is_file():
            return candidate.resolve(), False

        if not candidate.is_absolute():
            drive_candidate = drive_root / variant.lstrip("/\\")
            if drive_candidate.is_file():
                return drive_candidate.resolve(), False

    mounted_hint = (
        f"Drive montado en {drive_root}."
        if is_drive_mounted(drive_root)
        else f"Drive no está montado en {drive_root}."
    )
    raise FileNotFoundError(f"No se encontró el video fuente: {cleaned}. {mounted_hint}")


def default_output_path(
    source_path: Path,
    output_dir: str | Path | None = None,
    drive_root: Path = COLAB_DRIVE_ROOT,
) -> Path:
    """Calcula la ruta de salida '<stem>_resumen.mp4' sin colisionar con el origen."""
    if output_dir is not None and str(output_dir).strip():
        raw_dir = Path(str(output_dir).strip()).expanduser()
        if not raw_dir.is_absolute() and not raw_dir.exists() and is_drive_mounted(drive_root):
            target_dir = drive_root / str(output_dir).strip().lstrip("/\\")
        else:
            target_dir = raw_dir
    else:
        target_dir = source_path.parent

    candidate = target_dir / f"{source_path.stem}_resumen.mp4"
    if candidate.resolve() == source_path.resolve():
        candidate = target_dir / f"{source_path.stem}_resumen_ia.mp4"
    return candidate
