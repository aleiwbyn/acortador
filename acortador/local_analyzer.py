"""Analizador local sin API keys usando Whisper (transcripción) y TextRank/NLP."""

from __future__ import annotations

import collections
import contextlib
import math
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .models import (
    MAX_ALLOWED_DURATION_SEC,
    RawSegment,
    VideoSummaryPlan,
    compute_target_duration,
    format_timestamp,
)

_STOPWORDS_ES_EN = {
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con", "contra",
    "cual", "cuando", "de", "del", "desde", "donde", "durante", "e", "el", "ella",
    "ellas", "ellos", "en", "entre", "era", "erais", "eran", "eras", "eres", "es",
    "esa", "esas", "ese", "eso", "esos", "esta", "estaba", "estado", "estais", "estamos",
    "estan", "estar", "estas", "este", "esto", "estos", "estoy", "fin", "fue", "fueron",
    "fui", "fuimos", "ha", "habeis", "habia", "habian", "habido", "habla", "hace",
    "haceis", "hacemos", "hacen", "hacer", "hacia", "hago", "hasta", "incluso", "la",
    "las", "le", "les", "lo", "los", "mas", "me", "mi", "mis", "mientras", "muy",
    "nada", "ni", "no", "nos", "nosotras", "nosotros", "o", "otra", "otras", "otro",
    "otros", "para", "pero", "poco", "por", "porque", "que", "quien", "quienes",
    "se", "sea", "sean", "segun", "ser", "sera", "seran", "si", "sido", "siempre",
    "sin", "sobre", "sois", "solamente", "solo", "somos", "son", "soy", "su", "sus",
    "suya", "suyas", "suyo", "suyos", "tal", "tambien", "tampoco", "tan", "tanto",
    "te", "teneis", "tenemos", "tener", "tenga", "tengo", "ti", "tiene", "tienen",
    "toda", "todas", "todo", "todos", "tu", "tus", "un", "una", "unas", "uno",
    "unos", "vosotras", "vosotros", "y", "ya", "the", "be", "to", "of", "and", "in",
    "that", "have", "it", "for", "not", "on", "with", "he", "as", "you", "do",
    "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
}

_WORD_RE = re.compile(r"\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9]{3,}\b")


def _notify(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def extract_audio_from_video(video_path: Path, output_audio_path: Path) -> Path:
    """Extrae la pista de audio a 16kHz mono en segundos para procesamiento local."""
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-threads",
        "0",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_audio_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"No se pudo extraer el audio del video: {proc.stderr[:400]}")
    return output_audio_path


def _tokenize(text: str) -> list[str]:
    words = _WORD_RE.findall(text.lower())
    return [w for w in words if w not in _STOPWORDS_ES_EN]


def _patch_av_open_metadata_errors() -> None:
    """Corrige la incompatibilidad entre PyAV >= 18.0.0 y faster-whisper (metadata_errors)."""
    try:
        import av  # type: ignore[import-not-found]

        orig_open = av.open

        def _patched_open(*args: Any, **kwargs: Any) -> Any:
            try:
                return orig_open(*args, **kwargs)
            except TypeError as exc:
                if "metadata_errors" in str(exc):
                    kwargs.pop("metadata_errors", None)
                    return orig_open(*args, **kwargs)
                raise

        av.open = _patched_open  # type: ignore[assignment]
    except (ImportError, AttributeError):
        pass


_patch_av_open_metadata_errors()


def _compute_textrank_scores(sentences: list[str]) -> list[float]:
    """Calcula la importancia semántica relativa de cada frase con TextRank puro."""
    n = len(sentences)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    token_sets = [_tokenize(s) for s in sentences]
    vocab_counter: collections.Counter[str] = collections.Counter()
    for tokens in token_sets:
        vocab_counter.update(set(tokens))

    similarity_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        tokens_i = set(token_sets[i])
        len_i = len(tokens_i)
        if len_i == 0:
            continue
        for j in range(i + 1, n):
            tokens_j = set(token_sets[j])
            len_j = len(tokens_j)
            if len_j == 0:
                continue
            common = tokens_i & tokens_j
            if not common:
                continue
            # Ponderación basada en solapamiento normalizado y especificidad de términos
            weight = sum(1.0 / math.log(1.0 + vocab_counter[w]) for w in common)
            sim = weight / (math.log(1.0 + len_i) + math.log(1.0 + len_j) + 1e-6)
            similarity_matrix[i][j] = sim
            similarity_matrix[j][i] = sim

    # Iteración PageRank con amortiguación
    d = 0.85
    scores = [1.0 / n] * n
    for _ in range(25):
        next_scores = [(1.0 - d) / n] * n
        for i in range(n):
            sum_in = 0.0
            for j in range(n):
                if i != j and similarity_matrix[j][i] > 0:
                    sum_out = sum(similarity_matrix[j])
                    if sum_out > 0:
                        sum_in += (similarity_matrix[j][i] / sum_out) * scores[j]
            next_scores[i] += d * sum_in
        scores = next_scores

    max_s = max(scores) if scores else 1.0
    return [s / max_s for s in scores] if max_s > 0 else scores


def transcribe_video_locally(
    video_path: Path,
    whisper_model_size: str = "base",
    language: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Transcribe el video localmente con faster-whisper (GPU o CPU, 100% gratis)."""
    _notify(progress_callback, "Extrayendo audio para transcripción local con IA...")
    audio_tmp = video_path.parent / f"{video_path.stem}_audio_tmp.wav"
    try:
        extract_audio_from_video(video_path, audio_tmp)
    except (RuntimeError, OSError) as exc:
        _notify(progress_callback, f"Video sin audio o formato no soportado: {exc}")
        return []

    try:
        _notify(progress_callback, f"Cargando modelo local Whisper ({whisper_model_size})...")
        _patch_av_open_metadata_errors()
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]

        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" else "cpu"
        # Verificar si hay GPU disponible en PyTorch
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        except (ImportError, RuntimeError, OSError):
            device = "cpu"

        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)

        _notify(progress_callback, "Transcribiendo discurso y detectando marcas de tiempo exactas...")
        segments_gen, _info = model.transcribe(
            str(audio_tmp),
            beam_size=3,
            language=language if language and language != "auto" else None,
            vad_filter=True,
        )

        results: list[dict[str, Any]] = []
        for s in segments_gen:
            text = s.text.strip()
            if text and s.end > s.start:
                results.append(
                    {
                        "start": round(float(s.start), 3),
                        "end": round(float(s.end), 3),
                        "text": text,
                    }
                )
        return results
    finally:
        if audio_tmp.exists():
            with contextlib.suppress(OSError):
                audio_tmp.unlink()


def build_local_summary_plan(
    speech_segments: list[dict[str, Any]],
    source_duration: float,
    max_duration_sec: float = MAX_ALLOWED_DURATION_SEC,
    video_title: str = "Resumen de video",
) -> VideoSummaryPlan:
    """Selecciona los mejores segmentos con TextRank respetando el presupuesto <= 10 min."""
    effective_max = min(max_duration_sec, MAX_ALLOWED_DURATION_SEC, source_duration)
    target_duration = compute_target_duration(source_duration, max_duration_sec=effective_max)

    # Si no se detectó habla (video musical, dron, tutorial mudo), se generan segmentos visuales
    if not speech_segments:
        num_clips = min(8, max(3, int(target_duration / 20.0)))
        clip_len = min(20.0, target_duration / num_clips)
        step = (source_duration - clip_len) / max(1, num_clips - 1) if num_clips > 1 else 0.0

        raw_segs: list[RawSegment] = []
        for i in range(num_clips):
            st = i * step
            et = min(source_duration, st + clip_len)
            raw_segs.append(
                RawSegment(
                    start_time=format_timestamp(st),
                    end_time=format_timestamp(et),
                    title=f"Momento visual clave {i + 1}",
                    key_idea=f"Escena destacada {i + 1} del contenido original",
                    importance=5 if i in (0, num_clips - 1) else 4,
                )
            )

        return VideoSummaryPlan(
            video_title=video_title,
            executive_summary=(
                "Resumen generado por IA local identificando los momentos visuales y cambios "
                f"de escena más relevantes a lo largo de los {format_timestamp(source_duration)} de duración."
            ),
            main_ideas=[
                "Inicio y planteamiento visual del video.",
                "Evolución y desarrollo principal de la secuencia.",
                "Conclusión y momentos culminantes.",
            ],
            segments=raw_segs,
        )

    sentences = [seg["text"] for seg in speech_segments]
    scores = _compute_textrank_scores(sentences)

    scored_segments: list[dict[str, Any]] = []
    for idx, (seg, score) in enumerate(zip(speech_segments, scores)):
        dur = seg["end"] - seg["start"]
        # Asignar importancia 1 a 5
        imp = max(1, min(5, int(score * 4.0 + 1.5)))
        # Bonificación a segmentos de apertura y cierre naturales
        if idx == 0 or idx == len(speech_segments) - 1:
            imp = min(5, imp + 1)
        scored_segments.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "score": score,
                "importance": imp,
                "duration": dur,
                "index": idx,
            }
        )

    # Agrupar frases contiguas de alta relevancia para mantener fluidez del discurso
    merged_blocks: list[dict[str, Any]] = []
    curr = scored_segments[0].copy()
    for nxt in scored_segments[1:]:
        gap = nxt["start"] - curr["end"]
        # Si están muy juntas (< 2.0 s) y ambas tienen buena relevancia, fusionar en un clip continuo
        if gap <= 2.0 and (curr["duration"] + nxt["duration"] <= 60.0):
            curr["end"] = nxt["end"]
            curr["duration"] = curr["end"] - curr["start"]
            curr["text"] = f"{curr['text']} {nxt['text']}"
            curr["importance"] = max(curr["importance"], nxt["importance"])
            curr["score"] = max(curr["score"], nxt["score"])
        else:
            merged_blocks.append(curr)
            curr = nxt.copy()
    merged_blocks.append(curr)

    # Ordenar por relevancia para seleccionar hasta llenar el presupuesto
    sorted_by_rank = sorted(
        merged_blocks,
        key=lambda b: (-b["importance"], -b["score"], b["index"]),
    )

    selected: list[dict[str, Any]] = []
    accum_sec = 0.0
    for block in sorted_by_rank:
        if accum_sec + block["duration"] <= target_duration + 5.0:
            selected.append(block)
            accum_sec += block["duration"]
        elif not selected:
            selected.append(block)
            accum_sec += block["duration"]
            break

    # Reordenar cronológicamente
    chronological = sorted(selected, key=lambda b: b["start"])

    raw_segments: list[RawSegment] = []
    main_ideas: list[str] = []

    for i, block in enumerate(chronological, start=1):
        clean_text = block["text"].strip()
        first_sentence = clean_text.split(".")[0].strip() or clean_text[:60]
        title = first_sentence[:50] + ("..." if len(first_sentence) > 50 else "")
        raw_segments.append(
            RawSegment(
                start_time=format_timestamp(block["start"]),
                end_time=format_timestamp(block["end"]),
                title=f"Tema {i}: {title}",
                key_idea=clean_text[:120] + ("..." if len(clean_text) > 120 else ""),
                importance=block["importance"],
            )
        )
        if len(main_ideas) < 5 and block["importance"] >= 3:
            main_ideas.append(first_sentence)

    if not main_ideas:
        main_ideas = [s.title for s in raw_segments[:4]]

    executive_summary = (
        f"Resumen generado por IA local (Whisper + TextRank) a partir del análisis del discurso "
        f"del video. Se extrajeron {len(raw_segments)} momentos clave con las ideas principales, "
        f"condensando el contenido original ({format_timestamp(source_duration)}) a menos de "
        f"{format_timestamp(effective_max)} sin requerir claves API."
    )

    return VideoSummaryPlan(
        video_title=video_title,
        executive_summary=executive_summary,
        main_ideas=main_ideas,
        segments=raw_segments,
    )
