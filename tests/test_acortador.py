"""Pruebas automatizadas del paquete acortador y del notebook de Colab."""

from __future__ import annotations

import base64
import json
import sys
import types
import zlib
from pathlib import Path
from typing import Any

import pytest

from acortador import MAX_ALLOWED_DURATION_SEC, summarize_video
from acortador.ai_analyzer import analyze_video_with_gemini
from acortador.drive_utils import (
    default_output_path,
    extract_drive_file_id,
    list_drive_videos,
    resolve_video_source,
)
from acortador.local_analyzer import (
    _compute_textrank_scores,
    _patch_av_open_metadata_errors,
    _preload_cuda_libraries,
    build_local_summary_plan,
    transcribe_video_locally,
)
from acortador.models import (
    NormalizedSegment,
    RawSegment,
    SummaryResult,
    VideoMetadata,
    VideoSummaryPlan,
    compute_target_duration,
    format_srt_timestamp,
    format_timestamp,
    normalize_and_enforce_budget,
    parse_timestamp,
)
from acortador.video_editor import (
    PROXY_THRESHOLD_BYTES,
    build_segment_ffmpeg_cmd,
    needs_analysis_proxy,
    write_markdown_summary,
    write_srt_subtitles,
)


def test_timestamp_parsing_and_formatting() -> None:
    assert parse_timestamp("05:30") == pytest.approx(330.0)
    assert parse_timestamp("01:02:03.5") == pytest.approx(3723.5)
    assert parse_timestamp("42.25") == pytest.approx(42.25)
    assert parse_timestamp(90) == pytest.approx(90.0)

    assert format_timestamp(65) == "01:05"
    assert format_timestamp(3661) == "01:01:01"
    assert format_srt_timestamp(3723.456) == "01:02:03,456"

    for bad in ("", "   ", "-5", "12:60", "01:00:60", "abc", -1.0):
        with pytest.raises(ValueError):
            parse_timestamp(bad)
    with pytest.raises(TypeError):
        parse_timestamp(True)


def test_compute_target_duration_profiles_and_limits() -> None:
    short_20 = compute_target_duration(20.0, MAX_ALLOWED_DURATION_SEC)
    assert 8.0 <= short_20 <= 20.0

    short_4m = compute_target_duration(240.0, MAX_ALLOWED_DURATION_SEC)
    assert 90.0 <= short_4m <= 120.0

    med_12m = compute_target_duration(720.0, MAX_ALLOWED_DURATION_SEC)
    assert 200.0 <= med_12m <= 360.0

    long_90m = compute_target_duration(5400.0, MAX_ALLOWED_DURATION_SEC)
    assert long_90m == pytest.approx(540.0)
    assert long_90m <= MAX_ALLOWED_DURATION_SEC

    with pytest.raises(ValueError):
        compute_target_duration(1200.0, 601.0)
    with pytest.raises(ValueError):
        compute_target_duration(0.0, 300.0)


def test_normalize_and_enforce_budget_merges_prioritizes_and_caps() -> None:
    raw = [
        RawSegment(
            start_time="00:10",
            end_time="02:10",
            title="Intro contexto",
            key_idea="Contexto inicial",
            importance=2,
        ),
        RawSegment(
            start_time="02:11",
            end_time="05:10",
            title="Tesis central",
            key_idea="Argumento principal",
            importance=5,
        ),
        RawSegment(
            start_time="10:00",
            end_time="14:00",
            title="Demostración",
            key_idea="Prueba práctica",
            importance=5,
        ),
        RawSegment(
            start_time="20:00",
            end_time="25:00",
            title="Conclusiones",
            key_idea="Resumen de cierre",
            importance=4,
        ),
        RawSegment(
            start_time="26:00",
            end_time="25:00",
            title="Invertido",
            key_idea="Debe descartarse",
            importance=5,
        ),
    ]

    # Segmentos 1 y 2 se fusionan (brecha 1 s <= 1.5 s) -> duración 300 s (imp 5)
    # Segmento 3 -> 240 s (imp 5)
    # Segmento 4 -> 300 s (imp 4), debe recortarse al presupuesto restante de 60 s.
    normalized = normalize_and_enforce_budget(
        raw,
        source_duration=2400.0,
        max_duration_sec=900.0,  # Debe clampirse al límite estricto de 600 s.
    )

    total = sum(seg.duration for seg in normalized)
    assert total <= MAX_ALLOWED_DURATION_SEC
    assert total == pytest.approx(600.0)
    assert [seg.start for seg in normalized] == sorted(seg.start for seg in normalized)
    assert any("Tesis central" in seg.title for seg in normalized)
    assert not any(seg.title == "Invertido" for seg in normalized)

    with pytest.raises(ValueError):
        normalize_and_enforce_budget(
            [
                RawSegment(
                    start_time="05:00",
                    end_time="04:00",
                    title="Inválido",
                    key_idea="Nada",
                    importance=3,
                )
            ],
            source_duration=600.0,
        )


def test_drive_utils_url_listing_and_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert (
        extract_drive_file_id("https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQr/view?usp=sharing")
        == "1AbCdEfGhIjKlMnOpQr"
    )
    assert (
        extract_drive_file_id("https://drive.google.com/open?id=1AbCdEfGhIjKlMnOpQr")
        == "1AbCdEfGhIjKlMnOpQr"
    )
    assert (
        extract_drive_file_id("https://drive.google.com/uc?id=1AbCdEfGhIjKlMnOpQr&export=download")
        == "1AbCdEfGhIjKlMnOpQr"
    )
    assert extract_drive_file_id("/content/drive/MyDrive/mi_video.mp4") is None
    assert extract_drive_file_id(r"C:\videos\mi_video.mp4") is None
    assert extract_drive_file_id("https://www.youtube.com/watch?v=12345678901") is None

    drive_root = tmp_path / "MyDrive"
    (drive_root / "sub").mkdir(parents=True)
    (drive_root / ".hidden").mkdir()
    v1 = drive_root / "charla.mp4"
    v2 = drive_root / "sub" / "clase.mkv"
    v_hidden = drive_root / ".hidden" / "oculto.mp4"
    doc = drive_root / "notas.txt"
    for p in (v1, v2, v_hidden, doc):
        p.write_bytes(b"data")

    assert list_drive_videos(drive_root, recursive=True) == [v1, v2]
    assert list_drive_videos(drive_root, recursive=False) == [v1]

    resolved_rel, is_url = resolve_video_source(
        "sub/clase.mkv",
        download_dir=tmp_path / "dl",
        drive_root=drive_root,
    )
    assert resolved_rel == v2.resolve()
    assert is_url is False

    fake_downloaded = tmp_path / "dl" / "descargado.mp4"

    def fake_download(id: str, output: str, quiet: bool, fuzzy: bool) -> str:
        assert id == "1AbCdEfGhIjKlMnOpQr"
        fake_downloaded.parent.mkdir(parents=True, exist_ok=True)
        fake_downloaded.write_bytes(b"video")
        return str(fake_downloaded)

    fake_gdown = types.SimpleNamespace(download=fake_download)
    monkeypatch.setitem(sys.modules, "gdown", fake_gdown)

    resolved_url, is_url_2 = resolve_video_source(
        "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQr/view",
        download_dir=tmp_path / "dl",
        drive_root=drive_root,
    )
    assert resolved_url == fake_downloaded.resolve()
    assert is_url_2 is True

    out = default_output_path(v1, drive_root=drive_root)
    assert out == drive_root / "charla_resumen.mp4"


class _FakeState:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeGeminiFiles:
    def __init__(self, fail_processing: bool = False) -> None:
        self.fail_processing = fail_processing
        self.uploaded_names: list[str] = []
        self.deleted_names: list[str] = []
        self._get_calls = 0

    def upload(self, file: str) -> Any:
        # Simula la restricción de encabezados HTTP en Python estándar (http.client putheader)
        file.encode("ascii")
        name = f"files/{Path(file).stem}"
        self.uploaded_names.append(name)
        return types.SimpleNamespace(
            name=name,
            uri=f"https://generativelanguage.googleapis.com/v1beta/{name}",
            mime_type="video/mp4",
            state=_FakeState("PROCESSING"),
        )

    def get(self, name: str) -> Any:
        self._get_calls += 1
        final_state = "FAILED" if self.fail_processing else "ACTIVE"
        return types.SimpleNamespace(
            name=name,
            uri=f"https://generativelanguage.googleapis.com/v1beta/{name}",
            mime_type="video/mp4",
            state=_FakeState(final_state),
        )

    def delete(self, name: str) -> None:
        self.deleted_names.append(name)


class _FakeGeminiInteractions:
    def __init__(self, plan_json: str, fail_on_create: bool = False) -> None:
        self.plan_json = plan_json
        self.fail_on_create = fail_on_create
        self.calls: list[dict[str, Any]] = []
        self._polls = 0

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.fail_on_create:
            raise RuntimeError("Fallo simulado de API")
        return types.SimpleNamespace(id="int_001", status="in_progress")

    def get(self, interaction_id: str) -> Any:
        self._polls += 1
        status = "queued" if self._polls == 1 else "completed"
        return types.SimpleNamespace(
            id=interaction_id,
            status=status,
            output_text=self.plan_json if status == "completed" else None,
        )


class _FakeGeminiClient:
    def __init__(self, plan_json: str, fail_on_create: bool = False) -> None:
        self.files = _FakeGeminiFiles()
        self.interactions = _FakeGeminiInteractions(plan_json, fail_on_create=fail_on_create)


def test_ai_analyzer_interactions_agentic_and_cleanup(tmp_path: Path) -> None:
    video = tmp_path / "clase.mp4"
    video.write_bytes(b"mp4bytes")

    plan = VideoSummaryPlan(
        video_title="Clase de Arquitectura",
        executive_summary="Resumen ejecutivo de prueba.",
        main_ideas=["Idea 1", "Idea 2"],
        segments=[
            RawSegment(
                start_time="00:30",
                end_time="02:00",
                title="Idea 1",
                key_idea="Explicación clave",
                importance=5,
            )
        ],
    )
    wrapped_json = f"```json\n{plan.model_dump_json()}\n```"

    client_long = _FakeGeminiClient(wrapped_json)
    parsed = analyze_video_with_gemini(
        video_path=video,
        source_duration=600.0,
        model="gemini-4-argon",
        client=client_long,
        poll_interval=0.0,
    )
    assert parsed.video_title == "Clase de Arquitectura"
    assert client_long.files.deleted_names == ["files/clase"]

    call = client_long.interactions.calls[0]
    assert call["input"][0]["type"] == "video"
    assert call["input"][1]["type"] == "text"
    assert call["response_format"]["mime_type"] == "application/json"
    assert "schema" in call["response_format"]

    # Probar archivo con tildes NFD y caracteres no ASCII para verificar que no falle por ASCII encoding
    accent_video = tmp_path / "video\u0301_grabacio\u0301n_espan\u0303ol.mp4"
    accent_video.write_bytes(b"accentmp4bytes")
    client_accent = _FakeGeminiClient(wrapped_json)
    parsed_accent = analyze_video_with_gemini(
        video_path=accent_video,
        source_duration=600.0,
        model="gemini-3.5-flash-lite",
        client=client_accent,
        poll_interval=0.0,
    )
    assert parsed_accent.video_title == "Clase de Arquitectura"

    # Limpieza garantizada en finally aunque falle interactions.create
    failing_client = _FakeGeminiClient(plan.model_dump_json(), fail_on_create=True)
    with pytest.raises(RuntimeError):
        analyze_video_with_gemini(
            video_path=video,
            source_duration=400.0,
            model="gemini-4-hydrogen",
            client=failing_client,
            poll_interval=0.0,
        )
    assert failing_client.files.deleted_names == ["files/clase"]

    with pytest.raises(ValueError):
        analyze_video_with_gemini(
            video_path=video,
            source_duration=400.0,
            model="gemini-1.5-pro",
            client=client_long,
        )


def test_video_editor_cmds_proxy_and_outputs(tmp_path: Path) -> None:
    small_mp4 = VideoMetadata(
        path=tmp_path / "small.mp4",
        duration=100.0,
        width=1280,
        height=720,
        fps=30.0,
        has_audio=True,
        size_bytes=50 * 1024 * 1024,
    )
    large_mp4 = VideoMetadata(
        path=tmp_path / "large.mp4",
        duration=1800.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
        size_bytes=PROXY_THRESHOLD_BYTES + 1,
    )
    mkv_video = VideoMetadata(
        path=tmp_path / "grabacion.mkv",
        duration=120.0,
        width=1280,
        height=720,
        fps=30.0,
        has_audio=False,
        size_bytes=10 * 1024 * 1024,
    )

    assert needs_analysis_proxy(small_mp4) is False
    assert needs_analysis_proxy(large_mp4) is True
    assert needs_analysis_proxy(mkv_video) is True

    seg = NormalizedSegment(
        start=12.0,
        end=42.0,
        title="Punto clave",
        key_idea="Detalle importante",
        importance=5,
    )
    cmd_audio = build_segment_ffmpeg_cmd(small_mp4.path, seg, tmp_path / "c1.mp4", has_audio=True)
    assert "-af" in cmd_audio
    assert "-an" not in cmd_audio

    cmd_silent = build_segment_ffmpeg_cmd(mkv_video.path, seg, tmp_path / "c2.mp4", has_audio=False)
    assert "-an" in cmd_silent
    assert "-af" not in cmd_silent

    srt_path = write_srt_subtitles([seg], tmp_path / "out.srt")
    srt_text = srt_path.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:30,000" in srt_text
    assert "Punto clave" in srt_text

    result = SummaryResult(
        source_path=small_mp4.path,
        output_video_path=tmp_path / "out.mp4",
        summary_md_path=tmp_path / "out.md",
        subtitles_srt_path=srt_path,
        source_duration=100.0,
        summary_duration=30.0,
        video_title="Demo",
        executive_summary="Resumen breve.",
        main_ideas=["Idea A"],
        segments=[seg],
        model_used="gemini-4-argon",
        used_proxy=False,
    )
    md_path = write_markdown_summary(result, tmp_path / "out.md")
    md_text = md_path.read_text(encoding="utf-8")
    assert "# Demo" in md_text
    assert "| 1 | Punto clave | 00:12–00:42 | 00:00–00:30 | 5/5 | Detalle importante |" in md_text


def test_pipeline_end_to_end_mocked_and_hard_trim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_video = tmp_path / "conferencia.mp4"
    source_video.write_bytes(b"source")

    plan = VideoSummaryPlan(
        video_title="Conferencia IA",
        executive_summary="Puntos esenciales de la charla.",
        main_ideas=["Introducción", "Conclusión"],
        segments=[
            RawSegment(
                start_time="00:10",
                end_time="05:10",
                title="Introducción",
                key_idea="Planteamiento inicial",
                importance=5,
            ),
            RawSegment(
                start_time="10:00",
                end_time="15:00",
                title="Conclusión",
                key_idea="Resultados finales",
                importance=5,
            ),
        ],
    )
    fake_client = _FakeGeminiClient(plan.model_dump_json())

    monkeypatch.setattr("acortador.pipeline.ensure_ffmpeg_available", lambda: None)

    probe_calls: list[Path] = []
    trimmed_calls: list[float] = []

    def fake_probe(path: Path) -> VideoMetadata:
        probe_calls.append(path)
        if path == source_video.resolve():
            return VideoMetadata(
                path=path,
                duration=1200.0,
                width=1920,
                height=1080,
                fps=30.0,
                has_audio=True,
                size_bytes=50 * 1024 * 1024,
            )
        # Simula que en el primer render excede ligeramente para verificar hard_trim_video.
        duration = 605.0 if not trimmed_calls else 600.0
        return VideoMetadata(
            path=path,
            duration=duration,
            width=1920,
            height=1080,
            fps=30.0,
            has_audio=True,
            size_bytes=20 * 1024 * 1024,
        )

    def fake_render(
        input_path: Path,
        segments: list[NormalizedSegment],
        output_path: Path,
        has_audio: bool,
        progress_callback: Any = None,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"rendered")
        return output_path

    def fake_hard_trim(input_path: Path, output_path: Path, max_duration_sec: float) -> Path:
        trimmed_calls.append(max_duration_sec)
        output_path.write_bytes(b"trimmed")
        return output_path

    monkeypatch.setattr("acortador.pipeline.probe_video", fake_probe)
    monkeypatch.setattr("acortador.pipeline.render_summary_video", fake_render)
    monkeypatch.setattr("acortador.pipeline.hard_trim_video", fake_hard_trim)

    with pytest.raises(ValueError):
        summarize_video(str(source_video), max_duration_minutes=11.0, client=fake_client)

    result = summarize_video(
        source=str(source_video),
        max_duration_minutes=10.0,
        client=fake_client,
    )
    assert trimmed_calls == [600.0]
    assert result.summary_duration <= MAX_ALLOWED_DURATION_SEC
    assert result.output_video_path.is_file()
    assert result.summary_md_path.is_file()
    assert result.subtitles_srt_path.is_file()


def test_colab_notebook_structure_and_byte_exact_payload() -> None:
    root = Path(__file__).resolve().parent.parent
    nb_path = root / "colab" / "resumidor_video_ia.ipynb"
    assert nb_path.is_file()

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) == 6
    assert nb["cells"][0]["cell_type"] == "markdown"
    assert all(c["cell_type"] == "code" for c in nb["cells"][1:])

    unpack_src = "".join(nb["cells"][3]["source"])
    marker = '_PACKAGE_PAYLOAD = "'
    start = unpack_src.index(marker) + len(marker)
    end = unpack_src.index('"', start)
    payload_b64 = unpack_src[start:end]

    unpacked = json.loads(zlib.decompress(base64.b64decode(payload_b64)).decode("utf-8"))
    disk_modules = sorted((root / "acortador").glob("*.py"), key=lambda p: p.name)
    assert sorted(unpacked.keys()) == [p.name for p in disk_modules]

    for module in disk_modules:
        assert base64.b64decode(unpacked[module.name]) == module.read_bytes()


def test_local_analyzer_textrank_and_plan() -> None:
    # 1. TextRank computation
    texts = [
        "El aprendizaje automático permite a las computadoras aprender patrones a partir de datos.",
        "Los datos son el combustible del aprendizaje automático moderno en la inteligencia artificial.",
        "Hoy vamos a almorzar una pizza con queso.",
    ]
    scores = _compute_textrank_scores(texts)
    assert len(scores) == 3
    # Las dos primeras oraciones comparten términos clave, deben puntuar más que la pizza
    assert scores[0] > scores[2]
    assert scores[1] > scores[2]

    # 2. Plan generation with speech segments
    speech = [
        {
            "start": 10.0,
            "end": 70.0,
            "text": "Introducción clave al tema principal de inteligencia artificial y modelos.",
        },
        {
            "start": 75.0,
            "end": 180.0,
            "text": "Desarrollo profundo de los algoritmos y aplicaciones prácticas en la industria.",
        },
        {
            "start": 200.0,
            "end": 250.0,
            "text": "Comentario secundario y saludos al público presente en el evento.",
        },
    ]
    plan = build_local_summary_plan(
        speech_segments=speech,
        source_duration=600.0,
        max_duration_sec=300.0,
    )
    assert len(plan.segments) >= 1
    assert any(s.importance >= 4 for s in plan.segments)

    # 3. Plan generation with empty speech (fallback visual)
    empty_plan = build_local_summary_plan(
        speech_segments=[],
        source_duration=1200.0,
        max_duration_sec=300.0,
    )
    assert len(empty_plan.segments) >= 1
    total_sec = sum(
        parse_timestamp(s.end_time) - parse_timestamp(s.start_time)
        for s in empty_plan.segments
    )
    assert total_sec <= 300.0


def test_summarize_video_local_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_video = tmp_path / "charla_offline.mp4"
    source_video.write_bytes(b"dummy")

    monkeypatch.setattr("acortador.pipeline.ensure_ffmpeg_available", lambda: None)

    fake_speech = [
        {"start": 10.0, "end": 80.0, "text": "Concepto principal del proyecto."},
        {"start": 90.0, "end": 150.0, "text": "Implementación técnica del algoritmo."},
    ]
    monkeypatch.setattr(
        "acortador.pipeline.transcribe_video_locally",
        lambda *args, **kwargs: fake_speech,
    )

    def fake_probe(path: Path) -> VideoMetadata:
        return VideoMetadata(
            path=path,
            duration=300.0,
            width=1280,
            height=720,
            fps=30.0,
            has_audio=True,
            size_bytes=10 * 1024 * 1024,
        )

    def fake_render(
        input_path: Path,
        segments: list[NormalizedSegment],
        output_path: Path,
        has_audio: bool,
        progress_callback: Any = None,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"rendered_local")
        return output_path

    monkeypatch.setattr("acortador.pipeline.probe_video", fake_probe)
    monkeypatch.setattr("acortador.pipeline.render_summary_video", fake_render)

    res = summarize_video(
        source=str(source_video),
        max_duration_minutes=5.0,
        engine="local",
    )
    assert res.output_video_path.is_file()
    assert res.summary_duration <= 300.0
    assert len(res.segments) >= 1


def test_cli_parser_engine() -> None:
    from acortador.__main__ import build_parser

    parser = build_parser()
    args_default = parser.parse_args(["--source", "video.mp4"])
    assert args_default.engine == "local"

    args_gemini = parser.parse_args(["--source", "video.mp4", "--engine", "gemini"])
    assert args_gemini.engine == "gemini"


def test_patch_av_open_metadata_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class FakeAV:
        @staticmethod
        def open(*args: Any, **kwargs: Any) -> str:
            if "metadata_errors" in kwargs:
                raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")
            calls.append(kwargs)
            return "ok"

    fake_module = FakeAV()
    monkeypatch.setitem(sys.modules, "av", fake_module)

    _patch_av_open_metadata_errors()
    # Debe eliminar metadata_errors y llamar a open con éxito
    result = fake_module.open("dummy.wav", metadata_errors="ignore", mode="r")
    assert result == "ok"
    assert "metadata_errors" not in calls[0]
    assert calls[0]["mode"] == "r"


def test_transcribe_video_locally_cuda_fallback_on_libcublas_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")
    audio_tmp = tmp_path / "demo_audio_tmp.wav"

    def fake_extract(video_path: Path, output_audio_path: Path) -> Path:
        output_audio_path.write_bytes(b"wav")
        return output_audio_path

    monkeypatch.setattr("acortador.local_analyzer.extract_audio_from_video", fake_extract)
    monkeypatch.setattr("acortador.local_analyzer._preload_cuda_libraries", lambda: False)

    fake_torch = types.ModuleType("torch")
    fake_cuda = types.SimpleNamespace(is_available=lambda: True)
    fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    attempts: list[tuple[str, str]] = []

    class FakeWhisperModel:
        def __init__(self, size: str, device: str, compute_type: str) -> None:
            self.device = device
            attempts.append((device, compute_type))

        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
            if self.device == "cuda":
                def failing_gen() -> Any:
                    if False:
                        yield None
                    raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")

                return failing_gen(), None

            def ok_gen() -> Any:
                yield types.SimpleNamespace(start=1.0, end=12.5, text="Idea clave.")

            return ok_gen(), None

    fake_fw = types.ModuleType("faster_whisper")
    fake_fw.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

    messages: list[str] = []
    results = transcribe_video_locally(video, progress_callback=messages.append)

    assert attempts == [("cuda", "float16"), ("cpu", "int8")]
    assert results == [{"start": 1.0, "end": 12.5, "text": "Idea clave."}]
    assert any("CPU" in m for m in messages)
    assert not audio_tmp.exists()


def test_preload_cuda_libraries_loads_discovered_cublas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_lib_dir = tmp_path / "nvidia" / "cublas" / "lib"
    fake_lib_dir.mkdir(parents=True)
    cublas_file = fake_lib_dir / "libcublas.so.12"
    cublas_file.write_bytes(b"elf")

    monkeypatch.setattr("acortador.local_analyzer._discover_cuda_lib_dirs", lambda: [fake_lib_dir])
    monkeypatch.setattr("acortador.local_analyzer.ctypes.RTLD_GLOBAL", 256, raising=False)

    loaded_calls: list[tuple[str, int]] = []

    def fake_cdll(path: str, mode: int = 0) -> object:
        loaded_calls.append((path, mode))
        return object()

    monkeypatch.setattr("acortador.local_analyzer.ctypes.CDLL", fake_cdll)

    assert _preload_cuda_libraries() is True
    assert loaded_calls == [(str(cublas_file), 256)]



