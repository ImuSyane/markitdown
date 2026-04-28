import io
import subprocess
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from markitdown import StreamInfo
from markitdown_ocr._ocr_service import (
    OpenAICompatibleVisionOCRBackend,
    PaddleOCROCRBackend,
    PaddleOCRVLOCRBackend,
)
from markitdown_ocr._pdf_converter_with_ocr import PdfConverterWithOCR
from markitdown_ocr._plugin import register_converters


class DummyMarkItDown:
    def __init__(self) -> None:
        self.registrations = []

    def register_converter(self, converter, priority=0.0):
        self.registrations.append((converter, priority))


class FakePaddleResult:
    def __init__(self, markdown: str = "") -> None:
        self.markdown = markdown
        self.json = {"text": markdown}


class FakePaddleOCRResult:
    def __init__(self, texts: list[str], scores: list[float] | None = None) -> None:
        self.json = {"res": {"rec_texts": texts, "rec_scores": scores or []}}


class FakePaddleOCRPipeline:
    def __init__(self) -> None:
        self.paths = []

    def predict(self, path: str):
        self.paths.append(path)
        return [FakePaddleOCRResult(["Alpha", "Beta"], [0.9, 0.8])]


class FakePaddlePipeline:
    def __init__(self, markdown: str = "PADDLE_TEXT") -> None:
        self.markdown = markdown
        self.paths = []

    def predict(self, path: str):
        self.paths.append(path)
        return [FakePaddleResult(self.markdown)]


class FailingPaddlePipeline:
    def predict(self, path: str):
        raise RuntimeError(f"simulated runtime failure for {path}")


@pytest.mark.parametrize(
    ("quality", "max_dimension"),
    [("low", 1280), ("medium", 1800), ("high", 2400)],
)
def test_openai_backend_quality_profiles(quality: str, max_dimension: int) -> None:
    backend = OpenAICompatibleVisionOCRBackend(
        client=MagicMock(),
        model="gpt-4o",
        quality=quality,
    )
    assert backend._quality_max_dimension() == max_dimension


def test_register_converters_uses_paddleocr_backend_by_default() -> None:
    markitdown = DummyMarkItDown()

    register_converters(markitdown)

    assert len(markitdown.registrations) == 4
    for converter, priority in markitdown.registrations:
        assert priority == -1.0
        assert converter.ocr_service is not None
        assert converter.ocr_service.backend_name == "paddleocr"


def test_register_converters_uses_openai_backend_when_requested() -> None:
    markitdown = DummyMarkItDown()
    client = MagicMock()

    register_converters(
        markitdown,
        ocr_backend="openai_compatible",
        llm_client=client,
        llm_model="gpt-4o",
        ocr_quality="high",
    )

    assert len(markitdown.registrations) == 4
    for converter, priority in markitdown.registrations:
        assert priority == -1.0
        assert converter.ocr_service is not None
        assert converter.ocr_service.backend_name == "openai_compatible"
        assert converter.ocr_service.quality == "high"


def test_register_converters_uses_paddle_vl_local_mode_when_requested() -> None:
    markitdown = DummyMarkItDown()

    register_converters(
        markitdown,
        ocr_backend="paddleocr_vl",
        ocr_quality="low",
    )

    assert len(markitdown.registrations) == 4
    for converter, priority in markitdown.registrations:
        assert priority == -1.0
        assert converter.ocr_service is not None
        assert converter.ocr_service.backend_name == "paddleocr_vl"
        assert converter.ocr_service.requested_mode == "local"
        assert converter.ocr_service.quality == "low"


def test_paddleocr_backend_extracts_rec_texts(monkeypatch) -> None:
    backend = PaddleOCROCRBackend()
    pipeline = FakePaddleOCRPipeline()
    monkeypatch.setattr(backend, "_get_pipeline", lambda: pipeline)

    result = backend.extract_text(
        io.BytesIO(b"fake-image"),
        stream_info=StreamInfo(extension=".png", mimetype="image/png"),
    )

    assert result.text == "Alpha\nBeta"
    assert result.confidence == pytest.approx(0.85)
    assert result.backend_used == "paddleocr"


def test_paddle_mlx_model_alias_defaults_to_q8() -> None:
    backend = PaddleOCRVLOCRBackend(mode="local")

    assert backend._resolve_mlx_model_name() == "mlx-community/PaddleOCR-VL-1.5-8bit"
    assert (
        backend._resolve_mlx_model_name("PaddleOCR-VL-1.5")
        == "mlx-community/PaddleOCR-VL-1.5-8bit"
    )


def test_paddle_vl_uses_bundled_layout_model_when_available() -> None:
    backend = PaddleOCRVLOCRBackend(mode="local")

    kwargs = backend._with_bundled_layout_detection_model({})

    assert kwargs["layout_detection_model_name"] == "PP-DocLayoutV3"
    model_dir = Path(kwargs["layout_detection_model_dir"])
    assert model_dir.name == "PP-DocLayoutV3"
    assert (model_dir / "inference.json").is_file()
    assert (model_dir / "inference.yml").is_file()
    assert (model_dir / "inference.pdiparams").is_file()


def test_paddle_vl_preserves_explicit_layout_model_dir() -> None:
    backend = PaddleOCRVLOCRBackend(mode="local")

    kwargs = backend._with_bundled_layout_detection_model(
        {"layout_detection_model_dir": "/custom/layout"}
    )

    assert kwargs == {"layout_detection_model_dir": "/custom/layout"}


def test_local_mlx_server_candidates_increment_ports() -> None:
    backend = PaddleOCRVLOCRBackend(mode="local")

    candidates = backend._local_mlx_server_candidates("http://localhost:8111/")

    assert candidates[0] == (8111, "http://127.0.0.1:8111/")
    assert candidates[1] == (8112, "http://127.0.0.1:8112/")
    assert len(candidates) == 8


def test_register_converters_uses_paddle_server_mode_when_requested() -> None:
    markitdown = DummyMarkItDown()

    register_converters(
        markitdown,
        ocr_backend="paddleocr_vl",
        ocr_mode="server",
        ocr_server_url="http://localhost:8118/v1",
        ocr_server_backend="mlx-vlm-server",
        ocr_api_key="secret",
    )

    assert len(markitdown.registrations) == 4
    for converter, priority in markitdown.registrations:
        assert priority == -1.0
        assert converter.ocr_service is not None
        assert converter.ocr_service.backend_name == "paddleocr_vl"
        assert converter.ocr_service.requested_mode == "server"
        assert converter.ocr_service.server_url == "http://localhost:8118/v1"
        assert converter.ocr_service.server_backend == "mlx-vlm-server"
        assert converter.ocr_service.api_key == "secret"


def test_register_converters_maps_legacy_server_url_to_server_mode() -> None:
    markitdown = DummyMarkItDown()

    register_converters(
        markitdown,
        ocr_backend="paddleocr_vl",
        ocr_server_url="http://localhost:8118/v1",
    )

    for converter, _priority in markitdown.registrations:
        assert converter.ocr_service.requested_mode == "server"


def test_register_converters_passes_pdf_artifact_options() -> None:
    markitdown = DummyMarkItDown()

    register_converters(
        markitdown,
        llm_client=MagicMock(),
        llm_model="gpt-4o",
        ocr_artifact_export=False,
        ocr_artifact_dir="/tmp/ocr-artifacts",
        ocr_artifact_markdown_mode="image_and_text",
    )

    pdf_converter = next(
        converter
        for converter, _priority in markitdown.registrations
        if isinstance(converter, PdfConverterWithOCR)
    )

    assert pdf_converter.ocr_artifact_export is False
    assert str(pdf_converter.ocr_artifact_dir) == "/tmp/ocr-artifacts"
    assert pdf_converter.ocr_artifact_markdown_mode == "image_and_text"


def test_paddle_local_probe_prefers_cuda_when_runtime_confirms_it(monkeypatch) -> None:
    backend = PaddleOCRVLOCRBackend(mode="local")

    monkeypatch.setattr(
        backend,
        "_get_runtime_capabilities",
        lambda: {
            "platform": "linux",
            "machine": "x86_64",
            "available_device": "gpu:0",
            "compiled_with_cuda": True,
            "compiled_with_rocm": False,
            "compiled_with_xpu": False,
            "custom_devices": [],
        },
    )

    created = []

    def fake_instantiate_pipeline(**kwargs):
        created.append(kwargs)
        return FakePaddlePipeline()

    monkeypatch.setattr(backend, "_instantiate_pipeline", fake_instantiate_pipeline)

    pipeline = backend._get_pipeline()
    assert pipeline is not None
    assert backend.selected_mode == "local"
    assert backend.selected_device == "gpu"
    assert backend.probe_candidates[0] == "gpu"
    assert created[0]["device"] == "gpu"


def test_paddle_local_probe_autostarts_apple_local_server(monkeypatch) -> None:
    backend = PaddleOCRVLOCRBackend(mode="local")

    monkeypatch.setattr(
        backend,
        "_get_runtime_capabilities",
        lambda: {
            "platform": "darwin",
            "machine": "arm64",
            "available_device": "cpu",
            "compiled_with_cuda": False,
            "compiled_with_rocm": False,
            "compiled_with_xpu": False,
            "custom_devices": [],
        },
    )
    monkeypatch.setattr(
        backend,
        "_ensure_local_mlx_server",
        lambda url: (True, "started-managed-local-mlx-server"),
    )
    backend.api_model_name = "mlx-community/PaddleOCR-VL-1.5-8bit"

    created = []

    def fake_instantiate_pipeline(**kwargs):
        created.append(kwargs)
        return FakePaddlePipeline()

    monkeypatch.setattr(backend, "_instantiate_pipeline", fake_instantiate_pipeline)

    pipeline = backend._get_pipeline()
    assert pipeline is not None
    assert backend.selected_device == "apple-local-server"
    assert backend.probe_candidates == ["apple-local-server", "cpu"]
    assert created[0]["vl_rec_backend"] == "mlx-vlm-server"
    assert created[0]["vl_rec_api_model_name"] == "mlx-community/PaddleOCR-VL-1.5-8bit"


def test_paddle_local_probe_falls_back_to_cpu_when_managed_apple_local_server_fails(monkeypatch) -> None:
    backend = PaddleOCRVLOCRBackend(mode="local")

    monkeypatch.setattr(
        backend,
        "_get_runtime_capabilities",
        lambda: {
            "platform": "darwin",
            "machine": "arm64",
            "available_device": "cpu",
            "compiled_with_cuda": False,
            "compiled_with_rocm": False,
            "compiled_with_xpu": False,
            "custom_devices": [],
        },
    )
    monkeypatch.setattr(
        backend,
        "_ensure_local_mlx_server",
        lambda url: (False, "mlx auto-start failed"),
    )

    created = []

    def fake_instantiate_pipeline(**kwargs):
        created.append(kwargs)
        return FakePaddlePipeline()

    monkeypatch.setattr(backend, "_instantiate_pipeline", fake_instantiate_pipeline)

    pipeline = backend._get_pipeline()
    assert pipeline is not None
    assert backend.selected_device == "cpu"
    assert backend.probe_candidates == ["apple-local-server", "cpu"]
    assert backend.fallback_reason is not None
    assert any("apple-local-server" in item for item in backend.probe_diagnostics)
    assert created[-1]["device"] == "cpu"


def test_paddle_server_mode_uses_configured_server_backend(monkeypatch) -> None:
    backend = PaddleOCRVLOCRBackend(
        mode="server",
        server_url="http://localhost:8118/v1",
        server_backend="mlx-vlm-server",
        api_key="secret",
    )

    created = []

    def fake_instantiate_pipeline(**kwargs):
        created.append(kwargs)
        return FakePaddlePipeline()

    monkeypatch.setattr(backend, "_instantiate_pipeline", fake_instantiate_pipeline)
    monkeypatch.setattr(
        backend,
        "_select_mlx_model_for_server",
        lambda url: (True, "using requested mlx model mlx-community/PaddleOCR-VL-1.5-8bit"),
    )

    pipeline = backend._get_pipeline()
    assert pipeline is not None
    assert backend.selected_mode == "server"
    assert backend.selected_device == "server"
    assert created[0]["vl_rec_backend"] == "mlx-vlm-server"
    assert created[0]["vl_rec_server_url"] == "http://localhost:8118/v1"
    assert created[0]["vl_rec_api_key"] == "secret"


def test_paddle_server_model_probe_uses_api_key(monkeypatch) -> None:
    backend = PaddleOCRVLOCRBackend(
        mode="server",
        server_url="http://localhost:8118/v1",
        server_backend="mlx-vlm-server",
        api_key="secret",
    )
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return b'{"data":[{"id":"PaddleOCR-VL-1.5"}]}'

    def fake_urlopen(request, timeout):
        captured["authorization"] = request.headers.get("Authorization")
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert backend._fetch_server_models("http://localhost:8118/v1") == [
        "PaddleOCR-VL-1.5"
    ]
    assert captured["authorization"] == "Bearer secret"


def test_local_server_probe_treats_http_404_as_reachable(monkeypatch) -> None:
    backend = PaddleOCRVLOCRBackend(mode="local")

    def fake_urlopen(_request, timeout):
        raise urllib.error.HTTPError(
            url="http://localhost:8111/",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert backend._can_connect_to_local_server("http://localhost:8111/") is True


def test_select_mlx_model_prefers_requested_exact_match(monkeypatch) -> None:
    backend = PaddleOCRVLOCRBackend(
        mode="server",
        server_url="http://localhost:8111/",
        server_backend="mlx-vlm-server",
    )
    monkeypatch.setattr(
        backend,
        "_fetch_server_models",
        lambda url: [
            "mlx-community/PaddleOCR-VL-1.5-8bit",
            "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
        ],
    )

    ok, reason = backend._select_mlx_model_for_server("http://localhost:8111/")

    assert ok is True
    assert "requested mlx model" in reason
    assert backend.api_model_name == "mlx-community/PaddleOCR-VL-1.5-8bit"


def test_select_mlx_model_accepts_any_visual_model(monkeypatch) -> None:
    backend = PaddleOCRVLOCRBackend(
        mode="server",
        server_url="http://localhost:8111/",
        server_backend="mlx-vlm-server",
        api_model_name="PaddlePaddle/PaddleOCR-VL-1.5",
    )
    monkeypatch.setattr(
        backend,
        "_fetch_server_models",
        lambda url: [
            "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
            "mlx-community/Qwen3-TTS-0.6B-4bit",
        ],
    )

    ok, reason = backend._select_mlx_model_for_server("http://localhost:8111/")

    assert ok is True
    assert "detected VLM model" in reason
    assert backend.api_model_name == "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"


def test_select_mlx_model_rejects_non_visual_models(monkeypatch) -> None:
    backend = PaddleOCRVLOCRBackend(
        mode="server",
        server_url="http://localhost:8111/",
        server_backend="mlx-vlm-server",
    )
    monkeypatch.setattr(
        backend,
        "_fetch_server_models",
        lambda url: ["mlx-community/Qwen3-TTS-0.6B-4bit"],
    )

    ok, reason = backend._select_mlx_model_for_server("http://localhost:8111/")

    assert ok is False
    assert "not visual" in reason


def test_ensure_local_mlx_server_starts_with_q8_model(monkeypatch) -> None:
    backend = PaddleOCRVLOCRBackend(mode="local")
    backend.server_url = "http://127.0.0.1:8111/"
    snapshot_path = "/tmp/mlx-snapshot"

    monkeypatch.setattr(backend, "_can_connect_to_local_server", lambda url: False)
    monkeypatch.setattr(backend, "_mlx_server_command", lambda: ["mlx_vlm.server"])
    monkeypatch.setattr(backend, "_ensure_local_mlx_snapshot", lambda: snapshot_path)
    monkeypatch.setattr(backend, "_wait_for_local_server", lambda url, timeout_seconds=30.0: True)
    monkeypatch.setattr(
        backend,
        "_select_mlx_model_for_server",
        lambda url: (True, "using requested mlx model mlx-community/PaddleOCR-VL-1.5-8bit"),
    )

    created = {}

    class DummyProcess:
        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    def fake_popen(cmd, stdout, stderr, text):
        created["cmd"] = cmd
        return DummyProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    ok, _reason = backend._ensure_local_mlx_server("http://127.0.0.1:8111/")

    assert ok is True
    assert "--model" in created["cmd"]
    assert snapshot_path in created["cmd"]
    assert backend._managed_server_cleanup_registered is True
    backend.close()
    assert backend._managed_server_process is None


def test_ensure_local_mlx_server_starts_new_port_when_existing_server_is_not_visual(
    monkeypatch,
) -> None:
    backend = PaddleOCRVLOCRBackend(mode="local")
    backend.server_url = "http://127.0.0.1:8111/"
    snapshot_path = "/tmp/mlx-snapshot"

    monkeypatch.setattr(
        backend,
        "_local_mlx_server_candidates",
        lambda url: [
            (8111, "http://127.0.0.1:8111/"),
            (8112, "http://127.0.0.1:8112/"),
        ],
    )
    monkeypatch.setattr(
        backend,
        "_can_connect_to_local_server",
        lambda url: url == "http://127.0.0.1:8111/",
    )
    monkeypatch.setattr(
        backend,
        "_select_mlx_model_for_server",
        lambda url: (
            (False, "mlx-vlm server models are not visual")
            if url == "http://127.0.0.1:8111/"
            else (True, "using requested mlx model mlx-community/PaddleOCR-VL-1.5-8bit")
        ),
    )
    monkeypatch.setattr(backend, "_mlx_server_command", lambda: ["mlx_vlm.server"])
    monkeypatch.setattr(backend, "_ensure_local_mlx_snapshot", lambda: snapshot_path)
    monkeypatch.setattr(backend, "_wait_for_local_server", lambda url, timeout_seconds=30.0: True)

    created = {}

    class DummyProcess:
        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    def fake_popen(cmd, stdout, stderr, text):
        created["cmd"] = cmd
        return DummyProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    ok, _reason = backend._ensure_local_mlx_server("http://127.0.0.1:8111/")

    assert ok is True
    assert created["cmd"][2] == "8112"
    assert created["cmd"][-1] == snapshot_path
    assert backend.server_url == "http://127.0.0.1:8112/"


def test_ensure_local_mlx_server_stops_after_port_attempt_limit(monkeypatch) -> None:
    backend = PaddleOCRVLOCRBackend(mode="local")

    monkeypatch.setattr(
        backend,
        "_local_mlx_server_candidates",
        lambda url: [
            (8111, "http://127.0.0.1:8111/"),
            (8112, "http://127.0.0.1:8112/"),
            (8113, "http://127.0.0.1:8113/"),
        ],
    )
    monkeypatch.setattr(backend, "_can_connect_to_local_server", lambda url: False)
    monkeypatch.setattr(backend, "_mlx_server_command", lambda: None)

    ok, reason = backend._ensure_local_mlx_server("http://127.0.0.1:8111/")

    assert ok is False
    assert reason == "mlx_vlm.server command not found"


def test_paddle_local_runtime_falls_back_to_cpu_after_apple_server_failure(
    monkeypatch,
) -> None:
    backend = PaddleOCRVLOCRBackend(mode="local", quality="medium")

    monkeypatch.setattr(
        backend,
        "_get_runtime_capabilities",
        lambda: {
            "platform": "darwin",
            "machine": "arm64",
            "available_device": "cpu",
            "compiled_with_cuda": False,
            "compiled_with_rocm": False,
            "compiled_with_xpu": False,
            "custom_devices": [],
        },
    )
    monkeypatch.setattr(
        backend,
        "_ensure_local_mlx_server",
        lambda url: (True, "started-managed-local-mlx-server"),
    )
    monkeypatch.setattr(backend, "_stop_managed_local_server", lambda: None)

    created = []

    def fake_instantiate_pipeline(**kwargs):
        created.append(kwargs)
        if kwargs.get("vl_rec_backend") == "mlx-vlm-server":
            return FailingPaddlePipeline()
        return FakePaddlePipeline(markdown="CPU_TEXT")

    monkeypatch.setattr(backend, "_instantiate_pipeline", fake_instantiate_pipeline)

    result = backend.extract_text(
        io.BytesIO(b"fake-image-bytes"),
        stream_info=StreamInfo(extension=".png", mimetype="image/png"),
    )

    assert result.text == "CPU_TEXT"
    assert result.backend_used == "paddleocr_vl"
    assert result.metadata["selected_device"] == "cpu"
    assert result.metadata["probe_candidates"] == ["apple-local-server", "cpu"]
    assert "apple-local-server to cpu" in str(result.metadata["fallback_reason"])
    assert any("apple-local-server" in item for item in result.metadata["probe_diagnostics"])
    assert created[0]["vl_rec_backend"] == "mlx-vlm-server"
    assert created[-1]["device"] == "cpu"


def test_paddle_backend_returns_error_when_dependency_missing() -> None:
    backend = PaddleOCRVLOCRBackend(mode="local")
    result = backend.extract_text(
        io.BytesIO(b"fake-image"),
        stream_info=StreamInfo(extension=".png"),
    )
    assert result.text == ""
    assert result.backend_used == "paddleocr_vl"
    assert result.error is not None
    assert result.metadata["requested_mode"] == "local"


def test_paddle_backend_extract_text_preserves_selection_metadata(monkeypatch) -> None:
    backend = PaddleOCRVLOCRBackend(mode="local", quality="medium")
    backend._pipeline = FakePaddlePipeline(markdown="PADDLE_TEXT")
    backend.selected_mode = "local"
    backend.selected_device = "cpu"
    backend.probe_candidates = ["cpu"]
    backend.runtime_capabilities = {"available_device": "cpu"}

    result = backend.extract_text(
        io.BytesIO(b"fake-image-bytes"),
        stream_info=StreamInfo(extension=".png", mimetype="image/png"),
    )

    assert result.text == "PADDLE_TEXT"
    assert result.backend_used == "paddleocr_vl"
    assert result.metadata["selected_mode"] == "local"
    assert result.metadata["selected_device"] == "cpu"
    assert result.metadata["quality"] == "medium"
    assert result.metadata["probe_candidates"] == ["cpu"]
