import io
from typing import Any, BinaryIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from markitdown_ocr._ocr_service import (
    CallableOCRService,
    LocalVLMOCRService,
    OCRResult,
    OpenAICompatibleOCRService,
    PaddleOCRService,
    create_ocr_service,
)


def test_create_ocr_service_returns_explicit_service() -> None:
    service = MagicMock()
    service.extract_text.return_value = OCRResult(text="hello", backend_used="custom")

    result = create_ocr_service(ocr_service=service)

    assert result is service


def test_create_ocr_service_wraps_callable() -> None:
    def custom_ocr(
        image_stream: BinaryIO, **kwargs: Any
    ) -> str:  # noqa: ARG001
        return "callable text"

    service = create_ocr_service(ocr_service=custom_ocr)

    assert isinstance(service, CallableOCRService)
    result = service.extract_text(io.BytesIO(b"fake image"))
    assert result.text == "callable text"
    assert result.backend_used == "custom_callable"


def test_create_ocr_service_preserves_legacy_llm_configuration() -> None:
    client = MagicMock()
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="legacy text"))]
    )
    client.chat.completions.create.return_value = response

    service = create_ocr_service(llm_client=client, llm_model="gpt-4o-mini")

    assert service is not None
    result = service.extract_text(io.BytesIO(b"fake image"))
    assert result.text == "legacy text"
    assert result.backend_used == "llm_vision"


def test_create_ocr_service_builds_openai_compatible_client() -> None:
    fake_client = MagicMock()

    with patch("openai.OpenAI", return_value=fake_client) as mock_openai:
        service = create_ocr_service(
            ocr_backend="openai_compatible",
            ocr_model="glm-ocr",
            ocr_base_url="https://example.test/v1",
            ocr_api_key="secret",
        )

    assert isinstance(service, OpenAICompatibleOCRService)
    mock_openai.assert_called_once_with(
        api_key="secret",
        base_url="https://example.test/v1",
    )


def test_openai_compatible_service_handles_list_content() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[
                        {"type": "output_text", "text": "line 1"},
                        SimpleNamespace(text="line 2"),
                    ]
                )
            )
        ]
    )

    service = OpenAICompatibleOCRService(client=client, model="glm-ocr")

    result = service.extract_text(io.BytesIO(b"fake image"))

    assert result.text == "line 1\nline 2"
    assert result.backend_used == "openai_compatible"


def test_create_ocr_service_builds_paddleocr_service() -> None:
    engine = MagicMock()
    engine.ocr.return_value = [
        [
            ([[0, 0], [1, 0], [1, 1], [0, 1]], ("第一行", 0.9)),
            ([[0, 1], [1, 1], [1, 2], [0, 2]], ("second line", 0.7)),
        ]
    ]

    service = create_ocr_service(
        ocr_backend="paddleocr",
        ocr_paddle_engine=engine,
        ocr_lang="en",
    )

    assert isinstance(service, PaddleOCRService)
    result = service.extract_text(_png_stream())

    assert result.text == "第一行\nsecond line"
    assert result.backend_used == "paddleocr"
    assert result.confidence == 0.8


def test_create_ocr_service_builds_local_vlm_service_for_paddleocr_vl_alias() -> None:
    pipeline = MagicMock(return_value=[{"generated_text": "vlm text"}])

    service = create_ocr_service(
        ocr_backend="paddleocr-vl-1.5",
        ocr_vlm_pipeline=pipeline,
        ocr_model="paddleocr-vl-1.5",
    )

    assert isinstance(service, LocalVLMOCRService)
    result = service.extract_text(_png_stream())

    assert result.text == "vlm text"
    assert result.backend_used == "local_vlm"


def test_local_vlm_service_uses_prompt_argument() -> None:
    captured: dict[str, Any] = {}

    def pipeline(image: Any, prompt: str, **kwargs: Any) -> list[dict[str, str]]:
        captured["prompt"] = prompt
        return [{"generated_text": "recognized"}]

    service = LocalVLMOCRService(pipeline=pipeline, model="dummy")
    result = service.extract_text(_png_stream(), prompt="read tables")

    assert result.text == "recognized"
    assert captured["prompt"] == "read tables"


def _png_stream() -> io.BytesIO:
    return io.BytesIO(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
        b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
    )
