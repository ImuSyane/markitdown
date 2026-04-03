"""
OCR service layer for MarkItDown.

Supports built-in OpenAI-compatible OCR, explicit client/model configuration,
and user-provided OCR services/callables.
"""

import base64
import os
from dataclasses import dataclass
from typing import Any, BinaryIO, Callable, Protocol, runtime_checkable

from markitdown import StreamInfo


@dataclass
class OCRResult:
    """Result from OCR extraction."""

    text: str
    confidence: float | None = None
    backend_used: str | None = None
    error: str | None = None


@runtime_checkable
class OCRService(Protocol):
    """Minimal interface for OCR providers used by the converters."""

    def extract_text(
        self,
        image_stream: BinaryIO,
        prompt: str | None = None,
        stream_info: StreamInfo | None = None,
        **kwargs: Any,
    ) -> OCRResult: ...


class CallableOCRService:
    """Adapt a user callable to the OCRService protocol.

    The callable should accept an image stream plus optional keyword arguments
    such as prompt/stream_info, and may return OCRResult, str, or None.
    """

    def __init__(self, func: Callable[..., OCRResult | str | None]) -> None:
        self._func = func

    def extract_text(
        self,
        image_stream: BinaryIO,
        prompt: str | None = None,
        stream_info: StreamInfo | None = None,
        **kwargs: Any,
    ) -> OCRResult:
        result = self._func(
            image_stream,
            prompt=prompt,
            stream_info=stream_info,
            **kwargs,
        )
        if isinstance(result, OCRResult):
            return result
        if result is None:
            return OCRResult(text="", backend_used="custom_callable")
        return OCRResult(
            text=str(result).strip(),
            backend_used="custom_callable",
        )


class LLMVisionOCRService:
    """OCR service using LLM vision models (OpenAI-compatible)."""

    def __init__(
        self,
        client: Any,
        model: str,
        default_prompt: str | None = None,
    ) -> None:
        """
        Initialize LLM Vision OCR service.

        Args:
            client: OpenAI-compatible client
            model: Model name (e.g., 'gpt-4o', 'gemini-2.0-flash')
            default_prompt: Default prompt for OCR extraction
        """
        self.client = client
        self.model = model
        self.backend_name = "llm_vision"
        self.default_prompt = default_prompt or (
            "Extract all text from this image. "
            "Return ONLY the extracted text, maintaining the original "
            "layout and order. Do not add any commentary or description."
        )

    def extract_text(
        self,
        image_stream: BinaryIO,
        prompt: str | None = None,
        stream_info: StreamInfo | None = None,
        **kwargs: Any,
    ) -> OCRResult:
        """Extract text using LLM vision."""
        if self.client is None:
            return OCRResult(
                text="",
                backend_used=self.backend_name,
                error="LLM client not configured",
            )

        try:
            image_stream.seek(0)

            content_type: str | None = None
            if stream_info:
                content_type = stream_info.mimetype

            if not content_type:
                try:
                    from PIL import Image

                    image_stream.seek(0)
                    img = Image.open(image_stream)
                    fmt = img.format.lower() if img.format else "png"
                    content_type = f"image/{fmt}"
                except Exception:
                    content_type = "image/png"

            image_stream.seek(0)
            base64_image = base64.b64encode(image_stream.read()).decode("utf-8")
            data_uri = f"data:{content_type};base64,{base64_image}"

            actual_prompt = prompt or self.default_prompt
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": actual_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_uri},
                            },
                        ],
                    }
                ],
            )

            text = _extract_response_text(response)
            return OCRResult(
                text=text.strip() if text else "",
                backend_used=self.backend_name,
            )
        except Exception as e:
            return OCRResult(text="", backend_used=self.backend_name, error=str(e))
        finally:
            image_stream.seek(0)


class OpenAICompatibleOCRService(LLMVisionOCRService):
    """OCR service for remote providers exposing OpenAI-compatible APIs."""

    def __init__(
        self,
        client: Any,
        model: str,
        default_prompt: str | None = None,
    ) -> None:
        """Initialize a dedicated OpenAI-compatible OCR backend.

        This differs from LLMVisionOCRService only in backend labeling so
        callers can distinguish explicit remote OCR-provider usage.
        """
        super().__init__(client=client, model=model, default_prompt=default_prompt)
        self.backend_name = "openai_compatible"


def create_ocr_service(**kwargs: Any) -> OCRService | None:
    """Build an OCR service from kwargs/environment while preserving compatibility.

    Precedence is:
    1. explicit ``ocr_service`` object/callable
    2. explicit ``ocr_client`` with ``ocr_model``
    3. legacy ``llm_client`` with ``llm_model`` when no remote OCR backend is selected
    4. ``ocr_backend="openai_compatible"`` with OpenAI-compatible client settings

    Returns an OCRService instance when configuration is sufficient, else None.
    """
    explicit_service = kwargs.get("ocr_service")
    if explicit_service is not None:
        if hasattr(explicit_service, "extract_text"):
            return explicit_service
        if callable(explicit_service):
            return CallableOCRService(explicit_service)
        raise TypeError("ocr_service must define extract_text(...) or be callable")

    ocr_backend = _normalize_backend_name(
        kwargs.get("ocr_backend"),
        os.getenv("MARKITDOWN_OCR_BACKEND"),
    )
    ocr_client = kwargs.get("ocr_client")
    ocr_model = _first_non_empty(
        kwargs.get("ocr_model"),
        kwargs.get("llm_model"),
        os.getenv("MARKITDOWN_OCR_MODEL"),
    )
    ocr_prompt = _first_non_empty(
        kwargs.get("ocr_prompt"),
        kwargs.get("llm_prompt"),
        os.getenv("MARKITDOWN_OCR_PROMPT"),
    )

    if ocr_client is not None and ocr_model:
        return LLMVisionOCRService(
            client=ocr_client,
            model=ocr_model,
            default_prompt=ocr_prompt,
        )

    legacy_client = kwargs.get("llm_client")
    if ocr_backend in (None, "llm_vision") and legacy_client and ocr_model:
        return LLMVisionOCRService(
            client=legacy_client,
            model=ocr_model,
            default_prompt=ocr_prompt,
        )

    if ocr_backend != "openai_compatible":
        return None

    if not ocr_model:
        return None

    openai_client = _build_openai_compatible_client(**kwargs)
    if openai_client is None:
        return None

    return OpenAICompatibleOCRService(
        client=openai_client,
        model=ocr_model,
        default_prompt=ocr_prompt,
    )


def _build_openai_compatible_client(**kwargs: Any) -> Any | None:
    """Create an OpenAI client from OCR kwargs/environment.

    Supports ``ocr_base_url``, ``ocr_api_base``, ``ocr_api_key``, and optional
    ``ocr_client_options``. Returns None if the OpenAI SDK is unavailable.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return None

    base_url = _first_non_empty(
        kwargs.get("ocr_base_url"),
        kwargs.get("ocr_api_base"),
        os.getenv("MARKITDOWN_OCR_BASE_URL"),
    )
    api_key = _first_non_empty(
        kwargs.get("ocr_api_key"),
        os.getenv("MARKITDOWN_OCR_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
    )

    client_options = (kwargs.get("ocr_client_options") or {}).copy()
    if api_key:
        client_options["api_key"] = api_key
    if base_url:
        client_options["base_url"] = base_url

    try:
        return OpenAI(**client_options)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize OpenAI-compatible OCR client: {exc}"
        ) from exc


def _extract_response_text(response: Any) -> str:
    """Extract text from common OpenAI-compatible response payload shapes."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content or "")


def _first_non_empty(*values: Any) -> str | None:
    """Return the first non-empty value, stripping strings and stringifying others."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
            continue
        return str(value)
    return None


def _normalize_backend_name(*values: Any) -> str | None:
    """Normalize backend names to lowercase underscore form."""
    value = _first_non_empty(*values)
    if value is None:
        return None
    normalized = value.strip().lower().replace("-", "_")
    return normalized or None
