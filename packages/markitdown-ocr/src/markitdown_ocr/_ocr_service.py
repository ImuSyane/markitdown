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


class PaddleOCRService:
    """Traditional local OCR backend powered by the ``paddleocr`` package."""

    def __init__(
        self,
        engine: Any | None = None,
        *,
        lang: str | None = None,
        use_angle_cls: bool = True,
        **engine_options: Any,
    ) -> None:
        self._engine = engine
        self.lang = lang or "ch"
        self.use_angle_cls = use_angle_cls
        self.engine_options = engine_options
        self.backend_name = "paddleocr"

    def extract_text(
        self,
        image_stream: BinaryIO,
        prompt: str | None = None,
        stream_info: StreamInfo | None = None,
        **kwargs: Any,
    ) -> OCRResult:
        """Extract text with the local PaddleOCR detection/recognition pipeline."""
        try:
            image = _load_image_for_local_backends(image_stream)
            result = self._get_engine().ocr(image, cls=self.use_angle_cls)
            text, confidence = _parse_paddleocr_output(result)
            return OCRResult(
                text=text,
                confidence=confidence,
                backend_used=self.backend_name,
            )
        except Exception as exc:
            return OCRResult(
                text="",
                backend_used=self.backend_name,
                error=str(exc),
            )
        finally:
            image_stream.seek(0)

    def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "paddleocr backend requires the `paddleocr` package to be installed"
            ) from exc

        self._engine = PaddleOCR(
            lang=self.lang,
            use_angle_cls=self.use_angle_cls,
            **self.engine_options,
        )
        return self._engine


class LocalVLMOCRService:
    """Local VLM OCR backend for multimodal models such as PaddleOCR-VL."""

    def __init__(
        self,
        pipeline: Any | None = None,
        *,
        model: str | None = None,
        device: str | None = None,
        default_prompt: str | None = None,
        task: str = "image-text-to-text",
        call_mode: str | None = None,
        **pipeline_options: Any,
    ) -> None:
        self._pipeline = pipeline
        self.model = model
        self.device = device
        self.task = task
        self.call_mode = call_mode
        self.pipeline_options = pipeline_options
        self.backend_name = "local_vlm"
        self.default_prompt = default_prompt or (
            "Extract all text from this image. "
            "Return only the recognized text in reading order."
        )

    def extract_text(
        self,
        image_stream: BinaryIO,
        prompt: str | None = None,
        stream_info: StreamInfo | None = None,
        **kwargs: Any,
    ) -> OCRResult:
        """Extract text using a local multimodal generation pipeline."""
        try:
            image = _load_pil_image(image_stream)
            response = self._invoke_pipeline(
                image=image,
                prompt=prompt or self.default_prompt,
                **kwargs,
            )
            text = _extract_generated_text(response)
            return OCRResult(
                text=text.strip(),
                backend_used=self.backend_name,
            )
        except Exception as exc:
            return OCRResult(
                text="",
                backend_used=self.backend_name,
                error=str(exc),
            )
        finally:
            image_stream.seek(0)

    def _invoke_pipeline(self, *, image: Any, prompt: str, **kwargs: Any) -> Any:
        pipeline = self._get_pipeline()
        if self.call_mode is not None:
            return _invoke_vlm_pipeline(
                pipeline=pipeline,
                call_mode=self.call_mode,
                image=image,
                prompt=prompt,
                **kwargs,
            )

        attempts = [
            ("image_prompt", lambda: pipeline(image, prompt=prompt, **kwargs)),
            ("images_prompt", lambda: pipeline(images=image, prompt=prompt, **kwargs)),
            ("named_image_prompt", lambda: pipeline(image=image, prompt=prompt, **kwargs)),
            ("image_text", lambda: pipeline(image, text=prompt, **kwargs)),
            ("images_text", lambda: pipeline(images=image, text=prompt, **kwargs)),
            ("image_only", lambda: pipeline(image, **kwargs)),
        ]
        errors: list[str] = []
        for attempt_name, attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                errors.append(f"{attempt_name}: {exc}")
                continue
        raise RuntimeError(
            "Local VLM pipeline could not be invoked with any supported calling "
            f"pattern. Tried: {'; '.join(errors)}"
        )

    def _get_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        if not self.model:
            raise RuntimeError(
                "local_vlm backend requires either an `ocr_vlm_pipeline` object "
                "or an `ocr_model` identifier"
            )

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "local_vlm backend requires `transformers` or an explicit `ocr_vlm_pipeline`"
            ) from exc

        pipeline_options = self.pipeline_options.copy()
        if self.device is not None:
            pipeline_options.setdefault("device", self.device)
        self._pipeline = pipeline(
            self.task,
            model=self.model,
            **pipeline_options,
        )
        return self._pipeline


def create_ocr_service(**kwargs: Any) -> OCRService | None:
    """Build an OCR service from kwargs/environment while preserving compatibility.

    Precedence is:
    1. explicit ``ocr_service`` object/callable
    2. explicit ``ocr_client`` with ``ocr_model``
    3. legacy ``llm_client`` with ``llm_model`` when no remote OCR backend is selected
    4. ``ocr_backend="paddleocr"`` for classic local OCR
    5. ``ocr_backend="local_vlm"`` / ``paddleocr-vl-1.5`` for local multimodal OCR
    6. ``ocr_backend="openai_compatible"`` with OpenAI-compatible client settings

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

    if ocr_backend == "paddleocr":
        return _build_paddleocr_service(**kwargs)

    if ocr_backend in (
        "local_vlm",
        "vlm",
        "paddleocr_vl",
        "paddleocr_vl_1_5",
        "paddleocr_vl_15",
    ):
        vlm_kwargs = kwargs.copy()
        vlm_kwargs.pop("ocr_model", None)
        vlm_kwargs.pop("ocr_prompt", None)
        return _build_local_vlm_service(
            ocr_model=ocr_model,
            ocr_prompt=ocr_prompt,
            **vlm_kwargs,
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


def _build_paddleocr_service(**kwargs: Any) -> PaddleOCRService:
    """Create a local PaddleOCR service from kwargs/environment."""
    backend_options = (kwargs.get("ocr_backend_options") or {}).copy()
    env_use_angle_cls = _coerce_bool(
        os.getenv("MARKITDOWN_OCR_USE_ANGLE_CLS"),
        default=True,
    )
    return PaddleOCRService(
        engine=kwargs.get("ocr_paddle_engine"),
        lang=_first_non_empty(
            kwargs.get("ocr_lang"),
            os.getenv("MARKITDOWN_OCR_LANG"),
        ),
        use_angle_cls=_coerce_bool(
            kwargs.get("ocr_use_angle_cls"),
            default=env_use_angle_cls,
        ),
        **backend_options,
    )


def _build_local_vlm_service(
    *,
    ocr_model: str | None,
    ocr_prompt: str | None,
    **kwargs: Any,
) -> LocalVLMOCRService:
    """Create a local VLM OCR service from kwargs/environment."""
    backend_options = (kwargs.get("ocr_backend_options") or {}).copy()
    return LocalVLMOCRService(
        pipeline=kwargs.get("ocr_vlm_pipeline"),
        model=ocr_model,
        device=_first_non_empty(
            kwargs.get("ocr_device"),
            os.getenv("MARKITDOWN_OCR_DEVICE"),
        ),
        default_prompt=ocr_prompt,
        task=_first_non_empty(kwargs.get("ocr_task")) or "image-text-to-text",
        call_mode=_first_non_empty(
            kwargs.get("ocr_vlm_call_mode"),
            os.getenv("MARKITDOWN_OCR_VLM_CALL_MODE"),
        ),
        **backend_options,
    )


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


def _extract_generated_text(response: Any) -> str:
    """Extract text from common local VLM output payload shapes."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        for key in ("generated_text", "text", "content", "output"):
            value = response.get(key)
            if value:
                return _extract_generated_text(value)
        return ""
    if isinstance(response, list):
        parts = [_extract_generated_text(item) for item in response]
        return "\n".join([part for part in parts if part])
    text = getattr(response, "generated_text", None) or getattr(response, "text", None)
    if text:
        return _extract_generated_text(text)
    return str(response)


def _load_pil_image(image_stream: BinaryIO) -> Any:
    """Load a PIL image from a stream and normalize it to RGB."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Local OCR backends require Pillow to be installed") from exc

    image_stream.seek(0)
    image = Image.open(image_stream)
    return image.convert("RGB")


def _load_image_for_local_backends(image_stream: BinaryIO) -> Any:
    """Load an image into the array format local OCR libraries usually expect."""
    image = _load_pil_image(image_stream)
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Local OCR backends require numpy to be installed") from exc
    return np.array(image)


def _invoke_vlm_pipeline(
    *,
    pipeline: Any,
    call_mode: str,
    image: Any,
    prompt: str,
    **kwargs: Any,
) -> Any:
    """Invoke a VLM pipeline with an explicit calling convention."""
    if call_mode == "image_prompt":
        return pipeline(image, prompt=prompt, **kwargs)
    if call_mode == "images_prompt":
        return pipeline(images=image, prompt=prompt, **kwargs)
    if call_mode == "image_text":
        return pipeline(image, text=prompt, **kwargs)
    if call_mode == "images_text":
        return pipeline(images=image, text=prompt, **kwargs)
    if call_mode == "image_only":
        return pipeline(image, **kwargs)
    raise ValueError(f"Unsupported local VLM call mode: {call_mode}")


def _parse_paddleocr_output(result: Any) -> tuple[str, float | None]:
    """Parse PaddleOCR outputs into plain text and an average confidence score."""
    texts: list[str] = []
    scores: list[float] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            rec_texts = node.get("rec_texts")
            if isinstance(rec_texts, list):
                texts.extend([str(text).strip() for text in rec_texts if str(text).strip()])
            rec_scores = node.get("rec_scores")
            if isinstance(rec_scores, list):
                for score in rec_scores:
                    try:
                        scores.append(float(score))
                    except (TypeError, ValueError):
                        continue
            for value in node.values():
                visit(value)
            return

        if isinstance(node, (list, tuple)):
            if (
                len(node) == 2
                and isinstance(node[1], (list, tuple))
                and len(node[1]) > 0
                and isinstance(node[1][0], str)
            ):
                text_candidate = node[1][0]
                score_candidate = node[1][1] if len(node[1]) > 1 else None
                if text_candidate:
                    texts.append(str(text_candidate).strip())
                if score_candidate is not None:
                    try:
                        scores.append(float(score_candidate))
                    except (TypeError, ValueError):
                        pass
            for item in node:
                visit(item)

    visit(result)

    joined_text = "\n".join([text for text in texts if text])
    confidence = sum(scores) / len(scores) if scores else None
    return joined_text, confidence


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
    """Normalize backend names to lowercase underscore form.

    This lets users pass either hyphenated or dotted backend aliases such as
    ``paddleocr-vl-1.5`` while internal comparisons use a stable format.
    """
    value = _first_non_empty(*values)
    if value is None:
        return None
    normalized = value.strip().lower().replace("-", "_").replace(".", "_")
    return normalized or None


def _coerce_bool(value: Any, *, default: bool) -> bool:
    """Coerce common bool-like values while preserving a supplied default."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)
