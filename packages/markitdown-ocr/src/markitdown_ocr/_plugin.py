"""
Plugin registration for markitdown-ocr.
Registers OCR-enhanced converters with priority-based replacement strategy.
"""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from markitdown import MarkItDown

from ._ocr_service import (
    OCRBackend,
    OpenAICompatibleVisionOCRBackend,
    PaddleOCROCRBackend,
    PaddleOCRVLOCRBackend,
)
from ._pdf_converter_with_ocr import PdfConverterWithOCR
from ._docx_converter_with_ocr import DocxConverterWithOCR
from ._pptx_converter_with_ocr import PptxConverterWithOCR
from ._xlsx_converter_with_ocr import XlsxConverterWithOCR


__plugin_interface_version__ = 1


def register_converters(markitdown: "MarkItDown", **kwargs: Any) -> None:
    """
    Register OCR-enhanced converters with MarkItDown.

    This plugin provides OCR support for PDF, DOCX, PPTX, and XLSX files.
    The converters are registered with priority -1.0 to run BEFORE built-in
    converters (which have priority 0.0), effectively replacing them when
    the plugin is enabled.

    Args:
        markitdown: MarkItDown instance to register converters with
        **kwargs: Additional keyword arguments that may include:
            - ocr_backend: OCR backend selector ('paddleocr', 'openai_compatible', or 'paddleocr_vl')
            - ocr_mode: Paddle OCR mode ('local' or 'server')
            - ocr_server_url: optional PaddleOCR-VL server URL
            - ocr_server_backend: optional PaddleOCR-VL server backend transport
            - ocr_api_key / ocr_server_api_key: optional PaddleOCR-VL server API key
            - ocr_quality: OCR quality profile ('low', 'medium', 'high')
            - pdf_layout_backend: PDF layout prepass selector ('auto', 'none', 'docling')
            - pdf_layout_min_area_ratio: minimum embedded-image area ratio for layout prepass
            - pdf_layout_debug: warn on PDF layout prepass fallback reasons
            - ocr_artifact_export: enable PDF crop/full-page artifact export
            - ocr_artifact_dir: optional output directory for exported PDF artifacts
            - ocr_artifact_markdown_mode: markdown rendering mode for exported PDF artifacts
            - llm_client / llm_model: OpenAI-compatible OCR backend inputs and PPTX captioning inputs
            - llm_prompt: Custom prompt for OpenAI-compatible OCR text extraction
    """
    llm_client = kwargs.get("llm_client")
    llm_model = kwargs.get("llm_model")
    llm_prompt = kwargs.get("llm_prompt")
    ocr_debug = bool(kwargs.get("ocr_debug", False))
    ocr_backend_name = kwargs.get("ocr_backend")
    ocr_quality = kwargs.get("ocr_quality", "medium")
    ocr_backend_options = kwargs.get("ocr_backend_options") or {}

    ocr_service: OCRBackend | None = None
    if ocr_backend_name in {None, "paddleocr", "paddleocr_ppocr"}:
        ocr_service = PaddleOCROCRBackend(
            text_detection_model_name=kwargs.get(
                "ocr_text_detection_model_name", "PP-OCRv5_mobile_det"
            ),
            text_recognition_model_name=kwargs.get(
                "ocr_text_recognition_model_name", "PP-OCRv5_server_rec"
            ),
            backend_options=ocr_backend_options,
            quality=ocr_quality,
            debug=ocr_debug,
        )
    elif ocr_backend_name == "paddleocr_vl":
        ocr_mode = kwargs.get("ocr_mode")
        legacy_device = kwargs.get("ocr_device")
        if ocr_mode is None and kwargs.get("ocr_server_url"):
            ocr_mode = "server"
        if ocr_mode is None:
            ocr_mode = "local"

        ocr_service = PaddleOCRVLOCRBackend(
            mode=ocr_mode,
            server_url=kwargs.get("ocr_server_url"),
            server_backend=kwargs.get("ocr_server_backend"),
            api_key=kwargs.get("ocr_api_key") or kwargs.get("ocr_server_api_key"),
            api_model_name=kwargs.get(
                "ocr_api_model_name", "PaddlePaddle/PaddleOCR-VL-1.5"
            ),
            backend_options=ocr_backend_options,
            quality=ocr_quality,
            debug=ocr_debug,
            legacy_device=legacy_device,
        )
    elif ocr_backend_name in {"openai_compatible", "openai_vision"}:
        if llm_client and llm_model:
            ocr_service = OpenAICompatibleVisionOCRBackend(
                client=llm_client,
                model=llm_model,
                default_prompt=llm_prompt,
                quality=ocr_quality,
                debug=ocr_debug,
            )

    PRIORITY_OCR_ENHANCED = -1.0

    markitdown.register_converter(
        PdfConverterWithOCR(
            ocr_service=ocr_service,
            pdf_layout_backend=kwargs.get("pdf_layout_backend", "auto"),
            pdf_layout_min_area_ratio=kwargs.get("pdf_layout_min_area_ratio", 0.20),
            pdf_layout_debug=bool(kwargs.get("pdf_layout_debug", False)),
            ocr_artifact_export=bool(kwargs.get("ocr_artifact_export", True)),
            ocr_artifact_dir=kwargs.get("ocr_artifact_dir"),
            ocr_artifact_markdown_mode=kwargs.get(
                "ocr_artifact_markdown_mode", "image_and_text"
            ),
        ),
        priority=PRIORITY_OCR_ENHANCED,
    )

    markitdown.register_converter(
        DocxConverterWithOCR(ocr_service=ocr_service), priority=PRIORITY_OCR_ENHANCED
    )

    markitdown.register_converter(
        PptxConverterWithOCR(ocr_service=ocr_service), priority=PRIORITY_OCR_ENHANCED
    )

    markitdown.register_converter(
        XlsxConverterWithOCR(ocr_service=ocr_service), priority=PRIORITY_OCR_ENHANCED
    )
