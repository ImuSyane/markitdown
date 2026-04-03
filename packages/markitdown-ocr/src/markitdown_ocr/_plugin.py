"""
Plugin registration for markitdown-ocr.
Registers OCR-enhanced converters with priority-based replacement strategy.
"""

from typing import Any
from markitdown import MarkItDown

from ._ocr_service import OCRService, create_ocr_service
from ._pdf_converter_with_ocr import PdfConverterWithOCR
from ._docx_converter_with_ocr import DocxConverterWithOCR
from ._pptx_converter_with_ocr import PptxConverterWithOCR
from ._xlsx_converter_with_ocr import XlsxConverterWithOCR


__plugin_interface_version__ = 1


def register_converters(markitdown: MarkItDown, **kwargs: Any) -> None:
    """
    Register OCR-enhanced converters with MarkItDown.

    This plugin provides OCR support for PDF, DOCX, PPTX, and XLSX files.
    The converters are registered with priority -1.0 to run BEFORE built-in
    converters (which have priority 0.0), effectively replacing them when
    the plugin is enabled.

    Args:
        markitdown: MarkItDown instance to register converters with
        **kwargs: Additional keyword arguments that may include:
            - ocr_service: Explicit OCR service object or callable
            - ocr_client / ocr_model / ocr_prompt: explicit vision OCR client config
            - ocr_backend=openai_compatible plus ocr_base_url / ocr_api_key / ocr_model
            - llm_client / llm_model / llm_prompt: legacy OCR configuration
    """
    ocr_service: OCRService | None = create_ocr_service(**kwargs)

    # Register converters with priority -1.0 (before built-ins at 0.0)
    # This effectively "replaces" the built-in converters when plugin is installed
    # Pass the OCR service to each converter's constructor
    PRIORITY_OCR_ENHANCED = -1.0

    markitdown.register_converter(
        PdfConverterWithOCR(ocr_service=ocr_service), priority=PRIORITY_OCR_ENHANCED
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
