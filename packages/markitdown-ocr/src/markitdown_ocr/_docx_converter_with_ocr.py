"""
Enhanced DOCX Converter with OCR support for embedded images.
Extracts images from Word documents and performs OCR while maintaining context.
"""

import io
import os
import re
import sys
from typing import Any, BinaryIO, Optional

from markitdown.converters import HtmlConverter
from markitdown.converter_utils.docx.pre_process import pre_process_docx
from markitdown import DocumentConverterResult, StreamInfo
from markitdown._exceptions import (
    MissingDependencyException,
    MISSING_DEPENDENCY_MESSAGE,
)
from ._image_export import create_image_asset, image_export_enabled, render_image_block
from ._ocr_service import OCRService

# Try loading dependencies
_dependency_exc_info = None
try:
    import mammoth
    from docx import Document
except ImportError:
    _dependency_exc_info = sys.exc_info()

# Placeholder injected into HTML so that mammoth never sees the OCR markers.
# Must be a single token with no special markdown characters.
_PLACEHOLDER = "MARKITDOWNOCRBLOCK{}"


class DocxConverterWithOCR(HtmlConverter):
    """
    Enhanced DOCX Converter with OCR support for embedded images.
    Maintains document flow while extracting text from images inline.
    """

    def __init__(self, ocr_service: Optional[OCRService] = None):
        super().__init__()
        self._html_converter = HtmlConverter()
        self.ocr_service = ocr_service

    def accepts(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> bool:
        mimetype = (stream_info.mimetype or "").lower()
        extension = (stream_info.extension or "").lower()

        if extension == ".docx":
            return True

        if mimetype.startswith(
            "application/vnd.openxmlformats-officedocument.wordprocessingml"
        ):
            return True

        return False

    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> DocumentConverterResult:
        if _dependency_exc_info is not None:
            raise MissingDependencyException(
                MISSING_DEPENDENCY_MESSAGE.format(
                    converter=type(self).__name__,
                    extension=".docx",
                    feature="docx",
                )
            ) from _dependency_exc_info[1].with_traceback(
                _dependency_exc_info[2]
            )  # type: ignore[union-attr]

        # Get OCR service if available (from kwargs or instance)
        ocr_service: Optional[OCRService] = (
            kwargs.get("ocr_service") or self.ocr_service
        )
        export_images = image_export_enabled(**kwargs)

        if ocr_service or export_images:
            # 1. Extract image blocks for inline insertion
            file_stream.seek(0)
            image_blocks, assets = self._extract_image_blocks(
                file_stream,
                ocr_service=ocr_service,
                image_dir=kwargs.get("image_dir"),
            )

            # 2. Convert DOCX → HTML via mammoth
            file_stream.seek(0)
            pre_process_stream = pre_process_docx(file_stream)
            html_result = mammoth.convert_to_html(
                pre_process_stream, style_map=kwargs.get("style_map")
            ).value

            # 3. Replace <img> tags with plain placeholder tokens so that
            #    mammoth's HTML→markdown step never escapes our OCR markers.
            html_with_placeholders, replacement_blocks = self._inject_placeholders(
                html_result, image_blocks
            )

            # 4. Convert HTML → markdown
            md_result = self._html_converter.convert_string(
                html_with_placeholders, **kwargs
            )
            md = md_result.markdown

            # 5. Swap placeholders for the actual OCR blocks (post-conversion
            #    so * and _ are never escaped by the markdown converter).
            for i, replacement in enumerate(replacement_blocks):
                placeholder = _PLACEHOLDER.format(i)
                md = md.replace(placeholder, replacement)

            return DocumentConverterResult(markdown=md, assets=assets)
        else:
            # Standard conversion without OCR
            style_map = kwargs.get("style_map", None)
            pre_process_stream = pre_process_docx(file_stream)
            return self._html_converter.convert_string(
                mammoth.convert_to_html(pre_process_stream, style_map=style_map).value,
                **kwargs,
            )

    def _extract_image_blocks(
        self,
        file_stream: BinaryIO,
        *,
        ocr_service: Optional[OCRService],
        image_dir: Optional[str],
    ) -> tuple[list[str], list]:
        """
        Extract images from DOCX and build replacement markdown blocks.

        Returns:
            Ordered replacement blocks and exported assets.
        """
        blocks = []
        assets = []

        try:
            file_stream.seek(0)
            doc = Document(file_stream)

            for index, rel in enumerate(doc.part.rels.values(), start=1):
                if "image" in rel.target_ref.lower():
                    try:
                        image_bytes = rel.target_part.blob
                        asset = None
                        if image_dir:
                            asset = create_image_asset(
                                image_bytes=image_bytes,
                                image_dir=image_dir,
                                name=f"docx_image_{index}",
                                extension=os.path.splitext(rel.target_ref)[1],
                            )
                            assets.append(asset)

                        ocr_text = ""
                        if ocr_service:
                            image_stream = io.BytesIO(image_bytes)
                            ocr_result = ocr_service.extract_text(image_stream)
                            ocr_text = ocr_result.text.strip()

                        if asset is not None:
                            blocks.append(
                                render_image_block(
                                    asset,
                                    alt_text=f"docx image {index}",
                                    extracted_text=ocr_text,
                                )
                            )
                        elif ocr_text:
                            blocks.append(f"*[Image OCR]\n{ocr_text}\n[End OCR]*")

                    except Exception:
                        continue

        except Exception:
            pass

        return blocks, assets

    def _inject_placeholders(
        self, html: str, replacement_blocks: list[str]
    ) -> tuple[str, list[str]]:
        """
        Replace <img> tags with numbered placeholder tokens.

        Returns:
            (html_with_placeholders, ordered list of raw OCR texts)
        """
        if not replacement_blocks:
            return html, []

        used: list[int] = []

        def replace_img(match: re.Match) -> str:  # type: ignore[type-arg]
            for i in range(len(replacement_blocks)):
                if i not in used:
                    used.append(i)
                    return f"<p>{_PLACEHOLDER.format(i)}</p>"
            return ""  # remove image if all OCR texts already used

        result = re.sub(r"<img[^>]*>", replace_img, html)

        # Any OCR texts that had no matching <img> tag go at the end
        for i in range(len(replacement_blocks)):
            if i not in used:
                result += f"<p>{_PLACEHOLDER.format(i)}</p>"

        return result, replacement_blocks
