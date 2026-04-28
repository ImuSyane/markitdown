"""
Enhanced PDF converter for markitdown-ocr.

The converter keeps the official MarkItDown core untouched and adds PDF-only
layout/crop/artifact behavior inside the plugin:

- scanned pages: render page -> optional layout prepass -> OCR per region
- mixed PDFs: preserve native text extraction, OCR embedded images inline
- image_like / complex_like regions: export a cropped image artifact and emit
  image markdown plus OCR text
- layout failures: fall back to the prior full-page or single-image OCR path
"""

from __future__ import annotations

import io
import os
import re
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal

from markitdown import DocumentConverter, DocumentConverterResult, StreamInfo
from markitdown._exceptions import (
    MISSING_DEPENDENCY_MESSAGE,
    MissingDependencyException,
)

from ._ocr_service import OCRBackend
from ._pdf_layout import (
    PDFLayoutAnalyzer,
    PDFLayoutBackend,
    PDFLayoutCategory,
    PDFLayoutRegion,
    resolve_pdf_layout_analyzer,
)

_dependency_exc_info = None
try:
    import pdfminer.high_level
    import pdfplumber
    from PIL import Image
except ImportError:
    _dependency_exc_info = sys.exc_info()


PDFArtifactMarkdownMode = Literal["image_and_text"]
PDFVisualSource = Literal["docling", "ocr_placeholder"]
_OCR_IMAGE_PLACEHOLDER_RE = re.compile(
    r"<div[^>]*>\s*<img[^>]+src=[\"']imgs/[^\"']+[\"'][^>]*>\s*</div>",
    re.IGNORECASE,
)
_OCR_MARKDOWN_IMAGE_PLACEHOLDER_RE = re.compile(
    r"!\[[^\]]*\]\(\s*imgs/[^)]+\)",
    re.IGNORECASE,
)
_OCR_HTML_IMAGE_RE = re.compile(
    r"<img[^>]+src=[\"']imgs/[^\"']+[\"'][^>]*>",
    re.IGNORECASE,
)
_HTML_DIV_RE = re.compile(
    r"<div\b[^>]*>\s*(?P<content>.*?)\s*</div>",
    re.IGNORECASE | re.DOTALL,
)
_OCR_IMAGE_PLACEHOLDER_BOX_RE = re.compile(
    r"<div[^>]*>\s*<img[^>]+src=[\"']imgs/"
    r"img_in_image_box_(?P<x0>\d+)_(?P<y0>\d+)_(?P<x1>\d+)_(?P<y1>\d+)"
    r"\.[^\"']+[\"'][^>]*>\s*</div>",
    re.IGNORECASE,
)
_INLINE_LATEX_RE = re.compile(r"(?<!\$)\$\s*([^$\n]*?\S)\s*\$(?!\$)")


@dataclass(slots=True)
class _ArtifactContext:
    root_dir: Path
    markdown_dir: Path
    export_enabled: bool
    markdown_mode: PDFArtifactMarkdownMode


@dataclass(slots=True)
class _ContentItem:
    y_pos: float
    order: int
    markdown: str
    bbox: tuple[float, float, float, float] | None = None


@dataclass(slots=True)
class _VisualCandidate:
    y_pos: float
    order: int
    image_stream: io.BytesIO
    bbox: tuple[float, float, float, float] | None
    source: PDFVisualSource
    category: PDFLayoutCategory
    confidence: float | None = None


def _copy_stream(stream: BinaryIO) -> io.BytesIO:
    stream.seek(0)
    data = stream.read()
    stream.seek(0)
    copied = io.BytesIO(data)
    copied.seek(0)
    return copied


def _stream_to_png(stream: BinaryIO) -> io.BytesIO:
    copied = _copy_stream(stream)
    image = Image.open(copied)
    if image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")
    output = io.BytesIO()
    image.save(output, format="PNG")
    output.seek(0)
    return output


def _normalize_inline_latex(text: str) -> str:
    """Keep inline math tight for Markdown renderers: `$ x $` -> `$x$`."""

    return _INLINE_LATEX_RE.sub(lambda match: f"${match.group(1).strip()}$", text)


def _image_size(stream: BinaryIO) -> tuple[int, int] | None:
    try:
        copied = _copy_stream(stream)
        with Image.open(copied) as image:
            return image.size
    except Exception:
        return None


def _safe_bbox(
    x0: float | None,
    y0: float | None,
    x1: float | None,
    y1: float | None,
) -> tuple[float, float, float, float] | None:
    try:
        left = float(x0 or 0.0)
        top = float(y0 or 0.0)
        right = float(x1 or 0.0)
        bottom = float(y1 or 0.0)
    except Exception:
        return None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _extract_images_from_page(page: Any) -> list[dict[str, Any]]:
    """
    Extract embedded images from a PDF page.

    The return value keeps enough metadata for ordering and large-image routing.
    """

    images_info: list[dict[str, Any]] = []
    page_area = max(float(getattr(page, "width", 0.0) or 0.0), 1.0) * max(
        float(getattr(page, "height", 0.0) or 0.0), 1.0
    )

    try:
        images: list[dict[str, Any]] = []
        if hasattr(page, "images") and page.images:
            images = list(page.images)
        elif hasattr(page, "objects") and "image" in page.objects:
            images = list(page.objects.get("image", []))
        elif hasattr(page, "objects"):
            for obj_type, obj_values in page.objects.items():
                if "image" in obj_type.lower() or "xobject" in obj_type.lower():
                    if obj_values:
                        images = list(obj_values)
                        break

        for index, img_dict in enumerate(images):
            try:
                bbox = _safe_bbox(
                    img_dict.get("x0"),
                    img_dict.get("top"),
                    img_dict.get("x1"),
                    img_dict.get("bottom"),
                )
                if bbox is None:
                    continue

                y_pos = bbox[1]
                image_stream: io.BytesIO | None = None

                if "stream" in img_dict and hasattr(img_dict["stream"], "get_data"):
                    try:
                        image_stream = _stream_to_png(io.BytesIO(img_dict["stream"].get_data()))
                    except Exception:
                        image_stream = None

                if image_stream is None:
                    cropped_page = page.within_bbox(bbox)
                    page_image = cropped_page.to_image(resolution=200)
                    image_stream = io.BytesIO()
                    page_image.original.save(image_stream, format="PNG")
                    image_stream.seek(0)

                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area_ratio = (width * height) / page_area if page_area > 0 else 0.0

                images_info.append(
                    {
                        "stream": image_stream,
                        "bbox": bbox,
                        "name": f"page_{getattr(page, 'page_number', 1)}_img_{index}",
                        "y_pos": y_pos,
                        "area_ratio": area_ratio,
                    }
                )
            except Exception:
                continue
    except Exception:
        pass

    images_info.sort(key=lambda item: float(item.get("y_pos", 0.0)))
    return images_info


class PdfConverterWithOCR(DocumentConverter):
    """PDF converter with OCR, layout prepass, and artifact export support."""

    def __init__(
        self,
        *,
        ocr_service: OCRBackend | None = None,
        pdf_layout_backend: PDFLayoutBackend = "auto",
        pdf_layout_min_area_ratio: float = 0.20,
        pdf_layout_debug: bool = False,
        pdf_layout_analyzer: PDFLayoutAnalyzer | None = None,
        ocr_artifact_export: bool = True,
        ocr_artifact_dir: str | Path | None = None,
        ocr_artifact_markdown_mode: PDFArtifactMarkdownMode = "image_and_text",
    ) -> None:
        super().__init__()
        self.ocr_service = ocr_service
        self.pdf_layout_backend = (
            pdf_layout_backend if pdf_layout_backend in {"auto", "none", "docling"} else "auto"
        )
        self.pdf_layout_min_area_ratio = max(float(pdf_layout_min_area_ratio), 0.0)
        self.pdf_layout_debug = bool(pdf_layout_debug)
        self.ocr_artifact_export = bool(ocr_artifact_export)
        self.ocr_artifact_dir = Path(ocr_artifact_dir).expanduser() if ocr_artifact_dir else None
        self.ocr_artifact_markdown_mode = ocr_artifact_markdown_mode

        if pdf_layout_analyzer is not None:
            self.pdf_layout_analyzer = pdf_layout_analyzer
            self.resolved_pdf_layout_backend = getattr(
                pdf_layout_analyzer, "backend_name", "docling"
            )
        else:
            analyzer, resolved = resolve_pdf_layout_analyzer(
                self.pdf_layout_backend,
                debug=self.pdf_layout_debug,
            )
            self.pdf_layout_analyzer = analyzer
            self.resolved_pdf_layout_backend = resolved

    def accepts(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> bool:
        mimetype = (stream_info.mimetype or "").lower()
        extension = (stream_info.extension or "").lower()
        return extension == ".pdf" or mimetype.startswith("application/pdf") or mimetype.startswith(
            "application/x-pdf"
        )

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
                    extension=".pdf",
                    feature="pdf",
                )
            ) from _dependency_exc_info[1].with_traceback(  # type: ignore[union-attr]
                _dependency_exc_info[2]
            )

        ocr_service: OCRBackend | None = kwargs.get("ocr_service") or self.ocr_service

        file_stream.seek(0)
        pdf_raw = file_stream.read()
        pdf_bytes = io.BytesIO(pdf_raw)
        artifact_context = self._build_artifact_context(stream_info)
        artifact_counters: dict[str, int] = {}

        markdown = ""
        try:
            with pdfplumber.open(io.BytesIO(pdf_raw)) as pdf:
                page_sections: list[str] = []
                for page_num, page in enumerate(pdf.pages, 1):
                    page_markdown = self._convert_page(
                        page,
                        page_num,
                        ocr_service,
                        stream_info=stream_info,
                        artifact_context=artifact_context,
                        artifact_counters=artifact_counters,
                    )
                    section = f"## Page {page_num}"
                    if page_markdown.strip():
                        section = f"{section}\n\n{page_markdown.strip()}"
                    page_sections.append(section)
                markdown = "\n\n".join(section for section in page_sections if section.strip()).strip()
        except Exception as exc:
            if self.pdf_layout_debug:
                self._warn(f"pdfplumber page walk failed, falling back: {exc}")
            try:
                markdown = pdfminer.high_level.extract_text(io.BytesIO(pdf_raw)).strip()
            except Exception:
                markdown = ""

        if ocr_service and (not markdown or not markdown.strip()):
            markdown = self._ocr_full_pages(
                pdf_bytes,
                ocr_service,
                stream_info=stream_info,
                artifact_context=artifact_context,
                artifact_counters=artifact_counters,
            )
        elif not markdown.strip():
            try:
                markdown = pdfminer.high_level.extract_text(io.BytesIO(pdf_raw)).strip()
            except Exception:
                markdown = ""

        return DocumentConverterResult(markdown=markdown.strip())

    def _build_artifact_context(self, stream_info: StreamInfo) -> _ArtifactContext:
        if self.ocr_artifact_dir is not None:
            root_dir = self.ocr_artifact_dir
        else:
            source_path = Path(stream_info.local_path).expanduser() if stream_info.local_path else None
            if source_path is not None:
                source_stem = source_path.stem or "document"
                root_dir = source_path.parent / "markitdown-ocr-artifacts" / source_stem
            else:
                temp_root = Path(tempfile.gettempdir()) / "markitdown-ocr-artifacts"
                root_dir = temp_root / "document"

        if self.ocr_artifact_export:
            root_dir.mkdir(parents=True, exist_ok=True)

        return _ArtifactContext(
            root_dir=root_dir,
            markdown_dir=root_dir.parent,
            export_enabled=self.ocr_artifact_export,
            markdown_mode=self.ocr_artifact_markdown_mode,
        )

    def _warn(self, message: str) -> None:
        warnings.warn(f"markitdown-ocr: {message}", RuntimeWarning, stacklevel=2)

    def _extract_text_items(self, page: Any) -> list[tuple[float, str]]:
        chars = getattr(page, "chars", None) or []
        if chars:
            lines: list[tuple[float, str]] = []
            current_line: list[dict[str, Any]] = []
            current_y: float | None = None

            for char in sorted(chars, key=lambda item: (item["top"], item["x0"])):
                y = float(char.get("top", 0.0))
                if current_y is None:
                    current_y = y
                elif abs(y - current_y) > 2.5:
                    text = "".join(str(item.get("text", "")) for item in current_line).strip()
                    if text:
                        lines.append((current_y, text))
                    current_line = []
                    current_y = y
                current_line.append(char)

            if current_line and current_y is not None:
                text = "".join(str(item.get("text", "")) for item in current_line).strip()
                if text:
                    lines.append((current_y, text))
            return lines

        text = (page.extract_text() or "").strip()
        if not text:
            return []
        return [(float(index * 12), line.strip()) for index, line in enumerate(text.splitlines()) if line.strip()]

    def _convert_page(
        self,
        page: Any,
        page_num: int,
        ocr_service: OCRBackend | None,
        *,
        stream_info: StreamInfo | None = None,
        artifact_context: _ArtifactContext | None = None,
        artifact_counters: dict[str, int] | None = None,
    ) -> str:
        text_items = self._extract_text_items(page)
        if ocr_service is None:
            return "\n\n".join(text for _y, text in text_items if text.strip())

        artifact_context = artifact_context or _ArtifactContext(
            root_dir=Path(tempfile.gettempdir()) / "markitdown-ocr-artifacts" / "document",
            markdown_dir=Path(tempfile.gettempdir()) / "markitdown-ocr-artifacts",
            export_enabled=False,
            markdown_mode="image_and_text",
        )
        if stream_info is not None and artifact_context.export_enabled is False and self.ocr_artifact_export:
            artifact_context = self._build_artifact_context(stream_info)
        artifact_counters = artifact_counters or {}

        if not text_items:
            rendered_page = self._render_page_to_stream(page)
            page_pdf = self._render_page_to_pdf_stream(page)
            return self._process_scanned_page(
                rendered_page,
                page_num,
                ocr_service,
                stream_info=stream_info,
                artifact_context=artifact_context,
                artifact_counters=artifact_counters,
                layout_stream=page_pdf,
            )

        content_items = [
            _ContentItem(y_pos=y_pos, order=index, markdown=text)
            for index, (y_pos, text) in enumerate(text_items)
            if text.strip()
        ]

        for image_index, image_info in enumerate(_extract_images_from_page(page), start=1):
            content_items.extend(
                self._image_content_items(
                    image_info=image_info,
                    page_num=page_num,
                    base_order=1000 + image_index * 20,
                    ocr_service=ocr_service,
                    stream_info=stream_info,
                    artifact_context=artifact_context,
                    artifact_counters=artifact_counters,
                )
            )

        content_items.sort(key=lambda item: (item.y_pos, item.order))
        return "\n\n".join(item.markdown for item in content_items if item.markdown.strip())

    def _image_content_items(
        self,
        *,
        image_info: dict[str, Any],
        page_num: int,
        base_order: int,
        ocr_service: OCRBackend,
        stream_info: StreamInfo | None,
        artifact_context: _ArtifactContext,
        artifact_counters: dict[str, int],
    ) -> list[_ContentItem]:
        image_stream = _copy_stream(image_info["stream"])
        image_name = str(image_info.get("name") or f"page_{page_num}_image")
        y_pos = float(image_info.get("y_pos", 0.0))
        area_ratio = float(image_info.get("area_ratio", 0.0))

        if (
            self.pdf_layout_analyzer is not None
            and area_ratio >= self.pdf_layout_min_area_ratio
        ):
            layout_items = self._layout_region_items(
                source_stream=image_stream,
                source_name=f"{image_name}.png",
                page_num=page_num,
                y_offset=y_pos,
                order_offset=base_order,
                ocr_service=ocr_service,
                stream_info=stream_info,
                artifact_context=artifact_context,
                artifact_counters=artifact_counters,
                fallback_category="image_like",
            )
            if layout_items:
                return layout_items

        fallback_markdown = self._build_image_and_text_block(
            image_stream=image_stream,
            page_num=page_num,
            ocr_service=ocr_service,
            stream_info=stream_info,
            artifact_context=artifact_context,
            artifact_counters=artifact_counters,
            artifact_label="image",
        )
        return [_ContentItem(y_pos=y_pos, order=base_order, markdown=fallback_markdown)] if fallback_markdown else []

    def _process_scanned_page(
        self,
        rendered_page: io.BytesIO,
        page_num: int,
        ocr_service: OCRBackend,
        *,
        stream_info: StreamInfo | None,
        artifact_context: _ArtifactContext,
        artifact_counters: dict[str, int],
        layout_stream: io.BytesIO | None = None,
    ) -> str:
        layout_regions: list[PDFLayoutRegion] = []

        if self.pdf_layout_analyzer is not None:
            layout_regions = self._analyze_layout_regions(
                source_stream=layout_stream or rendered_page,
                source_name=(
                    f"page_{page_num}.pdf"
                    if layout_stream is not None
                    else f"page_{page_num}.png"
                ),
            )

        docling_candidates = self._visual_candidates_from_layout_regions(
            regions=layout_regions,
            y_offset=0.0,
            order_offset=0,
            source_size=_image_size(rendered_page),
        )
        full_text = self._ocr_image_to_text(
            rendered_page,
            ocr_service,
            stream_info,
        )
        placeholder_candidates: list[_VisualCandidate] = []
        if not layout_regions:
            placeholder_candidates = self._ocr_placeholder_visual_candidates(
                full_text=full_text,
                rendered_page=rendered_page,
            )
        selected_candidates = self._select_visual_candidates(
            docling_candidates,
            placeholder_candidates,
        )

        if full_text:
            visual_items = self._save_visual_candidates(
                candidates=selected_candidates,
                page_num=page_num,
                artifact_context=artifact_context,
                artifact_counters=artifact_counters,
            )
            return self._merge_scanned_ocr_with_visuals(
                full_text,
                visual_items,
            )

        if layout_regions:
            layout_markdown = self._layout_region_items_from_regions(
                regions=layout_regions,
                page_num=page_num,
                y_offset=0.0,
                order_offset=0,
                ocr_service=ocr_service,
                stream_info=stream_info,
                artifact_context=artifact_context,
                artifact_counters=artifact_counters,
                fallback_category="complex_like",
                as_markdown=not bool(full_text),
                emit_text_regions=not bool(full_text),
                ocr_visual_regions=not bool(full_text),
            )
            if isinstance(layout_markdown, str) and layout_markdown.strip():
                return layout_markdown

        return self._build_image_and_text_block(
            image_stream=rendered_page,
            page_num=page_num,
            ocr_service=ocr_service,
            stream_info=stream_info,
            artifact_context=artifact_context,
            artifact_counters=artifact_counters,
            artifact_label="full",
        )

    def _layout_region_items(
        self,
        *,
        source_stream: io.BytesIO,
        source_name: str,
        page_num: int,
        y_offset: float,
        order_offset: int,
        ocr_service: OCRBackend,
        stream_info: StreamInfo | None,
        artifact_context: _ArtifactContext,
        artifact_counters: dict[str, int],
        fallback_category: PDFLayoutCategory,
        as_markdown: bool = False,
        emit_text_regions: bool = True,
        ocr_visual_regions: bool = True,
    ) -> list[_ContentItem] | str:
        if self.pdf_layout_analyzer is None:
            return "" if as_markdown else []

        regions = self._analyze_layout_regions(
            source_stream=source_stream,
            source_name=source_name,
        )

        if not regions:
            return "" if as_markdown else []

        return self._layout_region_items_from_regions(
            regions=regions,
            page_num=page_num,
            y_offset=y_offset,
            order_offset=order_offset,
            ocr_service=ocr_service,
            stream_info=stream_info,
            artifact_context=artifact_context,
            artifact_counters=artifact_counters,
            fallback_category=fallback_category,
            as_markdown=as_markdown,
            emit_text_regions=emit_text_regions,
            ocr_visual_regions=ocr_visual_regions,
        )

    def _analyze_layout_regions(
        self,
        *,
        source_stream: io.BytesIO,
        source_name: str,
    ) -> list[PDFLayoutRegion]:
        if self.pdf_layout_analyzer is None:
            return []

        try:
            return self.pdf_layout_analyzer.analyze(
                _copy_stream(source_stream),
                source_name=source_name,
            )
        except Exception as exc:
            if self.pdf_layout_debug:
                self._warn(f"PDF layout prepass failed for {source_name}: {exc}")
            return []

    def _layout_region_items_from_regions(
        self,
        *,
        regions: list[PDFLayoutRegion],
        page_num: int,
        y_offset: float,
        order_offset: int,
        ocr_service: OCRBackend,
        stream_info: StreamInfo | None,
        artifact_context: _ArtifactContext,
        artifact_counters: dict[str, int],
        fallback_category: PDFLayoutCategory,
        as_markdown: bool = False,
        emit_text_regions: bool = True,
        ocr_visual_regions: bool = True,
    ) -> list[_ContentItem] | str:
        if not regions:
            return "" if as_markdown else []

        items: list[_ContentItem] = []
        for index, region in enumerate(regions, start=1):
            markdown = self._region_markdown(
                region=region,
                page_num=page_num,
                region_index=index,
                ocr_service=ocr_service,
                stream_info=stream_info,
                artifact_context=artifact_context,
                artifact_counters=artifact_counters,
                fallback_category=fallback_category,
                emit_text_regions=emit_text_regions,
                ocr_visual_regions=ocr_visual_regions,
            )
            if not markdown.strip():
                continue

            region_y = y_offset + float(index) * 0.001
            if region.bbox is not None:
                region_y = y_offset + float(region.bbox[1])
            items.append(
                _ContentItem(
                    y_pos=region_y,
                    order=order_offset + index,
                    markdown=markdown,
                    bbox=region.bbox,
                )
            )

        if not items:
            return "" if as_markdown else []

        items.sort(key=lambda item: (item.y_pos, item.order))
        if as_markdown:
            return "\n\n".join(item.markdown for item in items if item.markdown.strip())
        return items

    def _visual_candidates_from_layout_regions(
        self,
        *,
        regions: list[PDFLayoutRegion],
        y_offset: float,
        order_offset: int,
        source_size: tuple[int, int] | None = None,
    ) -> list[_VisualCandidate]:
        candidates: list[_VisualCandidate] = []
        for index, region in enumerate(regions, start=1):
            category = getattr(region, "category", "complex_like")
            if category not in {"image_like", "complex_like"}:
                continue
            if not self._should_embed_region_artifact(region, category):
                continue

            region_y = y_offset + float(index) * 0.001
            bbox = self._scale_layout_bbox(region.bbox, source_size)
            if bbox is not None:
                region_y = y_offset + float(bbox[1])

            candidates.append(
                _VisualCandidate(
                    y_pos=region_y,
                    order=order_offset + index,
                    image_stream=_copy_stream(region.image_stream),
                    bbox=bbox,
                    source="docling",
                    category=category,
                    confidence=getattr(region, "confidence", None),
                )
            )
        return candidates

    def _scale_layout_bbox(
        self,
        bbox: tuple[float, float, float, float] | None,
        source_size: tuple[int, int] | None,
    ) -> tuple[float, float, float, float] | None:
        if bbox is None:
            return None

        left, top, right, bottom = bbox
        if right <= left or bottom <= top:
            return None

        if source_size is None:
            return bbox

        width, height = source_size
        if width <= 0 or height <= 0:
            return bbox

        # Docling item provenance often returns normalized 0..1 coordinates,
        # while PaddleOCR-VL placeholders use rendered-page pixels.
        if all(0.0 <= value <= 1.0 for value in bbox):
            return (
                left * width,
                top * height,
                right * width,
                bottom * height,
            )

        return bbox

    def _region_markdown(
        self,
        *,
        region: PDFLayoutRegion,
        page_num: int,
        region_index: int,
        ocr_service: OCRBackend,
        stream_info: StreamInfo | None,
        artifact_context: _ArtifactContext,
        artifact_counters: dict[str, int],
        fallback_category: PDFLayoutCategory,
        emit_text_regions: bool = True,
        ocr_visual_regions: bool = True,
    ) -> str:
        category = getattr(region, "category", fallback_category)
        image_stream = _copy_stream(region.image_stream)

        if category in {"text_like", "table_like"} and not emit_text_regions:
            return ""

        should_embed_artifact = self._should_embed_region_artifact(region, category)
        if (
            category in {"image_like", "complex_like"}
            and not should_embed_artifact
            and not emit_text_regions
        ):
            return ""

        if category in {"text_like", "table_like"}:
            ocr_result = ocr_service.extract_text(
                image_stream,
                stream_info=self._image_stream_info(stream_info),
            )
            text = self._strip_ocr_image_placeholders((ocr_result.text or "").strip())
            return text if emit_text_regions else ""

        image_markdown = ""
        if should_embed_artifact:
            artifact_path = self._save_artifact(
                image_stream=image_stream,
                page_num=page_num,
                artifact_counters=artifact_counters,
                artifact_context=artifact_context,
                artifact_label="region",
            )
            if artifact_path is not None:
                image_markdown = self._format_image_markdown(
                    artifact_path,
                    artifact_context,
                )

        if image_markdown and not ocr_visual_regions:
            return image_markdown

        ocr_result = ocr_service.extract_text(
            image_stream,
            stream_info=self._image_stream_info(stream_info),
        )
        text = self._strip_ocr_image_placeholders((ocr_result.text or "").strip())
        if text:
            if image_markdown:
                return f"{image_markdown}\n\n{self._format_ocr_block(text)}"
            return self._format_ocr_block(text) if emit_text_regions else ""

        return image_markdown

    def _should_embed_region_artifact(
        self,
        region: PDFLayoutRegion,
        category: PDFLayoutCategory,
    ) -> bool:
        if category not in {"image_like", "complex_like"}:
            return False

        confidence = getattr(region, "confidence", None)
        if confidence is not None:
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                confidence_value = None
            if category == "image_like" and confidence_value is not None and confidence_value < 0.35:
                return False
            if category == "complex_like" and confidence_value is not None and confidence_value < 0.50:
                return False

        try:
            image_stream = _copy_stream(region.image_stream)
            with Image.open(image_stream) as image:
                width, height = image.size
        except Exception:
            # Unit-test stubs use byte streams instead of real PNGs; preserve the
            # previous behavior there and let real images use the size gates.
            return True

        if width <= 0 or height <= 0:
            return False

        area = width * height
        if category == "image_like":
            return width >= 48 and height >= 32 and area >= 2_000

        # Docling layout can label short formulas/page furniture as complex.
        # Only persist visual artifacts for blocks large enough to be figures,
        # charts, dense tables, or genuinely hard mixed regions.
        return width >= 120 and height >= 60 and area >= 7_200

    def _merge_scanned_ocr_with_visuals(
        self,
        full_text: str,
        visual_items: list[_ContentItem],
    ) -> str:
        visual_blocks = [
            item.markdown.strip() for item in sorted(visual_items, key=lambda item: (item.y_pos, item.order))
            if item.markdown.strip()
        ]
        if not visual_blocks:
            return self._format_ocr_block(self._strip_ocr_image_placeholders(full_text))

        parts = _OCR_IMAGE_PLACEHOLDER_RE.split(full_text)
        if len(parts) == 1:
            return "\n\n".join(
                [self._format_ocr_block(full_text.strip()), *visual_blocks]
            ).strip()

        blocks: list[str] = []
        inserted_visual_count = 0
        for index, part in enumerate(parts):
            cleaned = self._clean_ocr_markdown(part)
            if cleaned:
                blocks.append(self._format_ocr_block(cleaned))
            if index < len(parts) - 1 and index < len(visual_blocks):
                blocks.append(visual_blocks[index])
                inserted_visual_count += 1

        if len(visual_blocks) > inserted_visual_count:
            blocks.extend(visual_blocks[inserted_visual_count:])

        return "\n\n".join(block for block in blocks if block.strip()).strip()

    def _clean_ocr_markdown(self, text: str) -> str:
        text = _OCR_MARKDOWN_IMAGE_PLACEHOLDER_RE.sub("", text)
        text = _OCR_HTML_IMAGE_RE.sub("", text)
        text = _HTML_DIV_RE.sub(
            lambda match: str(match.group("content") or "").strip(),
            text,
        )
        return _normalize_inline_latex(text).strip()

    def _strip_ocr_image_placeholders(self, text: str) -> str:
        return self._clean_ocr_markdown(_OCR_IMAGE_PLACEHOLDER_RE.sub("", text))

    def _ocr_placeholder_visual_candidates(
        self,
        *,
        full_text: str,
        rendered_page: io.BytesIO,
    ) -> list[_VisualCandidate]:
        if not full_text:
            return []

        try:
            page_image_stream = _copy_stream(rendered_page)
            page_image = Image.open(page_image_stream).convert("RGB")
        except Exception:
            return []

        candidates: list[_VisualCandidate] = []
        for match_index, match in enumerate(
            _OCR_IMAGE_PLACEHOLDER_BOX_RE.finditer(full_text),
            start=1,
        ):
            bbox = self._placeholder_bbox_from_match(
                match,
                image_width=page_image.width,
                image_height=page_image.height,
            )
            if bbox is None:
                continue

            left, top, right, bottom = bbox
            width = right - left
            height = bottom - top
            if width < 32 or height < 24 or (width * height) < 1_000:
                continue

            crop = page_image.crop(
                (
                    int(round(left)),
                    int(round(top)),
                    int(round(right)),
                    int(round(bottom)),
                )
            )
            crop = self._refine_visual_crop(crop)
            crop_stream = io.BytesIO()
            crop.save(crop_stream, format="PNG")
            crop_stream.seek(0)

            candidates.append(
                _VisualCandidate(
                    y_pos=top,
                    order=match.start() + match_index,
                    image_stream=crop_stream,
                    bbox=bbox,
                    source="ocr_placeholder",
                    category="image_like",
                    confidence=None,
                )
            )

        return candidates

    def _select_visual_candidates(
        self,
        docling_candidates: list[_VisualCandidate],
        placeholder_candidates: list[_VisualCandidate],
    ) -> list[_VisualCandidate]:
        selected = list(docling_candidates)
        for candidate in sorted(
            placeholder_candidates,
            key=lambda item: self._bbox_area(item.bbox),
            reverse=True,
        ):
            if any(
                self._bbox_overlap_ratio(candidate.bbox, selected_candidate.bbox) >= 0.50
                for selected_candidate in selected
            ):
                continue
            selected.append(candidate)

        return sorted(selected, key=lambda item: (item.y_pos, item.order))

    def _bbox_area(self, bbox: tuple[float, float, float, float] | None) -> float:
        if bbox is None:
            return 0.0
        return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

    def _save_visual_candidates(
        self,
        *,
        candidates: list[_VisualCandidate],
        page_num: int,
        artifact_context: _ArtifactContext,
        artifact_counters: dict[str, int],
    ) -> list[_ContentItem]:
        items: list[_ContentItem] = []
        for index, candidate in enumerate(candidates, start=1):
            artifact_path = self._save_artifact(
                image_stream=_copy_stream(candidate.image_stream),
                page_num=page_num,
                artifact_counters=artifact_counters,
                artifact_context=artifact_context,
                artifact_label="region",
            )
            if artifact_path is None:
                continue

            items.append(
                _ContentItem(
                    y_pos=candidate.y_pos,
                    order=candidate.order if candidate.order else index,
                    markdown=self._format_image_markdown(
                        artifact_path,
                        artifact_context,
                    ),
                    bbox=candidate.bbox,
                )
            )

        return items

    def _bbox_overlap_ratio(
        self,
        first: tuple[float, float, float, float] | None,
        second: tuple[float, float, float, float] | None,
    ) -> float:
        if first is None or second is None:
            return 0.0

        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        if right <= left or bottom <= top:
            return 0.0

        overlap_area = (right - left) * (bottom - top)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        smaller_area = min(first_area, second_area)
        if smaller_area <= 0:
            return 0.0
        return overlap_area / smaller_area

    def _ocr_placeholder_visual_items(
        self,
        *,
        full_text: str,
        rendered_page: io.BytesIO,
        page_num: int,
        artifact_context: _ArtifactContext,
        artifact_counters: dict[str, int],
    ) -> list[_ContentItem]:
        if not full_text or not artifact_context.export_enabled:
            return []

        try:
            page_image_stream = _copy_stream(rendered_page)
            page_image = Image.open(page_image_stream).convert("RGB")
        except Exception:
            return []

        items: list[_ContentItem] = []
        for match_index, match in enumerate(
            _OCR_IMAGE_PLACEHOLDER_BOX_RE.finditer(full_text),
            start=1,
        ):
            bbox = self._placeholder_bbox_from_match(
                match,
                image_width=page_image.width,
                image_height=page_image.height,
            )
            if bbox is None:
                continue

            left, top, right, bottom = bbox
            width = right - left
            height = bottom - top
            if width < 32 or height < 24 or (width * height) < 1_000:
                continue

            crop = page_image.crop(
                (
                    int(round(left)),
                    int(round(top)),
                    int(round(right)),
                    int(round(bottom)),
                )
            )
            crop = self._refine_visual_crop(crop)
            crop_stream = io.BytesIO()
            crop.save(crop_stream, format="PNG")
            crop_stream.seek(0)

            artifact_path = self._save_artifact(
                image_stream=crop_stream,
                page_num=page_num,
                artifact_counters=artifact_counters,
                artifact_context=artifact_context,
                artifact_label="region",
            )
            if artifact_path is None:
                continue

            items.append(
                _ContentItem(
                    y_pos=top,
                    order=match.start() + match_index,
                    markdown=self._format_image_markdown(
                        artifact_path,
                        artifact_context,
                    ),
                    bbox=bbox,
                )
            )

        return items

    def _placeholder_bbox_from_match(
        self,
        match: re.Match[str],
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float, float, float] | None:
        try:
            left = float(match.group("x0"))
            top = float(match.group("y0"))
            right = float(match.group("x1"))
            bottom = float(match.group("y1"))
        except Exception:
            return None

        pad = 4.0
        left = max(0.0, left - pad)
        top = max(0.0, top - pad)
        right = min(float(image_width), right + pad)
        bottom = min(float(image_height), bottom + pad)
        if right <= left or bottom <= top:
            return None
        return left, top, right, bottom

    def _refine_visual_crop(self, image: Image.Image) -> Image.Image:
        """
        Tighten PaddleOCR-VL image placeholder crops.

        Paddle's `img_in_image_box_*` coordinates are in rendered-page pixels
        but can include nearby body text. Keep large connected dark regions
        such as axes/curves and discard isolated text fragments around them.
        """

        grayscale = image.convert("L")
        width, height = grayscale.size
        if width < 64 or height < 48:
            return image

        pixels = grayscale.load()
        seen: set[tuple[int, int]] = set()
        selected_boxes: list[tuple[int, int, int, int]] = []

        for y in range(height):
            for x in range(width):
                if (x, y) in seen or pixels[x, y] >= 190:
                    continue

                stack = [(x, y)]
                seen.add((x, y))
                min_x = max_x = x
                min_y = max_y = y
                area = 0

                while stack:
                    current_x, current_y = stack.pop()
                    area += 1
                    min_x = min(min_x, current_x)
                    max_x = max(max_x, current_x)
                    min_y = min(min_y, current_y)
                    max_y = max(max_y, current_y)

                    for next_y in range(current_y - 1, current_y + 2):
                        if next_y < 0 or next_y >= height:
                            continue
                        for next_x in range(current_x - 1, current_x + 2):
                            if next_x < 0 or next_x >= width:
                                continue
                            point = (next_x, next_y)
                            if point in seen or pixels[next_x, next_y] >= 190:
                                continue
                            seen.add(point)
                            stack.append(point)

                box_width = max_x - min_x + 1
                box_height = max_y - min_y + 1
                is_large_shape = area >= 500
                is_long_axis = box_height >= 70 and box_width >= 8
                is_wide_curve = box_width >= 90 and box_height >= 8
                if is_large_shape or is_long_axis or is_wide_curve:
                    selected_boxes.append((min_x, min_y, max_x + 1, max_y + 1))

        if not selected_boxes:
            return image

        left = min(box[0] for box in selected_boxes)
        top = min(box[1] for box in selected_boxes)
        right = max(box[2] for box in selected_boxes)
        bottom = max(box[3] for box in selected_boxes)
        padding = 8
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(width, right + padding)
        bottom = min(height, bottom + padding)

        original_area = width * height
        refined_area = max(0, right - left) * max(0, bottom - top)
        if refined_area < original_area * 0.20:
            return image

        return image.crop((left, top, right, bottom))

    def _merge_visual_item_candidates(
        self,
        primary_items: list[_ContentItem],
        secondary_items: list[_ContentItem],
    ) -> list[_ContentItem]:
        if not primary_items:
            return secondary_items
        return sorted(primary_items, key=lambda item: (item.y_pos, item.order))

    def _build_image_and_text_block(
        self,
        *,
        image_stream: io.BytesIO,
        page_num: int,
        ocr_service: OCRBackend,
        stream_info: StreamInfo | None,
        artifact_context: _ArtifactContext,
        artifact_counters: dict[str, int],
        artifact_label: str,
    ) -> str:
        image_stream = _copy_stream(image_stream)
        ocr_result = ocr_service.extract_text(
            image_stream,
            stream_info=self._image_stream_info(stream_info),
        )
        text = self._clean_ocr_markdown(ocr_result.text or "")
        artifact_path = self._save_artifact(
            image_stream=image_stream,
            page_num=page_num,
            artifact_counters=artifact_counters,
            artifact_context=artifact_context,
            artifact_label=artifact_label,
        )

        image_markdown = (
            self._format_image_markdown(artifact_path, artifact_context)
            if artifact_path
            else ""
        )
        if text:
            if image_markdown:
                return f"{image_markdown}\n\n{self._format_ocr_block(text)}"
            return self._format_ocr_block(text)
        return image_markdown

    def _format_ocr_block(self, text: str) -> str:
        return self._clean_ocr_markdown(text)

    def _ocr_image_to_text(
        self,
        image_stream: io.BytesIO,
        ocr_service: OCRBackend,
        stream_info: StreamInfo | None,
    ) -> str:
        result = ocr_service.extract_text(
            _copy_stream(image_stream),
            stream_info=self._image_stream_info(stream_info),
        )
        return (result.text or "").strip()

    def _image_stream_info(self, source_stream_info: StreamInfo | None) -> StreamInfo:
        if source_stream_info is None:
            return StreamInfo(extension=".png", mimetype="image/png")
        return source_stream_info.copy_and_update(
            mimetype="image/png",
            extension=".png",
        )

    def _format_image_markdown(
        self,
        artifact_path: Path,
        artifact_context: _ArtifactContext,
    ) -> str:
        try:
            display_path = os.path.relpath(
                artifact_path,
                start=artifact_context.markdown_dir,
            )
        except ValueError:
            display_path = artifact_path.as_posix()
        return f"![OCR region]({Path(display_path).as_posix()})"

    def _save_artifact(
        self,
        *,
        image_stream: io.BytesIO,
        page_num: int,
        artifact_counters: dict[str, int],
        artifact_context: _ArtifactContext,
        artifact_label: str,
    ) -> Path | None:
        if not artifact_context.export_enabled:
            return None

        counter_key = f"{page_num}:{artifact_label}"
        artifact_counters[counter_key] = artifact_counters.get(counter_key, 0) + 1
        index = artifact_counters[counter_key]

        if artifact_label == "full":
            filename = f"page-{page_num:04d}-full.png"
        elif artifact_label == "image":
            filename = f"page-{page_num:04d}-image-{index:04d}.png"
        else:
            filename = f"page-{page_num:04d}-region-{index:04d}.png"

        artifact_path = artifact_context.root_dir / filename
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        image_stream.seek(0)
        artifact_path.write_bytes(image_stream.read())
        image_stream.seek(0)
        return artifact_path

    def _render_page_to_stream(self, page: Any) -> io.BytesIO:
        page_image = page.to_image(resolution=220)
        stream = io.BytesIO()
        page_image.original.save(stream, format="PNG")
        stream.seek(0)
        return stream

    def _render_page_to_pdf_stream(self, page: Any) -> io.BytesIO | None:
        try:
            import fitz

            page_number = int(getattr(page, "page_number", 1))
            pdf = getattr(page, "pdf", None)
            stream = getattr(pdf, "stream", None)
            if stream is None:
                return None

            current_pos = stream.tell() if hasattr(stream, "tell") else None
            try:
                stream.seek(0)
                source_pdf = fitz.open(stream=stream.read(), filetype="pdf")
            finally:
                if current_pos is not None:
                    stream.seek(current_pos)

            single_page = fitz.open()
            try:
                single_page.insert_pdf(
                    source_pdf,
                    from_page=page_number - 1,
                    to_page=page_number - 1,
                )
                output = io.BytesIO(single_page.tobytes())
                output.seek(0)
                return output
            finally:
                single_page.close()
                source_pdf.close()
        except Exception as exc:
            if self.pdf_layout_debug:
                self._warn(f"could not build single-page PDF for layout: {exc}")
            return None

    def _ocr_full_pages(
        self,
        pdf_bytes: io.BytesIO,
        ocr_service: OCRBackend,
        *,
        stream_info: StreamInfo | None,
        artifact_context: _ArtifactContext,
        artifact_counters: dict[str, int],
    ) -> str:
        sections: list[str] = []

        try:
            pdf_bytes.seek(0)
            with pdfplumber.open(pdf_bytes) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    rendered_page = self._render_page_to_stream(page)
                    page_markdown = self._process_scanned_page(
                        rendered_page,
                        page_num,
                        ocr_service,
                        stream_info=stream_info,
                        artifact_context=artifact_context,
                        artifact_counters=artifact_counters,
                        layout_stream=None,
                    )
                    section = f"## Page {page_num}"
                    if page_markdown.strip():
                        section = f"{section}\n\n{page_markdown.strip()}"
                    sections.append(section)
            return "\n\n".join(sections).strip()
        except Exception as first_exc:
            if self.pdf_layout_debug:
                self._warn(f"pdfplumber full-page OCR failed, retrying with PyMuPDF: {first_exc}")

        try:
            import fitz

            pdf_bytes.seek(0)
            document = fitz.open(stream=pdf_bytes.read(), filetype="pdf")
            try:
                for page_num in range(1, document.page_count + 1):
                    page = document[page_num - 1]
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(220 / 72, 220 / 72))
                    rendered_page = io.BytesIO(pixmap.tobytes("png"))
                    rendered_page.seek(0)
                    page_markdown = self._process_scanned_page(
                        rendered_page,
                        page_num,
                        ocr_service,
                        stream_info=stream_info,
                        artifact_context=artifact_context,
                        artifact_counters=artifact_counters,
                        layout_stream=None,
                    )
                    section = f"## Page {page_num}"
                    if page_markdown.strip():
                        section = f"{section}\n\n{page_markdown.strip()}"
                    sections.append(section)
            finally:
                document.close()
            return "\n\n".join(sections).strip()
        except Exception:
            return "*[Error: Could not process scanned PDF]*"
