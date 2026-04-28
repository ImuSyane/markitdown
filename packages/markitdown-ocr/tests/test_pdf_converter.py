"""
Focused tests for the plugin PDF converter.

These tests validate the plugin-only PDF behavior added in markitdown-ocr:
- Docling-style layout prepass routing
- cropped artifact export for image/complex regions
- image+text markdown output
- clean fallback to prior OCR behavior when layout is unavailable or fails
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from markitdown import StreamInfo  # noqa: E402
from markitdown_ocr._ocr_service import OCRResult  # noqa: E402
from markitdown_ocr._pdf_converter_with_ocr import PdfConverterWithOCR  # noqa: E402
from markitdown_ocr._pdf_layout import (  # noqa: E402
    PDFLayoutRegion,
    resolve_pdf_layout_analyzer,
)

TEST_DATA_DIR = Path(__file__).parent / "ocr_test_data"


class StubLayoutAnalyzer:
    backend_name = "docling"

    def __init__(
        self,
        regions: list[PDFLayoutRegion] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.regions = regions or []
        self.error = error
        self.calls: list[str] = []

    def analyze(
        self,
        image_stream: Any,
        *,
        source_name: str = "document.png",
    ) -> list[PDFLayoutRegion]:
        self.calls.append(source_name)
        if self.error is not None:
            raise self.error
        image_stream.seek(0)
        return self.regions


class MappingOCRService:
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping
        self.calls: list[str] = []

    def extract_text(self, image_stream: Any, **kwargs: Any) -> OCRResult:
        image_stream.seek(0)
        key = image_stream.read().decode("utf-8")
        image_stream.seek(0)
        self.calls.append(key)
        return OCRResult(text=self.mapping.get(key, ""), backend_used="mock")


class StaticOCRService:
    def __init__(self, text: str = "MOCK_OCR_TEXT_12345") -> None:
        self.text = text

    def extract_text(self, image_stream: Any, **kwargs: Any) -> OCRResult:
        image_stream.seek(0)
        return OCRResult(text=self.text, backend_used="mock")


class FakeRenderedPage:
    def __init__(self, payload: bytes) -> None:
        self.original = self
        self.payload = payload

    def save(self, stream: io.BytesIO, format: str = "PNG") -> None:  # noqa: A002
        stream.write(self.payload)


class FakePageImage:
    def __init__(self, payload: bytes) -> None:
        self.original = FakeRenderedPage(payload)


class FakePage:
    def __init__(self, lines: list[str] | None = None, *, page_number: int = 1) -> None:
        self.page_number = page_number
        self.width = 100
        self.height = 100
        self._lines = lines or []
        self.chars = self._build_chars(self._lines)

    def extract_text(self) -> str:
        return "\n".join(self._lines)

    def to_image(self, resolution: int = 220) -> FakePageImage:
        return FakePageImage(f"page-{self.page_number}-render".encode("utf-8"))

    def within_bbox(self, bbox: tuple[float, float, float, float]) -> "FakePage":
        return self

    def _build_chars(self, lines: list[str]) -> list[dict[str, Any]]:
        chars: list[dict[str, Any]] = []
        for line_index, line in enumerate(lines):
            for column_index, char in enumerate(line):
                chars.append(
                    {
                        "text": char,
                        "top": line_index * 12.0,
                        "x0": float(column_index),
                    }
                )
        return chars


def make_png_stream(
    size: tuple[int, int] = (80, 70),
    color: str = "white",
) -> io.BytesIO:
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    stream.seek(0)
    return stream


def test_pdf_layout_auto_without_docling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "markitdown_ocr._pdf_layout.is_docling_available", lambda: False
    )

    analyzer, resolved = resolve_pdf_layout_analyzer("auto")

    assert analyzer is None
    assert resolved == "none"


def test_pdf_layout_auto_with_docling(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = StubLayoutAnalyzer()
    monkeypatch.setattr(
        "markitdown_ocr._pdf_layout.is_docling_available", lambda: True
    )
    monkeypatch.setattr(
        "markitdown_ocr._pdf_layout.DoclingPDFLayoutAnalyzer",
        lambda debug=False: stub,
    )

    analyzer, resolved = resolve_pdf_layout_analyzer("auto")

    assert analyzer is stub
    assert resolved == "docling"


def test_pdf_layout_docling_warns_and_falls_back_without_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "markitdown_ocr._pdf_layout.is_docling_available", lambda: False
    )

    with pytest.warns(RuntimeWarning, match="pdf_layout_backend='docling'"):
        analyzer, resolved = resolve_pdf_layout_analyzer("docling")

    assert analyzer is None
    assert resolved == "none"


def test_pdf_layout_none_bypasses_layout_path() -> None:
    converter = PdfConverterWithOCR(pdf_layout_backend="none")

    assert converter.pdf_layout_analyzer is None
    assert converter.resolved_pdf_layout_backend == "none"


def test_scanned_page_uses_layout_regions_and_exports_complex_artifact(
    tmp_path: Path,
) -> None:
    analyzer = StubLayoutAnalyzer(
        [
            PDFLayoutRegion(
                kind="text",
                category="text_like",
                image_stream=io.BytesIO(b"region-1"),
            ),
            PDFLayoutRegion(
                kind="figure",
                category="image_like",
                image_stream=io.BytesIO(b"region-2"),
            ),
        ]
    )
    converter = PdfConverterWithOCR(
        ocr_service=MappingOCRService({"region-1": "Alpha", "region-2": "Beta"}),
        pdf_layout_analyzer=analyzer,
        ocr_artifact_dir=tmp_path,
    )
    page = FakePage([])

    with patch.object(
        converter,
        "_render_page_to_stream",
        return_value=io.BytesIO(b"page-image"),
    ):
        markdown = converter._convert_page(
            page,
            1,
            converter.ocr_service,
            stream_info=StreamInfo(extension=".pdf", local_path=str(tmp_path / "doc.pdf")),
        )

    assert analyzer.calls == ["page_1.png"]
    assert "Alpha" in markdown
    assert "Beta" in markdown
    assert "![OCR region](" in markdown
    assert (tmp_path / "page-0001-region-0001.png").exists()


def test_large_embedded_image_uses_layout_prepass(tmp_path: Path) -> None:
    analyzer = StubLayoutAnalyzer(
        [
            PDFLayoutRegion(
                kind="text",
                category="text_like",
                image_stream=io.BytesIO(b"layout-region"),
            )
        ]
    )
    converter = PdfConverterWithOCR(
        ocr_service=MappingOCRService(
            {"layout-region": "LAYOUT_TEXT", "big-image": "RAW_TEXT"}
        ),
        pdf_layout_analyzer=analyzer,
        ocr_artifact_dir=tmp_path,
    )
    page = FakePage(["Before image"])
    image_info = {
        "stream": io.BytesIO(b"big-image"),
        "name": "page_1_img_0",
        "y_pos": 14,
        "area_ratio": 0.35,
    }

    with patch(
        "markitdown_ocr._pdf_converter_with_ocr._extract_images_from_page",
        return_value=[image_info],
    ):
        markdown = converter._convert_page(
            page,
            1,
            converter.ocr_service,
            stream_info=StreamInfo(extension=".pdf", local_path=str(tmp_path / "doc.pdf")),
        )

    assert analyzer.calls == ["page_1_img_0.png"]
    assert "LAYOUT_TEXT" in markdown
    assert "RAW_TEXT" not in markdown


def test_small_embedded_image_keeps_single_image_path_and_exports_artifact(
    tmp_path: Path,
) -> None:
    service = MappingOCRService({"small-image": "SMALL_IMAGE_TEXT"})
    converter = PdfConverterWithOCR(
        ocr_service=service,
        pdf_layout_analyzer=StubLayoutAnalyzer(),
        ocr_artifact_dir=tmp_path,
    )
    page = FakePage(["Before image"])
    image_info = {
        "stream": io.BytesIO(b"small-image"),
        "name": "page_1_img_0",
        "y_pos": 14,
        "area_ratio": 0.05,
    }

    with patch(
        "markitdown_ocr._pdf_converter_with_ocr._extract_images_from_page",
        return_value=[image_info],
    ):
        markdown = converter._convert_page(
            page,
            1,
            converter.ocr_service,
            stream_info=StreamInfo(extension=".pdf", local_path=str(tmp_path / "doc.pdf")),
        )

    assert "SMALL_IMAGE_TEXT" in markdown
    assert "![OCR region](" in markdown
    assert (tmp_path / "page-0001-image-0001.png").exists()


def test_docling_failure_falls_back_for_scanned_page(tmp_path: Path) -> None:
    converter = PdfConverterWithOCR(
        ocr_service=MappingOCRService({"page-image": "FULL_PAGE_TEXT"}),
        pdf_layout_analyzer=StubLayoutAnalyzer(error=RuntimeError("boom")),
        pdf_layout_debug=True,
        ocr_artifact_dir=tmp_path,
    )
    page = FakePage([])

    with patch.object(
        converter,
        "_render_page_to_stream",
        return_value=io.BytesIO(b"page-image"),
    ):
        markdown = converter._convert_page(
            page,
            1,
            converter.ocr_service,
            stream_info=StreamInfo(extension=".pdf", local_path=str(tmp_path / "doc.pdf")),
        )

    assert "FULL_PAGE_TEXT" in markdown
    assert "![OCR region](" not in markdown
    assert not (tmp_path / "page-0001-full.png").exists()


def test_scanned_page_uses_ocr_placeholder_boxes_as_visual_crops(
    tmp_path: Path,
) -> None:
    page_image = Image.new("RGB", (200, 200), "white")
    rendered = io.BytesIO()
    page_image.save(rendered, format="PNG")
    rendered.seek(0)
    ocr_text = (
        "Before\n\n"
        '<div style="text-align: center;"><img '
        'src="imgs/img_in_image_box_20_30_90_100.jpg" '
        'alt="Image" width="50%" /></div>\n\n'
        "After"
    )
    converter = PdfConverterWithOCR(
        ocr_service=StaticOCRService(ocr_text),
        pdf_layout_backend="none",
        ocr_artifact_dir=tmp_path / "artifacts",
    )
    page = FakePage([])

    with patch.object(
        converter,
        "_render_page_to_stream",
        return_value=rendered,
    ):
        markdown = converter._convert_page(
            page,
            1,
            converter.ocr_service,
            stream_info=StreamInfo(extension=".pdf", local_path=str(tmp_path / "doc.pdf")),
        )

    assert "Before" in markdown
    assert "After" in markdown
    assert "imgs/img_in_image_box" not in markdown
    assert "![OCR region](artifacts/page-0001-region-0001.png)" in markdown
    assert (tmp_path / "artifacts/page-0001-region-0001.png").exists()
    with Image.open(tmp_path / "artifacts/page-0001-region-0001.png") as crop:
        assert crop.size == (78, 78)


def test_scanned_page_prefers_docling_over_overlapping_ocr_placeholder(
    tmp_path: Path,
) -> None:
    rendered = make_png_stream((200, 200))
    analyzer = StubLayoutAnalyzer(
        [
            PDFLayoutRegion(
                kind="figure",
                category="image_like",
                image_stream=make_png_stream((80, 70), "black"),
                bbox=(18, 28, 92, 102),
            )
        ]
    )
    ocr_text = (
        "Before\n\n"
        '<div><img src="imgs/img_in_image_box_20_30_90_100.jpg" /></div>\n\n'
        "After"
    )
    converter = PdfConverterWithOCR(
        ocr_service=StaticOCRService(ocr_text),
        pdf_layout_analyzer=analyzer,
        ocr_artifact_dir=tmp_path,
    )

    with patch.object(converter, "_render_page_to_stream", return_value=rendered):
        markdown = converter._convert_page(
            FakePage([]),
            1,
            converter.ocr_service,
            stream_info=StreamInfo(extension=".pdf", local_path=str(tmp_path / "doc.pdf")),
        )

    artifacts = sorted(tmp_path.glob("page-0001-region-*.png"))
    assert len(artifacts) == 1
    assert markdown.count("![OCR region](") == 1
    assert "imgs/img_in_image_box" not in markdown
    with Image.open(artifacts[0]) as crop:
        assert crop.size == (80, 70)


def test_normalized_docling_bbox_is_scaled_before_placeholder_overlap(
    tmp_path: Path,
) -> None:
    rendered = make_png_stream((200, 200))
    analyzer = StubLayoutAnalyzer(
        [
            PDFLayoutRegion(
                kind="figure",
                category="image_like",
                image_stream=make_png_stream((80, 70), "black"),
                bbox=(0.10, 0.15, 0.45, 0.50),
            )
        ]
    )
    ocr_text = (
        "Before\n\n"
        '<div><img src="imgs/img_in_image_box_20_30_90_100.jpg" /></div>\n\n'
        "After"
    )
    converter = PdfConverterWithOCR(
        ocr_service=StaticOCRService(ocr_text),
        pdf_layout_analyzer=analyzer,
        ocr_artifact_dir=tmp_path,
    )

    with patch.object(converter, "_render_page_to_stream", return_value=rendered):
        markdown = converter._convert_page(
            FakePage([]),
            1,
            converter.ocr_service,
            stream_info=StreamInfo(extension=".pdf", local_path=str(tmp_path / "doc.pdf")),
        )

    artifacts = sorted(tmp_path.glob("page-0001-region-*.png"))
    assert len(artifacts) == 1
    assert markdown.count("![OCR region](") == 1


def test_docling_layout_regions_suppress_ocr_placeholder_supplements(
    tmp_path: Path,
) -> None:
    rendered = make_png_stream((220, 220))
    analyzer = StubLayoutAnalyzer(
        [
            PDFLayoutRegion(
                kind="figure",
                category="image_like",
                image_stream=make_png_stream((80, 70), "black"),
                bbox=(120, 120, 200, 190),
            )
        ]
    )
    ocr_text = (
        "Before\n\n"
        '<div><img src="imgs/img_in_image_box_20_30_90_100.jpg" /></div>\n\n'
        "After"
    )
    converter = PdfConverterWithOCR(
        ocr_service=StaticOCRService(ocr_text),
        pdf_layout_analyzer=analyzer,
        ocr_artifact_dir=tmp_path,
    )

    with patch.object(converter, "_render_page_to_stream", return_value=rendered):
        markdown = converter._convert_page(
            FakePage([]),
            1,
            converter.ocr_service,
            stream_info=StreamInfo(extension=".pdf", local_path=str(tmp_path / "doc.pdf")),
        )

    artifacts = sorted(tmp_path.glob("page-0001-region-*.png"))
    assert len(artifacts) == 1
    assert markdown.count("![OCR region](") == len(artifacts)
    assert "imgs/img_in_image_box" not in markdown


def test_docling_low_confidence_suppresses_visual_artifact(tmp_path: Path) -> None:
    analyzer = StubLayoutAnalyzer(
        [
            PDFLayoutRegion(
                kind="figure",
                category="image_like",
                image_stream=make_png_stream(),
                bbox=(20, 20, 100, 90),
                confidence=0.20,
            )
        ]
    )
    converter = PdfConverterWithOCR(
        ocr_service=StaticOCRService("Plain OCR text"),
        pdf_layout_analyzer=analyzer,
        ocr_artifact_dir=tmp_path,
    )

    with patch.object(converter, "_render_page_to_stream", return_value=make_png_stream((200, 200))):
        markdown = converter._convert_page(
            FakePage([]),
            1,
            converter.ocr_service,
            stream_info=StreamInfo(extension=".pdf", local_path=str(tmp_path / "doc.pdf")),
        )

    assert "Plain OCR text" in markdown
    assert "![OCR region](" not in markdown
    assert not list(tmp_path.glob("page-0001-region-*.png"))


def test_docling_missing_confidence_allows_visual_artifact(tmp_path: Path) -> None:
    analyzer = StubLayoutAnalyzer(
        [
            PDFLayoutRegion(
                kind="figure",
                category="image_like",
                image_stream=make_png_stream(),
                bbox=(20, 20, 100, 90),
                confidence=None,
            )
        ]
    )
    converter = PdfConverterWithOCR(
        ocr_service=StaticOCRService("Plain OCR text"),
        pdf_layout_analyzer=analyzer,
        ocr_artifact_dir=tmp_path,
    )

    with patch.object(converter, "_render_page_to_stream", return_value=make_png_stream((200, 200))):
        markdown = converter._convert_page(
            FakePage([]),
            1,
            converter.ocr_service,
            stream_info=StreamInfo(extension=".pdf", local_path=str(tmp_path / "doc.pdf")),
        )

    artifacts = list(tmp_path.glob("page-0001-region-*.png"))
    assert len(artifacts) == 1
    assert markdown.count("![OCR region](") == 1


def test_pdf_ocr_markdown_tightens_inline_latex_only() -> None:
    converter = PdfConverterWithOCR(pdf_layout_backend="none")

    markdown = converter._format_ocr_block(
        "函数 $ f(x) $ 连续，且 $ y=x^2$。\n\n$$ x + 1 $$"
    )

    assert "$f(x)$" in markdown
    assert "$y=x^2$" in markdown
    assert "$ f(x) $" not in markdown
    assert "$$ x + 1 $$" in markdown


def test_pdf_ocr_markdown_cleans_image_html_markers() -> None:
    converter = PdfConverterWithOCR(pdf_layout_backend="none")

    markdown = converter._format_ocr_block(
        '<div style="text-align: center;">图 1-1</div>\n'
        '<img src="imgs/img_in_image_box_1_2_3_4.jpg">\n'
        '![](imgs/img_in_image_box_1_2_3_4.jpg)'
    )

    assert "<div" not in markdown
    assert "<img" not in markdown
    assert "imgs/" not in markdown
    assert "图 1-1" in markdown


def test_docling_failure_falls_back_for_large_embedded_image(tmp_path: Path) -> None:
    service = MappingOCRService({"big-image": "RAW_FALLBACK_TEXT"})
    converter = PdfConverterWithOCR(
        ocr_service=service,
        pdf_layout_analyzer=StubLayoutAnalyzer(error=RuntimeError("boom")),
        pdf_layout_debug=True,
        ocr_artifact_dir=tmp_path,
    )
    page = FakePage(["Before image"])
    image_info = {
        "stream": io.BytesIO(b"big-image"),
        "name": "page_1_img_0",
        "y_pos": 14,
        "area_ratio": 0.25,
    }

    with patch(
        "markitdown_ocr._pdf_converter_with_ocr._extract_images_from_page",
        return_value=[image_info],
    ):
        markdown = converter._convert_page(
            page,
            1,
            converter.ocr_service,
            stream_info=StreamInfo(extension=".pdf", local_path=str(tmp_path / "doc.pdf")),
        )

    assert "RAW_FALLBACK_TEXT" in markdown
    assert (tmp_path / "page-0001-image-0001.png").exists()


def test_native_text_order_is_preserved_without_qualifying_image(tmp_path: Path) -> None:
    service = MappingOCRService({"small-image": "IMAGE_TEXT"})
    converter = PdfConverterWithOCR(
        ocr_service=service,
        pdf_layout_backend="none",
        pdf_layout_min_area_ratio=0.20,
        ocr_artifact_dir=tmp_path,
    )
    page = FakePage(["First line", "Second line"])
    image_info = {
        "stream": io.BytesIO(b"small-image"),
        "name": "page_1_img_0",
        "y_pos": 30,
        "area_ratio": 0.05,
    }

    with patch(
        "markitdown_ocr._pdf_converter_with_ocr._extract_images_from_page",
        return_value=[image_info],
    ):
        markdown = converter._convert_page(
            page,
            1,
            converter.ocr_service,
            stream_info=StreamInfo(extension=".pdf", local_path=str(tmp_path / "doc.pdf")),
        )

    assert markdown.startswith("First line\n\nSecond line")
    assert markdown.index("First line") < markdown.index("Second line") < markdown.index(
        "IMAGE_TEXT"
    )


def test_convert_scanned_pdf_with_text_does_not_export_full_page_artifact(tmp_path: Path) -> None:
    path = TEST_DATA_DIR / "pdf_scanned_minimal.pdf"
    if not path.exists():
        pytest.skip(f"Test file not found: {path}")

    converter = PdfConverterWithOCR(
        ocr_service=StaticOCRService("SCANNED_TEXT"),
        pdf_layout_backend="none",
        ocr_artifact_dir=tmp_path,
    )

    with path.open("rb") as handle:
        result = converter.convert(
            handle,
            StreamInfo(extension=".pdf", local_path=str(path)),
        )

    assert "## Page 1" in result.markdown
    assert "SCANNED_TEXT" in result.markdown
    assert "![OCR region](" not in result.markdown
    assert not any(tmp_path.glob("page-0001-full.png"))


def test_accepts_pdf_by_extension_and_mimetype() -> None:
    converter = PdfConverterWithOCR()

    assert converter.accepts(io.BytesIO(b""), StreamInfo(extension=".pdf"))
    assert converter.accepts(
        io.BytesIO(b""), StreamInfo(mimetype="application/pdf")
    )
    assert not converter.accepts(io.BytesIO(b""), StreamInfo(extension=".docx"))
