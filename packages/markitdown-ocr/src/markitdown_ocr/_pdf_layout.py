"""
Internal PDF layout prepass helpers.

This module keeps Docling usage intentionally narrow: it only detects PDF/image
regions, classifies them, and returns cropped region images in reading order.
Final OCR extraction and Markdown assembly remain in the PDF converter.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal, Protocol, runtime_checkable

PDFLayoutBackend = Literal["auto", "none", "docling"]
PDFLayoutCategory = Literal["text_like", "table_like", "image_like", "complex_like"]

_TEXTUAL_LABEL_HINTS = {
    "caption",
    "footnote",
    "list_item",
    "page_footer",
    "page_header",
    "paragraph",
    "section_header",
    "text",
    "title",
}
_TABLE_LABEL_HINTS = {"table"}
_IMAGE_LABEL_HINTS = {
    "barcode",
    "chart",
    "diagram",
    "figure",
    "image",
    "logo",
    "picture",
    "qr_code",
    "seal",
    "signature",
    "stamp",
}
_COMPLEX_LABEL_HINTS = {
    "code",
    "equation",
    "formula",
    "math",
}
_DOCLING_LAYOUT_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
)
_BUNDLED_DOCLING_ARTIFACTS_ROOT = Path(__file__).parent / "models" / "docling"
_DOCLING_ARTIFACTS_ROOT = Path.home() / ".cache" / "markitdown-ocr" / "docling-artifacts"


@dataclass(slots=True)
class PDFLayoutRegion:
    """A cropped PDF region returned in reading order."""

    kind: str
    category: PDFLayoutCategory
    image_stream: io.BytesIO
    bbox: tuple[float, float, float, float] | None = None
    confidence: float | None = None


@runtime_checkable
class PDFLayoutAnalyzer(Protocol):
    """Internal interface for PDF layout prepass implementations."""

    backend_name: str

    def analyze(
        self,
        image_stream: BinaryIO,
        *,
        source_name: str = "document.png",
    ) -> list[PDFLayoutRegion]: ...


def resolve_pdf_layout_analyzer(
    backend: PDFLayoutBackend = "auto",
    *,
    debug: bool = False,
) -> tuple[PDFLayoutAnalyzer | None, Literal["none", "docling"]]:
    """
    Resolve the effective PDF layout analyzer.

    `auto` activates Docling when available and otherwise becomes `none`.
    `docling` warns and falls back to `none` if the optional dependency is not
    installed.
    """

    backend = backend if backend in {"auto", "none", "docling"} else "auto"
    if backend == "none":
        return None, "none"

    if not is_docling_available():
        if backend == "docling":
            warnings.warn(
                "markitdown-ocr: pdf_layout_backend='docling' requested but "
                "Docling is not installed; falling back to existing PDF OCR path",
                RuntimeWarning,
                stacklevel=2,
            )
        return None, "none"

    try:
        return DoclingPDFLayoutAnalyzer(debug=debug), "docling"
    except Exception as exc:
        if backend == "docling" or debug:
            warnings.warn(
                "markitdown-ocr: Docling PDF layout analyzer could not be "
                f"initialized ({exc}); falling back to existing PDF OCR path",
                RuntimeWarning,
                stacklevel=2,
            )
        return None, "none"


def is_docling_available() -> bool:
    """Return True when the optional Docling dependency can be imported."""
    try:
        from docling.document_converter import DocumentConverter  # noqa: F401
    except Exception:
        return False
    return True


class DoclingPDFLayoutAnalyzer:
    """
    Thin Docling-backed layout prepass adapter.

    The adapter returns cropped regions in Docling reading order and classifies
    them into text/table/image/complex buckets for the PDF converter.
    """

    backend_name = "docling"

    def __init__(self, *, debug: bool = False) -> None:
        self.debug = debug
        self._converter = self._build_converter()

    def analyze(
        self,
        image_stream: BinaryIO,
        *,
        source_name: str = "document.png",
    ) -> list[PDFLayoutRegion]:
        from docling.datamodel.base_models import DocumentStream

        image_stream.seek(0)
        source = DocumentStream(
            name=source_name,
            stream=io.BytesIO(image_stream.read()),
        )

        try:
            result = self._converter.convert(source)
            document = getattr(result, "document", None)
            if document is None:
                return []

            regions: list[PDFLayoutRegion] = []
            for item, _level in document.iterate_items():
                region = self._region_from_item(document, item)
                if region is not None:
                    regions.append(region)
            if not regions:
                regions = self._regions_from_layout_clusters(result)
            return regions
        finally:
            image_stream.seek(0)

    def _build_converter(self) -> Any:
        from docling.document_converter import DocumentConverter

        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import ImageFormatOption, PdfFormatOption

            pdf_pipeline_options = PdfPipelineOptions()
            if hasattr(pdf_pipeline_options, "do_ocr"):
                pdf_pipeline_options.do_ocr = False
            if hasattr(pdf_pipeline_options, "do_table_structure"):
                pdf_pipeline_options.do_table_structure = False
            if hasattr(pdf_pipeline_options, "generate_page_images"):
                pdf_pipeline_options.generate_page_images = True
            layout_model_spec = getattr(
                getattr(pdf_pipeline_options, "layout_options", None),
                "model_spec",
                None,
            )
            repo_id = getattr(layout_model_spec, "repo_id", None)
            revision = getattr(layout_model_spec, "revision", "main")
            artifacts_root = self._ensure_docling_layout_artifacts(
                repo_id=repo_id,
                revision=revision,
            )
            if hasattr(pdf_pipeline_options, "artifacts_path"):
                pdf_pipeline_options.artifacts_path = artifacts_root

            format_options: dict[Any, Any] = {
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_pipeline_options
                )
            }
            allowed_formats = [InputFormat.PDF]

            input_image = getattr(InputFormat, "IMAGE", None)
            if input_image is not None:
                allowed_formats.append(input_image)
                format_options[input_image] = ImageFormatOption(
                    pipeline_options=pdf_pipeline_options
                )

            return DocumentConverter(
                allowed_formats=allowed_formats,
                format_options=format_options,
            )
        except Exception:
            return DocumentConverter()

    def _docling_model_dir(self, repo_id: str) -> Path:
        return _DOCLING_ARTIFACTS_ROOT / repo_id.replace("/", "--")

    def _bundled_docling_model_dir(self, repo_id: str) -> Path | None:
        model_dir = _BUNDLED_DOCLING_ARTIFACTS_ROOT / repo_id.replace("/", "--")
        if self._docling_model_complete(model_dir):
            return model_dir
        return None

    def _docling_model_complete(self, model_dir: Path) -> bool:
        return model_dir.is_dir() and all(
            (model_dir / filename).is_file() for filename in _DOCLING_LAYOUT_REQUIRED_FILES
        )

    def _run_curl(self, args: list[str]) -> bytes:
        completed = subprocess.run(args, check=True, capture_output=True)
        return completed.stdout

    def _fetch_hf_model_metadata(self, repo_id: str) -> dict[str, Any]:
        url = f"https://huggingface.co/api/models/{repo_id}"
        try:
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=20.0) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            curl = shutil.which("curl")
            if curl:
                payload = self._run_curl(
                    [curl, "-L", "--fail", "--silent", "--show-error", url]
                )
                return json.loads(payload.decode("utf-8"))
            raise RuntimeError(f"unable to fetch Docling layout metadata: {exc}") from exc

    def _download_hf_file(
        self,
        repo_id: str,
        revision: str,
        filename: str,
        destination: Path,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded_filename = "/".join(
            urllib.parse.quote(part, safe="") for part in filename.split("/")
        )
        url = (
            "https://huggingface.co/"
            f"{repo_id}/resolve/{revision}/{encoded_filename}"
        )

        curl = shutil.which("curl")
        if curl:
            subprocess.run(
                [
                    curl,
                    "-L",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--continue-at",
                    "-",
                    "--output",
                    str(destination),
                    url,
                ],
                check=True,
            )
            return

        temp_destination = destination.with_suffix(f"{destination.suffix}.tmp")
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=60.0) as response:
            with open(temp_destination, "wb") as handle:
                shutil.copyfileobj(response, handle)
        os.replace(temp_destination, destination)

    def _ensure_docling_layout_artifacts(
        self,
        *,
        repo_id: str | None,
        revision: str | None,
    ) -> Path:
        resolved_repo_id = repo_id or "docling-project/docling-layout-heron"
        resolved_revision = revision or "main"
        bundled_model_dir = self._bundled_docling_model_dir(resolved_repo_id)
        if bundled_model_dir is not None:
            return bundled_model_dir

        model_dir = self._docling_model_dir(resolved_repo_id)
        if self._docling_model_complete(model_dir):
            return model_dir

        metadata = self._fetch_hf_model_metadata(resolved_repo_id)
        filenames = [
            str(item.get("rfilename")).strip()
            for item in metadata.get("siblings", [])
            if isinstance(item, dict) and item.get("rfilename")
        ]
        if not filenames:
            filenames = list(_DOCLING_LAYOUT_REQUIRED_FILES)

        for filename in filenames:
            destination = model_dir / filename
            if destination.is_file():
                continue
            self._download_hf_file(
                resolved_repo_id,
                resolved_revision,
                filename,
                destination,
            )

        if not self._docling_model_complete(model_dir):
            raise RuntimeError(
                f"Docling layout artifacts are incomplete under {model_dir}"
            )
        return model_dir

    def _region_from_item(
        self,
        document: Any,
        item: Any,
    ) -> PDFLayoutRegion | None:
        label = self._label_name(item)
        category = self._classify_label(label)

        region_image = None
        get_image = getattr(item, "get_image", None)
        if callable(get_image):
            try:
                region_image = get_image(document)
            except Exception:
                region_image = None

        bbox = self._bbox_from_item(document, item)
        if region_image is None and bbox is not None:
            region_image = self._crop_from_bbox(document, item, bbox)
        if region_image is None:
            return None

        stream = io.BytesIO()
        region_image.save(stream, format="PNG")
        stream.seek(0)
        return PDFLayoutRegion(
            kind=label or "region",
            category=category,
            image_stream=stream,
            bbox=bbox,
            confidence=self._confidence_from_object(item),
        )

    def _regions_from_layout_clusters(self, result: Any) -> list[PDFLayoutRegion]:
        regions: list[PDFLayoutRegion] = []
        pages = getattr(result, "pages", None) or []

        for page in pages:
            page_image = self._page_image(page)
            if page_image is None:
                continue

            page_size = getattr(page, "size", None)
            page_width = float(getattr(page_size, "width", page_image.width) or page_image.width)
            page_height = float(getattr(page_size, "height", page_image.height) or page_image.height)
            scale_x = page_image.width / page_width if page_width else 1.0
            scale_y = page_image.height / page_height if page_height else 1.0

            layout = getattr(getattr(page, "predictions", None), "layout", None)
            clusters = getattr(layout, "clusters", None) or []
            for cluster in clusters:
                bbox = getattr(cluster, "bbox", None)
                if bbox is None:
                    continue

                label = self._cluster_label_name(cluster)
                category = self._classify_label(label)
                crop_box = self._cluster_crop_box(
                    bbox,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    image_width=page_image.width,
                    image_height=page_image.height,
                )
                if crop_box is None:
                    continue

                crop = page_image.crop(crop_box)
                stream = io.BytesIO()
                crop.save(stream, format="PNG")
                stream.seek(0)
                regions.append(
                    PDFLayoutRegion(
                        kind=label or "region",
                        category=category,
                        image_stream=stream,
                        bbox=(
                            float(getattr(bbox, "l", crop_box[0])),
                            float(getattr(bbox, "t", crop_box[1])),
                            float(getattr(bbox, "r", crop_box[2])),
                            float(getattr(bbox, "b", crop_box[3])),
                        ),
                        confidence=self._confidence_from_object(cluster),
                    )
                )

        return regions

    def _page_image(self, page: Any) -> Any | None:
        get_image = getattr(page, "get_image", None)
        if callable(get_image):
            for scale in (2.0, 1.0):
                try:
                    image = get_image(scale=scale)
                    if image is not None:
                        return image
                except Exception:
                    continue

        page_image = getattr(getattr(page, "image", None), "pil_image", None)
        if page_image is not None:
            return page_image
        return None

    def _cluster_label_name(self, cluster: Any) -> str:
        label = getattr(cluster, "label", None)
        value = getattr(label, "value", label)
        return str(value).strip().lower()

    def _confidence_from_object(self, obj: Any) -> float | None:
        for name in ("confidence", "score", "conf"):
            value = getattr(obj, name, None)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _cluster_crop_box(
        self,
        bbox: Any,
        *,
        scale_x: float,
        scale_y: float,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int] | None:
        left = getattr(bbox, "l", None)
        top = getattr(bbox, "t", None)
        right = getattr(bbox, "r", None)
        bottom = getattr(bbox, "b", None)
        if None in {left, top, right, bottom}:
            return None

        x0 = int(max(0.0, float(left) * scale_x))
        y0 = int(max(0.0, float(top) * scale_y))
        x1 = int(min(float(image_width), float(right) * scale_x))
        y1 = int(min(float(image_height), float(bottom) * scale_y))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def _label_name(self, item: Any) -> str:
        label = getattr(item, "label", None)
        value = getattr(label, "value", label)
        return str(value).strip().lower()

    def _classify_label(self, label: str) -> PDFLayoutCategory:
        if not label:
            return "complex_like"
        if label in _TABLE_LABEL_HINTS or "table" in label:
            return "table_like"
        if label in {"form", "key_value_region", "document_index"}:
            return "complex_like"
        if label in _IMAGE_LABEL_HINTS or any(token in label for token in _IMAGE_LABEL_HINTS):
            return "image_like"
        if label in _COMPLEX_LABEL_HINTS or any(
            token in label for token in _COMPLEX_LABEL_HINTS
        ):
            return "complex_like"
        if label in _TEXTUAL_LABEL_HINTS or any(
            token in label for token in ("text", "title", "caption", "paragraph", "list")
        ):
            return "text_like"
        return "complex_like"

    def _first_provenance(self, item: Any) -> Any | None:
        prov = getattr(item, "prov", None)
        if prov:
            return prov[0]
        return None

    def _bbox_from_item(
        self,
        document: Any,
        item: Any,
    ) -> tuple[float, float, float, float] | None:
        prov = self._first_provenance(item)
        if prov is None:
            return None

        bbox = getattr(prov, "bbox", None)
        if bbox is None:
            return None

        page = self._get_page(document, getattr(prov, "page_no", None))
        page_size = getattr(page, "size", None)
        page_height = getattr(page_size, "height", None)

        try:
            if hasattr(bbox, "to_top_left_origin") and page_height is not None:
                bbox = bbox.to_top_left_origin(page_height=page_height)
        except Exception:
            pass

        try:
            if hasattr(bbox, "normalized") and page_size is not None:
                normalized = bbox.normalized(page_size)
                return (
                    float(normalized.l),
                    float(normalized.t),
                    float(normalized.r),
                    float(normalized.b),
                )
        except Exception:
            pass

        left = getattr(bbox, "l", None)
        top = getattr(bbox, "t", None)
        right = getattr(bbox, "r", None)
        bottom = getattr(bbox, "b", None)
        if None in {left, top, right, bottom}:
            return None
        return (float(left), float(top), float(right), float(bottom))

    def _crop_from_bbox(
        self,
        document: Any,
        item: Any,
        bbox: tuple[float, float, float, float],
    ) -> Any | None:
        prov = self._first_provenance(item)
        page = self._get_page(document, getattr(prov, "page_no", None) if prov else None)
        if page is None:
            return None

        page_image = getattr(getattr(page, "image", None), "pil_image", None)
        if page_image is None:
            return None

        left, top, right, bottom = bbox
        if max(left, top, right, bottom) <= 1.0:
            width, height = page_image.size
            crop_box = (
                int(max(0.0, left) * width),
                int(max(0.0, top) * height),
                int(min(1.0, right) * width),
                int(min(1.0, bottom) * height),
            )
        else:
            crop_box = (
                int(max(0.0, left)),
                int(max(0.0, top)),
                int(max(0.0, right)),
                int(max(0.0, bottom)),
            )

        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            return None
        return page_image.crop(crop_box)

    def _get_page(self, document: Any, page_no: Any) -> Any | None:
        if page_no is None:
            return None
        pages = getattr(document, "pages", None)
        if pages is None:
            return None
        if isinstance(pages, dict):
            return pages.get(page_no) or pages.get(page_no - 1)
        if isinstance(pages, (list, tuple)):
            if isinstance(page_no, int) and 1 <= page_no <= len(pages):
                return pages[page_no - 1]
            if isinstance(page_no, int) and 0 <= page_no < len(pages):
                return pages[page_no]
        return None
