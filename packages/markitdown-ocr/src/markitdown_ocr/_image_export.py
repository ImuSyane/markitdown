import mimetypes
import posixpath
import re
from typing import Optional

from markitdown import DocumentAsset


def image_export_enabled(**kwargs) -> bool:
    image_dir = kwargs.get("image_dir")
    return isinstance(image_dir, str) and image_dir.strip() != ""


def create_image_asset(
    *,
    image_bytes: bytes,
    image_dir: str,
    name: str,
    extension: Optional[str] = None,
    mimetype: Optional[str] = None,
) -> DocumentAsset:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "image"
    suffix = _normalize_extension(extension, mimetype)
    relative_dir = image_dir.strip().strip("/").strip("\\")
    relative_path = (
        posixpath.join(relative_dir, f"{stem}{suffix}") if relative_dir else f"{stem}{suffix}"
    )
    return DocumentAsset(path=relative_path, data=image_bytes, mimetype=mimetype)


def render_image_block(
    asset: DocumentAsset,
    *,
    alt_text: str,
    extracted_text: Optional[str] = None,
    label: str = "Image OCR",
) -> str:
    parts = [f"![{sanitize_alt_text(alt_text)}]({asset.path})"]
    if extracted_text and extracted_text.strip():
        parts.append(f"*[{label}]\n{extracted_text.strip()}\n[End OCR]*")
    return "\n\n".join(parts)


def sanitize_alt_text(value: str) -> str:
    alt_text = re.sub(r"[\r\n\[\]]", " ", value or "")
    return re.sub(r"\s+", " ", alt_text).strip() or "image"


def _normalize_extension(
    extension: Optional[str], mimetype: Optional[str]
) -> str:
    if extension:
        return extension if extension.startswith(".") else f".{extension}"
    guessed = mimetypes.guess_extension(mimetype or "")
    return guessed or ".png"
