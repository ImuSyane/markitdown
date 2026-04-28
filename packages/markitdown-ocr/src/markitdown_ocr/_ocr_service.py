"""
OCR backend layer for MarkItDown.
Provides backend-neutral OCR interfaces plus OpenAI-compatible and Paddle-oriented implementations.
"""

import base64
import atexit
import json
import mimetypes
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from typing import Any, BinaryIO, Literal, Protocol, runtime_checkable

from markitdown import StreamInfo

OCRQuality = Literal["low", "medium", "high"]
OCRMode = Literal["local", "server"]

_DEFAULT_MLX_PADDLEOCR_MODEL = "mlx-community/PaddleOCR-VL-1.5-8bit"
_DEFAULT_PADDLEOCR_DET_MODEL = "PP-OCRv5_mobile_det"
_DEFAULT_PADDLEOCR_REC_MODEL = "PP-OCRv5_server_rec"
_BUNDLED_MODELS_ROOT = os.path.join(os.path.dirname(__file__), "models")
_BUNDLED_PADDLEOCR_ROOT = os.path.join(_BUNDLED_MODELS_ROOT, "paddleocr")
_BUNDLED_PADDLEX_OFFICIAL_MODELS_ROOT = os.path.join(
    _BUNDLED_MODELS_ROOT, "paddlex", "official_models"
)
_MATERIALIZED_PADDLEX_OFFICIAL_MODELS_ROOT = os.path.expanduser(
    "~/.cache/markitdown-ocr/paddlex/official_models"
)
_DEFAULT_MLX_LOCAL_PORT = 8111
_DEFAULT_MLX_LOCAL_PORT_ATTEMPTS = 8
_DEFAULT_HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")
_REQUIRED_MLX_SNAPSHOT_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "modeling_paddleocr_vl.py",
    "processor_config.json",
    "tokenizer.json",
)
_MLX_VLM_HINTS = (
    "paddleocr-vl",
    "vision",
    "llava",
    "pixtral",
    "internvl",
    "minicpm-v",
    "qwen2-vl",
    "qwen2.5-vl",
    "idefics",
    "molmo",
    "ovis",
    "vl-",
    "-vl",
)
_NON_VLM_HINTS = ("tts", "whisper", "asr", "embedding", "embed", "rerank")
_PADDLEX_MODEL_REQUIRED_FILES = (
    "inference.json",
    "inference.yml",
    "inference.pdiparams",
)


@dataclass
class OCRResult:
    """Result from OCR extraction."""

    text: str
    confidence: float | None = None
    backend_used: str | None = None
    error: str | None = None
    metadata: dict[str, Any] | None = None


@runtime_checkable
class OCRBackend(Protocol):
    backend_name: str
    quality: OCRQuality

    def extract_text(
        self,
        image_stream: BinaryIO,
        prompt: str | None = None,
        stream_info: StreamInfo | None = None,
        **kwargs: Any,
    ) -> OCRResult: ...


class BaseOCRBackend:
    backend_name = "ocr"

    def __init__(self, *, quality: OCRQuality = "medium", debug: bool = False) -> None:
        self.quality = quality if quality in {"low", "medium", "high"} else "medium"
        self.debug = debug

    def _warn(self, message: str) -> None:
        warnings.warn(f"markitdown-ocr: {message}", RuntimeWarning, stacklevel=2)

    def _debug(self, message: str) -> None:
        if self.debug:
            self._warn(message)

    def _quality_max_dimension(self) -> int:
        if self.quality == "low":
            return 1280
        if self.quality == "high":
            return 2400
        return 1800

    def _resolve_content_type(
        self, image_stream: BinaryIO, stream_info: StreamInfo | None = None
    ) -> str:
        content_type = stream_info.mimetype if stream_info and stream_info.mimetype else None
        if content_type:
            return content_type

        if stream_info and stream_info.extension:
            guessed, _ = mimetypes.guess_type(f"image{stream_info.extension}")
            if guessed:
                return guessed

        try:
            from PIL import Image

            image_stream.seek(0)
            img = Image.open(image_stream)
            fmt = img.format.lower() if img.format else "png"
            return f"image/{fmt}"
        except Exception:
            return "image/png"
        finally:
            image_stream.seek(0)

    def _resolve_extension(
        self, content_type: str, stream_info: StreamInfo | None = None
    ) -> str:
        if stream_info and stream_info.extension:
            return stream_info.extension
        guessed = mimetypes.guess_extension(content_type) or ".png"
        return guessed

    def _prepare_image_bytes(
        self, image_stream: BinaryIO, stream_info: StreamInfo | None = None
    ) -> tuple[bytes, str, str]:
        content_type = self._resolve_content_type(image_stream, stream_info)
        extension = self._resolve_extension(content_type, stream_info)

        image_stream.seek(0)
        original_bytes = image_stream.read()

        try:
            from PIL import Image
            import io

            image = Image.open(io.BytesIO(original_bytes))
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")

            max_dimension = self._quality_max_dimension()
            if max(image.size) > max_dimension:
                image.thumbnail((max_dimension, max_dimension))

            save_format = image.format or "PNG"
            if save_format.upper() not in {"PNG", "JPEG", "WEBP"}:
                save_format = "PNG"
                content_type = "image/png"
                extension = ".png"
            elif save_format.upper() == "JPEG":
                content_type = "image/jpeg"
                extension = ".jpg"
            elif save_format.upper() == "WEBP":
                content_type = "image/webp"
                extension = ".webp"
            else:
                content_type = "image/png"
                extension = ".png"

            output = io.BytesIO()
            save_kwargs: dict[str, Any] = {}
            if save_format.upper() in {"JPEG", "WEBP"}:
                save_kwargs["quality"] = (
                    70 if self.quality == "low" else 85 if self.quality == "medium" else 95
                )
            image.save(output, format=save_format, **save_kwargs)
            return output.getvalue(), content_type, extension
        except Exception:
            return original_bytes, content_type, extension
        finally:
            image_stream.seek(0)


class OpenAICompatibleVisionOCRBackend(BaseOCRBackend):
    """OCR backend using OpenAI-compatible vision chat completions."""

    backend_name = "openai_compatible"

    def __init__(
        self,
        client: Any,
        model: str,
        default_prompt: str | None = None,
        *,
        quality: OCRQuality = "medium",
        debug: bool = False,
    ) -> None:
        super().__init__(quality=quality, debug=debug)
        self.client = client
        self.model = model
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
        if self.client is None:
            self._warn("OCR requested but no llm_client is configured")
            return OCRResult(
                text="",
                backend_used=self.backend_name,
                error="LLM client not configured",
            )

        try:
            image_bytes, content_type, _ = self._prepare_image_bytes(
                image_stream, stream_info
            )
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            data_uri = f"data:{content_type};base64,{base64_image}"

            actual_prompt = prompt or self.default_prompt
            self._debug(
                f"sending OCR request with backend={self.backend_name}, model={self.model}, quality={self.quality}, content_type={content_type}, bytes={len(image_bytes)}"
            )
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

            text = response.choices[0].message.content
            extracted_text = text.strip() if text else ""
            if not extracted_text:
                self._debug("OCR response returned empty text")
            return OCRResult(
                text=extracted_text,
                backend_used=self.backend_name,
                metadata={"quality": self.quality, "model": self.model},
            )
        except Exception as exc:
            self._warn(f"OCR request failed: {exc}")
            return OCRResult(
                text="",
                backend_used=self.backend_name,
                error=str(exc),
            )
        finally:
            image_stream.seek(0)


class PaddleOCROCRBackend(BaseOCRBackend):
    """Lightweight local OCR backend using PP-OCRv5 detection/recognition models."""

    backend_name = "paddleocr"

    def __init__(
        self,
        *,
        text_detection_model_name: str = _DEFAULT_PADDLEOCR_DET_MODEL,
        text_recognition_model_name: str = _DEFAULT_PADDLEOCR_REC_MODEL,
        backend_options: dict[str, Any] | None = None,
        quality: OCRQuality = "medium",
        debug: bool = False,
    ) -> None:
        super().__init__(quality=quality, debug=debug)
        self.text_detection_model_name = text_detection_model_name
        self.text_recognition_model_name = text_recognition_model_name
        self.backend_options = backend_options or {}
        self._pipeline: Any | None = None
        self._pipeline_error: str | None = None

    def _bundled_model_dir(self, model_name: str) -> str | None:
        model_dir = os.path.join(_BUNDLED_PADDLEOCR_ROOT, model_name)
        required = ("inference.json", "inference.yml", "inference.pdiparams")
        if os.path.isdir(model_dir) and all(
            os.path.isfile(os.path.join(model_dir, filename)) for filename in required
        ):
            return model_dir
        return None

    def _pipeline_kwargs(self) -> dict[str, Any]:
        kwargs = dict(self.backend_options)
        kwargs.setdefault("text_detection_model_name", self.text_detection_model_name)
        kwargs.setdefault("text_recognition_model_name", self.text_recognition_model_name)
        kwargs.setdefault("use_doc_orientation_classify", False)
        kwargs.setdefault("use_doc_unwarping", False)
        kwargs.setdefault("use_textline_orientation", False)

        det_dir = self._bundled_model_dir(self.text_detection_model_name)
        rec_dir = self._bundled_model_dir(self.text_recognition_model_name)
        if det_dir is not None:
            kwargs.setdefault("text_detection_model_dir", det_dir)
        if rec_dir is not None:
            kwargs.setdefault("text_recognition_model_dir", rec_dir)
        return kwargs

    def _get_pipeline(self) -> Any | None:
        if self._pipeline is not None:
            return self._pipeline
        if self._pipeline_error is not None:
            return None

        try:
            from paddleocr import PaddleOCR

            self._pipeline = PaddleOCR(**self._pipeline_kwargs())
            return self._pipeline
        except Exception as exc:
            self._pipeline_error = str(exc)
            self._warn(f"PaddleOCR backend is unavailable: {exc}")
            return None

    def _extract_text_from_result(self, result: Any) -> tuple[list[str], list[float]]:
        json_value = getattr(result, "json", None)
        if isinstance(json_value, dict):
            json_value = json_value.get("res", json_value)
        if not isinstance(json_value, dict) and isinstance(result, dict):
            json_value = result

        texts: list[str] = []
        scores: list[float] = []
        if isinstance(json_value, dict):
            rec_texts = json_value.get("rec_texts") or []
            rec_scores = json_value.get("rec_scores") or []
            if isinstance(rec_texts, (list, tuple)):
                texts = [str(text).strip() for text in rec_texts if str(text).strip()]
            if isinstance(rec_scores, (list, tuple)):
                for score in rec_scores:
                    try:
                        scores.append(float(score))
                    except (TypeError, ValueError):
                        continue
        return texts, scores

    def extract_text(
        self,
        image_stream: BinaryIO,
        prompt: str | None = None,
        stream_info: StreamInfo | None = None,
        **kwargs: Any,
    ) -> OCRResult:
        pipeline = self._get_pipeline()
        if pipeline is None:
            return OCRResult(
                text="",
                backend_used=self.backend_name,
                error=self._pipeline_error or "PaddleOCR backend unavailable",
                metadata=self._build_metadata(),
            )

        temp_path: str | None = None
        try:
            image_bytes, _content_type, extension = self._prepare_image_bytes(
                image_stream, stream_info
            )
            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
                temp_file.write(image_bytes)
                temp_path = temp_file.name

            text_parts: list[str] = []
            scores: list[float] = []
            for result in pipeline.predict(temp_path):
                result_texts, result_scores = self._extract_text_from_result(result)
                text_parts.extend(result_texts)
                scores.extend(result_scores)

            confidence = sum(scores) / len(scores) if scores else None
            return OCRResult(
                text="\n".join(text_parts).strip(),
                confidence=confidence,
                backend_used=self.backend_name,
                metadata=self._build_metadata(),
            )
        except Exception as exc:
            self._warn(f"PaddleOCR request failed: {exc}")
            return OCRResult(
                text="",
                backend_used=self.backend_name,
                error=str(exc),
                metadata=self._build_metadata(),
            )
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def _build_metadata(self) -> dict[str, Any]:
        return {
            "quality": self.quality,
            "text_detection_model_name": self.text_detection_model_name,
            "text_recognition_model_name": self.text_recognition_model_name,
            "text_detection_model_dir": self._bundled_model_dir(
                self.text_detection_model_name
            ),
            "text_recognition_model_dir": self._bundled_model_dir(
                self.text_recognition_model_name
            ),
        }


class PaddleOCRVLOCRBackend(BaseOCRBackend):
    """OCR backend using PaddleOCR-VL through a local pipeline or a server transport."""

    backend_name = "paddleocr_vl"

    def __init__(
        self,
        *,
        mode: OCRMode = "local",
        server_url: str | None = None,
        server_backend: str | None = None,
        api_key: str | None = None,
        api_model_name: str = "PaddlePaddle/PaddleOCR-VL-1.5",
        backend_options: dict[str, Any] | None = None,
        quality: OCRQuality = "medium",
        debug: bool = False,
        legacy_device: str | None = None,
    ) -> None:
        super().__init__(quality=quality, debug=debug)
        self.requested_mode = self._normalize_mode(mode)
        self.server_url = server_url
        self.server_backend = server_backend or self._default_server_backend()
        self.requested_api_model_name = api_model_name
        self.api_model_name = api_model_name
        self.backend_options = backend_options or {}
        self.api_key = api_key or self.backend_options.get("vl_rec_api_key")
        self.legacy_device = self._normalize_legacy_device(legacy_device)

        self.selected_mode: OCRMode = self.requested_mode
        self.selected_device: str | None = None
        self.probe_candidates: list[str] = []
        self.probe_diagnostics: list[str] = []
        self.fallback_reason: str | None = None
        self.runtime_capabilities: dict[str, Any] | None = None

        self._pipeline: Any | None = None
        self._pipeline_error: str | None = None
        self._managed_server_process: subprocess.Popen[str] | None = None
        self._managed_server_log: str | None = None
        self._managed_server_cleanup_registered = False

    def _normalize_mode(self, mode: str | None) -> OCRMode:
        normalized = (mode or "local").strip().lower()
        if normalized in {"server", "remote"}:
            return "server"
        return "local"

    def _normalize_legacy_device(self, device: str | None) -> str | None:
        if not device:
            return None
        normalized = device.strip().lower()
        if normalized in {"cuda", "gpu", "nvidia"}:
            return "gpu"
        if normalized in {"cpu", "x86", "arm"}:
            return "cpu"
        if normalized in {"apple", "mps", "metal", "mac", "macos"}:
            return "mps"
        return normalized

    def _default_server_backend(self) -> str:
        system = platform.system().lower()
        machine = platform.machine().lower()
        if system == "darwin" and machine in {"arm64", "aarch64"}:
            return "mlx-vlm-server"
        return "vllm-server"

    def _default_local_server_url(self) -> str | None:
        if self._default_server_backend() == "mlx-vlm-server":
            return f"http://localhost:{_DEFAULT_MLX_LOCAL_PORT}/"
        return None

    def _local_mlx_server_candidates(self, url: str) -> list[tuple[int, str]]:
        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname or "localhost"
        if hostname not in {"localhost", "127.0.0.1"}:
            return [(parsed.port or _DEFAULT_MLX_LOCAL_PORT, url)]

        base_port = parsed.port or _DEFAULT_MLX_LOCAL_PORT
        path = parsed.path or "/"
        if not path.endswith("/"):
            path = f"{path}/"
        query = f"?{parsed.query}" if parsed.query else ""

        candidates = []
        for offset in range(_DEFAULT_MLX_LOCAL_PORT_ATTEMPTS):
            port = base_port + offset
            candidate_url = f"http://127.0.0.1:{port}{path}{query}"
            candidates.append((port, candidate_url))
        return candidates

    def _normalize_model_id(self, model_name: str | None) -> str:
        if not model_name:
            return ""
        return re.sub(r"[^a-z0-9]+", "", model_name.strip().lower())

    def _resolve_mlx_model_name(self, model_name: str | None = None) -> str:
        requested = model_name or self.api_model_name or self.requested_api_model_name
        normalized = self._normalize_model_id(requested)
        aliases = {
            self._normalize_model_id("PaddlePaddle/PaddleOCR-VL-1.5"): _DEFAULT_MLX_PADDLEOCR_MODEL,
            self._normalize_model_id("PaddleOCR-VL-1.5"): _DEFAULT_MLX_PADDLEOCR_MODEL,
            self._normalize_model_id("PaddleOCR-VL-1.5-8bit"): _DEFAULT_MLX_PADDLEOCR_MODEL,
            self._normalize_model_id(_DEFAULT_MLX_PADDLEOCR_MODEL): _DEFAULT_MLX_PADDLEOCR_MODEL,
        }
        return aliases.get(normalized, requested or _DEFAULT_MLX_PADDLEOCR_MODEL)

    def _mlx_cache_dir(self, model_name: str | None = None) -> str:
        resolved = self._resolve_mlx_model_name(model_name)
        org, _slash, repo = resolved.partition("/")
        safe_name = f"models--{org}--{repo}" if repo else f"models--{resolved}"
        return os.path.join(_DEFAULT_HF_CACHE_DIR, safe_name)

    def _mlx_snapshot_dir(self, revision: str, model_name: str | None = None) -> str:
        return os.path.join(self._mlx_cache_dir(model_name), "snapshots", revision)

    def _mlx_snapshot_complete(self, snapshot_dir: str) -> bool:
        if not snapshot_dir or not os.path.isdir(snapshot_dir):
            return False
        return all(
            os.path.isfile(os.path.join(snapshot_dir, filename))
            for filename in _REQUIRED_MLX_SNAPSHOT_FILES
        )

    def _existing_mlx_snapshot_path(self, model_name: str | None = None) -> str | None:
        cache_dir = self._mlx_cache_dir(model_name)
        refs_main = os.path.join(cache_dir, "refs", "main")
        if os.path.isfile(refs_main):
            with open(refs_main, "r", encoding="utf-8") as handle:
                revision = handle.read().strip()
            snapshot_dir = self._mlx_snapshot_dir(revision, model_name)
            if self._mlx_snapshot_complete(snapshot_dir):
                return snapshot_dir

        snapshots_dir = os.path.join(cache_dir, "snapshots")
        if not os.path.isdir(snapshots_dir):
            return None

        for entry in sorted(os.listdir(snapshots_dir)):
            snapshot_dir = os.path.join(snapshots_dir, entry)
            if self._mlx_snapshot_complete(snapshot_dir):
                return snapshot_dir
        return None

    def _run_curl(self, args: list[str]) -> bytes:
        completed = subprocess.run(
            args,
            check=True,
            capture_output=True,
        )
        return completed.stdout

    def _fetch_hf_model_metadata(self, model_name: str | None = None) -> dict[str, Any]:
        resolved = self._resolve_mlx_model_name(model_name)
        encoded = urllib.parse.quote(resolved, safe="")
        url = f"https://huggingface.co/api/models/{encoded}"

        try:
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=20.0) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            curl = shutil.which("curl")
            if curl:
                try:
                    payload = self._run_curl(
                        [curl, "-L", "--fail", "--silent", "--show-error", url]
                    )
                    return json.loads(payload.decode("utf-8"))
                except Exception:
                    pass
            raise RuntimeError(f"unable to fetch Hugging Face model metadata: {exc}") from exc

    def _download_hf_file(
        self,
        model_name: str,
        filename: str,
        destination: str,
    ) -> None:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        encoded_model = urllib.parse.quote(model_name, safe="/")
        encoded_filename = "/".join(
            urllib.parse.quote(part, safe="") for part in filename.split("/")
        )
        url = (
            f"https://huggingface.co/{encoded_model}/resolve/main/{encoded_filename}"
        )

        curl = shutil.which("curl")
        if curl:
            try:
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
                        destination,
                        url,
                    ],
                    check=True,
                )
                return
            except Exception as exc:
                self._debug(
                    f"curl download failed for {filename}: {type(exc).__name__}: {exc}; falling back to urllib"
                )

        temp_destination = f"{destination}.tmp"
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=60.0) as response:
            with open(temp_destination, "wb") as handle:
                shutil.copyfileobj(response, handle)
        os.replace(temp_destination, destination)

    def _ensure_local_mlx_snapshot(self, model_name: str | None = None) -> str:
        existing = self._existing_mlx_snapshot_path(model_name)
        if existing is not None:
            return existing

        resolved = self._resolve_mlx_model_name(model_name)
        metadata = self._fetch_hf_model_metadata(resolved)
        revision = str(metadata.get("sha") or "").strip()
        siblings = metadata.get("siblings") or []
        if not revision:
            raise RuntimeError(f"mlx model metadata for {resolved} did not include a revision")
        if not isinstance(siblings, list) or not siblings:
            raise RuntimeError(f"mlx model metadata for {resolved} did not include file listings")

        snapshot_dir = self._mlx_snapshot_dir(revision, resolved)
        os.makedirs(snapshot_dir, exist_ok=True)
        refs_dir = os.path.join(self._mlx_cache_dir(resolved), "refs")
        os.makedirs(refs_dir, exist_ok=True)

        filenames = [
            str(item.get("rfilename")).strip()
            for item in siblings
            if isinstance(item, dict) and item.get("rfilename")
        ]
        for filename in filenames:
            destination = os.path.join(snapshot_dir, filename)
            if os.path.isfile(destination):
                continue
            self._download_hf_file(resolved, filename, destination)

        with open(os.path.join(refs_dir, "main"), "w", encoding="utf-8") as handle:
            handle.write(revision)

        if not self._mlx_snapshot_complete(snapshot_dir):
            raise RuntimeError(
                f"mlx model snapshot is still incomplete after download: {snapshot_dir}"
            )
        return snapshot_dir

    def _mlx_models_endpoint(self, url: str) -> str:
        base = url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/models"
        if base.endswith("/v1/models"):
            return base
        return f"{base}/v1/models"

    def _is_visual_model(self, model_id: str) -> bool:
        lowered = model_id.strip().lower()
        if not lowered:
            return False
        if any(hint in lowered for hint in _NON_VLM_HINTS):
            return False
        return any(hint in lowered for hint in _MLX_VLM_HINTS)

    def _fetch_server_models(self, url: str) -> list[str]:
        endpoint = self._mlx_models_endpoint(url)
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(endpoint, headers=headers, method="GET")
        with urllib.request.urlopen(request, timeout=3.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        data = payload.get("data", [])
        return [
            str(item.get("id")).strip()
            for item in data
            if isinstance(item, dict) and item.get("id")
        ]

    def _select_mlx_model_for_server(self, url: str) -> tuple[bool, str]:
        try:
            model_ids = self._fetch_server_models(url)
        except Exception as exc:
            return False, f"unable to query mlx-vlm models: {exc}"

        if not model_ids:
            return False, "mlx-vlm server did not report any models"

        requested_exact = self._resolve_mlx_model_name()
        requested_normalized = self._normalize_model_id(requested_exact)
        for model_id in model_ids:
            if self._normalize_model_id(model_id) == requested_normalized:
                self.api_model_name = model_id
                return True, f"using requested mlx model {model_id}"

        for model_id in model_ids:
            if self._is_visual_model(model_id):
                self.api_model_name = model_id
                return True, f"using detected VLM model {model_id}"

        return False, f"mlx-vlm server models are not visual: {', '.join(model_ids)}"

    def _mlx_server_command(self) -> list[str] | None:
        command = shutil.which("mlx_vlm.server")
        if command:
            return [command]

        try:
            import importlib.util
            import sys

            if importlib.util.find_spec("mlx_vlm.server") is not None:
                return [sys.executable, "-m", "mlx_vlm.server"]
        except Exception:
            return None

        return None

    def _wait_for_local_server(self, url: str, timeout_seconds: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self._managed_server_process is not None:
                status = self._managed_server_process.poll()
                if status is not None:
                    return False
            if self._can_connect_to_local_server(url):
                return True
            time.sleep(0.5)
        return False

    def _stop_managed_local_server(self) -> None:
        if self._managed_server_process is None:
            return

        if self._managed_server_process.poll() is None:
            self._managed_server_process.terminate()
            try:
                self._managed_server_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._managed_server_process.kill()
                self._managed_server_process.wait(timeout=5.0)

        self._managed_server_process = None

    def _register_managed_server_cleanup(self) -> None:
        if self._managed_server_cleanup_registered:
            return
        atexit.register(self.close)
        self._managed_server_cleanup_registered = True

    def close(self) -> None:
        """Stop any MLX server process started by this backend instance."""

        self._stop_managed_local_server()

    def __enter__(self) -> "PaddleOCRVLOCRBackend":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _ensure_local_mlx_server(self, url: str) -> tuple[bool, str]:
        command = self._mlx_server_command()
        last_reason: str | None = None

        for port, candidate_url in self._local_mlx_server_candidates(url):
            if self._can_connect_to_local_server(candidate_url):
                ok, reason = self._select_mlx_model_for_server(candidate_url)
                if ok:
                    self.server_url = candidate_url
                    return True, reason
                last_reason = f"{candidate_url}: {reason}"
                continue

            if not command:
                last_reason = last_reason or "mlx_vlm.server command not found"
                continue

            try:
                snapshot_path = self._ensure_local_mlx_snapshot()
            except Exception as exc:
                last_reason = (
                    f"{candidate_url}: unable to prepare local mlx snapshot: {exc}"
                )
                continue

            self._stop_managed_local_server()
            log_file = tempfile.NamedTemporaryFile(
                prefix="markitdown-ocr-mlx-", suffix=".log", delete=False
            )
            log_file.close()
            self._managed_server_log = log_file.name

            with open(log_file.name, "w", encoding="utf-8") as handle:
                self._managed_server_process = subprocess.Popen(
                    command
                    + [
                        "--port",
                        str(port),
                        "--model",
                        snapshot_path,
                    ],
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                self._register_managed_server_cleanup()

            if self._wait_for_local_server(candidate_url):
                ok, reason = self._select_mlx_model_for_server(candidate_url)
                if ok:
                    self.server_url = candidate_url
                    return True, reason
                last_reason = f"{candidate_url}: {reason}"
            else:
                last_reason = (
                    f"{candidate_url}: managed-local-mlx-server failed to become usable; "
                    f"log={self._managed_server_log}"
                )
            self._stop_managed_local_server()

        if last_reason:
            return False, last_reason
        return False, "mlx_vlm.server command not found"

    def _get_runtime_capabilities(self) -> dict[str, Any]:
        capabilities: dict[str, Any] = {
            "platform": platform.system().lower(),
            "machine": platform.machine().lower(),
            "available_device": None,
            "compiled_with_cuda": False,
            "compiled_with_rocm": False,
            "compiled_with_xpu": False,
            "custom_devices": [],
        }
        try:
            import paddle

            capabilities["available_device"] = paddle.device.get_device()
            capabilities["compiled_with_cuda"] = paddle.device.is_compiled_with_cuda()
            capabilities["compiled_with_rocm"] = getattr(
                paddle.device, "is_compiled_with_rocm", lambda: False
            )()
            capabilities["compiled_with_xpu"] = getattr(
                paddle.device, "is_compiled_with_xpu", lambda: False
            )()
            capabilities["custom_devices"] = getattr(
                paddle.device, "get_all_custom_device_type", lambda: []
            )()
        except Exception as exc:
            capabilities["probe_error"] = str(exc)
        return capabilities

    def _can_connect_to_local_server(self, url: str) -> bool:
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=1.0) as response:
                status = getattr(response, "status", 200)
                return status < 500
        except urllib.error.HTTPError as exc:
            # A reachable server may legitimately reject GET / with 404/405 while
            # still being healthy enough for the PaddleOCR transport.
            return int(getattr(exc, "code", 500)) < 500
        except (urllib.error.URLError, TimeoutError, ValueError):
            return False

    def _build_local_probe_candidates(self) -> list[str]:
        capabilities = self.runtime_capabilities or self._get_runtime_capabilities()
        self.runtime_capabilities = capabilities

        candidates: list[str] = []
        if self.legacy_device:
            candidates.append(self.legacy_device)

        system = str(capabilities.get("platform") or "").lower()
        machine = str(capabilities.get("machine") or "").lower()
        available_device = str(capabilities.get("available_device") or "").lower()
        custom_devices = {
            str(device).lower() for device in capabilities.get("custom_devices") or []
        }

        if system == "darwin" and machine in {"arm64", "aarch64"}:
            candidates.append("apple-local-server")

        if capabilities.get("compiled_with_cuda") or available_device.startswith(
            ("gpu", "cuda")
        ):
            candidates.append("gpu")

        if capabilities.get("compiled_with_xpu") or available_device.startswith("xpu"):
            candidates.append("xpu")

        if {"mps", "apple", "metal"} & custom_devices:
            candidates.append("mps")

        candidates.append("cpu")

        deduped: list[str] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def _instantiate_pipeline(self, **kwargs: Any) -> Any:
        from paddleocr import PaddleOCRVL

        return PaddleOCRVL(**kwargs)

    def _paddlex_model_complete(self, model_dir: str) -> bool:
        return os.path.isdir(model_dir) and all(
            os.path.isfile(os.path.join(model_dir, filename))
            for filename in _PADDLEX_MODEL_REQUIRED_FILES
        )

    def _bundled_paddlex_model_dir(self, model_name: str) -> str | None:
        model_dir = os.path.join(_BUNDLED_PADDLEX_OFFICIAL_MODELS_ROOT, model_name)
        if self._paddlex_model_complete(model_dir):
            return model_dir

        materialized_dir = self._materialize_chunked_paddlex_model(model_name, model_dir)
        if materialized_dir is not None:
            return materialized_dir
        return None

    def _materialize_chunked_paddlex_model(
        self,
        model_name: str,
        bundled_model_dir: str,
    ) -> str | None:
        if not os.path.isdir(bundled_model_dir):
            return None

        param_prefix = "inference.pdiparams.part"
        param_parts = sorted(
            filename
            for filename in os.listdir(bundled_model_dir)
            if filename.startswith(param_prefix)
        )
        if not param_parts:
            return None

        materialized_dir = os.path.join(
            _MATERIALIZED_PADDLEX_OFFICIAL_MODELS_ROOT,
            model_name,
        )
        if self._paddlex_model_complete(materialized_dir):
            return materialized_dir

        os.makedirs(materialized_dir, exist_ok=True)
        for filename in os.listdir(bundled_model_dir):
            source = os.path.join(bundled_model_dir, filename)
            destination = os.path.join(materialized_dir, filename)
            if (
                os.path.isfile(source)
                and not filename.startswith(param_prefix)
                and filename != "inference.pdiparams"
            ):
                shutil.copy2(source, destination)

        temp_params = os.path.join(materialized_dir, "inference.pdiparams.tmp")
        final_params = os.path.join(materialized_dir, "inference.pdiparams")
        with open(temp_params, "wb") as output:
            for filename in param_parts:
                with open(os.path.join(bundled_model_dir, filename), "rb") as input_part:
                    shutil.copyfileobj(input_part, output)
        os.replace(temp_params, final_params)

        if self._paddlex_model_complete(materialized_dir):
            return materialized_dir
        return None

    def _with_bundled_layout_detection_model(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if kwargs.get("layout_detection_model_dir"):
            return kwargs

        bundled_layout_dir = self._bundled_paddlex_model_dir("PP-DocLayoutV3")
        if bundled_layout_dir is None:
            return kwargs

        resolved = dict(kwargs)
        resolved.setdefault("layout_detection_model_name", "PP-DocLayoutV3")
        resolved.setdefault("layout_detection_model_dir", bundled_layout_dir)
        return resolved

    def _record_probe_failure(self, candidate: str, exc: Exception) -> None:
        message = f"{candidate}: {type(exc).__name__}: {exc}"
        self.probe_diagnostics.append(message)

    def _build_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "quality": self.quality,
            "requested_mode": self.requested_mode,
            "selected_mode": self.selected_mode,
            "selected_device": self.selected_device,
            "layout_detection_model_dir": self._bundled_paddlex_model_dir(
                "PP-DocLayoutV3"
            ),
            "probe_candidates": list(self.probe_candidates),
            "probe_diagnostics": list(self.probe_diagnostics),
            "fallback_reason": self.fallback_reason,
        }
        if self.runtime_capabilities is not None:
            metadata["runtime_capabilities"] = self.runtime_capabilities
        if self.server_url:
            metadata["server_url"] = self.server_url
            metadata["server_backend"] = self.server_backend
            metadata["api_model_name"] = self.api_model_name
            metadata["requested_api_model_name"] = self.requested_api_model_name
        return metadata

    def _get_local_pipeline(self, candidates: list[str], kwargs: dict[str, Any]) -> Any | None:
        for candidate in candidates:
            candidate_kwargs = dict(kwargs)
            if candidate == "apple-local-server":
                server_url = self.server_url or self._default_local_server_url()
                ok, reason = self._ensure_local_mlx_server(server_url or "")
                if not ok:
                    self._record_probe_failure(candidate, RuntimeError(reason))
                    self.fallback_reason = (
                        "local Apple acceleration probe could not start or reach the MLX server; trying next candidate"
                    )
                    continue
                candidate_kwargs.setdefault("vl_rec_backend", "mlx-vlm-server")
                candidate_kwargs.setdefault("vl_rec_server_url", self.server_url or server_url)
                candidate_kwargs.setdefault("vl_rec_api_model_name", self.api_model_name)
            else:
                candidate_kwargs.setdefault("device", candidate)

            try:
                self._pipeline = self._instantiate_pipeline(**candidate_kwargs)
                self.selected_device = candidate
                if candidate != self.probe_candidates[0]:
                    self.fallback_reason = (
                        f"local probe fell back from {self.probe_candidates[0]} to {candidate}"
                    )
                return self._pipeline
            except Exception as exc:
                self._record_probe_failure(candidate, exc)
                if candidate != "cpu":
                    self.fallback_reason = (
                        f"local probe failed for {candidate}; trying next candidate"
                    )

        self.selected_device = "cpu" if "cpu" in self.probe_candidates else None
        self._pipeline_error = self.probe_diagnostics[-1] if self.probe_diagnostics else (
            "PaddleOCR-VL local backend unavailable"
        )
        self._warn(
            "PaddleOCR-VL local mode could not find a usable accelerator and did not complete initialization. "
            f"Diagnostics: {self._pipeline_error}"
        )
        return None

    def _retry_local_runtime_fallback(self, exc: Exception) -> bool:
        if self.requested_mode != "local":
            return False
        if not self.probe_candidates or not self.selected_device:
            return False
        if self.selected_device == "cpu":
            return False

        try:
            current_index = self.probe_candidates.index(self.selected_device)
        except ValueError:
            return False

        remaining_candidates = self.probe_candidates[current_index + 1 :]
        if not remaining_candidates:
            return False

        failed_candidate = self.selected_device
        self._record_probe_failure(failed_candidate, exc)
        self._pipeline = None
        self._pipeline_error = None
        self.selected_device = None
        if failed_candidate == "apple-local-server":
            self._stop_managed_local_server()
        self.fallback_reason = (
            f"local runtime failed for {failed_candidate}; retrying {remaining_candidates[0]}"
        )

        return self._get_local_pipeline(remaining_candidates, dict(self.backend_options)) is not None

    def _get_pipeline(self) -> Any | None:
        if self._pipeline is not None:
            return self._pipeline
        if self._pipeline_error is not None:
            return None

        kwargs = self._with_bundled_layout_detection_model(dict(self.backend_options))

        if self.requested_mode == "server":
            self.selected_mode = "server"
            self.selected_device = "server"
            if self.server_backend == "mlx-vlm-server" and self.server_url:
                ok, reason = self._select_mlx_model_for_server(self.server_url)
                if not ok:
                    self._pipeline_error = reason
                    self._warn(
                        "PaddleOCR-VL server mode found an mlx-vlm server, but it did not expose a usable vision model. "
                        f"Original error: {reason}"
                    )
                    return None
            kwargs.setdefault("vl_rec_backend", self.server_backend)
            kwargs.setdefault("vl_rec_server_url", self.server_url)
            kwargs.setdefault("vl_rec_api_model_name", self.api_model_name)
            if self.api_key:
                kwargs.setdefault("vl_rec_api_key", self.api_key)
            try:
                self._pipeline = self._instantiate_pipeline(**kwargs)
                return self._pipeline
            except Exception as exc:
                self._pipeline_error = str(exc)
                self._warn(
                    "PaddleOCR-VL server mode is unavailable. Check the configured server URL/backend. "
                    f"Original error: {exc}"
                )
                return None

        self.selected_mode = "local"
        self.probe_candidates = self._build_local_probe_candidates()
        return self._get_local_pipeline(self.probe_candidates, kwargs)

    def _extract_text_from_paddle_result(self, result: Any) -> str:
        markdown = getattr(result, "markdown", None)
        if isinstance(markdown, str) and markdown.strip():
            return markdown.strip()

        if isinstance(markdown, dict):
            markdown_texts = markdown.get("markdown_texts")
            if isinstance(markdown_texts, str) and markdown_texts.strip():
                return markdown_texts.strip()

        json_value = getattr(result, "json", None)
        collected: list[str] = []

        def visit(value: Any) -> None:
            if isinstance(value, str):
                stripped = value.strip()
                if stripped and not stripped.startswith("/"):
                    collected.append(stripped)
            elif isinstance(value, dict):
                for nested in value.values():
                    visit(nested)
            elif isinstance(value, (list, tuple)):
                for nested in value:
                    visit(nested)

        visit(json_value)
        return "\n".join(collected).strip()

    def extract_text(
        self,
        image_stream: BinaryIO,
        prompt: str | None = None,
        stream_info: StreamInfo | None = None,
        **kwargs: Any,
    ) -> OCRResult:
        pipeline = self._get_pipeline()
        if pipeline is None:
            return OCRResult(
                text="",
                backend_used=self.backend_name,
                error=self._pipeline_error or "PaddleOCR-VL backend unavailable",
                metadata=self._build_metadata(),
            )

        temp_path: str | None = None
        try:
            image_bytes, content_type, extension = self._prepare_image_bytes(
                image_stream, stream_info
            )
            self._debug(
                f"running Paddle OCR with mode={self.selected_mode}, device={self.selected_device}, quality={self.quality}, content_type={content_type}, bytes={len(image_bytes)}"
            )

            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
                temp_file.write(image_bytes)
                temp_path = temp_file.name

            results = list(pipeline.predict(temp_path))
            text_parts = []
            for result in results:
                text = self._extract_text_from_paddle_result(result)
                if text:
                    text_parts.append(text)
            extracted_text = "\n\n".join(text_parts).strip()
            if not extracted_text:
                self._debug("Paddle OCR response returned empty text")
            return OCRResult(
                text=extracted_text,
                backend_used=self.backend_name,
                metadata=self._build_metadata(),
            )
        except Exception as exc:
            if self._retry_local_runtime_fallback(exc):
                image_stream.seek(0)
                return self.extract_text(
                    image_stream,
                    prompt=prompt,
                    stream_info=stream_info,
                    **kwargs,
                )
            self._warn(f"Paddle OCR request failed: {exc}")
            return OCRResult(
                text="",
                backend_used=self.backend_name,
                error=str(exc),
                metadata=self._build_metadata(),
            )
        finally:
            image_stream.seek(0)
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)


LLMVisionOCRService = OpenAICompatibleVisionOCRBackend
