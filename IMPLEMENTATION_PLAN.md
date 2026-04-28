## Stage 1: Align the CLI contract and user-facing OCR usage
**Goal**: Make the documented OCR usage match the actual supported entrypoints, and narrow the stated scope away from universal VLM compatibility.
**Success Criteria**: Root docs and OCR package docs no longer advertise unsupported CLI flags or a generic OCR/VLM compatibility layer; supported Python API and plugin usage are documented consistently.
**Tests**: Audit updated docs against the actual CLI help and constructor signatures; verify every documented OCR invocation maps to a real code path.
**Status**: Complete

## Stage 2: Introduce explicit OCR backend selection
**Goal**: Split OCR backend selection from cloud image captioning and add a backend-neutral OCR interface for OpenAI-compatible and Paddle-oriented backends.
**Success Criteria**: The plugin can select an OCR backend through explicit configuration; converters depend on a backend-neutral OCR interface instead of a hard-wired OpenAI-compatible service; cloud captioning remains separate from OCR backend choice.
**Tests**: Add targeted tests for backend selection, missing-backend behavior, and compatibility with the existing OpenAI-compatible OCR path.
**Status**: Complete

## Stage 3: Add Paddle-oriented local OCR configuration and PDF layout prepass
**Goal**: Add a PaddleOCR-VL-oriented backend with device-aware local/server configuration and practical low/medium/high quality profiles, then complete the PDF-only optional layout prepass for scanned pages and large embedded PDF images.
**Success Criteria**: The plugin can configure a Paddle backend for CPU, Apple-friendly paths, or CUDA-capable environments through explicit device/server settings; Apple local MLX probing prefers a predictable incremental local port window before falling back to CPU; quality maps to inference/preprocessing profiles instead of pretending there are official model-size tiers; PDF conversion can optionally use a thin Docling-backed layout prepass without changing OCR backend semantics or touching non-PDF converters; PDF image-like and complex regions can be exported as plugin-managed image artifacts and written back into Markdown as image links plus OCR text.
**Tests**: Add tests for Paddle backend configuration, quality profile mapping, graceful fallback when Paddle dependencies are unavailable, PDF layout option resolution, scanned-page layout prepass, large-image threshold routing, artifact export, image+text Markdown output, and fallback to the existing PDF OCR path when Docling is unavailable or fails.
**Status**: Complete

## Stage 4: Document backend choices and operating modes
**Goal**: Update docs and examples so backend selection, quality levels, platform/device expectations, and the optional PDF layout path are clear.
**Success Criteria**: Root and package docs describe OpenAI-compatible OCR versus Paddle-oriented OCR, note that local Paddle support is optional, explicitly mention CPU/Apple/CUDA considerations, document `pdf_layout_backend` / `pdf_layout_min_area_ratio` / `pdf_layout_debug`, document `ocr_artifact_export` / `ocr_artifact_dir` / `ocr_artifact_markdown_mode`, and explain that Docling is PDF-only and used only as a layout prepass.
**Tests**: Review examples against real constructor kwargs and ensure every documented backend option maps to implemented code paths and the same default/fallback story appears in the root README, plugin README, and this plan.
**Status**: Complete
