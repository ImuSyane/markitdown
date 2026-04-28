# MarkItDown OCR Plugin

OCR plugin for MarkItDown that extracts text from images embedded in PDF, DOCX, PPTX, and XLSX files through an explicit OCR backend.

The plugin currently exposes three OCR backend tracks: a fast local PP-OCRv5 backend, an OpenAI-compatible vision backend, and a PaddleOCR-VL-oriented backend for higher-quality document/LaTeX recovery. It is still intentionally narrow: cloud image captioning stays separate from OCR backend choice, and Paddle local mode auto-probes the current environment with CPU fallback instead of forcing users to choose device strings up front. On Apple Silicon, local PaddleOCR-VL mode also tries to manage a local MLX service path automatically before falling back to CPU.

## Features

- **Enhanced PDF Converter**: Extracts text from images within PDFs, with full-page OCR fallback for scanned documents
- **Optional PDF Layout Prepass**: Uses Docling only as a PDF layout analyzer for scanned pages and large embedded PDF images before OCR runs
- **PDF Crop Artifact Export**: Saves image-like and complex PDF regions as plugin-managed PNG artifacts and writes Markdown image links back into the output
- **Enhanced DOCX Converter**: OCR for images in Word documents
- **Enhanced PPTX Converter**: OCR for images in PowerPoint presentations
- **Enhanced XLSX Converter**: OCR for images in Excel spreadsheets
- **Context Preservation**: Maintains document structure and flow when inserting extracted text

## Installation

```bash
pip install markitdown-ocr
```

To enable the optional PDF layout prepass:

```bash
pip install 'markitdown-ocr[layout]'
```

For the OpenAI-compatible OCR backend, install an OpenAI-compatible client SDK:

```bash
pip install openai
```

For local Paddle backends, install the Paddle runtime in the environment where you plan to run OCR, or point the PaddleOCR-VL backend at a compatible server:

```bash
pip install 'markitdown-ocr[paddle]'
```

The wheel includes bundled PP-OCRv5 fallback models and the `PP-DocLayoutV3` layout detector used by PaddleOCR-VL, so those models do not need a first-run download. The larger `mlx-community/PaddleOCR-VL-1.5-8bit` model is not bundled because it is roughly 1GB; Apple high-quality local mode downloads or reuses it from the Hugging Face cache.

The `layout` extra is intentionally narrow. It adds Docling only as an optional PDF layout analyzer; OCR extraction still happens through the configured OCR backend after regions are ordered.

## Usage

### Command Line

```bash
markitdown --use-plugins document.pdf
```

The CLI can enable the plugin, but it does not currently expose flags for constructing an LLM client. OCR-enabled runs should use the Python API.

### Python API

Choose an OCR backend explicitly. The OpenAI-compatible backend uses the same `llm_client` and `llm_model` path as image descriptions:

```python
from markitdown import MarkItDown
from openai import OpenAI

md = MarkItDown(
    enable_plugins=True,
    llm_client=OpenAI(),
    llm_model="gpt-4o",
)

result = md.convert("document_with_images.pdf")
print(result.text_content)
```

If no OCR backend is configured the plugin still loads, but OCR is skipped and the standard built-in converter handles the document instead.

### Optional PDF layout prepass

PDF layout analysis is PDF-only in the current stage. It does not apply to DOCX, PPTX, or XLSX.

```python
md = MarkItDown(
    enable_plugins=True,
    ocr_backend="paddleocr_vl",
    ocr_mode="local",
    pdf_layout_backend="auto",
    pdf_layout_min_area_ratio=0.20,
    pdf_layout_debug=False,
    ocr_artifact_export=True,
    ocr_artifact_dir="test-output/pdf-artifacts",
    ocr_artifact_markdown_mode="image_and_text",
)
```

The layout options are:

- `pdf_layout_backend="auto"`: if Docling is installed, use it as a prepass for scanned PDF pages and embedded PDF images whose bounding box covers at least `20%` of the page; otherwise keep the existing PDF OCR path with no error
- `pdf_layout_backend="none"`: disable the layout prepass entirely
- `pdf_layout_backend="docling"`: require the Docling prepass when available; if the dependency is missing or initialization fails, the plugin warns and falls back to the existing PDF OCR path

The layout prepass never replaces your OCR backend. It only decides regions and reading order, then sends those regions to the configured OCR backend (`paddleocr`, `openai_compatible`, or `paddleocr_vl`). For math-heavy scanned PDFs, `paddleocr_vl` preserves Markdown and LaTeX structure much better than the faster PP-OCRv5 backend.

The PDF artifact options are:

- `ocr_artifact_export=True`: export PDF crops/full-page fallback renders as PNG files
- `ocr_artifact_dir=None`: default to `markitdown-ocr-artifacts/<source-stem>/` beside the source PDF when a local path exists
- `ocr_artifact_markdown_mode="image_and_text"`: emit Markdown image links first, then the OCR text block

### Custom Prompt

Override the default extraction prompt for specialized documents:

```python
md = MarkItDown(
    enable_plugins=True,
    ocr_backend="openai_compatible",
    llm_client=OpenAI(),
    llm_model="gpt-4o",
    llm_prompt="Extract all text from this image, preserving table structure.",
    ocr_quality="high",
    ocr_debug=True,
)
```

`ocr_debug=True` turns on warning-based diagnostics for OCR requests and failures while you tune a backend.

### PaddleOCR-VL backend

Use the Paddle-oriented backend in either local or server mode. Local mode auto-probes the current runtime and falls back to CPU when no confirmed accelerator path succeeds. On Apple Silicon, local mode first tries to connect to or start a managed local MLX service path automatically. The managed MLX path prefers `mlx-community/PaddleOCR-VL-1.5-8bit`, probes `http://localhost:8111/` first, then increments through `8112` up to `8118`, and only falls back to CPU after those local ports are exhausted or fail runtime validation.

```python
md = MarkItDown(
    enable_plugins=True,
    ocr_backend="paddleocr_vl",
    ocr_mode="local",
    ocr_api_model_name="PaddlePaddle/PaddleOCR-VL-1.5",  # resolves to the MLX 8bit variant on Apple local mode
    ocr_quality="medium",
    ocr_debug=True,
)
```

For a server-backed Paddle runtime:

```python
md = MarkItDown(
    enable_plugins=True,
    ocr_backend="paddleocr_vl",
    ocr_mode="server",
    ocr_server_url="http://localhost:8118/v1",
    ocr_server_backend="mlx-vlm-server",  # or vllm-server on non-Apple paths
    ocr_api_key="local-or-provider-key",
    ocr_api_model_name="PaddleOCR-VL-1.5",
    ocr_quality="high",
)
```

For OpenAI-compatible hosted providers such as SiliconFlow, use the provider's `/v1`
base URL with `ocr_server_backend="vllm-server"` and the provider model name, for
example `PaddlePaddle/PaddleOCR-VL-1.5`.

### Any OpenAI-Compatible Client

Works with any client that follows the OpenAI API:

```python
from openai import AzureOpenAI

md = MarkItDown(
    enable_plugins=True,
    llm_client=AzureOpenAI(
        api_key="...",
        azure_endpoint="https://your-resource.openai.azure.com/",
        api_version="2024-02-01",
    ),
    llm_model="gpt-4o",
)
```

## How It Works

When `MarkItDown(enable_plugins=True, ...)` is called:

1. MarkItDown discovers the plugin via the `markitdown.plugin` entry point group
2. It calls `register_converters()`, forwarding OCR kwargs like `ocr_backend`, `ocr_mode`, `ocr_quality`, `ocr_server_url`, `ocr_server_backend`, `llm_client`, and `llm_model`
3. The plugin creates an OCR backend instance from those kwargs
4. Four OCR-enhanced converters are registered at **priority -1.0** — before the built-in converters at priority 0.0

When a file is converted:

1. The OCR converter accepts the file
2. It extracts embedded images from the document
3. Each image is sent to the configured OCR backend
4. The returned text is inserted inline, preserving document structure
5. If the OCR backend call fails, conversion continues without that image's text

## Supported File Formats

### PDF

- Embedded images are extracted by position (via `page.images` / page XObjects) and OCR'd inline, interleaved with the surrounding text in vertical reading order.
- If the optional layout prepass is enabled and Docling is installed, scanned PDF pages are rendered once, split into reading-order regions, and then merged with OCR output.
- For mixed PDFs, embedded images whose bounding box covers at least `20%` of the page area are first passed through Docling layout analysis, then OCR'd region by region; smaller images keep the existing single-image OCR behavior.
- Image-like and complex PDF regions are exported as PNG artifacts and written back into Markdown as `![OCR region](...)` plus their OCR text. If OCR text is empty, the image link is still kept.
- **Scanned PDFs** (pages with no extractable text) are detected automatically: each page is rendered at 300 DPI, analyzed for layout when available, and sent to the configured OCR backend.
- **Malformed PDFs** that pdfplumber/pdfminer cannot open (e.g. truncated EOF) are retried with PyMuPDF page rendering, so content is still recovered.
- If Docling is unavailable, raises, or returns no usable regions, the converter falls back to the existing full-page or single-image OCR path.

### DOCX

- Images are extracted via document part relationships (`doc.part.rels`).
- OCR is run before the DOCX→HTML→Markdown pipeline executes: placeholder tokens are injected into the HTML so that the markdown converter does not escape the OCR text, and the final placeholders are replaced with cleaned OCR text after conversion.
- Document flow (headings, paragraphs, tables) is fully preserved around the OCR blocks.

### PPTX

- Picture shapes, placeholder shapes with images, and images inside groups are all supported.
- Shapes are processed in top-to-left reading order per slide.
- If an `llm_client` is configured, the converter first tries the shared image-caption helper from core MarkItDown; OCR is used as the fallback when no description is returned.

### XLSX

- Images embedded in worksheets (`sheet._images`) are extracted per sheet.
- Cell position is calculated from the image anchor coordinates (column/row → Excel letter notation).
- Images are listed under a `### Images in this sheet:` section after the sheet's data table, with an `Image anchor: <cell>` label for traceability. They are not interleaved into the table rows.

### Output format

DOCX, PPTX, and XLSX emit cleaned OCR text directly:

```text
<extracted text>
```

For PDF image-like and complex regions, the default output is:

```markdown
![OCR region](artifacts/page-0001-region-0001.png)

<extracted text>
```

## Troubleshooting

### OCR text missing from output

The most likely cause is that no OCR backend is configured, or the selected backend is missing its runtime dependencies. Verify one of these patterns:

```python
from openai import OpenAI
from markitdown import MarkItDown

md = MarkItDown(
    enable_plugins=True,
    ocr_backend="openai_compatible",
    llm_client=OpenAI(),
    llm_model="gpt-4o",
)
```

```python
from markitdown import MarkItDown

md = MarkItDown(
    enable_plugins=True,
    ocr_backend="paddleocr_vl",
    ocr_mode="local",
    ocr_quality="medium",
)
```

### Plugin not loading

Confirm the plugin is installed and discovered:

```bash
markitdown --list-plugins   # should show: ocr
```

### API errors

The plugin emits warnings for OCR request failures and continues conversion. For the OpenAI-compatible backend, check your API key, quota, and that the chosen model supports vision inputs and `image_url` data URIs. For the Paddle backend, check that local mode has a usable runtime or that server mode points at a reachable Paddle service. On Apple Silicon, local mode will also try to auto-manage a local MLX service if the `mlx_vlm.server` command is available, preferring the `PaddleOCR-VL-1.5` MLX 8bit model and walking the local port range from `8111` through `8118`. In `ocr_debug=True` mode, the backend also records probe candidates, selected mode/device, fallback reasons, and managed-local-service diagnostics in OCR metadata.

## Development

### Running Tests

```bash
cd packages/markitdown-ocr
pytest tests/ -v
```

### Building from Source

```bash
git clone https://github.com/microsoft/markitdown.git
cd markitdown/packages/markitdown-ocr
pip install -e .
```

## Contributing

Contributions are welcome! See the [MarkItDown repository](https://github.com/microsoft/markitdown) for guidelines.

## License

MIT — see [LICENSE](LICENSE).

## Changelog

### 0.1.0 (Initial Release)

- LLM Vision OCR for PDF, DOCX, PPTX, XLSX
- Full-page OCR fallback for scanned PDFs
- Context-aware inline text insertion
- Priority-based converter replacement (no code changes required)
