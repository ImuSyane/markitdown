# SPDX-FileCopyrightText: 2024-present Adam Fourney <adamfo@microsoft.com>
#
# SPDX-License-Identifier: MIT
import argparse
import os
import sys
import codecs
from textwrap import dedent
from importlib.metadata import entry_points
from .__about__ import __version__
from ._markitdown import MarkItDown, StreamInfo, DocumentConverterResult


def main():
    parser = argparse.ArgumentParser(
        description="Convert various file formats to markdown.",
        prog="markitdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage=dedent(
            """
            SYNTAX:

                markitdown <OPTIONAL: FILENAME>
                If FILENAME is empty, markitdown reads from stdin.

            EXAMPLE:

                markitdown example.pdf

                OR

                cat example.pdf | markitdown

                OR

                markitdown < example.pdf

                OR to save to a file use

                markitdown example.pdf -o example.md

                OR

                markitdown example.pdf > example.md
            """
        ).strip(),
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="show the version number and exit",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output file name. If not provided, output is written to stdout.",
    )

    parser.add_argument(
        "-x",
        "--extension",
        help="Provide a hint about the file extension (e.g., when reading from stdin).",
    )

    parser.add_argument(
        "-m",
        "--mime-type",
        help="Provide a hint about the file's MIME type.",
    )

    parser.add_argument(
        "-c",
        "--charset",
        help="Provide a hint about the file's charset (e.g, UTF-8).",
    )

    parser.add_argument(
        "-d",
        "--use-docintel",
        action="store_true",
        help="Use Document Intelligence to extract text instead of offline conversion. Requires a valid Document Intelligence Endpoint.",
    )

    parser.add_argument(
        "-e",
        "--endpoint",
        type=str,
        help="Document Intelligence Endpoint. Required if using Document Intelligence.",
    )

    parser.add_argument(
        "-p",
        "--use-plugins",
        action="store_true",
        help="Use 3rd-party plugins to convert files. Use --list-plugins to see installed plugins.",
    )

    parser.add_argument(
        "--list-plugins",
        action="store_true",
        help="List installed 3rd-party plugins. Plugins are loaded when using the -p or --use-plugin option.",
    )

    parser.add_argument(
        "--keep-data-uris",
        action="store_true",
        help="Keep data URIs (like base64-encoded images) in the output. By default, data URIs are truncated.",
    )

    parser.add_argument(
        "--ocr-backend",
        help="OCR backend for plugins (for example: llm_vision, openai_compatible).",
    )

    parser.add_argument(
        "--ocr-model",
        help="Model name for OCR backends that use vision-capable APIs.",
    )

    parser.add_argument(
        "--ocr-prompt",
        help="Custom OCR extraction prompt for OCR-capable plugins.",
    )

    parser.add_argument(
        "--ocr-base-url",
        help="Base URL for an OpenAI-compatible OCR API provider.",
    )

    parser.add_argument(
        "--ocr-api-key",
        help="API key for an OpenAI-compatible OCR API provider. Defaults to MARKITDOWN_OCR_API_KEY or OPENAI_API_KEY.",
    )

    parser.add_argument(
        "--ocr-lang",
        help="Language hint for local OCR backends such as PaddleOCR.",
    )

    parser.add_argument(
        "--ocr-device",
        help="Device hint for local VLM OCR backends (for example cpu, cuda, cuda:0).",
    )

    parser.add_argument("filename", nargs="?")
    args = parser.parse_args()

    # Parse the extension hint
    extension_hint = args.extension
    if extension_hint is not None:
        extension_hint = extension_hint.strip().lower()
        if len(extension_hint) > 0:
            if not extension_hint.startswith("."):
                extension_hint = "." + extension_hint
        else:
            extension_hint = None

    # Parse the mime type
    mime_type_hint = args.mime_type
    if mime_type_hint is not None:
        mime_type_hint = mime_type_hint.strip()
        if len(mime_type_hint) > 0:
            if mime_type_hint.count("/") != 1:
                _exit_with_error(f"Invalid MIME type: {mime_type_hint}")
        else:
            mime_type_hint = None

    # Parse the charset
    charset_hint = args.charset
    if charset_hint is not None:
        charset_hint = charset_hint.strip()
        if len(charset_hint) > 0:
            try:
                charset_hint = codecs.lookup(charset_hint).name
            except LookupError:
                _exit_with_error(f"Invalid charset: {charset_hint}")
        else:
            charset_hint = None

    stream_info = None
    if (
        extension_hint is not None
        or mime_type_hint is not None
        or charset_hint is not None
    ):
        stream_info = StreamInfo(
            extension=extension_hint, mimetype=mime_type_hint, charset=charset_hint
        )

    if args.list_plugins:
        # List installed plugins, then exit
        print("Installed MarkItDown 3rd-party Plugins:\n")
        plugin_entry_points = list(entry_points(group="markitdown.plugin"))
        if len(plugin_entry_points) == 0:
            print("  * No 3rd-party plugins installed.")
            print(
                "\nFind plugins by searching for the hashtag #markitdown-plugin on GitHub.\n"
            )
        else:
            for entry_point in plugin_entry_points:
                print(f"  * {entry_point.name:<16}\t(package: {entry_point.value})")
            print(
                "\nUse the -p (or --use-plugins) option to enable 3rd-party plugins.\n"
            )
        sys.exit(0)

    markitdown_kwargs = _build_markitdown_kwargs(args)

    if args.use_docintel:
        if args.endpoint is None:
            _exit_with_error(
                "Document Intelligence Endpoint is required when using Document Intelligence."
            )
        elif args.filename is None:
            _exit_with_error("Filename is required when using Document Intelligence.")

        markitdown = MarkItDown(**markitdown_kwargs, docintel_endpoint=args.endpoint)
    else:
        markitdown = MarkItDown(**markitdown_kwargs)

    if args.filename is None:
        result = markitdown.convert_stream(
            sys.stdin.buffer,
            stream_info=stream_info,
            keep_data_uris=args.keep_data_uris,
        )
    else:
        result = markitdown.convert(
            args.filename, stream_info=stream_info, keep_data_uris=args.keep_data_uris
        )

    _handle_output(args, result)


def _handle_output(args, result: DocumentConverterResult):
    """Handle output to stdout or file"""
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result.markdown)
    else:
        # Handle stdout encoding errors more gracefully
        print(
            result.markdown.encode(sys.stdout.encoding, errors="replace").decode(
                sys.stdout.encoding
            )
        )


def _build_markitdown_kwargs(args: argparse.Namespace) -> dict:
    """Build MarkItDown constructor kwargs from CLI arguments and env fallbacks."""
    kwargs = {
        "enable_plugins": args.use_plugins,
    }

    if args.ocr_backend:
        kwargs["ocr_backend"] = args.ocr_backend
    elif os.getenv("MARKITDOWN_OCR_BACKEND"):
        kwargs["ocr_backend"] = os.getenv("MARKITDOWN_OCR_BACKEND")

    if args.ocr_model:
        kwargs["ocr_model"] = args.ocr_model
    elif os.getenv("MARKITDOWN_OCR_MODEL"):
        kwargs["ocr_model"] = os.getenv("MARKITDOWN_OCR_MODEL")

    if args.ocr_prompt:
        kwargs["ocr_prompt"] = args.ocr_prompt
    elif os.getenv("MARKITDOWN_OCR_PROMPT"):
        kwargs["ocr_prompt"] = os.getenv("MARKITDOWN_OCR_PROMPT")

    if args.ocr_base_url:
        kwargs["ocr_base_url"] = args.ocr_base_url
    elif os.getenv("MARKITDOWN_OCR_BASE_URL"):
        kwargs["ocr_base_url"] = os.getenv("MARKITDOWN_OCR_BASE_URL")

    if args.ocr_api_key:
        kwargs["ocr_api_key"] = args.ocr_api_key
    elif os.getenv("MARKITDOWN_OCR_API_KEY"):
        kwargs["ocr_api_key"] = os.getenv("MARKITDOWN_OCR_API_KEY")
    elif os.getenv("OPENAI_API_KEY"):
        kwargs["ocr_api_key"] = os.getenv("OPENAI_API_KEY")

    if args.ocr_lang:
        kwargs["ocr_lang"] = args.ocr_lang
    elif os.getenv("MARKITDOWN_OCR_LANG"):
        kwargs["ocr_lang"] = os.getenv("MARKITDOWN_OCR_LANG")

    if args.ocr_device:
        kwargs["ocr_device"] = args.ocr_device
    elif os.getenv("MARKITDOWN_OCR_DEVICE"):
        kwargs["ocr_device"] = os.getenv("MARKITDOWN_OCR_DEVICE")

    return kwargs


def _exit_with_error(message: str):
    print(message)
    sys.exit(1)


if __name__ == "__main__":
    main()
