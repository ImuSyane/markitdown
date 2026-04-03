#!/usr/bin/env python3 -m pytest
import subprocess
from argparse import Namespace

from markitdown import __version__
from markitdown.__main__ import _build_markitdown_kwargs

# This file contains CLI tests that are not directly tested by the FileTestVectors.
# This includes things like help messages, version numbers, and invalid flags.


def test_version() -> None:
    result = subprocess.run(
        ["python", "-m", "markitdown", "--version"], capture_output=True, text=True
    )

    assert result.returncode == 0, f"CLI exited with error: {result.stderr}"
    assert __version__ in result.stdout, f"Version not found in output: {result.stdout}"


def test_invalid_flag() -> None:
    result = subprocess.run(
        ["python", "-m", "markitdown", "--foobar"], capture_output=True, text=True
    )

    assert result.returncode != 0, f"CLI exited with error: {result.stderr}"
    assert (
        "unrecognized arguments" in result.stderr
    ), "Expected 'unrecognized arguments' to appear in STDERR"
    assert "SYNTAX" in result.stderr, "Expected 'SYNTAX' to appear in STDERR"


def test_build_markitdown_kwargs_prefers_cli_values(monkeypatch) -> None:
    monkeypatch.setenv("MARKITDOWN_OCR_MODEL", "env-model")
    args = Namespace(
        use_plugins=True,
        ocr_backend="openai_compatible",
        ocr_model="cli-model",
        ocr_prompt="prompt",
        ocr_base_url="https://ocr.example/v1",
        ocr_api_key="secret",
        ocr_lang="en",
        ocr_device="cpu",
    )

    kwargs = _build_markitdown_kwargs(args)

    assert kwargs == {
        "enable_plugins": True,
        "ocr_backend": "openai_compatible",
        "ocr_model": "cli-model",
        "ocr_prompt": "prompt",
        "ocr_base_url": "https://ocr.example/v1",
        "ocr_api_key": "secret",
        "ocr_lang": "en",
        "ocr_device": "cpu",
    }


def test_build_markitdown_kwargs_uses_environment_defaults(monkeypatch) -> None:
    monkeypatch.setenv("MARKITDOWN_OCR_BACKEND", "openai_compatible")
    monkeypatch.setenv("MARKITDOWN_OCR_MODEL", "glm-ocr")
    monkeypatch.setenv("MARKITDOWN_OCR_PROMPT", "extract everything")
    monkeypatch.setenv("MARKITDOWN_OCR_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("MARKITDOWN_OCR_API_KEY", "env-secret")
    monkeypatch.setenv("MARKITDOWN_OCR_LANG", "ch")
    monkeypatch.setenv("MARKITDOWN_OCR_DEVICE", "cuda:0")

    args = Namespace(
        use_plugins=False,
        ocr_backend=None,
        ocr_model=None,
        ocr_prompt=None,
        ocr_base_url=None,
        ocr_api_key=None,
        ocr_lang=None,
        ocr_device=None,
    )

    kwargs = _build_markitdown_kwargs(args)

    assert kwargs == {
        "enable_plugins": False,
        "ocr_backend": "openai_compatible",
        "ocr_model": "glm-ocr",
        "ocr_prompt": "extract everything",
        "ocr_base_url": "https://provider.example/v1",
        "ocr_api_key": "env-secret",
        "ocr_lang": "ch",
        "ocr_device": "cuda:0",
    }


if __name__ == "__main__":
    """Runs this file's tests from the command line."""
    test_version()
    test_invalid_flag()
    print("All tests passed!")
