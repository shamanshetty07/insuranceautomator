"""Local Qwen-VL inference with Hugging Face transformers (optional `vlm` extra)."""

import hashlib
import hmac
import importlib
import importlib.metadata
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.services.types import ModelInfo
from app.services.vlm.runner import VlmRequest

ALLOWED_MODEL_TYPES = frozenset({"qwen2_vl", "qwen2_5_vl"})
_PICKLE_SUFFIXES = frozenset({".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"})
_DTYPES = frozenset({"bfloat16", "float16", "float32"})


class RunnerRefused(RuntimeError):
    """A configured safeguard failed; the model was not loaded."""


def _model_files(model_dir: Path) -> list[Path]:
    files = []
    for path in sorted(model_dir.rglob("*")):
        rel = path.relative_to(model_dir)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if path.is_file():
            files.append(path)
    return files


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_dir_digest(model_dir: Path) -> str:
    """SHA-256 over the sorted manifest "relative/path<TAB>file-sha256" of every model file."""
    manifest = hashlib.sha256()
    for path in _model_files(model_dir):
        rel = path.relative_to(model_dir).as_posix()
        manifest.update(f"{rel}\t{_file_sha256(path)}\n".encode())
    return manifest.hexdigest()


def inspect_model_dir(model_dir: Path) -> str:
    """Return the model_type after the static checks; raise RunnerRefused otherwise."""
    if not model_dir.is_dir():
        raise RunnerRefused("VLM_MODEL_PATH is not a directory")
    files = _model_files(model_dir)
    if any(f.suffix.lower() in _PICKLE_SUFFIXES for f in files):
        raise RunnerRefused("the model directory contains pickle-format weights; only safetensors are accepted")
    if not any(f.suffix == ".safetensors" for f in files):
        raise RunnerRefused("the model directory contains no .safetensors weights")
    try:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RunnerRefused("the model directory has no readable config.json") from None
    model_type = config.get("model_type") if isinstance(config, dict) else None
    if model_type not in ALLOWED_MODEL_TYPES:
        raise RunnerRefused(f"unsupported model_type {model_type!r}")
    return model_type


def _installed(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


class TransformersQwenRunner:
    """VlmRunner for a local Qwen2-VL / Qwen2.5-VL snapshot directory."""

    def __init__(
        self,
        model_dir: Path,
        model_name: str,
        model_version: str,
        *,
        device: str = "cpu",
        dtype: str = "bfloat16",
        expected_sha256: str | None = None,
    ) -> None:
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
        inspect_model_dir(model_dir)
        digest = model_dir_digest(model_dir)
        if expected_sha256 is not None and not hmac.compare_digest(digest, expected_sha256):
            raise RunnerRefused("the model directory does not match VLM_WEIGHTS_SHA256")

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")

        self._torch = torch
        self._device = device
        self._processor = transformers.AutoProcessor.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False
        )
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=getattr(torch, dtype),
        )
        model.to(device)
        model.eval()
        self._model = model
        self.info = ModelInfo(name=model_name, version=model_version, weights_sha256=digest)
        self.runtime: Mapping[str, str | None] = {
            "transformers": _installed("transformers"),
            "torch": _installed("torch"),
        }

    def generate(self, request: VlmRequest) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": request.system}]},
            {
                "role": "user",
                "content": [*({"type": "image"} for _ in request.images), {"type": "text", "text": request.user_text}],
            },
        ]
        prompt = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(
            text=[prompt], images=list(request.images) or None, return_tensors="pt"
        ).to(self._device)
        with self._torch.inference_mode():
            output = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=request.max_new_tokens,
                max_time=request.max_time_seconds,
            )
        new_tokens = output[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
