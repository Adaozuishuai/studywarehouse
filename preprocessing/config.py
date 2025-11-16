"""Configuration models for the multimodal preprocessing pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence
import json


@dataclass
class ImageChannelConfig:
    """Configuration for image normalization."""

    size: Sequence[int] = (224, 224)
    mean: Sequence[float] = (0.48145466, 0.4578275, 0.40821073)
    std: Sequence[float] = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class TextChannelConfig:
    """Configuration for tokenizing text inputs."""

    tokenizer_name: str = "Qwen/Qwen-7B"
    max_length: int = 512
    padding: bool = False


@dataclass
class StructuredFieldConfig:
    """Describes how a structured field should be normalized."""

    name: str
    kind: str
    mean: float | None = None
    std: float | None = None
    vocabulary: List[str] = field(default_factory=list)
    default: str | float | int | None = None


@dataclass
class StructuredChannelConfig:
    """Configuration for structured data normalization."""

    fields: List[StructuredFieldConfig]


@dataclass
class PreprocessingConfig:
    """High-level configuration for the preprocessing pipeline."""

    image: ImageChannelConfig = field(default_factory=ImageChannelConfig)
    text: TextChannelConfig = field(default_factory=TextChannelConfig)
    structured: StructuredChannelConfig = field(default_factory=lambda: StructuredChannelConfig(fields=[]))

    @staticmethod
    def from_json(path: str | Path) -> "PreprocessingConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        return PreprocessingConfig(
            image=ImageChannelConfig(**raw.get("image", {})),
            text=TextChannelConfig(**raw.get("text", {})),
            structured=StructuredChannelConfig(
                fields=[StructuredFieldConfig(**item) for item in raw.get("structured", {}).get("fields", [])]
            ),
        )


__all__ = [
    "ImageChannelConfig",
    "TextChannelConfig",
    "StructuredFieldConfig",
    "StructuredChannelConfig",
    "PreprocessingConfig",
]
