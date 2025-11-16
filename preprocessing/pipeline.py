"""Utilities for aligning multimodal data prior to fine-tuning."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import json
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

from .config import (
    ImageChannelConfig,
    PreprocessingConfig,
    StructuredChannelConfig,
    StructuredFieldConfig,
    TextChannelConfig,
)


class ImageProcessor:
    """Normalizes images to a fixed size and channel statistics."""

    def __init__(self, config: ImageChannelConfig):
        self.config = config
        self.mean = np.array(config.mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(config.std, dtype=np.float32).reshape(3, 1, 1)

    def __call__(self, path: str | Path) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        resized = image.resize(tuple(self.config.size), Image.BICUBIC)
        array = np.asarray(resized).astype(np.float32) / 255.0
        array = array.transpose(2, 0, 1)
        normalized = (array - self.mean) / self.std
        return normalized


class TextProcessor:
    """Tokenizes textual prompts while respecting the config."""

    def __init__(self, config: TextChannelConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True)

    def __call__(self, text: str) -> Dict[str, List[int]]:
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length" if self.config.padding else False,
        )
        return {"input_ids": encoding["input_ids"], "attention_mask": encoding["attention_mask"]}


@dataclass
class StructuredProcessor:
    """Encodes structured information into normalized vectors and prompts."""

    config: StructuredChannelConfig

    def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        vector: List[float] = []
        description_parts: List[str] = []
        for field in self.config.fields:
            value = payload.get(field.name, field.default)
            description_parts.append(f"{field.name}: {value}")
            if field.kind == "continuous":
                vector.extend(self._encode_continuous(field, value))
            elif field.kind == "categorical":
                vector.extend(self._encode_categorical(field, value))
            else:
                raise ValueError(f"Unsupported field kind: {field.kind}")
        return {
            "structured_vector": vector,
            "structured_prompt": "\n".join(description_parts),
        }

    @staticmethod
    def _encode_continuous(field: StructuredFieldConfig, value: Any) -> Sequence[float]:
        if value is None:
            value = field.mean or 0.0
        value = float(value)
        if field.std and field.std != 0:
            return [(value - (field.mean or 0.0)) / field.std]
        return [value]

    @staticmethod
    def _encode_categorical(field: StructuredFieldConfig, value: Any) -> Sequence[float]:
        vocab = field.vocabulary
        vector = [0.0] * len(vocab)
        if value is None and field.default is not None:
            value = field.default
        if value in vocab:
            index = vocab.index(value)
            vector[index] = 1.0
        return vector


class MultimodalPreprocessor:
    """Central orchestrator that aligns every modality into a shared sample."""

    def __init__(self, config: PreprocessingConfig):
        self.image_processor = ImageProcessor(config.image)
        self.text_processor = TextProcessor(config.text)
        self.structured_processor = StructuredProcessor(config.structured)

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image_tensor = self.image_processor(sample["image_path"])
        text_features = self.text_processor(sample["text"])
        structured_features = self.structured_processor(sample.get("structured", {}))
        prompt = self._compose_prompt(sample["text"], structured_features["structured_prompt"])

        return {
            "pixel_values": image_tensor,
            "input_ids": text_features["input_ids"],
            "attention_mask": text_features["attention_mask"],
            **structured_features,
            "prompt": prompt,
        }

    @staticmethod
    def _compose_prompt(text_prompt: str, structured_prompt: str) -> str:
        return (
            "<image>\n"
            "<structured>\n"
            f"{structured_prompt}\n"
            "</structured>\n"
            "<instruction>\n"
            f"{text_prompt}\n"
            "</instruction>"
        )


def load_samples(manifest_path: str | Path) -> Iterable[Dict[str, Any]]:
    manifest = Path(manifest_path)
    with manifest.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def save_numpy(output_path: Path, data: Dict[str, Any]) -> None:
    arrays = {
        "pixel_values": data["pixel_values"].astype(np.float32),
        "input_ids": np.asarray(data["input_ids"], dtype=np.int64),
        "attention_mask": np.asarray(data["attention_mask"], dtype=np.int64),
        "structured_vector": np.asarray(data.get("structured_vector", []), dtype=np.float32),
    }
    np.savez(output_path, **arrays)


__all__ = [
    "ImageProcessor",
    "TextProcessor",
    "StructuredProcessor",
    "MultimodalPreprocessor",
    "load_samples",
    "save_numpy",
]
