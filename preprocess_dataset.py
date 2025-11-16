"""Command line interface for the multimodal preprocessing pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from preprocessing.config import PreprocessingConfig
from preprocessing.pipeline import MultimodalPreprocessor, load_samples, save_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal preprocessing orchestrator")
    parser.add_argument("manifest", type=str, help="Path to a JSONL manifest with image/text/structured fields")
    parser.add_argument("config", type=str, help="JSON file describing preprocessing hyperparameters")
    parser.add_argument("output", type=str, help="Directory where processed artifacts will be written")
    parser.add_argument("--metadata-name", default="metadata.jsonl", help="Filename for prompt/structured metadata")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PreprocessingConfig.from_json(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = MultimodalPreprocessor(config)
    metadata_path = output_dir / args.metadata_name
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for index, sample in enumerate(load_samples(args.manifest)):
            processed = preprocessor.process_sample(sample)
            output_file = output_dir / f"sample_{index:06d}.npz"
            save_numpy(output_file, processed)
            metadata = {
                "id": index,
                "prompt": processed["prompt"],
                "structured_prompt": processed["structured_prompt"],
                "image_path": sample["image_path"],
                "text": sample["text"],
            }
            metadata_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
