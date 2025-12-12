"""
Configuration helpers for the intelligent assistant service.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Settings:
    """Runtime configuration sourced from environment variables."""

    openai_api_key: Optional[str] = None
    langsmith_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    huggingface_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    source_urls: List[str] = field(
        default_factory=lambda: [
            "https://zh.wikipedia.org/wiki/%E5%A4%A7%E7%86%8A%E7%8C%AB",  # 大熊猫
            "https://zh.wikipedia.org/wiki/%E7%AB%B9",  # 竹类
            "https://zh.wikipedia.org/wiki/%E7%86%8A%E7%8C%AB%E4%B8%83%E5%9B%BD%E8%B5%A0%E9%80%81",  # 熊猫外交
            "https://zh.wikipedia.org/wiki/%E4%B8%AD%E5%9B%BD%E7%86%8A%E7%8C%AB%E5%8E%9F%E5%9C%B0",  # 熊猫栖息地与保护
        ]
    )
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chat_model: str = "gpt-5-nano"
    temperature: float = 0.0
    tavily_results: int = 3
    default_thread_ids: List[str] = field(default_factory=lambda: ["customer-care", "ops-monitoring"])

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings using environment variables with sensible defaults."""

        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            huggingface_model=os.getenv(
                "HUGGINGFACE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            source_urls=cls._parse_source_urls(),
            chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
            chat_model=os.getenv("CHAT_MODEL", "gpt-5-nano"),
            temperature=float(os.getenv("TEMPERATURE", 0.0)),
            tavily_results=int(os.getenv("TAVILY_RESULTS", 3)),
        )

    def validate(self) -> None:
        """Validate that required keys are present."""

        missing: list[str] = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")

        if missing:
            raise RuntimeError(
                "Missing required environment variables: " + ", ".join(sorted(missing))
            )

    @classmethod
    def _parse_source_urls(cls) -> List[str]:
        """Support comma-separated SOURCE_URLS or a single SOURCE_URL fallback."""

        urls = os.getenv("SOURCE_URLS")
        if urls:
            parsed = [url.strip() for url in urls.split(",") if url.strip()]
            if parsed:
                return parsed

        single = os.getenv("SOURCE_URL")
        if single:
            return [single]

        return cls().source_urls


settings = Settings.from_env()
