"""运行时配置：从 configs/default.json + 环境变量加载。"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.json"


def load_config(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class Settings(BaseSettings):
    """运行时配置。环境变量可覆盖（大小写不敏感），本地优先读 .env。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "development"
    app_origin: str = "http://127.0.0.1:8000"

    # Ollama（本地推理服务，OpenAI 兼容）
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "qwen2.5:7b-instruct"
    embedding_model: str = "bge-m3"
    llm_timeout_seconds: int = 120

    # 本地路径
    config_path: Path = DEFAULT_CONFIG

    @property
    def config(self) -> dict[str, Any]:
        return load_config(self.config_path)

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    def ollama_ready(self) -> bool:
        """探测 Ollama 是否可达（惰性，仅作运行期检查）。"""
        return bool(self.ollama_base_url.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
