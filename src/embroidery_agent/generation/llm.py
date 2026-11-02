"""本地 LLM 客户端：通过 OpenAI 兼容接口调用 Ollama。

关键设计（参考 private-gpt 的解耦范式）：
- 只面向 OpenAI 兼容 API 编程，换推理引擎（Ollama/llama.cpp/vLLM）不改业务代码。
- 支持流式输出（chat completions stream）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import httpx


@dataclass(frozen=True)
class LLMChunk:
    text: str
    kind: str = "content"  # content / reasoning
    done: bool = False


class LLMClient:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def is_ready(self) -> bool:
        """探测服务可用且目标模型已加载（仅服务在线但模型缺失时返回 False）。"""
        try:
            response = httpx.get(f"{self.base_url}/models", timeout=5)
            if response.status_code != 200:
                return False
            data = response.json()
            # OpenAI 兼容格式 {"data":[{"id":...}]}；Ollama 原生 {"models":[{"name":...}]}
            model_ids = [m.get("id", "") for m in data.get("data", [])]
            if not model_ids:
                model_ids = [m.get("name", "") for m in data.get("models", [])]
            if not model_ids:
                return False
            base = self.model.split(":")[0].lower()
            for mid in model_ids:
                low = str(mid).lower()
                if low == self.model.lower() or low.split(":")[0] == base:
                    return True
            return False
        except Exception:
            return False

    def missing_reason(self) -> str:
        """诊断信息：服务不可用 or 模型未安装（用于降级告警与状态页）。"""
        try:
            response = httpx.get(f"{self.base_url}/models", timeout=5)
            if response.status_code != 200:
                return f"Ollama 服务不可达（HTTP {response.status_code}）"
            data = response.json()
            model_ids = [m.get("id", "") or m.get("name", "") for m in (data.get("data") or data.get("models") or [])]
            if self.model.lower() not in [str(m).lower() for m in model_ids]:
                return f"模型 {self.model} 未安装，可用模型：{model_ids}"
            return ""
        except Exception as exc:
            return f"Ollama 连接失败：{exc}"

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        response_format: str | None = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if response_format:
            payload["response_format"] = {"type": response_format}
        response = httpx.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def stream_chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> Iterable[LLMChunk]:
        """流式聊天，兼容 OpenAI 兼容的 SSE 格式。"""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        with httpx.stream("POST", url, json=payload, timeout=self.timeout) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or ""
                if reasoning:
                    yield LLMChunk(reasoning, kind="reasoning")
                if text:
                    yield LLMChunk(text, kind="content")
        yield LLMChunk("", done=True)
