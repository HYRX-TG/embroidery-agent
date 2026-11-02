"""Rerank 精排：本地 bge-reranker（sentence-transformers），可插拔。

8GB 显存方案：bge-reranker-base 约 278M 参数，CPU 即可运行；
若未安装 sentence-transformers 或模型缺失，自动降级为不重排。
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from pathlib import Path
            # 离线优先：本地 HuggingFace 缓存无该模型时不触发在线下载（避免阻塞/外网依赖）
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_dir = cache_dir / f"models--{self.model_name.replace('/', '--')}"
            if not model_dir.exists():
                logger.warning(
                    "Reranker 模型 %s 未在本地缓存，跳过精排（离线模式）。"
                    "如需启用：提前执行 python -c \"from sentence_transformers import CrossEncoder; "
                    "CrossEncoder('%s')\" 完成下载。",
                    self.model_name, self.model_name,
                )
                self._model = False
                return
            import os
            # 强制离线加载：模型已入本地缓存即不再访问 HuggingFace（客户机无外网）
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
        except Exception as exc:  # pragma: no cover
            logger.warning("Reranker 不可用（%s），将跳过精排", exc)
            self._model = False

    def is_ready(self) -> bool:
        self._load()
        return self._model is not False

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str, float]],
        top_k: int = 5,
    ) -> list[tuple[str, str, float]]:
        """输入 [(doc_id, text, base_score)]，输出按重排分数降序的 [(doc_id, text, rerank_score)]。

        返回的 score 保留 base_score 用于展示，排序按 rerank 分。
        """
        if not self.is_ready() or not candidates:
            return candidates
        pairs = [(query, text) for _, text, _ in candidates]
        try:
            scores = self._model.predict(pairs)
        except Exception as exc:  # pragma: no cover
            logger.warning("Rerank 失败（%s），使用原始顺序", exc)
            return candidates
        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [(doc_id, text, base_score) for (doc_id, text, base_score), _ in ranked[:top_k]]
