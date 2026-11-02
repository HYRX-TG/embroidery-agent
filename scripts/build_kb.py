"""构建知识库索引：读取 data/knowledge + data/internal → 切块 → BM25 + FAISS。

用法：
    python scripts/build_kb.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embroidery_agent.agent.router import build_runtime  # noqa: E402


def main() -> None:
    runtime = build_runtime()
    stats = runtime.build_kb()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if not stats["total_chunks"]:
        print("警告：知识库为空，请检查 data/knowledge 与 data/internal 目录", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
