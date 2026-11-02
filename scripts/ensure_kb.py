"""确保知识库索引为最新：源文件变更自动重建（幂等，可安全重复执行）。

用法：
    python scripts/ensure_kb.py

输出 JSON：{total_chunks, vector_loaded, rebuilt, ...}；
知识库为空时返回非零退出码。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embroidery_agent.agent.router import build_runtime  # noqa: E402


def main() -> int:
    runtime = build_runtime()
    result = runtime.kb.load()  # stale 时内部自动重建
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("total_chunks"):
        print("警告：知识库为空，请检查 data/knowledge 与 data/internal 目录", file=sys.stderr)
        return 1
    if result.get("rebuilt"):
        print("[提示] 检测到知识库变更，索引已自动重建。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
