"""提示词外置仓库：prompts/team/*.md 按需热加载（mtime 缓存）。

文件格式：单个提示词一个文件；工具专家的文件按二级标题分节：

    # 标题（任意）
    ## system       ← 系统提示词（必需）
    ## params       ← 参数提取提示词（工具专家）
    ## missing      ← 补参话术（工具专家）

文件缺失或无对应小节时回退到 prompts_team.py 中的内置常量，
保证项目在任何情况下可运行；业务侧改 md 文件即生效，无需重启。
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

logger = logging.getLogger("embroidery_agent.prompt_store")

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts" / "team"

_cache: dict[str, tuple[float, dict[str, str]]] = {}
_lock = threading.Lock()


def _parse_sections(raw: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = "_head"
    for line in raw.splitlines():
        if line.startswith("## "):
            current = line[3:].strip().lower()
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if k != "_head"}


def load_prompt(name: str, default: str = "", section: str = "system") -> str:
    """读取 prompts/team/{name}.md 的指定小节；文件不存在/无小节时返回 default。"""
    path = PROMPTS_DIR / f"{name}.md"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return default

    with _lock:
        cached = _cache.get(name)
        if cached and cached[0] == mtime:
            return cached[1].get(section) or default

    sections = _parse_sections(path.read_text(encoding="utf-8"))
    with _lock:
        _cache[name] = (mtime, sections)
    return sections.get(section) or default
