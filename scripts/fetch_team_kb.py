"""团队外挂知识库摄取脚本：从开源数据集生成 data/team_kb/*.md。

当前数据源（均为公开开源数据集，出处随文件落盘声明）：
1. NCIFD 面向大模型的民族文化数据集（letsgoLakers/NCIFD, GitHub）
   - NCQA.json：8000 条民族文化问答 → 过滤刺绣/工艺/纺织相关条目
   - NCSI.json：补充信息型数据（原始文件有 JSON 语法瑕疵，容错解析）
2. xusi-zhixiu-site 刺绣知识库（ddccisme/xusi-zhixiu-site, GitHub，若已克隆）

用法：
    # 先克隆数据集到临时目录
    git clone --depth 1 https://github.com/letsgoLakers/NCIFD %TEMP%/ncifd
    python scripts/fetch_team_kb.py --ncifd %TEMP%/ncifd [--xusi %TEMP%/xusi]

产物：data/team_kb/ncifd_刺绣工艺.md（问答式知识条目）+ SOURCES.md（来源声明）
之后重建 RAG 索引：python -m embroidery_agent.agent.team_rag
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

KB_DIR = PROJECT_ROOT / "data" / "team_kb"

EMBROIDERY_RE = re.compile(
    r"刺绣|苏绣|湘绣|蜀绣|粤绣|苗绣|羌绣|顾绣|汴绣|瓯绣|汉绣|绣品|绣花|针法|缂丝|云锦|宋锦|蜀锦|织锦|四大名绣"
)
CRAFT_RE = re.compile(
    r"工艺|纺织|织造|蜡染|扎染|印染|手工艺|服饰|图案|纹样|织物|编织|染织"
)


def load_json_tolerant(path: Path):
    """NCIFD/NCSI.json 存在缺逗号等语法瑕疵，逐条容错解析。"""
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # 去掉数组外壳，按对象边界切片，逐个尝试解析
    raw = raw.strip()
    if raw.startswith("["):
        raw = raw[1:]
    if raw.endswith("]"):
        raw = raw[:-1]
    items = []
    for m in re.finditer(r"\{.*?\}(?=\s*[\[{]|\s*$)", raw, re.DOTALL):
        try:
            items.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
    return items


def collect_ncifd(ncifd_dir: Path) -> list[dict]:
    entries: list[dict] = []
    for name in ("NCQA.json", "NCSI.json"):
        path = ncifd_dir / "NCFID数据集" / name
        if not path.exists():
            # 目录结构可能是直接的
            path = ncifd_dir / name
        if not path.exists():
            print(f"[跳过] 未找到 {name}")
            continue
        data = load_json_tolerant(path)
        print(f"[读取] {path.name}: 容错解析出 {len(data)} 条")
        for item in data:
            if not isinstance(item, dict):
                continue
            instruction = str(item.get("instruction", "")).strip()
            supply = str(item.get("input", "") or "").strip()
            output = str(item.get("output", "")).strip()
            if not instruction or not output:
                continue
            entries.append(
                {"q": instruction, "supply": supply, "a": output, "src": name}
            )
    return entries


def filter_embroidery(entries: list[dict]) -> list[dict]:
    picked = []
    for e in entries:
        text = e["q"] + e["supply"] + e["a"]
        if EMBROIDERY_RE.search(text) or (CRAFT_RE.search(text) and len(e["a"]) > 40):
            picked.append(e)
    return picked


def write_kb(entries: list[dict], xusi_dir: Path | None) -> dict:
    KB_DIR.mkdir(parents=True, exist_ok=True)
    out = KB_DIR / "ncifd_刺绣工艺.md"
    lines = [
        "# 民族刺绣与工艺知识（NCIFD 数据集）",
        "",
        "> 来源：letsgoLakers/NCIFD《面向大模型的民族文化数据集》"
        "（GitHub 开源），经关键词过滤后的刺绣/工艺/纺织条目。",
        "> 问答式知识条目，供团队 RAG 检索使用。",
        "",
    ]
    for i, e in enumerate(entries, 1):
        q = e["supply"] and f"{e['q']}（{e['supply']}）" or e["q"]
        lines += [f"## 条目 {i}", f"问：{q}", "", f"答：{e['a']}", ""]
    out.write_text("\n".join(lines), encoding="utf-8")

    copied_xusi = 0
    if xusi_dir and xusi_dir.exists():
        dest_dir = KB_DIR / "xusi"
        dest_dir.mkdir(exist_ok=True)
        for md in sorted(xusi_dir.rglob("*.md")):
            rel = md.relative_to(xusi_dir)
            # 跳过站点工程文件（README/配置等），只要知识内容目录
            if any(seg in ("docs", "content", "knowledge", "kb", "notes", "Obsidian")
                   for seg in rel.parts) or "绣" in md.read_text(encoding="utf-8", errors="ignore")[:2000]:
                target = dest_dir / rel.name
                if not target.exists():
                    target.write_text(md.read_text(encoding="utf-8", errors="ignore"),
                                      encoding="utf-8")
                    copied_xusi += 1

    sources = KB_DIR / "SOURCES.md"
    sources.write_text(
        "# 团队知识库数据来源声明\n\n"
        "## NCIFD 民族文化数据集\n"
        "- 仓库：https://github.com/letsgoLakers/NCIFD\n"
        "- 说明：面向大模型的民族文化指令数据集（开源），本项目仅按"
        "刺绣/工艺关键词过滤后离线使用，文件：`ncifd_刺绣工艺.md`\n"
        + (
            "\n## 叙思织绣知识库\n"
            "- 仓库：https://github.com/ddccisme/xusi-zhixiu-site\n"
            "- 说明：刺绣藏品知识（Markdown），目录：`xusi/`\n"
            if copied_xusi else ""
        )
        + "\n> 替换为企业私有知识库时，删除上述文件并放入自有 md/txt 即可，"
        "然后运行 `python -m embroidery_agent.agent.team_rag` 重建索引。\n",
        encoding="utf-8",
    )
    return {"file": out.name, "entries": len(entries), "xusi_files": copied_xusi}


def main() -> int:
    parser = argparse.ArgumentParser(description="从开源数据集摄取团队知识库")
    parser.add_argument("--ncifd", required=True, help="NCIFD 仓库本地克隆路径")
    parser.add_argument("--xusi", default="", help="xusi-zhixiu-site 克隆路径（可选）")
    args = parser.parse_args()

    entries = collect_ncifd(Path(args.ncifd))
    if not entries:
        print("[错误] 未解析到任何数据，请检查 --ncifd 路径")
        return 1
    picked = filter_embroidery(entries)
    print(f"[过滤] 刺绣/工艺相关 {len(picked)} / {len(entries)} 条")
    stats = write_kb(picked, Path(args.xusi) if args.xusi else None)
    print(f"[完成] 写入 {stats['file']}（{stats['entries']} 条），"
          f"xusi 附带 {stats['xusi_files']} 个文件")
    print("[下一步] 重建索引: cd 项目根 && "
          "$env:PYTHONPATH='src'; python -m embroidery_agent.agent.team_rag")
    return 0


if __name__ == "__main__":
    sys.exit(main())
