"""基于种子集的安全扩充：仅变换客户问法，不新增任何事实。

原理：对 raw_qa 的每条知识问答，叠加口语化前缀/后缀变体，
帮助微调模型学会识别同一知识点的不同问法（鲁棒性提升）。
约束：
  - 不修改 answer，避免引入事实错误；
  - 变体与原 instruction 去重；
  - 输出到 data/finetune/generated/，必须经人工抽检后方可合入正式集。

用法：
    python scripts/generate_finetune.py --per 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINETUNE_DIR = ROOT / "data" / "finetune"
OUT_DIR = FINETUNE_DIR / "generated"

# 口语化前缀（客服场景真实问法）
PREFIXES = ["", "你好，", "您好，", "请问，", "你好呀，", "打扰一下，", "麻烦问下，"]
# 口语化后缀
SUFFIXES = ["", "呀", "呢", "啊", "吗", "吧"]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def variants(instruction: str, per: int) -> list[str]:
    base = instruction.strip().rstrip("？?！!。.，, ").strip()
    seen = {base}
    out: list[str] = []
    for prefix in PREFIXES:
        for suffix in SUFFIXES:
            v = (prefix + base + suffix).strip()
            if not v or v in seen:
                continue
            seen.add(v)
            out.append(v)
            if len(out) >= per:
                return out
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="问法变体扩充（安全：不改答案）")
    parser.add_argument("--per", type=int, default=2, help="每条生成的变体数量")
    parser.add_argument("--check-only", action="store_true", help="仅校验已生成文件")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = read_jsonl(FINETUNE_DIR / "raw_qa.jsonl")
    out_path = OUT_DIR / "generated_qa.jsonl"

    if not args.check_only:
        generated: list[dict] = []
        skipped = 0
        for rec in src:
            vs = variants(str(rec.get("instruction", "")), args.per)
            if not vs:
                skipped += 1
                continue
            for v in vs:
                generated.append({**rec, "instruction": v})
        with out_path.open("w", encoding="utf-8", newline="\n") as handle:
            for rec in generated:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[生成] {len(generated)} 条变体 → {out_path}（跳过无变体 {skipped} 条）")
        print("[提示] 生成数据仅供扩充候选，必须人工抽检 + validate_finetune.py 通过后合入正式集。")

    # 复用一个轻量自检：所有行可解析 + instruction 与答案非空
    records = read_jsonl(out_path)
    problems = 0
    for i, rec in enumerate(records, 1):
        if not str(rec.get("instruction", "")).strip() or not str(rec.get("answer", "")).strip():
            print(f"  [ERR] {out_path.name}:{i} 内容为空")
            problems += 1
    print(f"[自检] {len(records)} 条，问题 {problems}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
