"""数据飞轮：把真实客服对话导出为微调候选数据（人工审核后并入训练集）。

从 SQLite 的 customer_messages 表提取 (用户问 → 助手答) 对，
过滤固定话术与占位符，生成 data/finetune/candidates.jsonl 供人工审核：
  - reviewed=false 的记录不应直接进训练集；
  - 审核通过后把行移入 raw_qa.jsonl（去掉 reviewed 字段）再执行 export_finetune.py。

用法：
    python scripts/export_conversations.py                  # 导出全部候选
    python scripts/export_conversations.py --min-q 8 --min-a 20   # 自定义长度门槛
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "embroidery.db"
OUT_PATH = ROOT / "data" / "finetune" / "candidates.jsonl"

# 固定话术/占位符：不构成有训练价值的教学信号
EXCLUDED_ASSISTANT_PREFIXES = (
    "您好，该内容为门店专属业务信息",
    "您好呀~我是知绣坊的刺绣顾问",
    "非常理解您的心情",
    "（已生成回答）",
    "抱歉呀，我只能为您解答刺绣",
)


def load_pairs(min_q: int, min_a: int) -> list[dict]:
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT session_id, role, content, created_at FROM customer_messages ORDER BY session_id, id"
    ).fetchall()
    conn.close()

    pairs: list[dict] = []
    pending_user: dict | None = None
    for row in rows:
        content = (row["content"] or "").strip()
        if not content:
            continue
        if row["role"] == "user":
            pending_user = {"content": content, "session_id": row["session_id"], "created_at": row["created_at"]}
        elif row["role"] == "assistant" and pending_user is not None:
            user_text = pending_user["content"]
            ok = (
                len(user_text) >= min_q
                and len(content) >= min_a
                and not content.startswith(EXCLUDED_ASSISTANT_PREFIXES)
                and not user_text.startswith(EXCLUDED_ASSISTANT_PREFIXES)
            )
            if ok:
                pairs.append({
                    "instruction": user_text,
                    "output": content,
                    "session_id": pending_user["session_id"],
                    "created_at": pending_user["created_at"],
                    "reviewed": False,
                })
            pending_user = None
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-q", type=int, default=8, help="用户问题最小长度")
    parser.add_argument("--min-a", type=int, default=20, help="助手回答最小长度")
    args = parser.parse_args()

    pairs = load_pairs(args.min_q, args.min_a)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"已导出 {len(pairs)} 条候选 → {OUT_PATH}")
    if pairs:
        print("下一步：人工逐条审核（删掉低质量/含隐私的行，通过后把 reviewed 改为 true），")
        print("再把这些行合并进 data/finetune/raw_qa.jsonl 并执行 scripts/export_finetune.py。")
    else:
        print("暂无可导出的真实对话（系统上线积累后再运行）。")


if __name__ == "__main__":
    main()
