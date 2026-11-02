"""导出 LlamaFactory 微调数据（alpaca 格式）。

从 data/finetune/raw_qa.jsonl（Q/A 对）与 raw_dialogue.jsonl（多轮对话）出发，
生成 LlamaFactory 可用的 data/dataset.jsonl（alpaca 指令格式）。

微调目标：让基座模型学习"刺绣客服话术 + 需求表单结构化输出 + 双角色行为边界"。

用法：
    python scripts/export_finetune.py --output data/dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINETUNE_DIR = ROOT / "data" / "finetune"
SYSTEM_CUSTOMER = (
    "你是刺绣企业「知绣坊」的专属智能客服与刺绣方案顾问，风格温柔、专业、国风雅致。"
    "行业知识基于知识库回答；门店价格、工期等专属信息一律不编造，转人工；绝不私自报价。"
)
SYSTEM_STAFF = (
    "你是刺绣企业「知绣坊」的内部业务辅助助手，服务公司员工。"
    "可输出工艺、报价逻辑、工序规范等内部知识；输出简洁高效、可直接使用。"
)


def load_jsonl(path: Path) -> list[dict]:
    """解析 JSONL（兼容单行对象与跨行对象：按大括号配对切块）。"""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    records: list[dict] = []
    depth = 0
    buffer = ""
    for char in text:
        if char == "{":
            depth += 1
        if depth > 0:
            buffer += char
        if char == "}":
            depth -= 1
            if depth == 0:
                try:
                    records.append(json.loads(buffer))
                except json.JSONDecodeError:
                    pass
                buffer = ""
    return records


def build_alpaca_records() -> list[dict]:
    records: list[dict] = []

    # 单轮问答：C 端客服
    for item in load_jsonl(FINETUNE_DIR / "raw_qa.jsonl"):
        records.append({
            "instruction": item.get("instruction", item.get("query", "")),
            "input": "",
            "output": item["answer"],
            "system": SYSTEM_CUSTOMER,
        })

    # 多轮对话：C 端客服（把多轮拆成 instruction + history）
    for dialogue in load_jsonl(FINETUNE_DIR / "raw_dialogue.jsonl"):
        turns = dialogue.get("conversations", [])
        if len(turns) < 2:
            continue
        history = []
        for i in range(0, len(turns) - 1, 2):
            user_msg = turns[i]
            assistant_msg = turns[i + 1] if i + 1 < len(turns) else None
            if user_msg.get("role") != "user" or assistant_msg is None:
                continue
            history.append({"from": "user", "value": user_msg["content"]})
            history.append({"from": "assistant", "value": assistant_msg["content"]})
        if history:
            last_user = history.pop()["value"] if history else ""
            records.append({
                "instruction": last_user,
                "input": "",
                "output": turns[-1]["content"] if turns[-1].get("role") == "assistant" else "",
                "system": dialogue.get("system", SYSTEM_CUSTOMER),
                "history": history,
            })

    # 员工侧数据（复盘/文案/知识答疑）
    for item in load_jsonl(FINETUNE_DIR / "raw_staff.jsonl"):
        records.append({
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "output": item["answer"],
            "system": SYSTEM_STAFF,
        })

    # 需求表单结构化输出
    for item in load_jsonl(FINETUNE_DIR / "raw_form.jsonl"):
        records.append({
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "output": json.dumps(item["form"], ensure_ascii=False),
            "system": "输出客户定制需求表单 JSON，仅输出 JSON 对象。",
        })

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "dataset.jsonl")
    args = parser.parse_args()

    records = build_alpaca_records()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"已导出 {len(records)} 条微调数据 → {args.output}")
    print("字段说明：instruction/input/output/system/history（alpaca 格式，兼容 LlamaFactory）")


if __name__ == "__main__":
    main()
