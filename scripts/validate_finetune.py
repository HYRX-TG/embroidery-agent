"""微调数据校验：逐行检查四类原始数据的格式、必填项、长度与重复。

用法：
    python scripts/validate_finetune.py            # 校验 data/finetune/ 全部原始数据
    python scripts/validate_finetune.py --export   # 校验并执行导出
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINETUNE_DIR = ROOT / "data" / "finetune"

FORM_KEYS = {
    "pattern_theme", "size", "craft_type", "quantity", "usage",
    "budget_range", "deadline", "special_requirements", "core_preferences",
}

MIN_LEN = 5
MAX_LEN = 1200


def load_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    """解析 JSONL，返回 (记录列表, 错误列表)。"""
    records: list[dict] = []
    errors: list[str] = []
    if not path.exists():
        return records, [f"{path.name}: 文件不存在"]
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}:{i} JSON 解析失败: {exc}")
            continue
        if not isinstance(obj, dict):
            errors.append(f"{path.name}:{i} 不是对象")
            continue
        records.append(obj)
    return records, errors


def check_text(path: Path, idx: int, value, label: str, min_len: int = MIN_LEN) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path.name}:{idx} {label} 为空")
        return errors
    n = len(value.strip())
    if n < min_len:
        errors.append(f"{path.name}:{idx} {label} 过短({n}<{min_len})")
    if n > MAX_LEN:
        errors.append(f"{path.name}:{idx} {label} 过长({n}>{MAX_LEN})")
    return errors


def validate_qa(path: Path, records: list[dict]) -> tuple[int, list[str]]:
    seen: set[str] = set()
    errors: list[str] = []
    for i, rec in enumerate(records, 1):
        errors += check_text(path, i, rec.get("instruction"), "instruction")
        errors += check_text(path, i, rec.get("answer"), "answer")
        instr = (rec.get("instruction") or "").strip()
        if instr and instr in seen:
            errors.append(f"{path.name}:{i} 重复 instruction: {instr[:30]}")
        seen.add(instr)
    return len(records), errors


def validate_dialogue(path: Path, records: list[dict]) -> tuple[int, list[str]]:
    errors: list[str] = []
    for i, rec in enumerate(records, 1):
        conv = rec.get("conversations")
        if not isinstance(conv, list) or len(conv) < 2:
            errors.append(f"{path.name}:{i} conversations 需为≥2 的数组")
            continue
        for j, turn in enumerate(conv, 1):
            if not isinstance(turn, dict) or turn.get("role") not in ("user", "assistant"):
                errors.append(f"{path.name}:{i} 第{j}轮 role 非法: {turn}")
                continue
            # 对话轮次允许短句（如"就一幅"），长度下限放宽到 2 字
            errors += check_text(path, i, turn.get("content"), f"conversations[{j}].content", min_len=2)
        if conv[-1].get("role") != "assistant":
            errors.append(f"{path.name}:{i} 最后一轮必须是 assistant")
        # 交替检查
        for j in range(1, len(conv)):
            if conv[j].get("role") == conv[j - 1].get("role"):
                errors.append(f"{path.name}:{i} 第{j}轮与上一轮角色重复")
    return len(records), errors


def validate_staff(path: Path, records: list[dict]) -> tuple[int, list[str]]:
    seen: set[str] = set()
    errors: list[str] = []
    for i, rec in enumerate(records, 1):
        errors += check_text(path, i, rec.get("instruction"), "instruction")
        errors += check_text(path, i, rec.get("answer"), "answer")
        if "input" in rec and rec.get("input"):
            errors += check_text(path, i, rec.get("input"), "input")
        instr = (rec.get("instruction") or "").strip()
        if instr and instr in seen:
            errors.append(f"{path.name}:{i} 重复 instruction: {instr[:30]}")
        seen.add(instr)
    return len(records), errors


def validate_form(path: Path, records: list[dict]) -> tuple[int, list[str]]:
    errors: list[str] = []
    for i, rec in enumerate(records, 1):
        errors += check_text(path, i, rec.get("instruction"), "instruction")
        form = rec.get("form")
        if not isinstance(form, dict):
            errors.append(f"{path.name}:{i} form 必须是对象")
            continue
        extra = set(form.keys()) - FORM_KEYS
        missing = FORM_KEYS - set(form.keys())
        if extra:
            errors.append(f"{path.name}:{i} form 含非法字段: {sorted(extra)}")
        if missing:
            errors.append(f"{path.name}:{i} form 缺少字段: {sorted(missing)}")
        for key, value in form.items():
            if not isinstance(value, str):
                errors.append(f"{path.name}:{i} form.{key} 必须是字符串: {value!r}")
    return len(records), errors


VALIDATORS = {
    "raw_qa.jsonl": validate_qa,
    "raw_dialogue.jsonl": validate_dialogue,
    "raw_staff.jsonl": validate_staff,
    "raw_form.jsonl": validate_form,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="校验微调原始数据")
    parser.add_argument("--export", action="store_true", help="校验通过后执行导出")
    args = parser.parse_args()

    total_errors = 0
    grand_total = 0
    for name, fn in VALIDATORS.items():
        path = FINETUNE_DIR / name
        records, load_errors = load_jsonl(path)
        n, errs = fn(path, records)
        all_errs = load_errors + errs
        grand_total += n
        total_errors += len(all_errs)
        status = "OK " if not all_errs else "ERR"
        print(f"[{status}] {name}: {n} 条, 问题 {len(all_errs)}")
        for e in all_errs[:10]:
            print(f"      - {e}")
        if len(all_errs) > 10:
            print(f"      ... 另有 {len(all_errs) - 10} 条问题未展示")

    print(f"\n合计: {grand_total} 条, 问题 {total_errors}")
    if total_errors:
        return 1

    if args.export:
        sys.path.insert(0, str(ROOT / "src"))
        sys.path.insert(0, str(ROOT))  # 使 `import scripts.export_finetune` 可解析
        from scripts.export_finetune import build_alpaca_records  # noqa: E402

        records = build_alpaca_records()
        out = ROOT / "data" / "dataset.jsonl"
        with out.open("w", encoding="utf-8", newline="\n") as handle:
            for rec in records:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[OK] 已导出 {len(records)} 条 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
