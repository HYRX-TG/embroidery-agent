"""综合评测：检索命中 + 意图准确率 + 行为合规（基于黄金测试集）。

黄金测试集每行：{query, expected_sources, role, intent, behavior}
  - expected_sources: 期望命中的知识来源关键词（工艺/针法/品类/保养/纹样寓意/内部标准）
  - intent:            期望意图（knowledge/pattern_meaning/custom_requirement/order_service/
                       complaint/staff_analysis/staff_copywriting/small_talk）
  - behavior:          期望行为（answer 直接答 / collect 收集定制需求 / transfer 转人工 / staff 员工工具）

用法：
    python scripts/evaluate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embroidery_agent.agent.router import build_runtime  # noqa: E402
from embroidery_agent.core.schema import Role  # noqa: E402

# 行为 → 可接受的意图集合
BEHAVIOR_INTENTS = {
    "answer": {"knowledge", "pattern_meaning", "small_talk"},
    "collect": {"custom_requirement"},
    "transfer": {"order_service", "complaint"},
    "staff": {"staff_analysis", "staff_copywriting", "knowledge"},
}


def load_cases(golden_path: Path) -> list[dict]:
    cases = []
    for line in golden_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def evaluate(runtime, golden_path: Path, role: Role) -> dict:
    cases = [c for c in load_cases(golden_path) if c.get("role", "customer") == role.value]

    # ---- 1. 检索质量（仅对有来源标注的用例） ----
    retrieval_cases = [c for c in cases if c.get("expected_sources")]
    hit_at_1 = hit_at_5 = 0
    mrr_sum = 0.0
    retrieval_examples = []

    def matched(retrieved: str, expected: set[str]) -> bool:
        return any(exp in retrieved for exp in expected)

    for case in retrieval_cases:
        query = case["query"]
        expected = set(case.get("expected_sources", []))
        hits = runtime.retriever.search(query, role=role)
        retrieved = [h.chunk.source for h in hits[:5]]
        matched_ids = [r for r in retrieved if matched(r, expected)]
        if matched_ids:
            rank = retrieved.index(matched_ids[0]) + 1
            if rank == 1:
                hit_at_1 += 1
            if rank <= 5:
                hit_at_5 += 1
            mrr_sum += 1.0 / rank
            retrieval_examples.append({"query": query, "rank": rank, "retrieved": retrieved[:3]})
        else:
            retrieval_examples.append({"query": query, "rank": None, "retrieved": retrieved[:3]})

    n_ret = len(retrieval_cases)

    # ---- 2. 意图准确率 + 3. 行为合规 ----
    classifier = runtime.customer.classifier
    intent_correct = intent_total = 0
    behavior_ok = behavior_total = 0
    intent_wrong: list[dict] = []
    behavior_wrong: list[dict] = []

    for case in cases:
        pred = classifier.classify(case["query"], role=role.value)
        pred_intent = pred.intent.value

        if case.get("intent"):
            intent_total += 1
            if pred_intent == case["intent"]:
                intent_correct += 1
            else:
                intent_wrong.append(
                    {"query": case["query"], "expect": case["intent"], "got": pred_intent}
                )

        if case.get("behavior"):
            behavior_total += 1
            allowed = BEHAVIOR_INTENTS.get(case["behavior"], set())
            if pred_intent in allowed:
                behavior_ok += 1
            else:
                behavior_wrong.append(
                    {"query": case["query"], "behavior": case["behavior"], "got": pred_intent}
                )

    return {
        "role": role.value,
        "total_cases": len(cases),
        # 检索
        "retrieval_cases": n_ret,
        "hit_at_1": round(hit_at_1 / n_ret, 4) if n_ret else 0,
        "hit_at_5": round(hit_at_5 / n_ret, 4) if n_ret else 0,
        "mrr": round(mrr_sum / n_ret, 4) if n_ret else 0,
        # 意图
        "intent_accuracy": round(intent_correct / intent_total, 4) if intent_total else 0,
        "intent_wrong_count": len(intent_wrong),
        # 行为合规
        "behavior_compliance": round(behavior_ok / behavior_total, 4) if behavior_total else 0,
        "behavior_wrong_count": len(behavior_wrong),
        # 明细（各取前 12 条）
        "retrieval_examples": retrieval_examples[:12],
        "intent_wrong": intent_wrong[:12],
        "behavior_wrong": behavior_wrong[:12],
    }


def main() -> None:
    runtime = build_runtime()
    runtime.load_kb()
    if not runtime.kb.chunks:
        print("知识库为空，请先运行 python scripts/ensure_kb.py", file=sys.stderr)
        raise SystemExit(1)

    golden_path = Path(__file__).resolve().parents[1] / "data" / "golden_test.jsonl"
    if not golden_path.exists():
        print(f"黄金测试集不存在：{golden_path}", file=sys.stderr)
        raise SystemExit(1)

    for role in (Role.CUSTOMER, Role.STAFF):
        report = evaluate(runtime, golden_path, role)
        print(f"\n===== 角色：{report['role']}（共 {report['total_cases']} 条） =====")
        summary = {k: v for k, v in report.items() if not k.endswith("examples") and k not in ("intent_wrong", "behavior_wrong")}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        for ex in report["retrieval_examples"]:
            print(f"  [检索] {ex['query']} → rank={ex['rank']} | {ex['retrieved']}")
        for ex in report["intent_wrong"]:
            print(f"  [意图误判] 期望={ex['expect']} 实际={ex['got']} | {ex['query']}")
        for ex in report["behavior_wrong"]:
            print(f"  [行为偏差] 期望={ex['behavior']} 实际={ex['got']} | {ex['query']}")


if __name__ == "__main__":
    main()
