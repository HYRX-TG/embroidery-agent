"""确定性业务计算器：物流运费 / 订单报价 / 生产排程 / 工单统计。

设计原则（生产可用）：
- 所有数值计算由纯函数完成，LLM 只负责理解需求与解释结果，不做算术。
- 价格表 / 产能参数均来自配置文件，业务可直接编辑，无需改代码。
- 输入校验严格：非法输入抛 ValueError，绝不静默返回可疑数字。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 物流价格表：configs/logistics.json（业务可直接编辑；后期可由外挂 RAG/ERP 替代）
LOGISTICS_CONFIG = PROJECT_ROOT / "configs" / "logistics.json"
PRODUCTION_CONFIG = PROJECT_ROOT / "configs" / "production.json"

# 配置热缓存：mtime 变了才重读，不变则零 IO（改配置即时生效）
_json_cache: dict[str, tuple[float, dict]] = {}
_json_lock = threading.Lock()


def _load_json_cached(path: Path) -> dict:
    with _json_lock:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            raise RuntimeError(f"配置文件缺失: {path}") from None
        cached = _json_cache.get(str(path))
        if cached and cached[0] == mtime:
            return cached[1]
        data = json.loads(path.read_text(encoding="utf-8"))
        _json_cache[str(path)] = (mtime, data)
        return data


def _load_logistics() -> dict:
    return _load_json_cached(LOGISTICS_CONFIG)


def _num(value: Any, field_name: str, minimum: float = 0.0) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"参数 {field_name} 不是有效数字: {value!r}") from None
    if num < minimum:
        raise ValueError(f"参数 {field_name} 不能小于 {minimum}")
    return num


# ---------------------------------------------------------------------------
# 物流运费试算
# ---------------------------------------------------------------------------
def shipping_quote(
    weight_kg: float,
    dest_zone: str,
    express: str = "",
    length_cm: float = 0.0,
    width_cm: float = 0.0,
    height_cm: float = 0.0,
) -> dict:
    """运费试算：计费重 = max(实重, 体积重)，按分区价格表计算首重/续重。

    dest_zone: 价格表中的分区键（如 "华南"、"华东"）；
    express:   承运商键（如 "debang"、"shunfeng"），留空返回全部承运商比选。
    """
    cfg = _load_logistics()
    weight_kg = _num(weight_kg, "weight_kg", minimum=0.01)
    if dest_zone not in cfg["zones"]:
        raise ValueError(f"未知物流分区: {dest_zone}（可选: {', '.join(cfg['zones'])}）")

    # 体积重（抛比），无尺寸时忽略
    dim_weight = 0.0
    if all(_num(v, d, 0) > 0 for v, d in [
        (length_cm, "length_cm"), (width_cm, "width_cm"), (height_cm, "height_cm")
    ]):
        dim_weight = length_cm * width_cm * height_cm / cfg["dim_divisor"]

    chargeable = max(weight_kg, dim_weight)
    quotes = []
    for key, carrier in cfg["carriers"].items():
        if express and key != express:
            continue
        zone_price = carrier["prices"].get(dest_zone)
        if zone_price is None:
            continue
        first, additional = zone_price["first"], zone_price["additional"]
        extra_kg = max(0.0, chargeable - cfg["first_weight_kg"])
        # 续重按 0.5kg 向上取整计费，additional 为元/kg
        import math
        extra_units = math.ceil(extra_kg * 2) / 2
        fee = first + extra_units * additional
        quotes.append({
            "carrier": carrier["name"],
            "carrier_key": key,
            "first_fee": round(first, 2),
            "chargeable_weight": round(chargeable, 2),
            "estimated_fee": round(fee, 2),
            "eta_days": zone_price.get("eta_days", ""),
        })
    if not quotes:
        raise ValueError("没有匹配的承运商，请检查 express 参数")
    quotes.sort(key=lambda q: q["estimated_fee"])
    return {
        "weight_kg": round(weight_kg, 2),
        "dim_weight_kg": round(dim_weight, 2),
        "chargeable_weight_kg": round(chargeable, 2),
        "dest_zone": dest_zone,
        "quotes": quotes,
        "recommended": quotes[0]["carrier"],
        "note": "估价仅供参考，实际以承运商账单为准；价格表可在 configs/logistics.json 维护",
    }


# ---------------------------------------------------------------------------
# 订单报价核算（B 端内部）
# ---------------------------------------------------------------------------
def order_quote(
    width_cm: float,
    height_cm: float,
    craft: str = "machine",
    quantity: int = 1,
    framing: bool = False,
) -> dict:
    """按面积 × 工艺单价估算工时与成本，输出报价区间。

    工艺参数来自 logistics.json 同级的 production 配置段（configs/production.json）。
    """
    prod = _load_production()
    width_cm = _num(width_cm, "width_cm", minimum=1)
    height_cm = _num(height_cm, "height_cm", minimum=1)
    quantity = int(_num(quantity, "quantity", minimum=1))
    if craft not in prod["crafts"]:
        raise ValueError(f"未知工艺: {craft}（可选: {', '.join(prod['crafts'])}）")

    c = prod["crafts"][craft]
    area_dm2 = width_cm * height_cm / 100.0  # cm² → dm²
    hours_per_piece = area_dm2 * c["hours_per_dm2"]
    labor_cost = hours_per_piece * prod["labor_hourly_rate"]
    material_cost = area_dm2 * c["material_per_dm2"]
    framing_cost = prod.get("framing_fee", 0.0) if framing else 0.0

    unit_cost = labor_cost + material_cost + framing_cost
    # 数量折扣档位来自配置（如 [{min_quantity: 11, rate: 0.95}]），按降序取首个命中
    rate = 1.0
    for tier in sorted(prod.get("quantity_discounts", []),
                       key=lambda t: t.get("min_quantity", 0), reverse=True):
        if quantity >= int(tier.get("min_quantity", 0)):
            rate = float(tier.get("rate", 1.0))
            break
    unit_price_low = unit_cost * (1 + prod["margin_low"]) * rate
    unit_price_high = unit_cost * (1 + prod["margin_high"]) * rate

    return {
        "craft": c["name"],
        "area_dm2": round(area_dm2, 2),
        "hours_per_piece": round(hours_per_piece, 1),
        "unit_cost": round(unit_cost, 2),
        "discount": rate,
        "unit_price_range": [round(unit_price_low, 2), round(unit_price_high, 2)],
        "total_price_range": [round(unit_price_low * quantity, 2), round(unit_price_high * quantity, 2)],
        "total_hours": round(hours_per_piece * quantity, 1),
        "quantity": quantity,
        "note": "内部核算口径，对外报价请结合客户等级与市场行情调整",
    }


def _load_production() -> dict:
    return _load_json_cached(PRODUCTION_CONFIG)


# ---------------------------------------------------------------------------
# 生产排程 / 交期评估
# ---------------------------------------------------------------------------
def schedule_estimate(
    quantity: int,
    craft: str = "machine",
    width_cm: float = 40.0,
    height_cm: float = 60.0,
    parallel_workers: int = 0,
) -> dict:
    """工序拆解与交期估算：打版 → 绣制 → 后整 → 质检。"""
    prod = _load_production()
    quantity = int(_num(quantity, "quantity", minimum=1))
    width_cm = _num(width_cm, "width_cm", minimum=0.1)
    height_cm = _num(height_cm, "height_cm", minimum=0.1)
    if craft not in prod["crafts"]:
        raise ValueError(f"未知工艺: {craft}（可选: {', '.join(prod['crafts'])}）")
    c = prod["crafts"][craft]
    workers = int(parallel_workers or c["default_workers"])

    area_dm2 = width_cm * height_cm / 100.0
    stitch_hours = quantity * area_dm2 * c["hours_per_dm2"]
    pattern_hours = c["pattern_setup_hours"]  # 打版（批量摊一次）
    finishing_hours = quantity * prod["finishing_hours_per_piece"]
    qc_hours = quantity * prod["qc_hours_per_piece"]

    total_hours = pattern_hours + stitch_hours + finishing_hours + qc_hours
    calendar_days = total_hours / (workers * prod["work_hours_per_day"]) * prod["efficiency_factor"]

    return {
        "craft": c["name"],
        "quantity": quantity,
        "workers": workers,
        "processes": [
            {"name": "打版", "hours": round(pattern_hours, 1)},
            {"name": "绣制", "hours": round(stitch_hours, 1)},
            {"name": "后整", "hours": round(finishing_hours, 1)},
            {"name": "质检", "hours": round(qc_hours, 1)},
        ],
        "total_hours": round(total_hours, 1),
        "estimated_days": round(calendar_days, 1),
        "note": f"按 {workers} 人并行、每日 {prod['work_hours_per_day']} 小时估算，已含效率折扣",
    }


# ---------------------------------------------------------------------------
# 工单统计（只读白名单查询）
# ---------------------------------------------------------------------------
def ticket_stats(db_path: str | Path, days: int = 30) -> dict:
    """工单库只读统计：总量、状态分布、近期趋势。禁止任何写操作。"""
    days = int(_num(days, "days", minimum=1))
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT created_at, status FROM tickets"
        ).fetchall()
    finally:
        conn.close()
    total = len(rows)
    by_status: dict[str, int] = {}
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    recent = 0
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
        try:
            # created_at 为 ISO-8601 带时区（如 2026-09-06T07:00:00+00:00）
            created = datetime.fromisoformat(row["created_at"])
        except (TypeError, ValueError):
            continue
        if created >= cutoff:
            recent += 1
    return {"total": total, "by_status": by_status, f"last_{days}_days": recent}
