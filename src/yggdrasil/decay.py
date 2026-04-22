"""
衰减公式模块
实现量化软遗忘: Weight(t) = Score_initial × e^(-k × Δt)
"""

import math
from datetime import datetime, timezone


def calculate_weight(
    initial_score: float,
    k_value: float,
    updated_at: str | datetime,
    now: datetime | None = None,
) -> float:
    """
    计算节点当前的有效权重。

    参数:
        initial_score: 节点初始重要性分数
        k_value: 衰减系数 (0=永久钢印, 极大值=阅后即焚)
        updated_at: 节点最后更新时间 (ISO 格式字符串或 datetime)
        now: 当前时间 (默认 UTC now)

    返回:
        当前权重值
    """
    if now is None:
        now = datetime.now(timezone.utc)

    if isinstance(updated_at, str):
        # SQLite datetime 格式: "YYYY-MM-DD HH:MM:SS"
        updated_at = datetime.fromisoformat(updated_at).replace(tzinfo=timezone.utc)

    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    delta_hours = max((now - updated_at).total_seconds() / 3600.0, 0.0)

    if k_value == 0.0:
        return initial_score

    return initial_score * math.exp(-k_value * delta_hours)


def sort_by_weight(nodes: list[dict], now: datetime | None = None) -> list[dict]:
    """按衰减权重降序排列节点列表"""
    for n in nodes:
        n["_weight"] = calculate_weight(
            n["initial_score"], n["k_value"], n["updated_at"], now
        )
    nodes.sort(key=lambda n: n["_weight"], reverse=True)
    return nodes


def filter_top_k(nodes: list[dict], k: int) -> list[dict]:
    """保留权重最高的 k 个节点"""
    return nodes[:k]


def filter_top_p(nodes: list[dict], p: float) -> list[dict]:
    """
    Top-p (nucleus) 筛选: 保留累积权重占比达 p 的节点。
    要求 nodes 已按 _weight 降序排列。
    """
    if not nodes:
        return []
    total = sum(n["_weight"] for n in nodes)
    if total <= 0:
        return nodes[:1]
    cumulative = 0.0
    result = []
    for n in nodes:
        cumulative += n["_weight"]
        result.append(n)
        if cumulative / total >= p:
            break
    return result
