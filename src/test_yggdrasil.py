"""
世界树认知记忆引擎 - 核心功能测试
"""

import os
import sys
import json
import tempfile

# 确保可以导入 yggdrasil
sys.path.insert(0, os.path.dirname(__file__))

from yggdrasil.database import Database
from yggdrasil.memory_engine import MemoryEngine
from yggdrasil.tool_router import ToolRouter
from yggdrasil.decay import calculate_weight, sort_by_weight
from datetime import datetime, timezone, timedelta


def test_database_and_crud():
    """测试数据库层和核心 CRUD"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = Database(db_path)
        engine = MemoryEngine(db)

        # 1. 初始化根节点
        print("=== 初始化根节点 ===")
        root = engine.initialize_root()
        print(f"  根节点: {root['name']} (id={root['id']})")

        # 重复初始化应跳过
        root2 = engine.initialize_root()
        assert root2["id"] == root["id"], "重复初始化应返回同一根节点"
        print("  重复初始化: OK (跳过)")

        # 2. 猴子爬树 - 查看根节点
        print("\n=== retrieve_context (根节点) ===")
        ctx = engine.retrieve_context()
        print(f"  节点: {ctx['node']['name']}")
        print(f"  子节点: {[c['name'] for c in ctx['children']]}")
        assert len(ctx["children"]) == 3, "根节点应有3个子节点"

        # 3. 创建业务节点
        print("\n=== create_memory_node ===")
        proj = engine.create_memory_node(
            name="project_alpha",
            content="Alpha 项目：基于 React+Go 的电商后台",
            parent_name="global_map",
            k_value=0.0,
            initial_score=0.9,
        )
        print(f"  创建: {proj['name']} (parent=global_map, k={proj['k_value']})")

        temp = engine.create_memory_node(
            name="debug_log_0410",
            content="临时调试日志：TypeError at line 42...",
            parent_name="execution_state",
            k_value=1.0,  # 快速衰减
            initial_score=0.5,
        )
        print(f"  创建: {temp['name']} (k={temp['k_value']}, 阅后即焚)")

        # 4. 猴子爬树 - 深入查看
        print("\n=== retrieve_context (global_map) ===")
        ctx2 = engine.retrieve_context("global_map")
        print(f"  节点: {ctx2['node']['name']}")
        child_strs = [f"{c['name']}(w={c['weight']})" for c in ctx2['children']]
        print(f"  子节点: {child_strs}")

        # 5. 创建带边的节点
        engine.create_memory_node(
            name="auth_module",
            content="用户鉴权模块设计",
            parent_name="project_alpha",
            k_value=0.0,
            edges=[{"target": "project_alpha", "label": "BELONGS_TO"}],
        )
        print("\n=== 带边节点 ===")
        ctx3 = engine.retrieve_context("auth_module")
        print(f"  出边: {ctx3['outgoing_edges']}")

        # 6. Git 式更新
        print("\n=== update (Git式) ===")
        engine.update(
            "project_alpha",
            new_content="Alpha 项目 v2：已完成支付模块重构",
        )
        updated = engine.retrieve_context("project_alpha")
        print(f"  更新后内容: {updated['node']['content'][:50]}")

        # 查看历史
        history = engine.node_history("project_alpha")
        print(f"  历史版本数: {len(history)}")
        assert len(history) == 1, "应有1个历史快照"
        print(f"  v{history[0]['version']} 内容: {history[0]['content_preview'][:50]}")

        # 7. 追加内容
        engine.update(
            "project_alpha",
            new_content="附加: 支付网关对接完成",
            append_content=True,
        )
        appended = engine.retrieve_context("project_alpha")
        assert "附加" in appended["node"]["content"]
        print(f"\n=== 追加内容 OK ===")

        # 8. 物理删除
        print("\n=== hard_delete ===")
        result = engine.hard_delete("debug_log_0410")
        print(f"  删除: {result['deleted']}, 成功: {result['success']}")

        # 确认不存在
        try:
            engine.retrieve_context("debug_log_0410")
            assert False, "节点应已删除"
        except ValueError:
            print("  确认: 节点已不存在")

        # 9. 不允许删除根节点
        try:
            engine.hard_delete("root")
            assert False, "不应该能删除根节点"
        except ValueError as e:
            print(f"  根节点保护: {e}")

        print("\n✅ 所有 CRUD 测试通过!")

    finally:
        os.unlink(db_path)


def test_decay():
    """测试衰减公式"""
    print("\n=== 衰减公式测试 ===")
    now = datetime(2026, 4, 10, 12, 0, 0, tzinfo=timezone.utc)

    # k=0 永久钢印
    w1 = calculate_weight(1.0, 0.0, "2025-01-01 00:00:00", now)
    assert w1 == 1.0, f"k=0 应永不衰减, got {w1}"
    print(f"  k=0 (永久): weight={w1}")

    # k=0.01 缓慢衰减
    w2 = calculate_weight(1.0, 0.01, "2026-04-09 12:00:00", now)
    print(f"  k=0.01 (24h前): weight={w2:.4f}")
    assert 0 < w2 < 1

    # k=1.0 快速衰减
    w3 = calculate_weight(1.0, 1.0, "2026-04-09 12:00:00", now)
    print(f"  k=1.0 (24h前): weight={w3:.6f}")
    assert w3 < 0.001  # 几乎为零

    print("✅ 衰减公式测试通过!")


def test_tool_router():
    """测试泛型工具路由"""
    print("\n=== 泛型工具路由测试 ===")
    router = ToolRouter()

    # 注册一个测试工具
    def greet(name: str, greeting: str = "Hello"):
        return f"{greeting}, {name}!"

    router.register("greet", greet, {"name": "string", "greeting": "string"})

    # 正常调用
    r1 = router.execute("greet", '{"name": "World"}')
    assert r1["success"] and r1["result"] == "Hello, World!"
    print(f"  正常调用: {r1['result']}")

    # 工具不存在
    r2 = router.execute("nonexistent", "{}")
    assert not r2["success"]
    print(f"  不存在: {r2['error'][:40]}...")

    # JSON 格式错误
    r3 = router.execute("greet", "{bad json}")
    assert not r3["success"]
    print(f"  JSON错误: {r3['error'][:40]}...")

    # 参数错误
    r4 = router.execute("greet", '{"wrong_param": 1}')
    assert not r4["success"]
    print(f"  参数错误: {r4['error'][:40]}...")

    print("✅ 泛型工具路由测试通过!")


def test_top_k_top_p():
    """测试 top-k / top-p 筛选"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = Database(db_path)
        engine = MemoryEngine(db)
        engine.initialize_root()

        # 创建多个子节点
        for i, (name, score) in enumerate([
            ("high_priority", 1.0),
            ("medium_priority", 0.5),
            ("low_priority", 0.1),
        ]):
            engine.create_memory_node(
                name=name,
                content=f"测试节点 {name}",
                parent_name="global_map",
                k_value=0.0,
                initial_score=score,
            )

        print("\n=== top-k / top-p 筛选 ===")

        # top_k=2
        ctx_k = engine.retrieve_context("global_map", top_k=2)
        names_k = [c["name"] for c in ctx_k["children"]]
        print(f"  top_k=2: {names_k}")
        assert len(names_k) == 2

        # top_p=0.8
        ctx_p = engine.retrieve_context("global_map", top_p=0.8)
        names_p = [c["name"] for c in ctx_p["children"]]
        print(f"  top_p=0.8: {names_p}")

        print("✅ top-k/top-p 测试通过!")

    finally:
        os.unlink(db_path)


if __name__ == "__main__":
    test_decay()
    test_database_and_crud()
    test_tool_router()
    test_top_k_top_p()
    print("\n🌳 世界树 MVP 全部测试通过!")
