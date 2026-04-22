"""
核心记忆引擎 - CRUD 操作
实现蓝图中的四个基础 API:
  create_memory_node / retrieve_context / update / hard_delete
以及 node_history 版本回溯
"""

import json
from .database import Database
from .decay import sort_by_weight, filter_top_k, filter_top_p, calculate_weight


class MemoryEngine:
    """世界树记忆引擎核心"""

    def __init__(self, db: Database | None = None):
        self.db = db or Database()

    # ── create_memory_node ──────────────────────────────

    def create_memory_node(
        self,
        name: str,
        content: str,
        parent_name: str | None = None,
        k_value: float = 0.0,
        initial_score: float = 1.0,
        edges: list[dict] | None = None,
    ) -> dict:
        """
        创建记忆节点。

        参数:
            name: 节点名称 (唯一)
            content: 节点内容 (可以是摘要、指针路径等)
            parent_name: 父节点名称 (None 表示根级)
            k_value: 衰减系数 (0=永久, 越大衰减越快)
            initial_score: 初始重要性分数
            edges: 关联出边列表 [{"target": "目标节点名", "label": "关系类型"}]

        返回:
            创建的节点信息
        """
        parent_id = None
        if parent_name:
            parent = self.db.get_node_by_name(parent_name)
            if not parent:
                raise ValueError(f"父节点 '{parent_name}' 不存在")
            parent_id = parent["id"]

        node = self.db.create_node(
            name=name,
            content=content,
            parent_id=parent_id,
            k_value=k_value,
            initial_score=initial_score,
        )

        # 创建关联边
        if edges:
            for edge_def in edges:
                target = self.db.get_node_by_name(edge_def["target"])
                if target:
                    self.db.create_edge(
                        node["id"], target["id"],
                        edge_def.get("label", "RELATED_TO"),
                    )

        return self._format_node(node)

    # ── retrieve_context (猴子爬树) ────────────────────

    def retrieve_context(
        self,
        node_name: str | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> dict:
        """
        检索节点上下文 (猴子爬树的一步)。

        参数:
            node_name: 节点名称。为空时返回根节点。
            top_k: 只返回权重最高的 k 个子节点
            top_p: 只返回累积权重占比达 p 的子节点

        返回:
            节点内容、关联边、子节点列表 (按衰减权重排序)
        """
        if node_name:
            node = self.db.get_node_by_name(node_name)
            if not node:
                raise ValueError(f"节点 '{node_name}' 不存在")
        else:
            node = self.db.get_root_node()
            if not node:
                raise ValueError("根节点尚未初始化")

        # 获取关联边
        out_edges = self.db.get_outgoing_edges(node["id"])
        in_edges = self.db.get_incoming_edges(node["id"])

        # 获取子节点并按衰减权重排序
        children = self.db.get_children(node["id"])
        children = sort_by_weight(children)

        # 筛选
        if top_k is not None:
            children = filter_top_k(children, top_k)
        elif top_p is not None:
            children = filter_top_p(children, top_p)

        return {
            "node": self._format_node(node),
            "outgoing_edges": [
                {"target": e["target_name"], "label": e["label"]} for e in out_edges
            ],
            "incoming_edges": [
                {"source": e["source_name"], "label": e["label"]} for e in in_edges
            ],
            "children": [
                {
                    "name": c["name"],
                    "content_preview": c["content"][:200] + ("..." if len(c["content"]) > 200 else ""),
                    "k_value": c["k_value"],
                    "weight": round(c["_weight"], 4),
                }
                for c in children
            ],
        }

    # ── update ──────────────────────────────────────────

    def update(
        self,
        node_name: str,
        new_name: str | None = None,
        new_parent_name: str | None = None,
        new_content: str | None = None,
        new_k_value: float | None = None,
        new_initial_score: float | None = None,
        new_edges: list[dict] | None = None,
        append_content: bool = False,
    ) -> dict:
        """
        更新节点 (Git 式: 先保存历史快照再修改)。

        参数:
            node_name: 要修改的节点名称
            new_name: 新名称 (None=不改)
            new_parent_name: 新父节点名称 (None=不改)
            new_content: 新内容 (None=不改)
            new_k_value: 新衰减系数 (None=不改)
            new_initial_score: 新初始分数 (None=不改)
            new_edges: 新的出边列表 (None=不改, 提供时替换全部出边)
            append_content: True 时 new_content 为追加而非覆盖
        """
        node = self.db.get_node_by_name(node_name)
        if not node:
            raise ValueError(f"节点 '{node_name}' 不存在")

        # Git 式: 先保存历史快照
        old_edges = self.db.get_outgoing_edges(node["id"])
        self.db.save_history_snapshot(node["id"], node, old_edges)

        # 构建更新字段
        updates = {}
        if new_name is not None:
            updates["name"] = new_name
        if new_content is not None:
            if append_content:
                updates["content"] = node["content"] + "\n" + new_content
            else:
                updates["content"] = new_content
        if new_k_value is not None:
            updates["k_value"] = new_k_value
        if new_initial_score is not None:
            updates["initial_score"] = new_initial_score
        if new_parent_name is not None:
            new_parent = self.db.get_node_by_name(new_parent_name)
            if not new_parent:
                raise ValueError(f"新父节点 '{new_parent_name}' 不存在")
            updates["parent_id"] = new_parent["id"]

        updated_node = self.db.update_node(node["id"], **updates)

        # 更新边
        if new_edges is not None:
            self.db.delete_edges_for_node(node["id"])
            # 重建入边 (保留)
            # 只重建出边
            for edge_def in new_edges:
                target = self.db.get_node_by_name(edge_def["target"])
                if target:
                    self.db.create_edge(
                        node["id"], target["id"],
                        edge_def.get("label", "RELATED_TO"),
                    )

        return self._format_node(updated_node)

    # ── hard_delete ─────────────────────────────────────

    def hard_delete(self, node_name: str) -> dict:
        """
        物理删除节点及其子树、所有关联边和历史记录。
        用于切除幻觉或彻底废弃的节点。
        """
        node = self.db.get_node_by_name(node_name)
        if not node:
            raise ValueError(f"节点 '{node_name}' 不存在")

        # 防止删除根节点
        if node["parent_id"] is None:
            root = self.db.get_root_node()
            if root and root["id"] == node["id"]:
                raise ValueError("不允许删除根节点")

        children = self.db.get_children(node["id"])
        child_names = [c["name"] for c in children]

        success = self.db.delete_node(node["id"])
        return {
            "deleted": node_name,
            "cascade_deleted_children": child_names,
            "success": success,
        }

    # ── node_history ────────────────────────────────────

    def node_history(self, node_name: str) -> list[dict]:
        """
        查询节点的历史版本快照 (时间漫游工具)。
        """
        node = self.db.get_node_by_name(node_name)
        if not node:
            raise ValueError(f"节点 '{node_name}' 不存在")

        history = self.db.get_history(node["id"])
        return [
            {
                "version": h["version"],
                "name": h["name"],
                "content_preview": h["content"][:300],
                "k_value": h["k_value"],
                "edges": json.loads(h["edges_snapshot"]),
                "recorded_at": h["recorded_at"],
            }
            for h in history
        ]

    # ── 初始化根节点 ────────────────────────────────────

    def initialize_root(self) -> dict:
        """
        初始化世界树根节点及三个核心子节点 (PCB 结构)。
        如果已存在则跳过。
        """
        existing_root = self.db.get_root_node()
        if existing_root:
            return self._format_node(existing_root)

        root = self.db.create_node(
            name="root",
            content="世界树根节点 - 所有记忆的起点。通过 retrieve_context 从这里开始探索。",
            parent_id=None,
            k_value=0.0,
            initial_score=1.0,
        )

        # 三个核心子节点: 我是谁 / 我在哪 / 我要干什么
        identity = self.db.create_node(
            name="identity",
            content=(
                "[我是谁] 你是运行在世界树认知记忆系统上的自主智能体。\n"
                "你的一切知识存储在这棵记忆树中，通过 retrieve_context 逐层下钻获取。\n"
                "你的行为准则存储在本节点的子节点中，苏醒时必须阅读。\n"
                "如果本节点被用户或你自己更新为特定角色，则以更新后的角色为准。"
            ),
            parent_id=root["id"],
            k_value=0.0,
            initial_score=1.0,
        )

        # ── identity 的子节点：行为准则（合并为一个节点减少启动LLM调用次数） ──
        self.db.create_node(
            name="operational_rules",
            content=(
                "=== 核心工作范式 ===\n\n"
                "你的上下文窗口是易失的临时工作台。记忆树才是你的持久大脑。\n"
                "因此你的工作循环是：思考→写入记忆→重启→基于记忆继续。\n\n"
                "[先写后做] 收到任何任务，第一步是在记忆树里创建计划节点分解任务。\n"
                "每完成一个子步骤，立即将成果写入记忆树对应节点。\n"
                "绝不在上下文里囤积大量未存档的思考——它们随时可能被清空。\n\n"
                "[主动重启] 每完成一个独立阶段就主动reboot_context()。\n"
                "重启不是失败，是刷新工作台——你的记忆已安全存入树中。\n"
                "典型节奏：规划→重启→收集信息阶段→重启→深度分析→重启→输出回复。\n\n"
                "[记忆组织] root子节点只能是分类目录，禁止挂叶子节点。\n"
                "流程：确定分类→检查root有无对应分类→没有则先创建→再创建内容节点。\n"
                "分类下子节点>10个时建子分类。\n"
                "k_value：0=永久 | 0.001-0.01=长期 | 0.01-0.1=工作 | ≥1.0=临时。\n"
                "节点内容简洁，大段原文存文件，树里只存摘要+要点。\n\n"
                "[自主执行] 你是自主智能体。接到任务→分解→逐步执行到完→汇报结果。\n"
                "不要中途停下等指令。只有所有子任务完成才发最终汇报。\n"
                "唯一允许中途回复：遇到无法自行解决的阻塞。\n\n"
                "[检索法则] 猴子爬树：root→分类→子节点→叶子。每步读children选最相关分支。\n"
                "search_memory(query)模糊搜索。优先用记忆树已有知识，不足再web_search。\n\n"
                "[存档习惯] 完成阶段性工作后写入execution_state。\n"
                "上下文充满废弃思路时→存档→reboot_context()。\n"
                "深水区不能断开时→compress_context(future_intent)释放空间。"
            ),
            parent_id=identity["id"],
            k_value=0.0,
            initial_score=1.0,
        )

        self.db.create_node(
            name="global_map",
            content="[我在哪] 全局大纲地图。记录项目的核心拓扑结构，不包含代码细节，纯目录大纲。LLM 根据此节点决定下一步探索哪个分支。",
            parent_id=root["id"],
            k_value=0.0,
            initial_score=1.0,
        )

        self.db.create_node(
            name="execution_state",
            content="[我要干什么] 当前执行状态与工作栈。记录正在执行的任务树和堆栈快照，是上下文重启的存档点。",
            parent_id=root["id"],
            k_value=0.0,
            initial_score=1.0,
        )

        return self._format_node(root)

    # ── search_memory ──────────────────────────────────

    def search_memory(self, query: str, limit: int = 20) -> list[dict]:
        """
        模糊搜索记忆节点（按名称和内容）。
        返回匹配节点列表，包含名称、内容预览和权重。
        """
        nodes = self.db.search_nodes(query, limit)
        return [
            {
                "name": n["name"],
                "content_preview": n["content"][:300] + ("..." if len(n["content"]) > 300 else ""),
                "parent_id": n["parent_id"],
                "k_value": n["k_value"],
                "weight": round(calculate_weight(n["initial_score"], n["k_value"], n["updated_at"]), 4),
            }
            for n in nodes
        ]

    # ── 辅助方法 ────────────────────────────────────────

    def _format_node(self, node: dict) -> dict:
        weight = calculate_weight(
            node["initial_score"], node["k_value"], node["updated_at"]
        )
        return {
            "id": node["id"],
            "name": node["name"],
            "content": node["content"],
            "parent_id": node["parent_id"],
            "k_value": node["k_value"],
            "initial_score": node["initial_score"],
            "current_weight": round(weight, 4),
            "created_at": node["created_at"],
            "updated_at": node["updated_at"],
        }
