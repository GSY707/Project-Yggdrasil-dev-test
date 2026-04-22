# Project Yggdrasil (世界树项目) — 认知记忆引擎

LLM 智能体的确定性记忆系统，基于树图混合数据结构 + MCP 协议。

## 架构核心

- **树图混合存储**：树提供精确的层级寻址，图提供跨模块关联
- **猴子爬树检索**：`retrieve_context` 逐层下钻，100% 确定性路径
- **量化软遗忘**：$Weight = Score_{initial} \times e^{-k \cdot \Delta t}$
- **Git 式版本控制**：每次更新自动保存历史快照
- **泛型工具路由**：一个 `generic_tool_executor` 入口调用无限工具

## 快速启动

```bash
cd src && python -m yggdrasil.async_web
```

## MCP 工具清单

| 工具 | 用途 |
|------|------|
| `retrieve_context` | 猴子爬树 — 检索节点上下文、子节点列表 |
| `create_memory_node` | 创建记忆节点（指定 k 值衰减系数） |
| `update` | Git 式更新节点（自动保存历史快照） |
| `hard_delete` | 物理删除节点及子树 |
| `node_history` | 时间漫游 — 查询节点版本历史 |
| `generic_tool_executor` | 泛型工具路由 — 调用记忆树中注册的任意工具 |

## 项目结构

```
src/
  yggdrasil/
    __init__.py          # 包标识
    server.py            # MCP Server 入口 + 工具定义
    memory_engine.py     # 核心 CRUD 引擎
    database.py          # SQLite 持久化层
    decay.py             # 衰减公式 (指数衰减模型)
    tool_router.py       # 泛型工具分发器
  test_yggdrasil.py      # 核心功能测试
  pyproject.toml         # 项目配置
```

## 根节点结构 (PCB)

启动时自动初始化：

```
root (世界树根节点)
├── identity       [我是谁] 认知角色定义
├── global_map     [我在哪] 全局大纲地图
└── execution_state [我要干什么] 执行状态与工作栈
```

## 配置

通过环境变量 `YGGDRASIL_DB_PATH` 指定数据库路径，默认为 `src/yggdrasil.db`。
