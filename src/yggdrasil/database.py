"""
数据库层 - SQLite 存储引擎
负责节点、边、版本历史的持久化存储
"""

import sqlite3
import json
import os
from pathlib import Path
from contextlib import contextmanager


class OptimisticLockError(Exception):
    """乐观并发控制版本冲突异常"""
    def __init__(self, message: str, current_node: dict | None = None):
        super().__init__(message)
        self.current_node = current_node

# 默认数据库路径：项目根目录下的 yggdrasil.db
DEFAULT_DB_PATH = os.environ.get(
    "YGGDRASIL_DB_PATH",
    str(Path(__file__).parent.parent / "yggdrasil.db"),
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS nodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    content     TEXT    NOT NULL DEFAULT '',
    parent_id   INTEGER,
    k_value     REAL    NOT NULL DEFAULT 0.0,
    initial_score REAL  NOT NULL DEFAULT 1.0,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (parent_id) REFERENCES nodes(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_node_id  INTEGER NOT NULL,
    target_node_id  INTEGER NOT NULL,
    label           TEXT    NOT NULL DEFAULT 'RELATED_TO',
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (source_node_id) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_node_id) REFERENCES nodes(id) ON DELETE CASCADE,
    UNIQUE(source_node_id, target_node_id, label)
);

CREATE TABLE IF NOT EXISTS node_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id     INTEGER NOT NULL,
    version     INTEGER NOT NULL,
    name        TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    parent_id   INTEGER,
    k_value     REAL    NOT NULL,
    initial_score REAL  NOT NULL,
    edges_snapshot TEXT  NOT NULL DEFAULT '[]',
    recorded_at TEXT    NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_node_id);
CREATE INDEX IF NOT EXISTS idx_history_node ON node_history(node_id);
"""


class Database:
    """SQLite 数据库封装，提供节点/边/历史的原子操作"""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _tx(self):
        """事务上下文管理器"""
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self):
        with self._tx() as conn:
            conn.executescript(SCHEMA_SQL)
            # 增量迁移：并发控制 - 添加 version 列
            try:
                conn.execute("ALTER TABLE nodes ADD COLUMN version INTEGER NOT NULL DEFAULT 1")
            except sqlite3.OperationalError:
                pass  # 列已存在

    # ── 节点操作 ──────────────────────────────────────────

    def create_node(
        self,
        name: str,
        content: str,
        parent_id: int | None,
        k_value: float = 0.0,
        initial_score: float = 1.0,
    ) -> dict:
        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO nodes (name, content, parent_id, k_value, initial_score)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, content, parent_id, k_value, initial_score),
            )
            node_id = cur.lastrowid
            row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
            return dict(row)

    def get_node_by_name(self, name: str) -> dict | None:
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM nodes WHERE name = ?", (name,)).fetchone()
            return dict(row) if row else None

    def get_node_by_id(self, node_id: int) -> dict | None:
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
            return dict(row) if row else None

    def get_children(self, parent_id: int) -> list[dict]:
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT * FROM nodes WHERE parent_id = ?", (parent_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def update_node(self, node_id: int, **fields) -> dict:
        """更新节点，带乐观并发控制。
        如传入 expected_version，仅当版本匹配时才更新（CAS 语义）。
        不传 expected_version 时无条件更新（向后兼容）。
        """
        expected_version = fields.pop("expected_version", None)
        allowed = {"name", "content", "parent_id", "k_value", "initial_score"}
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return self.get_node_by_id(node_id)
        # version 自增
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        set_clause += ", updated_at = datetime('now'), version = version + 1"
        values = list(updates.values()) + [node_id]
        with self._tx() as conn:
            if expected_version is not None:
                # 乐观并发：检查版本号
                values_with_ver = list(updates.values()) + [node_id, expected_version]
                set_clause_ver = ", ".join(f"{k} = ?" for k in updates)
                set_clause_ver += ", updated_at = datetime('now'), version = version + 1"
                affected = conn.execute(
                    f"UPDATE nodes SET {set_clause_ver} WHERE id = ? AND version = ?",
                    values_with_ver,
                ).rowcount
                if affected == 0:
                    # 版本冲突：返回当前最新版本让调用者决策
                    current = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
                    if current is None:
                        raise ValueError(f"节点 id={node_id} 不存在")
                    raise OptimisticLockError(
                        f"版本冲突：期望 version={expected_version}，当前 version={current['version']}",
                        current_node=dict(current),
                    )
            else:
                conn.execute(f"UPDATE nodes SET {set_clause} WHERE id = ?", values)
            row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
            return dict(row)

    def delete_node(self, node_id: int) -> bool:
        with self._tx() as conn:
            # 级联删除子节点
            children = conn.execute(
                "SELECT id FROM nodes WHERE parent_id = ?", (node_id,)
            ).fetchall()
            for child in children:
                self.delete_node(child["id"])
            conn.execute("DELETE FROM edges WHERE source_node_id = ? OR target_node_id = ?", (node_id, node_id))
            conn.execute("DELETE FROM node_history WHERE node_id = ?", (node_id,))
            affected = conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,)).rowcount
            return affected > 0

    def get_root_node(self) -> dict | None:
        with self._tx() as conn:
            row = conn.execute(
                "SELECT * FROM nodes WHERE parent_id IS NULL ORDER BY id LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    # ── 边操作 ──────────────────────────────────────────

    def create_edge(self, source_id: int, target_id: int, label: str = "RELATED_TO") -> dict:
        with self._tx() as conn:
            cur = conn.execute(
                """INSERT OR IGNORE INTO edges (source_node_id, target_node_id, label)
                   VALUES (?, ?, ?)""",
                (source_id, target_id, label),
            )
            row = conn.execute("SELECT * FROM edges WHERE id = ?", (cur.lastrowid,)).fetchone()
            return dict(row) if row else {}

    def get_outgoing_edges(self, node_id: int) -> list[dict]:
        with self._tx() as conn:
            rows = conn.execute(
                """SELECT e.*, n.name AS target_name
                   FROM edges e JOIN nodes n ON e.target_node_id = n.id
                   WHERE e.source_node_id = ?""",
                (node_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_incoming_edges(self, node_id: int) -> list[dict]:
        with self._tx() as conn:
            rows = conn.execute(
                """SELECT e.*, n.name AS source_name
                   FROM edges e JOIN nodes n ON e.source_node_id = n.id
                   WHERE e.target_node_id = ?""",
                (node_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_edges_for_node(self, node_id: int):
        with self._tx() as conn:
            conn.execute(
                "DELETE FROM edges WHERE source_node_id = ? OR target_node_id = ?",
                (node_id, node_id),
            )

    # ── 版本历史 ──────────────────────────────────────────

    def save_history_snapshot(self, node_id: int, node: dict, edges: list[dict]):
        with self._tx() as conn:
            # 获取当前最大版本号
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS max_ver FROM node_history WHERE node_id = ?",
                (node_id,),
            ).fetchone()
            new_version = row["max_ver"] + 1
            edges_json = json.dumps(
                [{"target": e.get("target_name", ""), "label": e.get("label", "")} for e in edges],
                ensure_ascii=False,
            )
            conn.execute(
                """INSERT INTO node_history
                   (node_id, version, name, content, parent_id, k_value, initial_score, edges_snapshot)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    node_id, new_version, node["name"], node["content"],
                    node["parent_id"], node["k_value"], node["initial_score"],
                    edges_json,
                ),
            )
            return new_version

    def get_history(self, node_id: int) -> list[dict]:
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT * FROM node_history WHERE node_id = ? ORDER BY version DESC",
                (node_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── 搜索 ──────────────────────────────────────────

    def search_nodes(self, query: str, limit: int = 20) -> list[dict]:
        """按名称或内容模糊搜索节点"""
        pattern = f"%{query}%"
        with self._tx() as conn:
            rows = conn.execute(
                """SELECT * FROM nodes
                   WHERE name LIKE ? OR content LIKE ?
                   ORDER BY updated_at DESC
                   LIMIT ?""",
                (pattern, pattern, limit),
            ).fetchall()
            return [dict(r) for r in rows]
