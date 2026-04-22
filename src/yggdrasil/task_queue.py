"""
异步任务队列 - 管理后台并发任务
使用 asyncio + ThreadPoolExecutor 桥接同步 LLM 调用
"""

import asyncio
import logging
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskRecord:
    id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    progress: str = ""

    @property
    def elapsed(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at or time.time()
        return end - self.started_at


class TaskQueue:
    """
    后台任务管理器。
    - 同步函数通过 ThreadPoolExecutor 跑在线程里
    - 异步函数直接在事件循环中调度
    - 所有任务可通过 ID 查询状态和结果
    """

    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ygg-task")
        self._tasks: dict[str, TaskRecord] = {}
        self._futures: dict[str, asyncio.Task] = {}
        self._event_callback: Callable | None = None

    def on_task_event(self, callback: Callable):
        """注册任务事件回调 (status changes)"""
        self._event_callback = callback

    def _notify(self, task: TaskRecord, event_type: str = "task_update"):
        if self._event_callback:
            try:
                self._event_callback(event_type, {
                    "task_id": task.id,
                    "name": task.name,
                    "status": task.status.value,
                    "progress": task.progress,
                    "error": task.error,
                    "elapsed": task.elapsed,
                })
            except Exception as e:
                logger.error(f"Task event callback error: {e}")

    def submit_sync(self, name: str, fn: Callable, *args, **kwargs) -> str:
        """
        提交一个同步函数作为后台任务 (在线程池中执行)。
        返回 task_id。
        """
        task_id = str(uuid.uuid4())[:8]
        record = TaskRecord(id=task_id, name=name)
        self._tasks[task_id] = record
        self._notify(record, "task_submitted")

        loop = asyncio.get_event_loop()

        async def _run():
            record.status = TaskStatus.RUNNING
            record.started_at = time.time()
            self._notify(record, "task_started")
            try:
                result = await loop.run_in_executor(self._executor, lambda: fn(*args, **kwargs))
                record.status = TaskStatus.COMPLETED
                record.result = result
            except Exception as e:
                record.status = TaskStatus.FAILED
                record.error = str(e)
                logger.error(f"Task {task_id} ({name}) failed: {e}", exc_info=True)
            finally:
                record.finished_at = time.time()
                self._notify(record, "task_finished")

        self._futures[task_id] = asyncio.ensure_future(_run())
        return task_id

    def submit_async(self, name: str, coro: Coroutine) -> str:
        """
        提交一个协程作为后台任务。
        返回 task_id。
        """
        task_id = str(uuid.uuid4())[:8]
        record = TaskRecord(id=task_id, name=name)
        self._tasks[task_id] = record
        self._notify(record, "task_submitted")

        async def _run():
            record.status = TaskStatus.RUNNING
            record.started_at = time.time()
            self._notify(record, "task_started")
            try:
                result = await coro
                record.status = TaskStatus.COMPLETED
                record.result = result
            except Exception as e:
                record.status = TaskStatus.FAILED
                record.error = str(e)
                logger.error(f"Task {task_id} ({name}) failed: {e}", exc_info=True)
            finally:
                record.finished_at = time.time()
                self._notify(record, "task_finished")

        self._futures[task_id] = asyncio.ensure_future(_run())
        return task_id

    def get_task(self, task_id: str) -> TaskRecord | None:
        return self._tasks.get(task_id)

    def list_tasks(self, status: TaskStatus | None = None) -> list[TaskRecord]:
        tasks = list(self._tasks.values())
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def cancel(self, task_id: str) -> bool:
        future = self._futures.get(task_id)
        record = self._tasks.get(task_id)
        if future and not future.done():
            future.cancel()
            if record:
                record.status = TaskStatus.CANCELLED
                record.finished_at = time.time()
                self._notify(record, "task_cancelled")
            return True
        return False

    def update_progress(self, task_id: str, progress: str):
        record = self._tasks.get(task_id)
        if record:
            record.progress = progress
            self._notify(record, "task_progress")

    def cleanup_finished(self, max_age_seconds: float = 3600):
        """清理已完成超过指定时间的任务记录"""
        now = time.time()
        to_remove = [
            tid for tid, t in self._tasks.items()
            if t.finished_at and (now - t.finished_at) > max_age_seconds
        ]
        for tid in to_remove:
            del self._tasks[tid]
            self._futures.pop(tid, None)

    async def shutdown(self):
        """关闭任务队列"""
        for tid, future in self._futures.items():
            if not future.done():
                future.cancel()
        self._executor.shutdown(wait=False)
