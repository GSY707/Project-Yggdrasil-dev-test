"""
自举引擎 (Bootstrap Engine)
实现架构蓝图第五章：自我进化引擎

核心流程：
  1. Agent 感知到工具/代码问题 → 挂起业务任务
  2. Agent 阅读自身源码 → 修改 → 写入临时分支
  3. 沙箱测试：在隔离环境中跑自动化测试 (pytest)
  4. A/B 评测：Evaluator Agent 对比新旧版本
  5. 判定通过 → 合并；失败 → 回滚 + 反馈
  6. 连续 N 次被驳回 → ask_human 求助
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EvolutionStatus(str, Enum):
    """自举进化任务状态"""
    PROPOSED = "proposed"        # 提案已提交
    TESTING = "testing"          # 沙箱测试中
    EVALUATING = "evaluating"    # 法官评估中
    APPROVED = "approved"        # 已批准
    REJECTED = "rejected"        # 已驳回
    MERGED = "merged"            # 已合并
    ROLLED_BACK = "rolled_back"  # 已回滚


@dataclass
class EvolutionProposal:
    """一次自举进化提案"""
    id: str
    description: str               # 修改描述
    target_file: str               # 要修改的源文件 (相对于项目根目录)
    original_content: str          # 原始文件内容
    modified_content: str          # 修改后的文件内容
    reason: str                    # 修改原因
    status: EvolutionStatus = EvolutionStatus.PROPOSED
    test_results: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    rejection_count: int = 0       # 累计驳回次数


# ── 辅助工具 ─────────────────────────────────────────


def _file_to_import(target_file: str) -> str | None:
    """将目标文件路径转换为 Python 模块导入路径。
    例: "src/yggdrasil/decay.py" → "yggdrasil.decay"
         "portable_llm/providers/gemini.py" → "portable_llm.providers.gemini"
    非 .py 文件返回 None。"""
    p = Path(target_file)
    if p.suffix != ".py" or p.name.startswith("__"):
        return None
    parts = list(p.with_suffix("").parts)
    # 去掉 "src" 前缀
    if parts and parts[0] == "src":
        parts = parts[1:]
    return ".".join(parts) if parts else None


# ── 项目自省 ─────────────────────────────────────────


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).resolve().parent.parent.parent


def get_source_layout() -> dict:
    """获取项目源代码布局，供 Agent 了解自身结构。"""
    root = get_project_root()
    layout = {
        "project_root": str(root),
        "source_modules": {},
        "feature_docs": [],
    }

    # 扫描核心模块
    src_dir = root / "src" / "yggdrasil"
    if src_dir.exists():
        for py_file in sorted(src_dir.glob("*.py")):
            if py_file.name.startswith("__"):
                continue
            stat = py_file.stat()
            layout["source_modules"][py_file.name] = {
                "path": str(py_file.relative_to(root)),
                "size": stat.st_size,
                "lines": sum(1 for _ in py_file.open(encoding="utf-8", errors="replace")),
            }

    # 扫描 portable_llm 模块
    llm_dir = root / "portable_llm"
    if llm_dir.exists():
        for py_file in sorted(llm_dir.rglob("*.py")):
            if py_file.name.startswith("__"):
                continue
            stat = py_file.stat()
            layout["source_modules"][str(py_file.relative_to(root))] = {
                "path": str(py_file.relative_to(root)),
                "size": stat.st_size,
                "lines": sum(1 for _ in py_file.open(encoding="utf-8", errors="replace")),
            }

    # 扫描特性文档
    docs_dir = root / "特性"
    if docs_dir.exists():
        for md_file in sorted(docs_dir.glob("*.md")):
            layout["feature_docs"].append(md_file.name)

    return layout


def read_source_file(relative_path: str) -> str:
    """读取项目源文件内容（用于自省）。"""
    root = get_project_root()
    target = (root / relative_path).resolve()

    # 安全检查：必须在项目根目录内
    if not str(target).startswith(str(root)):
        raise PermissionError(f"路径 '{relative_path}' 超出项目范围")
    if not target.exists():
        raise FileNotFoundError(f"源文件不存在: {relative_path}")
    if not target.is_file():
        raise ValueError(f"不是文件: {relative_path}")

    return target.read_text(encoding="utf-8")


# ── 沙箱测试 ─────────────────────────────────────────


def run_sandbox_test(
    proposal: EvolutionProposal,
    test_commands: list[str] | None = None,
    timeout: int = 120,
) -> dict:
    """
    在隔离的临时目录中测试代码修改。

    流程：
    1. 复制项目到临时目录
    2. 应用修改
    3. 执行测试命令
    4. 返回测试结果
    """
    root = get_project_root()
    results = {
        "success": False,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_error": 0,
        "stdout": "",
        "stderr": "",
        "exit_code": -1,
    }

    # 创建沙箱目录
    sandbox_dir = Path(tempfile.mkdtemp(prefix="ygg_sandbox_"))
    try:
        # 复制核心源代码到沙箱（排除 data/、__pycache__/、.db 文件）
        _copy_project_to_sandbox(root, sandbox_dir)

        # 应用修改
        target_file = sandbox_dir / proposal.target_file
        if not target_file.parent.exists():
            target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(proposal.modified_content, encoding="utf-8")

        # 默认测试命令
        if not test_commands:
            # 自动构建导入检查：确保修改后的文件能正常导入
            import_module = _file_to_import(proposal.target_file)
            test_commands = []
            if import_module:
                test_commands.append(
                    f'python -c "import sys; sys.path.insert(0, \'src\'); sys.path.insert(0, \'.\'); import {import_module}"'
                )
            else:
                test_commands.append(
                    f'python -c "import sys; sys.path.insert(0, \'src\'); sys.path.insert(0, \'.\'); import yggdrasil"'
                )
            # 语法检查
            test_commands.append(f'python -m py_compile "{proposal.target_file}"')
            # pytest — 仅在有实际测试文件时运行
            tests_dir = sandbox_dir / "tests"
            if tests_dir.exists() and any(tests_dir.rglob("test_*.py")):
                test_commands.append("python -m pytest tests/ -x -q --tb=short")
            test_commands = [c for c in test_commands if c]

        # 执行测试
        combined_stdout = []
        combined_stderr = []
        all_passed = True

        for cmd in test_commands:
            try:
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=str(sandbox_dir),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(sandbox_dir / "src")},
                )
                combined_stdout.append(f"$ {cmd}\n{proc.stdout}")
                if proc.stderr:
                    combined_stderr.append(f"$ {cmd}\n{proc.stderr}")
                if proc.returncode != 0:
                    # pytest exit_code 5 = "no tests collected" → 不视为失败
                    if proc.returncode == 5 and "no tests ran" in (proc.stdout + proc.stderr).lower():
                        logger.info(f"[Sandbox] pytest returned 5 (no tests collected), treating as pass")
                    else:
                        all_passed = False
                        results["exit_code"] = proc.returncode
            except subprocess.TimeoutExpired:
                combined_stderr.append(f"$ {cmd}\n[TIMEOUT after {timeout}s]")
                all_passed = False

        results["success"] = all_passed
        results["stdout"] = "\n".join(combined_stdout)[:8000]
        results["stderr"] = "\n".join(combined_stderr)[:4000]
        if all_passed:
            results["exit_code"] = 0

        # 解析 pytest 输出
        for line in results["stdout"].split("\n"):
            if "passed" in line:
                try:
                    parts = line.strip().split()
                    for i, p in enumerate(parts):
                        if p == "passed":
                            results["tests_passed"] = int(parts[i - 1])
                        elif p == "failed":
                            results["tests_failed"] = int(parts[i - 1])
                        elif p == "error" or p == "errors":
                            results["tests_error"] = int(parts[i - 1])
                except (ValueError, IndexError):
                    pass

    finally:
        # 清理沙箱
        shutil.rmtree(sandbox_dir, ignore_errors=True)

    return results


def _copy_project_to_sandbox(src_root: Path, dest: Path):
    """复制项目到沙箱，排除不必要的文件。"""
    exclude_dirs = {"__pycache__", ".git", "data", "node_modules", ".venv", "venv", "static"}
    exclude_extensions = {".db", ".pyc", ".pyo"}

    for item in src_root.iterdir():
        if item.name in exclude_dirs:
            continue
        if item.name.startswith(".") and item.is_dir():
            continue

        dest_item = dest / item.name
        if item.is_dir():
            shutil.copytree(
                item, dest_item,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.db", "data", ".git"),
            )
        elif item.is_file():
            if item.suffix not in exclude_extensions:
                dest_item.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest_item)


# ── A/B 评测 ─────────────────────────────────────────


def build_evaluation_prompt(
    proposal: EvolutionProposal,
    test_results: dict,
) -> str:
    """
    构建给 Evaluator Agent 的评测提示词。
    """
    return f"""你是世界树认知系统的架构评审委员会主席。你的任务是严格评估一次代码修改是否应该被合并。

## 修改概要

- **描述**: {proposal.description}
- **修改原因**: {proposal.reason}
- **目标文件**: {proposal.target_file}

## 测试结果

- **测试通过**: {"✅ 全部通过" if test_results.get("success") else "❌ 存在失败"}
- **通过数**: {test_results.get("tests_passed", "N/A")}
- **失败数**: {test_results.get("tests_failed", "N/A")}
- **错误数**: {test_results.get("tests_error", "N/A")}

### 测试输出
```
{test_results.get("stdout", "")[:3000]}
```

### 错误日志
```
{test_results.get("stderr", "")[:2000]}
```

## 代码差异

### 修改前 (关键片段)
```python
{_extract_diff_context(proposal.original_content, proposal.modified_content, "before")[:3000]}
```

### 修改后 (关键片段)
```python
{_extract_diff_context(proposal.original_content, proposal.modified_content, "after")[:3000]}
```

## 评估维度

请从以下三个维度进行评估，每个维度 1-10 分：

1. **参数易用性** (Parameter Usability): 修改后的 API/参数是否更容易正确使用？
2. **容错能力** (Error Tolerance): 修改是否增强了错误处理和边界情况防护？
3. **代码简洁性** (Code Simplicity): 修改是否遵循奥卡姆剃刀原则，避免不必要的复杂度？

## 输出格式

请严格输出以下 JSON 格式：
```json
{{
  "verdict": "approve" 或 "reject",
  "scores": {{
    "parameter_usability": <1-10>,
    "error_tolerance": <1-10>,
    "code_simplicity": <1-10>
  }},
  "total_score": <三项平均>,
  "reasoning": "你的判断理由（2-3 句话）",
  "suggestions": "如果驳回，给出具体的改进建议"
}}
```

## 约束

- 测试未通过 → 必须驳回
- 总分低于 6.0 → 建议驳回
- 引入明显安全漏洞 → 必须驳回
- 代码复杂度显著增加但收益不明 → 驳回"""


def _extract_diff_context(original: str, modified: str, which: str) -> str:
    """提取差异上下文（简单实现：找不同的行）"""
    orig_lines = original.splitlines()
    mod_lines = modified.splitlines()

    # 找到差异起始行
    diff_start = 0
    for i, (a, b) in enumerate(zip(orig_lines, mod_lines)):
        if a != b:
            diff_start = max(0, i - 5)
            break
    else:
        # 行数不同
        diff_start = max(0, min(len(orig_lines), len(mod_lines)) - 5)

    diff_end = diff_start + 40  # 最多展示 40 行上下文

    if which == "before":
        return "\n".join(orig_lines[diff_start:diff_end])
    else:
        return "\n".join(mod_lines[diff_start:diff_end])


def evaluate_proposal(proposal: EvolutionProposal, llm_client, test_results: dict) -> dict:
    """
    调用 Evaluator Agent（LLM 法官）评估修改提案。

    返回评测结果 dict: {verdict, scores, reasoning, suggestions}
    """
    prompt = build_evaluation_prompt(proposal, test_results)

    response = llm_client.generate(
        system_prompt="你是一个严格的代码评审系统。只输出 JSON，不要有任何额外解释。",
        user_prompt=prompt,
        temperature=0.1,
    )

    # 解析评测结果
    try:
        # 尝试从回复中提取 JSON
        text = response.text.strip()
        # 处理 markdown code block 包裹
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        result = json.loads(text)

        # 验证必要字段
        if "verdict" not in result:
            result["verdict"] = "reject"
        if "scores" not in result:
            result["scores"] = {}
        if "reasoning" not in result:
            result["reasoning"] = "评测解析失败"

        return result

    except (json.JSONDecodeError, IndexError):
        logger.warning(f"Failed to parse evaluation response: {response.text[:200]}")
        return {
            "verdict": "reject",
            "scores": {},
            "reasoning": f"评测结果解析失败，原始回复: {response.text[:500]}",
            "suggestions": "请重新提交提案。",
        }


# ── 完整自举流程 ─────────────────────────────────────


def execute_evolution(
    proposal: EvolutionProposal,
    llm_client,
    test_commands: list[str] | None = None,
    max_rejections: int = 3,
) -> EvolutionProposal:
    """
    执行完整的自举进化流程：测试 → 评估 → 合并/回滚。

    参数:
        proposal: 进化提案
        llm_client: LLM 客户端 (UnifiedLLMClient)
        test_commands: 自定义测试命令
        max_rejections: 最大驳回次数，超过后触发 ask_human

    返回:
        更新后的 proposal (含 status, test_results, evaluation)
    """
    # 1. 沙箱测试
    proposal.status = EvolutionStatus.TESTING
    logger.info(f"[Bootstrap] Testing proposal: {proposal.description}")

    test_results = run_sandbox_test(proposal, test_commands)
    proposal.test_results = test_results

    if not test_results["success"]:
        proposal.status = EvolutionStatus.REJECTED
        proposal.rejection_count += 1
        proposal.evaluation = {
            "verdict": "reject",
            "reasoning": f"测试未通过: exit_code={test_results['exit_code']}",
            "suggestions": test_results.get("stderr", "")[:1000],
        }
        logger.warning(f"[Bootstrap] Tests failed for: {proposal.description}")
        return proposal

    # 2. A/B 评测
    proposal.status = EvolutionStatus.EVALUATING
    logger.info(f"[Bootstrap] Evaluating proposal: {proposal.description}")

    evaluation = evaluate_proposal(proposal, llm_client, test_results)
    proposal.evaluation = evaluation

    # 3. 判定
    if evaluation.get("verdict") == "approve":
        proposal.status = EvolutionStatus.APPROVED
        logger.info(f"[Bootstrap] Proposal APPROVED: {proposal.description}")
    else:
        proposal.status = EvolutionStatus.REJECTED
        proposal.rejection_count += 1
        logger.info(f"[Bootstrap] Proposal REJECTED (#{proposal.rejection_count}): {evaluation.get('reasoning', '')}")

    return proposal


def apply_approved_proposal(proposal: EvolutionProposal) -> bool:
    """
    将批准的提案实际应用到项目源码中。
    只在 proposal.status == APPROVED 时调用。

    返回是否成功。
    """
    if proposal.status != EvolutionStatus.APPROVED:
        logger.error(f"Cannot apply non-approved proposal: {proposal.status}")
        return False

    root = get_project_root()
    target = root / proposal.target_file

    # 备份原始文件
    backup_path = target.with_suffix(target.suffix + ".bak")
    try:
        if target.exists():
            shutil.copy2(target, backup_path)

        # 写入修改
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(proposal.modified_content, encoding="utf-8")

        proposal.status = EvolutionStatus.MERGED
        logger.info(f"[Bootstrap] Merged proposal to {proposal.target_file}")

        # 清理备份
        if backup_path.exists():
            backup_path.unlink()

        return True

    except Exception as e:
        # 回滚
        logger.error(f"[Bootstrap] Failed to apply proposal: {e}")
        if backup_path.exists():
            shutil.copy2(backup_path, target)
            backup_path.unlink()
        proposal.status = EvolutionStatus.ROLLED_BACK
        return False


def rollback_proposal(proposal: EvolutionProposal) -> bool:
    """回滚已合并的提案。"""
    if proposal.status != EvolutionStatus.MERGED:
        return False

    root = get_project_root()
    target = root / proposal.target_file

    try:
        target.write_text(proposal.original_content, encoding="utf-8")
        proposal.status = EvolutionStatus.ROLLED_BACK
        logger.info(f"[Bootstrap] Rolled back {proposal.target_file}")
        return True
    except Exception as e:
        logger.error(f"[Bootstrap] Rollback failed: {e}")
        return False
