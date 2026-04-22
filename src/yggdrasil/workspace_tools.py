"""
工作区文件工具 - 赋予 Agent 读写文件系统的能力
"""

import base64
import mimetypes
import os
import re
import subprocess
import shlex
from pathlib import Path

# 默认工作区根目录 (可通过环境变量覆盖)
WORKSPACE_ROOT = os.environ.get(
    "YGGDRASIL_WORKSPACE",
    str(Path.home() / "yggdrasil_workspace"),
)


def _safe_path(path: str) -> Path:
    """将用户路径解析为工作区内的安全绝对路径，防止路径穿越"""
    workspace = Path(WORKSPACE_ROOT).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    if os.path.isabs(path):
        target = Path(path).resolve()
    else:
        target = (workspace / path).resolve()

    # 路径穿越检查：目标必须在工作区内
    if not str(target).startswith(str(workspace)):
        raise PermissionError(f"路径 '{path}' 超出工作区范围 '{workspace}'")
    return target


def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> dict:
    """
    读取工作区中的文件内容。

    参数:
        path: 文件路径 (相对于工作区根目录，或绝对路径)
        start_line: 起始行号 (1-based，留空读全文)
        end_line: 结束行号 (1-based，含)
    """
    target = _safe_path(path)
    if not target.exists():
        raise FileNotFoundError(f"文件不存在: {target}")
    if not target.is_file():
        raise ValueError(f"不是文件: {target}")

    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    total = len(lines)

    if start_line is not None or end_line is not None:
        s = max((start_line or 1) - 1, 0)
        e = min(end_line or total, total)
        selected = lines[s:e]
        return {
            "path": str(target),
            "total_lines": total,
            "range": f"{s+1}-{e}",
            "content": "".join(selected),
        }

    return {
        "path": str(target),
        "total_lines": total,
        "content": text,
    }


def search_files(query: str, path: str | None = None, max_results: int = 30) -> dict:
    """
    在工作区中搜索包含指定文本/正则的文件。

    参数:
        query: 搜索文本或正则表达式
        path: 限定搜索的子目录 (相对于工作区)
        max_results: 最大结果数
    """
    workspace = Path(WORKSPACE_ROOT).resolve()
    search_root = _safe_path(path) if path else workspace

    if not search_root.exists():
        raise FileNotFoundError(f"搜索路径不存在: {search_root}")

    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error:
        pattern = re.compile(re.escape(query), re.IGNORECASE)

    matches = []
    for root_dir, _dirs, files in os.walk(search_root):
        for fname in files:
            if len(matches) >= max_results:
                break
            fpath = Path(root_dir) / fname
            # 跳过二进制/大文件
            if fpath.stat().st_size > 2 * 1024 * 1024:
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    matches.append({
                        "file": str(fpath.relative_to(workspace)),
                        "line": i,
                        "content": line.strip()[:200],
                    })
                    if len(matches) >= max_results:
                        break

    return {"query": query, "total_matches": len(matches), "matches": matches}


def list_directory(path: str | None = None) -> dict:
    """
    列出工作区中指定目录的内容。

    参数:
        path: 目录路径 (相对于工作区, 留空列出根目录)
    """
    target = _safe_path(path or ".")
    if not target.exists():
        raise FileNotFoundError(f"目录不存在: {target}")
    if not target.is_dir():
        raise ValueError(f"不是目录: {target}")

    entries = []
    for item in sorted(target.iterdir()):
        entry = {"name": item.name, "type": "dir" if item.is_dir() else "file"}
        if item.is_file():
            entry["size"] = item.stat().st_size
        entries.append(entry)

    return {
        "path": str(target),
        "entries": entries,
    }


def write_file(path: str, content: str) -> dict:
    """
    写入文件到工作区。自动创建父目录。

    参数:
        path: 文件路径 (相对于工作区)
        content: 文件内容
    """
    target = _safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {
        "path": str(target),
        "size": len(content),
        "message": f"已写入 {len(content)} 字符到 {target.name}",
    }


def append_file(path: str, content: str) -> dict:
    """
    追加内容到工作区文件末尾。文件不存在时自动创建。

    参数:
        path: 文件路径 (相对于工作区)
        content: 要追加的内容
    """
    target = _safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as f:
        f.write(content)
    return {
        "path": str(target),
        "appended": len(content),
        "message": f"已追加 {len(content)} 字符到 {target.name}",
    }


def delete_file(path: str) -> dict:
    """
    删除工作区中的文件或空目录。

    参数:
        path: 文件或空目录路径 (相对于工作区)
    """
    target = _safe_path(path)
    if not target.exists():
        raise FileNotFoundError(f"路径不存在: {target}")
    if target.is_file():
        target.unlink()
        return {"deleted": str(target), "type": "file"}
    elif target.is_dir():
        if any(target.iterdir()):
            raise PermissionError(f"目录非空，拒绝删除: {target}")
        target.rmdir()
        return {"deleted": str(target), "type": "directory"}
    else:
        raise ValueError(f"无法删除: {target}")


# 允许执行的命令白名单前缀
_ALLOWED_COMMANDS = {
    "python", "python3", "pip", "pip3",
    "node", "npm", "npx",
    "git", "ls", "dir", "cat", "head", "tail", "wc",
    "echo", "grep", "find", "sort", "uniq",
    "pytest", "unittest",
}


def execute_command(command: str, timeout: int = 30) -> dict:
    """
    在工作区目录下执行 shell 命令。
    有命令白名单限制，超时默认 30 秒，最长 120 秒。

    参数:
        command: 要执行的命令 (如 "python test.py", "ls -la")
        timeout: 超时秒数 (默认30, 最长120)
    """
    timeout = min(max(timeout, 1), 120)

    # 安全检查：提取首个命令词
    parts = shlex.split(command, posix=(os.name != "nt"))
    if not parts:
        return {"error": "空命令"}
    cmd_name = Path(parts[0]).stem.lower()

    if cmd_name not in _ALLOWED_COMMANDS:
        return {"error": f"命令 '{cmd_name}' 不在白名单中。允许的命令: {sorted(_ALLOWED_COMMANDS)}"}

    # 危险参数检查
    dangerous = {"--force", "-rf", "rm", "del", "rmdir", "format", "mkfs"}
    if any(d in parts for d in dangerous):
        return {"error": "检测到危险参数，拒绝执行"}

    workspace = Path(WORKSPACE_ROOT).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        stdout = result.stdout[:8000] if result.stdout else ""
        stderr = result.stderr[:4000] if result.stderr else ""
        output = {
            "exit_code": result.returncode,
            "stdout": stdout,
        }
        if stderr:
            output["stderr"] = stderr
        if len(result.stdout or "") > 8000:
            output["stdout_truncated"] = True
        return output
    except subprocess.TimeoutExpired:
        return {"error": f"命令超时 ({timeout}s)", "timeout": timeout}
    except Exception as e:
        return {"error": f"执行失败: {e}"}


# 支持的图片 MIME 类型
_IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
}


def view_image(path: str, max_size_mb: float = 5.0) -> dict:
    """
    读取工作区中的图片文件，返回 base64 data URL。
    可用于将图片传递给多模态模型分析。

    参数:
        path: 图片文件路径 (相对于工作区)
        max_size_mb: 最大文件大小 (MB, 默认5)
    """
    target = _safe_path(path)
    if not target.exists():
        raise FileNotFoundError(f"图片不存在: {target}")
    if not target.is_file():
        raise ValueError(f"不是文件: {target}")

    suffix = target.suffix.lower()
    mime = _IMAGE_MIME_TYPES.get(suffix)
    if not mime:
        raise ValueError(f"不支持的图片格式 '{suffix}'。支持: {list(_IMAGE_MIME_TYPES.keys())}")

    size = target.stat().st_size
    if size > max_size_mb * 1024 * 1024:
        raise ValueError(f"图片过大: {size / 1024 / 1024:.1f}MB (上限 {max_size_mb}MB)")

    raw = target.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    return {
        "path": str(target),
        "mime_type": mime,
        "size": size,
        "data_url": data_url,
    }


def save_uploaded_image(data_url: str, save_dir: str) -> str:
    """
    将 base64 data URL 保存为图片文件。
    返回保存后的文件路径。

    参数:
        data_url: data:image/...;base64,... 格式的 URL
        save_dir: 保存目录 (绝对路径)
    """
    import uuid

    # 解析 data URL
    if not data_url.startswith("data:"):
        raise ValueError("无效的 data URL：必须以 data: 开头")

    header, _, b64_data = data_url.partition(",")
    if not b64_data:
        raise ValueError("无效的 data URL：缺少 base64 数据")

    # 提取 mime type
    # header 格式: data:image/png;base64
    mime_part = header.split(";")[0].replace("data:", "")
    ext = mimetypes.guess_extension(mime_part) or ".png"
    # mimetypes 对 jpeg 返回 .jpeg
    if ext == ".jpe":
        ext = ".jpg"

    raw = base64.b64decode(b64_data)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex[:12]}{ext}"
    filepath = save_path / filename
    filepath.write_bytes(raw)

    return str(filepath)
