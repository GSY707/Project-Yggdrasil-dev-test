# Project Yggdrasil（世界树项目）

[English](README.en.md) | 中文（当前）

面向 LLM 智能体的确定性记忆树引擎。

这个项目从T-Graph memory system项目的基础上继续开发，使LLM可以完成远超原本上下文窗口的任务

## 仓库内容

- `src/`：Yggdrasil 记忆引擎与 Web 服务
- `portable_llm/`：可独立复用的多 Provider LLM 调用包
- `prompts/`：系统提示词与上下文压缩提示词
- `特性/`：架构设计与方案文档

## 快速启动

1. 进入 `src/` 并安装依赖。
2. 启动异步 Web 服务：

   `cd src && python -m yggdrasil.async_web`

3. 打开 `http://localhost:8000`。

核心引擎与工具说明见 [src/README.md](src/README.md)。

## 开源协议

本项目采用 GNU Affero General Public License v3.0（AGPL-3.0）。

详见 [LICENSE](LICENSE)。

## 安全披露

如发现安全问题，请不要公开提交细节，披露流程见 [SECURITY.md](SECURITY.md)。

## 贡献指南

欢迎提交 Issue 和 Pull Request，贡献前请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。
