# portable_llm

这是一个可直接拷贝到其他 Python 项目中的独立 LLM 调用包，来源于本项目现有的统一调用层，但已经做成了可搬运版本。

你只需要把整个 portable_llm 文件夹复制到目标项目里，然后安装依赖、补一个 keys.yaml 或设置环境变量，就可以直接调用。

## 包含内容

- `__init__.py`：统一导出入口。
- `llm_client.py`：统一客户端，负责模型到 provider 的路由、重试、会话日志。
- `llm_models.py`：响应对象和 token 用量结构。
- `error_handler.py`：错误分类、冷却、退避重试。
- `gcloud_auth.py`：Gemini Vertex 路径下的 gcloud 鉴权。
- `providers/`：DeepSeek、Gemini、LongCat、VectorEngine 适配层。
- `keys.yaml.example`：密钥模板。
- `config.example.yaml`：最小配置模板。
- `requirements.txt`：最小依赖列表。
- `example_usage.py`：最小调用示例。

## 复制方式

把整个 `portable_llm` 文件夹复制到目标项目根目录。

目标项目结构示例：

```text
your_project/
  portable_llm/
    __init__.py
    llm_client.py
    ...
  app.py
```

然后你可以直接这样导入：

```python
from portable_llm import UnifiedLLMClient
```

## 安装依赖

```bash
pip install -r portable_llm/requirements.txt
```

如果你需要 `GenericVannaLLM`，再额外安装：

```bash
pip install vanna
```

## 配置方式

支持两种方式。

### 方式 1：使用 keys.yaml

复制 `portable_llm/keys.yaml.example` 为 `portable_llm/keys.yaml`，再填入真实密钥。

`keys.yaml` 只用于本地开发，不应提交到代码仓库。项目根目录 `.gitignore` 已默认忽略该文件。

```yaml
api_keys:
  vectorengine: "your-vectorengine-key"
  gemini_api_key: "your-google-ai-api-key"
  deepseek_direct: "your-deepseek-api-key"
  longcat: "your-longcat-api-key"
```

### 方式 2：使用环境变量

可用环境变量：

- `VECTORENGINE_API_KEY`
- `GEMINI_API_KEY`
- `GEMINI_KEY`
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_API_BASE`
- `LONGCAT_API_KEY`
- `PORTABLE_LLM_KEYS_PATH`
- `GOOGLE_CLOUD_PROJECT`
- `VERTEX_SERVICE_ACCOUNT_EMAIL`

`PORTABLE_LLM_KEYS_PATH` 用来指定外部 `keys.yaml` 的绝对路径或相对路径。

## Provider 路由规则

客户端按 `model_name` 自动选择 provider：

- 包含 `longcat`：走 LongCat。
- 包含 `gemini`：走 Gemini。
- 包含 `deepseek`：走 DeepSeek。
- 其他模型名：默认走 VectorEngine。

也就是说，像 `gpt-4o-mini` 这种名字，在这个包里默认会走 VectorEngine，而不是 OpenAI 官方直连。

## Gemini 两条调用路径

### Google AI API

当你提供了 `gemini_api_key` 时，Gemini 走 Google AI API。

优点：

- 配置简单。
- 不依赖 gcloud。

### Vertex AI

当你没有提供 `gemini_api_key` 时，Gemini 自动回退到 Vertex AI。

你需要：

- 本机已安装并登录 `gcloud`
- 配置 `project_id`
- 如果需要 impersonation，再配 `service_account_email`

最小配置示例见 `config.example.yaml`。

## 最小调用示例

```python
from portable_llm import UnifiedLLMClient

config = {
    "vertex": {
        "project_id": "your-gcp-project-id",
        "service_account_email": "your-service-account@your-project.iam.gserviceaccount.com",
    }
}

client = UnifiedLLMClient(
    model_name="gemini-2.0-flash",
    config=config,
    keys_path="portable_llm/keys.yaml",
)

response = client.generate(
    system_prompt="You are a helpful assistant.",
    user_prompt="Reply with only OK.",
    temperature=0.0,
)

print(response.text)
print(response.usage.total_tokens)
```

## 这份可搬运版本做过的适配

相比原项目内联实现，这个文件夹额外解决了两个迁移常见问题：

- `keys.yaml` 不再绑定原仓库根目录，支持当前文件夹、工作目录、显式路径和环境变量。
- DeepSeek provider 不再把模型名写死为 `deepseek-chat`，会使用你传入的 `model_name`。

## 建议的落地方式

如果你想在目标项目里长期维护，推荐直接把这个目录当成一个内嵌小包来用，不要再拆散复制单文件；这样相对导入、provider 依赖和配置模板都能保持完整。