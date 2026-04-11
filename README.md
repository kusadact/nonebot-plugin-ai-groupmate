<div align="center">
  <a href="https://v2.nonebot.dev/store">
    <img src="https://raw.githubusercontent.com/fllesser/nonebot-plugin-template/refs/heads/resource/.docs/NoneBotPlugin.svg" width="310" alt="logo">
  </a>

## ✨ nonebot-plugin-ai-groupmate ✨

</div>

## 📖 介绍
这是基于原项目 [`yaowan233/nonebot-plugin-ai-groupmate`](https://github.com/yaowan233/nonebot-plugin-ai-groupmate) 持续演化的自用 fork。

README 以当前仓库 `main` 分支为准，不以 PyPI 版或原项目发布版为准。
当前 fork 已补齐 `uv` 项目管理文件，克隆后可直接在仓库根目录执行 `uv sync`。

当前这条分支的核心目标是：

- 使用 LangChain Agent + Tool Calling 驱动群聊回复
- 使用 Qdrant 承担聊天 RAG 与表情包检索
- 使用多模态模型做图片理解，不再依赖独立 `vlm.py`
- 保留并强化表情包拉黑机制
- 保留并强化好感度 V2 / 群体认知档案

## ✨ 当前分支能力

- **群聊 Agent**
  - 主对话模型使用 OpenAI 兼容接口，要求模型支持 Tool Calling
  - 支持联网搜索、历史聊天检索、表情包搜索/发送、年度报告、关系更新
- **群体认知档案**
  - 每 6 小时自动汇总群内常见话题、成员特征、内部梗和群氛围
- **聊天 RAG**
  - 聊天历史写入 Qdrant，检索结果会再走 rerank
  - Qdrant 不可用时自动降级，不会直接把 bot 拉死
- **表情包学习与检索**
  - 群图片先用多模态模型判定是否为表情包，再进入媒体向量库
  - 支持文字找图，也支持按历史消息 ID 找相似图
  - 图片处理已改为后台异步，不阻塞主回复链路
- **表情包拉黑**
  - superuser 回复 bot 发出的表情包消息，并带有“不能 / 不行 / 不可以 / 别发”等否定反馈时，会自动拉黑
  - 被拉黑表情包会同时从 SQL 标记并在 Qdrant 检索侧过滤
- **好感度 V2**
  - 使用 `favorability_raw` + `favorability` 双分值
  - 带状态机、每日上限、bank、bypass、道歉衰减、惩罚冷却
- **运维友好**
  - `/ai on|off|status` 可直接控制插件状态
  - 过期媒体和磁盘孤儿文件会定期清理
  - 多模态永久失败图片会自动跳过，避免无限重试

## 🚩 仓库定位

这个仓库现在已经补齐 `pyproject.toml`，可以直接用 `uv` 管理 Python、虚拟环境和依赖。

它仍然不是当前 fork 的 PyPI 发布仓库，更适合下面两种场景：

1. 直接克隆当前仓库，在仓库内用 `uv` 做本地开发和部署
2. 在你自己的 NoneBot bot 工程里，把这个仓库作为本地路径依赖或 git 依赖接入

如果你要对齐原项目的标准发布、测试、CI、PyPI 流程，可以继续参考原项目。

## 💿 接入方式

### 1. 直接在当前仓库内用 `uv` 管理（推荐）

当前仓库已经带有 `pyproject.toml` 和 `.python-version`，推荐直接在仓库根目录执行：

```bash
uv python install 3.12
uv sync
```

说明：

- 当前仓库声明的 Python 范围是 `>=3.10,<3.14`
- 仓库内 `.python-version` 固定为 `3.12`
- 如果你机器当前默认是 `3.14.x`，`uv` 会按项目要求单独准备兼容版本

插件开发或本地运维时，常用命令可以直接这样跑：

```bash
uv run nb orm upgrade
uv run python -m nonebot
```

### 2. 在 bot 工程里作为依赖接入

如果你的 bot 工程本身也在使用 `uv`，可以直接加本地路径依赖：

```bash
uv add /path/to/nonebot-plugin-ai-groupmate-main
```

或者加 git 依赖：

```bash
uv add git+https://github.com/kusadact/nonebot-plugin-ai-groupmate@dev
```

然后在 bot 项目的 `pyproject.toml` 里确保 `[tool.nonebot]` 已加载这个插件，例如：

```toml
[tool.nonebot]
plugins = ["nonebot_plugin_ai_groupmate"]
```

### 3. 继续按目录方式接入也可以

如果你仍然想保持“bot 工程内插件目录”的用法，也可以把 `nonebot_plugin_ai_groupmate/` 放进你的 bot 项目插件目录，例如：

```text
your-bot/
├─ src/plugins/
│  └─ nonebot_plugin_ai_groupmate/
```

### 4. 依赖说明

主运行时依赖已经写进当前仓库的 `pyproject.toml`。如果你是手动拷目录用，需要自己保证这些依赖已可用：

- `nonebot_plugin_alconna`
- `nonebot_plugin_orm`
- `nonebot_plugin_uninfo`
- `nonebot_plugin_localstore`
- `nonebot_plugin_apscheduler`
- `langchain`
- `langchain-openai`
- `langchain-tavily`
- `qdrant-client`
- `jieba`
- `wordcloud`
- `simpleeval`

### 5. 执行 ORM 迁移

在项目环境里执行：

```bash
uv run nb orm upgrade
```

如果你不是通过 `uv run` 执行，也可以继续用原来的方式：

```bash
nb -py /path/to/python orm upgrade
```

## ⚙️ 配置

### 基础说明

- `openai_*`：主对话模型 / 前置回复判断模型使用
- `summary_*`：群体认知档案总结模型使用
- `multimodal_*`：图片理解、表情包判定使用
- `qdrant_*`：聊天 / 媒体向量库
- `remote_embedding_*`：聊天文本 embedding
- `remote_rerank_*`：聊天文本 rerank
- `remote_media_embedding_*`：图片向量
- `remote_media_rerank_*`：图片 rerank

当前分支已经移除旧版统一 `/embed`、`/rerank`、`/clip` 接口兼容层，只保留分路配置。

### 配置表

| 配置项 | 必填 | 默认值 | 说明 |
|:--|:--:|:--|:--|
| `ai_groupmate__bot_name` | 是 | `bot` | bot 名称 |
| `ai_groupmate__reply_probability` | 否 | `0.01` | 群内主动发言概率 |
| `ai_groupmate__personality_setting` | 否 | 空 | 自定义人设补充 |
| `ai_groupmate__qdrant_uri` | 否 | 空 | Qdrant 地址；不填则禁用 RAG / 表情包向量功能 |
| `ai_groupmate__qdrant_api_key` | 否 | 空 | Qdrant API Key |
| `ai_groupmate__chat_vector_dim` | 否 | `1024` | 聊天文本向量维度 |
| `ai_groupmate__media_vector_dim` | 否 | `2560` | 图片向量维度 |
| `ai_groupmate__remote_embedding_base_url` | 否 | 空 | 文本 embedding 服务地址，OpenAI 风格 |
| `ai_groupmate__remote_embedding_api_key` | 否 | 空 | 文本 embedding API Key |
| `ai_groupmate__remote_embedding_model` | 否 | 空 | 文本 embedding 模型名 |
| `ai_groupmate__remote_embedding_dimensions` | 否 | `1024` | 文本 embedding 维度 |
| `ai_groupmate__remote_rerank_base_url` | 否 | 空 | 文本 rerank 服务地址 |
| `ai_groupmate__remote_rerank_api_key` | 否 | 空 | 文本 rerank API Key |
| `ai_groupmate__remote_rerank_model` | 否 | 空 | 文本 rerank 模型名 |
| `ai_groupmate__remote_media_embedding_provider` | 否 | `aliyun_dashscope` | 图片 embedding 提供方：`openai` / `aliyun_dashscope` |
| `ai_groupmate__remote_media_embedding_base_url` | 否 | 空 | 图片 embedding 地址 |
| `ai_groupmate__remote_media_embedding_api_key` | 否 | 空 | 图片 embedding API Key |
| `ai_groupmate__remote_media_embedding_model` | 否 | 空 | 图片 embedding 模型名 |
| `ai_groupmate__remote_media_embedding_dimensions` | 否 | `2560` | 图片 embedding 维度 |
| `ai_groupmate__remote_media_rerank_provider` | 否 | `openai` | 图片 rerank 提供方：`openai` / `aliyun_dashscope` |
| `ai_groupmate__remote_media_rerank_base_url` | 否 | 空 | 图片 rerank 地址 |
| `ai_groupmate__remote_media_rerank_api_key` | 否 | 空 | 图片 rerank API Key |
| `ai_groupmate__remote_media_rerank_model` | 否 | 空 | 图片 rerank 模型名 |
| `ai_groupmate__tavily_api_key` | 否 | 空 | Tavily 搜索 API Key |
| `ai_groupmate__qwen_token` | 否 | 空 | DashScope 通用 API Key，`summary` / `multimodal` 可回退使用 |
| `ai_groupmate__summary_model` | 否 | `qwen-flash` | 群体认知档案总结模型 |
| `ai_groupmate__summary_base_url` | 否 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 总结模型 base URL |
| `ai_groupmate__summary_api_key` | 否 | 空 | 总结模型 API Key；为空时回退用 `qwen_token` 或 `openai_token` |
| `ai_groupmate__multimodal_model` | 否 | `qwen-vl-max` | 图片理解模型 |
| `ai_groupmate__multimodal_base_url` | 否 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 多模态模型 base URL |
| `ai_groupmate__multimodal_api_key` | 否 | 空 | 多模态模型 API Key；为空时回退用 `qwen_token` |
| `ai_groupmate__openai_base_url` | 是 | 空 | 主对话模型 base URL，OpenAI 兼容 |
| `ai_groupmate__openai_model` | 是 | 空 | 主对话模型名，需支持 Tool Calling |
| `ai_groupmate__openai_token` | 是 | 空 | 主对话模型 API Key |

## 🔧 推荐配置

### 最小可运行配置

如果你只想先跑通“聊天 + 群体记忆 + 图片理解”，推荐：

```env
ai_groupmate__bot_name=bot

ai_groupmate__openai_base_url=https://dashscope.aliyuncs.com/compatible-mode/v1
ai_groupmate__openai_model=qwen3.5-plus
ai_groupmate__openai_token=sk-xxxx

ai_groupmate__qwen_token=sk-xxxx
ai_groupmate__summary_model=qwen3.5-plus
ai_groupmate__multimodal_model=qwen-vl-max
```

说明：

- 主对话模型走 `openai_*`
- `summary_model` 和 `multimodal_model` 默认可直接回退使用 `qwen_token`
- 此配置下若不填 `qdrant_uri`，RAG 和表情包向量功能会被禁用，但普通群聊仍可运行
- 如果后续启用 Qdrant，需补齐独立的文本 / 图片向量服务配置，不再支持旧版统一接口兜底

### 当前 fork 推荐配置

如果你要启用完整的 `Qdrant + RAG + 表情包检索`，至少需要把聊天 embedding 和图片 embedding 配起来：

```env
ai_groupmate__bot_name=bot
ai_groupmate__reply_probability=0.01

ai_groupmate__qdrant_uri=http://127.0.0.1:6333
ai_groupmate__qdrant_api_key=
ai_groupmate__chat_vector_dim=1024
ai_groupmate__media_vector_dim=2560

ai_groupmate__openai_base_url=https://dashscope.aliyuncs.com/compatible-mode/v1
ai_groupmate__openai_model=qwen3.5-plus
ai_groupmate__openai_token=sk-xxxx

ai_groupmate__qwen_token=sk-xxxx
ai_groupmate__summary_model=qwen3.5-plus
ai_groupmate__multimodal_model=qwen-vl-max

ai_groupmate__remote_embedding_base_url=https://api.siliconflow.cn/v1
ai_groupmate__remote_embedding_api_key=sk-xxxx
ai_groupmate__remote_embedding_model=Qwen/Qwen3-Embedding-4B
ai_groupmate__remote_embedding_dimensions=1024

ai_groupmate__remote_rerank_base_url=https://api.siliconflow.cn/v1
ai_groupmate__remote_rerank_api_key=sk-xxxx
ai_groupmate__remote_rerank_model=Qwen/Qwen3-Reranker-4B

ai_groupmate__remote_media_embedding_provider=aliyun_dashscope
ai_groupmate__remote_media_embedding_base_url=https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding
ai_groupmate__remote_media_embedding_api_key=sk-xxxx
ai_groupmate__remote_media_embedding_model=qwen3-vl-embedding
ai_groupmate__remote_media_embedding_dimensions=2560

ai_groupmate__remote_media_rerank_base_url=
ai_groupmate__remote_media_rerank_api_key=
ai_groupmate__remote_media_rerank_model=
```

## 🧭 使用

### 触发方式

- `@bot`、回复 bot、直接点名 bot，都会触发回复判断
- 也会按 `reply_probability` 概率主动发言
- 图片会后台异步入库，不再阻塞主回复

### 管理指令

| 指令 | 说明 |
|:--|:--|
| `/ai on` | 打开插件 |
| `/ai off` | 关闭插件 |
| `/ai status` | 查看插件状态与 Qdrant 连通性 |
| `/词频 <统计天数>` | 生成个人词频词云 |
| `/群词频 <统计天数>` | 生成群词频词云 |

说明：

- `/ai` 仅 superuser 可用
- `off` 主要关闭 AI 回复和向量查询/写入
- `status` 会实际检查 Qdrant 当前状态

### 聊天内能力

以下能力不需要额外命令，直接在聊天中触发：

- 询问历史群聊上下文
- 让 bot 查梗、查资料、联网搜索
- 让 bot 发合适的表情包
- 引用某张历史图片让 bot 找相似表情包
- 让 bot 生成年度报告 / 个人总结 / 成分分析

### 表情包拉黑

当前分支支持“回复式拉黑”：

1. bot 先发出一张表情包
2. superuser 回复这条消息
3. 回复内容包含 `不可以 / 不行 / 不能 / 别发 / 别再发 / 不要这张` 等否定反馈
4. 该表情包会被标记为黑名单，并从后续检索中排除

## 🔄 迁移提示

如果你是从旧的 Milvus / 独立 VLM 版本迁移过来，需要注意：

- 当前主线已经切到 **Qdrant**
- `vlm.py` 已移除，图片理解走 `multimodal_model`
- 旧版 `remote_model_*` / `remote_clip_*` 兼容配置已经移除，需要改成 `remote_embedding_*`、`remote_rerank_*`、`remote_media_embedding_*`、`remote_media_rerank_*`
- 聊天向量与图片向量默认维度为 **1024 / 2560**
- 如果旧向量库维度不一致，必须重建 `chat_collection` 和 `media_collection`
- 迁移后请执行 ORM 迁移，确保以下结构已经到位：
  - `GroupMemory`
  - `MediaStorage.blocked`
  - `UserRelation V2`
  - `ChatHistory(session_id, created_at)` 复合索引

## 📌 与原项目差异

下面这部分只描述当前 fork 相对原项目 `main` 的主要差异。

### 运行时架构差异

- 原项目更偏“标准插件发布版”；当前仓库现在也支持 `uv` 工程化管理，但整体仍偏“自用部署 / 本地维护版”
- 原项目 2.x 已切到 API 轻量化路线；当前 fork 在此基础上进一步做了分路配置和部署适配
- 当前 fork 不再使用独立 `vlm.py`，统一改为多模态模型图片理解

### 向量与检索差异

- 当前 fork 已完全切到 **Qdrant**
- 文本 embedding / 文本 rerank / 图片 embedding / 图片 rerank 均可分别配置
- 支持：
  - OpenAI 风格 `/v1/embeddings`
  - OpenAI 风格 `/v1/rerank`
  - 阿里百炼原生 `qwen3-vl-embedding`
- 图片检索支持：
  - 文字找表情包
  - 图片找相似图片
  - 按历史消息 ID 找相似图片
- 检索结果会过滤 `blocked` 表情包

### 功能差异

- 当前 fork 增加了 **表情包自动拉黑**
- 当前 fork 保留并强化了 **好感度 V2**
- 当前 fork 保留并启用了 **群体认知档案**
- 当前 fork 增加 `/ai on|off|status`
- 当前 fork 补了多处运行稳定性修复：
  - 图片并发入库冲突兜底
  - 多模态永久失败图片自动跳过
  - 图片后台异步处理
  - 媒体缓存与磁盘孤儿文件清理

### 文档和使用方式差异

- 原项目 README 更偏标准安装与发布说明
- 当前 README 以当前仓库代码和当前分支行为为准，也补充了当前 fork 的 `uv` 本地管理流程
- 如果你要完全对齐原项目的发布形态，请回原项目 README 和 `pyproject.toml`

## 🙏 致谢

- 原项目：[`yaowan233/nonebot-plugin-ai-groupmate`](https://github.com/yaowan233/nonebot-plugin-ai-groupmate)
- NoneBot2 社区及相关插件作者
