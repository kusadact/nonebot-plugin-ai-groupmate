<div align="center">
  <a href="https://v2.nonebot.dev/store">
    <img src="https://raw.githubusercontent.com/fllesser/nonebot-plugin-template/refs/heads/resource/.docs/NoneBotPlugin.svg" width="310" alt="logo">
  </a>

## ✨ nonebot-plugin-ai-groupmate ✨

</div>

## 📖 介绍
这是基于原项目 [`yaowan233/nonebot-plugin-ai-groupmate`](https://github.com/yaowan233/nonebot-plugin-ai-groupmate) 持续演化的自用 fork。

README 以当前仓库代码为准，不以 PyPI 版或原项目发布版为准。
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
  - 支持联网搜索、历史聊天检索、表情包搜索/发送、语音发送、年度报告、关系更新
  - 语音工具支持 GPT-SoVITS 接口；健康检查通过时才会注入 Agent，默认转为 NapCat 兼容的 AMR
  - 每个群只保留最新回复请求，旧请求会被取消
  - 工具调用带 `request_id` 过期保护，旧请求不会继续搜索、发消息或更新关系
  - 当前触发消息会作为本轮重点注入 prompt，降低顺着其他人支线接话的概率
  - 多段回复使用一次 `reply_user` 调用，按换行由程序串行拆成多条消息
  - `reply_user` 会把实际发送段落返回给 Agent，方便本轮后续工具知道刚刚说过什么
  - bot 有群管理权限时会注入 `mute_user` 工具，禁言 / 解除禁言成功后会写入 `ChatHistory`
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
  - Agent 工具内部数据库写入使用独立 session，降低异步工具并发导致的事务状态冲突
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
uv add git+https://github.com/kusadact/nonebot-plugin-ai-groupmate@main
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
- `voice_*`：GPT-SoVITS 语音合成、健康检查和 ffmpeg 转码使用
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
| `ai_groupmate__media_search_recall_limit` | 否 | `6` | 表情包检索从 Qdrant 召回的候选数；DashScope `qwen3-vl-rerank` 图片批量限制为 1-6，默认保持 6 |
| `ai_groupmate__media_search_return_limit` | 否 | `5` | 表情包检索最终返回给 Agent 的候选数 |
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
| `ai_groupmate__voice_enabled` | 否 | `false` | 是否启用语音工具；启用后仍需健康检查通过才会注入 Agent |
| `ai_groupmate__voice_base_url` | 否 | 空 | GPT-SoVITS 服务地址，例如 `http://127.0.0.1:9880` 或 frp 地址 |
| `ai_groupmate__voice_tts_path` | 否 | `/tts` | GPT-SoVITS 合成接口路径；也可填完整 URL |
| `ai_groupmate__voice_health_path` | 否 | `/` | 健康检查路径；返回非 5xx 即视为服务可达，也可配置为 `/docs` |
| `ai_groupmate__voice_health_timeout_seconds` | 否 | `2.0` | 健康检查超时 |
| `ai_groupmate__voice_tts_timeout_seconds` | 否 | `180.0` | TTS 请求超时 |
| `ai_groupmate__voice_health_cache_seconds` | 否 | `30.0` | 健康状态缓存时间 |
| `ai_groupmate__voice_unhealthy_cache_seconds` | 否 | `10.0` | 不健康状态缓存时间 |
| `ai_groupmate__voice_trust_env_proxy` | 否 | `false` | 语音请求是否使用容器环境代理；frp/直连地址建议保持 `false` |
| `ai_groupmate__voice_ref_audio_path` | 否 | `ref_audio/Azuma/Azuma_10.wav` | GPT-SoVITS 参考音频路径，按服务端可访问路径填写 |
| `ai_groupmate__voice_prompt_text` | 否 | `完了我找不到他之前的投稿了，反正就是有一个。` | 参考音频对应文本 |
| `ai_groupmate__voice_text_lang` | 否 | `zh` | 目标文本语言 |
| `ai_groupmate__voice_prompt_lang` | 否 | `zh` | 参考音频语言 |
| `ai_groupmate__voice_text_split_method` | 否 | `cut5` | GPT-SoVITS 分句方式 |
| `ai_groupmate__voice_request_media_type` | 否 | `wav` | GPT-SoVITS 返回格式 |
| `ai_groupmate__voice_output_format` | 否 | `wav` | bot 最终发送格式；NapCat 会自行转码发送语音，通常保持 `wav` 即可 |
| `ai_groupmate__voice_send_method` | 否 | `raw` | 语音发送方式；`raw` 使用 base64，`path` 使用本地文件路径 |
| `ai_groupmate__voice_send_dir` | 否 | 空 | `voice_send_method=path` 时的音频临时目录，需确保 OneBot/NapCat 进程可访问 |
| `ai_groupmate__voice_send_file_keep_seconds` | 否 | `300.0` | `path` 发送后临时音频文件保留秒数 |
| `ai_groupmate__voice_streaming_mode` | 否 | `0` | GPT-SoVITS 流式模式；机器人发送语音建议 `0` |
| `ai_groupmate__voice_batch_size` | 否 | `1` | GPT-SoVITS batch size |
| `ai_groupmate__voice_speed_factor` | 否 | `1.0` | 语速 |
| `ai_groupmate__voice_top_k` | 否 | `15` | GPT-SoVITS top_k |
| `ai_groupmate__voice_top_p` | 否 | `1.0` | GPT-SoVITS top_p |
| `ai_groupmate__voice_temperature` | 否 | `1.0` | GPT-SoVITS temperature |
| `ai_groupmate__voice_max_text_length` | 否 | `120` | 单条语音最大文本长度 |
| `ai_groupmate__voice_ffmpeg_path` | 否 | `ffmpeg` | ffmpeg 可执行文件路径 |
| `ai_groupmate__voice_ffmpeg_timeout_seconds` | 否 | `30.0` | ffmpeg 转码超时 |
| `ai_groupmate__voice_ffmpeg_audio_filter` | 否 | 空 | 自定义 ffmpeg 音频滤镜；填写后优先于 `voice_volume_gain`，例如 `loudnorm=I=-10:TP=-1.5:LRA=7,aresample=32000` |
| `ai_groupmate__voice_ffmpeg_audio_codec` | 否 | `libopencore_amrnb` | 转 AMR 时使用的音频编码器；如本机 ffmpeg 不支持可改配置 |
| `ai_groupmate__voice_volume_gain` | 否 | `1.5` | 转码时的音量倍率；`1.0` 表示不放大 |
| `ai_groupmate__voice_amr_sample_rate` | 否 | `8000` | AMR 采样率 |
| `ai_groupmate__voice_amr_bitrate` | 否 | `12.2k` | AMR 码率 |
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

### 可选语音工具配置

语音工具默认关闭。开启后，插件会在创建 Agent 前检查 GPT-SoVITS 服务健康状态；健康检查失败时，不会向 Agent 注入 `send_voice` 工具和语音相关 prompt。

```env
ai_groupmate__voice_enabled=true
ai_groupmate__voice_base_url=http://127.0.0.1:9880
ai_groupmate__voice_tts_path=/tts
ai_groupmate__voice_health_path=/
ai_groupmate__voice_trust_env_proxy=false

ai_groupmate__voice_ref_audio_path=ref_audio/Azuma/Azuma_10.wav
ai_groupmate__voice_prompt_text=完了我找不到他之前的投稿了，反正就是有一个。

# NapCat 发送语音时会自行转码；通常保留 WAV，避免先转 AMR 后再被二次转码。
ai_groupmate__voice_request_media_type=wav
ai_groupmate__voice_output_format=wav
ai_groupmate__voice_ffmpeg_path=ffmpeg
ai_groupmate__voice_volume_gain=1.5
# 如果 NoneBot 和 NapCat 共享同一个目录，可以用 path 发送；OneBot 会收到普通绝对路径。
# ai_groupmate__voice_send_method=path
# ai_groupmate__voice_send_dir=/app/.config/QQ/NapCat/temp
# 如需更稳的响度处理，可用 loudnorm 替代简单音量放大。
# ai_groupmate__voice_ffmpeg_audio_filter=loudnorm=I=-10:TP=-1.5:LRA=7,aformat=sample_fmts=s16:sample_rates=24000:channel_layouts=mono
```

### 当前 fork 推荐配置

如果你要启用完整的 `Qdrant + RAG + 表情包检索`，至少需要把聊天 embedding 和图片 embedding 配起来：

```env
ai_groupmate__bot_name=bot
ai_groupmate__reply_probability=0.01

ai_groupmate__qdrant_uri=http://127.0.0.1:6333
ai_groupmate__qdrant_api_key=
ai_groupmate__chat_vector_dim=1024
ai_groupmate__media_vector_dim=2560
ai_groupmate__media_search_recall_limit=6
ai_groupmate__media_search_return_limit=5

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

说明：

- `media_search_recall_limit` 控制送入图片 rerank 的候选池大小。使用阿里 DashScope `qwen3-vl-rerank` 时不要超过 `6`，否则接口会返回 `image batch size can should be [1, 6]`。
- `media_search_return_limit` 控制工具最终给 Agent 看的候选数量，默认 `5` 保持原有行为。

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
当前对照的上游基线是 `yaowan233/nonebot-plugin-ai-groupmate` 的 `f4f87e3`（2026-04-29，提交信息 `优化表现`）。

### 运行时架构差异

- 原项目更偏“标准插件发布版”；当前仓库现在也支持 `uv` 工程化管理，但整体仍偏“自用部署 / 本地维护版”
- 原项目 2.x 已切到 API 轻量化路线；当前 fork 在此基础上进一步做了分路配置和部署适配
- 当前 fork 不再使用独立 `vlm.py`，统一改为多模态模型图片理解

### 向量与检索差异

- 当前 fork 已完全切到 **Qdrant**
- 文本 embedding / 文本 rerank / 图片 embedding / 图片 rerank 均可分别配置
- 表情包检索召回数通过 `media_search_recall_limit` 配置，默认 `6`，避免阿里 DashScope `qwen3-vl-rerank` 图片批量超过 1-6 限制
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
- 当前 fork 增加 **群管理禁言工具**；成功禁言 / 解禁后会以系统记录形式写入 `ChatHistory`，让后续 Agent 能看到最近管理操作
- 当前 fork 补了多处运行稳定性修复：
  - 每群最新请求保护和 request 过期保护
  - Agent 当前触发消息锚定，降低回复串线
  - 当前消息图片 / 引用图片会强绑定到本轮 prompt，历史图片只作背景
  - 多段回复由一次 `reply_user` 调用串行拆发，避免模型并发多次调用 `reply_user`
  - `reply_user` 工具结果会回显实际发送段落
  - Agent 工具 DB 写入使用独立 session
  - 图片并发入库冲突兜底
  - 多模态永久失败图片自动跳过
  - 图片后台异步处理
  - 媒体缓存与磁盘孤儿文件清理

### 与上游实现方式差异

- 上游 `f4f87e3` 用 `SequentialReplyMiddleware` 允许 Agent 多轮调用 `reply_user`，每轮限制一次。当前 fork 选择一次 `reply_user` 调用内传入多行内容，由程序串行拆发；这样更省 token，也避免多轮发言在高峰期互相抢上下文。
- 上游 `reply_user` 返回刚发送的内容。当前 fork 保留这个思路，但返回的是程序实际发送的段落和被重复过滤跳过的段落，因为最终发出的内容可能经过拆行、去重和过期检查。
- 上游禁言成功后直接用传入工具的 `db_session` 写 `ChatHistory`。当前 fork 使用独立 `get_session()` 写禁言系统记录，避免和 Agent 主流程 / 其他工具共享同一个 `AsyncSession`。
- 上游图片并发入库失败后使用 `rollback()` 重新查询。当前 fork 使用 `begin_nested()` savepoint 兜底并发唯一约束冲突，避免回滚掉同一图片处理流程中的其他状态。
- 上游缓存清理主要做批量文件 / SQL 删除。当前 fork 还会同步删除 Qdrant 中的媒体向量，避免被清理的表情包继续参与检索。
- 上游媒体检索召回数仍是代码常量。当前 fork 将召回数 / 返回数做成配置项，默认 `6 / 5`，兼容 DashScope 图片 rerank 批量限制。

### 文档和使用方式差异

- 原项目 README 更偏标准安装与发布说明
- 当前 README 以当前仓库代码和当前分支行为为准，也补充了当前 fork 的 `uv` 本地管理流程
- 如果你要完全对齐原项目的发布形态，请回原项目 README 和 `pyproject.toml`

## 🙏 致谢

- 原项目：[`yaowan233/nonebot-plugin-ai-groupmate`](https://github.com/yaowan233/nonebot-plugin-ai-groupmate)
- NoneBot2 社区及相关插件作者
