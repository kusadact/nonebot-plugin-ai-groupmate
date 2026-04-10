<div align="center">
    <a href="https://v2.nonebot.dev/store">
    <img src="https://raw.githubusercontent.com/fllesser/nonebot-plugin-template/refs/heads/resource/.docs/NoneBotPlugin.svg" width="310" alt="logo"></a>

## ✨ nonebot-plugin-ai-groupmate ✨

</div>

## 📖 介绍
本插件主体使用使用 langchain 的 agent 进行决策，由 langchain 调用 tools 进行一系列任务。

tools 中包含 RAG ，可以自动对聊天历史储存，储存长记忆。学习群内群友发言习惯，使得 bot 更像真人。

对于群内的表情包，使用了多模态图片理解 + 多模态 embedding / rerank 模型，自动从群内学习并偷取表情包，然后从向量库内选取合适表情包进行回答。

对于模型选择方面：推荐使用 `qwen3.5-plus` 作为聊天 / 群体记忆模型，`qwen-vl-max` 负责图片理解，文本 embedding 与图片 embedding 独立配置。如果 embedding、rerank 和多模态 embedding / rerank 模型都使用显卡加速，大约需要 16G 显存；若显存不足，建议直接改为 API 调用。

## 改动说明
- 将 embedding / rerank / media embedding / media rerank 等模型计算远程化：插件只负责业务逻辑。图片 embedding 现已支持两种接法：
  - OpenAI 风格 `/v1/embeddings`
  - 阿里百炼原生多模态 embedding 接口 `qwen3-vl-embedding`
- 图片向量化与图片 rerank 使用 OpenAI 风格接口，插件侧会请求 `base_url + /embeddings` 与 `base_url + /rerank`，图片以 base64 data URL 传输。
- 文本与图片向量库默认维度调整为 `1024 / 2560`；升级后需要重建现有 Qdrant `chat_collection` / `media_collection`。
- 新增 superuser 手动开关指令：`/ai on|off|status`
  - `off` 仅停止 AI 回复与向量库查询/写入；图片识别与聊天记录入库不受影响。
  - `status` 会实际请求 Qdrant 并返回当前连通状态。
- 插件启动不再强依赖 Qdrant 在线，向量库初始化延迟到首次使用。
- RAG 在 Qdrant 不可用/超时时会自动跳过，降级为普通对话（不会导致 bot 退出）。
- 新增表情包黑名单（数据库列 `MediaStorage.blocked`）：
  - 仅当 superuser 回复 bot 发出的表情包消息，并包含`不可以/不行/不能`等否定反馈时，自动将该表情包标记为黑名单。
  - 被拉黑的表情包会在检索阶段与发送前双重过滤，不再被 bot 发送。
  - 重复拉黑同一表情包时会提示“已在黑名单中”。
- 已剥离图片/视频生成功能（Seedance 相关逻辑与配置），插件聚焦 AI 群聊、RAG 与表情包能力。
- 好感度系统 V2（Monika 风格）：
  - 新关系表：`nonebot_plugin_ai_groupmate_userrelation_v2`（旧表保留但不再作为主逻辑写入）。
  - 双分值模型：`favorability_raw` 为主驱动（范围 `[-1000, 1000]`），`favorability` 为映射显示分（范围 `[-100, 100]`，`RAW_PER_SCORE=10`）。
  - Agent 打分口径改为 raw：`update_user_impression.score_change` 按 raw 传入，常规建议 `±20`，单次硬上限 `±50`。
  - 状态机：`broken / distressed / upset / normal / happy / affectionate / enamored / love`，按 raw 分段决定关系描述与系数。
  - 正负向非线性：根据当前状态对加分/扣分使用不同倍率，并在接近上下限时自动“饱和”衰减。
  - 日限制机制（raw 单位）：每日普通正向最多 `+70`（映射 `+7`），每日普通负向最多 `-70`（映射 `-7`）。
  - bypass/bank（raw 单位）：`bypass` 每日上限 `100`（映射 `+10`），`bank` 存储上限 `200`（映射 `+20`）。
  - 道歉衰减：同类型道歉加分按“第一次全额、第二次半额、第三次及以后衰减至 0”处理。
  - 惩罚冷却（`last_penalty_at` 已接入）：连续负向变更按 5 分钟一档做阶梯衰减（0/1/2/3/4/5/6 档），冷却窗口 30 分钟（1800 秒）；超过 30 分钟后不再衰减。

## ⚙️ 配置

配置说明
| 配置项 | 必填 | 默认值 | 说明 |
|:-----:|:----:|:----:|:----:|
| ai_groupmate__bot_name | 是 | 无 | bot 名 |
| ai_groupmate__reply_probability | 否 | 0.01 | 群内发言概率 |
| ai_groupmate__personality_setting | 否 | 无 | 自定义人设 |
| ai_groupmate__qdrant_uri | 否 | 无 | Qdrant 地址 |
| ai_groupmate__qdrant_api_key | 否 | 无 | Qdrant API Key |
| ai_groupmate__chat_vector_dim | 否 | 1024 | 聊天文本向量维度 |
| ai_groupmate__media_vector_dim | 否 | 2560 | 图片向量维度 |
| ai_groupmate__remote_model_base_url | 否 | 无 | 旧版统一模型服务地址（`/embed` / `/rerank`） |
| ai_groupmate__remote_model_api_key | 否 | 无 | 远程模型服务 API Key |
| ai_groupmate__remote_embedding_base_url | 否 | 无 | embedding 分路地址（硅基流动/OpenAI 风格） |
| ai_groupmate__remote_embedding_api_key | 否 | 无 | embedding 分路 API Key（为空则回退用 remote_model_api_key） |
| ai_groupmate__remote_embedding_model | 否 | 无 | embedding 模型名（如 `BAAI/bge-m3`） |
| ai_groupmate__remote_embedding_dimensions | 否 | 1024 | 文本 embedding 维度 |
| ai_groupmate__remote_rerank_base_url | 否 | 无 | rerank 分路地址（硅基流动） |
| ai_groupmate__remote_rerank_api_key | 否 | 无 | rerank 分路 API Key（为空则回退用 remote_model_api_key） |
| ai_groupmate__remote_rerank_model | 否 | 无 | rerank 模型名（如 `BAAI/bge-reranker-v2-m3`） |
| ai_groupmate__remote_media_embedding_provider | 否 | openai | 图片 embedding 提供方：`openai` 或 `aliyun_dashscope` |
| ai_groupmate__remote_media_embedding_base_url | 否 | 无 | 图片 embedding 地址；`openai` 时填 `/v1` 前缀，`aliyun_dashscope` 时填百炼原生多模态 embedding 完整 URL |
| ai_groupmate__remote_media_embedding_api_key | 否 | 无 | 图片 embedding API Key（为空则回退用 remote_model_api_key） |
| ai_groupmate__remote_media_embedding_model | 否 | 无 | 图片 embedding 模型名 |
| ai_groupmate__remote_media_embedding_dimensions | 否 | 2560 | 图片 embedding 维度 |
| ai_groupmate__remote_media_rerank_base_url | 否 | 无 | 图片 rerank 分路地址（建议直接填带 `/v1` 的 base_url） |
| ai_groupmate__remote_media_rerank_api_key | 否 | 无 | 图片 rerank API Key（为空则回退用 remote_model_api_key） |
| ai_groupmate__remote_media_rerank_model | 否 | 无 | 图片 rerank 模型名 |
| ai_groupmate__tavily_api_key | 否 | 无 | tavily api 密钥 |
| ai_groupmate__openai_base_url | 否 | 无| openai 请求地址 |
| ai_groupmate__openai_token | 否 | 无 | openai token |
| ai_groupmate__openai_model | 否 | 无 | openai 模型名 |
| ai_groupmate__qwen_token | 否 | 无 | DashScope API Key 兼容字段 |
| ai_groupmate__summary_model | 否 | qwen-flash | 群体记忆总结模型 |
| ai_groupmate__summary_base_url | 否 | https://dashscope.aliyuncs.com/compatible-mode/v1 | 总结模型 base_url |
| ai_groupmate__summary_api_key | 否 | 无 | 总结模型 API Key |
| ai_groupmate__multimodal_model | 否 | qwen-vl-max | 图片理解模型 |
| ai_groupmate__multimodal_base_url | 否 | https://dashscope.aliyuncs.com/compatible-mode/v1 | 多模态模型 base_url |
| ai_groupmate__multimodal_api_key | 否 | 无 | 多模态模型 API Key |



### 远程模型服务示例配置（Qdrant + DashScope）
示例：聊天模型和群体记忆走 `qwen3.5-plus`，图片理解走 `qwen-vl-max`，向量库使用 Qdrant：

```env
ai_groupmate__qdrant_uri=http://127.0.0.1:6333
ai_groupmate__qdrant_api_key=
ai_groupmate__openai_base_url=https://dashscope.aliyuncs.com/compatible-mode/v1
ai_groupmate__openai_model=qwen3.5-plus
ai_groupmate__openai_token=sk-xxxx
ai_groupmate__summary_model=qwen3.5-plus
ai_groupmate__summary_base_url=https://dashscope.aliyuncs.com/compatible-mode/v1
ai_groupmate__summary_api_key=sk-xxxx
ai_groupmate__multimodal_model=qwen-vl-max
ai_groupmate__multimodal_base_url=https://dashscope.aliyuncs.com/compatible-mode/v1
ai_groupmate__multimodal_api_key=sk-xxxx
```

### embedding/rerank 走硅基流动 + 图片 embedding 直连阿里百炼示例
说明：`remote_embedding_base_url` 与 `remote_rerank_base_url` 请直接填写带 `/v1` 的完整前缀，例如 `https://api.siliconflow.cn/v1`。图片 embedding 如果使用阿里百炼，请把 `remote_media_embedding_provider` 设为 `aliyun_dashscope`，并直接填写百炼原生多模态 embedding 完整 URL。

```env
ai_groupmate__qdrant_uri=http://127.0.0.1:6333
ai_groupmate__qdrant_api_key=
ai_groupmate__chat_vector_dim=1024
ai_groupmate__media_vector_dim=2560

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

## 🎉 使用

ai会自动偷群内使用的表情包，增加至向量库当中，在回答时通过向量库内容搜索表情包。由于使用了多模态图片理解和图片向量，搜索准确率会更高。
![Screenshot_20251201_134203](https://github.com/user-attachments/assets/cbf95194-ac33-45e0-a83d-cb6639c204fb)
内置了好感度系统，增加了趣味性。
![Screenshot_20251201_134157](https://github.com/user-attachments/assets/68b8d563-7ad5-4d83-be4b-0a05f16df09a)
利用强大的 RAG，进行总结或进行任何检索聊天相关功能。
![Screenshot_20251201_133320](https://github.com/user-attachments/assets/b7e96bd0-8245-4da5-b28b-33e8aad5fc63)
发送群内偷学到的表情包
![Screenshot_20251201_132723](https://github.com/user-attachments/assets/6fbd036f-e7ec-4ced-9cd7-557976306553)

### 指令表
由于 AI 功能需要记录聊天记录，基于已记录的聊天记录，可以很轻松的做到词频统计，所以顺带加上了。

|     指令      |    说明    |
|:-----------:|:--------:|
|  /词频 <统计天数> | 生成个人词频词云 |
| /群词频 <统计天数> | 生成群词频词云  |
|  /ai on | 打开插件（启用 AI 回复 + 向量/RAG） |
|  /ai off | 关闭插件（停止 AI 回复 + 向量/RAG；图片识别/聊天记录仍会写入） |
|  /ai status | 查看插件开关与 Qdrant 状态 |
