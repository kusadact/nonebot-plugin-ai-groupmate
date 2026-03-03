<div align="center">
    <a href="https://v2.nonebot.dev/store">
    <img src="https://raw.githubusercontent.com/fllesser/nonebot-plugin-template/refs/heads/resource/.docs/NoneBotPlugin.svg" width="310" alt="logo"></a>

## ✨ nonebot-plugin-ai-groupmate ✨

</div>

## 📖 介绍
本插件主体使用使用 langchain 的 agent 进行决策，由 langchain 调用 tools 进行一系列任务。

tools 中包含 RAG ，可以自动对聊天历史储存，储存长记忆。学习群内群友发言习惯，使得 bot 更像真人。

对于群内的表情包，使用了 vlm + clip 模型，自动从群内学习并偷取表情包，然后从向量库内选取合适表情包进行回答。

对于模型选择方面：推荐使用 glm-4.6 模型，以及本地部署 qwen3-vl 作为 vlm 模型，如果 embedding、 rerank、vlm 和 clip 模型都使用了显卡加速的话，大约需要 16G 显存的显卡。若只有 8G 显存，将 vlm 模型调整为 api 调用的话，就可以流畅使用了。

## 改动说明
- 将 embedding / rerank / clip 等模型计算远程化：插件只负责业务逻辑，模型端使用 `server/app.py` 部署（默认监听 `8001`，可通过隧道/FRP 映射到 `18001`等端口）。
- 图片向量化支持 base64 传输（`/clip/image` 的 `images_base64`）。
- 新增 superuser 手动开关指令：`/ai on|off|status`
  - `off` 仅停止 AI 回复与向量库查询/写入；VLM 识别与聊天记录入库不受影响。
  - `status` 会实际请求 Milvus，并区分 `down (tunnel:...)` 与 `down (milvus:...)`。
- 插件启动不再强依赖 Milvus 在线，Milvus 初始化延迟到首次使用。
- RAG 在 Milvus 不可用/超时时会自动跳过，降级为普通对话（不会导致 bot 退出）。
- 新增表情包黑名单（数据库列 `MediaStorage.blocked`）：
  - 仅当 superuser 回复 bot 发出的表情包消息，并包含`不可以/不行/不能`等否定反馈时，自动将该表情包标记为黑名单。
  - 被拉黑的表情包会在检索阶段与发送前双重过滤，不再被 bot 发送。
  - 重复拉黑同一表情包时会提示“已在黑名单中”。
- 已剥离图片/视频生成功能（Seedance 相关逻辑与配置），插件聚焦 AI 群聊、RAG 与表情包能力。
- 好感度系统 V2（Monika 风格）：
  - 新关系表：`nonebot_plugin_ai_groupmate_userrelation_v2`（旧表保留但不再作为主逻辑写入）。
  - 双分值模型：`favorability_raw` 为主驱动（范围 `[-1000, 1000]`），`favorability` 为映射显示分（范围 `[-100, 100]`，`RAW_PER_SCORE=10`）。
  - 状态机：`broken / distressed / upset / normal / happy / affectionate / enamored / love`，按 raw 分段决定关系描述与系数。
  - 正负向非线性：根据当前状态对加分/扣分使用不同倍率，并在接近上下限时自动“饱和”衰减。
  - 日限制机制：普通正向受 `daily_cap`（默认 7）限制；`bypass` 通道有单独上限（默认 10），溢出进入 `daily_gain_bank`（默认上限 70）。
  - 道歉衰减：同类型道歉加分按“第一次全额、第二次半额、第三次及以后衰减至 0”处理。
  - 惩罚冷却（`last_penalty_at` 已接入）：连续负向变更会按上次惩罚时间衰减，冷却窗口 30 分钟（1800 秒），窗口内最小惩罚系数 0.25。

## ⚙️ 配置

配置说明
| 配置项 | 必填 | 默认值 | 说明 |
|:-----:|:----:|:----:|:----:|
| ai_groupmate__bot_name | 是 | 无 | bot 名 |
| ai_groupmate__reply_probability | 否 | 0.01 | 群内发言概率 |
| ai_groupmate__personality_setting | 否 | 无 | 自定义人设 |
| ai_groupmate__milvus_uri | 否 | 无 | milvus 地址 |
| ai_groupmate__milvus_user | 否 | 无| milvus 用户名 |
| ai_groupmate__milvus_password | 否 | 无 | milvus 密码 |
| ai_groupmate__remote_model_base_url | 否 | 无 | 远程模型服务地址（/embed /rerank /clip） |
| ai_groupmate__remote_model_api_key | 否 | 无 | 远程模型服务 API Key |
| ai_groupmate__remote_embedding_base_url | 否 | 无 | embedding 分路地址（硅基流动/OpenAI 风格） |
| ai_groupmate__remote_embedding_api_key | 否 | 无 | embedding 分路 API Key（为空则回退用 remote_model_api_key） |
| ai_groupmate__remote_embedding_model | 否 | 无 | embedding 模型名（如 `BAAI/bge-m3`） |
| ai_groupmate__remote_embedding_dimensions | 否 | 0 | embedding 维度；`0` 表示不传 `dimensions` 字段 |
| ai_groupmate__remote_rerank_base_url | 否 | 无 | rerank 分路地址（硅基流动） |
| ai_groupmate__remote_rerank_api_key | 否 | 无 | rerank 分路 API Key（为空则回退用 remote_model_api_key） |
| ai_groupmate__remote_rerank_model | 否 | 无 | rerank 模型名（如 `BAAI/bge-reranker-v2-m3`） |
| ai_groupmate__remote_clip_base_url | 否 | 无 | clip 分路地址（本地/远程 clip 服务） |
| ai_groupmate__remote_clip_api_key | 否 | 无 | clip 分路 API Key（为空则回退用 remote_model_api_key） |
| ai_groupmate__tavily_api_key | 否 | 无 | tavily api 密钥 |
| ai_groupmate__openai_base_url | 否 | 无| openai 请求地址 |
| ai_groupmate__openai_token | 否 | 无 | openai token |
| ai_groupmate__openai_model | 否 | 无 | openai 模型名 |
| ai_groupmate__vlm_ollama_base_url | 否 | 无| vlm 地址 |
| ai_groupmate__vlm_model | 否 | 无 | vlm 模型名 |
| ai_groupmate__vlm_provider | 否 | ollama| ollama 或 openai |
| ai_groupmate__vlm_openai_base_url | 否 | 无 | vlm openai 请求地址 |
| ai_groupmate__vlm_openai_api_key | 否 | 无 | vlm openai api key |



### 远程模型服务示例配置（端口映射）
示例：模型服务监听 `8001`，通过隧道/FRP 映射到云端 `18001`；Milvus `19530` 映射到 `19350`：

```env
ai_groupmate__remote_model_base_url=http://127.0.0.1:18001
ai_groupmate__milvus_uri=http://127.0.0.1:19350
```

### embedding/rerank 走硅基流动 + clip 走本地示例
说明：`remote_embedding_base_url` 与 `remote_rerank_base_url` 请填写 `https://api.siliconflow.cn`（不要再带 `/v1`，代码会自动拼接）。

```env
ai_groupmate__milvus_uri=http://127.0.0.1:19530

ai_groupmate__remote_embedding_base_url=https://api.siliconflow.cn
ai_groupmate__remote_embedding_api_key=sk-xxxx
ai_groupmate__remote_embedding_model=BAAI/bge-m3
ai_groupmate__remote_embedding_dimensions=0

ai_groupmate__remote_rerank_base_url=https://api.siliconflow.cn
ai_groupmate__remote_rerank_api_key=sk-xxxx
ai_groupmate__remote_rerank_model=BAAI/bge-reranker-v2-m3

ai_groupmate__remote_clip_base_url=http://127.0.0.1:18001
ai_groupmate__remote_clip_api_key=
```

## 🎉 使用

ai会自动偷群内使用的表情包，增加至向量库当中，在回答时通过向量库内容搜索表情包，由于使用了vlm模型，搜索的准确率十分高。
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
|  /ai off | 关闭插件（停止 AI 回复 + 向量/RAG；VLM/聊天记录仍会写入） |
|  /ai status | 查看插件开关与 Milvus 状态 |
