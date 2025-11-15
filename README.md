<div align="center">
  <a href="https://v2.nonebot.dev/store"><img src="https://github.com/A-kirami/nonebot-plugin-template/blob/resources/nbp_logo.png" width="180" height="180" alt="NoneBotPluginLogo"></a>
  <br>
  <p><img src="https://github.com/A-kirami/nonebot-plugin-template/blob/resources/NoneBotPlugin.svg" width="240" alt="NoneBotPluginText"></p>
</div>

<div align="center">

# nonebot-plugin-ai-groupmate

_✨ ai groupmate ✨_


<a href="./LICENSE">
    <img src="https://img.shields.io/github/license/yaowan233/nonebot-plugin-ai-groupmate.svg" alt="license">
</a>
<a href="https://pypi.python.org/pypi/nonebot-plugin-ai-groupmate">
    <img src="https://img.shields.io/pypi/v/nonebot-plugin-ai-groupmate.svg" alt="pypi">
</a>
<img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="python">

</div>


## 📖 介绍
使用 RAG 技术，自动对聊天历史储存，储存长记忆。学习群内群友发言习惯，使得 bot 更像真人。

接入vlm，并且自动学习表情包，自动在群内学习并偷取表情包。


## 💿 安装

<details>
<summary>使用 nb-cli 安装</summary>
在 nonebot2 项目的根目录下打开命令行, 输入以下指令即可安装

    nb plugin install nonebot-plugin-ai-groupmate

</details>

<details>
<summary>使用包管理器安装</summary>
在 nonebot2 项目的插件目录下, 打开命令行, 根据你使用的包管理器, 输入相应的安装命令

<details>
<summary>pip</summary>

    pip install nonebot-plugin-ai-groupmate
</details>
<details>
<summary>pdm</summary>

    pdm add nonebot-plugin-ai-groupmate
</details>
<details>
<summary>poetry</summary>

    poetry add nonebot-plugin-ai-groupmate
</details>


打开 nonebot2 项目的 `bot.py` 文件, 在其中写入

    nonebot.load_plugin('nonebot_plugin_ai_groupmate')

</details>




## ⚙️ 插件配置说明 (Config)


| 配置项 | 必填 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `bot_name` | 否 | `bot` | 机器人在群聊中使用的名称。 |
| `reply_probability` | 否 | `0.01` | 机器人在群内主动发言的概率。 |
| `personality_setting` | 否 | 无 | **自定义 Agent 人设**。 |
| `milvus_uri` | 否 | 无 | Milvus 向量数据库的连接地址。 |
| `milvus_user` | 否 | 无 | Milvus 连接用户名。 |
| `milvus_password` | 否 | 无 | Milvus 连接密码。 |
| `tavily_api_key` | 否 | 无 | **Tavily 搜索 API 密钥**。 |
| `siliconflow_bearer_token` | 否 | 无 | **SiliconFlow/其他云服务 VLM 的 API Token**。 |
| `use_cloud_vlm` | 否 | `True` | **是否使用云服务 VLM** (`True`) 或本地 Ollama VLM (`False`) 进行图片识别。 |
| `vlm_ollama_base_url` | 否 | `http://127.0.0.1:11434` | 本地 Ollama VLM 服务的 Base URL。 |
| `vlm_model` | 否 | 无 | 本地 Ollama VLM 的模型名称（例如 `llava`）。 |
| `API_ENDPOINTS` | 否 | 下表 | **API 故障转移配置列表**。配置一个或多个 LLM 服务端点。 |
| `API_RETRY_INTERVAL` | 否 | `300` | 某个 API 端点失败后，冷却（不可用）的秒数。 |


## 🛠️ APIConfig 子配置项

| 配置项 | 环境变量名 | 必填 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `name` | `API_NAME` | 是 | 无 | 该端点的唯一名称。 |
| `provider` | `API_PROVIDER` | 是 | 无 | API 提供商，必须是 `openai` 或 `gemini`。 |
| `api_key` | `API_KEY` | 是 | 无 | 对应的 API 密钥。 |
| `model_name` | `API_MODEL_NAME` | 是 | 无 | 实际使用的模型名称（如 `gpt-4o` 或 `gemini-2.5-flash`）。 |
| `base_url` | `API_BASE_URL` | 否 | 无 | 仅用于 `openai` Provider，指定自定义的 API 请求地址。 |
| `weight` | `API_WEIGHT` | 否 | `10` | **优先级权重**。值越高，越优先被调用。 |
| `timeout` | `API_TIMEOUT` | 否 | `20` | 调用该端点时的超时时间（秒）。 |

## 🎉 使用

待补充
### 指令
