from pydantic import BaseModel, Field


class ScopedConfig(BaseModel):
    bot_name: str = "bot"
    reply_probability: float = 0.01
    personality_setting: str = ""
    milvus_uri: str = "milvus_demo.db"
    milvus_user: str = ""
    milvus_password: str = ""
    # 兼容旧版：远程模型统一入口（/embed /rerank /clip）
    remote_model_base_url: str = ""
    remote_model_api_key: str = ""
    # embedding 分路（硅基流动/OpenAI 风格接口）
    remote_embedding_base_url: str = ""
    remote_embedding_api_key: str = ""
    remote_embedding_model: str = ""
    # >0 时才透传给 embeddings API；0 表示不传该字段
    remote_embedding_dimensions: int = 0
    # rerank 分路（硅基流动接口）
    remote_rerank_base_url: str = ""
    remote_rerank_api_key: str = ""
    remote_rerank_model: str = ""
    # clip 分路（你的本地/远程 clip API）
    remote_clip_base_url: str = ""
    remote_clip_api_key: str = ""
    tavily_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = ""
    openai_token: str = ""
    vlm_ollama_base_url: str = ""
    vlm_model: str = ""
    vlm_provider: str = "ollama"  # 可选: "ollama", "openai"
    vlm_openai_base_url: str = ""
    vlm_openai_api_key: str = ""
    # Seedance 文生图/视频配置（先用于白名单能力测试）
    seedance_base_url: str = ""
    seedance_api_key: str = ""
    # 即梦官方 OpenAPI（AK/SK）配置
    seedance_access_key_id: str = ""
    seedance_secret_access_key: str = ""
    seedance_endpoint: str = "https://visual.volcengineapi.com"
    seedance_region: str = "cn-north-1"
    seedance_service: str = "cv"
    seedance_action_submit: str = "CVSync2AsyncSubmitTask"
    seedance_action_result: str = "CVSync2AsyncGetResult"
    seedance_api_version: str = "2022-08-31"
    # 不同场景可分别指定 req_key（以控制台文档为准）
    seedance_image_t2i_req_key: str = "jimeng_t2i_v40"
    seedance_image_i2i_req_key: str = "jimeng_i2i_v30"
    seedance_video_t2v_req_key: str = "jimeng_t2v_v30"
    seedance_video_i2v_req_key: str = "jimeng_i2v_v30"
    seedance_image_model: str = ""
    seedance_video_model: str = ""
    # 仅这些 QQ 号可以调用 Seedance 相关工具
    seedance_tool_whitelist: list[str] = Field(default_factory=list)
    # 公开静态目录根路径（会写入 temp/<request_id>/xx.png）
    seedance_static_dir: str = ""
    # 与 seedance_static_dir 对应的公网 URL 前缀，例如 https://example.com/static
    seedance_static_base_url: str = ""
    # 临时参考图清理时间（分钟）
    seedance_temp_ttl_minutes: int = 30
    # 参考图最多使用数量
    seedance_max_reference_images: int = 2
    # 任务轮询配置
    seedance_poll_interval_seconds: int = 3
    seedance_poll_timeout_seconds: int = 90


class Config(BaseModel):
    ai_groupmate: ScopedConfig = Field(default_factory=ScopedConfig)
