from pydantic import BaseModel, Field


class ScopedConfig(BaseModel):
    bot_name: str = "bot"
    reply_probability: float = 0.01
    personality_setting: str = ""
    qdrant_uri: str = ""
    qdrant_api_key: str = ""
    chat_vector_dim: int = 1024
    media_vector_dim: int = 2560
    # 兼容旧版：远程模型统一入口（/embed /rerank /clip）
    remote_model_base_url: str = ""
    remote_model_api_key: str = ""
    # embedding 分路（硅基流动/OpenAI 风格接口）
    remote_embedding_base_url: str = ""
    remote_embedding_api_key: str = ""
    remote_embedding_model: str = ""
    # >0 时才透传给 embeddings API
    remote_embedding_dimensions: int = 1024
    # rerank 分路（硅基流动接口）
    remote_rerank_base_url: str = ""
    remote_rerank_api_key: str = ""
    remote_rerank_model: str = ""
    # 媒体 embedding 分路（OpenAI 风格 embeddings 接口）
    remote_media_embedding_provider: str = "aliyun_dashscope"  # 可选: "openai", "aliyun_dashscope"
    remote_media_embedding_base_url: str = ""
    remote_media_embedding_api_key: str = ""
    remote_media_embedding_model: str = ""
    remote_media_embedding_dimensions: int = 2560
    # 媒体 rerank 分路（/v1/rerank）
    remote_media_rerank_provider: str = "openai"  # 可选: "openai", "aliyun_dashscope"
    remote_media_rerank_base_url: str = ""
    remote_media_rerank_api_key: str = ""
    remote_media_rerank_model: str = ""
    # 旧版 clip 配置，保留作兼容别名
    remote_clip_base_url: str = ""
    remote_clip_api_key: str = ""
    tavily_api_key: str = ""
    qwen_token: str = ""
    summary_model: str = "qwen-flash"
    summary_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    summary_api_key: str = ""
    multimodal_model: str = "qwen-vl-max"
    multimodal_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    multimodal_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = ""
    openai_token: str = ""


class Config(BaseModel):
    ai_groupmate: ScopedConfig = Field(default_factory=ScopedConfig)
