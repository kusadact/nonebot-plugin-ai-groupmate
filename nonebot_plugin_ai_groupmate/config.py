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


class Config(BaseModel):
    ai_groupmate: ScopedConfig = Field(default_factory=ScopedConfig)
