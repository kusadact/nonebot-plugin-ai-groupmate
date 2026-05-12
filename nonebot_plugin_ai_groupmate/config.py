from pydantic import BaseModel, Field


class ScopedConfig(BaseModel):
    bot_name: str = "bot"
    reply_probability: float = 0.01
    personality_setting: str = ""
    qdrant_uri: str = ""
    qdrant_api_key: str = ""
    chat_vector_dim: int = 1024
    media_vector_dim: int = 2560
    media_search_recall_limit: int = 6
    media_search_return_limit: int = 5
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
    voice_enabled: bool = False
    voice_base_url: str = ""
    voice_tts_path: str = "/tts"
    voice_health_path: str = "/"
    voice_health_timeout_seconds: float = 2.0
    voice_tts_timeout_seconds: float = 180.0
    voice_health_cache_seconds: float = 30.0
    voice_unhealthy_cache_seconds: float = 10.0
    voice_trust_env_proxy: bool = False
    voice_ref_audio_path: str = "ref_audio/Azuma/Azuma_10.wav"
    voice_prompt_text: str = "完了我找不到他之前的投稿了，反正就是有一个。"
    voice_text_lang: str = "zh"
    voice_prompt_lang: str = "zh"
    voice_text_split_method: str = "cut5"
    voice_request_media_type: str = "wav"
    voice_output_format: str = "wav"
    voice_streaming_mode: int = 0
    voice_batch_size: int = 1
    voice_speed_factor: float = 1.0
    voice_top_k: int = 15
    voice_top_p: float = 1.0
    voice_temperature: float = 1.0
    voice_max_text_length: int = 120
    voice_ffmpeg_path: str = "ffmpeg"
    voice_ffmpeg_timeout_seconds: float = 30.0
    voice_ffmpeg_audio_codec: str = "libopencore_amrnb"
    voice_volume_gain: float = 1.5
    voice_amr_sample_rate: int = 8000
    voice_amr_bitrate: str = "12.2k"


class Config(BaseModel):
    ai_groupmate: ScopedConfig = Field(default_factory=ScopedConfig)
