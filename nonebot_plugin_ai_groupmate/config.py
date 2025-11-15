from pydantic import BaseModel, Field
from typing import Optional, Literal, List


class APIConfig(BaseModel):
    name: str = Field(..., alias="API_NAME")
    provider: Literal["openai", "gemini"] = Field(..., alias="API_PROVIDER")
    api_key: str = Field(..., alias="API_KEY")
    model_name: str = Field(..., alias="API_MODEL_NAME")
    base_url: Optional[str] = Field(default=None, alias="API_BASE_URL")
    weight: int = Field(default=10, alias="API_WEIGHT")
    timeout: int = Field(default=20, alias="API_TIMEOUT")


class Config(BaseModel):
    bot_name: str = "bot"
    reply_probability: float = 0.01
    personality_setting: str = ""
    milvus_uri: str = "http://localhost:19530"
    milvus_user: str = ""
    milvus_password: str = ""
    tavily_api_key: str = ""

    api_endpoints: List[APIConfig] = Field(
        default=[
            APIConfig(API_NAME="gemini-2.5-flash", API_PROVIDER="gemini", API_KEY="", API_MODEL_NAME="gemini-2.5-flash", API_WEIGHT=20),
            APIConfig(API_NAME="deepseek-chat", API_PROVIDER="openai", API_KEY="", API_MODEL_NAME="deepseek-chat", API_BASE_URL="https://api.deepseek.com/v1", API_WEIGHT=5)
        ],
        alias="API_ENDPOINTS"
    )

    api_retry_interval: int = Field(
        default=300,
        alias="API_RETRY_INTERVAL"  # 5分钟
    )

    siliconflow_bearer_token: str = Field(default="", description="SiliconFlow/其他云服务 VLM 的 API Token")
    use_cloud_vlm: bool = Field(default=True, description="是否使用云服务 VLM (True) 或本地 Ollama VLM (False)")

    # 本地 Ollama 配置
    vlm_ollama_base_url: str = "http://127.0.0.1:11434"
    vlm_model: str = ""
