import httpx
from .config import Config
from nonebot import logger
from typing import Optional, Any, Dict
from nonebot import get_plugin_config

plugin_config = get_plugin_config(Config)


# --- 1. 云服务 VLM 调用逻辑 ---
async def _cloud_image_vl(base64_image: str, prompt: str = "请描述一下这个图片") -> Optional[str]:
    """使用 SiliconFlow API 调用云服务 VLM 模型"""
    url = "https://api.siliconflow.cn/v1/chat/completions"
    # ... (保持原有 payload 结构不变，使用 base64_image 构建 content)
    payload = {
        "model": "Pro/Qwen/Qwen2.5-VL-7B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "stream": False,
        "max_tokens": 1024,
        "stop": None,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"},
    }
    headers = {
        "Authorization": f"Bearer {plugin_config.siliconflow_bearer_token}",
        "Content-Type": "application/json",
    }
    return await _make_vlm_request(url, payload, headers)


# --- 2. 本地服务 VLM 调用逻辑 (请根据实际本地服务接口调整) ---
async def _local_image_vl(base64_image: str, prompt: str = "请描述一下这个图片") -> Optional[str]:
    """使用本地 Ollama VLM 进行推理"""

    # Ollama Chat API URL
    url = f"{plugin_config.vlm_ollama_base_url}/api/chat"

    # Ollama 在 messages.content 中接收 Base64 编码的图片
    payload = {
        "model": plugin_config.vlm_model,  # 使用配置中的模型名
        "messages": [
            {
                "role": "user",
                # Ollama 的多模态输入是 contents 数组
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    # 也可以尝试简化为 Ollama 官方推荐的格式 (contents 数组)
                    # {"type": "text", "text": prompt},
                    # {"type": "image_data", "data": base64_image}
                ]
            }
        ],
        "stream": False,
        "options": {
            # Ollama 的参数在 options 字段内，如 temperature, num_predict (max_tokens)
            "temperature": 0.7,
            "num_predict": 1024
        }
    }

    headers = {
        "Content-Type": "application/json",
    }

    # ⚠️ 注意: Ollama 的返回结构略有不同，需要调整 _make_vlm_request 或在这里处理
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=300)

        response.raise_for_status()

        # Ollama 响应结构：response.json()["message"]["content"]
        return response.json()["message"]["content"]

    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama HTTP 错误 {e.response.status_code} on {url}: {e.response.text}")
        return None
    except Exception as e:
        logger.error(f"Ollama 请求失败: {e}")
        return None

# --- 3. 统一请求和错误处理函数 ---
async def _make_vlm_request(url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Optional[str]:
    """发送 HTTP 请求并处理响应"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=300)

        response.raise_for_status()

        # 假设响应结构都是 OpenAI 风格的
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"VLM 请求失败 ({url}): {e}")
        return None


# --- 4. 插件入口函数 (选择逻辑) ---
async def image_vl(base64_image: str, prompt: str = "请描述一下这个图片") -> Optional[str]:
    """
    根据配置选择调用云服务 VLM 或本地 VLM。
    """
    if plugin_config.use_cloud_vlm:
        logger.info("使用云服务 VLM 进行图片识别...")
        return await _cloud_image_vl(base64_image, prompt)
    else:
        logger.info("使用本地 Ollama VLM 进行图片识别...")
        return await _local_image_vl(base64_image, prompt)