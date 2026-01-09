"""
vLLM 客户端模块

直接使用 requests 调用 vLLM OpenAI 兼容 API，避免携带未定义的字段
"""
import os
import json
import requests
from typing import List, Dict, Generator, Optional

from config import (
    VLLM_API_URL,
    MODEL_NAME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    NOVEL_SYSTEM_PROMPT
)


class VLLMClient:
    """
    vLLM 客户端 - 直接 requests 调用
    """
    def __init__(
        self,
        api_base: str = None,
        model_name: str = None,
        api_key: str = "EMPTY"  # vLLM 不需要真实 API key
    ):
        self.api_base = api_base or VLLM_API_URL
        self.model_name = model_name or MODEL_NAME
        self.api_key = api_key
        self.session = requests.Session()
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        stream: bool = False,
    ) -> Dict:
        full_messages = [{"role": "system", "content": NOVEL_SYSTEM_PROMPT}] + messages
        payload = {
            "model": self.model_name,
            "messages": full_messages,
            "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
            "temperature": temperature if temperature is not None else DEFAULT_TEMPERATURE,
            "top_p": top_p if top_p is not None else DEFAULT_TOP_P,
            "stream": stream,
        }
        return payload
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        **kwargs
    ) -> Optional[str]:
        """非流式生成，仅发送允许字段"""
        payload = self._build_payload(messages, max_tokens, temperature, top_p, stream=False)
        try:
            resp = self.session.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=300,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            print(f"❌ 生成失败: {resp.status_code} - {resp.text}")
            return None
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return None
    
    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """流式生成，仅发送允许字段"""
        payload = self._build_payload(messages, max_tokens, temperature, top_p, stream=True)
        try:
            resp = self.session.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=300,
            )
            if resp.status_code != 200:
                yield f"❌ 生成失败: {resp.status_code} - {resp.text}"
                return
            full = ""
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith(b"data: "):
                    data = line[6:]
                    if data.strip() == b"[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta:
                            full += delta["content"]
                            yield full
                    except Exception:
                        continue
        except Exception as e:
            yield f"❌ 生成失败: {e}"
    
    def check_health(self) -> bool:
        """
        检查 vLLM 服务健康状态
        
        Returns:
            服务是否可用
        """
        import requests
        
        try:
            # 检查 /health 端点（如果有）
            response = requests.get(
                f"{self.api_base.rstrip('/v1')}/health",
                timeout=5
            )
            return response.status_code == 200
        except:
            pass
        
        try:
            # 检查 /models 端点
            response = requests.get(
                f"{self.api_base}/models",
                timeout=5,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            return response.status_code == 200
        except Exception as e:
            print(f"⚠️ 服务检查失败: {e}")
            return False
    
    def get_model_info(self) -> Optional[Dict]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        import requests
        
        try:
            response = requests.get(
                f"{self.api_base}/models",
                timeout=10,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"⚠️ 获取模型信息失败: {e}")
        
        return None


# ==================== 便捷函数 ====================
_client: Optional[VLLMClient] = None

def get_vllm_client() -> VLLMClient:
    """获取 vLLM 客户端单例"""
    global _client
    if _client is None:
        _client = VLLMClient()
    return _client

def generate(
    messages: List[Dict[str, str]],
    max_tokens: int = None,
    temperature: float = None,
    **kwargs
) -> Optional[str]:
    """非流式生成"""
    return get_vllm_client().generate(messages, max_tokens, temperature, **kwargs)

def generate_stream(
    messages: List[Dict[str, str]],
    max_tokens: int = None,
    temperature: float = None,
    **kwargs
) -> Generator[str, None, None]:
    """流式生成"""
    return get_vllm_client().generate_stream(messages, max_tokens, temperature, **kwargs)

def check_service() -> bool:
    """检查服务状态"""
    return get_vllm_client().check_health()

