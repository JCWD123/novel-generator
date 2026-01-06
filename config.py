"""
小说生成器配置文件
"""
import os
from pathlib import Path

# ==================== 模型配置 ====================
# DeepSeek-R1 蒸馏版模型 (72B 参数量太大，建议使用较小的蒸馏版本)
# 可选模型:
# - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (最小,测试用)
# - deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (适合单卡)
# - deepseek-ai/DeepSeek-R1-Distill-Qwen-14B (需要较大显存)
# - deepseek-ai/DeepSeek-R1-Distill-Qwen-32B (需要多卡)
# - deepseek-ai/DeepSeek-R1-Distill-Llama-70B (接近72B规模)

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

# 模型缓存目录
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/home/user/models")

# ==================== vLLM 服务配置 ====================
VLLM_HOST = os.getenv("VLLM_HOST", "0.0.0.0")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))

# GPU 配置
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))  # GPU 数量
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))

# ==================== 生成参数配置 ====================
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 50
DEFAULT_REPETITION_PENALTY = 1.1

# ==================== 历史记录配置 ====================
PROJECT_ROOT = Path(__file__).parent
HISTORY_DIR = PROJECT_ROOT / "chat_history"
HISTORY_DIR.mkdir(exist_ok=True)

# ==================== Streamlit 配置 ====================
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# ==================== 小说创作提示词 ====================
NOVEL_SYSTEM_PROMPT = """你是一位才华横溢的小说作家，擅长创作引人入胜的故事。你的写作特点：

1. **文笔优美**: 善于运用修辞手法，文字富有诗意和画面感
2. **情节紧凑**: 故事发展有张有弛，情节跌宕起伏
3. **人物鲜明**: 角色性格立体，对话生动自然
4. **细节丰富**: 场景描写细腻，能让读者身临其境
5. **连贯性强**: 能够根据之前的情节自然延续故事发展

请根据用户的要求进行小说创作。在续写时，要保持与之前内容的风格一致和情节连贯。"""
