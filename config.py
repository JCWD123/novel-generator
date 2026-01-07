"""
小说生成器配置文件

支持两种部署方式:
1. Gradio 前端 + transformers 直接加载 AWQ 模型 (推荐)
2. Streamlit 前端 + vLLM 服务器

使用方式1 (Gradio + AWQ):
    python gradio_app.py --model_path ./models/DeepSeek-R1-7B-AWQ --auto_load

使用方式2 (Streamlit + vLLM):
    python start_vllm_server.py --model ./models/DeepSeek-R1-7B-AWQ
    python -m streamlit run app.py
"""
import os
from pathlib import Path

# ==================== 模型配置 ====================
# 推荐的 DeepSeek-R1 蒸馏版模型:
# - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (最小,测试用, ~3GB 显存)
# - deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (适合单卡, ~14GB 显存)
# - deepseek-ai/DeepSeek-R1-Distill-Qwen-14B (需要较大显存, ~28GB)
# - deepseek-ai/DeepSeek-R1-Distill-Qwen-32B (需要多卡或大显存)
# - deepseek-ai/DeepSeek-R1-Distill-Llama-70B (接近72B规模)
#
# AWQ 量化后显存需求约为原始模型的 1/4

# 原始模型名称/路径 (用于下载或量化)
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

# AWQ 量化模型路径 (用于部署)
AWQ_MODEL_PATH = os.getenv("AWQ_MODEL_PATH", "./models/DeepSeek-R1-7B-AWQ")

# 模型缓存目录
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models")

# ==================== Gradio 配置 (方式1) ====================
GRADIO_HOST = os.getenv("GRADIO_HOST", "0.0.0.0")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"

# ==================== vLLM 服务配置 (方式2) ====================
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

# ==================== AWQ 量化配置 ====================
AWQ_CONFIG = {
    "w_bit": 4,              # 量化位宽
    "q_group_size": 128,     # 分组大小
    "zero_point": True,      # 零点量化
    "version": "GEMM",       # 量化版本
}

# 校准数据集选项
CALIBRATION_DATASETS = [
    "pileval",      # 默认，适合大多数场景
    "wikitext",     # 维基百科文本
    "c4",           # Common Crawl
]

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

请根据用户的要求进行小说创作。在续写时，要保持与之前内容的风格一致和情节连贯。
输出格式要求：直接输出小说内容，不要使用markdown格式，不要添加额外的解释。"""


# ==================== 快捷函数 ====================
def get_model_path(use_awq: bool = True) -> str:
    """获取模型路径"""
    if use_awq:
        return AWQ_MODEL_PATH
    return MODEL_NAME


def print_config():
    """打印当前配置"""
    print("="*60)
    print("📋 当前配置")
    print("="*60)
    print(f"  原始模型: {MODEL_NAME}")
    print(f"  AWQ模型: {AWQ_MODEL_PATH}")
    print(f"  模型缓存: {MODEL_CACHE_DIR}")
    print(f"  Gradio: {GRADIO_HOST}:{GRADIO_PORT}")
    print(f"  vLLM: {VLLM_HOST}:{VLLM_PORT}")
    print(f"  GPU数量: {TENSOR_PARALLEL_SIZE}")
    print("="*60)
