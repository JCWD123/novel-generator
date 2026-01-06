#!/usr/bin/env python3
"""
vLLM 模型服务启动脚本

支持本地 AWQ 量化模型和 HuggingFace 在线模型。

用法：
    # 使用本地 AWQ 量化模型（推荐）
    python start_vllm_server.py --preset local-awq-70b
    
    # 使用自定义本地模型路径
    python start_vllm_server.py --model /home/user/models/DeepSeek-R1-Distill-Llama-70B-AWQ --quantization awq
    
    # 使用 HuggingFace 在线模型
    python start_vllm_server.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    
    # 12GB 显卡优化配置
    python start_vllm_server.py --preset 12gb
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# ==================== 模型配置 ====================

# 本地模型路径配置
LOCAL_MODEL_PATHS = {
    # AWQ 量化后的本地模型
    "deepseek-r1-70b-awq": "/home/user/models/DeepSeek-R1-Distill-Llama-70B-AWQ",
    "deepseek-r1-7b-awq": "/home/user/models/DeepSeek-R1-Distill-Qwen-7B-AWQ",
    
    # 原始未量化模型
    "deepseek-r1-70b": "/home/user/models/deepseek-ai--DeepSeek-R1-Distill-Llama-70B",
    "deepseek-r1-7b": "/home/user/models/deepseek-ai--DeepSeek-R1-Distill-Qwen-7B",
}

# HuggingFace 在线模型配置
HF_MODEL_CONFIGS = {
    "1.5b": {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "description": "1.5B 参数，测试用",
        "gpu_memory": "4GB+",
    },
    "7b": {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "description": "7B 参数，单卡推荐",
        "gpu_memory": "16GB+",
    },
    "14b": {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "description": "14B 参数，需要较大显存",
        "gpu_memory": "32GB+",
    },
    "32b": {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "description": "32B 参数，需要多卡",
        "gpu_memory": "64GB+",
    },
    "70b": {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "description": "70B 参数，最高质量",
        "gpu_memory": "140GB+",
    },
}

# 预设配置
PRESETS = {
    # ========== 本地 AWQ 量化模型预设 ==========
    "local-awq-70b": {
        "description": "本地 AWQ 量化 70B 模型（推荐，显存占用约 35GB）",
        "model": "/home/user/models/DeepSeek-R1-Distill-Llama-70B-AWQ",
        "quantization": "awq",
        "max_model_len": 8192,
        "gpu_memory": 0.90,
        "tensor_parallel": 2,  # 70B 建议双卡
        "enforce_eager": True,
    },
    "local-awq-70b-single": {
        "description": "本地 AWQ 量化 70B 模型（单卡模式，需要 48GB+ 显存）",
        "model": "/home/user/models/DeepSeek-R1-Distill-Llama-70B-AWQ",
        "quantization": "awq",
        "max_model_len": 4096,
        "gpu_memory": 0.95,
        "tensor_parallel": 1,
        "enforce_eager": True,
    },
    "local-awq-7b": {
        "description": "本地 AWQ 量化 7B 模型",
        "model": "/home/user/models/DeepSeek-R1-Distill-Qwen-7B-AWQ",
        "quantization": "awq",
        "max_model_len": 8192,
        "gpu_memory": 0.90,
        "tensor_parallel": 1,
        "enforce_eager": False,
    },
    
    # ========== 显存优化预设 ==========
    "12gb": {
        "description": "RTX 4080/3080 12GB 优化配置",
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "quantization": None,  # 使用 BNB 或 AWQ
        "max_model_len": 4096,
        "gpu_memory": 0.92,
        "tensor_parallel": 1,
        "enforce_eager": True,
    },
    "24gb": {
        "description": "RTX 4090/A5000 24GB 配置",
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "quantization": "awq",
        "max_model_len": 8192,
        "gpu_memory": 0.90,
        "tensor_parallel": 1,
        "enforce_eager": False,
    },
    "48gb": {
        "description": "A6000/双卡 48GB 配置",
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "quantization": "awq",
        "max_model_len": 16384,
        "gpu_memory": 0.85,
        "tensor_parallel": 1,
        "enforce_eager": False,
    },
    "multi-gpu": {
        "description": "多卡配置（自动检测 GPU 数量）",
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "quantization": "awq",
        "max_model_len": 8192,
        "gpu_memory": 0.85,
        "tensor_parallel": "auto",
        "enforce_eager": False,
    },
}

# 默认配置
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90


def check_local_model(model_path: str) -> bool:
    """检查本地模型是否存在"""
    path = Path(model_path)
    if path.exists():
        # 检查是否有模型文件
        model_files = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
        config_file = path / "config.json"
        if model_files and config_file.exists():
            return True
    return False


def get_gpu_count() -> int:
    """获取可用 GPU 数量"""
    try:
        import torch
        return torch.cuda.device_count()
    except:
        return 1


def list_models():
    """列出所有可用模型"""
    print("\n" + "="*70)
    print("📦 可用模型配置")
    print("="*70)
    
    print("\n🔸 本地 AWQ 量化模型:")
    print("-"*70)
    for key, path in LOCAL_MODEL_PATHS.items():
        status = "✅ 已就绪" if check_local_model(path) else "❌ 未找到"
        print(f"  {key}")
        print(f"    路径: {path}")
        print(f"    状态: {status}")
        print()
    
    print("\n🔸 HuggingFace 在线模型:")
    print("-"*70)
    for key, config in HF_MODEL_CONFIGS.items():
        print(f"  {key}")
        print(f"    模型: {config['name']}")
        print(f"    显存: {config['gpu_memory']}")
        print(f"    说明: {config['description']}")
        print()
    
    print("="*70)


def list_presets():
    """列出所有预设配置"""
    print("\n" + "="*70)
    print("⚙️ 预设配置")
    print("="*70)
    
    for key, preset in PRESETS.items():
        print(f"\n  --preset {key}")
        print(f"    说明: {preset['description']}")
        print(f"    模型: {preset['model']}")
        print(f"    量化: {preset.get('quantization') or '无'}")
        print(f"    最大长度: {preset.get('max_model_len', '默认')}")
        tp = preset.get('tensor_parallel', 1)
        print(f"    张量并行: {tp if tp != 'auto' else '自动检测'}")
    
    print("\n" + "="*70)


def start_vllm_server(
    model: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    max_model_len: int = None,
    quantization: str = None,
    download_dir: str = None,
    trust_remote_code: bool = True,
    enforce_eager: bool = False,
    dtype: str = None,
    served_model_name: str = "deepseek-r1",
):
    """
    启动 vLLM OpenAI 兼容服务器
    """
    # 检查模型路径
    is_local = model.startswith("/") or model.startswith("./")
    if is_local and not check_local_model(model):
        print(f"\n❌ 本地模型不存在: {model}")
        print("\n💡 请先运行量化脚本创建本地模型:")
        print(f"   python awq_quantization.py --model_path <原始模型路径> --quant_path {model}")
        print("\n   或者使用 HuggingFace 在线模型:")
        print("   python start_vllm_server.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        sys.exit(1)
    
    # 构建命令
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--served-model-name", served_model_name,
    ]
    
    # 可选参数
    if max_model_len:
        cmd.extend(["--max-model-len", str(max_model_len)])
    
    if quantization:
        cmd.extend(["--quantization", quantization])
    
    if download_dir:
        cmd.extend(["--download-dir", download_dir])
    
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    
    if enforce_eager:
        cmd.append("--enforce-eager")
    
    if dtype:
        cmd.extend(["--dtype", dtype])
    
    # 打印启动信息
    print("\n" + "="*70)
    print("🚀 启动 vLLM 服务器")
    print("="*70)
    print(f"📦 模型: {model}")
    print(f"🔗 服务地址: http://{host}:{port}")
    print(f"🎯 API 模型名: {served_model_name}")
    print(f"💾 显存利用率: {gpu_memory_utilization:.0%}")
    print(f"🖥️ GPU 数量: {tensor_parallel_size}")
    if max_model_len:
        print(f"📏 最大上下文: {max_model_len}")
    if quantization:
        print(f"🔧 量化方式: {quantization.upper()}")
    if enforce_eager:
        print(f"⚡ CUDA 图: 禁用（节省显存）")
    print("="*70)
    
    print(f"\n📝 执行命令:")
    print(f"   {' '.join(cmd)}\n")
    
    # 设置环境变量
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    
    # 执行命令
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n⏹️ 服务器已停止")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 服务器启动失败: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n❌ 找不到 vllm 命令，请确保已安装:")
        print("   pip install vllm>=0.6.0")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="启动 vLLM 服务器（支持本地 AWQ 量化模型）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 使用示例：

  🔹 推荐：使用本地 AWQ 量化 70B 模型
     python start_vllm_server.py --preset local-awq-70b

  🔹 使用自定义本地模型路径
     python start_vllm_server.py \\
         --model /home/user/models/DeepSeek-R1-Distill-Llama-70B-AWQ \\
         --quantization awq --tp 2

  🔹 使用 HuggingFace 在线 7B 模型
     python start_vllm_server.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

  🔹 12GB 显卡优化配置
     python start_vllm_server.py --preset 12gb

  🔹 列出所有可用模型
     python start_vllm_server.py --list-models

  🔹 列出所有预设配置
     python start_vllm_server.py --list-presets

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 量化工作流：
  1. 先运行量化脚本：
     python awq_quantization.py \\
         --model_path /home/user/models/deepseek-ai--DeepSeek-R1-Distill-Llama-70B \\
         --quant_path /home/user/models/DeepSeek-R1-Distill-Llama-70B-AWQ

  2. 启动量化后的模型：
     python start_vllm_server.py --preset local-awq-70b
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
    )
    
    # 模型选择
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="模型名称或本地路径"
    )
    
    # 预设配置
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()),
        help="使用预设配置 (local-awq-70b/12gb/24gb/...)"
    )
    
    # 服务配置
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"监听地址 (默认: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=DEFAULT_PORT,
        help=f"监听端口 (默认: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default="deepseek-r1",
        help="API 模型名称 (默认: deepseek-r1)"
    )
    
    # GPU 配置
    parser.add_argument(
        "--tp", "--tensor-parallel-size",
        type=int,
        default=1,
        dest="tensor_parallel_size",
        help="张量并行 GPU 数量 (默认: 1)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help=f"GPU 显存利用率 (默认: {DEFAULT_GPU_MEMORY_UTILIZATION})"
    )
    
    # 模型配置
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="最大上下文长度"
    )
    parser.add_argument(
        "--quantization", "-q",
        type=str,
        choices=["awq", "gptq", "squeezellm", "fp8", "bitsandbytes"],
        default=None,
        help="量化方式 (awq/gptq/fp8/bitsandbytes)"
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="禁用 CUDA 图，减少显存占用"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float16", "float32", "auto"],
        default=None,
        help="数据类型"
    )
    
    # 其他选项
    parser.add_argument(
        "--download-dir",
        type=str,
        default=None,
        help="模型下载目录"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="列出所有可用模型"
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="列出所有预设配置"
    )
    
    args = parser.parse_args()
    
    # 信息查询
    if args.list_models:
        list_models()
        return 0
    
    if args.list_presets:
        list_presets()
        return 0
    
    # 应用预设配置
    if args.preset:
        preset = PRESETS[args.preset]
        print(f"\n📋 应用预设配置: {args.preset}")
        print(f"   {preset['description']}")
        
        # 使用预设值（但用户显式指定的参数优先）
        model = args.model or preset.get("model")
        quantization = args.quantization or preset.get("quantization")
        max_model_len = args.max_model_len or preset.get("max_model_len")
        enforce_eager = args.enforce_eager or preset.get("enforce_eager", False)
        
        # 处理张量并行
        tp = preset.get("tensor_parallel", 1)
        if tp == "auto":
            tp = get_gpu_count()
            print(f"   自动检测到 {tp} 个 GPU")
        tensor_parallel_size = args.tensor_parallel_size if args.tensor_parallel_size != 1 else tp
        
        gpu_memory = preset.get("gpu_memory", DEFAULT_GPU_MEMORY_UTILIZATION)
        if args.gpu_memory_utilization != DEFAULT_GPU_MEMORY_UTILIZATION:
            gpu_memory = args.gpu_memory_utilization
    else:
        # 使用命令行参数或默认值
        model = args.model or "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        quantization = args.quantization
        max_model_len = args.max_model_len
        enforce_eager = args.enforce_eager
        tensor_parallel_size = args.tensor_parallel_size
        gpu_memory = args.gpu_memory_utilization
    
    # 启动服务器
    start_vllm_server(
        model=model,
        host=args.host,
        port=args.port,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory,
        max_model_len=max_model_len,
        quantization=quantization,
        download_dir=args.download_dir,
        enforce_eager=enforce_eager,
        dtype=args.dtype,
        served_model_name=args.served_model_name,
    )


if __name__ == "__main__":
    main()
