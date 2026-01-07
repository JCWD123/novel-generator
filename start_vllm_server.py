#!/usr/bin/env python3
"""
vLLM 服务启动脚本 - 模仿 Inference-parts-eedi 项目

支持张量并行部署，将模型平均分配到多张 GPU。
自动检测本地已下载的模型，避免重复下载。
强制使用镜像站加速。

用法：
    # 单卡部署
    python start_vllm_server.py --model 1.5b
    
    # 双卡张量并行（推荐用于大模型）
    python start_vllm_server.py --model 70b --tensor-parallel-size 2
    
    # 使用本地模型路径
    python start_vllm_server.py --model /path/to/model --tensor-parallel-size 2
"""

import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path

# ========== 环境变量设置 ==========
# 强制使用镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 检测 hf_transfer 是否安装，只有安装了才启用
def _check_hf_transfer():
    try:
        import hf_transfer
        return True
    except ImportError:
        return False

if _check_hf_transfer():
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    print("⚡ hf_transfer 已安装，启用加速下载")
else:
    # 关键：如果没装 hf_transfer，必须移除或设为 0
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    print("⚠️ hf_transfer 未安装，使用普通下载模式")
    print("   提示：pip install hf_transfer 可加速下载")

# 模型配置
MODELS = {
    "1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
}

# HuggingFace 缓存目录
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")


def get_local_model_path(model_id: str) -> tuple:
    """
    检查模型是否已下载到本地，返回本地 snapshot 路径
    
    直接扫描 HuggingFace 缓存目录结构
    
    Args:
        model_id: HuggingFace 模型 ID (如 deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
        
    Returns:
        (local_path, status) - status: "complete" / "incomplete" / "not_found"
    """
    # 将模型 ID 转换为缓存目录名格式
    cache_dir_name = "models--" + model_id.replace("/", "--")
    model_cache_dir = os.path.join(HF_CACHE_DIR, cache_dir_name)
    
    if not os.path.exists(model_cache_dir):
        return None, "not_found"
    
    # 检查 blobs 目录是否有 .incomplete 文件
    blobs_dir = os.path.join(model_cache_dir, "blobs")
    if os.path.exists(blobs_dir):
        for f in os.listdir(blobs_dir):
            if f.endswith(".incomplete"):
                return None, "incomplete"
    
    # 查找 snapshots 目录
    snapshots_dir = os.path.join(model_cache_dir, "snapshots")
    if not os.path.exists(snapshots_dir):
        return None, "not_found"
    
    # 遍历所有 snapshot
    for snapshot_name in os.listdir(snapshots_dir):
        snapshot_path = os.path.join(snapshots_dir, snapshot_name)
        if not os.path.isdir(snapshot_path):
            continue
        
        # 检查是否有 config.json
        config_path = os.path.join(snapshot_path, "config.json")
        if not os.path.exists(config_path):
            continue
        
        # 检查是否有模型权重文件（可能是软链接）
        has_model = False
        for f in os.listdir(snapshot_path):
            file_path = os.path.join(snapshot_path, f)
            # 检查文件名或软链接目标
            if f.endswith(".safetensors") or f.endswith(".bin"):
                # 如果是软链接，检查目标是否存在
                if os.path.islink(file_path):
                    target = os.path.realpath(file_path)
                    if os.path.exists(target) and os.path.getsize(target) > 100_000_000:  # > 100MB
                        has_model = True
                        break
                elif os.path.isfile(file_path) and os.path.getsize(file_path) > 100_000_000:
                    has_model = True
                    break
        
        if has_model:
            print(f"✅ 检测到本地模型: {snapshot_path}")
            return snapshot_path, "complete"
    
    # 有缓存目录但没有模型权重
    return None, "incomplete"


def resolve_model_path(model: str) -> str:
    """
    解析模型路径 - 核心逻辑
    
    关键：必须返回本地绝对路径给 vLLM，避免触发 snapshot_download()
    """
    # 如果是本地路径且存在
    if os.path.exists(model):
        abs_path = os.path.abspath(model)
        print(f"✅ 使用本地模型: {abs_path}")
        return abs_path
    
    # 解析简写为完整模型 ID
    model_id = MODELS.get(model.lower(), model)
    
    # 查找本地缓存
    local_path, status = get_local_model_path(model_id)
    
    if status == "complete" and local_path:
        return local_path
    
    # 模型不存在或不完整
    if status == "incomplete":
        print(f"\n⚠️ 模型下载不完整: {model_id}")
        print(f"   发现 .incomplete 文件，说明下载被中断")
        print(f"\n   解决方案:")
        print(f"   1. 清除缓存: rm -rf {HF_CACHE_DIR}/models--{model_id.replace('/', '--')}")
        print(f"   2. 重新下载: python download_model.py --model {model}")
    else:
        print(f"\n❌ 模型未在本地找到: {model_id}")
        print(f"   缓存目录: {HF_CACHE_DIR}")
        print(f"\n   请先下载模型:")
        print(f"   python download_model.py --model {model}")
    
    print(f"\n   或者指定本地路径:")
    print(f"   python start_vllm_server.py --model /path/to/model")
    sys.exit(1)


def start_openai_server(
    model: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.95,
    max_model_len: int = 4096,
    dtype: str = "bfloat16",
    enforce_eager: bool = True,
    quantization: str = None,
    served_model_name: str = "deepseek-r1",
):
    """
    启动 OpenAI 兼容的 API 服务器
    """
    # 解析模型路径（必须是本地路径）
    model_path = resolve_model_path(model)
    
    print(f"\n{'='*60}")
    print(f"🚀 启动 vLLM OpenAI 兼容服务器")
    print(f"{'='*60}")
    print(f"模型路径: {model_path}")
    print(f"地址: {host}:{port}")
    print(f"张量并行: {tensor_parallel_size} GPU(s)")
    print(f"显存利用率: {gpu_memory_utilization}")
    print(f"最大序列长度: {max_model_len}")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--dtype", dtype,
        "--served-model-name", served_model_name,
        "--trust-remote-code",
    ]
    
    if enforce_eager:
        cmd.append("--enforce-eager")
    
    if quantization:
        cmd.extend(["--quantization", quantization])
    
    print(f"命令: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n⏹️ 服务器已停止")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 服务器启动失败: {e}")
        sys.exit(1)


def interactive_generate(
    model: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.95,
    max_model_len: int = 4096,
    dtype: str = "bfloat16",
):
    """
    交互式生成 - 模仿 vllm_generate.py 的方式
    """
    import torch
    from vllm import LLM, SamplingParams
    
    # 解析模型路径（必须是本地路径）
    model_path = resolve_model_path(model)
    
    print(f"\n{'='*60}")
    print(f"🚀 创建 vLLM 实例")
    print(f"{'='*60}")
    print(f"模型路径: {model_path}")
    print(f"张量并行: {tensor_parallel_size} GPU(s)")
    print(f"{'='*60}\n")
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype=dtype,
        enforce_eager=True,
        max_model_len=max_model_len,
        disable_log_stats=True,
    )
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=1.0,
        max_tokens=2048,
    )
    
    print("\n" + "="*60)
    print("🎭 交互式生成模式")
    print("输入 'quit' 退出")
    print("="*60 + "\n")
    
    while True:
        try:
            prompt = input(">>> ").strip()
            if prompt.lower() in ["quit", "exit", "q"]:
                break
            if not prompt:
                continue
            
            outputs = llm.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text
            print(f"\n{generated_text}\n")
            
        except KeyboardInterrupt:
            break
    
    del llm
    torch.cuda.empty_cache()
    gc.collect()
    print("\n✅ 已清理")


def main():
    parser = argparse.ArgumentParser(
        description="vLLM 服务器 - 张量并行部署（使用本地模型）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单卡部署（需先下载模型）
  python download_model.py --model 7b
  python start_vllm_server.py --model 7b
  
  # 双卡张量并行（70B 模型）
  python start_vllm_server.py --model 70b --tensor-parallel-size 2
  
  # 使用本地模型路径
  python start_vllm_server.py --model /path/to/model --tensor-parallel-size 2
  
  # 使用 AWQ 量化 + 双卡
  python start_vllm_server.py --model /path/to/awq-model --quantization awq --tensor-parallel-size 2
  
  # 交互式模式
  python start_vllm_server.py --model 7b --interactive

注意: 
  - 必须先用 download_model.py 下载模型
  - 传给 vLLM 的是本地路径，不会触发额外下载
        """
    )
    
    # 模型选择
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="模型名称 (1.5b/7b/14b/32b/70b) 或本地路径")
    
    # vLLM 配置
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=1,
                        help="张量并行 GPU 数量 (默认: 1)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95,
                        help="GPU 显存利用率 (默认: 0.95)")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="最大序列长度 (默认: 4096)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "half"],
                        help="数据类型 (默认: bfloat16)")
    parser.add_argument("--quantization", "-q", type=str, default=None,
                        choices=["awq", None],
                        help="量化方式")
    parser.add_argument("--enforce-eager", action="store_true", default=True,
                        help="禁用 CUDA 图 (默认: True)")
    parser.add_argument("--no-enforce-eager", action="store_false", dest="enforce_eager",
                        help="启用 CUDA 图")
    
    # 服务器配置
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="服务器地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=8000,
                        help="服务器端口 (默认: 8000)")
    parser.add_argument("--served-model-name", type=str, default="deepseek-r1",
                        help="API 模型名称 (默认: deepseek-r1)")
    
    # 模式选择
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="交互式生成模式")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_generate(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
        )
    else:
        start_openai_server(
            model=args.model,
            host=args.host,
            port=args.port,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            enforce_eager=args.enforce_eager,
            quantization=args.quantization,
            served_model_name=args.served_model_name,
        )


if __name__ == "__main__":
    main()
