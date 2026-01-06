#!/usr/bin/env python3
"""
使用 vLLM 部署 DeepSeek-R1 蒸馏版模型
提供 OpenAI 兼容的 API 接口
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# 默认配置
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9


def start_vllm_server(
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    max_model_len: int = None,
    quantization: str = None,
    download_dir: str = None,
    trust_remote_code: bool = True,
):
    """
    启动 vLLM OpenAI 兼容服务器
    """
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--served-model-name", "deepseek-r1",
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
    
    print("="*60)
    print("启动 vLLM 服务器")
    print("="*60)
    print(f"模型: {model}")
    print(f"地址: http://{host}:{port}")
    print(f"GPU 数量: {tensor_parallel_size}")
    print(f"显存利用率: {gpu_memory_utilization}")
    print("="*60)
    print("\n运行命令:")
    print(" ".join(cmd))
    print("\n")
    
    # 执行命令
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except subprocess.CalledProcessError as e:
        print(f"服务器启动失败: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="启动 vLLM OpenAI 兼容服务器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认启动 7B 模型
  python start_vllm_server.py
  
  # 指定模型和 GPU 数量
  python start_vllm_server.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --tp 2
  
  # 使用量化减少显存占用
  python start_vllm_server.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --quantization awq
  
  # 限制上下文长度
  python start_vllm_server.py --max-model-len 8192
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"模型名称或路径 (默认: {DEFAULT_MODEL})"
    )
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
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="最大模型上下文长度 (默认: 模型默认值)"
    )
    parser.add_argument(
        "--quantization", "-q",
        type=str,
        choices=["awq", "gptq", "squeezellm", "fp8"],
        default=None,
        help="量化方式 (可选: awq, gptq, squeezellm, fp8)"
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=None,
        help="模型下载目录"
    )
    
    args = parser.parse_args()
    
    start_vllm_server(
        model=args.model,
        host=args.host,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        quantization=args.quantization,
        download_dir=args.download_dir,
    )


if __name__ == "__main__":
    main()
