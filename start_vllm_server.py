#!/usr/bin/env python3
"""
vLLM 服务启动脚本 - 模仿 Inference-parts-eedi 项目

支持张量并行部署，将模型平均分配到多张 GPU。

用法：
    # 单卡部署
    python start_vllm_server.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    
    # 双卡张量并行（推荐用于大模型）
    python start_vllm_server.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B --tensor-parallel-size 2
    
    # 使用本地模型路径
    python start_vllm_server.py --model /path/to/model --tensor-parallel-size 2
"""

import argparse
import gc
import os

import torch
from vllm import LLM, SamplingParams

# 模型配置
MODELS = {
    "1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
}


def create_llm(
    model: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.95,
    max_model_len: int = 4096,
    dtype: str = "bfloat16",
    enforce_eager: bool = True,
    quantization: str = None,
) -> LLM:
    """
    创建 vLLM 实例 - 模仿 Inference-parts-eedi 项目的 run_expert_tutor.py
    
    Args:
        model: 模型路径或 HuggingFace 模型 ID
        tensor_parallel_size: 张量并行 GPU 数量
        gpu_memory_utilization: GPU 显存利用率
        max_model_len: 最大序列长度
        dtype: 数据类型 (bfloat16/float16/half)
        enforce_eager: 禁用 CUDA 图以节省显存
        quantization: 量化方式 (awq/None)
    """
    print(f"\n{'='*60}")
    print(f"🚀 Creating vLLM instance")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Tensor Parallel Size: {tensor_parallel_size}")
    print(f"GPU Memory Utilization: {gpu_memory_utilization}")
    print(f"Max Model Length: {max_model_len}")
    print(f"Dtype: {dtype}")
    print(f"Enforce Eager: {enforce_eager}")
    print(f"Quantization: {quantization or 'None'}")
    print(f"{'='*60}\n")
    
    # 参考 run_expert_tutor.py 的配置
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype=dtype,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
        disable_log_stats=True,
        quantization=quantization,
    )
    
    return llm


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
    import subprocess
    import sys
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting vLLM OpenAI-compatible server")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Host: {host}:{port}")
    print(f"Tensor Parallel Size: {tensor_parallel_size}")
    print(f"GPU Memory Utilization: {gpu_memory_utilization}")
    print(f"Max Model Length: {max_model_len}")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
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
    
    print(f"Command: {' '.join(cmd)}\n")
    
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n⏹️ Server stopped")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server failed: {e}")


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
    llm = create_llm(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
        enforce_eager=True,
    )
    
    # 参考 vllm_generate.py 的采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=1.0,
        max_tokens=2048,
    )
    
    print("\n" + "="*60)
    print("🎭 Interactive Generation Mode")
    print("Type 'quit' to exit")
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
    
    # 清理
    del llm
    torch.cuda.empty_cache()
    gc.collect()
    print("\n✅ Cleaned up")


def main():
    parser = argparse.ArgumentParser(
        description="vLLM Server - Tensor Parallel Deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single GPU
  python start_vllm_server.py --model 7b
  
  # Dual GPU tensor parallel (for 70B model)
  python start_vllm_server.py --model 70b --tensor-parallel-size 2
  
  # Custom model path
  python start_vllm_server.py --model /path/to/model --tensor-parallel-size 2
  
  # With AWQ quantization
  python start_vllm_server.py --model /path/to/awq-model --quantization awq --tensor-parallel-size 2
  
  # Interactive mode
  python start_vllm_server.py --model 7b --interactive
        """
    )
    
    # 模型选择
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Model name (1.5b/7b/14b/32b/70b) or full model path")
    
    # vLLM 配置 - 模仿 Inference-parts-eedi
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (default: 1)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95,
                        help="GPU memory utilization (default: 0.95)")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Maximum sequence length (default: 4096)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "half"],
                        help="Data type (default: bfloat16)")
    parser.add_argument("--quantization", "-q", type=str, default=None,
                        choices=["awq", None],
                        help="Quantization method")
    parser.add_argument("--enforce-eager", action="store_true", default=True,
                        help="Disable CUDA graph (default: True)")
    parser.add_argument("--no-enforce-eager", action="store_false", dest="enforce_eager",
                        help="Enable CUDA graph")
    
    # 服务器配置
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=8000,
                        help="Server port (default: 8000)")
    parser.add_argument("--served-model-name", type=str, default="deepseek-r1",
                        help="Model name for API (default: deepseek-r1)")
    
    # 模式选择
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive generation mode")
    
    args = parser.parse_args()
    
    # 解析模型名称
    model = MODELS.get(args.model, args.model)
    
    print(f"\n{'='*60}")
    print(f"📦 Model: {model}")
    print(f"🖥️ Tensor Parallel: {args.tensor_parallel_size} GPU(s)")
    print(f"{'='*60}")
    
    if args.interactive:
        interactive_generate(
            model=model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
        )
    else:
        start_openai_server(
            model=model,
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
