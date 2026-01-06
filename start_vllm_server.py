#!/usr/bin/env python3
"""
vLLM 模型服务启动脚本

支持本地模型和 HuggingFace 在线模型，带有交互式配置和自动张量并行。

新增功能：
- 交互式模式：自动检测 GPU 显存并给出部署建议
- 自动张量并行：当单卡显存不足时，自动提示使用多卡
- 智能配置：根据显存自动选择最优配置

用法：
    # 交互式模式（推荐，自动检测显存）
    python start_vllm_server.py --interactive
    
    # 自动张量并行（显存不足时自动提示）
    python start_vllm_server.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B --auto-tp
    
    # 双卡张量并行 + BNB 量化
    python start_vllm_server.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \\
        --quantization bnb --tensor-parallel 2
    
    # 使用本地模型路径
    python start_vllm_server.py --model /home/user/models/my-model --tensor-parallel 2
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List, Dict

# ==================== GPU 检测功能 ====================

def get_gpu_info() -> List[Dict]:
    """获取所有 GPU 的详细信息"""
    gpu_list = []
    
    try:
        import torch
        if not torch.cuda.is_available():
            return gpu_list
        
        num_gpus = torch.cuda.device_count()
        
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / (1024**3)
            
            torch.cuda.set_device(i)
            free_mem, _ = torch.cuda.mem_get_info(i)
            free_mem = free_mem / (1024**3)
            used_mem = total_mem - free_mem
            
            gpu_list.append({
                "index": i,
                "name": props.name,
                "total_memory": round(total_mem, 1),
                "free_memory": round(free_mem, 1),
                "used_memory": round(used_mem, 1),
            })
    except ImportError:
        print("⚠️ PyTorch 未安装，无法检测 GPU")
    except Exception as e:
        print(f"⚠️ GPU 检测失败: {e}")
    
    return gpu_list


def print_gpu_info(gpu_list: List[Dict]) -> float:
    """打印 GPU 信息并返回总可用显存"""
    if not gpu_list:
        print("\n❌ 未检测到可用的 NVIDIA GPU")
        return 0
    
    print(f"\n{'='*70}")
    print(f"🖥️  GPU 信息检测")
    print(f"{'='*70}")
    
    total_available = 0
    for gpu in gpu_list:
        print(f"\n  GPU {gpu['index']}: {gpu['name']}")
        print(f"    ├─ 总显存: {gpu['total_memory']:.1f} GB")
        print(f"    ├─ 已使用: {gpu['used_memory']:.1f} GB")
        print(f"    └─ 可用:   {gpu['free_memory']:.1f} GB")
        total_available += gpu['free_memory']
    
    print(f"\n  📊 总可用显存: {total_available:.1f} GB ({len(gpu_list)} 张卡)")
    print(f"{'='*70}")
    
    return total_available


def estimate_model_memory(model_name: str, quantization: str = None, max_model_len: int = 8192) -> float:
    """估算模型所需显存 (GB)"""
    # 基础显存估算
    model_memory = {
        "1.5b": 3.0,
        "7b": 14.0,
        "14b": 28.0,
        "32b": 64.0,
        "70b": 140.0,
    }
    
    # 尝试从模型名推断大小
    base_memory = 14.0  # 默认 7B
    model_lower = model_name.lower()
    for size, mem in model_memory.items():
        if size in model_lower:
            base_memory = mem
            break
    
    # 量化后显存估算
    if quantization in ["bnb", "bitsandbytes"]:
        base_memory *= 0.30
    elif quantization == "awq":
        base_memory *= 0.25
    elif quantization == "gptq":
        base_memory *= 0.25
    elif quantization == "fp8":
        base_memory *= 0.50
    
    # KV cache 估算
    kv_cache = (max_model_len / 8192) * 2.0
    
    # 额外开销
    overhead = 2.0
    
    return base_memory + kv_cache + overhead


def suggest_deployment(
    gpu_list: List[Dict],
    model_name: str,
    quantization: str = None,
    max_model_len: int = 8192
) -> Dict:
    """根据 GPU 配置建议最优部署方案"""
    if not gpu_list:
        return {"error": "没有可用的 GPU", "can_deploy": False}
    
    estimated = estimate_model_memory(model_name, quantization, max_model_len)
    single_gpu = gpu_list[0]['free_memory']
    total_mem = sum(g['free_memory'] for g in gpu_list)
    num_gpus = len(gpu_list)
    
    result = {
        "estimated_memory": estimated,
        "single_gpu_memory": single_gpu,
        "total_memory": total_mem,
        "num_gpus": num_gpus,
        "tensor_parallel": 1,
        "can_deploy": False,
        "message": "",
        "suggestions": [],
    }
    
    # 单卡可以部署
    if single_gpu >= estimated * 1.1:
        result["can_deploy"] = True
        result["tensor_parallel"] = 1
        result["message"] = f"✅ 单卡即可部署（需要 {estimated:.1f}GB，可用 {single_gpu:.1f}GB）"
        return result
    
    # 需要多卡
    if num_gpus > 1 and total_mem >= estimated * 1.1:
        needed_gpus = 2
        for i in range(2, num_gpus + 1):
            if (total_mem / num_gpus) * i >= estimated:
                needed_gpus = i
                break
        
        result["can_deploy"] = True
        result["tensor_parallel"] = needed_gpus
        result["message"] = f"⚡ 建议使用 {needed_gpus} 张卡张量并行（每卡约 {estimated/needed_gpus:.1f}GB）"
        return result
    
    # 显存不足，给出建议
    result["message"] = f"❌ 显存不足（需要 {estimated:.1f}GB，可用 {total_mem:.1f}GB）"
    
    if quantization is None:
        quantized_mem = estimate_model_memory(model_name, "bnb", max_model_len)
        if single_gpu >= quantized_mem * 1.1:
            result["suggestions"].append(f"启用量化 (--quantization bnb)，需要约 {quantized_mem:.1f}GB")
        elif num_gpus > 1 and total_mem >= quantized_mem * 1.1:
            result["suggestions"].append(f"启用量化 + 多卡 (--quantization bnb --tensor-parallel {num_gpus})")
    
    if max_model_len > 4096:
        result["suggestions"].append("减少序列长度 (--max-model-len 4096)")
    
    result["suggestions"].append("使用更小的模型")
    
    return result


def interactive_mode():
    """交互式模式"""
    print("\n" + "="*70)
    print("🚀 vLLM 服务部署向导")
    print("="*70)
    
    gpu_list = get_gpu_info()
    total_mem = print_gpu_info(gpu_list)
    
    if not gpu_list:
        print("\n❌ 无法继续，请确保有可用的 NVIDIA GPU")
        return None
    
    # 选择模型
    print("\n" + "-"*70)
    print("📦 选择模型:")
    print("-"*70)
    
    models = [
        ("1", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "1.5B 参数（测试用）"),
        ("2", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "7B 参数（推荐）"),
        ("3", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "14B 参数"),
        ("4", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "32B 参数"),
        ("5", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "70B 参数（最高质量）"),
        ("6", "custom", "自定义模型路径"),
    ]
    
    for idx, model, desc in models:
        print(f"  [{idx}] {desc}")
        print(f"      {model}")
    
    choice = input("\n请选择 [1-6，默认 2]: ").strip() or "2"
    
    if choice == "6":
        model_name = input("请输入模型路径: ").strip()
    elif choice in ["1", "2", "3", "4", "5"]:
        model_name = models[int(choice) - 1][1]
    else:
        model_name = models[1][1]  # 默认 7B
    
    # 选择量化
    print("\n" + "-"*70)
    print("🔧 选择量化方式:")
    print("-"*70)
    print("  [1] 无量化（全精度，需要更多显存）")
    print("  [2] BitsAndBytes 4-bit（推荐，显存降低 70%）")
    print("  [3] AWQ 4-bit（需要预量化模型）")
    
    quant_choice = input("\n请选择 [1-3，默认 2]: ").strip() or "2"
    quantization = {
        "1": None,
        "2": "bnb",
        "3": "awq"
    }.get(quant_choice, "bnb")
    
    # 选择序列长度
    print("\n" + "-"*70)
    print("📏 选择最大序列长度:")
    print("-"*70)
    print("  [1] 2048（节省显存）")
    print("  [2] 4096（推荐）")
    print("  [3] 8192（标准）")
    print("  [4] 16384（长文本）")
    
    len_choice = input("\n请选择 [1-4，默认 2]: ").strip() or "2"
    max_model_len = {
        "1": 2048,
        "2": 4096,
        "3": 8192,
        "4": 16384
    }.get(len_choice, 4096)
    
    # 分析配置
    print("\n" + "-"*70)
    print("📊 配置分析:")
    print("-"*70)
    
    suggestion = suggest_deployment(gpu_list, model_name, quantization, max_model_len)
    
    print(f"\n  模型: {model_name}")
    print(f"  量化: {quantization or '无'}")
    print(f"  序列长度: {max_model_len}")
    print(f"  预估显存: {suggestion['estimated_memory']:.1f} GB")
    print(f"\n  {suggestion['message']}")
    
    tensor_parallel = suggestion.get('tensor_parallel', 1)
    
    # 显存不足处理
    if not suggestion['can_deploy'] and suggestion.get('suggestions'):
        print("\n  💡 建议:")
        for i, sug in enumerate(suggestion['suggestions'], 1):
            print(f"    [{i}] {sug}")
        
        fix = input("\n选择解决方案编号 (或直接回车继续): ").strip()
        
        if fix == "1" and "量化" in suggestion['suggestions'][0]:
            quantization = "bnb"
            if "多卡" in suggestion['suggestions'][0]:
                tensor_parallel = len(gpu_list)
            suggestion = suggest_deployment(gpu_list, model_name, quantization, max_model_len)
        elif fix == "2" and len(suggestion['suggestions']) > 1:
            max_model_len = 4096
            suggestion = suggest_deployment(gpu_list, model_name, quantization, max_model_len)
    
    # 确认
    print("\n" + "-"*70)
    print("🚀 最终配置:")
    print("-"*70)
    print(f"  模型: {model_name}")
    print(f"  量化: {quantization or '无'}")
    print(f"  序列长度: {max_model_len}")
    print(f"  张量并行: {tensor_parallel} GPU(s)")
    
    confirm = input("\n确认启动？[Y/n]: ").strip().lower()
    
    if confirm != "n":
        return {
            "model": model_name,
            "quantization": quantization,
            "max_model_len": max_model_len,
            "tensor_parallel": tensor_parallel,
        }
    
    return None


# ==================== 模型配置 ====================

HF_MODELS = {
    "1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
}

PRESETS = {
    "12gb": {
        "description": "12GB 显卡配置",
        "quantization": "bnb",
        "max_model_len": 4096,
        "gpu_memory": 0.92,
        "enforce_eager": True,
    },
    "24gb": {
        "description": "24GB 显卡配置",
        "quantization": "bnb",
        "max_model_len": 8192,
        "gpu_memory": 0.90,
        "enforce_eager": False,
    },
    "48gb": {
        "description": "48GB 显卡配置",
        "quantization": None,
        "max_model_len": 8192,
        "gpu_memory": 0.85,
        "enforce_eager": False,
    },
    "dual-48gb": {
        "description": "双卡 48GB 配置（适合 70B）",
        "quantization": "bnb",
        "max_model_len": 8192,
        "gpu_memory": 0.90,
        "tensor_parallel": 2,
        "enforce_eager": True,
    },
}

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_GPU_MEMORY = 0.90


def start_vllm_server(
    model: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY,
    max_model_len: int = None,
    quantization: str = None,
    enforce_eager: bool = False,
    dtype: str = None,
    served_model_name: str = "deepseek-r1",
):
    """启动 vLLM 服务器"""
    
    # 构建命令
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--served-model-name", served_model_name,
        "--trust-remote-code",
    ]
    
    if max_model_len:
        cmd.extend(["--max-model-len", str(max_model_len)])
    
    if quantization:
        if quantization == "bnb":
            quantization = "bitsandbytes"
        cmd.extend(["--quantization", quantization])
    
    if enforce_eager:
        cmd.append("--enforce-eager")
    
    if dtype:
        cmd.extend(["--dtype", dtype])
    
    # 打印配置
    print("\n" + "="*70)
    print("🚀 启动 vLLM 服务器")
    print("="*70)
    print(f"📦 模型: {model}")
    print(f"🔗 地址: http://{host}:{port}")
    print(f"🎯 API 名称: {served_model_name}")
    print(f"💾 显存利用率: {gpu_memory_utilization:.0%}")
    print(f"🖥️ GPU 数量: {tensor_parallel_size}")
    if max_model_len:
        print(f"📏 最大长度: {max_model_len}")
    if quantization:
        print(f"🔧 量化: {quantization}")
    print("="*70)
    
    print(f"\n📝 命令: {' '.join(cmd)}\n")
    
    # 设置环境变量
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n⏹️ 服务器已停止")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 启动失败: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n❌ 找不到 vllm，请安装: pip install vllm>=0.6.0")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="vLLM 服务启动脚本（支持交互式配置和自动张量并行）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 使用示例：

  🔹 交互式模式（推荐）：
     python start_vllm_server.py --interactive

  🔹 自动张量并行：
     python start_vllm_server.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B --auto-tp

  🔹 双卡 + BNB 量化：
     python start_vllm_server.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \\
         --quantization bnb --tensor-parallel 2

  🔹 使用本地模型：
     python start_vllm_server.py --model /home/user/models/my-model --tensor-parallel 2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
    )
    
    parser.add_argument("--interactive", "-i", action="store_true", help="交互式模式")
    parser.add_argument("--auto-tp", action="store_true", help="自动张量并行")
    parser.add_argument("--model", "-m", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT)
    parser.add_argument("--tensor-parallel", "--tp", type=int, default=1)
    parser.add_argument("--gpu-memory", type=float, default=DEFAULT_GPU_MEMORY)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--quantization", "-q", type=str, choices=["bnb", "bitsandbytes", "awq", "gptq", "fp8"])
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16", "auto"])
    parser.add_argument("--preset", type=str, choices=list(PRESETS.keys()))
    parser.add_argument("--gpu-info", action="store_true", help="显示 GPU 信息")
    
    args = parser.parse_args()
    
    # GPU 信息
    if args.gpu_info:
        gpu_list = get_gpu_info()
        print_gpu_info(gpu_list)
        return 0
    
    # 交互式模式
    if args.interactive:
        config = interactive_mode()
        if config:
            start_vllm_server(
                model=config["model"],
                host=args.host,
                port=args.port,
                tensor_parallel_size=config.get("tensor_parallel", 1),
                gpu_memory_utilization=args.gpu_memory,
                max_model_len=config.get("max_model_len"),
                quantization=config.get("quantization"),
                enforce_eager=args.enforce_eager,
            )
        return 0
    
    # 自动张量并行
    tensor_parallel = args.tensor_parallel
    if args.auto_tp:
        gpu_list = get_gpu_info()
        if gpu_list:
            print_gpu_info(gpu_list)
            suggestion = suggest_deployment(
                gpu_list, args.model, args.quantization, args.max_model_len or 8192
            )
            
            if suggestion['tensor_parallel'] > 1:
                print(f"\n⚡ 建议使用张量并行: {suggestion['tensor_parallel']} GPUs")
                confirm = input(f"使用 {suggestion['tensor_parallel']} 张卡？[Y/n]: ").strip().lower()
                if confirm != "n":
                    tensor_parallel = suggestion['tensor_parallel']
    
    # 应用预设
    quantization = args.quantization
    max_model_len = args.max_model_len
    enforce_eager = args.enforce_eager
    gpu_memory = args.gpu_memory
    
    if args.preset:
        preset = PRESETS[args.preset]
        quantization = quantization or preset.get("quantization")
        max_model_len = max_model_len or preset.get("max_model_len")
        enforce_eager = enforce_eager or preset.get("enforce_eager", False)
        gpu_memory = preset.get("gpu_memory", gpu_memory)
        if tensor_parallel == 1:
            tensor_parallel = preset.get("tensor_parallel", 1)
    
    start_vllm_server(
        model=args.model,
        host=args.host,
        port=args.port,
        tensor_parallel_size=tensor_parallel,
        gpu_memory_utilization=gpu_memory,
        max_model_len=max_model_len,
        quantization=quantization,
        enforce_eager=enforce_eager,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
