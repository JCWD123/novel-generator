#!/usr/bin/env python3
"""
模型量化脚本

支持多种量化方式：
1. BitsAndBytes 4-bit 量化（推荐，直接在 vLLM 中使用）
2. GPTQ 量化
3. 使用预量化模型

由于 autoawq 库存在兼容性问题，推荐直接使用 vLLM 的运行时量化功能。

用法：
    # 方式一（推荐）：直接在 vLLM 中启用量化，无需预处理
    python start_vllm_server.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --quantization bnb
    
    # 方式二：下载预量化模型
    python awq_quantization.py --download-quantized deepseek-r1-7b
    
    # 方式三：使用 GPTQ 量化（需要 auto-gptq）
    python awq_quantization.py --gptq --model-path /path/to/model --output /path/to/output

注意：
    - vLLM >= 0.6.0 支持 BitsAndBytes 运行时 4-bit 量化
    - 无需预先量化模型，直接在启动时添加 --quantization bnb 即可
    - 预量化模型可以加快启动速度，但灵活性较低
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional


# 预量化模型列表（HuggingFace 上可用的预量化版本）
PRE_QUANTIZED_MODELS = {
    # DeepSeek-R1 AWQ 版本
    "deepseek-r1-7b-awq": "TheBloke/DeepSeek-R1-Distill-Qwen-7B-AWQ",
    "deepseek-r1-14b-awq": "TheBloke/DeepSeek-R1-Distill-Qwen-14B-AWQ",
    "deepseek-r1-32b-awq": "TheBloke/DeepSeek-R1-Distill-Qwen-32B-AWQ",
    
    # DeepSeek-R1 GPTQ 版本
    "deepseek-r1-7b-gptq": "TheBloke/DeepSeek-R1-Distill-Qwen-7B-GPTQ",
    "deepseek-r1-14b-gptq": "TheBloke/DeepSeek-R1-Distill-Qwen-14B-GPTQ",
    
    # 其他常用模型
    "qwen2-7b-awq": "Qwen/Qwen2-7B-Instruct-AWQ",
    "llama3-8b-awq": "casperhansen/llama-3-8b-instruct-awq",
}


def check_dependencies():
    """检查必要的依赖"""
    print("🔍 检查依赖...\n")
    
    try:
        import torch
        print(f"✅ PyTorch 版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用: {torch.cuda.get_device_name(0)}")
            print(f"✅ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️ CUDA 不可用")
    except ImportError:
        print("❌ PyTorch 未安装")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers 版本: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers 未安装")
        return False
    
    # 检查 bitsandbytes（可选）
    try:
        import bitsandbytes
        print(f"✅ BitsAndBytes 版本: {bitsandbytes.__version__}")
    except ImportError:
        print("⚠️ BitsAndBytes 未安装（可选，用于 4-bit 量化）")
    
    return True


def download_model(model_name: str, use_mirror: bool = False) -> bool:
    """下载模型"""
    print(f"\n{'='*70}")
    print(f"📥 下载模型: {model_name}")
    print(f"{'='*70}")
    
    env = os.environ.copy()
    
    if use_mirror:
        env["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"🌐 使用镜像: https://hf-mirror.com")
    
    # 尝试启用 hf_transfer
    try:
        import hf_transfer
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print(f"⚡ 启用 hf_transfer 加速")
    except ImportError:
        print(f"📡 使用普通下载模式")
    
    cmd = [
        sys.executable, "-m", "huggingface_hub.commands.huggingface_cli",
        "download", model_name
    ]
    
    try:
        process = subprocess.run(cmd, env=env)
        return process.returncode == 0
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False


def list_quantized_models():
    """列出可用的预量化模型"""
    print("\n" + "="*70)
    print("📦 可用的预量化模型")
    print("="*70)
    
    print("\n🔸 DeepSeek-R1 系列 (AWQ 4-bit):")
    for key, model in PRE_QUANTIZED_MODELS.items():
        if "deepseek" in key and "awq" in key:
            print(f"  {key}: {model}")
    
    print("\n🔸 DeepSeek-R1 系列 (GPTQ 4-bit):")
    for key, model in PRE_QUANTIZED_MODELS.items():
        if "deepseek" in key and "gptq" in key:
            print(f"  {key}: {model}")
    
    print("\n🔸 其他模型:")
    for key, model in PRE_QUANTIZED_MODELS.items():
        if "deepseek" not in key:
            print(f"  {key}: {model}")
    
    print("\n" + "="*70)
    print("\n💡 使用方法:")
    print("  python awq_quantization.py --download-quantized deepseek-r1-7b-awq --mirror")
    print("\n💡 或者直接使用 vLLM 运行时量化（推荐）:")
    print("  python start_vllm_server.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --quantization bnb")
    print("="*70)


def convert_to_bnb_4bit(
    model_path: str,
    output_path: str,
    compute_dtype: str = "bfloat16"
):
    """
    将模型转换为 BitsAndBytes 4-bit 格式
    
    注意：这会创建一个新的模型目录，但 vLLM 更推荐直接使用运行时量化
    """
    print(f"\n{'='*70}")
    print(f"🔧 BitsAndBytes 4-bit 转换")
    print(f"{'='*70}")
    print(f"📂 源模型: {model_path}")
    print(f"📂 输出路径: {output_path}")
    print(f"{'='*70}\n")
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        # 配置量化
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, compute_dtype),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        print("📥 加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        print("📥 加载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        print("💾 保存量化模型...")
        Path(output_path).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        print(f"\n✅ 转换完成!")
        print(f"📂 输出路径: {output_path}")
        
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请安装: pip install bitsandbytes accelerate")
        return False
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="模型量化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 推荐方式：

  🔹 方式一（最简单）：直接使用 vLLM 运行时量化
     python start_vllm_server.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --quantization bnb
     
  🔹 方式二：下载预量化模型
     python awq_quantization.py --download-quantized deepseek-r1-7b-awq --mirror
     python start_vllm_server.py --model TheBloke/DeepSeek-R1-Distill-Qwen-7B-AWQ --quantization awq

  🔹 方式三：双卡张量并行 + 量化
     python start_vllm_server.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \\
         --quantization bnb --tensor-parallel 2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 说明：
  - vLLM >= 0.6.0 支持 BitsAndBytes 运行时 4-bit 量化
  - 运行时量化无需预处理，直接添加 --quantization bnb 参数
  - 预量化模型启动更快，但需要额外下载
  - 张量并行可将模型分布到多张 GPU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
    )
    
    # 列出可用模型
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出可用的预量化模型"
    )
    
    # 下载预量化模型
    parser.add_argument(
        "--download-quantized",
        type=str,
        metavar="MODEL_KEY",
        help="下载预量化模型（如 deepseek-r1-7b-awq）"
    )
    
    # BNB 转换
    parser.add_argument(
        "--convert-bnb",
        action="store_true",
        help="将模型转换为 BitsAndBytes 4-bit 格式"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="源模型路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出路径"
    )
    
    # 通用选项
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="使用 HuggingFace 镜像 (hf-mirror.com)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查依赖"
    )
    
    args = parser.parse_args()
    
    # 列出模型
    if args.list:
        list_quantized_models()
        return 0
    
    # 检查依赖
    if args.check:
        check_dependencies()
        return 0
    
    # 下载预量化模型
    if args.download_quantized:
        model_key = args.download_quantized
        
        if model_key in PRE_QUANTIZED_MODELS:
            model_name = PRE_QUANTIZED_MODELS[model_key]
        else:
            # 尝试作为完整模型名使用
            model_name = model_key
        
        print(f"\n下载预量化模型: {model_name}")
        
        if download_model(model_name, use_mirror=args.mirror):
            print(f"\n✅ 下载完成!")
            print(f"\n💡 使用方法:")
            
            if "awq" in model_key.lower() or "awq" in model_name.lower():
                print(f"  python start_vllm_server.py --model {model_name} --quantization awq")
            elif "gptq" in model_key.lower() or "gptq" in model_name.lower():
                print(f"  python start_vllm_server.py --model {model_name} --quantization gptq")
            else:
                print(f"  python start_vllm_server.py --model {model_name}")
        else:
            print(f"\n❌ 下载失败")
            return 1
        
        return 0
    
    # BNB 转换
    if args.convert_bnb:
        if not args.model_path or not args.output:
            print("❌ 请指定 --model-path 和 --output")
            return 1
        
        check_dependencies()
        
        if convert_to_bnb_4bit(args.model_path, args.output):
            print(f"\n💡 使用方法:")
            print(f"  python start_vllm_server.py --model {args.output} --quantization bnb")
        else:
            return 1
        
        return 0
    
    # 默认显示帮助
    print("\n" + "="*70)
    print("📚 模型量化指南")
    print("="*70)
    
    print("""
💡 推荐方式：直接使用 vLLM 运行时量化（最简单）

  # 单卡 + BNB 4-bit 量化
  python start_vllm_server.py \\
      --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\
      --quantization bnb

  # 双卡张量并行 + 量化（适合 70B 模型）
  python start_vllm_server.py \\
      --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \\
      --quantization bnb \\
      --tensor-parallel 2

  # 使用本地模型路径
  python start_vllm_server.py \\
      --model /home/user/models/deepseek-ai--DeepSeek-R1-Distill-Llama-70B \\
      --quantization bnb \\
      --tensor-parallel 2
""")
    
    print("="*70)
    print("\n运行 --help 查看更多选项")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
