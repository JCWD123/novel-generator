#!/usr/bin/env python3
"""
AWQ 4-bit 量化脚本

将 DeepSeek-R1 蒸馏版模型量化为 AWQ 4-bit 格式，大幅减少显存占用。

用法：
    # 量化 70B 模型（需要较大内存）
    python awq_quantization.py \
        --model_path /home/user/models/deepseek-ai--DeepSeek-R1-Distill-Llama-70B \
        --quant_path /home/user/models/DeepSeek-R1-Distill-Llama-70B-AWQ \
        --max_calib_seq_len 2048

    # 量化 7B 模型
    python awq_quantization.py \
        --model_path /home/user/models/deepseek-ai--DeepSeek-R1-Distill-Qwen-7B \
        --quant_path /home/user/models/DeepSeek-R1-Distill-Qwen-7B-AWQ
        
    # 使用自定义校准数据
    python awq_quantization.py \
        --model_path /home/user/models/deepseek-ai--DeepSeek-R1-Distill-Llama-70B \
        --quant_path /home/user/models/DeepSeek-R1-Distill-Llama-70B-AWQ \
        --calib_data pileval \
        --max_calib_seq_len 2048 \
        --max_calib_samples 512

量化配置说明：
    - w_bit: 权重位数，4表示4-bit量化
    - q_group_size: 量化分组大小，64是常用值
    - zero_point: 是否使用零点量化
    - version: GEMM表示使用矩阵乘法优化的内核
"""

import argparse
import os
import sys
from pathlib import Path


def check_dependencies():
    """检查必要的依赖是否已安装"""
    missing = []
    
    try:
        import awq
        print(f"✅ AutoAWQ 版本: {awq.__version__}")
    except ImportError:
        missing.append("autoawq")
    
    try:
        import transformers
        print(f"✅ Transformers 版本: {transformers.__version__}")
    except ImportError:
        missing.append("transformers")
    
    try:
        import torch
        print(f"✅ PyTorch 版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用: {torch.cuda.get_device_name(0)}")
            print(f"✅ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️ CUDA 不可用，量化将在 CPU 上进行（非常慢）")
    except ImportError:
        missing.append("torch")
    
    if missing:
        print(f"\n❌ 缺少依赖: {', '.join(missing)}")
        print("请先安装依赖: pip install " + " ".join(missing))
        sys.exit(1)
    
    return True


def quantize_model(
    model_path: str,
    quant_path: str,
    calib_data: str = "pileval",
    max_calib_seq_len: int = 2048,
    max_calib_samples: int = 512,
    w_bit: int = 4,
    q_group_size: int = 64,
    zero_point: bool = True,
    version: str = "GEMM"
):
    """
    执行 AWQ 量化
    
    Args:
        model_path: 原始模型路径
        quant_path: 量化后模型保存路径
        calib_data: 校准数据集名称或路径
        max_calib_seq_len: 校准序列最大长度
        max_calib_samples: 校准样本数量
        w_bit: 权重位数 (4)
        q_group_size: 量化分组大小 (64/128)
        zero_point: 是否使用零点
        version: 量化版本 (GEMM/GEMV)
    """
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer, AwqConfig
    
    print(f"\n{'='*70}")
    print(f"🔧 AWQ 4-bit 量化")
    print(f"{'='*70}")
    print(f"📂 源模型: {model_path}")
    print(f"📂 目标路径: {quant_path}")
    print(f"📊 校准数据: {calib_data}")
    print(f"📏 最大序列长度: {max_calib_seq_len}")
    print(f"📈 校准样本数: {max_calib_samples}")
    print(f"{'='*70}")
    print(f"⚙️ 量化配置:")
    print(f"   ├─ 权重位数: {w_bit}-bit")
    print(f"   ├─ 分组大小: {q_group_size}")
    print(f"   ├─ 零点量化: {'是' if zero_point else '否'}")
    print(f"   └─ 版本: {version}")
    print(f"{'='*70}\n")
    
    # 检查源模型是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        sys.exit(1)
    
    # 创建输出目录
    Path(quant_path).mkdir(parents=True, exist_ok=True)
    
    # 量化配置
    quant_config = {
        "zero_point": zero_point,
        "q_group_size": q_group_size,
        "w_bit": w_bit,
        "version": version
    }
    
    print("📥 加载模型...")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        safetensors=True,  # 优先使用 safetensors 格式
    )
    
    print("📥 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    print(f"\n🔄 开始量化（这可能需要较长时间）...")
    print(f"   使用校准数据: {calib_data}")
    
    # 执行量化
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_data,
        max_calib_seq_len=max_calib_seq_len,
        max_calib_samples=max_calib_samples,
    )
    
    print("\n💾 保存量化模型...")
    
    # 创建量化配置
    quantization_config = AwqConfig(
        bits=quant_config["w_bit"],
        group_size=quant_config["q_group_size"],
        zero_point=quant_config["zero_point"],
        version=quant_config["version"].lower(),
    ).to_dict()
    
    # 更新模型配置
    model.model.config.quantization_config = quantization_config
    
    # 保存模型和 tokenizer
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)
    
    # 计算压缩后大小
    quant_size = sum(f.stat().st_size for f in Path(quant_path).rglob("*") if f.is_file())
    quant_size_gb = quant_size / (1024**3)
    
    print(f"\n{'='*70}")
    print(f"✅ 量化完成!")
    print(f"{'='*70}")
    print(f"📂 保存路径: {quant_path}")
    print(f"📦 模型大小: {quant_size_gb:.2f} GB")
    print(f"{'='*70}")
    print(f"\n💡 使用方法:")
    print(f"   python start_vllm_server.py --model {quant_path} --quantization awq")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="AWQ 4-bit 模型量化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 量化 DeepSeek-R1 70B 蒸馏版
  python awq_quantization.py \\
      --model_path /home/user/models/deepseek-ai--DeepSeek-R1-Distill-Llama-70B \\
      --quant_path /home/user/models/DeepSeek-R1-Distill-Llama-70B-AWQ
      
  # 量化 7B 模型（适合测试）
  python awq_quantization.py \\
      --model_path /home/user/models/deepseek-ai--DeepSeek-R1-Distill-Qwen-7B \\
      --quant_path /home/user/models/DeepSeek-R1-Distill-Qwen-7B-AWQ \\
      --max_calib_seq_len 1024

校准数据集选项:
  - pileval (默认): AutoAWQ 内置的 WikiText 数据集
  - wikitext: HuggingFace wikitext 数据集
  - 自定义 HuggingFace 数据集路径
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/user/models/deepseek-ai--DeepSeek-R1-Distill-Llama-70B",
        help="原始模型路径"
    )
    parser.add_argument(
        "--quant_path",
        type=str,
        default="/home/user/models/DeepSeek-R1-Distill-Llama-70B-AWQ",
        help="量化后模型保存路径"
    )
    parser.add_argument(
        "--calib_data",
        type=str,
        default="pileval",
        help="校准数据集 (pileval/wikitext/自定义HF数据集)"
    )
    parser.add_argument(
        "--max_calib_seq_len",
        type=int,
        default=2048,
        help="校准序列最大长度 (默认: 2048)"
    )
    parser.add_argument(
        "--max_calib_samples",
        type=int,
        default=512,
        help="校准样本数量 (默认: 512)"
    )
    parser.add_argument(
        "--w_bit",
        type=int,
        default=4,
        choices=[4, 8],
        help="量化位数 (默认: 4)"
    )
    parser.add_argument(
        "--q_group_size",
        type=int,
        default=64,
        choices=[32, 64, 128],
        help="量化分组大小 (默认: 64)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查依赖，不执行量化"
    )
    
    args = parser.parse_args()
    
    # 检查依赖
    print("🔍 检查依赖...\n")
    check_dependencies()
    
    if args.check_only:
        print("\n✅ 依赖检查完成")
        return 0
    
    # 执行量化
    quantize_model(
        model_path=args.model_path,
        quant_path=args.quant_path,
        calib_data=args.calib_data,
        max_calib_seq_len=args.max_calib_seq_len,
        max_calib_samples=args.max_calib_samples,
        w_bit=args.w_bit,
        q_group_size=args.q_group_size,
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
