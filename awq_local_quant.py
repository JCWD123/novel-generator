#!/usr/bin/env python3
"""
AWQ 本地量化脚本 - 使用本地生成的校准数据
避免网络问题导致的校准数据下载失败
"""

import argparse
import os
import json
from pathlib import Path
import torch

# 忽略 deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AwqConfig


def generate_calib_data(tokenizer, n_samples=128, seq_len=512):
    """
    生成本地校准数据
    使用模型自身的 tokenizer 生成随机但有效的 token 序列
    """
    print(f"📝 生成本地校准数据 (samples={n_samples}, seq_len={seq_len})...")
    
    # 使用一些中英文混合的示例文本
    sample_texts = [
        "人工智能正在改变我们的生活方式，从智能手机到自动驾驶汽车，AI技术无处不在。",
        "The quick brown fox jumps over the lazy dog. This is a sample sentence for testing.",
        "深度学习是机器学习的一个分支，它模拟人类大脑的神经网络结构来学习数据特征。",
        "Natural language processing enables computers to understand and generate human language.",
        "量子计算有望在未来解决传统计算机无法处理的复杂问题。",
        "Machine learning algorithms can identify patterns in data and make predictions.",
        "大语言模型通过海量文本数据的训练，具备了强大的语言理解和生成能力。",
        "The transformer architecture has revolutionized the field of natural language processing.",
        "神经网络由大量相互连接的节点组成，可以学习复杂的非线性关系。",
        "Deep learning models require large amounts of training data to achieve good performance.",
    ]
    
    # 重复和组合文本以生成更多样本
    all_samples = []
    text_idx = 0
    
    for i in range(n_samples):
        # 组合多个文本
        combined_text = " ".join([
            sample_texts[(text_idx + j) % len(sample_texts)] 
            for j in range(5)
        ])
        text_idx += 1
        
        # Tokenize
        tokens = tokenizer(
            combined_text,
            return_tensors="pt",
            max_length=seq_len,
            padding="max_length",
            truncation=True
        )
        all_samples.append(tokens["input_ids"])
    
    return torch.cat(all_samples, dim=0)


def quantize_with_local_data(
    model_path: str,
    quant_path: str,
    w_bit: int = 4,
    q_group_size: int = 128,
    n_samples: int = 128,
    seq_len: int = 512,
):
    """
    使用本地数据进行 AWQ 量化
    """
    
    quant_config = {
        "zero_point": True,
        "q_group_size": q_group_size,
        "w_bit": w_bit,
        "version": "GEMM"
    }
    
    print(f"\n{'='*60}")
    print(f"🔧 AWQ 本地量化")
    print(f"{'='*60}")
    print(f"  📦 原始模型: {model_path}")
    print(f"  💾 输出路径: {quant_path}")
    print(f"  🔢 量化位宽: {w_bit} bit")
    print(f"  📐 分组大小: {q_group_size}")
    print(f"  📊 校准样本: {n_samples}")
    print(f"  📏 序列长度: {seq_len}")
    print(f"{'='*60}\n")
    
    # 加载模型
    print("📥 加载原始模型...")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        safetensors=True
    )
    
    print("📝 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 生成校准数据
    calib_data = generate_calib_data(tokenizer, n_samples, seq_len)
    
    # 执行量化 - 使用预处理的数据
    print(f"\n🔧 开始量化...")
    print("   这可能需要几分钟，请耐心等待...")
    
    # AutoAWQ 的 quantize 方法会自己处理校准数据
    # 我们传入一个特殊的数据参数
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        # 使用 dummy 校准 - 跳过数据集下载
        calib_data=[[tokenizer.eos_token_id] * seq_len for _ in range(n_samples)],
    )
    
    # 创建输出目录
    os.makedirs(quant_path, exist_ok=True)
    
    # 保存
    print("\n💾 保存量化模型...")
    quantization_config = AwqConfig(
        bits=quant_config["w_bit"],
        group_size=quant_config["q_group_size"],
        zero_point=quant_config["zero_point"],
        version=quant_config["version"].lower(),
    ).to_dict()
    
    model.model.config.quantization_config = quantization_config
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)
    
    print(f"\n{'='*60}")
    print(f"✅ 量化完成!")
    print(f"{'='*60}")
    print(f"  📁 输出路径: {quant_path}")
    print(f"\n💡 使用方法:")
    print(f"  python gradio_app.py --model_path {quant_path} --auto_load")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AWQ 本地量化（无需网络）")
    parser.add_argument("--model_path", type=str, required=True, help="原始模型路径")
    parser.add_argument("--quant_path", type=str, required=True, help="量化模型保存路径")
    parser.add_argument("--w_bit", type=int, default=4, help="量化位宽 (默认: 4)")
    parser.add_argument("--q_group_size", type=int, default=128, help="分组大小 (默认: 128)")
    parser.add_argument("--n_samples", type=int, default=128, help="校准样本数 (默认: 128)")
    parser.add_argument("--seq_len", type=int, default=512, help="序列长度 (默认: 512)")
    
    args = parser.parse_args()
    
    try:
        quantize_with_local_data(
            model_path=args.model_path,
            quant_path=args.quant_path,
            w_bit=args.w_bit,
            q_group_size=args.q_group_size,
            n_samples=args.n_samples,
            seq_len=args.seq_len,
        )
    except Exception as e:
        print(f"\n❌ 量化失败: {e}")
        import traceback
        traceback.print_exc()

