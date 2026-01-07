"""
AWQ 量化脚本 - 将原始模型量化为 AWQ 格式

支持两种部署方式：
1. 使用 transformers 直接加载 (Gradio 前端)
2. 使用 vLLM 服务器部署 (Streamlit 前端)

用法：
    # 基本用法
    python awq_quantization.py --model_path /path/to/model --quant_path /path/to/output
    
    # 使用自定义校准数据集
    python awq_quantization.py --model_path /path/to/model --quant_path /path/to/output --calib_data wikitext
    
    # 指定量化位宽
    python awq_quantization.py --model_path /path/to/model --quant_path /path/to/output --w_bit 4 --q_group_size 128

依赖：
    pip install autoawq transformers torch

量化后使用 Gradio 前端部署：
    python gradio_app.py --model_path /path/to/output --auto_load
"""

import argparse
import os
import json
from pathlib import Path

try:
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer, AwqConfig
    HAS_AWQ = True
except ImportError:
    HAS_AWQ = False


def check_dependencies():
    """检查依赖是否安装"""
    if not HAS_AWQ:
        print("="*60)
        print("❌ autoawq 未安装")
        print("="*60)
        print("\n安装方法:")
        print("  pip install autoawq")
        print("\n或者使用 CUDA 特定版本:")
        print("  pip install autoawq --extra-index-url https://download.pytorch.org/whl/cu121")
        print("="*60)
        return False
    return True


def validate_model_path(model_path: str) -> bool:
    """验证模型路径是否有效"""
    path = Path(model_path)
    
    # 检查是否为 HuggingFace 模型 ID 格式
    if "/" in model_path and not path.exists():
        print(f"📦 检测到 HuggingFace 模型 ID: {model_path}")
        print("   将自动从 HuggingFace Hub 下载模型")
        return True
    
    # 检查本地路径
    if not path.exists():
        print(f"❌ 模型路径不存在: {model_path}")
        return False
    
    # 检查必要文件
    config_file = path / "config.json"
    if not config_file.exists():
        print(f"❌ 找不到 config.json: {config_file}")
        return False
    
    print(f"✅ 模型路径有效: {model_path}")
    return True


def get_model_info(model_path: str) -> dict:
    """获取模型信息"""
    info = {"name": model_path, "size": "unknown"}
    
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
            info["hidden_size"] = config.get("hidden_size", "unknown")
            info["num_layers"] = config.get("num_hidden_layers", "unknown")
            info["vocab_size"] = config.get("vocab_size", "unknown")
            info["model_type"] = config.get("model_type", "unknown")
    
    return info


def quantize_model(
    model_path: str,
    quant_path: str,
    calib_data: str = "pileval",
    max_calib_seq_len: int = 1024,
    w_bit: int = 4,
    q_group_size: int = 128,
    zero_point: bool = True,
    version: str = "GEMM"
):
    """
    执行 AWQ 量化
    
    Args:
        model_path: 原始模型路径或 HuggingFace 模型 ID
        quant_path: 量化模型保存路径
        calib_data: 校准数据集 (pileval, wikitext, c4, 或自定义 HuggingFace 数据集)
        max_calib_seq_len: 校准时的最大序列长度
        w_bit: 量化位宽 (通常为 4)
        q_group_size: 量化分组大小 (64 或 128)
        zero_point: 是否使用零点量化
        version: 量化版本 (GEMM 或 GEMV)
    """
    
    # 量化配置
    quant_config = {
        "zero_point": zero_point,
        "q_group_size": q_group_size,
        "w_bit": w_bit,
        "version": version
    }
    
    print(f"\n{'='*60}")
    print(f"🔧 AWQ 量化配置")
    print(f"{'='*60}")
    print(f"  📦 原始模型: {model_path}")
    print(f"  💾 输出路径: {quant_path}")
    print(f"  📊 校准数据: {calib_data}")
    print(f"  📏 最大序列长度: {max_calib_seq_len}")
    print(f"  🔢 量化位宽: {w_bit} bit")
    print(f"  📐 分组大小: {q_group_size}")
    print(f"  🎯 零点量化: {zero_point}")
    print(f"  ⚙️ 版本: {version}")
    print(f"{'='*60}\n")
    
    # 获取模型信息
    model_info = get_model_info(model_path)
    if model_info.get("hidden_size") != "unknown":
        print(f"📋 模型信息:")
        print(f"   类型: {model_info.get('model_type', 'unknown')}")
        print(f"   隐藏层维度: {model_info.get('hidden_size', 'unknown')}")
        print(f"   层数: {model_info.get('num_layers', 'unknown')}")
        print(f"   词表大小: {model_info.get('vocab_size', 'unknown')}")
        print()
    
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
    
    # 执行量化
    print(f"\n🔧 开始量化 (使用 {calib_data} 数据集)...")
    print("   这可能需要几分钟到几小时，取决于模型大小...")
    
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_data,
        max_calib_seq_len=max_calib_seq_len
    )
    
    # 创建输出目录
    os.makedirs(quant_path, exist_ok=True)
    
    # 保存量化配置到 config.json
    print("\n💾 保存量化模型...")
    quantization_config = AwqConfig(
        bits=quant_config["w_bit"],
        group_size=quant_config["q_group_size"],
        zero_point=quant_config["zero_point"],
        version=quant_config["version"].lower(),
    ).to_dict()
    
    model.model.config.quantization_config = quantization_config
    
    # 保存模型和 tokenizer
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)
    
    # 计算压缩率
    original_size = sum(p.numel() * 2 for p in model.model.parameters()) / (1024**3)  # 假设 fp16
    quantized_size = sum(p.numel() * w_bit / 8 for p in model.model.parameters()) / (1024**3)
    compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"✅ 量化完成!")
    print(f"{'='*60}")
    print(f"  📁 输出路径: {quant_path}")
    print(f"  📊 压缩比: ~{compression_ratio:.1f}x")
    print(f"\n💡 使用方法:")
    print(f"\n  方式1: Gradio 前端 (transformers 直接加载)")
    print(f"  ─────────────────────────────────────────")
    print(f"  python gradio_app.py --model_path {quant_path} --auto_load")
    print(f"\n  方式2: vLLM 服务器 (Streamlit 前端)")
    print(f"  ─────────────────────────────────────────")
    print(f"  python start_vllm_server.py --model {quant_path} --quantization awq")
    print(f"  python -m streamlit run app.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AWQ 量化脚本 - 将模型量化为 4-bit AWQ 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置量化 DeepSeek-R1 7B
  python awq_quantization.py \\
      --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\
      --quant_path ./models/DeepSeek-R1-7B-AWQ

  # 使用自定义配置
  python awq_quantization.py \\
      --model_path /path/to/model \\
      --quant_path /path/to/output \\
      --calib_data wikitext \\
      --w_bit 4 \\
      --q_group_size 128

校准数据集选项:
  - pileval: 默认，适合大多数场景
  - wikitext: 维基百科文本
  - c4: Common Crawl 数据集
  - 也可以使用 HuggingFace 上的其他数据集
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="原始模型路径或 HuggingFace 模型 ID"
    )
    parser.add_argument(
        "--quant_path",
        type=str,
        required=True,
        help="量化模型保存路径"
    )
    parser.add_argument(
        "--calib_data",
        type=str,
        default="pileval",
        help="校准数据集 (默认: pileval)"
    )
    parser.add_argument(
        "--max_calib_seq_len",
        type=int,
        default=1024,
        help="校准时的最大序列长度 (默认: 1024)"
    )
    parser.add_argument(
        "--w_bit",
        type=int,
        default=4,
        choices=[2, 3, 4, 8],
        help="量化位宽 (默认: 4)"
    )
    parser.add_argument(
        "--q_group_size",
        type=int,
        default=128,
        choices=[32, 64, 128, 256],
        help="量化分组大小 (默认: 128)"
    )
    parser.add_argument(
        "--no_zero_point",
        action="store_true",
        help="禁用零点量化"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="GEMM",
        choices=["GEMM", "GEMV"],
        help="量化版本 (默认: GEMM)"
    )
    
    args = parser.parse_args()
    
    # 检查依赖
    if not check_dependencies():
        exit(1)
    
    # 验证模型路径
    if not validate_model_path(args.model_path):
        exit(1)
    
    # 执行量化
    try:
        quantize_model(
            model_path=args.model_path,
            quant_path=args.quant_path,
            calib_data=args.calib_data,
            max_calib_seq_len=args.max_calib_seq_len,
            w_bit=args.w_bit,
            q_group_size=args.q_group_size,
            zero_point=not args.no_zero_point,
            version=args.version
        )
    except Exception as e:
        print(f"\n❌ 量化失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
