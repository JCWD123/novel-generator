#!/usr/bin/env python3
"""
模型下载脚本 - 模仿 eedi-mining-misconceptions 项目

支持两种下载方式：
1. 从 HuggingFace 下载（使用 hf_transfer 加速）
2. 从 Kaggle 下载（使用 kagglehub）

用法：
    # 从 HuggingFace 下载（推荐）
    HF_HUB_ENABLE_HF_TRANSFER=1 python download_model.py --model 7b
    
    # 使用镜像加速
    HF_HUB_ENABLE_HF_TRANSFER=1 python download_model.py --model 7b --mirror
    
    # 下载所有模型
    python download_model.py --all
"""

import os
import subprocess
import sys

# DeepSeek-R1 蒸馏版模型列表
MODELS = {
    "1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
}


def download_from_hf(model_id: str, use_mirror: bool = False) -> None:
    """
    从 HuggingFace 下载模型
    
    借鉴 Train-parts-eedi 项目的下载方式：
    HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download <model>
    """
    print(f"\n{'='*60}")
    print(f"📥 Downloading: {model_id}")
    print(f"{'='*60}")
    
    env = os.environ.copy()
    
    # 启用 hf_transfer 加速
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    # 使用镜像
    if use_mirror:
        env["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("🌐 Using mirror: hf-mirror.com")
    
    # 使用 huggingface-cli download
    cmd = ["huggingface-cli", "download", model_id]
    
    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"\n✅ Downloaded: {model_id}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to download: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ huggingface-cli not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub[hf_transfer]", "-q"])
        subprocess.run(cmd, env=env, check=True)


def download_from_kaggle(handle: str) -> None:
    """
    从 Kaggle 下载模型 - 模仿 eedi-mining-misconceptions 项目
    """
    try:
        import kagglehub
        
        print(f"\n{'='*60}")
        print(f"📥 Downloading from Kaggle: {handle}")
        print(f"{'='*60}")
        
        local_dir = kagglehub.model_download(handle)
        print(f"✅ Downloaded to: {local_dir}")
        
    except Exception as e:
        print(f"❌ Failed to download: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download DeepSeek-R1 models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models:
  1.5b  -> deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  (~3GB)
  7b    -> deepseek-ai/DeepSeek-R1-Distill-Qwen-7B    (~14GB)
  14b   -> deepseek-ai/DeepSeek-R1-Distill-Qwen-14B   (~28GB)
  32b   -> deepseek-ai/DeepSeek-R1-Distill-Qwen-32B   (~64GB)
  70b   -> deepseek-ai/DeepSeek-R1-Distill-Llama-70B  (~140GB)

Examples:
  # Download with hf_transfer acceleration
  HF_HUB_ENABLE_HF_TRANSFER=1 python download_model.py --model 7b
  
  # Use mirror (for China)
  HF_HUB_ENABLE_HF_TRANSFER=1 python download_model.py --model 7b --mirror
        """
    )
    
    parser.add_argument("--model", "-m", type=str, choices=list(MODELS.keys()),
                        help="Model to download (1.5b/7b/14b/32b/70b)")
    parser.add_argument("--model-id", type=str, help="Full HuggingFace model ID")
    parser.add_argument("--mirror", action="store_true", help="Use hf-mirror.com")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--kaggle", type=str, help="Kaggle model handle")
    
    args = parser.parse_args()
    
    # 从 Kaggle 下载
    if args.kaggle:
        download_from_kaggle(args.kaggle)
        return
    
    # 下载所有模型
    if args.all:
        for key, model_id in MODELS.items():
            download_from_hf(model_id, args.mirror)
        return
    
    # 下载指定模型
    if args.model:
        model_id = MODELS[args.model]
    elif args.model_id:
        model_id = args.model_id
    else:
        parser.print_help()
        print("\n❌ Please specify --model or --model-id")
        sys.exit(1)
    
    download_from_hf(model_id, args.mirror)
    
    print("\n" + "="*60)
    print("✅ Download complete!")
    print("Next: python start_vllm_server.py --model", model_id)
    print("="*60)


if __name__ == "__main__":
    main()
