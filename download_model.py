#!/usr/bin/env python3
"""
模型下载脚本 - 强制使用镜像站加速

用法：
    python download_model.py --model 7b
    python download_model.py --model 1.5b
    python download_model.py --all
"""

import os
import subprocess
import sys

# 强制使用镜像站 - 移除所有直连逻辑
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# DeepSeek-R1 蒸馏版模型列表
MODELS = {
    "1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
}


def check_local_model(model_id: str) -> bool:
    """检查模型是否已下载到本地"""
    try:
        from huggingface_hub import scan_cache_dir
        
        cache_info = scan_cache_dir()
        
        for repo in cache_info.repos:
            if repo.repo_id == model_id:
                if repo.revisions:
                    for revision in repo.revisions:
                        snapshot_path = revision.snapshot_path
                        if snapshot_path.exists():
                            model_files = list(snapshot_path.glob("*.safetensors")) + list(snapshot_path.glob("*.bin"))
                            if model_files:
                                print(f"✅ 模型已存在: {snapshot_path}")
                                return True
        return False
        
    except Exception:
        return False


def download_from_hf(model_id: str) -> None:
    """
    从 HuggingFace 镜像站下载模型
    """
    # 检查是否已下载
    if check_local_model(model_id):
        print(f"⏭️ 跳过下载，模型已存在")
        return
    
    print(f"\n{'='*60}")
    print(f"📥 下载模型: {model_id}")
    print(f"🌐 镜像站: {os.environ.get('HF_ENDPOINT')}")
    print(f"{'='*60}")
    
    # 使用 huggingface-cli download
    cmd = ["huggingface-cli", "download", model_id]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ 下载完成: {model_id}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 下载失败: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ huggingface-cli 未找到，正在安装...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub[hf_transfer]", "-q"])
        subprocess.run(cmd, check=True)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="下载 DeepSeek-R1 模型（强制使用镜像站）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
模型列表:
  1.5b  -> deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  (~3GB)
  7b    -> deepseek-ai/DeepSeek-R1-Distill-Qwen-7B    (~14GB)
  14b   -> deepseek-ai/DeepSeek-R1-Distill-Qwen-14B   (~28GB)
  32b   -> deepseek-ai/DeepSeek-R1-Distill-Qwen-32B   (~64GB)
  70b   -> deepseek-ai/DeepSeek-R1-Distill-Llama-70B  (~140GB)

示例:
  python download_model.py --model 7b
  python download_model.py --model 1.5b
  python download_model.py --all
        """
    )
    
    parser.add_argument("--model", "-m", type=str, choices=list(MODELS.keys()),
                        help="模型名称 (1.5b/7b/14b/32b/70b)")
    parser.add_argument("--model-id", type=str, help="完整 HuggingFace 模型 ID")
    parser.add_argument("--all", action="store_true", help="下载所有模型")
    
    args = parser.parse_args()
    
    print(f"🌐 使用镜像站: {os.environ.get('HF_ENDPOINT')}")
    print(f"⚡ hf_transfer 加速: 已启用")
    
    # 下载所有模型
    if args.all:
        for key, model_id in MODELS.items():
            download_from_hf(model_id)
        return
    
    # 下载指定模型
    if args.model:
        model_id = MODELS[args.model]
    elif args.model_id:
        model_id = args.model_id
    else:
        parser.print_help()
        print("\n❌ 请指定 --model 或 --model-id")
        sys.exit(1)
    
    download_from_hf(model_id)
    
    print("\n" + "="*60)
    print("✅ 下载完成!")
    print(f"下一步: python start_vllm_server.py --model {args.model or model_id}")
    print("="*60)


if __name__ == "__main__":
    main()
