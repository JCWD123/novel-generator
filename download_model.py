#!/usr/bin/env python3
"""
模型下载脚本 - 强制使用镜像站加速

用法：
    # 先登录 HuggingFace（避免限流）
    huggingface-cli login
    
    # 然后下载
    python download_model.py --model 7b
    python download_model.py --model 1.5b
"""

import os
import subprocess
import sys
import time

# 强制使用镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 检测 hf_transfer
def _check_hf_transfer():
    try:
        import hf_transfer
        return True
    except ImportError:
        return False

if _check_hf_transfer():
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    print("⚡ hf_transfer 加速: 已启用")
else:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    print("⚠️ hf_transfer 未安装，使用普通下载")

# DeepSeek-R1 蒸馏版模型列表
MODELS = {
    "1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
}


def check_hf_login():
    """检查是否已登录 HuggingFace"""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        print(f"✅ 检测到 HF_TOKEN 环境变量")
        return True
    
    # 检查 token 文件
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        print(f"✅ 检测到已登录 HuggingFace")
        return True
    
    return False


def download_with_retry(model_id: str, max_retries: int = 3) -> bool:
    """
    带重试的下载函数
    """
    for attempt in range(max_retries):
        print(f"\n{'='*60}")
        print(f"📥 下载模型: {model_id} (尝试 {attempt + 1}/{max_retries})")
        print(f"🌐 镜像站: {os.environ.get('HF_ENDPOINT')}")
        print(f"{'='*60}")
        
        # 使用新命令 hf download（避免 deprecated 警告）
        cmd = ["hf", "download", model_id]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f"\n✅ 下载完成: {model_id}")
            return True
        except FileNotFoundError:
            # hf 命令不存在，回退到 huggingface-cli
            cmd = ["huggingface-cli", "download", model_id]
            try:
                subprocess.run(cmd, check=True)
                print(f"\n✅ 下载完成: {model_id}")
                return True
            except subprocess.CalledProcessError as e:
                if "429" in str(e) or attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 30
                    print(f"\n⚠️ 请求被限流，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    return False
        except subprocess.CalledProcessError as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30
                print(f"\n⚠️ 下载失败，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                return False
    
    return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="下载 DeepSeek-R1 模型（镜像站加速）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
模型列表:
  1.5b  -> deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  (~3GB)
  7b    -> deepseek-ai/DeepSeek-R1-Distill-Qwen-7B    (~14GB)
  14b   -> deepseek-ai/DeepSeek-R1-Distill-Qwen-14B   (~28GB)
  32b   -> deepseek-ai/DeepSeek-R1-Distill-Qwen-32B   (~64GB)
  70b   -> deepseek-ai/DeepSeek-R1-Distill-Llama-70B  (~140GB)

如果遇到 429 限流错误，请先登录:
  huggingface-cli login

或设置 token:
  export HF_TOKEN="hf_xxxxxxxxxxxx"
        """
    )
    
    parser.add_argument("--model", "-m", type=str, choices=list(MODELS.keys()),
                        help="模型名称 (1.5b/7b/14b/32b/70b)")
    parser.add_argument("--model-id", type=str, help="完整 HuggingFace 模型 ID")
    parser.add_argument("--retry", type=int, default=3, help="重试次数 (默认: 3)")
    
    args = parser.parse_args()
    
    print(f"🌐 使用镜像站: {os.environ.get('HF_ENDPOINT')}")
    
    # 检查登录状态
    if not check_hf_login():
        print("\n⚠️ 未检测到 HuggingFace 登录")
        print("   如果遇到 429 限流错误，请先登录:")
        print("   huggingface-cli login")
        print("")
    
    # 确定模型 ID
    if args.model:
        model_id = MODELS[args.model]
    elif args.model_id:
        model_id = args.model_id
    else:
        parser.print_help()
        print("\n❌ 请指定 --model 或 --model-id")
        sys.exit(1)
    
    # 下载
    success = download_with_retry(model_id, max_retries=args.retry)
    
    if success:
        print("\n" + "="*60)
        print("✅ 下载完成!")
        print(f"下一步: python start_vllm_server.py --model {args.model or model_id}")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ 下载失败!")
        print("   可能的原因:")
        print("   1. 网络问题")
        print("   2. 镜像站限流 (429)")
        print("")
        print("   解决方案:")
        print("   1. huggingface-cli login  # 登录获取 token")
        print("   2. 稍后重试")
        print("   3. 尝试官方源: export HF_ENDPOINT=https://huggingface.co")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()

