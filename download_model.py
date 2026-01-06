#!/usr/bin/env python3
"""
使用 huggingface_hub 下载 DeepSeek-R1 蒸馏版模型
支持断点续传和加速下载
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 可用的 DeepSeek-R1 蒸馏版模型列表
AVAILABLE_MODELS = {
    "1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
}


def download_model(
    model_name: str,
    cache_dir: str = "/home/user/models",
    use_mirror: bool = False,
    max_workers: int = 8,
    resume_download: bool = True,
):
    """
    下载指定的模型
    
    Args:
        model_name: 模型名称或 HuggingFace 仓库 ID
        cache_dir: 模型缓存目录
        use_mirror: 是否使用镜像站 (hf-mirror.com)
        max_workers: 下载并发数
        resume_download: 是否支持断点续传
    """
    # 如果是简写形式，转换为完整模型名
    if model_name.lower() in AVAILABLE_MODELS:
        model_id = AVAILABLE_MODELS[model_name.lower()]
    else:
        model_id = model_name
    
    logger.info(f"开始下载模型: {model_id}")
    logger.info(f"缓存目录: {cache_dir}")
    
    # 创建缓存目录
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置镜像站 (可选,用于中国大陆加速)
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        logger.info("使用 HuggingFace 镜像站: hf-mirror.com")
    
    try:
        # 使用 snapshot_download 下载完整模型
        local_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_dir=os.path.join(cache_dir, model_id.replace("/", "--")),
            local_dir_use_symlinks=False,  # Windows 兼容
            resume_download=resume_download,
            max_workers=max_workers,
            ignore_patterns=["*.md", "*.txt", "LICENSE*"],  # 忽略非必要文件
        )
        
        logger.info(f"模型下载完成!")
        logger.info(f"模型路径: {local_dir}")
        return local_dir
        
    except HfHubHTTPError as e:
        logger.error(f"下载失败: {e}")
        logger.info("提示: 如果遇到网络问题，可以尝试:")
        logger.info("  1. 使用 --mirror 参数启用镜像站")
        logger.info("  2. 设置代理: export https_proxy=http://127.0.0.1:7890")
        logger.info("  3. 使用 huggingface-cli login 登录账户")
        sys.exit(1)
    except Exception as e:
        logger.error(f"下载出错: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="下载 DeepSeek-R1 蒸馏版模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
可用的模型简写:
  1.5b  -> deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  (约 3GB, 测试用)
  7b    -> deepseek-ai/DeepSeek-R1-Distill-Qwen-7B    (约 14GB, 推荐)
  14b   -> deepseek-ai/DeepSeek-R1-Distill-Qwen-14B   (约 28GB)
  32b   -> deepseek-ai/DeepSeek-R1-Distill-Qwen-32B   (约 64GB)
  70b   -> deepseek-ai/DeepSeek-R1-Distill-Llama-70B  (约 140GB)

示例:
  python download_model.py 7b
  python download_model.py 7b --mirror
  python download_model.py deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --cache-dir /data/models
        """
    )
    
    parser.add_argument(
        "model",
        type=str,
        help="模型名称 (如 7b) 或完整 HuggingFace 仓库 ID"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/home/user/models",
        help="模型缓存目录 (默认: /home/user/models)"
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="使用 HuggingFace 镜像站 (hf-mirror.com) 加速下载"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="下载并发数 (默认: 8)"
    )
    
    args = parser.parse_args()
    
    download_model(
        model_name=args.model,
        cache_dir=args.cache_dir,
        use_mirror=args.mirror,
        max_workers=args.workers,
    )
    
    print("\n" + "="*50)
    print("下载完成! 接下来可以运行 vLLM 服务:")
    print("  python start_vllm_server.py")
    print("="*50)


if __name__ == "__main__":
    main()
