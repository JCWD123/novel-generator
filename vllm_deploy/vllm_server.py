#!/usr/bin/env python3
"""
vLLM 服务启动脚本

用于在 Docker 容器内启动 vLLM OpenAI 兼容 API 服务器
"""
import os
import sys
import subprocess

from config import (
    MODEL_PATH,
    VLLM_HOST,
    VLLM_PORT,
    MODEL_NAME,
    TENSOR_PARALLEL_SIZE,
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    DTYPE,
    print_config
)


def check_model_path(model_path: str) -> bool:
    """检查模型路径是否有效"""
    from pathlib import Path
    
    path = Path(model_path)
    if not path.exists():
        print(f"❌ 模型路径不存在: {model_path}")
        return False
    
    # 检查必要文件
    config_file = path / "config.json"
    if not config_file.exists():
        print(f"❌ 未找到 config.json: {model_path}")
        return False
    
    # 检查模型权重文件
    has_weights = any(
        path.glob("*.safetensors")
    ) or any(
        path.glob("*.bin")
    )
    
    if not has_weights:
        print(f"❌ 未找到模型权重文件: {model_path}")
        return False
    
    print(f"✅ 模型路径验证通过: {model_path}")
    return True


def start_vllm_server():
    """启动 vLLM 服务器"""
    
    print("\n" + "=" * 60)
    print("🚀 vLLM OpenAI 兼容服务器")
    print("=" * 60)
    
    # 打印配置
    print_config()
    
    # 检查模型路径
    model_path = os.getenv("MODEL_PATH", MODEL_PATH)
    if not check_model_path(model_path):
        print("\n请确保模型已正确挂载到容器中")
        sys.exit(1)
    
    # 构建启动命令
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", os.getenv("VLLM_HOST", "0.0.0.0"),
        "--port", str(os.getenv("VLLM_PORT", VLLM_PORT)),
        "--tensor-parallel-size", str(os.getenv("TENSOR_PARALLEL_SIZE", TENSOR_PARALLEL_SIZE)),
        "--gpu-memory-utilization", str(os.getenv("GPU_MEMORY_UTILIZATION", GPU_MEMORY_UTILIZATION)),
        "--max-model-len", str(os.getenv("MAX_MODEL_LEN", MAX_MODEL_LEN)),
        "--dtype", os.getenv("DTYPE", DTYPE),
        "--served-model-name", os.getenv("MODEL_NAME", MODEL_NAME),
        "--trust-remote-code",
    ]
    
    # 是否使用 AWQ 量化
    quantization = os.getenv("QUANTIZATION", "")
    if quantization:
        cmd.extend(["--quantization", quantization])
    
    # 是否强制 eager 模式
    if os.getenv("ENFORCE_EAGER", "true").lower() == "true":
        cmd.append("--enforce-eager")
    
    print(f"\n📋 启动命令:\n{' '.join(cmd)}\n")
    print("=" * 60 + "\n")
    
    try:
        # 启动服务器
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n⏹️ 服务器已停止")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 服务器启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    start_vllm_server()

