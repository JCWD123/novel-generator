#!/bin/bash
# ========================================
# 服务器部署脚本
# ========================================
# 用法:
#   chmod +x deploy.sh
#   ./deploy.sh [镜像文件路径]
# ========================================

set -e

echo "=========================================="
echo "🚀 小说生成器 - 服务器部署"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 镜像文件
IMAGE_FILE="${1:-novel-images.tar}"

# 检查 nvidia-docker
check_nvidia_docker() {
    echo -e "${YELLOW}🔍 检查 NVIDIA Docker 环境...${NC}"
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}❌ 未检测到 NVIDIA 驱动${NC}"
        echo "   请安装 NVIDIA 驱动 >= 535.xx"
        return 1
    fi
    
    echo "GPU 信息:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    
    if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
        echo -e "${YELLOW}⚠️ 未检测到 nvidia-container-toolkit${NC}"
        echo "   安装命令:"
        echo "   distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
        echo "   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -"
        echo "   curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list"
        echo "   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
        echo "   sudo systemctl restart docker"
        return 1
    fi
    
    echo -e "${GREEN}✅ NVIDIA Docker 环境正常${NC}"
    return 0
}

# 加载镜像
load_images() {
    if [ -f "$IMAGE_FILE" ]; then
        echo -e "\n${YELLOW}📥 加载 Docker 镜像: $IMAGE_FILE${NC}"
        docker load -i "$IMAGE_FILE"
        echo -e "${GREEN}✅ 镜像加载完成${NC}"
    else
        echo -e "${YELLOW}⚠️ 未找到镜像文件 $IMAGE_FILE${NC}"
        echo "   如果镜像已存在，可以跳过此步骤"
    fi
}

# 创建必要目录
create_directories() {
    echo -e "\n${YELLOW}📁 创建必要目录...${NC}"
    mkdir -p models
    mkdir -p chat_history
    mkdir -p logs/vllm
    echo -e "${GREEN}✅ 目录创建完成${NC}"
}

# 检查模型
check_model() {
    echo -e "\n${YELLOW}🔍 检查模型文件...${NC}"
    
    MODEL_DIR="${MODEL_DIR:-./models}"
    
    if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
        echo -e "${RED}❌ 模型目录为空: $MODEL_DIR${NC}"
        echo ""
        echo "请将模型文件放置到 $MODEL_DIR 目录"
        echo "例如: $MODEL_DIR/DeepSeek-R1-7B-AWQ/"
        echo ""
        echo "模型目录应包含:"
        echo "  - config.json"
        echo "  - *.safetensors 或 *.bin"
        echo "  - tokenizer.json"
        echo "  - tokenizer_config.json"
        return 1
    fi
    
    echo "模型目录内容:"
    ls -la "$MODEL_DIR"
    echo -e "${GREEN}✅ 模型检查完成${NC}"
    return 0
}

# 启动服务
start_services() {
    echo -e "\n${YELLOW}🚀 启动服务...${NC}"
    
    # 停止旧容器
    docker compose down 2>/dev/null || true
    
    # 启动服务
    docker compose up -d
    
    echo -e "\n${GREEN}✅ 服务启动完成!${NC}"
    echo ""
    echo "=========================================="
    echo -e "${BLUE}服务地址:${NC}"
    echo "  - vLLM API:   http://localhost:8000"
    echo "  - Streamlit:  http://localhost:8501"
    echo ""
    echo -e "${BLUE}查看日志:${NC}"
    echo "  - docker compose logs -f vllm-server"
    echo "  - docker compose logs -f streamlit"
    echo ""
    echo -e "${BLUE}停止服务:${NC}"
    echo "  - docker compose down"
    echo "=========================================="
}

# 主流程
main() {
    echo ""
    
    # 1. 检查 NVIDIA Docker
    if ! check_nvidia_docker; then
        echo -e "\n${RED}❌ 环境检查失败${NC}"
        exit 1
    fi
    
    # 2. 加载镜像
    load_images
    
    # 3. 创建目录
    create_directories
    
    # 4. 检查模型
    if ! check_model; then
        echo -e "\n${YELLOW}⚠️ 请先准备模型文件，然后重新运行此脚本${NC}"
        exit 1
    fi
    
    # 5. 启动服务
    start_services
}

main "$@"

