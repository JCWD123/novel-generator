#!/bin/bash
# ========================================
# 本地构建脚本
# ========================================
# 用法:
#   chmod +x build.sh
#   ./build.sh
# ========================================

set -e

echo "=========================================="
echo "🏗️  小说生成器 - Docker 镜像构建"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker 未安装${NC}"
    exit 1
fi

# 检查 Docker Compose
if ! command -v docker compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose 未安装${NC}"
    exit 1
fi

echo -e "${YELLOW}📦 开始构建镜像...${NC}"

# 构建 vLLM 镜像
echo -e "\n${GREEN}[1/2] 构建 vLLM 服务镜像...${NC}"
docker build -t novel-vllm:latest -f Dockerfile.vllm .

# 构建 Streamlit 镜像
echo -e "\n${GREEN}[2/2] 构建 Streamlit 前端镜像...${NC}"
docker build -t novel-streamlit:latest -f Dockerfile.streamlit .

echo -e "\n${GREEN}✅ 镜像构建完成!${NC}"
echo ""
echo "已构建镜像:"
docker images | grep -E "novel-vllm|novel-streamlit"

echo ""
echo "=========================================="
echo "📤 导出镜像命令:"
echo "   docker save -o novel-images.tar novel-vllm:latest novel-streamlit:latest"
echo ""
echo "🚀 本地启动命令:"
echo "   docker compose up -d"
echo "=========================================="

