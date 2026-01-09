#!/bin/bash
# ========================================
# 镜像导出脚本
# ========================================
# 用法:
#   chmod +x export.sh
#   ./export.sh
# ========================================

set -e

echo "=========================================="
echo "📤 小说生成器 - Docker 镜像导出"
echo "=========================================="

OUTPUT_FILE="${1:-novel-images.tar}"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 检查镜像是否存在
echo -e "${YELLOW}🔍 检查镜像...${NC}"

if ! docker image inspect novel-vllm:latest &> /dev/null; then
    echo "❌ 未找到 novel-vllm:latest 镜像"
    echo "   请先运行: ./build.sh"
    exit 1
fi

if ! docker image inspect novel-streamlit:latest &> /dev/null; then
    echo "❌ 未找到 novel-streamlit:latest 镜像"
    echo "   请先运行: ./build.sh"
    exit 1
fi

# 导出镜像
echo -e "${YELLOW}💾 导出镜像到 ${OUTPUT_FILE}...${NC}"
docker save -o "$OUTPUT_FILE" novel-vllm:latest novel-streamlit:latest

# 显示文件信息
FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
echo ""
echo -e "${GREEN}✅ 导出完成!${NC}"
echo ""
echo "文件信息:"
echo "  - 路径: $(pwd)/$OUTPUT_FILE"
echo "  - 大小: $FILE_SIZE"
echo ""
echo "=========================================="
echo "📥 服务器端加载命令:"
echo "   docker load -i $OUTPUT_FILE"
echo ""
echo "🚀 服务器端启动命令:"
echo "   docker compose up -d"
echo "=========================================="

