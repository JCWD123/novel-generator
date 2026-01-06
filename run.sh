#!/bin/bash
# 小说生成器快速启动脚本

echo "========================================"
echo "📚 AI 小说生成器启动脚本"
echo "========================================"

# 激活 conda 环境
echo "激活 conda 环境: vllm_embedding"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_embedding

# 检查是否安装依赖
if ! python -c "import streamlit" 2>/dev/null; then
    echo "安装依赖..."
    pip install -r requirements.txt
fi

# 检查模型是否下载
if [ ! -d "/home/user/models" ]; then
    echo "模型未下载，开始下载 7B 模型..."
    python download_model.py 7b --mirror
fi

# 启动 vLLM 服务 (后台运行)
echo "启动 vLLM 服务..."
python start_vllm_server.py &
VLLM_PID=$!

# 等待 vLLM 服务启动
echo "等待 vLLM 服务启动..."
sleep 30

# 启动 Streamlit
echo "启动 Streamlit 前端..."
streamlit run app.py --server.port 8501

# 清理
kill $VLLM_PID 2>/dev/null
