# 📚 AI 小说生成器 - vLLM Docker 部署

基于 **vLLM + Streamlit + LangChain** 的小说生成器，支持 Docker 一键部署。

## 🎯 特性

- ✅ **vLLM 推理服务**: 高性能 LLM 推理，支持 AWQ 量化
- ✅ **Streamlit 前端**: 现代化 Web 界面，支持流式生成
- ✅ **LangChain 集成**: 完整的历史对话管理
- ✅ **Docker 部署**: 一次构建，多处部署
- ✅ **国内镜像加速**: 使用阿里云、清华源加速下载

## 📋 环境要求

### 稳定组合 (推荐)

| 组件 | 版本 |
|------|------|
| CUDA Runtime | 12.1 |
| NVIDIA Driver | ≥ 535.xx |
| Python | 3.10 |
| PyTorch | 2.1.2 + cu121 |
| Triton | 2.1.0 |
| vLLM | 0.3.3 |
| OS | Ubuntu 22.04 |
| Docker | nvidia-container-toolkit |

### 硬件要求

- **GPU**: NVIDIA GPU (显存 ≥ 8GB，推荐 ≥ 16GB)
- **内存**: ≥ 16GB RAM
- **存储**: ≥ 50GB (镜像 + 模型)

## 🚀 快速开始

### 1. 本地构建镜像

```bash
cd vllm_deploy

# 构建镜像
chmod +x build.sh
./build.sh
```

### 2. 导出镜像

```bash
# 导出为 tar 文件
chmod +x export.sh
./export.sh

# 生成: novel-images.tar
```

### 3. 传输到服务器

```bash
# 使用 scp 传输
scp novel-images.tar user@server:/path/to/deploy/

# 或使用其他方式 (rsync, sftp 等)
```

### 4. 服务器部署

```bash
# SSH 登录服务器
ssh user@server

cd /path/to/deploy/

# 复制部署文件
# - docker-compose.yml
# - deploy.sh

# 运行部署脚本
chmod +x deploy.sh
./deploy.sh novel-images.tar
```

### 5. 准备模型

将模型文件放置到 `models/` 目录：

```
models/
└── DeepSeek-R1-7B-AWQ/
    ├── config.json
    ├── model-00001-of-00002.safetensors
    ├── model-00002-of-00002.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

### 6. 启动服务

```bash
docker compose up -d
```

## 📁 目录结构

```
vllm_deploy/
├── config.py              # 配置文件
├── langchain_history.py   # LangChain 历史管理
├── vllm_client.py         # vLLM 客户端
├── streamlit_app.py       # Streamlit 前端
├── vllm_server.py         # vLLM 服务启动脚本
├── requirements.txt       # Python 依赖
├── Dockerfile.vllm        # vLLM 服务镜像
├── Dockerfile.streamlit   # Streamlit 镜像
├── docker-compose.yml     # Docker Compose 配置
├── streamlit_config.toml  # Streamlit 配置
├── build.sh               # 构建脚本
├── export.sh              # 导出脚本
├── deploy.sh              # 部署脚本
├── chat_history/          # 历史记录存储
└── README.md              # 本文档
```

## ⚙️ 配置说明

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `/models/DeepSeek-R1-7B-AWQ` | 模型路径 |
| `MODEL_NAME` | `deepseek-r1` | API 模型名称 |
| `VLLM_PORT` | `8000` | vLLM 服务端口 |
| `STREAMLIT_PORT` | `8501` | Streamlit 端口 |
| `TENSOR_PARALLEL_SIZE` | `1` | GPU 数量 |
| `GPU_MEMORY_UTILIZATION` | `0.9` | 显存利用率 |
| `MAX_MODEL_LEN` | `4096` | 最大序列长度 |
| `QUANTIZATION` | `awq` | 量化方式 |

### 修改配置

编辑 `docker-compose.yml` 中的 `environment` 部分：

```yaml
environment:
  - MODEL_PATH=/models/your-model
  - TENSOR_PARALLEL_SIZE=2  # 使用 2 张 GPU
  - GPU_MEMORY_UTILIZATION=0.85
```

## 🔧 常用命令

### Docker Compose

```bash
# 启动服务
docker compose up -d

# 查看状态
docker compose ps

# 查看日志
docker compose logs -f
docker compose logs -f vllm-server
docker compose logs -f streamlit

# 停止服务
docker compose down

# 重启服务
docker compose restart
```

### 镜像管理

```bash
# 查看镜像
docker images | grep novel

# 删除镜像
docker rmi novel-vllm:latest novel-streamlit:latest

# 重新构建
docker compose build --no-cache
```

## 🐛 问题排查

### 1. GPU 不可用

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 nvidia-container-toolkit
docker info | grep -i nvidia

# 测试 GPU Docker
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### 2. vLLM 服务启动失败

```bash
# 查看详细日志
docker compose logs vllm-server

# 常见原因:
# - 模型路径错误
# - 显存不足
# - CUDA 版本不匹配
```

### 3. 连接 vLLM 失败

```bash
# 检查服务状态
curl http://localhost:8000/health

# 检查模型列表
curl http://localhost:8000/v1/models
```

### 4. 显存不足

- 降低 `GPU_MEMORY_UTILIZATION` (如 0.7)
- 使用量化模型 (AWQ)
- 减少 `MAX_MODEL_LEN`

## 🌐 访问地址

部署完成后访问:

- **Streamlit 前端**: http://localhost:8501
- **vLLM API**: http://localhost:8000
  - 模型列表: http://localhost:8000/v1/models
  - 健康检查: http://localhost:8000/health

## 📝 API 使用示例

vLLM 提供 OpenAI 兼容 API:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="deepseek-r1",
    messages=[
        {"role": "system", "content": "你是一位小说作家"},
        {"role": "user", "content": "写一个开头"}
    ],
    max_tokens=1024,
    temperature=0.8
)

print(response.choices[0].message.content)
```

## 📄 License

MIT License

