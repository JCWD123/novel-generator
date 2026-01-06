# 📚 AI 小说生成器

基于 vLLM + DeepSeek-R1 蒸馏版模型的本地小说创作助手，支持连续小说创作和历史对话管理。

## ✨ 功能特性

- 🤖 **本地部署**: 使用 vLLM 高效部署 DeepSeek-R1 蒸馏版模型
- 📝 **连续创作**: 支持多轮对话，实现连续小说写作
- 💾 **历史管理**: 自动保存对话历史到本地 TXT 文件
- 🎨 **精美界面**: 现代化 Streamlit 前端，古典美学风格
- 📤 **导出功能**: 支持导出小说和完整对话记录

## 🚀 快速开始

### 1. 环境准备

```bash
# 进入项目目录
cd novel-generator

# 激活 conda 环境
conda activate vllm_embedding

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载模型

```bash
# 下载 7B 模型 (推荐，约 14GB)
python download_model.py 7b

# 如果在中国大陆，可使用镜像加速
python download_model.py 7b --mirror

# 其他可选模型
python download_model.py 1.5b   # 测试用，约 3GB
python download_model.py 14b   # 需要较大显存
python download_model.py 32b   # 需要多卡
python download_model.py 70b   # 接近完整版性能
```

### 3. 启动 vLLM 服务

```bash
# 默认启动 7B 模型
python start_vllm_server.py

# 指定模型和参数
python start_vllm_server.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --tp 2

# 显存不够可以限制上下文长度
python start_vllm_server.py --max-model-len 8192
```

### 4. 启动前端页面

```bash
# 新开一个终端
streamlit run app.py
```

访问 http://localhost:8501 开始创作！

## 📁 项目结构

```
novel-generator/
├── app.py                 # Streamlit 前端应用
├── config.py              # 配置文件
├── download_model.py      # 模型下载脚本
├── start_vllm_server.py   # vLLM 服务启动脚本
├── requirements.txt       # Python 依赖
├── README.md              # 项目说明
└── chat_history/          # 历史对话存储目录
    └── *.txt              # 对话记录文件
```

## 🔧 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `VLLM_API_URL` | vLLM API 地址 | `http://localhost:8000/v1` |
| `MODEL_NAME` | 模型名称 | `deepseek-r1` |
| `STREAMLIT_PORT` | Streamlit 端口 | `8501` |

### 模型选择指南

| 模型 | 显存需求 | 推荐场景 |
|------|----------|----------|
| 1.5B | ~4GB | 测试/开发 |
| 7B | ~16GB | 单卡推荐 |
| 14B | ~32GB | 高质量创作 |
| 32B | ~64GB | 多卡部署 |
| 70B | ~140GB | 最高质量 |

## 📖 使用指南

1. **开始新小说**: 在侧边栏输入小说标题，点击"开始创作"
2. **输入创作指令**: 描述故事情节、人物、场景等
3. **续写功能**: 点击"续写"让 AI 自动延续故事
4. **查看历史**: 侧边栏显示所有历史作品，点击可加载
5. **导出作品**: 支持导出纯小说文本或完整对话记录

## ⚠️ 注意事项

- 首次运行需要下载模型，请确保有足够的磁盘空间和稳定的网络
- 模型推理需要 GPU，请确保 CUDA 环境正确配置
- 如遇网络问题，可使用 `--mirror` 参数启用镜像加速

## 📄 License

MIT License
