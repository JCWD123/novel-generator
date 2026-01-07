# 📚 AI 小说生成器

基于 DeepSeek-R1 蒸馏版模型的本地小说创作助手，支持 AWQ 4-bit 量化，提供两种部署方式。

## ✨ 功能特性

- 🤖 **本地部署**: 完全本地运行，无需云服务
- ⚡ **AWQ 量化**: 4-bit 量化，显存占用降低 75%
- 🎨 **双前端支持**: Gradio (推荐) 或 Streamlit
- 📝 **连续创作**: 支持多轮对话，实现连续小说写作
- 💾 **历史管理**: 自动保存对话历史
- 📤 **导出功能**: 支持导出小说和完整对话记录

## 🚀 快速开始

### 方式一：Gradio 前端 + transformers 直接加载（推荐）

这种方式更简单，无需启动额外的 vLLM 服务器。

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 量化模型（如果还没有 AWQ 模型）
python awq_quantization.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --quant_path ./models/DeepSeek-R1-7B-AWQ

# 3. 启动 Gradio 前端
python gradio_app.py --model_path ./models/DeepSeek-R1-7B-AWQ --auto_load
```

访问 http://localhost:7860 开始创作！

### 方式二：Streamlit 前端 + vLLM 服务器

这种方式适合需要更高性能或多用户访问的场景。

```bash
# 1. 启动 vLLM 服务
python start_vllm_server.py \
    --model ./models/DeepSeek-R1-7B-AWQ \
    --quantization awq

# 2. 启动 Streamlit 前端（新终端）
streamlit run app.py
```

访问 http://localhost:8501 开始创作！

## 📦 AWQ 量化

### 量化命令

```bash
# 基本用法
python awq_quantization.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --quant_path ./models/DeepSeek-R1-7B-AWQ

# 高级配置
python awq_quantization.py \
    --model_path /path/to/model \
    --quant_path /path/to/output \
    --calib_data pileval \
    --max_calib_seq_len 1024 \
    --w_bit 4 \
    --q_group_size 128
```

### 量化参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--w_bit` | 4 | 量化位宽 (2/3/4/8) |
| `--q_group_size` | 128 | 分组大小 (32/64/128/256) |
| `--calib_data` | pileval | 校准数据集 |
| `--max_calib_seq_len` | 1024 | 校准最大序列长度 |

### 显存需求对比

| 模型 | 原始大小 | AWQ 4-bit | 显存需求 |
|------|----------|-----------|----------|
| 1.5B | ~3GB | ~1GB | 4GB+ |
| 7B | ~14GB | ~4GB | 8GB+ |
| 14B | ~28GB | ~8GB | 12GB+ |
| 32B | ~64GB | ~18GB | 24GB+ |
| 70B | ~140GB | ~35GB | 48GB+ |

## 📁 项目结构

```
novel-generator/
├── gradio_app.py          # Gradio 前端（推荐，直接加载 AWQ 模型）
├── app.py                 # Streamlit 前端（需要 vLLM 服务器）
├── awq_quantization.py    # AWQ 量化脚本
├── config.py              # 配置文件
├── download_model.py      # 模型下载脚本
├── start_vllm_server.py   # vLLM 服务启动脚本
├── requirements.txt       # Python 依赖
├── README.md              # 项目说明
├── run.sh                 # 一键启动脚本
└── chat_history/          # 历史对话存储目录
```

## 🔧 两种部署方式对比

| 特性 | Gradio + transformers | Streamlit + vLLM |
|------|----------------------|------------------|
| 部署复杂度 | ⭐ 简单（单进程） | ⭐⭐ 中等（需启动服务器） |
| 推理速度 | ⭐⭐ 中等 | ⭐⭐⭐ 快（连续批处理） |
| 显存效率 | ⭐⭐ 中等 | ⭐⭐⭐ 高（PagedAttention） |
| 多用户支持 | ⭐ 单用户 | ⭐⭐⭐ 多用户 |
| 流式输出 | ✅ 支持 | ✅ 支持 |
| 依赖复杂度 | ⭐ 低 | ⭐⭐ 中 |

**推荐**：
- 个人使用/测试 → Gradio + transformers
- 生产环境/多用户 → Streamlit + vLLM

## 📖 使用指南

### Gradio 界面

1. **加载模型**: 在右侧面板输入 AWQ 模型路径，点击"加载模型"
2. **开始创作**: 输入小说标题和创作指令
3. **生成内容**: 点击"生成"按钮
4. **续写**: 点击"续写"让 AI 继续故事
5. **保存/导出**: 使用右侧面板的保存和导出功能

### Streamlit 界面

1. **新建小说**: 在侧边栏输入小说标题
2. **输入指令**: 描述故事情节、人物、场景
3. **续写功能**: 点击"续写"自动延续故事
4. **历史管理**: 侧边栏查看和加载历史作品
5. **导出作品**: 支持 TXT 格式导出

## 🛠️ 环境配置

### 推荐配置

```bash
# 创建 conda 环境
conda create -n novel-gen python=3.10
conda activate novel-gen

# 安装 PyTorch (根据你的 CUDA 版本)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
pip install -r requirements.txt
```

### 环境变量

```bash
# 模型路径
export AWQ_MODEL_PATH="./models/DeepSeek-R1-7B-AWQ"

# Gradio 配置
export GRADIO_HOST="0.0.0.0"
export GRADIO_PORT="7860"

# vLLM 配置
export VLLM_HOST="0.0.0.0"
export VLLM_PORT="8000"
```

## ⚠️ 注意事项

1. **显存要求**: AWQ 量化后的 7B 模型需要约 8GB 显存
2. **量化时间**: 量化过程可能需要 10 分钟到数小时，取决于模型大小
3. **量化内存**: 量化时需要足够的系统内存（约为模型大小的 2 倍）
4. **首次加载**: 首次加载模型可能较慢，之后会更快

## 🐛 常见问题

### Q: 模型加载失败怎么办？

A: 检查以下几点：
- 模型路径是否正确
- 是否有足够的显存
- autoawq 是否正确安装

### Q: 生成速度太慢？

A: 尝试以下方法：
- 减小 `max_new_tokens` 参数
- 使用 vLLM 部署方式
- 使用更小的模型

### Q: 显存不足？

A: 尝试以下方法：
- 使用更小的模型（如 1.5B 或 7B）
- 减小 `max_length` 参数
- 关闭其他占用显存的程序

## 📄 License

MIT License
