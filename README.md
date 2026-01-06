# 📚 AI 小说生成器

基于 vLLM + DeepSeek-R1 蒸馏版模型的本地小说创作助手，支持 AWQ 4-bit 量化和连续小说创作。

## ✨ 功能特性

- 🤖 **本地部署**: 使用 vLLM 高效部署 DeepSeek-R1 蒸馏版模型
- ⚡ **AWQ 量化**: 支持 4-bit 量化，显存占用降低 70%+
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

### 2. AWQ 量化（推荐）

如果你有本地模型权重文件，建议先进行 AWQ 量化以减少显存占用：

```bash
# 量化 70B 模型
python awq_quantization.py \
    --model_path /home/user/models/deepseek-ai--DeepSeek-R1-Distill-Llama-70B \
    --quant_path /home/user/models/DeepSeek-R1-Distill-Llama-70B-AWQ \
    --max_calib_seq_len 2048

# 量化 7B 模型（测试用）
python awq_quantization.py \
    --model_path /home/user/models/deepseek-ai--DeepSeek-R1-Distill-Qwen-7B \
    --quant_path /home/user/models/DeepSeek-R1-Distill-Qwen-7B-AWQ
```

### 3. 启动 vLLM 服务

```bash
# 方式一：使用本地 AWQ 量化模型（推荐）
python start_vllm_server.py --preset local-awq-70b

# 方式二：自定义本地模型路径
python start_vllm_server.py \
    --model /home/user/models/DeepSeek-R1-Distill-Llama-70B-AWQ \
    --quantization awq \
    --tp 2

# 方式三：使用 HuggingFace 在线模型
python start_vllm_server.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

# 查看所有可用预设
python start_vllm_server.py --list-presets

# 查看所有可用模型
python start_vllm_server.py --list-models
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
├── awq_quantization.py    # AWQ 4-bit 量化脚本
├── config.py              # 配置文件
├── download_model.py      # 模型下载脚本
├── start_vllm_server.py   # vLLM 服务启动脚本（支持本地 AWQ 模型）
├── requirements.txt       # Python 依赖
├── README.md              # 项目说明
├── run.sh                 # 一键启动脚本
└── chat_history/          # 历史对话存储目录
```

## 🔧 量化说明

### AWQ 量化优势

| 模型 | 原始大小 | AWQ 4-bit | 显存需求 |
|------|----------|-----------|----------|
| 7B   | ~14GB    | ~4GB      | 8GB+     |
| 14B  | ~28GB    | ~8GB      | 12GB+    |
| 32B  | ~64GB    | ~18GB     | 24GB+    |
| 70B  | ~140GB   | ~35GB     | 48GB+（双卡）|

### 预设配置

| 预设 | 说明 | 适用显卡 |
|------|------|----------|
| `local-awq-70b` | 本地 AWQ 量化 70B 模型 | 双卡 24GB+ |
| `local-awq-70b-single` | 单卡模式 70B | 单卡 48GB+ |
| `local-awq-7b` | 本地 AWQ 量化 7B 模型 | 单卡 8GB+ |
| `12gb` | 12GB 显卡优化 | RTX 4080/3080 |
| `24gb` | 24GB 显卡配置 | RTX 4090/A5000 |
| `48gb` | 48GB 配置 | A6000/双卡 |

## 📖 使用指南

1. **开始新小说**: 在侧边栏输入小说标题，点击"开始创作"
2. **输入创作指令**: 描述故事情节、人物、场景等
3. **续写功能**: 点击"续写"让 AI 自动延续故事
4. **查看历史**: 侧边栏显示所有历史作品，点击可加载
5. **导出作品**: 支持导出纯小说文本或完整对话记录

## ⚠️ 注意事项

- 量化 70B 模型需要约 150GB+ 系统内存
- 量化过程可能需要 1-2 小时
- 推荐使用 AWQ 量化后的本地模型，避免重复下载
- 如遇显存不足，可尝试减小 `--max-model-len` 参数

## 📄 License

MIT License
