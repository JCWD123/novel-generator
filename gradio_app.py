#!/usr/bin/env python3
"""
小说生成器 Gradio 前端
直接使用 transformers 加载 AWQ 量化模型（不依赖 vLLM）

用法:
    python gradio_app.py --model_path /path/to/awq_model
    
    或使用环境变量:
    MODEL_PATH=/path/to/awq_model python gradio_app.py
"""

import gradio as gr
import torch
import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Generator, Optional, Tuple
from threading import Thread

# LangChain 消息类型（轻量级实现，不依赖完整 langchain）
class BaseMessage:
    """消息基类"""
    def __init__(self, content: str, role: str = ""):
        self.content = content
        self.role = role
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}
    
    @classmethod
    def from_dict(cls, data: Dict) -> "BaseMessage":
        role = data.get("role", "")
        content = data.get("content", "")
        if role == "user" or role == "human":
            return HumanMessage(content)
        elif role == "assistant" or role == "ai":
            return AIMessage(content)
        elif role == "system":
            return SystemMessage(content)
        return BaseMessage(content, role)

class HumanMessage(BaseMessage):
    """用户消息"""
    def __init__(self, content: str):
        super().__init__(content, "user")

class AIMessage(BaseMessage):
    """AI 助手消息"""
    def __init__(self, content: str):
        super().__init__(content, "assistant")

class SystemMessage(BaseMessage):
    """系统消息"""
    def __init__(self, content: str):
        super().__init__(content, "system")


class ChatHistory:
    """聊天历史管理器 - 使用 LangChain 风格的消息对象"""
    
    def __init__(self):
        self.messages: List[BaseMessage] = []
    
    def add_user_message(self, content: str):
        """添加用户消息"""
        self.messages.append(HumanMessage(content))
    
    def add_ai_message(self, content: str):
        """添加 AI 消息"""
        self.messages.append(AIMessage(content))
    
    def update_last_ai_message(self, content: str):
        """更新最后一条 AI 消息（用于流式生成）"""
        if self.messages and isinstance(self.messages[-1], AIMessage):
            self.messages[-1].content = content
    
    def get_messages_for_model(self) -> List[Dict[str, str]]:
        """获取用于模型的消息列表"""
        return [msg.to_dict() for msg in self.messages]
    
    def to_gradio_format(self) -> List[Dict[str, str]]:
        """转换为 Gradio 6.x Chatbot 格式 [{"role": "...", "content": "..."}, ...]"""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]
    
    @classmethod
    def from_gradio_format(cls, messages: List[Dict[str, str]]) -> "ChatHistory":
        """从 Gradio 6.x Chatbot 格式创建"""
        history = cls()
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                history.add_user_message(content)
            elif role == "assistant":
                history.add_ai_message(content)
        return history
    
    def to_json_serializable(self) -> List[Dict]:
        """转换为可 JSON 序列化的格式"""
        return [msg.to_dict() for msg in self.messages]
    
    @classmethod
    def from_json(cls, data: List[Dict]) -> "ChatHistory":
        """从 JSON 数据恢复"""
        history = cls()
        for item in data:
            history.messages.append(BaseMessage.from_dict(item))
        return history
    
    def clear(self):
        """清空历史"""
        self.messages = []
    
    def __len__(self):
        return len(self.messages)
    
    def __bool__(self):
        return len(self.messages) > 0

# ==================== 配置 ====================
MODEL_PATH = os.getenv("MODEL_PATH", "./models/DeepSeek-R1-AWQ")
HISTORY_DIR = Path(__file__).parent / "chat_history"
HISTORY_DIR.mkdir(exist_ok=True)

# 检测设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ 运行设备: {DEVICE}")

# 小说创作系统提示词
NOVEL_SYSTEM_PROMPT = """你是一位才华横溢的小说作家，擅长创作引人入胜的故事。你的写作特点：

1. **文笔优美**: 善于运用修辞手法，文字富有诗意和画面感
2. **情节紧凑**: 故事发展有张有弛，情节跌宕起伏
3. **人物鲜明**: 角色性格立体，对话生动自然
4. **细节丰富**: 场景描写细腻，能让读者身临其境
5. **连贯性强**: 能够根据之前的情节自然延续故事发展

请根据用户的要求进行小说创作。在续写时，要保持与之前内容的风格一致和情节连贯。
输出格式要求：直接输出小说内容，不要使用markdown格式，不要添加额外的解释。"""


# ==================== 全局模型变量 ====================
model = None
tokenizer = None


def load_awq_model(model_path: str):
    """
    加载 AWQ 量化模型
    支持两种方式：
    1. 使用 transformers 原生加载（推荐）
    2. 使用 AutoAWQ 加载
    """
    global model, tokenizer
    
    print(f"\n{'='*60}")
    print(f"📥 加载 AWQ 模型: {model_path}")
    print(f"{'='*60}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # 检查是否为 AWQ 模型
    config_path = Path(model_path) / "config.json"
    is_awq = False
    
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
            if "quantization_config" in config:
                quant_config = config["quantization_config"]
                if quant_config.get("quant_method") == "awq":
                    is_awq = True
                    print(f"✅ 检测到 AWQ 量化配置:")
                    print(f"   - bits: {quant_config.get('bits', 4)}")
                    print(f"   - group_size: {quant_config.get('group_size', 128)}")
    
    # 加载 tokenizer
    print("📝 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        use_fast=True
    )
    
    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print("🔧 加载模型权重...")
    
    # 对于 AWQ 模型，transformers 4.36+ 原生支持
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",  # 自动分配到可用设备
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    model.eval()  # 设置为评估模式
    
    print(f"\n{'='*60}")
    print(f"✅ 模型加载完成!")
    print(f"   - 设备: {next(model.parameters()).device}")
    print(f"   - 参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"{'='*60}\n")
    
    return model, tokenizer


def format_messages(messages: List[Dict], system_prompt: str = NOVEL_SYSTEM_PROMPT) -> str:
    """
    将对话消息格式化为模型输入
    支持多种对话模板格式
    """
    # 构建完整的消息列表
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    
    # 尝试使用 tokenizer 的 chat 模板
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt
        except Exception as e:
            print(f"⚠️ apply_chat_template 失败: {e}")
    
    # 回退到手动构建 (ChatML 格式)
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    
    return prompt


@torch.inference_mode()
def generate_response(
    messages: List[Dict],
    max_new_tokens: int = 2048,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
) -> str:
    """
    生成模型响应（非流式）
    """
    if model is None or tokenizer is None:
        return "❌ 模型未加载，请先加载模型"
    
    # 清理 GPU 缓存
    torch.cuda.empty_cache()
    
    # 格式化输入
    prompt = format_messages(messages)
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096
    ).to(model.device)
    
    # 生成
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # 解码（只取新生成的部分）
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    return response.strip()


@torch.inference_mode()
def generate_response_stream(
    messages: List[Dict],
    max_new_tokens: int = 2048,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
) -> Generator[str, None, None]:
    """
    生成模型响应（流式）
    """
    if model is None or tokenizer is None:
        yield "❌ 模型未加载，请先加载模型"
        return
    
    from transformers import TextIteratorStreamer
    
    # 清理 GPU 缓存
    torch.cuda.empty_cache()
    
    # 格式化输入
    prompt = format_messages(messages)
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096
    ).to(model.device)
    
    # 创建流式输出器
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    
    # 在后台线程中生成
    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # 流式输出
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield generated_text
    
    thread.join()


# ==================== 历史记录管理 ====================
def get_history_files() -> List[Dict]:
    """获取所有历史记录文件"""
    files = []
    for f in sorted(HISTORY_DIR.glob("*.json"), key=os.path.getmtime, reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                files.append({
                    "path": str(f),
                    "name": data.get("title", f.stem),
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                    "messages": data.get("messages", [])
                })
        except:
            continue
    return files


def save_history(title: str, chat_history_gradio: List[Dict[str, str]]):
    """保存历史对话"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = re.sub(r'[\\/*?:"<>|]', "", title)[:50]
    filename = f"{timestamp}_{safe_title}.json"
    filepath = HISTORY_DIR / filename
    
    # 使用 ChatHistory 转换格式
    history = ChatHistory.from_gradio_format(chat_history_gradio)
    
    data = {
        "title": title,
        "created": datetime.now().isoformat(),
        "messages": history.to_json_serializable()
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return str(filepath)


def load_history_file(filepath: str) -> Tuple[str, List[Dict[str, str]]]:
    """加载历史记录"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    title = data.get("title", "未命名")
    messages = data.get("messages", [])
    
    # 使用 ChatHistory 转换为 Gradio 格式
    history = ChatHistory.from_json(messages)
    return title, history.to_gradio_format()


def export_novel(chat_history_gradio: List[Dict[str, str]], title: str) -> str:
    """导出小说为纯文本"""
    lines = [f"《{title}》\n", "="*50 + "\n\n"]
    
    for msg in chat_history_gradio:
        if msg.get("role") == "assistant":  # 只导出 AI 生成的内容
            content = msg.get("content", "")
            if content:
                lines.append(content)
                lines.append("\n\n")
    
    return "".join(lines)


# ==================== Gradio 界面 ====================
def create_gradio_interface():
    """创建 Gradio 界面"""
    
    # 自定义 CSS
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=Ma+Shan+Zheng&display=swap');
    
    .gradio-container {
        max-width: 1200px !important;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
    }
    
    .main-title {
        font-family: 'Ma Shan Zheng', cursive !important;
        font-size: 3rem !important;
        text-align: center !important;
        background: linear-gradient(90deg, #e94560, #f39c12, #e94560) !important;
        background-size: 200% auto !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        animation: shine 3s linear infinite !important;
        margin-bottom: 0.5rem !important;
    }
    
    @keyframes shine {
        to { background-position: 200% center; }
    }
    
    .subtitle {
        font-family: 'Noto Serif SC', serif !important;
        text-align: center !important;
        color: #a0a0a0 !important;
        font-size: 1.1rem !important;
        margin-bottom: 2rem !important;
    }
    
    #chatbot {
        font-family: 'Noto Serif SC', serif !important;
        min-height: 500px !important;
    }
    
    #chatbot .message {
        font-size: 1.1rem !important;
        line-height: 1.8 !important;
    }
    
    .user-message {
        background: rgba(233, 69, 96, 0.1) !important;
        border-left: 4px solid #f39c12 !important;
    }
    
    .bot-message {
        background: rgba(255, 255, 255, 0.05) !important;
        border-left: 4px solid #e94560 !important;
    }
    
    .generate-btn {
        background: linear-gradient(135deg, #e94560, #f39c12) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
    }
    
    .generate-btn:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4) !important;
    }
    
    .parameter-slider label {
        color: #e0e0e0 !important;
    }
    """
    
    with gr.Blocks(
        title="📚 AI 小说生成器",
    ) as demo:
        
        # 状态变量
        current_title = gr.State("")
        
        # 标题
        gr.HTML("""
        <h1 class="main-title">📚 AI 小说生成器</h1>
        <p class="subtitle">基于 AWQ 量化模型的智能小说创作助手 | 使用 Transformers 直接部署</p>
        """)
        
        with gr.Row():
            # 左侧：主聊天区域
            with gr.Column(scale=3):
                # 小说标题
                novel_title = gr.Textbox(
                    label="📖 小说标题",
                    placeholder="输入你的小说标题...",
                    lines=1
                )
                
                # 聊天界面
                chatbot = gr.Chatbot(
                    label="创作区",
                    elem_id="chatbot",
                    height=500,
                )
                
                # 输入区域
                with gr.Row():
                    user_input = gr.Textbox(
                        label="创作指令",
                        placeholder="描述你想要的故事情节、人物、场景，或者输入'继续'让 AI 续写...",
                        lines=3,
                        scale=5
                    )
                
                with gr.Row():
                    generate_btn = gr.Button(
                        "✍️ 生成", 
                        variant="primary",
                        elem_classes="generate-btn",
                        scale=2
                    )
                    continue_btn = gr.Button(
                        "⏩ 续写",
                        scale=1
                    )
                    clear_btn = gr.Button(
                        "🗑️ 清空",
                        scale=1
                    )
            
            # 右侧：参数和历史
            with gr.Column(scale=1):
                # 模型状态
                with gr.Accordion("🔧 模型状态", open=True):
                    model_status = gr.Textbox(
                        label="状态",
                        value="等待加载模型...",
                        interactive=False,
                        lines=2
                    )
                    model_path_input = gr.Textbox(
                        label="模型路径",
                        value=MODEL_PATH,
                        placeholder="输入 AWQ 模型路径"
                    )
                    load_model_btn = gr.Button("📥 加载模型", variant="secondary")
                
                # 生成参数
                with gr.Accordion("⚙️ 生成参数", open=True):
                    max_tokens = gr.Slider(
                        label="最大生成长度",
                        minimum=256,
                        maximum=4096,
                        value=2048,
                        step=128,
                        elem_classes="parameter-slider"
                    )
                    temperature = gr.Slider(
                        label="创意程度 (Temperature)",
                        minimum=0.1,
                        maximum=1.5,
                        value=0.8,
                        step=0.1
                    )
                    top_p = gr.Slider(
                        label="Top-P",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.95,
                        step=0.05
                    )
                    repetition_penalty = gr.Slider(
                        label="重复惩罚",
                        minimum=1.0,
                        maximum=2.0,
                        value=1.1,
                        step=0.1
                    )
                
                # 历史记录
                with gr.Accordion("📚 历史作品", open=False):
                    history_dropdown = gr.Dropdown(
                        label="选择历史记录",
                        choices=[],
                        interactive=True
                    )
                    refresh_history_btn = gr.Button("🔄 刷新")
                    load_history_btn = gr.Button("📂 加载")
                
                # 导出
                with gr.Accordion("📥 导出", open=False):
                    save_btn = gr.Button("💾 保存当前对话")
                    export_btn = gr.Button("📄 导出为TXT")
                    export_output = gr.File(label="下载文件")
        
        # ==================== 事件处理 ====================
        
        def on_load_model(path):
            """加载模型"""
            try:
                load_awq_model(path)
                return f"✅ 模型加载成功!\n设备: {DEVICE}"
            except Exception as e:
                return f"❌ 模型加载失败: {str(e)}"
        
        def on_generate(user_msg, chat_history_gradio, title, max_tok, temp, top_p_val, rep_penalty):
            """生成响应"""
            import sys
            print(f"\n📝 [on_generate] 收到请求:", flush=True)
            print(f"   - 用户消息: {user_msg[:50]}..." if len(user_msg) > 50 else f"   - 用户消息: {user_msg}", flush=True)
            print(f"   - 当前历史长度: {len(chat_history_gradio)}", flush=True)
            print(f"   - 模型状态: {'已加载' if model is not None else '未加载'}", flush=True)
            sys.stdout.flush()
            
            if not user_msg.strip():
                print("   ⚠️ 用户消息为空，跳过生成")
                return chat_history_gradio, ""
            
            if model is None:
                print("   ❌ 模型未加载")
                chat_history_gradio.append({"role": "user", "content": user_msg})
                chat_history_gradio.append({"role": "assistant", "content": "❌ 请先加载模型"})
                return chat_history_gradio, ""
            
            # 使用 ChatHistory 管理消息
            history = ChatHistory.from_gradio_format(chat_history_gradio)
            
            # 添加用户消息
            history.add_user_message(user_msg)
            
            # 获取用于模型的消息格式
            messages = history.get_messages_for_model()
            print(f"   - 发送给模型的消息数: {len(messages)}")
            
            # 添加空的 AI 消息占位
            history.add_ai_message("")
            
            # 流式生成
            print("   🚀 开始流式生成...")
            token_count = 0
            for response in generate_response_stream(
                messages,
                max_new_tokens=max_tok,
                temperature=temp,
                top_p=top_p_val,
                repetition_penalty=rep_penalty
            ):
                token_count += 1
                history.update_last_ai_message(response)
                yield history.to_gradio_format(), ""
            
            print(f"   ✅ 生成完成，共 {token_count} 次更新")
        
        def on_continue(chat_history_gradio, title, max_tok, temp, top_p_val, rep_penalty):
            """续写"""
            import sys
            print(f"\n⏩ [on_continue] 续写请求", flush=True)
            print(f"   - 当前历史长度: {len(chat_history_gradio)}", flush=True)
            sys.stdout.flush()
            
            if not chat_history_gradio:
                print("   ⚠️ 历史为空，无法续写")
                # 返回提示消息
                yield [{"role": "assistant", "content": "⚠️ 请先输入内容开始创作，再点击续写"}], ""
                return
            
            continue_msg = "请继续创作，延续上文的故事情节。"
            
            # 使用生成器
            for result in on_generate(
                continue_msg, chat_history_gradio, title, 
                max_tok, temp, top_p_val, rep_penalty
            ):
                yield result
        
        def on_clear():
            """清空对话"""
            return [], ""
        
        def on_refresh_history():
            """刷新历史记录列表"""
            files = get_history_files()
            choices = [(f"{f['name']} ({f['modified']})", f['path']) for f in files]
            return gr.Dropdown(choices=choices)
        
        def on_load_history(selected):
            """加载选中的历史记录"""
            if not selected:
                return [], ""
            title, messages = load_history_file(selected)
            return messages, title
        
        def on_save(chat_history, title):
            """保存对话"""
            if not chat_history:
                return "没有可保存的对话"
            if not title:
                title = "未命名小说"
            filepath = save_history(title, chat_history)
            return f"✅ 已保存到: {filepath}"
        
        def on_export(chat_history, title):
            """导出为TXT"""
            if not chat_history:
                return None
            if not title:
                title = "未命名小说"
            
            content = export_novel(chat_history, title)
            
            # 保存临时文件
            export_path = HISTORY_DIR / f"{title}_export.txt"
            with open(export_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            return str(export_path)
        
        # 绑定事件
        load_model_btn.click(
            on_load_model,
            inputs=[model_path_input],
            outputs=[model_status]
        )
        
        generate_btn.click(
            on_generate,
            inputs=[user_input, chatbot, novel_title, max_tokens, temperature, top_p, repetition_penalty],
            outputs=[chatbot, user_input]
        )
        
        # Enter 键触发生成
        user_input.submit(
            on_generate,
            inputs=[user_input, chatbot, novel_title, max_tokens, temperature, top_p, repetition_penalty],
            outputs=[chatbot, user_input]
        )
        
        continue_btn.click(
            on_continue,
            inputs=[chatbot, novel_title, max_tokens, temperature, top_p, repetition_penalty],
            outputs=[chatbot, user_input]
        )
        
        clear_btn.click(
            on_clear,
            outputs=[chatbot, user_input]
        )
        
        refresh_history_btn.click(
            on_refresh_history,
            outputs=[history_dropdown]
        )
        
        load_history_btn.click(
            on_load_history,
            inputs=[history_dropdown],
            outputs=[chatbot, novel_title]
        )
        
        save_btn.click(
            on_save,
            inputs=[chatbot, novel_title],
            outputs=[model_status]
        )
        
        export_btn.click(
            on_export,
            inputs=[chatbot, novel_title],
            outputs=[export_output]
        )
        
        # 启动时刷新历史记录
        demo.load(on_refresh_history, outputs=[history_dropdown])
    
    return demo


# ==================== 主程序 ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="小说生成器 Gradio 前端")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="AWQ 模型路径")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=7860, help="服务器端口")
    parser.add_argument("--share", action="store_true", help="创建公共链接")
    parser.add_argument("--auto_load", action="store_true", help="启动时自动加载模型")
    args = parser.parse_args()
    
    # 更新模型路径
    MODEL_PATH = args.model_path
    
    # 自动加载模型
    if args.auto_load:
        print("🚀 自动加载模型...")
        try:
            load_awq_model(args.model_path)
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}")
            print("💡 可以在界面中手动加载模型")
    
    # 创建并启动 Gradio 界面
    demo = create_gradio_interface()
    
    print(f"\n{'='*60}")
    print(f"🚀 启动 Gradio 服务器")
    print(f"{'='*60}")
    print(f"   地址: http://{args.host}:{args.port}")
    if args.share:
        print(f"   公共链接: 生成中...")
    print(f"{'='*60}\n")
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True
    )
