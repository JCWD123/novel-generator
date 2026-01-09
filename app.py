#!/usr/bin/env python3
"""
小说生成器 Streamlit 前端
支持历史对话记录和连续小说创作
"""

import streamlit as st
import requests
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import re

# ==================== 配置 ====================
VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-r1")
HISTORY_DIR = Path(__file__).parent / "chat_history"
HISTORY_DIR.mkdir(exist_ok=True)

# 小说创作系统提示词
NOVEL_SYSTEM_PROMPT = """你是一位才华横溢的小说作家，擅长创作引人入胜的故事。你的写作特点：

1. **文笔优美**: 善于运用修辞手法，文字富有诗意和画面感
2. **情节紧凑**: 故事发展有张有弛，情节跌宕起伏
3. **人物鲜明**: 角色性格立体，对话生动自然
4. **细节丰富**: 场景描写细腻，能让读者身临其境
5. **连贯性强**: 能够根据之前的情节自然延续故事发展

请根据用户的要求进行小说创作。在续写时，要保持与之前内容的风格一致和情节连贯。
输出格式要求：直接输出小说内容，不要使用markdown格式，不要添加额外的解释。"""


# ==================== 历史记录管理 ====================
def get_history_files() -> List[Dict]:
    """获取所有历史记录文件"""
    files = []
    for f in sorted(HISTORY_DIR.glob("*.txt"), key=os.path.getmtime, reverse=True):
        files.append({
            "path": f,
            "name": f.stem,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "size": f.stat().st_size
        })
    return files


def load_history(filepath: Path) -> List[Dict]:
    """从文件加载历史对话"""
    messages = []
    if filepath.exists():
        content = filepath.read_text(encoding="utf-8")
        # 解析格式: [角色] 内容
        current_role = None
        current_content = []
        
        for line in content.split("\n"):
            if line.startswith("[用户]:"):
                if current_role and current_content:
                    messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
                current_role = "user"
                current_content = [line[5:].strip()]
            elif line.startswith("[AI助手]:"):
                if current_role and current_content:
                    messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
                current_role = "assistant"
                current_content = [line[7:].strip()]
            elif line.startswith("---"):
                continue
            elif current_role:
                current_content.append(line)
        
        # 添加最后一条消息
        if current_role and current_content:
            messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
    
    return messages


def save_history(filepath: Path, messages: List[Dict], title: str = ""):
    """保存历史对话到文件"""
    lines = []
    if title:
        lines.append(f"# 小说标题: {title}")
        lines.append(f"# 创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("---\n")
    
    for msg in messages:
        if msg["role"] == "user":
            lines.append(f"[用户]: {msg['content']}")
        elif msg["role"] == "assistant":
            lines.append(f"[AI助手]: {msg['content']}")
        lines.append("")
    
    filepath.write_text("\n".join(lines), encoding="utf-8")


def create_new_session(title: str) -> Path:
    """创建新的对话会话"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = re.sub(r'[\\/*?:"<>|]', "", title)[:50]  # 清理文件名
    filename = f"{timestamp}_{safe_title}.txt"
    return HISTORY_DIR / filename


# ==================== vLLM API 调用 ====================
def call_vllm_api(messages: List[Dict], max_tokens: int = 2048, temperature: float = 0.8) -> Optional[str]:
    """调用 vLLM OpenAI 兼容 API"""
    try:
        # 添加系统提示词
        full_messages = [{"role": "system", "content": NOVEL_SYSTEM_PROMPT}] + messages
        
        response = requests.post(
            f"{VLLM_API_URL}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": full_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.95,
                "stream": False,
            },
            timeout=300,  # 5分钟超时
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            st.error(f"API 错误: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("无法连接到 vLLM 服务器，请确保服务器已启动")
        return None
    except Exception as e:
        st.error(f"请求错误: {e}")
        return None


def stream_vllm_api(messages: List[Dict], max_tokens: int = 2048, temperature: float = 0.8):
    """流式调用 vLLM API"""
    try:
        full_messages = [{"role": "system", "content": NOVEL_SYSTEM_PROMPT}] + messages
        
        response = requests.post(
            f"{VLLM_API_URL}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": full_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.95,
                "stream": True,
            },
            stream=True,
            timeout=300,
        )
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"]
                            if "content" in delta:
                                yield delta["content"]
                        except json.JSONDecodeError:
                            continue
        else:
            st.error(f"API 错误: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        st.error("无法连接到 vLLM 服务器")
    except Exception as e:
        st.error(f"请求错误: {e}")


# ==================== Streamlit UI ====================
def main():
    # 页面配置
    st.set_page_config(
        page_title="📚 AI 小说生成器",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 自定义 CSS 样式
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=Ma+Shan+Zheng&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    .main-title {
        font-family: 'Ma Shan Zheng', cursive;
        font-size: 3.5rem;
        text-align: center;
        background: linear-gradient(90deg, #e94560, #f39c12, #e94560);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    @keyframes shine {
        to { background-position: 200% center; }
    }
    
    .subtitle {
        font-family: 'Noto Serif SC', serif;
        text-align: center;
        color: #a0a0a0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .novel-content {
        font-family: 'Noto Serif SC', serif;
        font-size: 1.15rem;
        line-height: 2;
        color: #e0e0e0;
        background: rgba(255, 255, 255, 0.05);
        padding: 2rem;
        border-radius: 12px;
        border-left: 4px solid #e94560;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .user-input {
        font-family: 'Noto Serif SC', serif;
        background: rgba(233, 69, 96, 0.1);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #f39c12;
        margin: 0.5rem 0;
        color: #f0f0f0;
    }
    
    .history-item {
        background: rgba(255, 255, 255, 0.08);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    
    .history-item:hover {
        background: rgba(233, 69, 96, 0.15);
        border-color: #e94560;
        transform: translateX(5px);
    }
    
    .sidebar .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #e94560, #f39c12);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .sidebar .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4);
    }
    
    .stTextArea textarea {
        font-family: 'Noto Serif SC', serif;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(233, 69, 96, 0.3);
        border-radius: 12px;
        color: #e0e0e0;
    }
    
    .stTextArea textarea:focus {
        border-color: #e94560;
        box-shadow: 0 0 10px rgba(233, 69, 96, 0.3);
    }
    
    /* 隐藏 Streamlit 默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stats-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(233, 69, 96, 0.2);
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: 700;
        color: #e94560;
    }
    
    .stats-label {
        color: #a0a0a0;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 初始化 session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    if "novel_title" not in st.session_state:
        st.session_state.novel_title = ""
    
    # 侧边栏
    with st.sidebar:
        st.markdown("### 📖 小说管理")
        
        # 新建小说
        with st.expander("✨ 新建小说", expanded=False):
            new_title = st.text_input("小说标题", placeholder="请输入小说标题...")
            if st.button("🚀 开始创作", use_container_width=True):
                if new_title:
                    st.session_state.messages = []
                    st.session_state.novel_title = new_title
                    st.session_state.current_file = create_new_session(new_title)
                    st.success(f"已创建新小说: {new_title}")
                    st.rerun()
                else:
                    st.warning("请输入小说标题")
        
        st.markdown("---")
        
        # 历史记录列表
        st.markdown("### 📚 历史作品")
        history_files = get_history_files()
        
        if history_files:
            for f in history_files:
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(
                        f"📄 {f['name'][:20]}...\n_{f['modified']}_" if len(f['name']) > 20 else f"📄 {f['name']}\n_{f['modified']}_",
                        key=f"load_{f['path']}",
                        use_container_width=True
                    ):
                        st.session_state.messages = load_history(f['path'])
                        st.session_state.current_file = f['path']
                        st.session_state.novel_title = f['name']
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_{f['path']}"):
                        f['path'].unlink()
                        st.rerun()
        else:
            st.info("暂无历史作品")
        
        st.markdown("---")
        
        # 生成参数
        st.markdown("### ⚙️ 生成参数")
        max_tokens = st.slider("最大生成长度", 256, 4096, 2048, 128)
        temperature = st.slider("创意程度", 0.1, 1.5, 0.8, 0.1)
        
        st.markdown("---")
        
        # 统计信息
        if st.session_state.messages:
            total_chars = sum(len(m["content"]) for m in st.session_state.messages)
            st.markdown("### 📊 当前作品统计")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-number">{len(st.session_state.messages)}</div>
                    <div class="stats-label">对话轮次</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-number">{total_chars}</div>
                    <div class="stats-label">总字数</div>
                </div>
                """, unsafe_allow_html=True)
    
    # 主内容区
    st.markdown('<h1 class="main-title">📚 AI 小说生成器</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">基于 DeepSeek-R1 的智能小说创作助手 | 让 AI 成为你的故事伙伴</p>', unsafe_allow_html=True)
    
    # 当前小说标题
    if st.session_state.novel_title:
        st.markdown(f"### 📖 当前作品: {st.session_state.novel_title}")
    
    # 显示对话历史
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-input">💭 <strong>创作指令:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="novel-content">{msg["content"]}</div>', unsafe_allow_html=True)
    
    # 用户输入区
    st.markdown("---")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_area(
            "创作指令",
            placeholder="描述你想要的故事情节、人物、场景，或者输入'继续'让 AI 续写...",
            height=100,
            label_visibility="collapsed"
        )
    
    with col2:
        st.write("")  # 占位
        generate_btn = st.button("✍️ 生成", use_container_width=True, type="primary")
        continue_btn = st.button("⏩ 续写", use_container_width=True)
    
    # 处理生成请求
    if generate_btn and user_input:
        # 如果没有当前文件，创建一个
        if not st.session_state.current_file:
            st.session_state.current_file = create_new_session(user_input[:20])
            st.session_state.novel_title = user_input[:20]
        
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 显示加载状态
        with st.spinner("🎭 AI 正在创作中..."):
            response = call_vllm_api(
                st.session_state.messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
        
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})
            # 保存历史
            save_history(st.session_state.current_file, st.session_state.messages, st.session_state.novel_title)
            st.rerun()
    
    elif continue_btn:
        # 续写模式
        if st.session_state.messages:
            st.session_state.messages.append({"role": "user", "content": "请继续创作，延续上文的故事情节。"})
            
            with st.spinner("🎭 AI 正在续写中..."):
                response = call_vllm_api(
                    st.session_state.messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})
                save_history(st.session_state.current_file, st.session_state.messages, st.session_state.novel_title)
                st.rerun()
        else:
            st.warning("请先开始创作一个故事")
    
    # 导出功能
    if st.session_state.messages:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # 导出为 TXT
            full_text = "\n\n".join([m["content"] for m in st.session_state.messages if m["role"] == "assistant"])
            st.download_button(
                "📥 导出小说 (TXT)",
                full_text,
                file_name=f"{st.session_state.novel_title or 'novel'}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # 导出完整对话
            full_dialogue = ""
            for m in st.session_state.messages:
                if m["role"] == "user":
                    full_dialogue += f"[创作指令]\n{m['content']}\n\n"
                else:
                    full_dialogue += f"[AI 创作]\n{m['content']}\n\n{'='*50}\n\n"
            
            st.download_button(
                "📥 导出对话记录",
                full_dialogue,
                file_name=f"{st.session_state.novel_title or 'dialogue'}_对话.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col3:
            if st.button("🗑️ 清空当前对话", use_container_width=True):
                st.session_state.messages = []
                st.session_state.novel_title = ""
                st.session_state.current_file = None
                st.rerun()


if __name__ == "__main__":
    main()

