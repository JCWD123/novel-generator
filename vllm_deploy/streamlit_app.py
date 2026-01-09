#!/usr/bin/env python3
"""
小说生成器 Streamlit 前端

基于 vLLM 服务 + LangChain 历史管理
支持流式生成、历史记录管理、导出功能
"""
import streamlit as st
from pathlib import Path
from datetime import datetime

from config import (
    VLLM_API_URL,
    MODEL_NAME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    print_config
)
from langchain_history import (
    NovelChatHistory,
    HistoryManager,
    list_all_histories,
    create_new_history,
    load_history
)
from vllm_client import (
    VLLMClient,
    get_vllm_client,
    check_service
)

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="📚 AI 小说生成器 - vLLM",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义样式 ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=ZCOOL+XiaoWei&display=swap');

/* 主题色彩变量 */
:root {
    --primary-color: #c0392b;
    --secondary-color: #8e44ad;
    --accent-color: #d4af37;
    --bg-dark: #0d1117;
    --bg-card: rgba(22, 27, 34, 0.95);
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
}

/* 背景渐变 */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 40%, #21262d 100%);
}

/* 主标题 */
.main-title {
    font-family: 'ZCOOL XiaoWei', serif;
    font-size: 3.2rem;
    text-align: center;
    background: linear-gradient(135deg, var(--primary-color), var(--accent-color), var(--secondary-color));
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient-shift 5s ease infinite;
    margin-bottom: 0.3rem;
    letter-spacing: 0.1em;
}

@keyframes gradient-shift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

/* 副标题 */
.subtitle {
    font-family: 'Noto Serif SC', serif;
    text-align: center;
    color: var(--text-secondary);
    font-size: 1rem;
    margin-bottom: 2rem;
    letter-spacing: 0.05em;
}

/* 小说内容卡片 */
.novel-content {
    font-family: 'Noto Serif SC', serif;
    font-size: 1.1rem;
    line-height: 2.2;
    color: var(--text-primary);
    background: var(--bg-card);
    padding: 1.8rem 2rem;
    border-radius: 12px;
    border-left: 4px solid var(--primary-color);
    margin: 1rem 0;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(10px);
}

/* 用户输入卡片 */
.user-input {
    font-family: 'Noto Serif SC', serif;
    background: rgba(192, 57, 43, 0.1);
    padding: 1rem 1.5rem;
    border-radius: 10px;
    border-left: 4px solid var(--accent-color);
    margin: 0.8rem 0;
    color: var(--text-primary);
    font-size: 1rem;
}

/* 统计卡片 */
.stats-card {
    background: var(--bg-card);
    padding: 1.2rem;
    border-radius: 12px;
    text-align: center;
    border: 1px solid rgba(192, 57, 43, 0.3);
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
}

.stats-number {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--primary-color);
    font-family: 'Noto Serif SC', serif;
}

.stats-label {
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin-top: 0.3rem;
}

/* 服务状态指示器 */
.status-online {
    color: #2ecc71;
    font-weight: 600;
}

.status-offline {
    color: #e74c3c;
    font-weight: 600;
}

/* 侧边栏美化 */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
}

section[data-testid="stSidebar"] .stButton button {
    width: 100%;
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
}

section[data-testid="stSidebar"] .stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(192, 57, 43, 0.4);
}

/* 输入框美化 */
.stTextArea textarea {
    font-family: 'Noto Serif SC', serif !important;
    background: var(--bg-card) !important;
    border: 1px solid rgba(192, 57, 43, 0.3) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-size: 1rem !important;
}

.stTextArea textarea:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 15px rgba(192, 57, 43, 0.2) !important;
}

/* 隐藏默认元素 */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* 分隔线 */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(192, 57, 43, 0.5), transparent);
    margin: 1.5rem 0;
}

/* 历史记录项 */
.history-item {
    background: var(--bg-card);
    padding: 0.8rem 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 3px solid var(--secondary-color);
    transition: all 0.2s ease;
}

.history-item:hover {
    transform: translateX(5px);
    border-left-color: var(--primary-color);
}
</style>
""", unsafe_allow_html=True)


# ==================== 初始化 Session State ====================
def init_session_state():
    """初始化会话状态"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None  # NovelChatHistory 实例
    if "novel_title" not in st.session_state:
        st.session_state.novel_title = ""
    if "vllm_client" not in st.session_state:
        st.session_state.vllm_client = None
    if "service_status" not in st.session_state:
        st.session_state.service_status = False

init_session_state()


# ==================== 辅助函数 ====================
def check_vllm_service():
    """检查 vLLM 服务状态"""
    try:
        client = get_vllm_client()
        st.session_state.vllm_client = client
        st.session_state.service_status = client.check_health()
    except Exception as e:
        st.session_state.service_status = False
    return st.session_state.service_status


def create_new_novel(title: str):
    """创建新小说"""
    st.session_state.chat_history = create_new_history(title)
    st.session_state.novel_title = title


def load_novel(filepath: Path):
    """加载小说"""
    history = load_history(filepath)
    st.session_state.chat_history = history
    st.session_state.novel_title = history.title


def generate_response(user_input: str, max_tokens: int, temperature: float, top_p: float):
    """生成 AI 响应"""
    if st.session_state.chat_history is None:
        create_new_novel(user_input[:20] if len(user_input) > 20 else user_input)
    
    history = st.session_state.chat_history
    
    # 添加用户消息
    history.add_user_message(user_input)
    
    # 获取 LLM 格式的消息
    messages = history.get_messages_for_llm()
    
    # 流式生成
    client = get_vllm_client()
    
    # 创建占位 AI 消息
    history.add_ai_message("")
    
    # 流式生成并更新
    response_placeholder = st.empty()
    full_response = ""
    
    for partial_response in client.generate_stream(
        messages=[msg for msg in messages if msg["role"] != "system"],  # 系统消息已在客户端处理
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    ):
        full_response = partial_response
        history.update_last_ai_message(full_response)
        response_placeholder.markdown(
            f'<div class="novel-content">{full_response}</div>',
            unsafe_allow_html=True
        )
    
    # 保存历史
    history.save()
    
    return full_response


# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("### 📖 小说管理")
    
    # 服务状态
    with st.expander("🔌 服务状态", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**API**: `{VLLM_API_URL}`")
            st.markdown(f"**模型**: `{MODEL_NAME}`")
        with col2:
            if st.button("🔄", help="刷新状态"):
                check_vllm_service()
        
        if st.session_state.service_status:
            st.markdown('<span class="status-online">● 服务在线</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-offline">● 服务离线</span>', unsafe_allow_html=True)
            st.warning("请确保 vLLM 服务已启动")
    
    st.markdown("---")
    
    # 新建小说
    with st.expander("✨ 新建小说", expanded=False):
        new_title = st.text_input("小说标题", placeholder="请输入小说标题...", key="new_title")
        if st.button("🚀 开始创作", use_container_width=True, key="create_btn"):
            if new_title:
                create_new_novel(new_title)
                st.success(f"已创建: {new_title}")
                st.rerun()
            else:
                st.warning("请输入小说标题")
    
    st.markdown("---")
    
    # 历史记录
    st.markdown("### 📚 历史作品")
    
    history_files = list_all_histories()
    
    if history_files:
        for i, f in enumerate(history_files[:10]):  # 只显示最近 10 个
            col1, col2 = st.columns([4, 1])
            with col1:
                display_title = f['title'][:18] + "..." if len(f['title']) > 18 else f['title']
                if st.button(
                    f"📄 {display_title}\n_{f['modified']}_",
                    key=f"load_{i}",
                    use_container_width=True
                ):
                    load_novel(f['path'])
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{i}", help="删除"):
                    HistoryManager().delete_history(f['path'])
                    st.rerun()
    else:
        st.info("暂无历史作品")
    
    st.markdown("---")
    
    # 生成参数
    st.markdown("### ⚙️ 生成参数")
    max_tokens = st.slider(
        "最大生成长度",
        min_value=256,
        max_value=4096,
        value=DEFAULT_MAX_TOKENS,
        step=128
    )
    temperature = st.slider(
        "创意程度",
        min_value=0.1,
        max_value=1.5,
        value=DEFAULT_TEMPERATURE,
        step=0.1
    )
    top_p = st.slider(
        "Top-P",
        min_value=0.1,
        max_value=1.0,
        value=DEFAULT_TOP_P,
        step=0.05
    )
    
    st.markdown("---")
    
    # 统计信息
    if st.session_state.chat_history:
        st.markdown("### 📊 当前统计")
        history = st.session_state.chat_history
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{len(history.messages) // 2}</div>
                <div class="stats-label">对话轮次</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{history.get_ai_chars()}</div>
                <div class="stats-label">AI 字数</div>
            </div>
            """, unsafe_allow_html=True)


# ==================== 主内容区 ====================
st.markdown('<h1 class="main-title">📚 AI 小说生成器</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">基于 vLLM + LangChain 的智能小说创作助手 | Docker 部署版</p>',
    unsafe_allow_html=True
)

# 当前小说标题
if st.session_state.novel_title:
    st.markdown(f"### 📖 当前作品: {st.session_state.novel_title}")

# 显示对话历史
if st.session_state.chat_history:
    for msg in st.session_state.chat_history.to_streamlit_format():
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-input">💭 <strong>创作指令:</strong> {msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="novel-content">{msg["content"]}</div>',
                unsafe_allow_html=True
            )

st.markdown("---")

# 用户输入区
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_area(
        "创作指令",
        placeholder="描述你想要的故事情节、人物、场景，或者输入'继续'让 AI 续写...",
        height=100,
        label_visibility="collapsed",
        key="user_input"
    )

with col2:
    st.write("")  # 占位
    generate_btn = st.button("✍️ 生成", use_container_width=True, type="primary")
    continue_btn = st.button("⏩ 续写", use_container_width=True)

# 处理生成请求
if generate_btn and user_input:
    if not st.session_state.service_status:
        # 重新检查服务
        check_vllm_service()
        if not st.session_state.service_status:
            st.error("❌ vLLM 服务未连接，请先启动服务")
            st.stop()
    
    with st.spinner("🎭 AI 正在创作中..."):
        response = generate_response(user_input, max_tokens, temperature, top_p)
    
    if response:
        st.rerun()

elif continue_btn:
    if st.session_state.chat_history and len(st.session_state.chat_history.messages) > 0:
        if not st.session_state.service_status:
            check_vllm_service()
            if not st.session_state.service_status:
                st.error("❌ vLLM 服务未连接")
                st.stop()
        
        with st.spinner("🎭 AI 正在续写中..."):
            response = generate_response(
                "请继续创作，延续上文的故事情节。",
                max_tokens, temperature, top_p
            )
        
        if response:
            st.rerun()
    else:
        st.warning("请先开始创作一个故事")

# 导出功能
if st.session_state.chat_history and len(st.session_state.chat_history.messages) > 0:
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # 导出小说
        novel_text = st.session_state.chat_history.export_novel()
        st.download_button(
            "📥 导出小说 (TXT)",
            novel_text,
            file_name=f"{st.session_state.novel_title or 'novel'}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col2:
        # 导出对话
        dialogue_text = st.session_state.chat_history.export_dialogue()
        st.download_button(
            "📥 导出对话记录",
            dialogue_text,
            file_name=f"{st.session_state.novel_title or 'dialogue'}_对话.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col3:
        if st.button("🗑️ 清空当前对话", use_container_width=True):
            st.session_state.chat_history = None
            st.session_state.novel_title = ""
            st.rerun()

# 初始化时检查服务
if not st.session_state.service_status:
    check_vllm_service()

