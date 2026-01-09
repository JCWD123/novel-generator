"""
LangChain 历史对话管理模块

使用 LangChain 的消息类型和历史记录管理
支持会话持久化到文件系统
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    messages_to_dict,
    messages_from_dict
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory

from config import HISTORY_DIR, NOVEL_SYSTEM_PROMPT


class NovelChatHistory(BaseChatMessageHistory):
    """
    小说创作专用的聊天历史管理器
    
    继承自 LangChain 的 BaseChatMessageHistory
    支持消息持久化、加载、导出等功能
    """
    
    def __init__(self, session_id: str = None, title: str = "未命名小说"):
        """
        初始化历史管理器
        
        Args:
            session_id: 会话 ID（用于文件名）
            title: 小说标题
        """
        self.title = title
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._messages: List[BaseMessage] = []
        self._metadata: Dict[str, Any] = {
            "title": title,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # 文件路径
        safe_title = re.sub(r'[\\/*?:"<>|]', "", title)[:30]
        self._file_path = HISTORY_DIR / f"{self.session_id}_{safe_title}.json"
    
    @property
    def messages(self) -> List[BaseMessage]:
        """获取所有消息"""
        return self._messages
    
    def add_message(self, message: BaseMessage) -> None:
        """添加消息"""
        self._messages.append(message)
        self._metadata["updated_at"] = datetime.now().isoformat()
    
    def add_user_message(self, content: str) -> None:
        """添加用户消息"""
        self.add_message(HumanMessage(content=content))
    
    def add_ai_message(self, content: str) -> None:
        """添加 AI 消息"""
        self.add_message(AIMessage(content=content))
    
    def update_last_ai_message(self, content: str) -> None:
        """更新最后一条 AI 消息（用于流式生成）"""
        if self._messages and isinstance(self._messages[-1], AIMessage):
            self._messages[-1] = AIMessage(content=content)
    
    def clear(self) -> None:
        """清空消息历史"""
        self._messages = []
        self._metadata["updated_at"] = datetime.now().isoformat()
    
    def get_messages_for_llm(self, include_system: bool = True) -> List[Dict[str, str]]:
        """
        获取用于 LLM API 调用的消息格式
        
        Args:
            include_system: 是否包含系统提示词
            
        Returns:
            OpenAI 格式的消息列表
        """
        messages = []
        
        if include_system:
            messages.append({
                "role": "system",
                "content": NOVEL_SYSTEM_PROMPT
            })
        
        for msg in self._messages:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                messages.append({"role": "system", "content": msg.content})
        
        return messages
    
    def to_streamlit_format(self) -> List[Dict[str, str]]:
        """
        转换为 Streamlit chat_message 格式
        
        Returns:
            [{"role": "user/assistant", "content": "..."}, ...]
        """
        result = []
        for msg in self._messages:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
        return result
    
    @classmethod
    def from_streamlit_format(
        cls, 
        messages: List[Dict[str, str]], 
        session_id: str = None,
        title: str = "未命名小说"
    ) -> "NovelChatHistory":
        """
        从 Streamlit 格式创建历史
        
        Args:
            messages: Streamlit 格式的消息列表
            session_id: 会话 ID
            title: 小说标题
        """
        history = cls(session_id=session_id, title=title)
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                history.add_user_message(content)
            elif role == "assistant":
                history.add_ai_message(content)
        return history
    
    def save(self, filepath: Path = None) -> str:
        """
        保存历史到文件
        
        Args:
            filepath: 自定义文件路径（可选）
            
        Returns:
            保存的文件路径
        """
        save_path = filepath or self._file_path
        
        data = {
            "session_id": self.session_id,
            "title": self.title,
            "metadata": self._metadata,
            "messages": messages_to_dict(self._messages)
        }
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return str(save_path)
    
    @classmethod
    def load(cls, filepath: Path) -> "NovelChatHistory":
        """
        从文件加载历史
        
        Args:
            filepath: 文件路径
            
        Returns:
            NovelChatHistory 实例
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        history = cls(
            session_id=data.get("session_id"),
            title=data.get("title", "未命名小说")
        )
        history._metadata = data.get("metadata", {})
        history._messages = messages_from_dict(data.get("messages", []))
        history._file_path = filepath
        
        return history
    
    def export_novel(self) -> str:
        """
        导出小说内容（仅 AI 生成的部分）
        
        Returns:
            纯文本格式的小说
        """
        lines = [f"《{self.title}》\n", "=" * 50 + "\n\n"]
        
        for msg in self._messages:
            if isinstance(msg, AIMessage):
                lines.append(msg.content)
                lines.append("\n\n")
        
        return "".join(lines)
    
    def export_dialogue(self) -> str:
        """
        导出完整对话记录
        
        Returns:
            包含用户指令和 AI 生成的完整对话
        """
        lines = [
            f"《{self.title}》- 创作对话记录\n",
            f"创建时间: {self._metadata.get('created_at', 'N/A')}\n",
            "=" * 50 + "\n\n"
        ]
        
        for msg in self._messages:
            if isinstance(msg, HumanMessage):
                lines.append(f"[创作指令]\n{msg.content}\n\n")
            elif isinstance(msg, AIMessage):
                lines.append(f"[AI 创作]\n{msg.content}\n\n{'─' * 30}\n\n")
        
        return "".join(lines)
    
    def get_total_chars(self) -> int:
        """获取总字数"""
        return sum(len(msg.content) for msg in self._messages)
    
    def get_ai_chars(self) -> int:
        """获取 AI 生成的字数"""
        return sum(
            len(msg.content) 
            for msg in self._messages 
            if isinstance(msg, AIMessage)
        )


class HistoryManager:
    """
    历史记录文件管理器
    
    用于列出、加载、删除历史记录文件
    """
    
    def __init__(self, history_dir: Path = HISTORY_DIR):
        self.history_dir = history_dir
        self.history_dir.mkdir(exist_ok=True)
    
    def list_histories(self) -> List[Dict[str, Any]]:
        """
        列出所有历史记录
        
        Returns:
            历史记录信息列表，按修改时间倒序
        """
        files = []
        
        for f in sorted(
            self.history_dir.glob("*.json"), 
            key=os.path.getmtime, 
            reverse=True
        ):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    files.append({
                        "path": f,
                        "session_id": data.get("session_id", f.stem),
                        "title": data.get("title", f.stem),
                        "created_at": data.get("metadata", {}).get("created_at", ""),
                        "updated_at": data.get("metadata", {}).get("updated_at", ""),
                        "modified": datetime.fromtimestamp(
                            f.stat().st_mtime
                        ).strftime("%Y-%m-%d %H:%M"),
                        "size": f.stat().st_size,
                        "message_count": len(data.get("messages", []))
                    })
            except Exception as e:
                print(f"⚠️ 无法读取历史文件 {f}: {e}")
                continue
        
        return files
    
    def load_history(self, filepath: Path) -> NovelChatHistory:
        """加载指定的历史记录"""
        return NovelChatHistory.load(filepath)
    
    def delete_history(self, filepath: Path) -> bool:
        """
        删除历史记录
        
        Args:
            filepath: 文件路径
            
        Returns:
            是否删除成功
        """
        try:
            filepath.unlink()
            return True
        except Exception as e:
            print(f"❌ 删除失败: {e}")
            return False
    
    def create_new_session(self, title: str = "未命名小说") -> NovelChatHistory:
        """
        创建新的会话
        
        Args:
            title: 小说标题
            
        Returns:
            新的 NovelChatHistory 实例
        """
        return NovelChatHistory(title=title)


# ==================== 便捷函数 ====================
_manager = HistoryManager()

def get_history_manager() -> HistoryManager:
    """获取历史管理器单例"""
    return _manager

def list_all_histories() -> List[Dict[str, Any]]:
    """列出所有历史记录"""
    return _manager.list_histories()

def create_new_history(title: str = "未命名小说") -> NovelChatHistory:
    """创建新的历史记录"""
    return _manager.create_new_session(title)

def load_history(filepath: Path) -> NovelChatHistory:
    """加载历史记录"""
    return _manager.load_history(filepath)

