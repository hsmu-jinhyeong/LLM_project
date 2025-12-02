"""Phase 1-B: Conversation Memory Management.

This module implements conversation memory using LangChain's memory components
integrated with Streamlit's session state.
"""
import uuid
from typing import List, Dict, Any
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
import logging

logger = logging.getLogger("memory_manager")


def get_session_id() -> str:
    """Get or create a unique session ID for conversation tracking.
    
    Returns:
        Unique session ID string.
    """
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
        logger.info(f"[MEMORY] Created new session ID: {st.session_state['session_id'][:8]}...")
    return st.session_state["session_id"]


def create_memory(window_size: int = 5):
    """Create conversation memory with Streamlit integration.
    
    Args:
        window_size: Number of conversation turns to remember (default: 5).
    
    Returns:
        StreamlitChatMessageHistory instance (simplified).
    """
    # Use StreamlitChatMessageHistory for persistence
    message_history = StreamlitChatMessageHistory(key="chat_messages")
    
    logger.info(f"[MEMORY] Created message history (window size reference: {window_size})")
    return message_history


def get_conversation_history() -> List[BaseMessage]:
    """Get formatted conversation history from session state.
    
    Returns:
        List of BaseMessage objects.
    """
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    
    return st.session_state["chat_messages"]


def add_user_message(content: str) -> None:
    """Add user message to conversation history.
    
    Args:
        content: User message content.
    """
    msg = HumanMessage(content=content)
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    st.session_state["chat_messages"].append(msg)
    logger.info(f"[MEMORY] Added user message: {content[:50]}...")


def add_ai_message(content: str) -> None:
    """Add AI message to conversation history.
    
    Args:
        content: AI message content.
    """
    msg = AIMessage(content=content)
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    st.session_state["chat_messages"].append(msg)
    logger.info(f"[MEMORY] Added AI message: {content[:50]}...")


def clear_memory() -> None:
    """Clear conversation history."""
    if "chat_messages" in st.session_state:
        st.session_state["chat_messages"] = []
    if "session_id" in st.session_state:
        old_id = st.session_state["session_id"][:8]
        st.session_state["session_id"] = str(uuid.uuid4())
        logger.info(f"[MEMORY] Cleared memory, new session: {st.session_state['session_id'][:8]} (old: {old_id})")


def get_memory_summary() -> Dict[str, Any]:
    """Get summary of current memory state.
    
    Returns:
        Dictionary with memory statistics.
    """
    messages = get_conversation_history()
    user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
    
    return {
        "total_messages": len(messages),
        "user_messages": len(user_msgs),
        "ai_messages": len(ai_msgs),
        "session_id": get_session_id()[:8] + "..."
    }


def wrap_chain_with_memory(chain, memory):
    """Wrap a LangChain runnable with conversation memory.
    
    Args:
        chain: LangChain runnable chain.
        memory: StreamlitChatMessageHistory instance.
    
    Returns:
        RunnableWithMessageHistory wrapper.
    """
    def get_session_history(session_id: str):
        """Callback to get session history."""
        return memory
    
    wrapped_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="user_input",
        history_messages_key="chat_history",
    )
    
    logger.info("[MEMORY] Wrapped chain with message history")
    return wrapped_chain


def format_memory_for_context(max_turns: int = 3) -> str:
    """Format recent conversation history as context string.
    
    Args:
        max_turns: Maximum number of recent turns to include.
    
    Returns:
        Formatted context string.
    """
    messages = get_conversation_history()
    
    if not messages:
        return "이전 대화 없음"
    
    # Get last N messages (each turn = user + AI message = 2 messages)
    recent = messages[-(max_turns * 2):]
    
    formatted_parts = []
    for msg in recent:
        role = "사용자" if isinstance(msg, HumanMessage) else "AI"
        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        formatted_parts.append(f"{role}: {content}")
    
    return "\n".join(formatted_parts)
