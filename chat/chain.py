import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# langchain_anthropicが使うhttpxクライアント生成関数にSSL無効化パッチを当てる
import os
from functools import lru_cache
from typing import Any
import langchain_anthropic.chat_models as _cam
from anthropic._base_client import SyncHttpxClientWrapper as _SyncWrapper
from anthropic._base_client import AsyncHttpxClientWrapper as _AsyncWrapper
from anthropic import NOT_GIVEN as _NOT_GIVEN

@lru_cache
def _patched_sync_client(*, base_url, timeout=_NOT_GIVEN, anthropic_proxy=None):
    kwargs: dict[str, Any] = {
        "base_url": base_url or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com",
        "verify": False,
    }
    if timeout is not _NOT_GIVEN:
        kwargs["timeout"] = timeout
    if anthropic_proxy is not None:
        kwargs["proxy"] = anthropic_proxy
    return _SyncWrapper(**kwargs)

@lru_cache
def _patched_async_client(*, base_url, timeout=_NOT_GIVEN, anthropic_proxy=None):
    kwargs: dict[str, Any] = {
        "base_url": base_url or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com",
        "verify": False,
    }
    if timeout is not _NOT_GIVEN:
        kwargs["timeout"] = timeout
    if anthropic_proxy is not None:
        kwargs["proxy"] = anthropic_proxy
    return _AsyncWrapper(**kwargs)

_cam._get_default_httpx_client = _patched_sync_client
_cam._get_default_async_httpx_client = _patched_async_client

from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LLM_MODEL, LLM_TEMPERATURE, SYSTEM_PROMPT, get_anthropic_api_key
from rag.retriever import get_tools


def build_agent():
    api_key = get_anthropic_api_key()
    llm = ChatAnthropic(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        anthropic_api_key=api_key,
    )
    tools = get_tools()
    return create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)


def run_agent(agent, user_input: str, chat_history: list) -> dict:
    history_messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            history_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history_messages.append(AIMessage(content=msg["content"]))

    history_messages.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": history_messages})

    answer = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            answer = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    retrieved_docs = []
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            retrieved_docs.append({
                "tool": msg.name,
                "query": "",
                "result": msg.content,
            })

    return {
        "answer": answer,
        "retrieved_docs": retrieved_docs,
    }
