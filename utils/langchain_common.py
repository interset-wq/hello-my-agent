"""LangChain 模型构建与调用的公共函数。"""

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage


def build_model(model: str | None = None) -> BaseChatModel:
    """创建 OpenAI 兼容模型。"""
    return init_chat_model(
        f"openai:{model or os.getenv('MODEL')}",
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY", "ollama"),
    )


def ask(model: BaseChatModel, prompt: str) -> str:
    """发送单条用户消息并返回文本。"""
    return model.invoke([HumanMessage(prompt)]).text
