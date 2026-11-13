"""OpenAI 兼容的最小 LLM 工具集。

包含 timed 装饰器、LLMChat 客户端和 ToolExecutor 执行器。"""


import inspect
import sys
import time
from collections.abc import Callable, Iterator
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast

from openai import OpenAI

__all__ = ["LLMChat", "ToolExecutor", "timed"]

P = ParamSpec("P")
R = TypeVar("R")


def timed(func: Callable[P, R]) -> Callable[P, R]:
    """统计函数耗时；生成器函数统计到迭代结束，结果输出到 stderr。"""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = func(*args, **kwargs)

        if inspect.isgenerator(result):
            def _timed_generator() -> Iterator[Any]:
                try:
                    yield from result
                finally:
                    _report(func.__qualname__, time.perf_counter() - start)
            return cast(R, _timed_generator())

        _report(func.__qualname__, time.perf_counter() - start)
        return result

    return wrapper


def _report(name: str, elapsed: float) -> None:
    print(f"[timing] {name}: {elapsed:.3f}s", file=sys.stderr)


class LLMChat:
    """兼容 OpenAI 接口的最小客户端."""

    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        self.model: str = model
        self.client: OpenAI = OpenAI(api_key=api_key, base_url=base_url)

    def _create(
        self,
        messages: list,
        temperature: float,
        enable_thinking: bool = False,
        stream: bool = False,
    ) -> Any:
        extra: dict[str, Any] = {} if enable_thinking else {"reasoning_effort": "none"}
        return self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature,
            stream=stream, **extra,
        )

    def chat(
        self,
        messages: list,
        temperature: float = 0.7,
        enable_thinking: bool = True,
        include_reasoning: bool = False,
    ) -> str | None | tuple[Any, Any]:
        """通用对话；include_reasoning=True 时返回 (补全, 思考)。"""
        resp = self._create(messages, temperature, enable_thinking)
        message = resp.choices[0].message
        if include_reasoning:
            return message.content, getattr(message, "reasoning", None)
        return message.content

    def invoke(
        self,
        messages: list,
        temperature: float = 0.7,
    ) -> str | None:
        """只返回补全内容。"""
        return cast(
            str | None,
            self.chat(messages, temperature, enable_thinking=False),
        )

    def stream_chat(
        self,
        messages: list,
        temperature: float = 0.7,
        enable_thinking: bool = True,
        include_reasoning: bool = False,
    ) -> Iterator[str | tuple[str, str]]:
        """通用流式对话；include_reasoning=True 时产出 (kind, text)。"""
        for chunk in self._create(messages, temperature, enable_thinking, stream=True):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if include_reasoning:
                reasoning = getattr(delta, "reasoning", None)
                if reasoning:
                    yield "reasoning", reasoning
                if delta.content:
                    yield "content", delta.content
            elif delta.content:
                yield delta.content

    def stream(
        self,
        messages: list,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """只流式返回补全内容。"""
        yield from cast(
            Iterator[str],
            self.stream_chat(messages, temperature, enable_thinking=False),
        )


class ToolExecutor:
    def __init__(self) -> None:
        self.tools: dict[str, dict[str, Any]] = {}

    def register(
        self, name: str, description: str, func: Callable[..., Any]
    ) -> None:
        self.tools[name] = {"desc": description, "func": func}

    def describe_tools(self) -> str:
        return "\n".join(f"- {n}: {info['desc']}" for n, info in self.tools.items())

    def run(self, tool_name: str, /, *args: Any, **kwargs: Any) -> Any:
        tool = self.tools.get(tool_name)
        if tool is None:
            return f"错误: 工具 {tool_name} 不存在"
        return tool["func"](*args, **kwargs)
