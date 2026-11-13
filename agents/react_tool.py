"""ReAct：基于 Responses API 内置工具调用的实现。"""

import inspect
import json
from typing import Any

from utils.core import LLMChat, ToolExecutor

_TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _tool_specs(tools: ToolExecutor) -> list[dict[str, Any]]:
    """根据已注册函数生成 Responses API 的工具定义。"""
    specs = []
    for name, info in tools.tools.items():
        params = inspect.signature(info["func"]).parameters
        specs.append({
            "type": "function",
            "name": name,
            "description": info["desc"],
            "parameters": {
                "type": "object",
                "properties": {
                    p: {"type": _TYPE_MAP.get(v.annotation, "string")}
                    for p, v in params.items()
                },
                "required": [
                    p for p, v in params.items()
                    if v.default is inspect.Parameter.empty
                ],
            },
        })
    return specs


def _calls(response: Any) -> list[Any]:
    return [item for item in response.output if item.type == "function_call"]


class ReActAgent:
    def __init__(self, llm: LLMChat, tools: ToolExecutor, max_steps: int = 5) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.tool_specs = _tool_specs(tools)

    def _create(self, data: Any) -> Any:
        return self.llm.client.responses.create(
            model=self.llm.model,
            input=data,
            tools=self.tool_specs,
        )

    def run(self, question: str) -> str:
        conversation: list[Any] = [{"role": "user", "content": question}]
        response = self._create(conversation)
        for _ in range(self.max_steps):
            calls = _calls(response)
            if not calls:
                return response.output_text

            conversation += response.output
            for call in calls:
                try:
                    args = json.loads(call.arguments or "{}")
                    result = str(self.tools.run(call.name, **args))
                except Exception as exc:
                    result = f"错误: {exc}"
                conversation.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": result,
                })
            response = self._create(conversation)

        if _calls(response):
            return "已达到最大步数，未能得到最终答案。"
        return response.output_text
