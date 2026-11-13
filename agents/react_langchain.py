"""ReAct：基于 LangChain 模型与工具，手动实现工具调用循环。"""

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from utils.langchain_common import build_model
from utils.tools import calculator as _calculator

SYSTEM_PROMPT = "你是一个可以使用工具的助手，需要时调用工具，得到结果后再给出最终答案。"


@tool
def calculator(expression: str) -> str:
    """计算算术表达式，例如 (12 + 8) * 3。"""
    return _calculator(expression)


class ReActAgent:
    def __init__(
        self,
        tools: list[BaseTool],
        model: str | None = None,
        max_steps: int = 5,
    ) -> None:
        self.tools = {item.name: item for item in tools}
        self.model = build_model(model).bind_tools(tools)
        self.max_steps = max_steps

    def run(self, question: str) -> str:
        messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(question)]
        for _ in range(self.max_steps):
            response = self.model.invoke(messages)
            messages.append(response)
            if not response.tool_calls:
                return response.text

            for call in response.tool_calls:
                item = self.tools.get(call["name"])
                if item is None:
                    content = f"错误: 工具 {call['name']} 不存在"
                else:
                    try:
                        content = str(item.invoke(call["args"]))
                    except Exception as exc:
                        content = f"错误: {exc}"
                messages.append(
                    ToolMessage(content=content, tool_call_id=call["id"])
                )
        return "已达到最大步数，未能得到最终答案。"
