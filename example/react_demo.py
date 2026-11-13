"""演示 ReAct：多轮思考与行动，并把完整过程写入文件。"""

import os

from dotenv import load_dotenv

from agents.react import ReActAgent
from utils.core import LLMChat, ToolExecutor
from utils.tools import calculator, query_database


def main() -> None:
    load_dotenv()
    llm = LLMChat(
        model=os.getenv("MODEL") or "",
        api_key=os.getenv("API_KEY") or "",
        base_url=os.getenv("BASE_URL") or "",
    )
    tools = ToolExecutor()
    tools.register(
        "query_database",
        "查询模拟数据库，例如查询星尘兽或月影兽的体重（单位：吨）",
        query_database,
    )
    tools.register("calculator", "计算算术表达式，例如 (12 + 8) * 3", calculator)
    agent = ReActAgent(llm, tools)
    agent.demo(
        "星尘兽的体重是月影兽的多少倍？",
        verbose="full",
        trace_file="temp/react_demo_trace.txt",
    )


if __name__ == "__main__":
    main()
