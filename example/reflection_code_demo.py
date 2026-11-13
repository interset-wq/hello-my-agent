"""演示代码 Reflection：生成初稿、评审、改进与收敛，并把完整过程写入文件。"""

import os

from dotenv import load_dotenv

from agents.reflection_code import ReflectionAgent
from utils.core import LLMChat


def main() -> None:
    load_dotenv()
    llm = LLMChat(
        model=os.getenv("MODEL") or "",
        api_key=os.getenv("API_KEY") or "",
        base_url=os.getenv("BASE_URL") or "",
    )
    agent = ReflectionAgent(llm)
    task = "实现一个函数，返回 1 到 n 之间的所有质数。"
    agent.demo(
        task,
        verbose="full",
        trace_file="temp/reflection_code_demo_trace.txt",
    )


if __name__ == "__main__":
    main()
