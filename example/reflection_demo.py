"""演示 Reflection：生成初稿、自我批评、改进答案，并把完整过程写入文件。"""

import os

from dotenv import load_dotenv

from agents.reflection import ReflectionAgent
from utils.core import LLMChat


def main() -> None:
    load_dotenv()
    llm = LLMChat(
        model=os.getenv("MODEL") or "",
        api_key=os.getenv("API_KEY") or "",
        base_url=os.getenv("BASE_URL") or "",
    )
    agent = ReflectionAgent(llm)
    agent.demo(
        "用一句话解释什么是递归。",
        verbose="full",
        trace_file="temp/reflection_demo_trace.txt",
    )


if __name__ == "__main__":
    main()
