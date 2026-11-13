"""演示 Plan-and-Solve：先制定计划，再逐步执行，并把完整过程写入文件。"""

import os

from dotenv import load_dotenv

from agents.plan_and_solve import PlanAndSolveAgent
from utils.core import LLMChat


def main() -> None:
    load_dotenv()
    llm = LLMChat(
        model=os.getenv("MODEL") or "",
        api_key=os.getenv("API_KEY") or "",
        base_url=os.getenv("BASE_URL") or "",
    )
    agent = PlanAndSolveAgent(llm)
    tasks = [
        (
            "一个班有 40 名学生，60% 是女生，女生中 25% 戴眼镜。戴眼镜的女生有多少人？",
            "temp/pas_demo_trace.txt",
        ),
        (
            "笼子里有若干只鸡和兔，从上面数有 35 个头，从下面数有 94 只脚。问鸡和兔各有多少只？",
            "temp/pas_demo_trace_chickens_rabbits.txt",
        ),
    ]
    for task, trace_file in tasks:
        agent.demo(task, verbose="full", trace_file=trace_file)


if __name__ == "__main__":
    main()
