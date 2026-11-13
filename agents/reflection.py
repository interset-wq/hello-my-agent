"""Reflection：生成、批评、改进循环（带过程展示）。"""

import os
from pathlib import Path

from dotenv import load_dotenv

from agents.core.reflection import Critic, Generator, Reviser
from utils.trace import (
    TraceWriter,
    TracedLLM,
    default_trace_path,
    resolve_verbose,
)
from utils.core import LLMChat


class ReflectionAgent:
    def __init__(self, llm: LLMChat, max_rounds: int = 2) -> None:
        self.llm = llm
        self.generator = Generator(llm)
        self.critic = Critic(llm)
        self.reviser = Reviser(llm)
        self.max_rounds = max_rounds

    def run(
        self,
        task: str,
        verbose: bool | str = False,
        trace_file: str | Path | None = None,
    ) -> str:
        """verbose=True 打印摘要；verbose="full" 或指定 trace_file 时记录完整过程。"""
        full, show = resolve_verbose(verbose, trace_file)
        if full:
            path = Path(trace_file) if trace_file else default_trace_path("reflection")
            with TraceWriter(path) as trace:
                trace.section("模型")
                trace.write(self.llm.model)
                trace.section("任务")
                trace.write(task)
                answer = self._run(task, show, trace)
            print(f"完整过程已写入: {path}")
            return answer
        return self._run(task, show)

    def _run(
        self,
        task: str,
        show: bool,
        trace: TraceWriter | None = None,
    ) -> str:
        if trace:
            llm = TracedLLM(self.llm, trace)
            generator = Generator(llm)
            critic = Critic(llm)
            reviser = Reviser(llm)
        else:
            generator, critic, reviser = (
                self.generator,
                self.critic,
                self.reviser,
            )

        answer = generator.generate(task)
        if trace:
            trace.section("初稿")
            trace.write(answer)
        if show:
            print(f"--- 初稿 ---\n{answer}\n")

        for round_no in range(1, self.max_rounds + 1):
            critique = critic.critique(task, answer)
            if trace:
                trace.section(f"第 {round_no} 轮批评")
                trace.write(critique)
            if show:
                print(f"--- 第 {round_no} 轮批评 ---\n{critique}\n")
            answer = reviser.revise(task, answer, critique)
            if trace:
                trace.section(f"第 {round_no} 轮改进")
                trace.write(answer)
            if show:
                print(f"--- 第 {round_no} 轮改进 ---\n{answer}\n")

        if trace:
            trace.section("最终答案")
            trace.write(answer)
        return answer

    def demo(
        self,
        task: str,
        verbose: bool | str = True,
        trace_file: str | Path | None = None,
    ) -> str:
        """完整展示生成、批评、改进的循环过程。"""
        print("=" * 20, "Reflection Demo", "=" * 20)
        print(f"任务: {task}\n")
        return self.run(task, verbose=verbose, trace_file=trace_file)


if __name__ == "__main__":
    load_dotenv()
    llm = LLMChat(
        model=os.getenv("MODEL") or "",
        api_key=os.getenv("API_KEY") or "",
        base_url=os.getenv("BASE_URL") or "",
    )
    agent = ReflectionAgent(llm)
    print(agent.run("用一句话解释什么是递归。"))
