"""Plan-and-Solve：先制定计划，再逐步求解（带过程展示）。"""

from pathlib import Path

from agents.core.plan_and_solve import Executor, Planner
from utils.trace import (
    TraceWriter,
    TracedLLM,
    default_trace_path,
    resolve_verbose,
)
from utils.core import LLMChat


class PlanAndSolveAgent:
    def __init__(self, llm: LLMChat) -> None:
        self.llm = llm
        self.planner = Planner(llm)
        self.executor = Executor(llm)

    def run(
        self,
        task: str,
        verbose: bool | str = False,
        trace_file: str | Path | None = None,
    ) -> str:
        """verbose=True 打印摘要；verbose="full" 或指定 trace_file 时记录完整过程。"""
        full, show = resolve_verbose(verbose, trace_file)

        if full:
            return self._run_traced(task, show, trace_file)

        plan = self.planner.plan(task)
        if show:
            self._print_plan(plan)

        step_no = 0

        def report(step: str, answer: str) -> None:
            nonlocal step_no
            step_no += 1
            print(f"--- 步骤 {step_no} ---\n{step}\n结果: {answer}\n")

        answer = self.executor.execute(task, plan, report if show else None)
        if show:
            print(f"--- 最终答案 ---\n{answer}\n")
        return answer

    def _run_traced(
        self,
        task: str,
        show: bool,
        trace_file: str | Path | None,
    ) -> str:
        path = Path(trace_file) if trace_file else default_trace_path("pas")
        with TraceWriter(path) as trace:
            trace.section("模型")
            trace.write(self.llm.model)
            trace.section("任务")
            trace.write(task)

            llm = TracedLLM(self.llm, trace)
            plan = Planner(llm).plan(task)
            trace.section("解析后的计划")
            trace.write(
                "\n".join(
                    f"{index}. {step}"
                    for index, step in enumerate(plan, 1)
                )
                or "<空>"
            )
            if show:
                self._print_plan(plan)

            step_no = 0

            def report(step: str, answer: str) -> None:
                nonlocal step_no
                step_no += 1
                trace.section(f"步骤 {step_no}: {step}")
                trace.write(f"[步骤结果]\n{answer}")
                if show:
                    print(f"--- 步骤 {step_no} ---\n{step}\n结果: {answer}\n")

            answer = Executor(llm).execute(task, plan, report)
            trace.section("最终答案")
            trace.write(answer)

        print(f"完整过程已写入: {path}")
        if show:
            print(f"--- 最终答案 ---\n{answer}\n")
        return answer

    @staticmethod
    def _print_plan(plan: list[str]) -> None:
        plan_text = "\n".join(
            f"{index}. {step}" for index, step in enumerate(plan, 1)
        )
        print(f"--- 计划 ---\n{plan_text}\n")

    def demo(
        self,
        task: str,
        verbose: bool | str = True,
        trace_file: str | Path | None = None,
    ) -> str:
        """完整展示制定计划、逐步执行的过程。"""
        print("=" * 20, "Plan-and-Solve Demo", "=" * 20)
        print(f"任务: {task}\n")
        return self.run(task, verbose=verbose, trace_file=trace_file)

