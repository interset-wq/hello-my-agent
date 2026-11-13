"""Plan-and-Solve：基于 LangChain 的实现。"""

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel

from utils.langchain_common import ask, build_model
from agents.core.plan_and_solve import EXEC_PROMPT, PLAN_PROMPT, parse_steps


class Planner:
    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def plan(self, task: str) -> list[str]:
        """生成步骤列表。"""
        return parse_steps(ask(self.model, PLAN_PROMPT.format(task=task)))


class Executor:
    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def execute(
        self,
        question: str,
        plan: list[str],
        on_step: Callable[[str, str], None] | None = None,
    ) -> str:
        """逐步执行计划，最后一步的结果即最终答案。"""
        if not plan:
            plan = [question]

        plan_text = "\n".join(
            f"{index}. {step}" for index, step in enumerate(plan, 1)
        )
        history = ""
        answer = ""
        for step in plan:
            prompt = EXEC_PROMPT.format(
                question=question,
                plan=plan_text,
                history=history or "无",
                current_step=step,
            )
            answer = ask(self.model, prompt)
            history += f"步骤: {step}\n结果: {answer}\n\n"
            if on_step:
                on_step(step, answer)
        return answer


class PlanAndSolveAgent:
    def __init__(self, model: str | None = None) -> None:
        chat = build_model(model)
        self.planner = Planner(chat)
        self.executor = Executor(chat)

    def run(self, task: str) -> str:
        return self.executor.execute(task, self.planner.plan(task))
