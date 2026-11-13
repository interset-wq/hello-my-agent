"""Reflection 代码智能体：基于 LangChain 的实现。"""

from utils.langchain_common import ask, build_model
from agents.reflection_code import (
    INITIAL_PROMPT,
    REFINE_PROMPT,
    REFLECT_PROMPT,
    Memory,
    clean_code,
)


class ReflectionAgent:
    def __init__(
        self, model: str | None = None, max_iterations: int = 3
    ) -> None:
        self.model = build_model(model)
        self.memory = Memory()
        self.max_iterations = max_iterations

    def run(self, task: str) -> str:
        code = clean_code(ask(self.model, INITIAL_PROMPT.format(task=task)))
        self.memory.add("execution", code)

        for _ in range(self.max_iterations):
            feedback = ask(
                self.model,
                REFLECT_PROMPT.format(
                    task=task, code=self.memory.last_execution()
                ),
            )
            self.memory.add("reflection", feedback)
            if "无需改进" in feedback:
                break

            code = clean_code(
                ask(
                    self.model,
                    REFINE_PROMPT.format(
                        task=task,
                        last_code=self.memory.last_execution(),
                        feedback=feedback,
                    ),
                ) or self.memory.last_execution()
            )
            self.memory.add("execution", code)

        return self.memory.last_execution()
