"""Reflection：基于 LangChain 的实现。"""

from langchain_core.language_models import BaseChatModel

from utils.langchain_common import ask, build_model
from agents.core.reflection import ANSWER_PROMPT, CRITIQUE_PROMPT, REVISE_PROMPT


class Generator:
    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def generate(self, task: str) -> str:
        """生成初稿。"""
        return ask(self.model, ANSWER_PROMPT.format(task=task))


class Critic:
    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def critique(self, task: str, answer: str) -> str:
        """指出答案中的错误、遗漏或可改进之处。"""
        return ask(self.model, CRITIQUE_PROMPT.format(task=task, answer=answer))


class Reviser:
    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def revise(self, task: str, answer: str, critique: str) -> str:
        """根据批评改进答案；失败时保留原答案。"""
        return ask(
            self.model,
            REVISE_PROMPT.format(task=task, answer=answer, critique=critique),
        ) or answer


class ReflectionAgent:
    def __init__(self, model: str | None = None, max_rounds: int = 2) -> None:
        chat = build_model(model)
        self.generator = Generator(chat)
        self.critic = Critic(chat)
        self.reviser = Reviser(chat)
        self.max_rounds = max_rounds

    def run(self, task: str) -> str:
        answer = self.generator.generate(task)
        for _ in range(self.max_rounds):
            critique = self.critic.critique(task, answer)
            answer = self.reviser.revise(task, answer, critique)
        return answer
