"""Reflection 核心逻辑：生成、批评、改进循环。"""

from utils.core import LLMChat

# 初稿只要求作答，不附加额外约束
ANSWER_PROMPT = """请回答下面的任务：
{task}"""

# 只输出批评，避免模型在批评阶段直接改写答案
CRITIQUE_PROMPT = """任务：
{task}

当前答案：
{answer}

请指出当前答案中的错误、遗漏或可改进之处，只输出批评意见。"""

# 结合批评给出改进后的答案
REVISE_PROMPT = """任务：
{task}

当前答案：
{answer}

修改意见：
{critique}

请根据修改意见写出改进后的最终答案，只输出最终答案。"""


class Generator:
    def __init__(self, llm: LLMChat) -> None:
        self.llm = llm

    def generate(self, task: str) -> str:
        """生成初稿。"""
        # invoke 固定关闭思考，返回空串时由调用方决定如何处理
        return self.llm.invoke(
            [{"role": "user", "content": ANSWER_PROMPT.format(task=task)}]
        ) or ""


class Critic:
    def __init__(self, llm: LLMChat) -> None:
        self.llm = llm

    def critique(self, task: str, answer: str) -> str:
        """只提出批评，不直接改写答案。"""
        return self.llm.invoke([
            {
                "role": "user",
                "content": CRITIQUE_PROMPT.format(task=task, answer=answer),
            }
        ]) or ""


class Reviser:
    def __init__(self, llm: LLMChat) -> None:
        self.llm = llm

    def revise(self, task: str, answer: str, critique: str) -> str:
        """根据批评改进答案；失败时保留原答案。"""
        # 空响应通常表示调用失败，此时退回上一版答案
        return self.llm.invoke([
            {
                "role": "user",
                "content": REVISE_PROMPT.format(
                    task=task, answer=answer, critique=critique
                ),
            }
        ]) or answer


class ReflectionAgent:
    def __init__(self, llm: LLMChat, max_rounds: int = 2) -> None:
        # 三个角色共享同一个 LLM，仅通过提示词区分职责
        self.generator = Generator(llm)
        self.critic = Critic(llm)
        self.reviser = Reviser(llm)
        # 每轮包含一次批评和一次改进
        self.max_rounds = max_rounds

    def run(self, task: str) -> str:
        answer = self.generator.generate(task)
        # 固定执行 max_rounds 轮，不额外判断是否收敛
        for _ in range(self.max_rounds):
            critique = self.critic.critique(task, answer)
            answer = self.reviser.revise(task, answer, critique)
        return answer
