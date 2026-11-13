"""Plan-and-Solve 核心逻辑：先制定计划，再逐步执行。"""

import re
from collections.abc import Callable

from utils.core import LLMChat

# 要求按编号逐行输出，便于解析成步骤列表
PLAN_PROMPT = """请为下面的任务制定解决计划，按编号逐行列出步骤，不要给出答案。
每个步骤单独一行，格式为“1. ...”，不要添加其他内容。
最后一步应得出最终答案。

任务：{task}"""

# 同时提供完整计划和历史，避免执行时偏离整体目标
EXEC_PROMPT = """你是一位执行专家，严格按计划一步步解决问题。
只专注于解决“当前步骤”，仅输出该步骤的最终答案，不要多余解释。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:"""

# 兼容 "1. ..."、"1) ..."、"步骤 1：..."、"第 1 步：..."
STEP_RE = re.compile(
    r"^\s*(?:\d+[.、)]|步骤\s*\d+[:：]|第\s*\d+\s*步[:：])\s*(.+?)\s*$"
)


def parse_steps(text: str) -> list[str]:
    """从编号列表中提取步骤；没有编号时按非空行拆分。"""
    steps = [
        match.group(1).strip()
        for line in text.splitlines()
        if (match := STEP_RE.match(line))
    ]
    if not steps:
        steps = [line.strip() for line in text.splitlines() if line.strip()]
    return steps


class Planner:
    """规划"""
    def __init__(self, llm: LLMChat) -> None:
        self.llm = llm

    def plan(self, task: str) -> list[str]:
        """生成步骤列表。"""
        output = self.llm.invoke(
            [{"role": "user", "content": PLAN_PROMPT.format(task=task)}]
        ) or ""
        return parse_steps(output)


class Executor:
    """执行"""
    def __init__(self, llm: LLMChat) -> None:
        self.llm = llm

    def execute(
        self,
        question: str,
        plan: list[str],
        on_step: Callable[[str, str], None] | None = None,
    ) -> str:
        """逐步执行计划，最后一步的结果即最终答案。"""
        # 计划为空时退化为直接执行原问题
        if not plan:
            plan = [question]

        # 重新编号后放入提示词，保证模型看到的计划结构清晰
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
            answer = self.llm.invoke([{"role": "user", "content": prompt}]) or ""
            # 累积真实结果，作为后续步骤的上下文
            history += f"步骤: {step}\n结果: {answer}\n\n"
            # 回调仅用于展示进度，不影响核心逻辑
            if on_step:
                on_step(step, answer)
        return answer


class PlanAndSolveAgent:
    def __init__(self, llm: LLMChat) -> None:
        self.planner = Planner(llm)
        self.executor = Executor(llm)

    def run(self, task: str) -> str:
        return self.executor.execute(task, self.planner.plan(task))
