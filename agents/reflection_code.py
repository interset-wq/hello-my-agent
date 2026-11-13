"""Reflection 代码智能体：生成、评审、优化代码。"""

import re
from pathlib import Path

from utils.trace import (
    TraceWriter,
    TracedLLM,
    default_trace_path,
    resolve_verbose,
)
from utils.core import LLMChat

INITIAL_PROMPT = """你是一位资深程序员.请根据以下要求编写 Python 函数，
代码需包含完整函数签名、文档字符串，并遵循 PEP 8.
要求: {task}
请直接输出代码，不要额外解释."""

REFLECT_PROMPT = """你是一位极其严格的代码评审专家，对算法效率有极致要求.
请审查以下代码，重点找出其在算法效率上的主要瓶颈.

# 原始任务:
{task}

# 待审查的代码:
```python
{code}
```

分析其时间复杂度，判断是否存在算法上更优的解法（例如用筛法替代试除法）.
若已最优，才回答“无需改进”.请直接输出反馈，不要额外解释."""

REFINE_PROMPT = """你是一位资深程序员，正根据评审专家反馈优化代码.

# 原始任务:
{task}

# 你上一轮尝试的代码:
{last_code}

# 评审员的反馈:
{feedback}

请据此生成优化后的新版本代码，遵循 PEP 8.请直接输出代码，不要额外解释."""

_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def clean_code(text: str) -> str:
    """去掉模型可能添加的 Markdown 代码围栏。"""
    match = _FENCE_RE.search(text)
    return (match.group(1) if match else text).strip()


class Memory:
    """保存执行结果和反思意见。"""

    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []

    def add(self, kind: str, content: str) -> None:
        self.entries.append((kind, content))

    def last_execution(self) -> str:
        for kind, content in reversed(self.entries):
            if kind == "execution":
                return content
        return ""


class ReflectionAgent:
    def __init__(self, llm: LLMChat, max_iterations: int = 3) -> None:
        self.llm = llm
        self.memory = Memory()
        self.max_iterations = max_iterations

    def run(
        self,
        task: str,
        verbose: bool | str = False,
        trace_file: str | Path | None = None,
    ) -> str:
        """verbose=True 打印摘要；verbose="full" 或指定 trace_file 时记录完整过程。"""
        full, show = resolve_verbose(verbose, trace_file)
        if full:
            path = (
                Path(trace_file)
                if trace_file
                else default_trace_path("reflection_code")
            )
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
        llm = TracedLLM(self.llm, trace) if trace else self.llm
        code = clean_code(
            llm.invoke([
                {"role": "user", "content": INITIAL_PROMPT.format(task=task)}
            ]) or ""
        )
        self.memory.add("execution", code)
        if trace:
            trace.section("初稿")
            trace.write(code)
        if show:
            print(f"--- 初稿 ---\n{code}\n")

        for round_no in range(1, self.max_iterations + 1):
            feedback = llm.invoke([
                {
                    "role": "user",
                    "content": REFLECT_PROMPT.format(
                        task=task, code=self.memory.last_execution()
                    ),
                }
            ]) or ""
            self.memory.add("reflection", feedback)
            if trace:
                trace.section(f"第 {round_no} 轮评审")
                trace.write(feedback)
            if show:
                print(f"--- 第 {round_no} 轮评审 ---\n{feedback}\n")
            if "无需改进" in feedback:
                if trace:
                    trace.section("评审通过")
                    trace.write("无需改进，停止优化。")
                if show:
                    print("--- 评审通过，停止改进 ---\n")
                break

            code = clean_code(
                llm.invoke([
                    {
                        "role": "user",
                        "content": REFINE_PROMPT.format(
                            task=task,
                            last_code=self.memory.last_execution(),
                            feedback=feedback,
                        ),
                    }
                ]) or self.memory.last_execution()
            )
            self.memory.add("execution", code)
            if trace:
                trace.section(f"第 {round_no} 轮改进")
                trace.write(code)
            if show:
                print(f"--- 第 {round_no} 轮改进 ---\n{code}\n")

        answer = self.memory.last_execution()
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
        """完整展示生成、评审、改进与收敛的过程。"""
        print("=" * 20, "Code Reflection Demo", "=" * 20)
        print(f"任务: {task}\n")
        return self.run(task, verbose=verbose, trace_file=trace_file)
