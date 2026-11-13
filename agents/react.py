"""ReAct：思考（Reason）与行动（Act）交替进行的最简实现。"""

import re
from pathlib import Path

from agents.core.react import PROMPT, parse
from utils.trace import (
    TraceWriter,
    TracedLLM,
    default_trace_path,
    resolve_verbose,
)
from utils.core import LLMChat, ToolExecutor


class ReActAgent:
    def __init__(self, llm: LLMChat, tools: ToolExecutor, max_steps: int = 5) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps

    def run(
        self,
        question: str,
        verbose: bool | str = False,
        trace_file: str | Path | None = None,
    ) -> str:
        """verbose=True 打印摘要；verbose="full" 或指定 trace_file 时记录完整过程。"""
        full, show = resolve_verbose(verbose, trace_file)
        if full:
            path = Path(trace_file) if trace_file else default_trace_path("react")
            with TraceWriter(path) as trace:
                trace.section("模型")
                trace.write(self.llm.model)
                trace.section("可用工具")
                trace.write(self.tools.describe_tools())
                trace.section("问题")
                trace.write(question)
                answer = self._run(question, show, trace)
            print(f"完整过程已写入: {path}")
            return answer
        return self._run(question, show)

    def _run(
        self,
        question: str,
        show: bool,
        trace: TraceWriter | None = None,
    ) -> str:
        llm = TracedLLM(self.llm, trace) if trace else self.llm
        history = ""
        for step in range(1, self.max_steps + 1):
            if trace:
                trace.section(f"第 {step} 步")
            prompt = PROMPT.format(
                tools=self.tools.describe_tools(),
                question=question,
                history=history or "（无）",
            )
            output = llm.invoke([{"role": "user", "content": prompt}]) or ""
            output = output.split("Observation:")[0].strip()
            action, action_input, final = parse(output)
            thought = re.search(r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)", output, re.S)
            thought_text = thought.group(1).strip() if thought else ""
            if trace:
                trace.write("[解析结果]")
                if final is not None:
                    trace.write(f"Thought: {thought_text}\nFinal Answer: {final}")
                elif action is not None:
                    trace.write(
                        f"Thought: {thought_text}\n"
                        f"Action: {action}\nAction Input: {action_input}"
                    )
                else:
                    trace.write(f"[未解析出 Action/Final Answer]\n{output}")
            if show:
                print(f"--- 第 {step} 步 ---")
                if final is not None:
                    print(f"Thought: {thought_text}")
                    print(f"Final Answer: {final}\n")
                elif action is not None:
                    print(f"Thought: {thought_text}")
                    print(f"Action: {action}")
                    print(f"Action Input: {action_input}")
                else:
                    print(output)
            if final is not None:
                if trace:
                    trace.section("最终答案")
                    trace.write(final)
                return final
            if action is None:
                if trace:
                    trace.section("最终答案")
                    trace.write(output)
                return output.strip()
            observation = self.tools.run(action, action_input)
            if trace:
                trace.write(
                    f"[工具执行] {action}({action_input})\n"
                    f"[Observation]\n{observation}"
                )
            if show:
                print(f"Observation: {observation}\n")
            history += (
                f"Thought: {thought_text}\n"
                f"Action: {action}\nAction Input: {action_input}\n"
                f"Observation: {observation}\n"
            )
        if trace:
            trace.section("结束")
            trace.write("已达到最大步数，未能得到最终答案。")
        return "已达到最大步数，未能得到最终答案。"

    def demo(
        self,
        question: str,
        verbose: bool | str = True,
        trace_file: str | Path | None = None,
    ) -> str:
        """完整展示思考、行动、观察的循环过程。"""
        print("=" * 20, "ReAct Demo", "=" * 20)
        print(f"可用工具:\n{self.tools.describe_tools()}\n")
        print(f"问题: {question}\n")
        return self.run(question, verbose=verbose, trace_file=trace_file)
