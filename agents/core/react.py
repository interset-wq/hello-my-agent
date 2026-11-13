"""ReAct 核心逻辑：思考、行动、观察循环。"""

import re

from utils.core import LLMChat, ToolExecutor

# 提示词中的严格规则用于抑制模型编造 Observation、一次输出多步
PROMPT = """你是一个可以使用工具的助手，请按以下格式逐步解决问题。

可用工具：
{tools}

严格规则：
1. 每次只输出一个 Thought 和一个 Action，或一个 Final Answer。
2. 不要自己编造 Observation，Observation 由系统在执行 Action 后提供。
3. 需要多步时，等待系统返回 Observation 后再继续。

输出格式：
Thought: 你的思考
Action: 工具名称（必须是上面列表中的一个）
Action Input: 工具输入

当已经得到最终答案时，输出：
Thought: 我已经知道答案
Final Answer: 最终答案

问题：{question}

已执行的过程：
{history}
"""


def parse(output: str) -> tuple[str | None, str | None, str | None]:
    """解析补全，返回 (Action, Action Input, Final Answer)。"""
    # 分别捕获最终答案、工具名和工具输入
    final = re.search(r"Final Answer:\s*(.+)", output, re.S)
    action = re.search(r"Action:\s*([^\n]+)", output)
    action_input = re.search(r"Action Input:\s*([^\n]+)", output)
    if action and action_input:
        # 模型可能先给出 Action，又自行续写 Final Answer；
        # 用 start() 比较出现位置，只保留先出现的那个，丢弃后面的内容
        if final and final.start() < action.start():
            return None, None, final.group(1).strip()
        return action.group(1).strip(), action_input.group(1).strip(), None
    if final:
        return None, None, final.group(1).strip()
    return None, None, None


class ReActAgent:
    def __init__(self, llm: LLMChat, tools: ToolExecutor, max_steps: int = 5) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps

    def run(self, question: str) -> str:
        history = ""
        for _ in range(self.max_steps):
            prompt = PROMPT.format(
                tools=self.tools.describe_tools(),
                question=question,
                history=history or "（无）",
            )
            output = self.llm.invoke([{"role": "user", "content": prompt}]) or ""
            # 模型有时会自己续写 Observation；只保留第一个之前的内容，
            # 避免把伪造的观察和后续推理带入历史
            output = output.split("Observation:")[0].strip()

            action, action_input, final = parse(output)
            if final is not None:
                return final
            # 既没有 Action 也没有 Final Answer 时，直接返回原始输出
            if action is None:
                return output

            # Thought 截止到下一个 Action、Final Answer 或文本末尾
            thought = re.search(
                r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)", output, re.S
            )
            thought_text = thought.group(1).strip() if thought else ""
            observation = self.tools.run(action, action_input)
            # 只记录真实执行得到的 Observation，作为下一轮的上下文
            history += (
                f"Thought: {thought_text}\n"
                f"Action: {action}\nAction Input: {action_input}\n"
                f"Observation: {observation}\n"
            )
        return "已达到最大步数，未能得到最终答案。"
