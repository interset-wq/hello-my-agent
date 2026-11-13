# hello-my-agent

一个用于学习 Agent 原理的最小项目：手写 ReAct、Plan-and-Solve、Reflection，并提供 LangChain 低层实现（不使用 `create_agent` 等高层封装）。

## 特性

- 纯手写核心逻辑，提示词、解析和执行循环均可读
- 同一套 Agent 提供多种实现：手写解析、Responses API 工具调用、LangChain
- 支持摘要与完整追踪：记录每轮提示词、LLM 原始输出、解析结果和工具调用
- 内置模拟数据库与安全计算器等工具
- 使用 OpenAI 兼容接口，默认适配本地 Ollama

## 环境要求

- Python >= 3.14
- [uv](https://docs.astral.sh/uv/)
- 一个 OpenAI 兼容的模型服务（默认 Ollama）

## 快速开始

```bash
uv sync
cp .sample.env .env
```

Windows PowerShell:

```powershell
uv sync
Copy-Item .sample.env .env
```

`.sample.env` 默认配置：

```dotenv
MODEL=qwen3:8b
API_KEY=ollama
BASE_URL=http://localhost:11434/v1
```

修改 `.env` 以适配你的模型服务。`.env` 已被 `.gitignore` 忽略。

## 运行示例

```bash
uv run python -m example.react_demo
uv run python -m example.plan_and_solve_demo
uv run python -m example.reflection_demo
uv run python -m example.reflection_code_demo
```

`plan_and_solve_demo` 包含两个任务：百分比应用题和鸡兔同笼。上述 Agent demo 默认使用 `verbose="full"`，完整过程写入项目 `temp/` 目录。

## 项目结构

```text
agents/
  core/                     # 纯核心逻辑，不包含展示代码
    react.py                # ReAct 循环与解析
    plan_and_solve.py       # 规划 + 逐步执行
    reflection.py           # 生成 + 批评 + 改进
  react.py                  # ReAct 演示版，支持 verbose/追踪
  react_tool.py             # ReAct：Responses API 原生工具调用
  react_langchain.py        # ReAct：LangChain 手动工具循环
  plan_and_solve.py         # Plan-and-Solve 演示版
  plan_and_solve_langchain.py
  reflection.py             # Reflection 演示版
  reflection_langchain.py
  reflection_code.py        # 代码 Reflection
  reflection_code_langchain.py
utils/
  core.py                   # LLMChat、ToolExecutor、timed
  tools.py                  # calculator、word_count、current_time、query_database
  trace.py                  # TraceWriter、TracedLLM
  langchain_common.py       # build_model、ask
data/
  db.json                   # query_database 读取的模拟数据库
example/                    # 可运行示例
```

## Agent 选择

| Agent | 适合场景 |
| --- | --- |
| ReAct | 需要调用外部工具、多轮查询与计算 |
| Plan-and-Solve | 多步数学/逻辑应用题，先规划再执行 |
| Reflection | 需要自我批评、改进答案质量 |
| Code Reflection | 代码生成、复杂度评审与优化 |

## 完整过程追踪

带演示逻辑的 Agent 支持三种输出模式：

```python
agent.run(task)                                      # 默认不输出
agent.run(task, verbose=True)                        # 控制台打印摘要
agent.run(task, verbose="full")                      # 摘要 + 写入默认追踪文件
agent.run(task, verbose="full", trace_file="logs/run.txt")
agent.run(task, trace_file="logs/run.txt")           # 只写文件
```

默认文件路径为 `temp/<agent>_<时间戳>.txt`，包含：

- 模型、任务/问题、可用工具
- 每轮完整提示词
- LLM 原始输出与耗时
- 解析后的 Thought / Action / 参数
- 工具执行结果（Observation）
- 计划、批评、改进等中间结果
- 最终答案

LangChain 版本目前只保留最小实现，不包含追踪逻辑。

## 在代码中使用

### ReAct

```python
import os

from dotenv import load_dotenv

from agents.react import ReActAgent
from utils.core import LLMChat, ToolExecutor
from utils.tools import calculator, query_database

load_dotenv()

llm = LLMChat(
    model=os.getenv("MODEL") or "",
    api_key=os.getenv("API_KEY") or "",
    base_url=os.getenv("BASE_URL") or "",
)
tools = ToolExecutor()
tools.register(
    "query_database",
    "查询模拟数据库，例如查询星尘兽或月影兽的体重（单位：吨）",
    query_database,
)
tools.register("calculator", "计算算术表达式，例如 (12 + 8) * 3", calculator)

agent = ReActAgent(llm, tools)
print(agent.run("星尘兽的体重是月影兽的多少倍？", verbose="full"))
```

### Plan-and-Solve

```python
from agents.plan_and_solve import PlanAndSolveAgent

agent = PlanAndSolveAgent(llm)
print(
    agent.run(
        "笼子里有若干只鸡和兔，从上面数有 35 个头，从下面数有 94 只脚。"
        "问鸡和兔各有多少只？",
        verbose="full",
    )
)
```

### Reflection

```python
from agents.reflection import ReflectionAgent

agent = ReflectionAgent(llm, max_rounds=2)
print(agent.run("用一句话解释什么是递归。", verbose="full"))
```

### Code Reflection

```python
from agents.reflection_code import ReflectionAgent

agent = ReflectionAgent(llm, max_iterations=3)
print(agent.run("实现一个函数，返回 1 到 n 之间的所有质数。", verbose="full"))
```

## 工具与数据

- `calculator`：基于 `ast` 的安全算术求值，不执行任意代码
- `word_count`：统计字符数和词数
- `current_time`：返回当前时间
- `query_database`：读取 `data/db.json`，支持精确和部分名称匹配

示例数据库：

```json
{
  "星尘兽": 150,
  "月影兽": 6
}
```

## LangChain 实现

`agents/*_langchain.py` 使用 `utils/langchain_common.py` 中的 `build_model()` 和 `ask()`，手写工具调用/执行循环，不使用 `create_agent` 等高层封装：

```python
from agents.react_langchain import ReActAgent, calculator

agent = ReActAgent([calculator])
print(agent.run("计算 (12 + 8) * 3"))
```

## 说明

- `LLMChat.invoke()` 固定关闭思考，`LLMChat.chat()` 默认开启思考，便于对比
- 如果 Windows 控制台出现中文编码问题，可设置 `PYTHONIOENCODING=utf-8`
- 支持任意 OpenAI 兼容接口，只需修改 `.env`
