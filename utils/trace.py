"""Agent 过程追踪：把提示词、原始输出和中间结果写入文件。"""

import time
from datetime import datetime
from pathlib import Path
from typing import TextIO

from utils.core import LLMChat


def default_trace_path(prefix: str) -> Path:
    """在项目 temp/ 下生成带时间戳的追踪文件路径。"""
    root = Path(__file__).resolve().parent.parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / "temp" / f"{prefix}_{stamp}.txt"


def resolve_verbose(
    verbose: bool | str, trace_file: str | Path | None
) -> tuple[bool, bool]:
    """返回 (是否写追踪文件, 是否打印摘要)。"""
    if verbose not in (False, True, "full"):
        raise ValueError('verbose 只能是 False、True 或 "full"')
    full = verbose == "full" or trace_file is not None
    show = verbose is True or verbose == "full"
    return full, show


class TraceWriter:
    """按段落写入追踪文件，并立即刷新，便于实时查看。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file: TextIO = self.path.open("w", encoding="utf-8")

    def section(self, title: str) -> None:
        self.write(f"\n{'=' * 20} {title} {'=' * 20}")

    def write(self, text: str) -> None:
        self.file.write(text)
        if not text.endswith("\n"):
            self.file.write("\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class TracedLLM(LLMChat):
    """包装 LLMChat，记录每次 invoke 的完整提示词和原始输出。"""

    def __init__(self, llm: LLMChat, trace: TraceWriter) -> None:
        self.model = llm.model
        self.client = llm.client
        self.trace = trace
        self.calls = 0

    def invoke(self, messages: list, temperature: float = 0.7) -> str | None:
        self.calls += 1
        self.trace.section(f"LLM 调用 #{self.calls} @ {datetime.now():%H:%M:%S}")
        self.trace.write(f"[参数] temperature={temperature}")
        self.trace.write("[提示词]")
        for message in messages:
            role = message.get("role", "?")
            content = message.get("content") or ""
            self.trace.write(f"<{role}>\n{content}")

        start = time.perf_counter()
        output = super().invoke(messages, temperature)
        self.trace.write(f"[原始输出] (耗时 {time.perf_counter() - start:.3f}s)")
        self.trace.write(output or "<空>")
        return output
