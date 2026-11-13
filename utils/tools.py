"""agents 可直接使用的基础工具。"""

import ast
import json
import operator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.core import ToolExecutor

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "db.json"

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        return _BINARY_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    raise ValueError("不支持的表达式")


def calculator(expression: str) -> str:
    """安全计算算术表达式，例如 (12 + 8) * 3。"""
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as exc:
        return f"错误: {exc}"


def word_count(text: str) -> str:
    """统计文本的字符数与空格分隔的词数。"""
    return f"字符数: {len(text)}，词数: {len(text.split())}"


def current_time() -> str:
    """返回当前日期时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def query_database(name: str) -> str:
    """查询模拟数据库中的数值记录，支持部分名称匹配。"""
    try:
        data = json.loads(_DB_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"错误: {exc}"
    if name in data:
        return str(data[name])
    for key, value in data.items():
        if name in key or key in name:
            return str(value)
    return f"未找到 {name}"


def register_builtin_tools(tools: "ToolExecutor") -> None:
    """把内置工具注册到 ToolExecutor。"""
    tools.register("calculator", "计算算术表达式，例如 (12 + 8) * 3", calculator)
    tools.register("word_count", "统计文本的字符数和词数", word_count)
    tools.register("current_time", "获取当前日期时间，无需参数", current_time)
    tools.register(
        "query_database",
        "查询模拟数据库，例如查询星尘兽或月影兽的体重（单位：吨）",
        query_database,
    )
