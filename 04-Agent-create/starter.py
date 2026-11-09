"""
练习 3：从零实现 ReAct（不用 framework）— starter.py（Path A、Ollama 默认）

用一小段 Python 把「assistant 文字 → Tool Call → Tool Result」循环写出来。
不要 LangChain、不要 LangGraph，就是纯 while loop。

跑法：
    pip install -r requirements.txt
    ollama pull qwen2.5:3b   # Stage 3+ tool-use 默认 model
    ollama serve             # 默认 port 11434
    python starter.py

验证：
    python test.py   （test.py 跨 backend 通用、用 mock、不打 API）

想看 Anthropic Claude 版本：
    python starter_anthropic.py   （需 ANTHROPIC_API_KEY；每次先保留 $0.05）
"""

from __future__ import annotations

import ast #抽象语法树模块
import json
import math
import os
import sys
from typing import Any
from LLM_factory.client import get_llm_client
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI

# === 1. Tools 定义（含实现）===

MAX_EXPRESSION_LENGTH = 200
MAX_AST_NODES = 50
MAX_AST_DEPTH = 12
MAX_ABS_NUMBER = 1_000_000_000_000


def _within_bounds(value: int | float) -> bool:
    if isinstance(value, int):
        return abs(value) <= MAX_ABS_NUMBER
    return math.isfinite(value) and abs(value) <= MAX_ABS_NUMBER


"""
          [Expression]
               │
            [BinOp]  (最外层是加法)
           /   │   \
          /    │    \
    [UnaryOp] [Add] [Constant: 2]
      /   \     (加号)
  [USub] [Constant: 5]
  (负号)  (数字 5)
"""

def _evaluate_arithmetic(expression: str) -> int | float:
    """只计算小型数学式；模型文字永远不会变成代码。"""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("expression is too long")
    try:
        tree = ast.parse(expression, mode="eval") #todo：将字符串解析为节点数，eval表示，只接受表达式，比如：1+2
    except (SyntaxError, ValueError, TypeError):
        raise ValueError("calculator only accepts basic arithmetic") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES: #todo：广度优先搜索遍历总节点数
        raise ValueError("expression has too many parts")

    def visit(node: ast.AST, depth: int = 0) -> int | float:
        if depth > MAX_AST_DEPTH:
            raise ValueError("expression is too deeply nested")
        if isinstance(node, ast.Expression): #遇到根节点节点，审差下一层
            return visit(node.body, depth + 1)
        #检查数字及类型
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
            if not _within_bounds(value):
                raise ValueError("number is out of bounds")
            return value
        #正负运算，todo：提取运算符;ast.BinOp 代表二元运算; ast.UnartOp代表一元运算
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand, depth + 1) #将负号后括号中的运算先进性
            result = value if isinstance(node.op, ast.UAdd) else -value
        #加减乘除运算
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = visit(node.left, depth + 1)
            right = visit(node.right, depth + 1)
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ValueError("division by zero")
                result = left / right
            elif isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            else:
                result = left * right
        else:
            raise ValueError("calculator only accepts numbers and + - * /")
        if not _within_bounds(result):
            raise ValueError("result is out of bounds")
        return result

    return visit(tree)


def tool_calculator(expression: str) -> str:
    """安全的计算器：只允许 + - * / 跟数字。"""
    try:
        return str(_evaluate_arithmetic(expression))
    except (TypeError, ValueError) as exc:
        return f"error: {exc}"


def tool_lookup_fact(query: str) -> str:
    """假的事实查询（教学专用、避免依赖外部 API）。"""
    facts = {
        "台北人口": "2602000",
        "纽约人口": "8336000",
        "光速": "299792458",  # m/s
    }
    return facts.get(query.strip(), f"unknown: {query}")


# OpenAI-compatible 的 tools schema wrap 在 {"type":"function", "function":{...}} 里
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "做基本算术运算（加减乘除）。输入是表达式字符串。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "算术表达式"},
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_fact",
            "description": "查询一个事实（人口 / 物理常数等）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "查询关键字"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_IMPL = {
    "calculator": lambda inp: tool_calculator(inp["expression"]),
    "lookup_fact": lambda inp: tool_lookup_fact(inp["query"]),
}

TOOL_FIELDS = {
    "calculator": "expression",
    "lookup_fact": "query",
}


def execute_tool(name: str, raw_arguments: str) -> tuple[dict, str, bool]:
    """Validate an untrusted tool call before dispatch."""
    if name not in TOOL_IMPL:
        return {}, f"error: tool not allowed: {name}", True
    try:
        args = json.loads(raw_arguments)
    except (TypeError, json.JSONDecodeError):
        return {}, "error: arguments must be valid JSON", True
    if not isinstance(args, dict):
        return {}, "error: arguments must be a JSON object", True
    field = TOOL_FIELDS[name]
    if set(args) != {field}:
        return {}, f"error: arguments must contain exactly {field}", True
    if not isinstance(args[field], str) or not args[field].strip():
        return {}, f"error: {field} must be a non-empty string", True
    try:
        observation = TOOL_IMPL[name](args)
    except (KeyError, TypeError, ValueError) as exc:
        return args, f"error: invalid arguments: {exc}", True
    return args, observation, observation.startswith("error:")


# === 2. ReAct loop ===

def react_loop(question: str, max_iter: int = 6, client: Any = None) -> dict:
    """
    OpenAI-compatible ReAct loop。每轮：
      1. 问 LLM（含 tools）
      2. finish_reason: 'tool_calls' → 执行 tool、observation 接回、继续
                       'stop' → 结束、最后 message 是答案
    回传 {final, trace, steps}。
    """
    MODEL,api_key,base_url,timeout = get_llm_client("qwen")
    client = client or OpenAI(base_url=base_url, api_key=api_key)
    messages = [{"role": "user", "content": question}]
    trace: list[dict] = []

    for step in range(1,max_iter+1):
        resp = client.chat.completions.create(
            model=MODEL,
            tools=TOOLS_SPEC,
            messages=messages,
        )
        msg = resp.choices[0].message
        assistant_text = msg.content or ""
        tool_calls = msg.tool_calls or []

        # 把 assistant message 加进 messages（OpenAI 格式）
        assistant_entry: dict = {"role": "assistant", "content": assistant_text}
        if tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        messages.append(assistant_entry)

        finish_reason = resp.choices[0].finish_reason
        if not tool_calls:
            trace.append({"step": step, "assistant_text": assistant_text, "tool": None, "obs": None})
            if finish_reason == "stop":
                return {"final": assistant_text, "trace": trace, "steps": step + 1}
            return {
                "final": None,
                "trace": trace,
                "steps": step + 1,
                "terminal_reason": finish_reason or "missing_tool_call",
                "truncated": finish_reason == "length",
            }

        # 执行 tool calls、observation 接回（OpenAI 用 role="tool"）
        last_obs = ""
        for tc in tool_calls:
            args, obs, _ = execute_tool(tc.function.name, tc.function.arguments)
            last_obs = obs
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": obs,
            })

            trace.append({
                "step": step,
                "assistant_text": assistant_text,
                "tool": tc.function.name,
                "tool_input": args,
                "obs": last_obs,
            })

    return {"final": None, "trace": trace, "steps": max_iter, "truncated": True}


# === 3. 自我验证 ===

if __name__ == "__main__":
    question = "'台北人口' 除以 '纽约人口'、答案保留 4 位小数。"
    print(f"❓ 问题：{question}")
    print("-" * 60)

    result = react_loop(question, max_iter=5)

    for entry in result["trace"]:
        print(f"[step {entry['step']}] assistant: {(entry['assistant_text'] or '')[:80]}...")
        if entry["tool"]:
            print(f"           tool: {entry['tool']}({entry.get('tool_input')}) → {entry['obs']}")
    print("-" * 60)
    print(f"✅ 最终答案：{result['final']}")
    print(f"   共 {result['steps']} 轮")

    # 宽松验证（小 model 不一定精确到 4 位小数）
    assert result.get("final") is not None or result.get("truncated"), "loop 应收尾或显式 truncate"
    assert result["steps"] <= 5, "loop 不可超过 max_iter"