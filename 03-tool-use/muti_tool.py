import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


# allowlist：只有这里登记的函数才允许被模型调用
available_functions = {
    "add": add,
    "multiply": multiply,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two integers. Use when you need the sum of two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "The first number"},
                    "b": {"type": "integer", "description": "The second number"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two integers. Use when you need the product of two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "The first number"},
                    "b": {"type": "integer", "description": "The second number"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    },
]

messages = [{"role": "user", "content": "What is (11434+12341)*412?"}]

MAX_STEPS = 10  # 有界循环，防止模型无限调工具
for step in range(MAX_STEPS):
    response = client.chat.completions.create(
        model="qwen2.5:3b",
        messages=messages,
        tools=TOOLS,
    )
    choice = response.choices[0]
    msg = choice.message

    # 把这一轮 assistant 输出（可能带 tool_calls）追加进历史
    messages.append(msg.model_dump(exclude_none=True))

    print(f"\n--- step {step + 1} | finish_reason={choice.finish_reason}")
    print("content:", msg.content)
    print("tool_calls:", msg.tool_calls)

    if not msg.tool_calls:
        print("\nFINAL:", msg.content)
        break

    for tc in msg.tool_calls:
        name = tc.function.name

        # 1) allowlist 检查：未知工具不执行，回报错误
        if name not in available_functions:
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps({"error": f"unknown tool: {name}"}),
            })
            continue

        # 2) arguments 是 JSON 字符串，先解析
        args = json.loads(tc.function.arguments)

        # 3) 参数验证：必须是 {a:int, b:int}
        if (
            not isinstance(args, dict)
            or set(args) != {"a", "b"}
            or isinstance(args["a"], bool)
            or isinstance(args["b"], bool)
            or not isinstance(args["a"], int)
            or not isinstance(args["b"], int)
        ):
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps({"error": "a and b must be integers"}),
            })
            continue

        # 4) 执行
        print(f"Calling {name}({args})")
        result = available_functions[name](**args)
        print(f"Result: {result}")

        # 5) 把结果追加进历史，用 tool_call_id 配对
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": str(result),
        })
else:
    print("达到 MAX_STEPS，停止（模型可能一直在调工具）")