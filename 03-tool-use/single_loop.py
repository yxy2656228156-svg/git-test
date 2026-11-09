import json
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

def get_temperature(city: str) -> str:
    """Get the current temperature for a city"""
    temperatures = {
        "New York": "22°C",
        "London": "15°C",
        "Tokyo": "18°C",
    }
    return temperatures.get(city, "Unknown")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_temperature",
        "description": "Get the current temperature for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city, e.g. New York",
                },
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}]

messages = [{"role": "user", "content": "What is the temperature in New York?"}]

first = client.chat.completions.create(
    model="qwen2.5:3b",
    messages=messages,
    tools=TOOLS,
)

assistant = first.choices[0].message
""" 数据情况如下(注意assistance是类，在访问数据时，应当遵循类的规则）
{
    "role": "assistant",
    “content”：None
    "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_temperature",
                "arguments": "{\"city\": \"New York\"}",
            },
        }
    ],
}
"""
messages.append(assistant.model_dump(exclude_none=True)) #todo: 将数据转为字典，转换后数据形式如下，exclude_none=True，则是删除所有None字段数据
"""此时经过model_dump转换之后，得到的数据为字典类型
{
    "role": "assistant",
    "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_temperature",
                "arguments": "{\"city\": \"New York\"}",
            },
        }
    ],
}
"""

if assistant.tool_calls:
    call = assistant.tool_calls[0]
    args = json.loads(call.function.arguments)   #todo: json.loads控制格式为字符串
    result = get_temperature(**args) #todo：**args是字典解包，args："{\"city\": \"New York\"}"，解析为：city=“NEW YORK”
    messages.append({
        "role": "tool",
        "tool_call_id": call.id,                 # 注意：不是 tool_name
        "content": str(result),
    })

    #todo：单loop指的是依次工具调用不是一个api请求，完整流程：请求1 → 模型要工具 → 执行 → 请求2 → 模型给答案
    final = client.chat.completions.create(
        model="qwen2.5:3b",
        messages=messages,
        tools=TOOLS,
    )
    print(final.choices[0].message.content)
else:
    print("模型没调工具，直接回答：", assistant.content)