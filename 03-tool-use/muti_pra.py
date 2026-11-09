import json
from openai import OpenAI

with open("data.json","r",encoding="utf-8") as f:
    data = json.load(f)
    weather_data = data["weather"]
    attraction_data = data["attractions"]

#设计工具
#获取气温
def get_weather(city:str) -> dict:
    if city in weather_data:
        return {"city":city,
                "temperature":weather_data[city]["temperature"],
                "unit":weather_data[city]["unit"]}
    return {"city":city,"temperature":"Unknown","unit":"Unknown"}

def validate_get_weather(args: dict) -> str | None:
    if set(args) != {"city"}:
        return "get_weather 需要且仅需要参数city"
    if not isinstance(args["city"], str) or not args["city"].strip():
        return "city 必须是非空字符串"
    return None

#获取景点
def search_attractions(city:str) -> dict:
    if city in attraction_data:
        return {"city": city,
                "attractions": attraction_data[city]}
    return {"city":city,"attractions":"Unknown"}

def validate_search_attraction(args: dict) -> str | None:
    if set(args) != {"city"}:
        return "search_attractions 需要且仅需要参数city"
    if not isinstance(args["city"], str) or not args["city"].strip():
        return "city 必须是非空字符串"
    return None

#计算外汇
def convert_currency(amount,from_currency:str) -> float:
    if from_currency == "USD":
        return amount * 7.2
    return amount / 7.2

def validate_convert_currency(args: dict) -> str | None:
    if set(args) != {"from_currency", "amount"}:
        return "convert_currency 需要且仅需要from_currency, amount"
    if not isinstance(args["from_currency"], str) or not args["from_currency"].strip():
        return "from_currency必须是非空字符串"
    if args["amount"] <= 0.0:
        return "amount必须是非负数"
    return None


#分发
def dispatch(call) -> dict:
    name = call.function.name
    if name not in allow_tools:
        raise ValueError(f"未知工具：{name}")

    try:
        args = json.loads(call.function.arguments)
    except json.JSONDecodeError as e:
        return {"role": "tool", "tool_call_id": call.id,
                "content": json.dumps({"ok": False, "error": f"参数不是合法 JSON：{e}"}, ensure_ascii=False)}

    if not isinstance(args, dict):
        return {"role": "tool", "tool_call_id": call.id,"content":json.dumps({"ok": False, "error": "参数必须是对象"}, ensure_ascii=False)}

    #逐工具验证
    err = validate_tool[name](args)
    if err:
        return {"role": "tool", "tool_call_id": call.id,"content":json.dumps({"ok": False, "error": err}, ensure_ascii=False)}

    #调用工具
    res = allow_tools[name](**args)
    return {"role": "tool", "tool_call_id": call.id,"content":json.dumps(res, ensure_ascii=False)}

#工具定义函数
def _wrap(name: str, description: str, field: str, field_description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {field: {"type": "string", "description": field_description}},
                "required": [field],
                "additionalProperties": False,
            },
        },
    }

#定义工具
TOOLS = [
    _wrap("get_weather","查某个城市当前气温","city","市名，例如“东京”“北京”"),
    _wrap("search_attractions","查某个城市有哪些景点或活动","city","市名，例如“东京”“北京”"),
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "把一种货币金额换算成另一种货币",
            "parameters": {
                "type": "object",
                "properties": {"from_currency": {"type": "string", "enum": ["USD", "CNY"]},
                               "amount": {"type": "number", "description": "金额，必须大于0"}
                },
                "required": ["from_currency","amount"],
                "additionalProperties": False,
            },
        },
    }
]

#合法工具列表
allow_tools = {
    "get_weather":get_weather,
    "search_attractions":search_attractions,
    "convert_currency":convert_currency,
}

validate_tool = {
    "get_weather":validate_get_weather,
    "search_attractions":validate_search_attraction,
    "convert_currency":validate_convert_currency
}

#最大循环次数
MAX_LOOP = 10
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
while True:
    print("有什么需要帮助的吗：",end="")
    prompt = input()
    if type(prompt) != str:
        break
    messages = [{"role":"user","content":prompt}]

    for _ in range(MAX_LOOP):
        response = client.chat.completions.create(
            model="gemma4:e4b",
            messages=messages,
            tools=TOOLS
        )
        choice = response.choices[0]
        mesg = choice.message

        #追加对话
        messages.append(mesg.model_dump(exclude_none=True))

        #对话日志
        print(f"--- step{_+1} | finish_reason:{choice.finish_reason}")
        print("content",mesg.content)
        print("tool_call",mesg.tool_calls)

        if not mesg.tool_calls:
            print("\n开始回答：",mesg.content)
            break

        for call in mesg.tool_calls:
           try:
               tool_msg = dispatch(call)
           except ValueError as e:
               raise
           messages.append(tool_msg)

    else:
        print("达到 MAX_STEPS，停止（模型可能一直在调工具）")