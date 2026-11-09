import json
from openai import OpenAI

#todo 目标：构建一个查阅天气的单工具非调用api的Agent

#工具
def get_weather(city:str , unit:str) -> dict:
    #资料表
    datas = {
        "北京": 10,
        "武汉":20,
        "广东":25
    }

    #判断输入数据是否正确
    if type(city) is not str or unit != "celsius":
        raise ValueError("输入数据错误，检查city是否为str，或者unit是否为celsius!")

    return {"city":city,"temperature":datas[city],"unit":unit}

#构建client
client = OpenAI(base_url="http://localhost:11434/v1",api_key="ollama")

#工具定义
TOOLS = [
    {
        "type": "function",
        "function":{
            "name": "get_weather",
            "description":"用于获取需要查阅的城市的当天气温",
            "parameters":{
                "type":"object",
                "properties":{
                    "city": {"type":"string" , "description":"城市名称，例如北京"},
                    "unit": {"type":"string" , "enum":["celsius"]}, #todo：此处的enum为schema字段，后必须跟非空列表，列举出所有合法取值
                },
                "required": ["city","unit"],
                "additionalProperties":False,
            },
        },
    }
]

#历史记录
messages = [{"role":"user" , "content":"武汉现在几度？"}]
#第一次调用llm
first = client.chat.completions.create(
    model = "qwen2.5:3b" , messages = messages , tools = TOOLS
)
#获取llm的回答
assistant = first.choices[0].message
messages.append(assistant.model_dump(exclude_none=True))

#遍历工具列表，依次调用工具
for call in assistant.tool_calls or []:
    if call.function.name != "get_weather":
        raise ValueError(f"未知工具:{call.funtion.name}")
    args = json.loads(call.function.arguments)

    #判断输入参数是否合法
    if (
        not isinstance(args,dict)
        or set(args) != {"city","unit"}
        or not isinstance(args["city"],str)
        or not isinstance(args["unit"],str)
        or args["unit"] != "celsius"
    ):
        raise ValueError("city 必须是非空字符串，unit 必须是 celsius，并且传入参数只有city和unit")

    result = get_weather(args["city"],args["unit"])
    messages.append(
        {"role":"tool","tool_call_id":call.id,"content":json.dumps(result,ensure_ascii=False),}
    )

    if not assistant.tool_calls:
        raise ValueError("模型没有调用工具，检查模型与schema")

    #回答llm
    final = client.chat.completions.create(
        model = "qwen2.5:3b", messages = messages ,tools = TOOLS
    )

    print(final.choices[0].message.content)

assert assistant.tool_calls[0].function.name == "get_weather"
assert any(message["role"] == "tool" for message in messages)