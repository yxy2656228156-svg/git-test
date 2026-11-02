import sys
import time
from openai import OpenAI
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

client_local = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

with open('text_1_tech.txt', 'r' , encoding='utf-8') as f:
    text = f.read()

print("本地模型运行开始")
output_local = list()
t_l,i_l,o_l,p_l=0,0,0,0
for _ in range(5):
    mes_local = dict()
    start_local = time.time()
    result_local = client_local.chat.completions.create(
        model="gemma4:e4b",
        max_tokens=800,
        messages=[
        {"role": "system",
         "content": "你是一个专业的文本摘要助手。你只输出摘要本身，不添加任何解释、前言或客套话。"},
        {"role": "user",
         "content": ("请用不超过 50 字概括下面这段文本的核心内容，保留关键事实"
                     "（工具名称、端口号、接口类型、优势与代价）。\n\n"
                     "文本如下：\n<text>\n"
                     f"{text}\n"
                     "</text>")},
    ],
    )
    mes_local["回答次数"] = _ + 1
    mes_local["延迟"] = time.time() - start_local
    mes_local["input_token"] = result_local.usage.prompt_tokens
    mes_local["output_token"] = result_local.usage.completion_tokens
    mes_local["price"] = 0
    mes_local["summary"] = result_local.choices[0].message.content
    output_local.append(mes_local)
    t_l+=time.time() - start_local
    i_l+=result_local.usage.prompt_tokens
    o_l+=result_local.usage.completion_tokens

output_local.append({"平均延迟":t_l / 5 ,
                     "平均输入tokens":i_l / 5 ,
                     "平均输出tokens":o_l / 5 ,
                     "平均价格":p_l / 5 })
print("本地模型运行结束")

#todo: ----------------------------分割线---------------------------
PRICING_CNY = {
    "deepseek-flash":  {"input": 2.00, "output": 8.00},
}

client_dep = OpenAI(
    base_url="https://api.deepseek.com",
    api_key="",
)
MODEL = "deepseek-flash"

print("云端模型运行开始")
output_dep = list()
t_d,i_d,o_d,p_d=0,0,0,0
for _ in range(5):
    mes_dep = dict()
    start_dep = time.time()
    result_dep = client_dep.chat.completions.create(
        model=MODEL,
        max_tokens=800,
        messages=[
        {"role": "system",
         "content": "你是一个专业的文本摘要助手。你只输出摘要本身，不添加任何解释、前言或客套话。"},
        {"role": "user",
         "content": ("请用不超过 50 字概括下面这段文本的核心内容，保留关键事实"
                     "（工具名称、端口号、接口类型、优势与代价）。\n\n"
                     "文本如下：\n<text>\n"
                     f"{text}\n"
                     "</text>")},
    ],
    )
    mes_dep["回答次数"] = _ + 1
    mes_dep["延迟"] = time.time() - start_dep
    mes_dep["input_token"] = result_dep.usage.prompt_tokens
    mes_dep["output_token"] = result_dep.usage.completion_tokens
    mes_dep["price"] = (result_dep.usage.prompt_tokens * PRICING_CNY[MODEL]["input"] + result_dep.usage.completion_tokens * PRICING_CNY[MODEL]["output"]) / 1_000_000
    mes_dep["summary"] = result_dep.choices[0].message.content
    output_dep.append(mes_dep)
    t_d += time.time() - start_dep
    i_d += result_dep.usage.prompt_tokens
    o_d += result_dep.usage.completion_tokens
    p_d += (result_dep.usage.prompt_tokens * PRICING_CNY[MODEL]["input"] + result_dep.usage.completion_tokens * PRICING_CNY[MODEL]["output"]) / 1_000_000

output_dep.append({"平均延迟": t_d / 5,
                     "平均输入tokens": i_d / 5,
                     "平均输出tokens": o_d / 5,
                     "平均价格": p_d / 5})

with open('res.json', 'w', encoding='utf-8') as f:
    json.dump({"local": output_local, "cloud": output_dep}, f,
              ensure_ascii=False, indent=4)

print("云端模型运行结束")