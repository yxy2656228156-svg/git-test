#todo: json
'''import json
import time

data = {"model": "deepseek-pro" , "runs":5 , "tags":["test","本地"]}

start = time.time()
with open("config.json","r",encoding="utf-8") as f:
    config = json.load(f)
print(config)

with open("config.json","a",encoding="utf-8") as f:
    json.dump(data,f,ensure_ascii=False,indent=2)'''

#todo: txt
'''#整文件读取
with open("test_txt1.txt","r",encoding="utf-8") as f:
    data1 = f.read()
print("以下内容为prompt示例文本：\n" + data1)
print("-----"*5)

#逐行读取
with open("test_txt2.txt","r",encoding="utf-8") as f:
    print("以下内容为用户问题：")
    for line in f:
        if line:
            print(line,end="")'''

#todo: API
import os
import requests

r = requests.get("https://jsonplaceholder.typicode.com/posts/1", timeout=10)
r.raise_for_status()          # 非 2xx 直接抛异常，比手动判断 status_code 省事
data = r.json()               # 自动解析 JSON 响应
print(data["title"])

api_key = ""
if not api_key:
    raise SystemExit("请先设置环境变量 DEEPSEEK_API_KEY")

url = "https://api.deepseek.com/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}
payload = {
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "用一句话解释什么是 API"}],
}

try:
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    resp = r.json()
    """
    resp示例：
    {'id': 'dc97fa20-21af-4d8a-b6b7-155039ac8d9e',
     'object': 'chat.completion',
      'created': 1793607461, 
      'model': 'deepseek-flash', 
      'choices': [{'index': 0, 
      'message': {'role': 'assistant', 
      'content': 'API 就像餐厅的服务员：你（程序）把需求告诉服务员（API），服务员去厨房（服务器）下单，再把结果端回来给你——它是一套让不同软件之间互相“点单”和“上菜”的约定。'}, 'logprobs': None, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 9, 'completion_tokens': 52, 'total_tokens': 61, 'prompt_tokens_details': {'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 9}, 'system_fingerprint': 'bc19ba97cbe016479903222bb1578db6'}
    """

    print(resp["choices"][0]["message"]["content"])
except requests.exceptions.Timeout:
    print("请求超时，检查网络或加大 timeout")
except requests.exceptions.HTTPError as e:
    print("HTTP 错误：", r.status_code)
    print(r.text)
except requests.exceptions.RequestException as e:
    print("请求失败：", e)