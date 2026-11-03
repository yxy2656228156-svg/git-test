import json
from openai import OpenAI

with open("data.json","r",encoding="utf-8") as f:
    datas = json.load(f)

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=""
)

score = 0
result = dict()
for data in datas:
    mes = data["text"]
    prompt = f"""
    目标：在资料中有客户向客服发送的客服留言，现在需要对客服留言信息进性分类，来判断客户遇到的是什么类型的问题，将客服留言分到 billing（账单中的服务、费用变化等问题）、bug（系统问题导致出现的卡顿、无法加载等问题）或 other（非billing以及bug问题）。
    资料：<input_data>{mes}</input_data>
    规则：只根据资料分类；不知道时选 other。判断时参考以下范例：
        输入：信用卡又扣了一次
        输出：billing
        
        输入：提交表单后没有反应
        输出：bug
        
        输入：可以更改联系邮箱吗
        输出：other
    输出：只返回一个小写标签。"""

    reply = client.chat.completions.create(
        model="deepseek-flash",
        max_tokens=5000,
        messages = [{"role": "user" , "content": prompt}],
        temperature= 0.5
    )
    res = reply.choices[0].message.content
    result[mes] = res

    if res == data["label"]:
        score += 1

print(result)
print(f'''
    原版｜改了什么：没有改｜分数：2 / 6
    新版｜改了什么：重新写了目标，让目标更清楚｜分数：{score} / 6
    ''')