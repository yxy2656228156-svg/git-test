import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI, DefaultHttpxClient # 1. 导入 DefaultHttpxClient

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    # 2. 关键：创建客户端并禁用环境/注册表代理探测
    http_client=DefaultHttpxClient(trust_env=False),
    timeout=600.0, # 3. 可选：为模型首次加载预留充足时间
)

r = client.chat.completions.create(
    model="gemma4:e4b",
    max_tokens=500,
    messages=[{"role": "user", "content": "用一句话自我介绍。"}],
)

# === 自我验证 ===
choice = r.choices[0]
text = choice.message.content or ""
print("响应：", text)
print("usage:", r.usage)

assert choice.finish_reason == "stop" and text.strip(), f"Expected complete non-empty response; got finish_reason={choice.finish_reason!r}"
assert r.usage.completion_tokens > 0, "output token 应大于 0"
print("✅ 练习 1 通过 — Ollama gemma4:e4b 已能在本地响应，每次 $0")