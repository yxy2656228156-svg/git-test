import sys, os, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI

# DeepSeek 官方人民币定价（元 / 每 1M token）
# 取“空闲时段、缓存未命中”费率；高峰时段（工作日 9:00-12:00、14:00-18:00）价格翻倍
PRICING_CNY = {
    "deepseek-flash":  {"input": 1.00, "output": 4.00},   # 空闲时段，缓存未命中
    "deepseek-v4-pro": {"input": 4.50, "output": 13.50},  # 空闲时段，缓存未命中
}

client = OpenAI(
    api_key="#####",
    base_url="https://api.deepseek.com",
)
MODEL = "deepseek-flash"
RUNS = 5

latencies, out_tokens_list = [], []
total_cost_cny = 0.0

for i in range(RUNS):
    start = time.perf_counter()
    msg = client.chat.completions.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": "你好！请自我介绍一下。"}],
    )
    elapsed = time.perf_counter() - start

    in_tok = msg.usage.prompt_tokens
    out_tok = msg.usage.completion_tokens
    cost_cny = (in_tok * PRICING_CNY[MODEL]["input"] + out_tok * PRICING_CNY[MODEL]["output"]) / 1_000_000

    latencies.append(elapsed)
    out_tokens_list.append(out_tok)
    total_cost_cny += cost_cny

    print(f"[{i+1}/{RUNS}] {elapsed:.2f}s | input={in_tok} output={out_tok} | ¥{cost_cny:.6f}")

avg_latency = sum(latencies) / RUNS
avg_out = sum(out_tokens_list) / RUNS
throughput = avg_out / avg_latency

print(f"\n--- {RUNS} 次汇总（人民币计价）---")
print(f"平均延迟: {avg_latency:.2f} s")
print(f"平均输出: {avg_out:.1f} tokens")
print(f"平均吞吐: {throughput:.1f} tokens/s")
print(f"{RUNS} 次总成本: ¥{total_cost_cny:.6f}")
print(f"估算 1000 次成本: ¥{total_cost_cny / RUNS * 1000:.4f}")

assert total_cost_cny > 0, "Cloud LLM 调用应有正成本"
print(f"\n✅ 完成 — {RUNS} 次调用，总成本 ¥{total_cost_cny:.6f}")