--------------------------------主流程-------------------------
#ReAct流程固定的事件
① 发请求    messages（当前状态）+ tools → LLM
② 收响应    assistant message + finish_reason
③ 无条件记录 messages.append(assistant_entry)
④ 分支判断  有 tool_calls？没有 → 决定返回；有 → 执行
⑤ 逐个执行  每个 tc：执行 → messages.append(role="tool") → trace.append  ； 然后回到 ①