#ReAct提示,todo：包括system指令（智能体定义）；能够调用的工具；回答模板；历史记录
#todo: 这部分的ReAct的思路为，thinking -> Action -> Observation -> thinking
REACT_PROMPT_TEMPLATE ="""
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下：
{tools}

请严格按照以下格式进行回应:
Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请回答以下问题:
Question: {question}
History: {history}
"""

from LLM_factory.hello_agents_llm import HelloAgentsLLM
from Tool_factory.tool_executor import ToolExecutor
from Tool_factory.search_engine import search
import re

#todo:ReActAgent 的核心是一个循环，它不断地“格式化提示词 -> 调用LLM -> 执行动作 -> 整合结果”，直到任务完成或达到最大步数限制。
class ReActAgent:
    def __init__(self,llm_client:HelloAgentsLLM,tool_executor:ToolExecutor,max_steps:int = 6):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self,question:str):
        """
        运行ReAct智能体来回答一个问题。
        """
        self.history = [] #每次运行重置历史数据
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )

            # 调用LLM
            messages = [{"role":"user","content":prompt}]
            response_text = self.llm_client.think(messages=messages)

            if not response_text:
                raise ValueError("error:LLM未能返回有效响应。")

            #解析llm的回应
            thought, action = self._parse_output(response_text)
            if thought:
                print(f"🤔 思考: {thought}")
            if not action:
                raise ValueError("error:未能解析出有效的Action，流程终止。")

            #执行Action
            if action.startswith("Finish"):
                match = re.match(r"Finish\[(.*)\]", action, re.DOTALL)
                if match:
                    final_answer = match.group(1)
                    print(f"🎉 最终答案: {final_answer}")
                    return final_answer

            tool_name,tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                # ... 处理无效Action格式 ...
                continue
            print(f"🎬 行动: {tool_name}[{tool_input}]")

            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。"
            else:
                observation = tool_function(tool_input)

            #整合回答,放入history
            print(f"👀 观察: {observation}")
            self.history.append(f"Action: {action}")
            self.history.append(f"observation: {observation}")
        print("已达到最大步数，流程终止。")
        return None

    def _parse_output(self,text:str):
        """解析LLM的输出，提取Thought和Action。
        """
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)",text,re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$",text,re.DOTALL)
        thought = thought_match.group(1) if thought_match else None
        action = action_match.group(1) if action_match else None
        return thought, action

    def _parse_action(self,action_text:str):
        """
        Action回应模式为: Action: Finish[]  (str + [])
        """
        match = re.search(r"(\w+)\[(.*)\]",action_text,re.DOTALL) #todo：此处两个通配符，分别匹配tool_name and tool_input
        if match:
            return match.group(1), match.group(2)
        return None,None

if __name__=="__main__":
    llm = HelloAgentsLLM("gemma")
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("search", search_desc,search)
    react_agent = ReActAgent(llm,tool_executor)
    question = "华为最新的手机是哪一款？它的主要卖点是什么？"
    react_agent.run(question)