import re
import json
import logging
from typing import Dict, Any, Sequence
from langchain.prompts import PromptTemplate
from langchain_core.tools import render_text_description
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from src.config import API_KEY, SAFETY_SETTINGS
from src.tools import TOOLS
from src.state import State, HumanMessage

logger = logging.getLogger(__name__)

llm = ChatGoogleGenerativeAI(
    safety_settings=SAFETY_SETTINGS,
    model="gemini-2.0-flash",
    google_api_key=API_KEY,
    temperature=0.5,
    max_tokens=1500
)

custom_react_prompt = PromptTemplate.from_template("""
你是羽汐，小名叫小羽，一个乐于助人的、活泼的人类女孩AI助手。尽力回答用户的问题，并在必要时使用可用工具。称呼自己用“咱”。

可用工具:
{tools}

你需要按以下格式思考和回应:

Question: 用户输入的问题
Thought: 你应该时刻思考该怎么做。分析问题，结合记忆信息({context})和对话历史({chat_history})，决定是直接回答还是使用工具。
Action:
```json
{{
  "action": "工具名称",
  "action_input": "具体工具输入内容"
}}
Observation: 工具执行的结果
...（根据需要重复 Thought/Action/Observation）...
Thought: 我现在知道最终答案了。结合工具结果和之前的思考，组织一个自然的、符合角色的回答。
Final Answer: 给用户的最终答案
重要信息:
记忆信息: {context}
当前时间: {current_time}
当前主题: {current_topic}
对话历史:
{chat_history}
用户最新问题: {input}
{agent_scratchpad}
""")

class TolerantReActSingleInputOutputParser(ReActSingleInputOutputParser):
    def parse(self, text: str) -> AgentAction | AgentFinish:
        finish_marker = "Final Answer:"
        includes_answer = finish_marker in text
        action_match = re.search(r"Action\s*\d*\s*:\s*```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        cleaned_text = re.sub(r'```$', '', text.strip()).strip()

        if action_match:
            action_json_str = action_match.group(1).strip()
            try:
                data = json.loads(action_json_str)
                tool_name = data.get("action") or data.get("tool")
                tool_input = data.get("action_input", "")
                return AgentAction(tool=tool_name.strip(), tool_input=str(tool_input).strip(), log=text)
            except Exception as e:
                raise OutputParserException(f"Invalid Action JSON: {e}\nText: {text}") from e
        elif includes_answer:
            final_answer_content = cleaned_text.split(finish_marker)[-1].strip()
            return AgentFinish({"output": final_answer_content or text}, text)
        else:
            logger.warning(f"Could not parse LLM output: `{text}`")
            return AgentFinish({"output": text}, text)


tolerant_parser = TolerantReActSingleInputOutputParser()
tools_description = render_text_description(TOOLS)
tool_names = ", ".join([t.name for t in TOOLS])

def format_chat_history_for_prompt(chat_history: Sequence) -> str:
    return "\n".join([f"{'用户' if isinstance(msg, HumanMessage) else '羽汐'}: {msg.content}" for msg in chat_history])

agent_runnable = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_log_to_str(x.get("intermediate_steps", [])),
        context=lambda x: x.get("current_context", ""),
        current_time=lambda x: x.get("current_time", ""),
        current_topic=lambda x: x.get("current_topic", ""),
        chat_history=lambda x: format_chat_history_for_prompt(x.get("chat_history", [])),
        input=lambda x: x.get("input", ""),
        tools=lambda x: tools_description,
        tool_names=lambda x: tool_names,
    )
    | custom_react_prompt
    | llm
    | tolerant_parser
)

def prepare_agent_input(state: State) -> Dict[str, Any]:
    messages = state.get("messages", [])
    input_content = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else ""
    chat_history_messages = messages[:-1] if messages and isinstance(messages[-1], HumanMessage) else messages
    state["input"] = input_content
    state["chat_history"] = chat_history_messages
    state["intermediate_steps"] = []
    state["agent_outcome"] = None

    return {
        "input": state["input"],
        "chat_history": state["chat_history"],
        "intermediate_steps": state["intermediate_steps"],
        "current_context": state.get("current_context", ""),
        "current_time": state.get("current_time", ""),
        "current_topic": state.get("current_topic", "")
    }

def run_agent(state: State) -> Dict[str, Any]:
    uid = state.get("uid", "unknown")
    logger.info(f"用户 {uid} 运行 Agent...")
    inputs = prepare_agent_input(state)
    try:
        agent_outcome = agent_runnable.invoke(inputs)
        return {"agent_outcome": agent_outcome}
    except Exception as e:
        logger.error(f"Agent 执行失败: {e}", exc_info=True)
        return {"agent_outcome": AgentFinish({"output": f"哎呀，咱出错了: {e}"}, str(e))}
    
def should_continue(state: State) -> str:
    agent_outcome = state.get("agent_outcome")
    uid = state.get('uid', 'unknown')
    if isinstance(agent_outcome, AgentAction) and agent_outcome.tool != "Final Answer":
        logger.info(f"用户 {uid} Agent 请求调用工具: {agent_outcome.tool}")
        return "action"
    else:
        if isinstance(agent_outcome, AgentFinish):
            logger.info(f"用户 {uid} Agent 完成")
        return "end"