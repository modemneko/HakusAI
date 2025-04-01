from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from src.state import State, HumanMessage, AIMessage, AgentFinish
from src.agent import run_agent, should_continue
from src.memory import memory_retrieval, extract_memory, consolidation, reflection
from src.tools import TOOLS
import logging

logger = logging.getLogger(__name__)

def user_input(state: State) -> State:
    query_content = state["current_query"]
    full_content = query_content + (f" [图片描述: {state.get('image_description', '')}]" if state.get("image_description") else "")
    state["messages"] = state.get("messages", []) + [HumanMessage(content=full_content)]
    state["current_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state["short_term_memory"] = state["history"][-3:]
    return state

def history_storage(state: State) -> State:
    agent_outcome = state.get("agent_outcome")
    final_response = agent_outcome.return_values.get('output', "没找到答案") if isinstance(agent_outcome, AgentFinish) else state.get("response", "出错了")
    state["response"] = final_response
    state["messages"] = state.get("messages", []) + [AIMessage(content=final_response)]
    state["history"] = state.get("history", []) + [{"query": state["current_query"], "response": final_response}]
    state["current_step"] = state.get("current_step", 0) + 1
    return extract_memory(state)

def consolidation_trigger(state: State) -> str:
    steps_since_last = state["current_step"] - state["last_consolidation"]
    current_time = datetime.strptime(state["current_time"], "%Y-%m-%d %H:%M:%S")
    last_reflection_time = datetime.strptime(state["last_reflection_time"], "%Y-%m-%d %H:%M:%S")
    days_since_last = (current_time - last_reflection_time).days
    if steps_since_last >= 5 or days_since_last >= state.get("reflection_interval", 1):
        return "consolidation"
    return "check_reflection"

def reflection_trigger(state: State) -> str:
    if state.get("new_observations", 0) >= 3:
        return "reflection"
    return END

def build_react_graph():
    workflow = StateGraph(State)
    workflow.add_node("user_input", user_input)
    workflow.add_node("memory_retrieval", memory_retrieval)
    workflow.add_node("agent", run_agent)
    workflow.add_node("action", ToolNode(TOOLS))
    workflow.add_node("history_storage", history_storage)
    workflow.add_node("consolidation", consolidation)
    workflow.add_node("check_reflection", lambda state: state)
    workflow.add_node("reflection", reflection)

    workflow.set_entry_point("user_input")
    workflow.add_edge("user_input", "memory_retrieval")
    workflow.add_edge("memory_retrieval", "agent")
    workflow.add_conditional_edges("agent", should_continue, {"action": "action", "end": "history_storage"})
    workflow.add_edge("action", "agent")
    workflow.add_conditional_edges("history_storage", consolidation_trigger, {"consolidation": "consolidation", "check_reflection": "check_reflection"})
    workflow.add_edge("consolidation", "check_reflection")
    workflow.add_conditional_edges("check_reflection", reflection_trigger, {"reflection": "reflection", END: END})
    workflow.add_edge("reflection", END)

    return workflow.compile()

graph = build_react_graph()