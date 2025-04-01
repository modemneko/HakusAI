from src.workflow import graph
from src.state import State, initialize_state
from src.agent import llm
from langchain_core.messages import HumanMessage, AIMessage
import logging

logger = logging.getLogger(__name__)
user_states = {}

def process_message(uid: str, message: str, img_data_list: list = []) -> str:
    if uid not in user_states:
        user_states[uid] = initialize_state(uid)
        logger.info(f"Initialized state for user {uid} via initialize_state")

    state = user_states[uid]
    state["current_query"] = message
    state["img_data_list"] = img_data_list
    state["messages"].append(HumanMessage(content=message))

    try:
        final_state = graph.invoke(state)
        user_states[uid] = final_state
        if not final_state["messages"] or not isinstance(final_state["messages"][-1], AIMessage):
            logger.error(f"Last message is not an AIMessage: {final_state['messages']}")
            return "处理出错: Agent 未正确响应"
        response = final_state["messages"][-1].content
        return response
    except Exception as e:
        logger.error(f"Graph 执行失败: {e}", exc_info=True)
        return f"处理出错: {str(e)}"