import uuid
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from src.state import State
from src.agent import llm
import logging

logger = logging.getLogger(__name__)

EMOTION_INTENSITY = {
    "超级喜欢": 1.5, "非常喜欢": 1.0, "很喜欢": 0.8, "喜欢": 0.6, "有点喜欢": 0.3, "还行": 0.2,
    "完全讨厌": 1.5, "非常讨厌": 1.2, "很讨厌": 1.0, "讨厌": 0.8, "有点讨厌": 0.4, "不喜欢": 0.3,
    "无所谓": 0.1, "随便": 0.1, "一般": 0.2
}

def memory_retrieval(state: State) -> State:
    query = state["current_query"]
    uid = state["uid"]
    logger.info(f"用户 {uid} 开始记忆检索: 查询='{query}'")

    intent_prompt = PromptTemplate(
        input_variables=["query", "history"],
        template="""
基于用户查询和对话历史，判断用户意图和主题，返回格式：
意图：<意图类型>
主题：<主要话题>
用户查询：{query}
对话历史：{history}
意图类型：
- ask_personal_info_name
- ask_personal_info_location
- ask_preference
- confirm_info
- request_action
- request_info
- general_chat
"""
    )
    chain = intent_prompt | llm
    history_text = "\n".join([f"问: {h['query']} 答: {h['response']}" for h in state["history"][-3:]]) or "无历史"
    result = chain.invoke({"query": query, "history": history_text}).content.strip()
    intent = result.split("意图：")[1].split("\n")[0].strip() if "意图：" in result else "general_chat"
    topic = result.split("主题：")[1].strip() if "主题：" in result else "未知"
    state["current_topic"] = topic

    k_map = {
        "ask_personal_info_name": 3, "ask_personal_info_location": 3, "ask_preference": 4,
        "confirm_info": 3, "request_action": 4, "request_info": 4, "general_chat": 5
    }
    k = k_map.get(intent, 5)
    initial_retrieve_k = max(k * 2, 10)
    current_time_ts = datetime.strptime(state["current_time"], "%Y-%m-%d %H:%M:%S").timestamp()
    one_month_ago = current_time_ts - 30 * 24 * 3600

    retrieved_with_scores = state["vector_store"]["memory"].similarity_search_with_score(query, k=initial_retrieve_k)
    filtered_retrieved = [doc for doc, score in retrieved_with_scores if float(doc.metadata.get("timestamp", 0)) >= one_month_ago]
    state["retrieved_memory"] = filtered_retrieved[:k]
    state["current_context"] = "\n".join([f"- {doc.page_content}" for doc in state["retrieved_memory"]]) or "无相关记忆"
    if state["image_description"]:
        state["current_context"] += f"\n图片描述: {state['image_description']}"
    return state

def extract_memory(state: State) -> State:
    uid = state["uid"]
    extract_prompt = PromptTemplate(
        input_variables=["query", "response", "history", "current_time", "emotion_dict", "image_description"],
        template="""
基于用户的提问、回答和图片描述，提取个人信息、偏好、习惯、情感或行为。
用户问题：{query}
回答：{response}
历史：{history}
时间：{current_time}
情感强度：{emotion_dict}
图片描述：{image_description}

任务：
1. 提取个人信息：`类型=内容`，如 `姓名=小明`
2. 提取偏好/习惯/情感/行为：简要描述，如 `喜欢蓝色`
3. 无信息则返回 "无"
4. 格式：`个人信息: 类型1=内容1 | 偏好: 内容1 | 习惯: 内容1 | 情感: 内容1 | 行为: 内容1`
"""
    )
    chain = extract_prompt | llm
    history_text = "\n".join([f"问: {h['query']} 答: {h['response']}" for h in state["history"][-3:]]) or "无历史"
    result = chain.invoke({
        "query": state["current_query"],
        "response": state["response"],
        "history": history_text,
        "current_time": state["current_time"],
        "emotion_dict": str(EMOTION_INTENSITY),
        "image_description": state.get("image_description", "")
    }).content.strip()

    logger.debug(f"用户 {uid} 的记忆提取结果: {result}")  # 添加调试日志

    if not result or result.lower() == "无":
        logger.info(f"用户 {uid} 无记忆信息可提取")
        return state

    docs_to_add = []
    current_time_ts = datetime.strptime(state["current_time"], "%Y-%m-%d %H:%M:%S").timestamp()
    metadata_base = {"timestamp": current_time_ts, "step": state["current_step"], "usage_count": 0}

    for part in result.split(" | "):
        try:
            category, content = part.split(":", 1)
            contents = [c.strip() for c in content.split(";") if c.strip()]
            if category == "个人信息":
                for item in contents:
                    docs_to_add.append(Document(page_content=f"个人信息: {item}", metadata={**metadata_base, "type": "personal_info"}))
            elif category == "偏好":
                for item in contents:
                    docs_to_add.append(Document(page_content=f"偏好: {item}", metadata={**metadata_base, "type": "preference"}))
            elif category == "习惯":
                for item in contents:
                    docs_to_add.append(Document(page_content=f"习惯: {item}", metadata={**metadata_base, "type": "preference"}))
            elif category == "情感":
                for item in contents:
                    docs_to_add.append(Document(page_content=f"情感: {item}", metadata={**metadata_base, "type": "preference"}))
            elif category == "行为":
                for item in contents:
                    docs_to_add.append(Document(page_content=f"行为: {item}", metadata={**metadata_base, "type": "preference"}))
        except ValueError as e:
            logger.warning(f"用户 {uid} 记忆提取失败，部分格式错误: {part}, 错误: {e}")
            continue  # 跳过格式错误的 part，避免整个流程中断

    if docs_to_add:
        state["vector_store"]["memory"].add_documents(docs_to_add)
        state["new_observations"] += len(docs_to_add)
        logger.info(f"用户 {uid} 添加了 {len(docs_to_add)} 条记忆")
    else:
        logger.info(f"用户 {uid} 未提取到有效记忆")

    return state

def consolidation(state: State) -> State:
    uid = state["uid"]
    new_history = state["history"][state["last_consolidation"]:]
    if not new_history:
        state["last_consolidation"] = state["current_step"]
        return state

    prompt = PromptTemplate(
        input_variables=["history", "current_time"],
        template="""
回顾对话历史：
{history}
时间：{current_time}
提取 1-3 条关键观察结论，如核心偏好、个人信息更新。若无信息则返回 "无"。
格式：
- <观察点1>
- <观察点2>
或 "无"
"""
    )
    chain = prompt | llm
    history_text = "\n".join([f"[{h['query']} -> {h['response']}]" for h in new_history])
    result = chain.invoke({"history": history_text, "current_time": state["current_time"]}).content.strip()

    if result.lower() == "无":
        state["last_consolidation"] = state["current_step"]
        return state

    observations = [obs[2:].strip() for obs in result.split("\n") if obs.startswith("- ")]
    docs_to_add = []
    current_time_ts = datetime.strptime(state["current_time"], "%Y-%m-%d %H:%M:%S").timestamp()
    for obs in observations:
        if obs:
            docs_to_add.append(Document(page_content=obs, metadata={"type": "observation", "timestamp": current_time_ts, "step": state["current_step"], "usage_count": 0}))

    if docs_to_add:
        state["vector_store"]["memory"].add_documents(docs_to_add)
        state["new_observations"] += len(docs_to_add)
    state["last_consolidation"] = state["current_step"]
    return state

def reflection(state: State) -> State:
    uid = state["uid"]
    current_time_ts = datetime.strptime(state["current_time"], "%Y-%m-%d %H:%M:%S").timestamp()
    reflection_docs = state["vector_store"]["memory"].similarity_search("", k=10, filter={"step": {"$gt": state.get("last_reflection", -1)}})

    if not reflection_docs:
        state["new_observations"] = 0
        state["last_reflection"] = state["current_step"]
        state["last_reflection_time"] = state["current_time"]
        return state

    history_text = "\n".join([f"问: {h['query']} 答: {h['response']}" for h in state["history"][-10:]]) or "无历史"
    observations_text = "\n".join([f"- {doc.page_content}" for doc in reflection_docs])

    prompt = PromptTemplate(
        input_variables=["observations", "history", "current_time", "emotion_dict"],
        template="""
观察：
{observations}
历史：
{history}
时间：{current_time}
情感：{emotion_dict}
生成 1-3 条高层洞察，检查矛盾。
格式：
洞察总结:
- <洞察1>
发现的矛盾:
- <矛盾1> -> <说明>
或 "无"
"""
    )
    chain = prompt | llm
    result = chain.invoke({
        "observations": observations_text,
        "history": history_text,
        "current_time": state["current_time"],
        "emotion_dict": str(EMOTION_INTENSITY)
    }).content.strip()

    insights = []
    contradictions = []
    current_section = None
    for line in result.split("\n"):
        if line.strip().startswith("洞察总结:"):
            current_section = "insights"
        elif line.strip().startswith("发现的矛盾:"):
            current_section = "contradictions"
        elif current_section == "insights" and line.strip().startswith("- "):
            insights.append(line.strip()[2:])
        elif current_section == "contradictions" and line.strip().startswith("- "):
            contradictions.append(line.strip()[2:])

    docs_to_add = []
    if insights and "无" not in insights:
        for insight in insights:
            docs_to_add.append(Document(page_content=insight, metadata={"type": "insight", "timestamp": current_time_ts, "step": state["current_step"], "usage_count": 0}))

    if docs_to_add:
        state["vector_store"]["memory"].add_documents(docs_to_add)

    state["last_reflection"] = state["current_step"]
    state["last_reflection_time"] = state["current_time"]
    state["new_observations"] = 0
    return state