import logging
from langchain_core.tools import tool
from langchain_google_community import GoogleSearchAPIWrapper
from src.config import SEARCH_API_KEY, SEARCH_CSE_ID

logger = logging.getLogger(__name__)

# 全局用户状态（实际应用中应通过参数传递）
user_states = {}

@tool
def search(query: str) -> str:
    """使用 Google 搜索获取实时信息。输入应为清晰的搜索查询。"""
    current_uid_state = user_states.get("current_uid", {})
    uid = current_uid_state.get("uid", "unknown")
    if not uid or uid == "unknown":
        logger.warning("Tool 'search' called without a valid UID.")
        if user_states:
            uid = next(iter(user_states)).get('uid', 'unknown')

    if uid != "unknown" and uid in user_states and query in user_states[uid].get("search_cache", {}):
        logger.info(f"Using cached result for query: {query}")
        return user_states[uid]["search_cache"][query]

    logger.info(f"Performing Google Search for query: {query}")
    search_api_wrapper = GoogleSearchAPIWrapper(
        google_api_key=SEARCH_API_KEY,
        google_cse_id=SEARCH_CSE_ID
    )
    try:
        results = search_api_wrapper.results(query, num_results=3)
        if not results:
            return "搜索没有返回结果。"
        formatted_result = "\n".join([
            f"Title: {res.get('title', 'N/A')}\nSnippet: {res.get('snippet', 'N/A')}\nLink: {res.get('link', 'N/A')}\n---"
            for res in results
        ])
        if uid != "unknown" and uid in user_states:
            if "search_cache" not in user_states[uid]:
                user_states[uid]["search_cache"] = {}
            user_states[uid]["search_cache"][query] = formatted_result
        return formatted_result
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        return f"哎呀，搜索出错了: {e}"

TOOLS = [search]