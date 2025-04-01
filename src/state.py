import os
from typing import Dict, List, Any, TypedDict, Annotated, Sequence, Tuple, Union
from datetime import datetime
from langchain.schema import Document, HumanMessage, AIMessage
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.agents import AgentAction, AgentFinish
from src.config import API_KEY
import logging

logger = logging.getLogger(__name__)

class State(TypedDict):
    messages: Annotated[Sequence[Any], "add_messages"]
    history: List[Dict[str, str]]
    short_term_memory: List[Dict[str, str]]
    current_step: int
    last_consolidation: int
    new_observations: int
    last_reflection: int
    current_query: str
    response: str
    retrieved_memory: List[Document]
    vector_store: Dict[str, Chroma]
    uid: str
    search_cache: Dict[str, str]
    current_time: str
    last_reflection_time: str
    reflection_interval: int
    current_topic: str
    current_context: str
    img_data_list: List[bytes]
    image_description: str
    input: str
    chat_history: Sequence[Any]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[List[Tuple[AgentAction, str]], lambda x, y: x + y]

def initialize_state(uid: str) -> State:
    """初始化用户状态"""
    user_states: Dict[str, State] = {}
    if uid not in user_states:
        logger.info(f"Initializing state for new user {uid}")
        persist_dir = os.path.abspath(f"hakusai_memory_db/user_{uid}")
        os.makedirs(persist_dir, exist_ok=True)

        embedding_function = GoogleGenerativeAIEmbeddings(google_api_key=API_KEY, model="models/text-embedding-004")
        vector_store = Chroma(
            collection_name=f"user_{uid}_memory",
            embedding_function=embedding_function,
            persist_directory=persist_dir
        )

        current_time = datetime.now()
    return State(
        messages=[],
        history=[],
        short_term_memory=[],
        current_step=0,
        last_consolidation=0,
        new_observations=0,
        last_reflection=0,
        current_query="",
        response="",
        retrieved_memory=[],
        vector_store={"memory": vector_store},
        uid=uid,
        search_cache={},
        current_time=current_time.strftime("%Y-%m-%d %H:%M:%S"),
        last_reflection_time=current_time.strftime("%Y-%m-%d %H:%M:%S"),
        reflection_interval=1,
        current_topic="",
        current_context="",
        img_data_list=[],
        image_description="",
        input="",
        chat_history=[],
        agent_outcome=None,
        intermediate_steps=[]
    )