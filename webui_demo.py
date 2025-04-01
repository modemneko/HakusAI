import gradio as gr
from src.main import process_message
from src.state import State
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def chat_handler(message: str, image: bytes, history: list, state: State = None) -> tuple[str, bytes, list, str]:
    """处理用户输入并返回聊天历史和日志"""
    uid = "user123"
    img_data_list = [image] if image else []

    response = process_message(uid, message, img_data_list)
    logger.info(f"User {uid} sent: '{message}', Response: '{response[:50]}...'")

    if message or image:
        user_input = message
        if image:
            user_input += " [图片已上传]"
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

    from src.main import user_states
    current_state = user_states.get(uid, {})

    log_output = "=== 实时日志 ===\n"
    
    intermediate_steps = current_state.get("intermediate_steps", [])
    if intermediate_steps:
        log_output += "Observation:\n"
        for step in intermediate_steps:
            action, result = step
            log_output += f"- {action.tool}: {result[:100]}...\n"
    
    retrieved_memory = current_state.get("retrieved_memory", [])
    if retrieved_memory:
        log_output += "Memory:\n"
        for doc in retrieved_memory[:3]: 
            log_output += f"- {doc.page_content} (时间: {datetime.fromtimestamp(float(doc.metadata.get('timestamp', 0))).strftime('%Y-%m-%d %H:%M:%S')})\n"

    vector_store = current_state.get("vector_store", {}).get("memory")
    if vector_store:
        insights = vector_store.similarity_search("洞察", k=2, filter={"type": "insight"})
        reflections = vector_store.similarity_search("反思", k=2, filter={"type": "insight"})
        
        if insights:
            log_output += "Insights:\n"
            for insight in insights:
                log_output += f"- {insight.page_content}\n"
        
        if reflections:
            log_output += "Reflections:\n"
            for reflection in reflections:
                log_output += f"- {reflection.page_content}\n"
    
    if log_output == "=== 实时日志 ===\n":
        log_output += "暂无日志信息。\n"

    return "", None, history, log_output

with gr.Blocks(title="HakusAI - 小羽助手") as demo:
    gr.Markdown("# HakusAI - 小羽助手")
    gr.Markdown("跟小羽聊聊天吧！支持文本和图片输入，右侧显示实时日志。")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="聊天记录", type="messages", height=400)
            with gr.Row():
                msg = gr.Textbox(label="输入消息", placeholder="跟小羽说点啥吧...", scale=3)
                img = gr.Image(type="pil", label="上传图片", scale=1)
            submit_btn = gr.Button("发送")
        with gr.Column(scale=1):
            log_box = gr.Textbox(label="实时日志", lines=20, max_lines=20, interactive=False)

    submit_btn.click(
        fn=chat_handler,
        inputs=[msg, img, chatbot],
        outputs=[msg, img, chatbot, log_box]
    )
    msg.submit(
        fn=chat_handler,
        inputs=[msg, img, chatbot],
        outputs=[msg, img, chatbot, log_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7800, inbrowser=True)