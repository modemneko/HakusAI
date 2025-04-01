import json
from aiohttp import web
import asyncio
import base64
import logging
from src.main import process_message, user_states
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def chat_handler(request: web.Request) -> web.Response:
    """处理 POST /chat 请求"""
    try:
        data = await request.json()
        message = data.get("message", "")
        uid = data.get("uid")
        api_key = data.get("api_key")

        if not uid:
            return web.json_response({"error": "请提供 'uid'"}, status=400)
        if not api_key:
            return web.json_response({"error": "请提供 'api_key'"}, status=400)

        img_data_list = []
        if "image" in data and data["image"]:
            try:
                img_base64 = data["image"].split(",")[1] if "," in data["image"] else data["image"]
                img_data = base64.b64decode(img_base64)
                img_data_list.append(img_data)
                logger.info(f"Received image data for UID {uid}, size: {len(img_data)} bytes")
            except Exception as e:
                return web.json_response({"error": f"Invalid image data: {e}"}, status=400)

        response = await asyncio.to_thread(process_message, uid, message, img_data_list, api_key)

        log_output = "=== 实时日志 ===\n"
        current_state = user_states.get(uid, {})
        
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

        return web.json_response({
            "response": response,
            "uid": uid,
            "log": log_output
        })

    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON format"}, status=400)
    except Exception as e:
        logger.error(f"Error in chat_handler: {e}", exc_info=True)
        return web.json_response({"error": str(e)}, status=500)

async def status_handler(request: web.Request) -> web.Response:
    """处理 GET /status 请求"""
    return web.json_response({
        "status": "ok",
        "message": "HakusAI API is running",
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

app = web.Application()
app.add_routes([
    web.post('/chat', chat_handler),
    web.get('/status', status_handler),
])

async def start_server():
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8950)
    await site.start()
    logger.info("HakusAI API server started at http://0.0.0.0:8950")
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(start_server())