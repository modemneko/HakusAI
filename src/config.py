import os
import google.generativeai as genai
from google.generativeai import types as genai_types

# API 密钥配置
API_KEY = ""
SEARCH_API_KEY = ""
SEARCH_CSE_ID = ""

# 配置 Google Generative AI
genai.configure(api_key=API_KEY)

# 安全设置
SAFETY_SETTINGS = {
    genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai_types.HarmBlockThreshold.BLOCK_NONE,
    genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai_types.HarmBlockThreshold.BLOCK_NONE,
    genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai_types.HarmBlockThreshold.BLOCK_NONE,
    genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai_types.HarmBlockThreshold.BLOCK_NONE,
}

# 可选：代理设置（根据需要启用）
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"