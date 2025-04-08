# HakusAI

**HakusAI** 是一个基于 LangChain 和 LangGraph 构建的智能 AI 助手，化身为活泼的“羽汐”（由提示词性格设定）。它具备一定的学习能力，能记住用户的偏好、爱好等个人信息，旨在为用户提供个性化、智能化的交互体验。

望大佬们指点江山，提出改进建议

## 功能特性

- **个性化对话**：以“羽汐”的活泼角色与用户互动，使用“咱”自称，语言自然亲切。
- **记忆管理**：
  - 短期记忆：保留最近对话上下文。
  - 长期记忆：通过向量数据库存储用户信息、偏好和洞察。
  - 巩固与反思：定期整理对话历史，提取关键观察并检测矛盾。
- **工具集成**：内置 Google 搜索工具，支持实时信息获取，可扩展其他工具。
- **图片处理**：支持上传图片并生成描述（依赖 Gemini API）。
- **状态持久化**：用户状态存储在本地文件系统，确保对话连续性。

## 技术栈

- **Python**: 3.9+
- **LangChain**: 用于构建 Agent 和工具链。
- **LangGraph**: 管理对话工作流。
- **Google Generative AI**: 提供 LLM 和嵌入支持。
- **Chroma**: 向量数据库，用于记忆存储。
- **Google Search API**: 实时搜索功能。

## Graph 流程
```bash
graph TD
    A[用户输入] --> B[记忆检索]
    B --> C[准备 Agent 输入]
    C --> D[运行 ReAct Agent]
    D -->|需要工具| E[执行工具]
    E --> D
    D -->|完成| F[存储历史]
    F --> G{需要巩固?}
    G -->|是| H[记忆巩固]
    G -->|否| I{需要反思?}
    H --> I
    I -->|是| J[记忆反思]
    I -->|否| K[结束]
    J --> K
   ```
### 工作流程说明
1. **用户输入**：接收用户的问题或指令。
2. **记忆检索**：从短期和长期记忆中提取相关信息。
3. **准备 Agent 输入**：整合上下文和历史，准备给 ReAct Agent。
4. **运行 ReAct Agent**：通过思考、行动、观察循环生成回答，可能调用外部工具。
5. **存储历史**：将对话存入历史记录。
6. **记忆巩固**：定期总结对话，提取关键观察。
7. **记忆反思**：分析积累的观察，生成深层洞察。

### 记忆检索节点

记忆检索节点负责从向量数据库（Chroma）中提取与用户查询最相关的信息。它结合了语义相似性搜索、意图识别和动态过滤机制，确保返回的信息既准确又具有时效性。以下是技术细节：

- **意图与主题识别**：
  - 使用 LLM（Google Gemini）通过提示模板分析用户查询和最近三轮对话历史，识别意图（如 `ask_preference`、`request_info`）和主题。
  - 示例输出：`意图：request_info 主题：天气`，为后续检索提供上下文。

- **分层检索策略**：
  - **初步检索**：基于查询，使用 Chroma 的 `similarity_search_with_score` 方法，结合 Google Generative AI 的嵌入模型（`text-embedding-004`），执行向量相似性搜索。初始检索数量（`k`）动态调整，例如意图为 `general_chat` 时 `k=10`，其他特定意图时根据需求调整。
  - **类型过滤**：根据意图应用元数据过滤。例如，`ask_preference` 只检索类型为 `preference` 的文档，`request_info` 优先检索 `dialogue` 类型。

- **时间与权重优化**：
  - **时间过滤**：仅保留最近一个月（30天）的记忆，基于文档元数据中的 `timestamp` 与当前时间的差值。
  - **权重计算**：
    - **相似性得分**：由向量余弦相似度计算（Chroma 返回的 `score`）。
    - **时间衰减**：使用公式 `time_decay = 0.95 ^ (days_diff)`，其中 `days_diff` 是文档创建时间与当前时间的差值（天数），使较旧的记忆权重逐渐降低。
    - **使用频率提升**：通过 `usage_boost = 1 + usage_count / 10`，增加经常被引用的记忆权重。
    - **综合权重**：`weight = similarity_score * time_decay * usage_boost`，综合考虑相似性、时效性和重要性。
  - **阈值筛选**：设置权重阈值（如 0.3），仅保留权重高于阈值的文档，并按权重排序，取前 `k` 个（`k` 由意图决定，如 `ask_preference` 取 4）。

- **上下文整合**：
  - 将短期记忆（最近三轮对话）和长期记忆（检索结果）整合为上下文。
  - 示例上下文格式：
    ```
    短期记忆:
    问: 你喜欢啥颜色？ 答: 咱喜欢蓝色喵～
    长期记忆 (相关性最高):
    - 偏好: 喜欢蓝色 (权重: 0.85)
    - 对话: 用户问喜欢的颜色，回答蓝色 (权重: 0.62)
    ```

- **使用频率更新**：
  - 被检索到的长期记忆文档，其 `usage_count` 增加 1，并通过 Chroma 的 `update_document` 方法更新数据库，提升其未来被选中的概率。

- **技术实现**：
  - **嵌入模型**：Google 的 `text-embedding-004`，将查询和记忆文本转化为 768 维向量。
  - **向量数据库**：Chroma，使用 HNSW（Hierarchical Navigable Small World）算法进行高效近似最近邻搜索。
  - **异常处理**：若检索失败或无结果，返回空记忆并依赖短期上下文回答。

这种检索机制确保了系统能够快速定位最相关的记忆，同时兼顾信息的时效性和用户交互历史的重要性。

## 安装

### 前置条件
- Python 3.9 或更高版本
- Git

### 步骤
1. 克隆仓库：
   ```bash
   git clone https://github.com/modemneko/HakusAI.git
   cd HakusAI
   ```
2. 创建并激活虚拟环境（可选但推荐）：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. 安装依赖：
   ```bash
   pip install -U -r requirements.txt
   ```
4. 配置密钥：
   - Gemini API密钥：
      1. 访问[Google AI Studio](https://aistudio.google.com/apikey?)。
      2. 创建API密钥。
   - Google Search Engine密钥及引擎ID：
      1. 参考[Custom Search JSON API](https://developers.google.com/custom-search/v1/overview?hl=zh-cn)，获取密钥和自定义搜索引擎ID
   - 设置`src/config.py`文件中`API_KEY`和`SEARCH_API_KEY`、`SEARCH_CSE_ID`参数配置

## 运行程序
1. 确保密钥配置完成。
2. 在项目根目录下运行主脚本：
   ```bash
   python webui_demo.py
   ```
   - 浏览器会自动打开，默认地址为`http://localhost7800`

## API
- HakusAI 提供 RESTful API 用于与AI交互。
- 主要端点为 POST /chat，接收 JSON 请求体（包含 `message`（消息）、`uid`（用户ID，必填）、`api_key`（Google API 密钥，必填）、`image`（可选 base64 图片）），返回 JSON 响应（包含 `response`（羽汐回复）、`uid` 和 `log`（实时日志））。
- 示例：
```bash
curl -X POST http://localhost:8950/chat -H "Content-Type: application/json" -d '{"message": "今天天气咋样？", "uid": "user1", "api_key": "YOUR_API_KEY"}'。
```
另有 `GET /status` 检查服务器状态。



## 贡献
欢迎提交Issue或Pull Request到[Github仓库](https://github.com/modemneko/HakusAI.git)。请遵循以下步骤：
1. Fork 仓库。
2. 创建你的功能分支（`git checkout -b feature/xxx`）。
3. 提交更改（`git commit -m "Add xxx feature"`）。
4. 推送到分支（`git push origin feature/xxx`）。
5. 创建 Pull Request。

## 许可
本项目采用MIT许可证。详情见[LICENCE](https://github.com/modemneko/HakusAI/blob/main/LICENSE)文件。
