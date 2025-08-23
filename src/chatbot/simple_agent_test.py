from langchain.agents import AgentType, Tool, initialize_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import init_chat_model
from langchain.memory import ConversationBufferMemory
from langchain_tavily import TavilySearch
from utils.load_env import get_openai_api_key, get_tavily_api_key


# -----------------------------
# 自定義 Callback 來捕捉所有步驟
# -----------------------------
class DetailedCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.step_count = 0

    def on_agent_action(self, action, **kwargs):
        self.step_count += 1
        print(f"\n🔧 STEP {self.step_count}: ACTION")
        print(f"Tool: {action.tool}")
        print(f"Input: {action.tool_input}")
        print(f"Reasoning: {action.log}")
        print("-" * 50)

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"🛠️  TOOL EXECUTION: {serialized.get('name', 'Unknown')}")
        print(f"Input: {input_str}")

    def on_tool_end(self, output, **kwargs):
        print("📤 TOOL RESULT:")
        print(f"Output: {output}")
        print("-" * 50)

    def on_agent_finish(self, finish, **kwargs):
        print("✅ FINAL ANSWER:")
        print(f"Output: {finish.return_values}")
        print("=" * 60)


# -----------------------------
# 1️⃣ 取得 API key
# -----------------------------
api_key = get_openai_api_key()
tavily_api_key = get_tavily_api_key()

# -----------------------------
# 2️⃣ 初始化模型
# -----------------------------
model = init_chat_model(
    model="gpt-4o-mini", model_provider="openai", api_key=api_key, temperature=0, max_tokens=1000
)

# -----------------------------
# 3️⃣ 初始化工具
# -----------------------------
search_tool = TavilySearch(api_key=tavily_api_key, max_results=3)

tools = [
    Tool(
        name="Search",
        func=search_tool.run,
        description="Use this tool to search the web for current information. Input should be a search query.",
    )
]

# -----------------------------
# 4️⃣ 初始化記憶體
# -----------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -----------------------------
# 5️⃣ 建立 AgentExecutor（使用 callback）
# -----------------------------
callback_handler = DetailedCallbackHandler()

agent_executor = initialize_agent(
    tools=tools,
    llm=model,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    max_iterations=5,
    handle_parsing_errors=True,
    verbose=True,  # 啟用內建的 verbose
    callbacks=[callback_handler],  # 添加自定義 callback
)

# -----------------------------
# 6️⃣ 系統 prompt
# -----------------------------
system_message = """You are a helpful assistant. 
When you need current information, use the search tool.
Think step by step and show your reasoning process clearly."""

# -----------------------------
# 7️⃣ 多輪對話示例
# -----------------------------
inputs = [
    "Hi, I'm Bob and I live in SF.",
    "What's my name and where do I live?",
    "What's the current weather in the place I am living?",
]

for i, user_input in enumerate(inputs, 1):
    print(f"\n{'=' * 60}")
    print(f"CONVERSATION ROUND {i}")
    print(f"{'=' * 60}")
    print(f"👤 USER: {user_input}")
    print(f"{'=' * 60}")

    # 重置步驟計數器
    callback_handler.step_count = 0

    try:
        # 執行對話
        result = agent_executor.run(input=f"{system_message}\n\nUser: {user_input}")

        print("\n🤖 FINAL RESPONSE:")
        print(f"{result}")

    except Exception as e:
        print(f"❌ ERROR: {e}")

    print("\n📝 CURRENT MEMORY:")
    print(f"Chat History: {memory.chat_memory.messages}")
    print("\n" + "=" * 80 + "\n")

# -----------------------------
# 8️⃣ 額外：顯示完整記憶體內容
# -----------------------------
print("🧠 COMPLETE CONVERSATION MEMORY:")
for i, message in enumerate(memory.chat_memory.messages):
    print(f"{i + 1}. {message.__class__.__name__}: {message.content}")
