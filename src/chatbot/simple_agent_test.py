from typing import Annotated, Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from utils.load_env import get_openai_api_key, get_tavily_api_key


# -----------------------------
# 1️⃣ 定義狀態
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: dict  # 儲存使用者資訊（姓名、地點等）


# -----------------------------
# 2️⃣ 初始化工具
# -----------------------------
api_key = get_openai_api_key()
tavily_api_key = get_tavily_api_key()

search_tool = TavilySearch(api_key=tavily_api_key, max_results=3)
checkpointer = InMemorySaver()


@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    try:
        results = search_tool.run(query)
        return f"Search results: {results}"
    except Exception as e:
        return f"Search failed: {str(e)}"


tools = [web_search]

# -----------------------------
# 3️⃣ 初始化模型
# -----------------------------
model = init_chat_model(
    model="gpt-4o-mini", model_provider="openai", api_key=api_key, temperature=0
).bind_tools(tools)


# -----------------------------
# 4️⃣ 定義節點函數
# -----------------------------
def agent_node(state: AgentState) -> AgentState:
    """主要的 agent 推理節點"""
    print("🤖 AGENT THINKING...")

    # 系統訊息
    system_msg = SystemMessage(
        content="""
    You are a helpful assistant. Remember user information from the conversation.
    When you need current information, use the web_search tool.
    Think step by step and show your reasoning clearly.
    
    Always check if you have the information needed before using tools.
    """
    )

    # 建構訊息列表
    messages = [system_msg] + state["messages"]

    # 調用模型
    response = model.invoke(messages)

    print(f"💭 AGENT RESPONSE: {response.content}")
    if response.tool_calls:
        print(f"🛠️  TOOLS TO CALL: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": [response]}


def extract_user_info(state: AgentState) -> AgentState:
    """提取並記憶使用者資訊"""
    user_info = state.get("user_info", {})

    # 簡單的資訊提取邏輯
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            content = message.content.lower()
            if "my name is" in content or "i'm" in content:
                # 提取姓名
                words = content.split()
                for i, word in enumerate(words):
                    if word in ["i'm", "name", "is"] and i + 1 < len(words):
                        name = words[i + 1].strip(",.!")
                        if name.isalpha():
                            user_info["name"] = name

            if "live in" in content or "from" in content:
                # 提取地點
                if "live in" in content:
                    location = content.split("live in")[-1].strip(",.!")
                elif "from" in content:
                    location = content.split("from")[-1].strip(",.!")
                user_info["location"] = location.strip()

    print(f"📝 UPDATED USER INFO: {user_info}")
    return {"user_info": user_info}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """決定是否需要使用工具"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("➡️  CONTINUING TO TOOLS")
        return "tools"
    print("✅ ENDING CONVERSATION")
    return "end"


# -----------------------------
# 5️⃣ 建立圖
# -----------------------------
def create_agent_graph():
    workflow = StateGraph(AgentState)

    # 添加節點
    workflow.add_node("extract_info", extract_user_info)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    # 定義邊
    workflow.set_entry_point("extract_info")
    workflow.add_edge("extract_info", "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=checkpointer)


# -----------------------------
# 6️⃣ 執行對話
# -----------------------------
def run_conversation():
    app = create_agent_graph()

    # 初始狀態
    initial_state = {"messages": [], "user_info": {}}

    inputs = [
        "Hi, I'm Bob and I live in SF.",
        "What's my name and where do I live?",
        "What's the current weather in the place I am living?",
    ]

    current_state = initial_state

    for i, user_input in enumerate(inputs, 1):
        print(f"\n{'=' * 70}")
        print(f"CONVERSATION ROUND {i}")
        print(f"{'=' * 70}")
        print(f"👤 USER: {user_input}")
        print("-" * 50)

        # 添加使用者訊息
        current_state["messages"].append(HumanMessage(content=user_input))

        # 執行圖並顯示每個步驟
        step_count = 0
        for step in app.stream(
            current_state, config={"configurable": {"thread_id": "1"}}, stream_mode="updates"
        ):
            step_count += 1
            print(f"\n📋 STEP {step_count}:")

            for node_name, node_output in step.items():
                print(f"  🔹 NODE: {node_name}")

                # 更新狀態
                current_state.update(node_output)

                # 顯示新訊息
                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        if isinstance(msg, AIMessage):
                            print(f"    🤖 AI: {msg.content}")
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    print(f"    🔧 TOOL CALL: {tc['name']} with {tc['args']}")
                        elif isinstance(msg, ToolMessage):
                            print(f"    🛠️  TOOL RESULT: {msg.content[:200]}...")

                if "user_info" in node_output:
                    print(f"    📝 USER INFO: {node_output['user_info']}")

        print("\n💾 FINAL STATE:")
        print(f"   Messages: {len(current_state['messages'])}")
        print(f"   User Info: {current_state['user_info']}")
        print(f"\n{'=' * 70}")


# -----------------------------
# 7️⃣ 另一個版本：更詳細的 streaming
# -----------------------------
def run_conversation_detailed_stream():
    """使用更詳細的 streaming 模式"""
    app = create_agent_graph()

    initial_state = {"messages": [], "user_info": {}}

    inputs = [
        "Hi, I'm Bob and I live in SF.",
        "What's my name and where do I live?",
        "What's the current weather in the place I am living?",
    ]

    current_state = initial_state

    for user_input in inputs:
        print(f"\n👤 USER: {user_input}")
        print("=" * 60)

        current_state["messages"].append(HumanMessage(content=user_input))

        # 使用 debug stream 模式
        for chunk in app.stream(
            current_state,
            stream_mode="debug",  # 顯示更多細節
            debug=True,
        ):
            print(f"📦 CHUNK: {chunk}")

        print("=" * 60)


if __name__ == "__main__":
    print("🚀 Starting LangGraph Agent...")
    run_conversation()
