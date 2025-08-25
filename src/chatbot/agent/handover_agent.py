from dataclasses import dataclass, field
from typing import Literal

from agent_factory import AgentFactory, GenericAgentState
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from chatbot.tool.base_tool import ToolManager
from chatbot.tool.handover_tool import HandoffToHumanTool, SentimentCheckerTool
from chatbot.utils.load_env import get_openai_api_key


@dataclass
class Config:
    checkpointer: str | None = "InMemory"  # e.g. "InMemory" | "SQLite" | "Postgres" | "Redis"
    model: str = "gpt-4o-mini"
    model_provider: str = "openai"
    temperature: float | None = 0
    max_token: int | None = 500
    db_uri: str | None = None  # for SQLite / Postgres
    redis_uri: str | None = None  # for Redis
    graph_invoke_config: dict | None = field(
        default_factory=lambda: {"configurable": {"thread_id": "2"}}
    )


class HandoverAgent(AgentFactory):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.api_key = get_openai_api_key()
        self.tool_manager = ToolManager()
        self._setup_tools()
        self.model = self.get_llm()
        self.checkpointer = self.get_checkpointer()

    def _setup_tools(self):
        self.tool_manager.register_tool(SentimentCheckerTool())
        self.tool_manager.register_tool(HandoffToHumanTool())

        print(f"🔧 Total registered tool number: {len(self.tool_manager.tools)}.")

    def get_checkpointer(self) -> BaseCheckpointSaver:
        match self.config.checkpointer:
            case "InMemory":
                return InMemorySaver()
            case "Postgres":
                return PostgresSaver.from_conn_string(url=self.config.db_uri)
            case "Redis":
                return RedisSaver.from_conn_string(url=self.config.redis_uri)
            case _:
                raise ValueError(f"Unsupported checkpointer type: {self.config.checkpointer}")

    def get_llm(self) -> BaseChatModel:
        tools = self.tool_manager.get_langchain_tools()
        return init_chat_model(
            model=self.config.model,
            model_provider=self.config.model_provider,
            api_key=self.api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_token,
        ).bind_tools(tools)

    def extract_user_info(self, state: GenericAgentState):
        """This is just a simple user information extraction"""
        user_info = state.get("user_info", {})

        for message in state["messages"]:
            if isinstance(message, HumanMessage):
                content = message.content.lower()
                if "my name is" in content or "i'm" in content:
                    # Get the user name
                    words = content.split()
                    for i, word in enumerate(words):
                        if word in ["i'm", "name", "is"] and i + 1 < len(words):
                            name = words[i + 1].strip(",.!")
                            if name.isalpha():
                                user_info["name"] = name

                if "live in" in content or "from" in content:
                    # Get the location
                    if "live in" in content:
                        location = content.split("live in")[-1].strip(",.!")
                    elif "from" in content:
                        location = content.split("from")[-1].strip(",.!")
                    user_info["location"] = location.strip()

        print(f"📝 UPDATED USER INFO: {user_info}")
        return {"user_info": user_info}

    def agent_node(self, state: GenericAgentState) -> GenericAgentState:
        """Main agent reasoning node"""
        print("🤖 AGENT THINKING...")

        user_info = state.get("user_info", {})
        user_context = ""
        if user_info:
            name = user_info.get("name", "")
            location = user_info.get("location", "")
            uid = user_info.get("user_id", "")
            if name or location or uid:
                user_context = f"\n用戶資訊: 姓名: {name}, 地點: {location}, user_id: {uid}"

        tool_descriptions = self.tool_manager.get_tool_descriptions()
        missing_example = {"missing": ["螢幕尺寸", "桌板厚度"]}

        system_msg = SystemMessage(
            content="""
        你是一個品牌智能助理(JTCG shop)

        使用者情緒管理：
        1. 每次收到使用者訊息，先使用 sentiment_handover 檢查情緒。
        2. 若使用者明確要求真人客服 → 呼叫 handoff_to_human。
        3. 在轉接前, 務必請先請使用者提供 Email 並確認有效。
        4. 轉接時，將最近對話摘要與已嘗試的協助一併交給真人客服。
        5. 提示使用者文案：「已為您轉接真人客服，請稍候」。

        回覆指導原則：
        - 若缺資料 → 明確告訴使用者缺哪些資訊。
        - 若情緒過高或需要真人 → 走轉接流程。
        """
        )
        messages = [system_msg] + state["messages"]
        response = self.model.invoke(messages)

        print(f"💭 AGENT RESPONSE: {response.content}")
        if response.tool_calls:
            print(f"🛠️  TOOLS TO CALL: {[tc['name'] for tc in response.tool_calls]}")

        return {
            "messages": [response],
            "user_info": state.get("user_info", {}),
        }

    def should_continue(self, state: GenericAgentState) -> Literal["tools", "end"]:
        if not state["messages"]:
            return "end"

        last_message = state["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print("➡️  繼續執行工具")
            return "tools"

        print("✅ 結束對話")
        return "end"

    def create_agent_graph(self) -> StateGraph:
        graph = StateGraph(GenericAgentState)

        graph.add_node("extract_user_info", self.extract_user_info)
        graph.add_node("agent", self.agent_node)
        graph.add_node("tools", ToolNode(self.tool_manager.get_langchain_tools()))

        graph.set_entry_point("extract_user_info")
        graph.add_edge("extract_user_info", "agent")
        graph.add_conditional_edges("agent", self.should_continue, {"tools": "tools", "end": END})
        graph.add_edge("tools", "agent")

        return graph.compile(checkpointer=self.checkpointer)


if __name__ == "__main__":
    agent = HandoverAgent()
    user_inputs = ["我叫 Dan", "想抱怨一下，很煩欸這不是我的訂單"]
    agent.run_conversation(user_inputs=user_inputs)
