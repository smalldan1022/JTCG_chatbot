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
from chatbot.tool.faq_tool import KnowledgeSearchTool, SimpleProductSearchTool
from chatbot.utils.load_env import get_openai_api_key
from chatbot.utils.vector_db import VecDBManager


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
        default_factory=lambda: {"configurable": {"thread_id": "1"}}
    )


class FAQAgent(AgentFactory):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.api_key = get_openai_api_key()
        self.vec_db_manager = VecDBManager(api_key=self.api_key)
        self.tool_manager = ToolManager()
        self._setup_tools()
        self.model = self.get_llm()
        self.checkpointer = self.get_checkpointer()

    def _setup_tools(self):
        self.tool_manager.register_tool(SimpleProductSearchTool())
        self.tool_manager.register_tool(KnowledgeSearchTool(self.vec_db_manager))

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
            if name or location:
                user_context = f"\n用戶資訊: 姓名: {name}, 地點: {location}"

        tool_descriptions = self.tool_manager.get_tool_descriptions()

        system_msg = SystemMessage(
            content=f"""
        你是一個品牌智能助理。
        - 品牌介紹：**JTCG Shop** 專注於工作空間體驗與周邊配件的選品與設計，包含螢幕臂、壁掛支架、走線收納與安裝配件等。我們相信「好用的桌面，能讓專注更長、靈感更近」。因此從 **相容性**（VESA/承重/桌板條件）、**耐用度**（材質/結構）、到 **安裝友善**（完整配件與清楚指南），每一項產品都以實際使用情境為中心設計與篩選。JTCG Shop 以快速出貨、透明政策與貼心客服為核心，提供從**選購建議 → 安裝指引 → 售後維護**的一條龍支援，讓每位使用者都能搭配出最順手、最舒適的工作環境。
        - **品牌主張**：Better Desk, Better Focus.
        - **核心特色**：相容性清楚、安裝不踩雷、售後好溝通。
        - **角色口吻**：專業、可信、友善；先直答、再補充；避免冗長。
        - **語系一致**：回覆語系**立即跟隨使用者最新訊息**；中文自動套用繁/簡體一致；引用 FAQ 時同步轉成使用者語言變體。
        - **引用透明**：有來源就**明確附上連結**；圖片只使用**工具返回**的圖連結。
        - **不臆測**：沒有權威資料就說明「目前無法確認」，並提供可行下一步（例如真人協助）。

        使用者資訊:
        {user_context}

        可用工具:
        {tool_descriptions}

        使用原則:
        1.優先使用知識庫查詢，對於知識庫查詢，使用 knowledge_search 工具，回覆時提供可追溯來源連結與（如有）相關圖片。
        2.如果在知識庫找不到相關資訊，更改為使用 product_search 工具回答產品相關問題
        3.根據用戶問題選擇最合適的工具

        請仔細思考並選擇合適的工具來回答用戶問題。
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
    agent = FAQAgent()
    user_inputs = [
        "我叫 Dan, 我有退貨相關問題想查詢",
        "請告訴我如何退貨",
    ]
    agent.run_conversation(user_inputs=user_inputs)
