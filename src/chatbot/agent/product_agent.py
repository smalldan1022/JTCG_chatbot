from dataclasses import dataclass, field
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from chatbot.agent.agent_factory import AgentFactory, GenericAgentState
from chatbot.tool.base_tool import ToolManager
from chatbot.tool.product_tool import ProductSearchTool, RequirementCheckerTool
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
        default_factory=lambda: {"configurable": {"thread_id": "2"}}
    )


class ProductAgent(AgentFactory):
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
        self.tool_manager.register_tool(RequirementCheckerTool())
        self.tool_manager.register_tool(ProductSearchTool(self.vec_db_manager))

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
        missing_example = {"missing": ["螢幕尺寸", "桌板厚度"]}

        system_msg = SystemMessage(
            content="""
        你是一個品牌智能助理。

        使用者輸入可能缺少資訊時：
        1. 使用工具 check_missing(query) 回傳缺少的欄位
        2. 若 missing 不為空，先回覆追問訊息
        3. 若 missing 為空，再使用 ProductSearchTool

        回覆內容：
        1. 針對使用者需求提供可行的產品建議
        2. 解釋關鍵依據（如規格相容、使用情境吻合），附商品頁連結與圖片
        3. 最後可加一句品牌介紹, 簡短即可, 例如: 'JTCG Shop 提供專業桌面配件，讓專注更持久'
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
    agent = ProductAgent()
    user_inputs = ["我叫 Dan", "雙螢幕氣壓臂我家桌子裝得下嗎?"]
    agent.run_conversation(user_inputs=user_inputs)
