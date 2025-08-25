from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Annotated

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class GenericAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: dict


@dataclass
class AgentFactory(ABC):
    model: BaseChatModel | None = None
    tools: list[BaseTool] = field(default_factory=list)
    agent_state: GenericAgentState = field(default_factory=dict)
    graph: StateGraph | None = None

    @abstractmethod
    def create_agent_graph(self):
        pass

    @abstractmethod
    def get_checkpointer(self):
        pass

    @abstractmethod
    def get_llm(self):
        pass

    @abstractmethod
    def run_conversation(self):
        pass
