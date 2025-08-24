from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


class GenericAgentState(TypedDict, ABC):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: dict


class GenericTools(ABC):
    """Base class for agent tools."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_tools(self) -> list[BaseTool]:
        """Return an assembled list of tools"""
        pass


class SearchTools(GenericTools):
    @tool
    def search(query: str) -> str:
        """Search something from database or API."""
        return f"Found info for {query}"

    def get_tools(self) -> list[BaseTool]:
        return [self.search]


@dataclass
class AgentFactory(ABC):
    model: BaseChatModel
    tools: list[BaseTool] = field(default_factory=list)
    agent_state: GenericAgentState = field(default_factory=dict)
    graph: StateGraph | None = None

    @abstractmethod
    def create_agent_graph(self):
        pass

    @abstractmethod
    def run_conversation(self):
        pass
