from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Annotated

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
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

    def run_conversation(self, user_inputs: list[str]):
        if not self.graph:
            self.graph = self.create_agent_graph()

        initial_state = {"messages": [], "user_info": {}}
        current_state = initial_state
        last_ai_message = None

        for i, user_input in enumerate(user_inputs):
            print(f"\n{'=' * 70}")
            print(f"CONVERSATION ROUND {i}")
            print(f"{'=' * 70}")
            print(f"👤 USER: {user_input}")
            print("-" * 50)
            current_state["messages"].append(HumanMessage(content=user_input))

            step_count = 0
            for step in self.graph.stream(
                current_state, config=self.config.graph_invoke_config, stream_mode="updates"
            ):
                step_count += 1
                print(f"\n📋 STEP {step_count}:")

                for node_name, node_output in step.items():
                    print(f"  🔹 NODE: {node_name}")
                    current_state.update(node_output)
                    if "messages" in node_output:
                        for msg in node_output["messages"]:
                            if isinstance(msg, AIMessage):
                                last_ai_message = msg.content
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
        return last_ai_message
