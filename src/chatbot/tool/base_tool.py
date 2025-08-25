from abc import ABC, abstractmethod

from langchain_core.tools import BaseTool


class BaseAgentTool(ABC):
    @abstractmethod
    def get_tool_name(self) -> str:
        pass

    @abstractmethod
    def get_tool_description(self) -> str:
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        pass

    def create_langchain_tool(self) -> BaseTool:
        return self._create_tool()

    @abstractmethod
    def _create_tool(self) -> BaseTool:
        pass


class ToolManager:
    def __init__(self):
        self.tools: list[BaseTool] = []
        self._tool_map = {}

    def register_tool(self, tool: BaseTool):
        self.tools.append(tool)
        self._tool_map[tool.get_tool_name()] = tool
        print(f"✅ Registered Tools: {tool.get_tool_name()}")

    def get_langchain_tools(self) -> list[BaseTool]:
        return [tool.create_langchain_tool() for tool in self.tools]

    def get_tool_descriptions(self) -> str:
        descriptions = []
        for tool in self.tools:
            descriptions.append(f"- {tool.get_tool_name()}: {tool.get_tool_description()}")
        return "\n".join(descriptions)

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        if tool_name in self._tool_map:
            return self._tool_map[tool_name].execute(**kwargs)
        else:
            return f"Can not find the tool: {tool_name}"
