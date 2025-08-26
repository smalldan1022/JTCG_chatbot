from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool

from chatbot.tool.base_tool import BaseAgentTool
from chatbot.utils.vector_db import VecDBManager


class ProductSearchTool(BaseAgentTool):
    def __init__(self, vec_db_manager: VecDBManager):
        self.vec_db_manager = vec_db_manager
        csv_path = "chatbot/data/raw/ai-eng-test-sample-products.csv"
        self.vec_db_manager.init_from_csv(csv_path=csv_path)
        self.vec_db = self.vec_db_manager.vec_db

    def get_tool_name(self) -> str:
        return "knowledge_search"

    def get_tool_description(self) -> str:
        return "Search product information through database"

    def _format_documents(self, documents: list[Document]) -> str:
        if not documents:
            return "Documents not found"

        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            content = doc.page_content
            content_preview = content[:200] + "..." if len(content) > 200 else content

            metadata_info = ""
            if hasattr(doc, "metadata") and doc.metadata:
                title = doc.metadata.get("title", "")
                source = doc.metadata.get("source", "")
                if title:
                    metadata_info = f" [標題: {title}]"
                if source:
                    metadata_info += f" [來源: {source}]"

            formatted_docs.append(f"文件 {i}:{metadata_info}\n{content_preview}")

        return "\n\n".join(formatted_docs)

    def execute(self, query: str, k: int = 3) -> str:
        try:
            print(f"📚 Searching through knowledge database: {query}")
            results = self.vec_db.similarity_search(query, k=k)
            formatted_results = self._format_documents(results)
            return f"在知識庫中找到 {len(results)} 筆相關文件:\n\n{formatted_results}"
        except Exception as e:
            print(f"Searching error: {e}")
            return f"Searching error: {str(e)}"

    def _create_tool(self) -> BaseTool:
        @tool
        def product_search(query: str) -> str:
            """Search knowledge through database"""
            return self.execute(query=query)

        return product_search


class RequirementCheckerTool(BaseAgentTool):
    # We can extend the fields in the future
    REQUIRED_FIELDS: dict = {
        "氣壓臂": ["螢幕尺寸", "螢幕重量", "桌板厚度", "VESA孔距"],
        "壁掛支架": ["螢幕尺寸", "VESA孔距", "牆面材質"],
        "走線收納": ["桌面空間", "線材種類", "桌板尺寸"],
    }

    def get_tool_name(self) -> str:
        return "check_missing"

    def get_tool_description(self) -> str:
        return "Check if the information is adequate"

    def _call(self, query: str) -> dict:
        missing = set()

        for category, fields in self.REQUIRED_FIELDS.items():
            if category in query:
                for f in fields:
                    if f not in query:
                        missing.add(f)

        return {"missing": list(missing)}

    def execute(self, query: str) -> str:
        try:
            print(f"📚 Checking missing information: {query}")
            return self._call(query)

        except Exception as e:
            print(f"Checking error: {e}")
            return f"Checking error: {str(e)}"

    def _create_tool(self) -> BaseTool:
        @tool
        def check_missing(query: str) -> str:
            """Check if we miss some crucial information from users"""
            return self.execute(query=query)

        return check_missing
