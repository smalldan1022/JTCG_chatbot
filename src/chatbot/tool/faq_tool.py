import pandas as pd
from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool

from chatbot.tool.base_tool import BaseAgentTool
from chatbot.utils.vector_db import VecDBManager


class ProductSearchTool(BaseAgentTool):
    name: str = "product_search"
    description: str = (
        "Search relevant products for the user's query and provide links and brief descriptions."
    )

    def __init__(self):
        self.product_df = pd.read_csv(
            "/home/smalldan/chatbot/data/raw/ai-eng-test-sample-products.csv"
        )

    def get_tool_name(self) -> str:
        return "product_search"

    def get_tool_description(self) -> str:
        return "Search for product information and return relevant product results based on the user’s query."

    def run(self, query: str):
        # We can use simple word match or vector search
        results = []
        for _, row in self.product_df.iterrows():
            if (
                query.lower() in row["name"].lower()
                or query.lower() in row["compatibility_notes"].lower()
            ):
                results.append(
                    {
                        "title": row["name"],
                        "compatibility_notes": row["compatibility_notes"],
                        "link": row["url"],
                    }
                )
            if len(results) >= 5:
                break
        return results

    def execute(self, query: str) -> str:
        try:
            print(f"🔍 Searching product: {query}")
            results = self.run(query)
            if results:
                formatted_results = "\n".join([f"- {r['title']}: {r['link']}" for r in results])
                return f"Found related products:\n{formatted_results}"
            else:
                return "No relevant product information was found."
        except Exception as e:
            return f"Error finding product information: {str(e)}"

    def _create_tool(self) -> BaseTool:
        @tool
        def product_search(query: str) -> str:
            """Search for product information and return relevant product results based on the user’s query."""
            return self.execute(query=query)

        return product_search


class VectorSearchTool(BaseAgentTool):
    def __init__(self, vec_db_manager: VecDBManager):
        self.vec_db_manager = vec_db_manager
        csv_path = "/home/smalldan/chatbot/data/raw/ai-eng-test-sample-knowledges.csv"
        text_columns = [
            "id",
            "title",
            "content",
            "urls/0/label",
            "urls/0/href",
            "images/0",
            "tags/0",
            "tags/1",
            "tags/2",
        ]
        self.vec_db_manager.init_from_csv(csv_path=csv_path, text_columns=text_columns)
        self.vec_db = self.vec_db_manager.vec_db

    def get_tool_name(self) -> str:
        return "knowledge_search"

    def get_tool_description(self) -> str:
        return "Search knowledge through database"

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
        def knowledge_search(query: str) -> str:
            """Search knowledge through database"""
            return self.execute(query=query)

        return knowledge_search
