import json

from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool

from chatbot.tool.base_tool import BaseAgentTool
from chatbot.utils.vector_db import VecDBManager


class OrderSearchTool(BaseAgentTool):
    def __init__(self, vec_db_manager: VecDBManager):
        self.vec_db_manager = vec_db_manager
        json_path = "data/raw/ai-eng-test-sample-order.json"
        self.vec_db_manager.init_from_json(json_path=json_path)
        self.vec_db = self.vec_db_manager.vec_db

        with open(json_path, encoding="utf-8") as f:
            self.orders_data = json.load(f)["orders_db"]

    def get_tool_name(self) -> str:
        return "order_search"

    def get_tool_description(self) -> str:
        return "Search order information through database"

    def _structure_search(self, query: str) -> dict:
        query_lower = query.lower()
        user_id = None
        order_id = None

        if "user_id=" in query_lower:
            user_id = query_lower.split("user_id=")[-1].split(",")[0].strip()
        if "order_id=" in query_lower:
            order_id = query_lower.split("order_id=")[-1].split(",")[0].strip()

        if not user_id:
            return {"error": "缺少 user_id，請提供 user_id 以查詢訂單。"}

        user_orders = self.orders_data.get(user_id)
        if not user_orders:
            return {"error": f"查無此 user_id ({user_id}) 的訂單紀錄，請確認是否正確。"}

        orders = user_orders.get("orders", [])
        if not orders:
            return {"message": f"user_id={user_id} 沒有任何訂單紀錄。"}

        if not order_id:
            return {
                "user_id": user_id,
                "orders": [
                    {
                        "order_id": o["order_id"],
                        "status": o["status"],
                        "eta": o["eta"],
                        "items": o["items"],
                    }
                    for o in orders
                ],
            }

        target_order = next((o for o in orders if o["order_id"].lower() == order_id.lower()), None)
        if not target_order:
            return {
                "error": f"user_id={user_id} 下查無此訂單 {order_id}，請確認 order_id 是否正確。"
            }

        return {"user_id": user_id, "order": target_order}

    def _format_documents(self, documents: list[Document]) -> str:
        if not documents:
            return "Documents not found"

        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            content = doc.page_content

            metadata_info = ""
            if hasattr(doc, "metadata") and doc.metadata:
                title = doc.metadata.get("title", "")
                source = doc.metadata.get("source", "")
                if title:
                    metadata_info = f" [標題: {title}]"
                if source:
                    metadata_info += f" [來源: {source}]"

            formatted_docs.append(f"文件 {i}:{metadata_info}\n{content}")

        return "\n\n".join(formatted_docs)

    def _semantic_search(self, query: str, k: int = 3) -> str:
        results = self.vec_db.similarity_search(query, k=k)
        formatted_results = self._format_documents(results)
        return f"在知識庫中找到 {len(results)} 筆相關文件:\n\n{formatted_results}"

    def execute(self, query: str, k: int = 3) -> str:
        try:
            if "user_id=" in query.lower():
                return json.dumps(self._search(query), ensure_ascii=False, indent=2)
            else:
                return self._semantic_search(query, k=k)
        except Exception as e:
            print(f"Searching error: {e}")
            return f"Searching error: {str(e)}"

    def _create_tool(self) -> BaseTool:
        @tool
        def order_search(query: str) -> str:
            """Search orders through database"""
            return self.execute(query=query)

        return order_search


class RequirementCheckerTool(BaseAgentTool):
    def get_tool_name(self) -> str:
        return "check_missing"

    def get_tool_description(self) -> str:
        return "Check if we're missing needed information: user_id, order_id"

    def _check(self, query: str) -> dict:
        query_lower = query.lower()
        missing = []

        if "user_id" not in query_lower:
            missing.append("user_id")
        elif "order_id" not in query_lower:
            missing.append("order_id")
        return {"missing": missing}

    def execute(self, query: str) -> dict:
        try:
            print(f"📚 Checking missing information: {query}")
            return self._check(query)
        except Exception as e:
            print(f"Checking error: {e}")
            return {"error": str(e)}

    def _create_tool(self) -> BaseTool:
        @tool
        def check_missing(query: str) -> dict:
            """Check if we miss some crucial information (user_id, order_id) from users"""
            return self.execute(query=query)

        return check_missing
