from langchain_core.tools import BaseTool, tool

from chatbot.tool.base_tool import BaseAgentTool


class RedirectTopicTool(BaseAgentTool):
    def get_tool_name(self) -> str:
        return "redirect_topic"

    def get_tool_description(self) -> str:
        return (
            "When the user asks about topics unrelated to BenQ shopping or products, "
            "politely redirect them to the topics we can help with: FAQ, products, orders, human agent"
        )

    def _check_and_redirect(self, query: str) -> dict:
        query_lower = query.lower()
        shopping_keywords = ["benq", "訂單", "產品", "購買", "價格", "折扣", "保固", "型號", "商品"]

        is_related = any(k in query_lower for k in shopping_keywords)
        if is_related:
            return {"message": "👍 我們可以協助您的購物/產品相關問題。"}
        else:
            redirect_msg = (
                "您好，關於您提問的議題，我們可能無法直接提供完整答案\n"
                "不過我們可以幫您：\n"
                "1. 查詢 FAQ 常見問題\n"
                "2. 查詢產品資訊\n"
                "3. 查詢訂單狀態\n"
                "4. 轉接真人客服\n"
                "請問您想從哪一個開始？"
            )
            return {"message": redirect_msg}

    def execute(self, query: str) -> dict:
        try:
            print(f"🔄 Redirect check: {query}")
            return self._check_and_redirect(query)
        except Exception as e:
            print(f"Redirect tool error: {e}")
            return {"error": str(e)}

    def _create_tool(self) -> BaseTool:
        from langchain_core.tools import tool

        @tool
        def redirect_topic(query: str) -> dict:
            """Redirect user politely if the topic is unrelated to BenQ shopping or products"""
            return self.execute(query=query)

        return redirect_topic


class TopicCheckerTool(BaseAgentTool):
    def get_tool_name(self) -> str:
        return "check_topic"

    def get_tool_description(self) -> str:
        return "Check if the user query is related to BenQ shopping, orders, or products"

    def _check_topic(self, query: str) -> dict:
        query_lower = query.lower()

        # Keywords related to shopping
        shopping_keywords = ["benq", "訂單", "產品", "購買", "價格", "折扣", "保固", "型號", "商品"]

        is_related = any(k in query_lower for k in shopping_keywords)
        return {"related_to_shopping": is_related}

    def execute(self, query: str) -> dict:
        try:
            print(f"Checking topic relevance: {query}")
            return self._check_topic(query)
        except Exception as e:
            print(f"Topic check error: {e}")
            return {"error": str(e)}

    def _create_tool(self) -> BaseTool:
        @tool
        def check_topic(query: str) -> dict:
            """Check if the user query is about BenQ shopping or products"""
            return self.execute(query=query)

        return check_topic
