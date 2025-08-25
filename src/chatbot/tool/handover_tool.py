import json
import re
import uuid

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool

from chatbot.tool.base_tool import BaseAgentTool
from chatbot.utils.load_env import get_openai_api_key


class SentimentCheckerTool(BaseAgentTool):
    NEGATIVE_THRESHOLD = 0.4

    def __init__(self):
        super().__init__()
        self.semantic_model = self._get_semantic_model()

    def get_tool_name(self) -> str:
        return "detect_sentiment"

    def get_tool_description(self) -> str:
        return "Detect user sentiment. If too negative, suggest transferring to human agent."

    def _get_semantic_model(self) -> None:
        api_key = get_openai_api_key()
        return init_chat_model(
            model="gpt-4o-mini",
            model_provider="openai",
            api_key=api_key,
            temperature=0,
            max_tokens=100,
        )

    def _analyze_sentiment(self, message: str) -> dict:
        """分析情緒"""
        prompt = f"""請對以下訊息評估情緒，給出一個情緒分數(score)，範圍 0到1：
        - 0: 非常負面（憤怒、沮喪、抱怨）
        - 0.5: 中性
        - 1: 非常正面（開心、滿意）

        只回傳 JSON 格式：{{"score": 分數, "reason": "簡短原因"}}

        訊息: {message}"""

        try:
            # 修正：正確調用模型並提取內容
            response = self.semantic_model.invoke([HumanMessage(content=prompt)])
            response_content = response.content if hasattr(response, "content") else str(response)

            print(f"🎭 情緒分析回應: {response_content}")

            # 提取JSON
            match = re.search(r"\{.*?\}", response_content, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                    score = float(result.get("score", 0.5))
                    reason = result.get("reason", "無法判斷")
                    return {"score": score, "reason": reason, "message": message}
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"JSON 解析錯誤: {e}")
                    return {"score": 0.5, "reason": "解析失敗", "message": message}
            else:
                print("未找到有效的JSON格式")
                return {"score": 0.5, "reason": "格式錯誤", "message": message}

        except Exception as e:
            print(f"情緒分析模型調用錯誤: {e}")
            return {"score": 0.5, "reason": f"模型錯誤: {str(e)}", "message": message}

    def execute(self, message: str) -> str:
        try:
            result = self._analyze_sentiment(message)
            score = result["score"]

            if score < self.NEGATIVE_THRESHOLD:
                return "使用者情緒高，建議轉接客服"
            else:
                return "情緒正常"

        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return f"Error analyzing sentiment: {str(e)}"

    def _create_tool(self) -> BaseTool:
        @tool
        def detect_sentiment(message: str) -> str:
            """Detect sentiment and trigger human transfer if necessary."""
            return self.execute(message)

        return detect_sentiment


class HandoffToHumanTool(BaseAgentTool):
    def get_tool_name(self) -> str:
        return "handoff_human"

    def get_tool_description(self) -> str:
        return "Transfer the conversation to a human customer service agent"

    def _is_valid_email(self, text: str) -> bool:
        return re.match(r"[^@]+@[^@]+\.[^@]+", text) is not None

    def _handoff(self, query: str, history: list[str]) -> dict:
        email = None
        if self._is_valid_email(query):
            email = query
        else:
            return {"error": "請提供一個有效的 Email 以便轉接真人客服。"}

        ticket_id = str(uuid.uuid4())[:8]
        summary = "\n".join(history[-5:]) if history else "（無對話紀錄）"

        # TODO: can save into db
        print(f"📑 Human Handoff (ticket={ticket_id}):\n{summary}")

        return {
            "message": f"已為您轉接真人客服，請稍候（案件編號：{ticket_id}）。",
            "email": email,
            "ticket_id": ticket_id,
        }

    def execute(self, query: str, history: list[str] = None) -> str:
        try:
            print(f"🙋 Handoff request: {query}")
            return self._handoff(query, history or [])
        except Exception as e:
            return {"error": f"Handoff error: {str(e)}"}

    def _create_tool(self):
        @tool
        def handoff_human(query: str) -> dict:
            """Transfer the conversation to a human customer service agent"""
            return self.execute(query=query)

        return handoff_human
