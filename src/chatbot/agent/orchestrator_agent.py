import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from chatbot.agent.faq_agent import FAQAgent
from chatbot.agent.handover_agent import HandoverAgent
from chatbot.agent.order_agent import OrderAgent
from chatbot.agent.product_agent import ProductAgent
from chatbot.agent.redirect_agent import RedirectAgent
from chatbot.tool.handover_tool import SentimentCheckerTool
from chatbot.utils.load_env import get_openai_api_key


class AgentType(Enum):
    HANDOVER = "handover_agent"
    ORDER = "order_agent"
    FAQ = "faq_agent"
    PRODUCT = "product_agent"
    REDIRECT = "redirect_agent"


@dataclass
class RoutingResult:
    agent_type: AgentType
    confidence: float
    reason: str
    sentiment_score: float | None = None
    should_handover: bool = False


class LLMRouter:
    def __init__(self):
        self.api_key = get_openai_api_key()
        self.router_model = init_chat_model(
            model="gpt-4o-mini",
            model_provider="openai",
            api_key=self.api_key,
            temperature=0.1,
            max_tokens=300,
        )
        self.sentiment_tool = SentimentCheckerTool()

    def _create_routing_prompt(self, message: str, user_info: dict[str, Any]) -> str:
        user_context = ""
        if user_info:
            context_parts = []
            if user_info.get("name"):
                context_parts.append(f"姓名: {user_info['name']}")
            if user_info.get("email"):
                context_parts.append(f"信箱: {user_info['email']}")
            if user_info.get("location"):
                context_parts.append(f"地點: {user_info['location']}")

            if context_parts:
                user_context = f"\n用戶資訊: {', '.join(context_parts)}"

        return f"""你是 JTCG shop 的智能客服路由系統。請分析用戶訊息並決定應該路由到哪個專門代理。

        {user_context}

        用戶訊息: "{message}"

        可用的代理類型及其職責:
        1. handover_agent: 處理客訴、情緒激動、要求真人客服的情況
        2. order_agent: 處理訂單查詢、訂單狀態、
        3. faq_agent: 處理常見問題、退換貨、物流運送問題、保固、發票、安裝指南、一般資訊
        4. product_agent: 處理產品資訊、規格查詢、兼容性問題
        5. redirect_agent: 處理與本店無關的問題（其他品牌、無關話題）

        路由規則:
        - 如果用戶情緒負面、抱怨、憤怒 → handover_agent
        - 如果提到訂單號、訂單狀態、退貨、物流 → order_agent
        - 如果詢問政策、退換貨規定、常見問題 → faq_agent
        - 如果詢問產品規格、兼容性、產品比較 → product_agent
        - 如果詢問其他品牌或無關話題 → redirect_agent

        請回傳 JSON 格式:
        {{
            "agent_type": "代理類型",
            "confidence": 0.9,
            "reason": "選擇理由",
            "keywords": ["關鍵字1", "關鍵字2"]
        }}"""

    def route_message(self, message: str, user_info: dict[str, Any] = None) -> RoutingResult:
        user_info = user_info or {}

        try:
            # 1. Check semantic first
            sentiment_result = self.sentiment_tool.execute(message)
            sentiment_score = self._extract_sentiment_score(sentiment_result)

            if sentiment_score is not None and sentiment_score <= 0.4:
                return RoutingResult(
                    agent_type=AgentType.HANDOVER,
                    confidence=0.95,
                    reason="檢測到負面情緒，需要人工客服介入",
                    sentiment_score=sentiment_score,
                    should_handover=True,
                )

            # 2. Use the LLM to route the msg
            prompt = self._create_routing_prompt(message, user_info)
            response = self.router_model.invoke([HumanMessage(content=prompt)])

            # 3. parse the response of the LLM output
            routing_result = self._parse_routing_response(response.content, sentiment_score)

            print(
                f"🎯 路由決策: {routing_result.agent_type.value} (信心度: {routing_result.confidence:.2f})"
            )
            print(f"   理由: {routing_result.reason}")
            if sentiment_score:
                print(f"   情緒分數: {sentiment_score:.2f}")

            return routing_result

        except Exception as e:
            print(f"route_message Error: {e}")
            return RoutingResult(
                agent_type=AgentType.FAQ,
                confidence=0.5,
                reason=f"路由失敗, 降級到FAQ代理: {str(e)}",
                sentiment_score=sentiment_score,
            )

    def _extract_sentiment_score(self, sentiment_result: str) -> float | None:
        try:
            # Findinf (分數: X.XX) format
            match = re.search(r"分數:\s*([0-9.]+)", sentiment_result)
            if match:
                return float(match.group(1))
        except Exception:
            pass
        return None

    def _parse_routing_response(
        self, response_content: str, sentiment_score: float | None
    ) -> RoutingResult:
        try:
            match = re.search(r"\{.*\}", response_content, re.DOTALL)
            if match:
                result = json.loads(match.group())

                agent_type_str = result.get("agent_type", "faq_agent")
                confidence = float(result.get("confidence", 0.7))
                reason = result.get("reason", "LLM路由決策")

                try:
                    agent_type = AgentType(agent_type_str)
                except ValueError:
                    agent_type = AgentType.FAQ  # default: FAQ

                should_handover = agent_type == AgentType.HANDOVER or (
                    sentiment_score is not None and sentiment_score <= 0.3
                )

                return RoutingResult(
                    agent_type=agent_type,
                    confidence=confidence,
                    reason=reason,
                    sentiment_score=sentiment_score,
                    should_handover=should_handover,
                )

        except Exception as e:
            print(f"解析路由回應錯誤: {e}")

        # Go back to the FAQ
        return RoutingResult(
            agent_type=AgentType.FAQ,
            confidence=0.5,
            reason="解析失敗，使用預設路由",
            sentiment_score=sentiment_score,
        )


class OrchestratorAgent:
    def __init__(self):
        self.agents = {
            AgentType.HANDOVER: HandoverAgent(),
            AgentType.ORDER: OrderAgent(),
            AgentType.FAQ: FAQAgent(),
            AgentType.PRODUCT: ProductAgent(),
            AgentType.REDIRECT: RedirectAgent(),
        }
        self.router = LLMRouter()
        self.conversation_state = {agent_type: [] for agent_type in self.agents}

        print("🚀 OrchestratorAgent 初始化完成")
        print(f"   已載入 {len(self.agents)} 個專門代理")

    def route_and_execute(
        self, message: str, user_info: dict[str, Any] = None, is_display: bool = None
    ) -> dict[str, Any]:
        user_info = user_info or {}

        print(f"\n📨 收到訊息: {message}")
        print(f"👤 用戶資訊: {user_info}")

        try:
            # 1. Use the LLM to determine which agent should we use
            routing_result = self.router.route_message(message, user_info)

            # 2. Use that specific agent
            selected_agent = self.agents[routing_result.agent_type]

            # 3. Save the message history
            self.conversation_state[routing_result.agent_type].append(message)

            # 4. Excute the corresponding results
            response = self._execute_agent(
                selected_agent,
                routing_result.agent_type,
                self.conversation_state,
                is_display,
                routing_result,
            )

            return {
                "message": response,
                "agent_type": routing_result.agent_type.value,
                "confidence": routing_result.confidence,
                "reason": routing_result.reason,
                "sentiment_score": routing_result.sentiment_score,
                "should_handover": routing_result.should_handover,
            }

        except Exception as e:
            print(f"route_and_execute Error: {e}")
            return {
                "message": f"抱歉, 處理您的請求時發生錯誤: {str(e)}",
                "agent_type": "error",
                "confidence": 0.0,
                "reason": "系統錯誤",
            }

    def _execute_agent(
        self,
        agent,
        agent_type: AgentType,
        conversation_state: dict,
        is_display: bool,
        routing_result: RoutingResult,
    ) -> str:
        try:
            match agent_type:
                case AgentType.HANDOVER:
                    return self._execute_handover_agent(
                        agent, conversation_state[agent_type], is_display, routing_result
                    )

                case AgentType.ORDER:
                    return agent.run_conversation(conversation_state[agent_type], is_display)

                case AgentType.FAQ:
                    return agent.run_conversation(conversation_state[agent_type], is_display)

                case AgentType.PRODUCT:
                    return agent.run_conversation(conversation_state[agent_type], is_display)

                case AgentType.REDIRECT:
                    return agent.run_conversation(conversation_state[agent_type], is_display)

                case _:
                    return agent.run_conversation(conversation_state[agent_type], is_display)

        except Exception as e:
            print(f"_execute_agent Error ({agent_type.value}): {e}")
            return f"Agent Excuting Error ({agent_type.value}): {e}"

    def _execute_handover_agent(
        self, agent, message: str, is_display: bool, routing_result: RoutingResult
    ) -> str:
        try:
            if routing_result.should_handover and routing_result.sentiment_score:
                comfort_message = f"""我理解您現在可能感到不滿，非常抱歉造成您的困擾
                情緒分析: {routing_result.sentiment_score:.2f} / 1.0
                {routing_result.reason}

                讓我來幫助您解決這個問題"""
                print(comfort_message)
            return agent.run_conversation(message, is_display)
        except Exception as e:
            print(f"_execute_handover_agent Error: {e}")
            return "我們已記錄您的問題，客服將盡快與您聯繫。"
