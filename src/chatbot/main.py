from chatbot.agent.orchestrator_agent import OrchestratorAgent


class Chatbot:
    def __init__(self, user_info: dict = None) -> None:
        self.orchestrator = OrchestratorAgent()
        self.default_test_messages = [
            "我想查詢訂單 12345",
            "user_id=u_123456",
            "我想抱怨一下，很煩，這根本不是我的訂單！",
            "我想問其他品牌的產品",
            "你們的退貨政策是什麼？",
            "這個螢幕支援什麼尺寸？",
        ]
        self.default_user_info = {"name": "測試用戶", "email": "test@example.com"}

    @staticmethod
    def pretty_print(response: dict) -> None:
        print("\n🤖 路由結果:")
        print(f"   代理類型: {response['agent_type']}")
        print(f"   信心度: {response['confidence']:.2f}")
        print(f"   理由: {response['reason']}")
        if response.get("sentiment_score"):
            print(f"   情緒分數: {response['sentiment_score']:.2f}")
        if response.get("message"):
            print(f"\n💬 回應: {response['message']}")

    def process_single_user_message(self, message: str, user_info: dict = None) -> dict:
        return self.orchestrator.route_and_execute(message, user_info)

    def dry_run(self, user_info: dict = None, messages: list[str] = None) -> None:
        user_info = user_info or {"name": "測試用戶", "email": "test@example.com"}
        messages = messages or self.default_test_messages
        for i, message in enumerate(messages):
            print(f"\n{'='*60}")
            print(f"測試 {i}: {message}")
            print("=" * 60)
            response = self.process_single_user_message(message, user_info)
            self.pretty_print(response)


if __name__ == "__main__":
    chatbot = Chatbot()
    chatbot.dry_run()
