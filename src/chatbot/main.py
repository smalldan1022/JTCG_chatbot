from chatbot.agent.orchestrator_agent import OrchestratorAgent


def main():
    orchestrator = OrchestratorAgent()

    test_messages = [
        "我想查詢訂單 12345",
        "u_123456",
        "我想抱怨一下，很煩，這根本不是我的訂單！",
        "我想問其他品牌的產品",
        "你們的退貨政策是什麼？",
        "這個螢幕支援什麼尺寸？",
    ]

    user_info = {"name": "測試用戶", "email": "test@example.com"}

    for i, message in enumerate(test_messages):
        print(f"\n{'='*60}")
        print(f"測試 {i}: {message}")
        print("=" * 60)

        response = orchestrator.route_and_execute(message, user_info)

        print("\n🤖 路由結果:")
        print(f"   代理類型: {response['agent_type']}")
        print(f"   信心度: {response['confidence']:.2f}")
        print(f"   理由: {response['reason']}")
        if response.get("sentiment_score"):
            print(f"   情緒分數: {response['sentiment_score']:.2f}")
        if response.get("message"):
            print(f"\n💬 回應: {response['message'][:200]}...")


if __name__ == "__main__":
    main()
