import argparse

from chatbot.main import Chatbot


def main() -> None:
    arg_parser = argparse.ArgumentParser(
        prog="chatbot", description="Use this agentic AI system as a customer service"
    )

    arg_parser.add_argument(
        "--question",
        "-q",
        type=str,
        default="You don't provide a question, please type your question in so that the agent can help you",
        help="The question you want to ask the chatbot",
    )

    arg_parser.add_argument(
        "--user_id",
        "-uid",
        type=str,
        help="The user ID of the chatbot user",
    )

    arg_parser.add_argument(
        "--user_name",
        "-u",
        type=str,
        help="The user name of the chatbot user",
    )

    arg_parser.add_argument(
        "--email",
        "-e",
        type=str,
        help="The user email of the chatbot user",
    )

    arg_parser.add_argument(
        "--interactive", "-i", action="store_true", help="Run the chatbot in interactive mode"
    )

    arg_parser.add_argument(
        "--display", "-d", action="store_true", help="Show detailed conversation output"
    )

    arg_parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["AI", "human"],
        default="AI",
        help="Choose between AI agent or live human customer service",
    )

    arg_parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Run the chatbot in test mode with sample messages",
    )

    args = arg_parser.parse_args()
    chatbot = Chatbot()

    # set up user basic information
    if args.user_id:
        chatbot.user_id = args.user_id
    if args.user_name:
        chatbot.user_name = args.user_name
    if args.email:
        chatbot.email = args.email
    if args.display:
        chatbot.is_display = True

    if args.interactive:
        chatbot.run_interactive()
        exit(0)
    if args.test:
        chatbot.dry_run()
    if args.mode == "human":
        chatbot.process_single_user_message(message="我要真人客服")
        exit(0)
    if args.question:
        chatbot.process_single_user_message(args.question)


if __name__ == "__main__":
    main()
