import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
root_dir = Path(__file__).parent.parent.parent.parent
load_dotenv(root_dir / ".env")


def get_openai_api_key():
    """Get OpenAI API key from environment or prompt user input."""
    # Export environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        import getpass

        OPENAI_API_KEY = getpass.getpass("Enter API key for OpenAI: ")
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    return OPENAI_API_KEY


def get_tavily_api_key():
    """Get Tavily API key from environment or prompt user input."""
    # Export environment variables
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    if not TAVILY_API_KEY:
        import getpass

        TAVILY_API_KEY = getpass.getpass("Enter API key for Tavily: ")
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    return TAVILY_API_KEY
