# Chatbot

An chatbot project on JTCG

## 📋 Table of Contents
- [About](#about)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🤖 About

This is a chatbot project developed for JTCG. The chatbot is designed to provide automated conversational capabilities for customer support.

## ✨ Features

- **Multi Agents System**: Understands and responds to user queries with multi-Agents
- **Multi-turn Conversations**: Maintains context across multiple interactions
- **Customizable Responses**: Easy to configure and extend response patterns
- **Reasoning and Flexible**: Give you the answers with the reasons and can generate flexible content


## 📋 Prerequisites

Before you begin, ensure you have the following installed:
- [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)
- [Python 3.11+](https://www.python.org/downloads/) (or appropriate language version)
- [poetry](https://python-poetry.org/)
- [Git](https://git-scm.com/downloads)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/smalldan1022/chatbot.git
   cd chatbot
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   conda create -n YOUR_ENV_NAME python=3.11
   ```

3. **Install dependencies**
   ```bash
   poetry install
   ```

4. **Set up environment variables**
   ```bash
   # Set up your own OpenAI API Key
   # at the root of the folder
   vim .env

   (Inside the .env)
   OPENAI_API_KEY=xxxxxxx
   ```

## 💬 Usage

### Running the Chatbot

1. **Start the application**
   ```bash
   # interactive mode
   chatbot -i

   # interactive mode without other display
   chatbot -d -i
   ```

2. **Access the chatbot**
    ```bash
    # or at the root folder
    python src/chatbot/main.py
   ```

### Basic Example

```python
# Example of using the chatbot
from chatbot import ChatBot

chatbot = ChatBot()
response = chatbot.process_single_user_message("Hello, how are you?")
print(response)
```

## ⚙️ Configuration
- The Agent
    
    The agent can be configured through the `config.yaml` files:

    ```yaml
    
    ```
- The Prompt

    The prompt can be configured through the `config.yaml` files:

    ```yaml
    
    ```
- The Orchestrator Route

    The orchestrator route can be configured through the `config.yaml` files:

    ```yaml
    
    ```


### Development Guidelines

- Follow PEP 8 style guide for Python code
- Write unit tests for new features
- Update documentation as needed
- Keep commits atomic and well-described

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## 👤 Contact

**smalldan1022**
- GitHub: [@smalldan1022](https://github.com/smalldan1022)
- Email: [asign1022@gmail.com](https://github.com/smalldan1022/chatbot)

## 🙏 Acknowledgments

- Thanks for JTCG team's project support

---

⭐ If you find this project useful, please consider giving it a star on GitHub!