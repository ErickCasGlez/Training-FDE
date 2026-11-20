# Copy .env.example to .env and add your AWS credentials and region.
# With venv (Windows): py -m venv .venv
#   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
#   .\.venv\Scripts\python.exe chatbot.py
# With venv (macOS/Linux): python3 -m venv .venv
#   .venv/bin/python -m pip install -r requirements.txt
#   .venv/bin/python chatbot.py
# Without venv (Windows): py -m pip install -r requirements.txt; py chatbot.py
# Without venv (macOS/Linux): python3 -m pip install -r requirements.txt; python3 chatbot.py

import boto3
from dotenv import load_dotenv

MODEL_ID = "mistral.mistral-7b-instruct-v0:2"


def main() -> None:
    load_dotenv()
    client = boto3.client("bedrock-runtime")
    messages = []

    print("Chat started. Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "salir", "/salir"}:
            break

        messages.append({"role": "user", "content": [{"text": user_input}]})
        response = client.converse(
            modelId=MODEL_ID,
            messages=messages,
            inferenceConfig={"temperature": 0.7, "maxTokens": 1024},
        )
        answer = "\n".join(
            block["text"]
            for block in response["output"]["message"]["content"]
            if "text" in block
        )
        print(f"Assistant: {answer}")
        messages.append({"role": "assistant", "content": [{"text": answer}]})


if __name__ == "__main__":
    main()
