import os
from dotenv import load_dotenv
from openai import OpenAI

# Load variables from .env
load_dotenv()

api_key = os.getenv("QWEN_API_KEY")

if not api_key:
    raise ValueError("QWEN_API_KEY is not set")

# Create OpenAI-compatible client
client = OpenAI(
    base_url="https://rimless-operator-abacus.ngrok-free.dev/v1",
    api_key=api_key
)

print("=================================")
print("       Qwen AI Chatbot")
print("=================================")
print("Type 'exit' or 'quit' to stop.\n")

while True:

    # Get custom input from user
    user_input = input("You: ")

    # Exit condition
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Ignore empty input
    if not user_input.strip():
        continue

    try:
        # Send user's content to the AI model
        response = client.chat.completions.create(
            model="qwen2.5-0.5b",
            messages=[
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            max_tokens=150,
            temperature=0.7
        )

        # Display AI response
        answer = response.choices[0].message.content

        print("\nAI:", answer)
        print()

    except Exception as e:
        print("\nError:", e)
