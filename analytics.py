from openai import OpenAI
from dotenv import load_dotenv
import os
import parse_system_prompt
import json

# Load the .env file
load_dotenv()

# --- Use the new API endpoint ---
client = OpenAI()

# Access the API key from the environment
client.api_key = os.getenv("OPENAI_API_KEY")

parse_system_prompt.parse("system_prompt.md")

with open("system_prompt.json", "r", encoding="utf-8") as f:
    messages = json.load(f)

print("Enter 'exit' to quit the conversation.")

while True:
    # Get user input
    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        break

    # Append user message to the conversation history
    messages.append({"role": "user", "content": user_input})
    
    # Get the response from the API
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.5,
        )
    except Exception as e:
        print("Error during API call:", e)
        continue

    reply = response.choices[0].message.content
    print("Assistant:", reply)
    messages.append({"role": "assistant", "content": reply})