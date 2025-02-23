import openai
from dotenv import load_dotenv
import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_utils import get_image_messages

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_play_analysis(play_number):
    # Get base64 encoded images
    image_messages = get_image_messages(play_number)

    # Load conversation history
    with open("../system_prompt.json", "r", encoding="utf-8") as f:
        messages = json.load(f)

    # Construct the message with images
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "This is the starting formation and play movement analysis.",
            }
        ],
    }

    # Add images to the content
    for img in image_messages:
        user_message["content"].append(
            {"type": "image", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
        )

    messages.append(user_message)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.5,
        )
        analysis = response.choices[0].message["content"]

        # Add the response to the conversation history
        messages.append({"role": "assistant", "content": analysis})

        return analysis
    except Exception as e:
        print(f"Error getting AI analysis: {e}")
        return None
